import argparse
import json
import os
import shutil
import sys  # for command line argument dumping
import time
from collections import OrderedDict
from datetime import datetime  # for timing

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import progressbar

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

try:
    from azureml.core.run import Run

    run = Run.get_context()
except ImportError:
    run = None

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

BASE_LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


def main():
    args, doLoad, doTest, doValidate, dataPath, checkpointsPath, \
    exportONNX, saveCheckpoints, using_cuda, workers, epochs, dataset_limit, verbose = parse_commandline_arguments()

    if using_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        using_cuda = True
    else:
        device = torch.device('cpu')

    if using_cuda and torch.cuda.device_count() > 0:
        batch_size = torch.cuda.device_count() * 100  # Change if out of cuda memory
    else:
        batch_size = 5

    best_prec1 = 1e20
    lr = BASE_LR

    if verbose:
        print('')
        if using_cuda:
            print('CUDA DEVICE_COUNT {0}'.format(torch.cuda.device_count()))
        print('')

    model = ITrackerModel().to(device=device)

    if using_cuda:
        model = torch.nn.DataParallel(model).to(device=device)

    image_size = (224, 224)
    cudnn.benchmark = False

    epoch = 1
    if doLoad:
        saved = load_checkpoint(checkpointsPath, device)
        if saved:
            print(
                'Loading checkpoint for epoch %05d with loss %.5f '
                '(which is the mean squared error not the actual linear error)...' % (
                    saved['epoch'],
                    saved['best_prec1'])
            )

            try:
                state = saved['state_dict']
                model.load_state_dict(state)
            except RuntimeError:
                # The most likely cause of a failure to load is that there is a leading "module." from training. This is
                # normal for models trained with DataParallel. If not using DataParallel, then the "module." needs to be
                # removed.
                state = remove_module_from_state(saved)
                model.load_state_dict(state)

            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    print('epoch = %d' % epoch)

    totalstart_time = datetime.now()

    datasets = load_all_data(dataPath, image_size, workers, batch_size, verbose)

    print('')  # print blank line after loading data

    #     criterion = nn.MSELoss(reduction='sum').to(device=device)
    criterion = nn.MSELoss(reduction='mean').to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    if doTest:
        # Quick test
        start_time = datetime.now()
        precision = test(datasets, model, criterion, 1, checkpointsPath, batch_size, device, dataset_limit, verbose)
        time_elapsed = datetime.now() - start_time
        print('')
        print('Testing loss %.5f' % precision)
        print('Testing Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif doValidate:
        # Validation of saved checkpoint
        start_time = datetime.now()
        precision = validate(datasets, model, criterion, 1, checkpointsPath, batch_size, device, dataset_limit, verbose)
        time_elapsed = datetime.now() - start_time
        print('')  # print blank line after loading data
        print('Validation loss %.5f' % precision)
        print('Validation Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif exportONNX:
        # export the model for use in other frameworks
        export_onnx_model(model, device, verbose)
    else:  # Train
        # first make a learning_rate correction suitable for epoch from saved checkpoint
        # epoch will be non-zero if a checkpoint was loaded
        for epoch in range(1, epoch):
            if verbose:
                print('Epoch %05d of %05d - adjust learning rate only' % (epoch, epochs))
                start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)
            if verbose:
                time_elapsed = datetime.now() - start_time
                print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

        # now start training from last best epoch
        for epoch in range(epoch, epochs + 1):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, epochs))
            start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            print('\nEpoch:{} [device:{}, lr:{}]'.format(epoch, device, lr))
            train_error = train(datasets['train'], model, criterion, optimizer, epoch, batch_size, device, dataset_limit, verbose)

            # evaluate on validation set
            prec1 = validate(datasets, model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit, verbose)

            # remember best prec1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)

            time_elapsed = datetime.now() - start_time

            if run:
                run.log('precision', prec1)
                run.log('best precision', best_prec1)
                run.log('epoch time', time_elapsed)

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1
            },
                is_best,
                checkpointsPath,
                saveCheckpoints)

            print('')
            print('Epoch %05d with loss %.5f' % (epoch, best_prec1))
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))


def train(dataset, model, criterion, optimizer, epoch, batch_size, device, dataset_limit=None, verbose=False):
    data_size = dataset['size']
    loader = dataset['loader']
    split = dataset['split']

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    num_samples = 0

    if not verbose:
        progress_meter = ProgressMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame) in enumerate(loader):
        batchNum = i + 1
        num_samples += imFace.size(0)

        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.to(device=device)
        imEyeL = imEyeL.to(device=device)
        imEyeR = imEyeR.to(device=device)
        faceGrid = faceGrid.to(device=device)
        gaze = gaze.to(device=device)

        imFace = torch.autograd.Variable(imFace, requires_grad=True)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad=True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad=True)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad=True)
        gaze = torch.autograd.Variable(gaze, requires_grad=False)

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)

        lossLin = output - gaze
        lossLin = torch.mul(lossLin, lossLin)
        lossLin = torch.sum(lossLin, 1)
        # MSE vs RMS error
        #         lossLin = torch.sum(lossLin)
        lossLin = torch.sum(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item() / batch_size, imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Epoch (train): [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'MSELoss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'RMSErr {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                epoch, batchNum, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lossLin=lossesLin))
        else:
            progress_meter.update(num_samples, data_size, split, lossesLin.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    print('')

    return lossesLin.avg


def evaluate(dataset, model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit=None, verbose=False):
    data_size = dataset['size']
    loader = dataset['loader']
    split = dataset['split']

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    num_samples = 0

    if not verbose:
        progress_meter = ProgressMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    results = []

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame) in enumerate(loader):
        batchNum = i + 1
        num_samples += imFace.size(0)

        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.to(device=device)
        imEyeL = imEyeL.to(device=device)
        imEyeR = imEyeR.to(device=device)
        faceGrid = faceGrid.to(device=device)
        gaze = gaze.to(device=device)

        imFace = torch.autograd.Variable(imFace, requires_grad=False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad=False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad=False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad=False)
        gaze = torch.autograd.Variable(gaze, requires_grad=False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        # Combine the tensor results together into a collated list so that we have the gazePoint and gazePrediction
        # for each frame
        f1 = frame.cpu().numpy().tolist()
        g1 = gaze.cpu().numpy().tolist()
        o1 = output.cpu().numpy().tolist()
        r1 = [list(r) for r in zip(f1, g1, o1)]

        def convertResult(result):
            return {'frame': result[0], 'gazePoint': result[1], 'gazePrediction': result[2]}

        results += list(map(convertResult, r1))
        loss = criterion(output, gaze)

        lossLin = output - gaze
        lossLin = torch.mul(lossLin, lossLin)
        lossLin = torch.sum(lossLin, 1)
        # MSE vs RMS error
        #         lossLin = torch.sum(lossLin)
        lossLin = torch.sum(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item() / batch_size, imFace.size(0))

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Epoch ({}): [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'MSELoss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'RMSErr {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                split, epoch, batchNum, len(loader), batch_time=batch_time,
                loss=losses, lossLin=lossesLin))
        else:
            progress_meter.update(num_samples, data_size, split, lossesLin.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    resultsFileName = os.path.join(checkpointsPath, 'results.json')
    with open(resultsFileName, 'w+') as outfile:
        json.dump(results, outfile)

    print('')

    return lossesLin.avg


def validate(datasets,
             model,
             criterion,
             epoch,
             checkpointsPath,
             batch_size,
             device,
             dataset_limit=None,
             verbose=False):
    return evaluate(datasets['val'], model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit, verbose)


def test(datasets,
         model,
         criterion,
         epoch,
         checkpointsPath,
         batch_size,
         device,
         dataset_limit=None,
         verbose=False):
    return evaluate(datasets['test'], model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit, verbose)


def export_onnx_model(model, device, verbose):
    # switch to evaluate mode
    model.eval()

    batch_size = 1
    color_depth = 3  # 3 bytes for RGB color space
    dim_width = 224
    dim_height = 224
    face_grid_size = 25 * 25

    imFace = torch.randn(batch_size, color_depth, dim_width, dim_height).to(device=device).float()
    imEyeL = torch.randn(batch_size, color_depth, dim_width, dim_height).to(device=device).float()
    imEyeR = torch.randn(batch_size, color_depth, dim_width, dim_height).to(device=device).float()
    faceGrid = torch.zeros((batch_size, face_grid_size)).to(device=device).float()

    dummy_in = (imFace, imEyeL, imEyeR, faceGrid)

    in_names = ["face", "eyesLeft", "eyesRight", "faceGrid"]
    out_names = ["data"]

    try:
        torch.onnx.export(model.module,
                          dummy_in,
                          "itracker.onnx",
                          input_names=in_names,
                          output_names=out_names,
                          verbose=verbose)
    except AttributeError:
        torch.onnx.export(model,
                          dummy_in,
                          "itracker.onnx",
                          input_names=in_names,
                          output_names=out_names,
                          verbose=verbose)


def load_checkpoint(checkpointsPath, device, filename='checkpoint.pth.tar'):
    filename = os.path.join(checkpointsPath, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename, map_location=device)
    return state


def save_checkpoint(state, is_best, checkpointsPath, saveCheckpoints, filename='checkpoint.pth.tar'):
    resultsFilename = os.path.join(checkpointsPath, 'results.json')
    checkpointFilename = os.path.join(checkpointsPath, filename)

    torch.save(state, checkpointFilename)

    if saveCheckpoints:
        shutil.copyfile(checkpointFilename,
                        os.path.join(checkpointsPath, 'checkpoint' + str(state['epoch']) + '.pth.tar'))
        shutil.copyfile(resultsFilename, os.path.join(checkpointsPath, 'results' + str(state['epoch']) + '.json'))

    bestFilename = os.path.join(checkpointsPath, 'best_' + filename)
    bestResultsFilename = os.path.join(checkpointsPath, 'best_results.json')

    if is_best:
        shutil.copyfile(checkpointFilename, bestFilename)
        shutil.copyfile(resultsFilename, bestResultsFilename)


def remove_module_from_state(saved_state):
    # when using Cuda for training we use DataParallel. When using DataParallel, there is a
    # 'module.' added to the namespace of the item in the dictionary.
    # remove 'module.' from the front of the name to make it compatible with cpu only
    state = OrderedDict()

    for key, value in saved_state['state_dict'].items():
        state[key[7:]] = value.to(device='cpu')

    return state


def load_data(split, path, image_size, workers, batch_size, verbose):
    data = ITrackerData(path, split=split, imSize=image_size, silent=not verbose)
    size = len(data.indices)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    return {
        'split': split,
        'data': data,
        'size': size,
        'loader': loader
    }


def load_all_data(path, image_size, workers, batch_size, verbose):
    return {
        # training data : model sees and learns from this data
        'train': load_data('train', path, image_size, workers, batch_size, verbose),
        # validation data : model sees but never learns from this data
        'val': load_data('val', path, image_size, workers, batch_size, verbose),
        # test data : model never sees or learns from this data
        'test': load_data('test', path, image_size, workers, batch_size, verbose)
    }


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter(object):
    def __init__(self):
        self.widgets = [
            'Progress',  # 0
            ' ',  # 1
            progressbar.Bar(marker='X', left='|', right='|', fill='-'),  # 2
            '[', progressbar.SimpleProgress(), ']',  # 4
            '[', progressbar.ETA(), ']',  # 7
            '[', 'RMSError', ']',  # 10
        ]
        self.bar = progressbar.ProgressBar(maxval=0, widgets=self.widgets)
        self.bar.start()

    def update(self, value, maxval, label, error):
        # update label
        label = '{:5}'.format(label)
        if self.bar.widgets[0] != label:
            self.bar.widgets[0] = label

        # update metric
        metric = '{metric:.4f}'.format(metric=error)
        if self.bar.widgets[10] != metric:
            self.bar.widgets[10] = metric

        # update max_value
        if self.bar.maxval != maxval:
            self.bar.maxval = maxval
        # update value
        self.bar.update(value)
        # update finish
        if value >= self.bar.maxval:
            self.bar.finish()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = BASE_LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
    parser.add_argument('--data_path',
                        help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.",
                        default='/data/gc-data-prepped/')
    parser.add_argument('--output_path', help="Path to checkpoint", default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--save_checkpoints', type=str2bool, nargs='?', const=True, default=False,
                        help="Save each of the checkpoints as the run progresses.")
    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help="Just test and terminate.")
    parser.add_argument('--validate', type=str2bool, nargs='?', const=True, default=False,
                        help="Just validate and terminate.")
    parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False,
                        help="Start from scratch (do not load).")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dataset_limit', type=int, default=0, help="Limits the dataset size, useful for debugging")
    parser.add_argument('--exportONNX', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--disable-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False,
                        help="verbose mode - print details every batch")

    args = parser.parse_args()

    if args.verbose:
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))
        print('===================================================')
        print('args.epochs           = %s' % args.epochs)
        print('args.reset            = %s' % args.reset)
        print('args.test             = %s' % args.test)
        print('args.validate         = %s' % args.validate)
        print('args.workers          = %s' % args.workers)
        print('args.data_path        = %s' % args.data_path)
        print('args.output_path      = %s' % args.output_path)
        print('args.save_checkpoints = %s' % args.save_checkpoints)
        print('args.exportONNX       = %s' % args.exportONNX)
        print('args.disable_cuda     = %d' % args.disable_cuda)
        print('args.verbose          = %d' % args.verbose)
        print('===================================================')

    # Change there flags to control what happens.
    doLoad = not args.reset  # Load checkpoint at the beginning
    doTest = args.test  # Only run test, no training
    doValidate = args.validate  # Only run validation, no training
    dataPath = args.data_path
    checkpointsPath = args.output_path
    exportONNX = args.exportONNX
    saveCheckpoints = args.save_checkpoints
    using_cuda = not args.disable_cuda
    verbose = args.verbose
    workers = args.workers
    epochs = args.epochs
    dataset_limit = args.dataset_limit

    if verbose:
        print('===================================================')
        print('doLoad                = %d' % doLoad)
        print('doTest                = %d' % doTest)
        print('doValidate            = %d' % doValidate)
        print('dataPath              = %s' % dataPath)
        print('checkpointsPath       = %s' % checkpointsPath)
        print('saveCheckpoints       = %d' % saveCheckpoints)
        print('workers               = %d' % workers)
        print('epochs                = %d' % epochs)
        print('exportONNX            = %d' % exportONNX)
        print('===================================================')

    return args, doLoad, doTest, doValidate, dataPath, checkpointsPath, exportONNX, saveCheckpoints, \
           using_cuda, workers, epochs, dataset_limit, verbose


if __name__ == "__main__":
    main()
    print('')
    print('DONE')
    print('')
