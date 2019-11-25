import argparse
import json
import os
import shutil
import sys  # for command line argument dumping
import time
from collections import OrderedDict
from datetime import datetime  # for timing
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel
from Utilities import AverageMeter, ProgressBar, SamplingBar

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
    exportONNX, saveCheckpoints, using_cuda, workers, epochs, \
    dataset_limit, verbose, device, deviceId = parse_commandline_arguments()

    if using_cuda and torch.cuda.device_count() > 0:
         # Change batch_size in commandLine args if out of cuda memory
        if args.deviceId < 0:
            batch_size = torch.cuda.device_count() * args.batch_size
        else:
            batch_size = args.batch_size
    else:
        batch_size = 1

    eval_RMSError= math.inf
    best_RMSError = math.inf
    lr = BASE_LR

    if verbose:
        print('')
        if using_cuda:
            print('CUDA DEVICE_COUNT {0}'.format(torch.cuda.device_count()))
        print('')

    # Retrieve model
    model = ITrackerModel().to(device=device)
    if using_cuda and args.deviceId < 0:
        model = torch.nn.DataParallel(model).to(device=device)

    image_size = (224, 224)
    cudnn.benchmark = True

    epoch = 1
    if doLoad:
        saved = load_checkpoint(checkpointsPath, device)
        if saved:
            epoch = saved['epoch']
            best_RMSError = saved['best_RMSError']
            print(
                'Loading checkpoint : [Epoch: %d | RMSError: %.5f].' % (
                    epoch,
                    best_RMSError)
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
        else:
            print('Warning: Could not read checkpoint!')

    print('epoch = %d' % epoch)

    totalstart_time = datetime.now()

    datasets = load_all_data(dataPath, image_size, workers, batch_size, verbose)

    #     criterion = nn.MSELoss(reduction='sum').to(device=device)
    criterion = nn.MSELoss(reduction='mean').to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    if doTest:
        # Quick test
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = test(datasets, model, criterion, 1, checkpointsPath, batch_size, device, dataset_limit, verbose)
        time_elapsed = datetime.now() - start_time
        print('')
        print('Testing MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
        print('Testing Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif doValidate:
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = validate(datasets, model, criterion, 1, checkpointsPath, batch_size, device, dataset_limit, verbose)
        time_elapsed = datetime.now() - start_time
        print('')  # print blank line after loading data
        print('Validation MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
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

        if args.hsm:
            args.multinomial_weights = torch.ones(datasets['train']['size'], dtype=torch.double)
            if not verbose:
                args.sampling_bar = SamplingBar('HSM')

        # now start training from last best epoch
        for epoch in range(epoch, epochs + 1):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, epochs))
            start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            print('\nEpoch:{} [device:{}, lr:{}, best_RMSError:{:2.4f}, hsm:{}, adv:{}]'.format(epoch, device, lr, best_RMSError, args.hsm, args.adv))
            train_MSELoss, train_RMSError = train(datasets['train'], model, criterion, optimizer, epoch, batch_size, device, dataset_limit, verbose, args)

            # evaluate on validation set
            eval_MSELoss, eval_RMSError = validate(datasets, model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit, verbose)

            # remember best RMSError and save checkpoint
            is_best = eval_RMSError < best_RMSError
            best_RMSError = min(eval_RMSError, best_RMSError)

            time_elapsed = datetime.now() - start_time

            if run:
                run.log('MSELoss', eval_MSELoss)
                run.log('RMSLoss', eval_RMSError)
                run.log('best MSELoss', best_RMSError)
                run.log('epoch time', time_elapsed)

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_RMSError': best_RMSError,
            },
                is_best,
                checkpointsPath,
                saveCheckpoints)

            print('')
            print('Epoch %05d with RMSError %.5f' % (epoch, best_RMSError))
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))

# Fast Gradient Sign Attack (FGSA)
def adversarialAttack(image, data_grad, epsilon=0.1):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def euclideanBatchError(output, target):
    """ For a batch of output and target returns corresponding batch of euclidean errors
    """
    # Batch Euclidean Distance sqrt(dx^2 + dy^2)
    return torch.sqrt(torch.sum(torch.pow(output - target, 2),1))


def train(dataset, model, criterion, optimizer, epoch, batch_size, device, dataset_limit=None, verbose=False, args=None):
    data_size = dataset['size']
    loader = dataset['loader']
    split = dataset['split']

    batch_time = AverageMeter()
    data_time = AverageMeter()
    MSELosses = AverageMeter()
    RMSErrors = AverageMeter()
    num_samples = 0

    if not verbose:
        progress_bar = ProgressBar(max_value=data_size, label=split)

    # switch to train mode
    model.train()

    end = time.time()

	# HSM Update - Every epoch
    if args.hsm:
        # Reset every few epoch (hsm_cycle)
        if epoch%args.hsm_cycle == 0:
            args.multinomial_weights = torch.ones(data_size, dtype=torch.double)
        # update dataloader and sampler
        sampler = torch.utils.data.WeightedRandomSampler(args.multinomial_weights, int(len(args.multinomial_weights)), replacement=True)
        loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=batch_size, sampler=sampler,
            num_workers=args.workers)
        # Line-space for HSM meter
        print('')
        if not verbose:
            args.sampling_bar.display(args.multinomial_weights)

	# load data samples and train
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(loader):
        batchNum = i + 1
        actual_batch_size = imFace.size(0)
        num_samples += actual_batch_size

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
        error = euclideanBatchError(output, gaze)

        if args.hsm:
            # update sample weights to be the loss, so that harder samples have larger chances to be drawn in the next epoch
            # normalize and threshold prob values at max value '1'
            batch_loss = error.detach().cpu().div_(10.0)
            # batch_loss = error.detach().cpu().div_(best_MSELoss*2)
            batch_loss[batch_loss>1.0] = 1.0
            args.multinomial_weights.scatter_(0, indices, batch_loss.type_as(torch.DoubleTensor()))

        # average over the batch
        error = torch.mean(error)
        MSELosses.update(loss.data.item(), actual_batch_size)
        RMSErrors.update(error.item(), actual_batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.adv:
            # Backprop the loss while retaining the graph to backprop again
            loss.backward(retain_graph=True)

            # Collect gradInput
            imFace_grad = imFace.grad.data
            imEyeL_grad = imEyeL.grad.data
            imEyeR_grad = imEyeR.grad.data
            faceGrid_grad = faceGrid.grad.data

            # Generate perturbed input for Adversarial Attack
            perturbed_imFace = adversarialAttack(imFace, imFace_grad)
            perturbed_imEyeL = adversarialAttack(imEyeL, imEyeL_grad)
            perturbed_imEyeR = adversarialAttack(imEyeR, imEyeR_grad)
            perturbed_faceGrid = adversarialAttack(faceGrid, faceGrid_grad)

            # Regenerate output for the perturbed input
            output_adv = model(perturbed_imFace, perturbed_imEyeL, perturbed_imEyeR, perturbed_faceGrid)
            loss_adv = criterion(output_adv, gaze)

            # concatenate both real and adversarial loss functions
            loss = loss + loss_adv

        # backprop the loss
        loss.backward()

        # optimize
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Epoch (train): [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'MSELoss {MSELosses.val:.4f} ({MSELosses.avg:.4f})\t'
                  'RMSError {RMSErrors.val:.4f} ({RMSErrors.avg:.4f})\t'.format(
                    epoch, batchNum, len(loader), batch_time=batch_time,
                    data_time=data_time, MSELosses=MSELosses, RMSErrors=RMSErrors))
        else:
            progress_bar.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    return MSELosses.avg, RMSErrors.avg

def evaluate(dataset, model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit=None, verbose=False):
    data_size = dataset['size']
    loader = dataset['loader']
    split = dataset['split']

    batch_time = AverageMeter()
    data_time = AverageMeter()
    MSELosses = AverageMeter()
    RMSErrors = AverageMeter()
    num_samples = 0

    if not verbose:
        progress_bar = ProgressBar(max_value=data_size, label=split)

    # switch to evaluate mode
    model.eval()

    end = time.time()

    results = []

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(loader):
        batchNum = i + 1
        actual_batch_size = imFace.size(0)
        num_samples += actual_batch_size

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
        error = euclideanBatchError(output, gaze)

        # average over the batch
        error = torch.mean(error)
        MSELosses.update(loss.data.item(), actual_batch_size)
        RMSErrors.update(error.item(), actual_batch_size)

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Epoch ({}): [{}][{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'MSELoss {MSELosses.val:.4f} ({MSELosses.avg:.4f})\t'
              'RMSError {RMSErrors.val:.4f} ({RMSErrors.avg:.4f})\t'.format(
                stage, epoch, batchNum, len(eval_loader), batch_time=batch_time,
                MSELosses=MSELosses, RMSErrors=RMSErrors))
        else:
            progress_bar.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    resultsFileName = os.path.join(checkpointsPath, 'results.json')
    with open(resultsFileName, 'w+') as outfile:
        json.dump(results, outfile)

    return MSELosses.avg, RMSErrors.avg


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
    shuffle = True if split == 'train' else False
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True)

    return {
        'split': split,
        'data': data,
        'size': size,
        'loader': loader
    }

def centeredText(infoString, marker='-', length=40):
    marker = marker*length
    index = (len(marker)-len(infoString))//2
    return marker[:index] + infoString + marker[index + len(infoString):]

def load_all_data(path, image_size, workers, batch_size, verbose):
    print(centeredText('Loading Data'))
    all_data = {
        # training data : model sees and learns from this data
        'train': load_data('train', path, image_size, workers, batch_size, verbose),
        # validation data : model sees but never learns from this data
        'val': load_data('val', path, image_size, workers, batch_size, verbose),
        # test data : model never sees or learns from this data
        'test': load_data('test', path, image_size, workers, batch_size, verbose)
    }
    return all_data

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
    # Experimental options
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--deviceId', type=int, default=0)
    parser.add_argument('--hsm', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--hsm_cycle', type=int, default=8)
    parser.add_argument('--adv', type=str2bool, nargs='?', const=True, default=False, help="")
    args = parser.parse_args()

    args.device = None
    usingCuda = False
    if not args.disable_cuda and torch.cuda.is_available():
        usingCuda = True
        if args.deviceId < 0:
            deviceId = -1
            args.device = torch.device('cuda')
        else:
            if 0 <= args.deviceId < torch.cuda.device_count():
                torch.cuda.set_device(args.deviceId)
            else:
                print("Device id can't exeed {}, default to currently set device gpu{}.".format(torch.cuda.device_count()-1), torch.cuda.current_device())

            deviceId = torch.cuda.current_device()
            args.device = torch.device('cuda:'+str(deviceId)) 
    else:
        args.device = torch.device('cpu')
        deviceId = 0

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
    device = args.device

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
        using_cuda, workers, epochs, dataset_limit, verbose, device, deviceId


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print('Thread is killed.')
        sys.exit()
    print('')
    print('DONE')
    print('')
