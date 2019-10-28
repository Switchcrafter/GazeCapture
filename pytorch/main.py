import argparse
import json
import os
import shutil
import sys  # for command line argument dumping
import time
from collections import OrderedDict
from datetime import datetime  # for timing
import progressbar

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

try:
    from azureml.core.run import Run
except ImportError:
    Run = None

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

OUTPUT_PATH = os.path.dirname(os.path.realpath(__file__))


# widgets=[
#         'Progress',#0
#         ' ',#1
#         progressbar.Bar(marker='■', left='|', right='|', fill='-'),#2
#         progressbar.SimpleProgress(),#3
#         '[',progressbar.ETA(),']', #5 
#         '[','RMSError',']',#8
#     ]
# bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=widgets)
# def progressbar_update(value, max, label, error):
#     # update label
#     label = '{:8}'.format(label)
#     if bar.widgets[0] != label:
#         bar.widgets[0] = label
#     if bar.widgets[8] != error:
#         bar.widgets[8] = error
#     # update max_value
#     if bar.max_value != max:
#         bar.max_value = max
#     # update value
#     bar.update(value)
#     # update finish
#     if value >= bar.max_value:
#         bar.finish()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if Run:
    run = Run.get_context()

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
parser.add_argument('--dataset_size', type=int, default=0)
parser.add_argument('--exportONNX', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False,
                    help="verbose mode - print details every batch")
args = parser.parse_args()
args.device = None
usingCuda = False
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    usingCuda = True
else:
    args.device = torch.device('cpu')

# Change there flags to control what happens.
doLoad = not args.reset  # Load checkpoint at the beginning
doTest = args.test  # Only run test, no training
doValidate = args.validate  # Only run validation, no training
dataPath = args.data_path
checkpointsPath = args.output_path
exportONNX = args.exportONNX
saveCheckpoints = args.save_checkpoints

workers = args.workers
epochs = args.epochs

if usingCuda and torch.cuda.device_count() > 0:
    batch_size = torch.cuda.device_count() * 100  # Change if out of cuda memory
else:
    batch_size = 5

base_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

dataset_size = args.dataset_size

def main():
    global args, best_prec1, weight_decay, momentum, data_size
    
    if args.verbose:
        print('CUDA DEVICE_COUNT {0}'.format(torch.cuda.device_count()))
        print('')
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

    model = ITrackerModel().to(device=args.device)

    if usingCuda:
        model = torch.nn.DataParallel(model).to(device=args.device)

    imSize = (224, 224)
    cudnn.benchmark = False

    epoch = 1
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print(
                'Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (
                    saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']

            state = remove_module_from_state(state)

            model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')
    
    print('epoch = %d' % epoch)
    
    totalstart_time = datetime.now()
    
    # training data : model sees and learns from this data 
    data_train = ITrackerData(dataPath, split='train', imSize=imSize, silent = not args.verbose)
    # validation data : model sees but never learns from this data 
    data_val = ITrackerData(dataPath, split='val', imSize=imSize, silent = not args.verbose)
    # test data : model never sees or learns from this data 
    data_test = ITrackerData(dataPath, split='test', imSize=imSize, silent = not args.verbose)
    
    data_size = {'train':len(data_train.indices), 'val':len(data_val.indices), 'test':len(data_test.indices)}
    
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    
#     criterion = nn.MSELoss(reduction='sum').to(device=args.device)
    criterion = nn.MSELoss(reduction='mean').to(device=args.device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    
    # Quick test
    if doTest:
        start_time = datetime.now()
        precision = test(test_loader, model, criterion, epoch=0)
        time_elapsed = datetime.now() - start_time
        print('Testing loss %.5f' % (precision))
        print('Testing Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif doValidate:
        start_time = datetime.now()
        precision = validate(val_loader, model, criterion, epoch=0)
        time_elapsed = datetime.now() - start_time
        print('Validation loss %.5f' % (precision))
        print('Validation Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif exportONNX:
        export_onnx_model(val_loader, model)
    else:#Train
        # first cmake a learning_rate correction suitable for epoch from saved checkpoint
        # epoch will be non-zero if a checkpoint was loaded
        for epoch in range(1, epoch):
            if args.verbose:
                print('Epoch %05d of %05d - adjust learning rate only' % (epoch, epochs))
                start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)
            if args.verbose:
                time_elapsed = datetime.now() - start_time
                print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
        
        # now start training from last best epoch
        for epoch in range(epoch, epochs):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, epochs))
            start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            print('\nEpoch:{} [device:{}, id:{}, lr:{}]'.format(epoch, args.device, torch.cuda.current_device(), lr))
            train_error = train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(test_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)

            time_elapsed = datetime.now() - start_time

            if Run:
                run.log('precision', prec1)
                run.log('best precision', best_prec1)
                run.log('epoch time', time_elapsed)

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

            print('Epoch %05d with loss %.5f' % (epoch, best_prec1))
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))


def train(train_loader, model, criterion, optimizer, epoch):
    global count
    global dataset_size
    global data_train
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    progress_meter = ProgressMeter()
    num_samples = 0

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame) in enumerate(train_loader):
        batchNum = i+1
        num_samples += imFace.size(0)
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.to(device=args.device)
        imEyeL = imEyeL.to(device=args.device)
        imEyeR = imEyeR.to(device=args.device)
        faceGrid = faceGrid.to(device=args.device)
        gaze = gaze.to(device=args.device)

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
        lossesLin.update(lossLin.item()/batch_size, imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count = count + 1
        
        if args.verbose:
            print('Epoch (train): [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'MSELoss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'RMSErr {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, batchNum, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, lossLin=lossesLin))
        else:
            progress_meter.update(num_samples, data_size['train'], 'train', lossesLin.avg)
            
        if 0 < dataset_size < batchNum:
            breakvalvalvalvalval
    
    return lossesLin.avg

def evaluate(eval_loader, model, criterion, epoch, stage):
    global count_test
    global dataset_size
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    progress_meter = ProgressMeter()
    num_samples = 0

    # switch to evaluate mode
    model.eval()
    end = time.time()

    results = []

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame) in enumerate(eval_loader):
        batchNum = i+1
        num_samples += imFace.size(0)
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.to(device=args.device)
        imEyeL = imEyeL.to(device=args.device)
        imEyeR = imEyeR.to(device=args.device)
        faceGrid = faceGrid.to(device=args.device)
        gaze = gaze.to(device=args.device)

        imFace = torch.autograd.Variable(imFace, requires_grad=False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad=False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad=False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad=False)
        gaze = torch.autograd.Variable(gaze, requires_grad=False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        # Combine the tensor results together into a colated list so that we have the gazePoint and gazePrediction
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
        lossesLin.update(lossLin.item()/batch_size, imFace.size(0))

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.verbose:
            print('Epoch ({}): [{}][{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'MSELoss {loss.val:.4f} ({loss.avg:.4f})\t'
              'RMSErr {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                stage, epoch, batchNum, len(eval_loader), batch_time=batch_time,
                loss=losses, lossLin=lossesLin))
        else:
            progress_meter.update(num_samples, data_size[stage], stage, lossesLin.avg)
            
        if 0 < dataset_size < batchNum:
            break
    
    resultsFileName = os.path.join(checkpointsPath, 'results.json')
    with open(resultsFileName, 'w+') as outfile:
        json.dump(results, outfile)

    return lossesLin.avg

def validate(val_loader, model, criterion, epoch):
    return evaluate(val_loader, model, criterion, epoch, 'val')

def test(test_loader, model, criterion, epoch):
    return evaluate(test_loader, model, criterion, epoch, 'test')

def export_onnx_model(val_loader, model):
    global count_test
    global dataset_size

    # switch to evaluate mode
    model.eval()

    batch_size = 1
    color_depth = 3  # 3 bytes for RGB color space
    dim_width = 224
    dim_height = 224
    face_grid_size = 25 * 25

    imFace = torch.randn(batch_size, color_depth, dim_width, dim_height).to(device=args.device).float()
    imEyeL = torch.randn(batch_size, color_depth, dim_width, dim_height).to(device=args.device).float()
    imEyeR = torch.randn(batch_size, color_depth, dim_width, dim_height).to(device=args.device).float()
    faceGrid = torch.randn(batch_size, face_grid_size).to(device=args.device).float()

    dummy_in = (imFace, imEyeL, imEyeR, faceGrid)

    in_names = ["faces", "eyesLeft", "eyesRight", "faceGrids"]
    out_names = ["x"]

    torch.onnx.export(model.module,
                      dummy_in,
                      "itracker.onnx",
                      input_names=in_names,
                      output_names=out_names,
                      opset_version=7,
                      verbose=True)


def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(checkpointsPath, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename, map_location=args.device)
    return state


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
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
        self.widgets=[
            'Progress',#0
            ' ',#1
            progressbar.Bar(marker='■', left='|', right='|', fill='-'),#2
            '[',progressbar.SimpleProgress(),']',#4
            '[',progressbar.ETA(),']', #7 
            '[','RMSError',']',#10
        ]
        self.bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=self.widgets)
     
    def update(self, value, max, label, error):
        # update label
        label = '{:5}'.format(label)
        if self.bar.widgets[0] != label:
            self.bar.widgets[0] = label
        
        #update metric
        metric = '{metric:.4f}'.format(metric=error)
        if self.bar.widgets[10] != metric:
            self.bar.widgets[10] = metric
        
        # update max_value
        if self.bar.max_value != max:
            self.bar.max_value = max
        # update value
        self.bar.update(value)
        # update finish
        if value >= self.bar.max_value:
            self.bar.finish()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('')
    print('DONE')
    print('')
