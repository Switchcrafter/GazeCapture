import math, shutil, os, time, argparse, json
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

from datetime import datetime # for timing
import sys # for command line argument dumping

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
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.", default='/data/gc-data-prepped/')
parser.add_argument('--output_path', help="Path to checkpoint", default=os.path.dirname(os.path.realpath(__file__)))
parser.add_argument('--save_checkpoints', type=str2bool, nargs='?', const=True, default=False, help="Save each of the checkpoints as the run progresses.")
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--dataset-size', type=int, default=0)
parser.add_argument('--ONNX', type=str2bool, nargs='?', const=True, default=False)
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training
dataPath = args.data_path
checkpointsPath = args.output_path
outputONNX = args.ONNX
saveCheckpoints = args.save_checkpoints

workers = args.workers
epochs = args.epochs
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

base_lr = 0.0001
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
    global args, best_prec1, weight_decay, momentum

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print('')
    print('DEVICE_COUNT {0}'.format(torch.cuda.device_count()))
    print('args.epochs           = %s' % args.epochs)
    print('args.reset            = %s' % args.reset)
    print('args.sink             = %s' % args.sink)
    print('args.workers          = %s' % args.workers)
    print('args.data_path        = %s' % args.data_path)
    print('args.output_path      = %s' % args.output_path)
    print('args.save_checkpoints = %s' % args.save_checkpoints)
    print('args.ONNX             = %s' % args.ONNX)
    print('')
    print('doLoad                = %d' % doLoad)
    print('doTest                = %d' % doTest)
    print('dataPath              = %s' % dataPath)
    print('checkpointsPath       = %s' % checkpointsPath)
    print('saveCheckpoints       = %d' % saveCheckpoints)
    print('workers               = %d' % workers)
    print('epochs                = %d' % epochs)
    print('outputONNX            = %d' % outputONNX)


    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(224,224)
    cudnn.benchmark = True

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    print('epoch = %d' % (epoch))

    totalstart_time = datetime.now()
    
    dataTrain = ITrackerData(dataPath, split='train', imSize = imSize)
    dataVal = ITrackerData(dataPath, split='test', imSize = imSize)
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # Quick test
    if doTest:
        print('doTest - Validating only')
        print('\nValidation Started')
        validate(val_loader, model, criterion, epoch)
        print('\nValidation Completed')
    elif outputONNX:
        exportONNX(val_loader, model)
    else:
        for epoch in range(0, epoch):
            print('Epoch %05d of %05d - adjust learning rate only' % (epoch, epochs))
            start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)
            time_elapsed = datetime.now() - start_time
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

        for epoch in range(epoch, epochs):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, epochs))
            start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            print('\nTraining Started')
            train(train_loader, model, criterion, optimizer, epoch)
            print('\nTraining Completed')

            # evaluate on validation set
            print('\nValidation Started')
            prec1 = validate(val_loader, model, criterion, epoch)
            print('\nValidation Completed')

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)

            time_elapsed = datetime.now() - start_time

            if Run:
                run.log('precision', prec1)
                run.log('best precision', best_prec1)
                run.log('epoch time', time_elapsed)

            save_checkpoint({
               'epoch': epoch + 1,
               'state_dict': model.state_dict(),
               'best_prec1': best_prec1,
                }, is_best)

            print('Epoch %05d with loss %.5f' % (epoch, best_prec1))
            
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))

def train(train_loader, model, criterion,optimizer, epoch):
    global count
    global dataset_size
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = True)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        losses.update(loss.data.item(), imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        if dataset_size > 0 and dataset_size < i + 1:
            break

def validate(val_loader, model, criterion, epoch):
    global count_test
    global dataset_size
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    results = []

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        # Combine the tensor results together into a colated list so that we have the gazePoint and gazePrediction for each frame
        f1 = frame.cpu().numpy().tolist()
        g1 = gaze.cpu().numpy().tolist()
        o1 = output.cpu().numpy().tolist()
        r1 = [list(r) for r in zip(f1, g1, o1)]

        def convertResult(result):
            
            r = {}

            r['frame'] = result[0]
            r['gazePoint'] = result[1]
            r['gazePrediction'] = result[2]

            return r

        results += list(map(convertResult, r1))

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))
        if dataset_size > 0 and dataset_size < i + 1:
            break

    resultsFileName = os.path.join(checkpointsPath, 'results.json')
    with open(resultsFileName, 'w+') as outfile:
        json.dump(results, outfile)

    return lossesLin.avg

def exportONNX(val_loader, model):
    global count_test
    global dataset_size

    # switch to evaluate mode
    model.eval()

    batch_size = 1
    color_depth = 3 # 3 bytes for RGB color space
    dim_width = 224
    dim_height = 224
    face_grid_size = 25 * 25

    imFace = torch.randn(batch_size, color_depth, dim_width, dim_height).cuda()
    imEyeL = torch.randn(batch_size, color_depth, dim_width, dim_height).cuda()
    imEyeR = torch.randn(batch_size, color_depth, dim_width, dim_height).cuda()
    faceGrid = torch.randn(batch_size, face_grid_size).cuda()

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
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    resultsFilename = os.path.join(checkpointsPath, 'results.json')
    checkpointFilename = os.path.join(checkpointsPath, filename)

    torch.save(state, checkpointFilename)
    
    if saveCheckpoints:
        shutil.copyfile(checkpointFilename, os.path.join(checkpointsPath, 'checkpoint' + str(state['epoch']) + '.pth.tar'))
        shutil.copyfile(resultsFilename, os.path.join(checkpointsPath, 'results' + str(state['epoch']) + '.json'))

    bestFilename = os.path.join(checkpointsPath, 'best_' + filename)
    bestResultsFilename = os.path.join(checkpointsPath, 'best_results.json')

    if is_best:
        shutil.copyfile(checkpointFilename, bestFilename)
        shutil.copyfile(resultsFilename, bestResultsFilename)

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
