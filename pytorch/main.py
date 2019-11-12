import argparse
import json
import os
import shutil
import sys  # for command line argument dumping
import time
from collections import OrderedDict
from datetime import datetime  # for timing
import shutil
import math

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
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--dataset_size', type=int, default=0)
parser.add_argument('--exportONNX', type=str2bool, nargs='?', const=True, default=False)
# GPU Settings options
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--deviceId', type=int, default=0)
parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help="verbose mode: print every batch")
# Advanced experimental options
parser.add_argument('--hsm', type=str2bool, nargs='?', const=True, default=False, help="")
parser.add_argument('--adv', type=str2bool, nargs='?', const=True, default=False, help="")
args = parser.parse_args()

args.device = None
usingCuda = False
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    usingCuda = True
    if 0 <= args.deviceId < torch.cuda.device_count():
        torch.cuda.set_device(args.deviceId)
    else:
        print("Device id can't exeed {}, default to currently set device gpu{}.".format(torch.cuda.device_count()-1), torch.cuda.current_device())
    deviceId = torch.cuda.current_device()
else:
    args.device = torch.device('cpu')
    deviceId = 0

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

batch_size = args.batch_size
# if usingCuda and torch.cuda.device_count() > 0:
#     batch_size = torch.cuda.device_count() * 100  # Change if out of cuda memory
# else:
#     batch_size = 5

base_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
eval_MSELoss = math.inf
best_MSELoss = math.inf
lr = base_lr

count_test = 0
count = 0

dataset_size = args.dataset_size

def main():
    global args, best_MSELoss, weight_decay, momentum
    global data_size, data_train, sampler, sampling_meter
    global multinomial_weights

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
        if args.validate or args.test:
            saved = load_checkpoint(filename='best_checkpoint.pth.tar')
        else:
            saved = load_checkpoint()
        if saved:
            print(
                'Loading checkpoint [ Epoch:%05d => MSELoss %.5f ].' % (
                    saved['epoch'], saved['best_MSELoss']))
            state = saved['state_dict']

            if not usingCuda:
                # when using Cuda for training we use DataParallel. When using DataParallel, there is a
                # 'module.' added to the namespace of the item in the dictionary.
                # remove 'module.' from the front of the name to make it compatible with cpu only
                state = OrderedDict()
                for key, value in saved['state_dict'].items():
                    state[key[7:]] = value.to(device=args.device)

            model.load_state_dict(state)
            epoch = saved['epoch']
            best_MSELoss = saved['best_MSELoss']
        else:
            print('Warning: Could not read checkpoint!')

    print('epoch = %d' % epoch)

    totalstart_time = datetime.now()

    # Dataset
    # training data : model sees and learns from this data
    data_train = ITrackerData(dataPath, split='train', imSize=imSize, silent = not args.verbose)
    # validation data : model sees but never learns from this data
    data_val = ITrackerData(dataPath, split='val', imSize=imSize, silent = not args.verbose)
    # test data : model never sees or learns from this data
    data_test = ITrackerData(dataPath, split='test', imSize=imSize, silent = not args.verbose)

    data_size = {'train':len(data_train.indices), 'val':len(data_val.indices), 'test':len(data_test.indices)}

    # Sampling Strategy
    sampler = torch.utils.data.RandomSampler(data_train, replacement=False, num_samples=None)
    if args.hsm:
        multinomial_weights = torch.ones(len(data_train), dtype=torch.double)

    # TODO: Batch sampling - Will implement in future
    # batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
    # Dataloader using the sampling strategy
    # train_loader = torch.utils.data.DataLoader(
    #     data_train,
    #     batch_sampler=batch_sampler,
    #     num_workers=workers)

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size, sampler=sampler,
        num_workers=workers)

    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    criterion = nn.MSELoss(reduction='mean').to(device=args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr,
    #                             weight_decay=weight_decay)

    # Quick test
    if doTest:
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = test(test_loader, model, criterion, epoch=1)
        time_elapsed = datetime.now() - start_time
        print('Testing MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
        print('Testing Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif doValidate:
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = validate(val_loader, model, criterion, epoch=1)
        time_elapsed = datetime.now() - start_time
        print('Validation MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
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

        if args.hsm and not args.verbose:
            sampling_meter = SamplingMeter('HSM')

        # now start training from last best epoch
        for epoch in range(epoch, epochs):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, epochs))
            start_time = datetime.now()
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            print('\nEpoch:{} [device:{}{}, lr:{}, best_MSELoss:{:2.4f}, hcm:{}, adv:{}]'.format(epoch, args.device, deviceId, lr, best_MSELoss, args.hsm, args.adv))
            train_MSELoss, train_RMSError = train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            eval_MSELoss, eval_RMSError = validate(val_loader, model, criterion, epoch)

            # remember best MSELoss and save checkpoint
            is_best = eval_MSELoss < best_MSELoss
            best_MSELoss = min(eval_MSELoss, best_MSELoss)

            time_elapsed = datetime.now() - start_time

            if Run:
                run.log('MSELoss', eval_MSELoss)
                run.log('best MSELoss', best_MSELoss)
                run.log('epoch time', time_elapsed)

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_MSELoss': best_MSELoss,
            }, is_best)

            print('Epoch %05d with loss %.5f' % (epoch, best_MSELoss))
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))

# adversarial attack code
# Fast Gradient Sign Attack (FGSA)
def adversarial_attack(image, data_grad, epsilon=0.1):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def train(train_loader, model, criterion, optimizer, epoch):
    global count
    global dataset_size
    global multinomial_weights
    global sampler
    global data_train
    global sampling_meter

    stage = 'train'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    MSELosses = AverageMeter()
    RMSErrors = AverageMeter()

    # todo: init only for verbose disabled
    if not args.verbose:
        progress_meter = ProgressMeter(max_value=data_size[stage], label=stage)
    num_samples = 0

    # switch to train mode
    model.train()

    end = time.time()

    # HSM Update - Every epoch
    if args.hsm:
        # Reset every 15th epoch
        if epoch%15 == 0:
            multinomial_weights = torch.ones(len(data_train), dtype=torch.double)
        # update dataloader and sampler
        sampler = torch.utils.data.WeightedRandomSampler(multinomial_weights, int(len(multinomial_weights)), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=batch_size, sampler=sampler,
            num_workers=workers)
        # Line-space for HSM meter
        print('')
        if not args.verbose:
            sampling_meter.display(multinomial_weights)

    # load data samples and train
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(train_loader):
        batchNum = i+1
        actual_batch_size = imFace.size(0)
        num_samples += actual_batch_size

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

        error = output - gaze
        error = torch.mul(error, error)
        error = torch.sum(error, 1)
        error = torch.sqrt(error) #Batch Eucledean Distance sqrt(dx^2 + dy^2)

        if args.hsm:
            # update sample weights to be the loss, so that harder samples have larger chances to be drawn in the next epoch
            # normalize and threshold prob values at max value '1'
            batch_loss = error.detach().cpu().div_(10.0)
            # batch_loss = error.detach().cpu().div_(best_MSELoss*2)
            batch_loss[batch_loss>1.0] = 1.0
            multinomial_weights.scatter_(0, indices, batch_loss.type_as(torch.DoubleTensor()))
            # if not args.verbose:
            #     sampling_meter.display(multinomial_weights)

        # average over the batch
        error = torch.mean(error)
        MSELosses.update(loss.data.item(), actual_batch_size)
        RMSErrors.update(error.item(), actual_batch_size)

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.adv:
            # Collect gradInput
            imFace_grad = imFace.grad.data
            imEyeL_grad = imEyeL.grad.data
            imEyeR_grad = imEyeR.grad.data
            faceGrid_grad = faceGrid.grad.data

            # Call Adversarial Attack
            perturbed_imFace = adversarial_attack(imFace, imFace_grad)
            perturbed_imEyeL = adversarial_attack(imEyeL, imEyeL_grad)
            perturbed_imEyeR = adversarial_attack(imEyeR, imEyeR_grad)
            perturbed_faceGrid = adversarial_attack(faceGrid, faceGrid_grad)

            # Re-classify the perturbed image
            output_adv = model(perturbed_imFace, perturbed_imEyeL, perturbed_imEyeR, perturbed_faceGrid)
            loss_adv = criterion(output_adv, gaze)
            # loss_adv.backward()
            # concatenate both real and adversarial loss functions
            loss = loss + loss_adv
            loss.backward()

        # optimize
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count = count + 1

        if args.verbose:
            print('Epoch (train): [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'MSELoss {MSELosses.val:.4f} ({MSELosses.avg:.4f})\t'
                  'RMSError {RMSErrors.val:.4f} ({RMSErrors.avg:.4f})\t'.format(
                    epoch, batchNum, len(train_loader), batch_time=batch_time,
                    data_time=data_time, MSELosses=MSELosses, RMSErrors=RMSErrors))
        else:
            progress_meter.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if 0 < dataset_size < batchNum:
            break

    return MSELosses.avg, RMSErrors.avg

def evaluate(eval_loader, model, criterion, epoch, stage):
    global count_test
    global dataset_size
    batch_time = AverageMeter()
    data_time = AverageMeter()
    MSELosses = AverageMeter()
    RMSErrors = AverageMeter()
    progress_meter = ProgressMeter(max_value=data_size[stage], label=stage)
    num_samples = 0

    # switch to evaluate mode
    model.eval()
    end = time.time()

    results = []

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(eval_loader):
        batchNum = i+1
        actual_batch_size = imFace.size(0)
        num_samples += actual_batch_size

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

        # Combine the tensor results together into a colated list so that we have the gazePoint and gazePrediction for each frame
        f1 = frame.cpu().numpy().tolist()
        g1 = gaze.cpu().numpy().tolist()
        o1 = output.cpu().numpy().tolist()
        r1 = [list(r) for r in zip(f1, g1, o1)]

        def convertResult(result):
            r = {'frame': result[0], 'gazePoint': result[1], 'gazePrediction': result[2]}
            return r

        results += list(map(convertResult, r1))
        loss = criterion(output, gaze)

        error = output - gaze
        error = torch.mul(error, error)
        error = torch.sum(error, 1) #Batch MSDistance
        error = torch.sqrt(error) #Batch RMSDistance

        # average over the batch
        error = torch.mean(error)
        MSELosses.update(loss.data.item(), actual_batch_size)
        RMSErrors.update(error.item(), actual_batch_size)

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose:
            print('Epoch ({}): [{}][{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'MSELoss {MSELosses.val:.4f} ({MSELosses.avg:.4f})\t'
              'RMSError {RMSErrors.val:.4f} ({RMSErrors.avg:.4f})\t'.format(
                stage, epoch, batchNum, len(eval_loader), batch_time=batch_time,
                MSELosses=MSELosses, RMSErrors=RMSErrors))
        else:
            progress_meter.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if 0 < dataset_size < batchNum:
            break

    resultsFileName = os.path.join(checkpointsPath, 'results.json')
    with open(resultsFileName, 'w+') as outfile:
        json.dump(results, outfile)

    return MSELosses.avg, RMSErrors.avg

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def cleanup_stop_thread():
    print('Thread is killed.')

################################################################
# Utility classes
################################################################
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
    '''A progress bar which stretches to fill the line.'''
    def __init__(self, max_value=100, label='', marker='#', left='|', right='|', arror = '>', fill='-'):
        '''Creates a customizable progress bar.
        max_value - max possible value for the progressbar
        label - title for the progressbar as prefix
        marker - string or callable object to use as a marker
        left - string or callable object to use as a left border
        right - string or callable object to use as a right border
        fill - character to use for the empty part of the progress bar
        '''
        self.label = '{:5}'.format(label)
        self.left = '|'
        self.marker = '■' # alt: '#'
        self.arrow = '▶' # alt: '>'
        self.right = '|'
        self.fill = '-'
        self.max_value = max_value
        self.start_time = datetime.now()

    def create_marker(self, value, width):
        if self.max_value > 0:
            length = int(value / self.max_value * width)
            if length == width:
                return (self.marker * length)
            elif length == 0:
                return ''
            else:
                marker = (self.marker * (length-1)) + self.arrow
        else:
            marker = self.marker
        return marker

    def getTerminalWidth(self):
        default_width = 80
        default_height = 20
        size_tuple = shutil.get_terminal_size((default_width, default_height))  # pass fallback
        return size_tuple.columns

    def update(self, value, metric, error):
        '''Updates the progress bar and its subcomponents'''

        metric = '[{metric:.4f}]'.format(metric=metric) if metric else ''
        error = '[{error:.4f}]'.format(error=error) if error else ''

        time_elapsed = ' [Time: '+str(datetime.now() - self.start_time)+']'
        assert( value <= self.max_value), 'ProgressBar value (' + str(value) + ') can not exceed max_value ('+ str(self.max_value)+').'
        width = self.getTerminalWidth() - (len(self.label)+len(self.left)+len(self.right)+len(metric)+len(error)+len(time_elapsed))
        marker = self.create_marker(value, width).ljust(width, self.fill)
        marker = self.left + marker + self.right
        # append infoString at the center
        infoString = ' {val:d}/{max:d} ({percent:d}%) '.format(val=value, max=self.max_value, percent=int(value/self.max_value*100))
        index = (len(marker)-len(infoString))//2
        marker = marker[:index] + infoString + marker[index + len(infoString):]
        # print('\r'+self.label + marker + metric + time_elapsed, end= '' if value < self.max_value else '\n')
        print(self.label + marker + metric + error + time_elapsed, end= '\r' if value < self.max_value else '\n')

class SamplingMeter(object):
    '''A sampling hotness bar which stretches to fill the line.'''
    def __init__(self, label='', left='|', right='|'):
        '''Creates a multinomial sampling hotness bar.
        '''
        self.label = '{:5}'.format(label)
        self.left = '|'
        self.right = '|'

    def getTerminalWidth(self):
        default_width = 80
        default_height = 20
        size_tuple = shutil.get_terminal_size((default_width, default_height))  # pass fallback
        return size_tuple.columns

    ##  colorCodes = {black, VIBGYOR, White}
    def getCode(self, value=0.1, max=1.0, s='█'):
        colorCodes = ["\033[30m", "\033[1;30m", "\033[35m", "\033[1;35m", "\033[34m", "\033[1;34m", "\033[36m", "\033[32m", "\033[1;32m", "\033[1;33m", "\033[33m",  "\033[1;31m", "\033[31m", "\033[37m", "\033[1;37m"]
        index = int((len(colorCodes)-1) * (value/max))
        return colorCodes[index] + s + "\033[0m"

    # creates numBins of equal length
    # (except last bin which contains remaining items)
    # returns max val in each bucket
    def bucket(self, data, numBins):
        dataLength = len(data)
        if dataLength <= numBins:
            return data
        windowLength = dataLength//numBins
        limit = numBins * windowLength
        output = torch.Tensor(numBins)
        for i in range(0, numBins):
            start = i*windowLength
            stop = (i+1)*windowLength
            if (stop == limit):
                stop = dataLength
            output[i] = torch.max(data[start:stop])
        return output

    def display(self, data):
        barLength = self.getTerminalWidth() - 30 - len(self.label)
        normalizedData = torch.floor((1.0*data*barLength)/torch.max(data))
        bucketData = self.bucket(normalizedData, barLength)
        maxValue = torch.max(bucketData)
        code = ''
        for i in range(1, len(bucketData)):
            code = code + self.getCode(bucketData[i], maxValue, '█')
        # For Live Heatmap: print in previous line and comeback
        print('\033[F'+self.label + self.left + code + self.right, end='\n')


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        cleanup_stop_thread()
        sys.exit()
    print('')
    print('DONE')
    print('')
