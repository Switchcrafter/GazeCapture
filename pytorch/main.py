import argparse
import json
import math
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

from ITrackerData import load_all_data
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

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
GRID_SIZE = 25
FACE_GRID_SIZE = (GRID_SIZE, GRID_SIZE)

START_LR = 1
END_LR = 3E-3
LR_FACTOR = 6


def main():
    args, doLoad, doTest, doValidate, dataPath, checkpointsPath, \
    exportONNX, saveCheckpoints, using_cuda, workers, epochs, \
    dataset_limit, verbose, device = parse_commandline_arguments()

    if using_cuda and torch.cuda.device_count() > 0:
        # Change batch_size in commandLine args if out of cuda memory
        batch_size = len(args.local_rank) * args.batch_size
    else:
        batch_size = 1

    eval_RMSError = math.inf
    best_RMSError = math.inf

    if verbose:
        print('')
        if using_cuda:
            print('Using cuda devices:', args.local_rank)
            # print('CUDA DEVICE_COUNT {0}'.format(torch.cuda.device_count()))
        print('')

    # make sure checkpoints directory exists
    if not os.path.exists(checkpointsPath):
        print('{0} does not exist, creating...'.format(checkpointsPath))
        os.mkdir(checkpointsPath)

    # Retrieve model
    model = ITrackerModel().to(device=device)

    # GPU optimizations and modes
    cudnn.benchmark = True
    if using_cuda and len(args.local_rank) > 1:
        if args.mode == 'dp':
            print('Using DataParallel Backend')
            model = torch.nn.DataParallel(model, device_ids=args.local_rank).to(device=device)
        elif args.mode == 'ddp1':
            print('Using DistributedDataParallel Backend - Single-Process Multi-GPU')
            # Single-Process Multi-GPU
            torch.distributed.init_process_group(backend="nccl")
            model = torch.nn.DistributedDataParallel(model)
        else:
            print('Using DistributedDataParallel Backend - Multi-Process Single-GPU')
            # Multi-Process Single-GPU
            args.world_size = os.environ.get('WORLD_SIZE') or 1
            # torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, init_method='env://')
            torch.distributed.init_process_group(backend='nccl')
            model = torch.nn.DistributedDataParallel(model, device_ids=args.local_rank,
                                                     output_device=args.local_rank[0])

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

    datasets = load_all_data(dataPath, IMAGE_SIZE, FACE_GRID_SIZE, workers, batch_size, verbose)

    #     criterion = nn.MSELoss(reduction='sum').to(device=device)
    criterion = nn.MSELoss(reduction='mean').to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), START_LR,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    step_size = 4 * (datasets['train'].size / batch_size)
    clr = cyclical_lr(step_size, min_lr=END_LR / LR_FACTOR, max_lr=END_LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    if doTest:
        # Quick test
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = test(datasets, model, criterion, 1, checkpointsPath, batch_size, device,
                                           dataset_limit, verbose)
        time_elapsed = datetime.now() - start_time
        print('')
        print('Testing MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
        print('Testing Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif doValidate:
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = validate(datasets, model, criterion, 1, checkpointsPath, batch_size, device,
                                               dataset_limit, verbose)
        time_elapsed = datetime.now() - start_time
        print('')  # print blank line after loading data
        print('Validation MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
        print('Validation Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif exportONNX:
        # export the model for use in other frameworks
        export_onnx_model(model, device, verbose)
    else:  # Train
        learning_rates = [0] * epochs
        best_RMSErrors = [0] * epochs
        RMSErrors = [0] * epochs

        # first make a learning_rate correction suitable for epoch from saved checkpoint
        # epoch will be non-zero if a checkpoint was loaded
        for epoch in range(1, epoch):
            if verbose:
                print('Epoch %05d of %05d - adjust learning rate only' % (epoch, epochs))
                start_time = datetime.now()
            lr = adjust_learning_rate(optimizer, epoch)

            learning_rates[epoch - 1] = lr

            if verbose:
                time_elapsed = datetime.now() - start_time
                print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))

        if args.hsm:
            args.multinomial_weights = torch.ones(datasets['train'].size, dtype=torch.double)
            if not verbose:
                args.sampling_bar = SamplingBar('HSM')

        # now start training from last best epoch
        for epoch in range(epoch, epochs + 1):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, epochs))
            start_time = datetime.now()

            # train for one epoch
            print('\nEpoch:{} [device:{}, best_RMSError:{:2.4f}, hsm:{}, adv:{}]'.format(epoch, device, best_RMSError,
                                                                                         args.hsm, args.adv))
            train_MSELoss, train_RMSError = train(datasets['train'], model, criterion, optimizer, scheduler, epoch,
                                                  batch_size, device, dataset_limit, verbose, args)

            # evaluate on validation set
            eval_MSELoss, eval_RMSError = validate(datasets, model, criterion, epoch, checkpointsPath, batch_size,
                                                   device, dataset_limit, verbose)

            # remember best RMSError and save checkpoint
            is_best = eval_RMSError < best_RMSError
            best_RMSError = min(eval_RMSError, best_RMSError)

            best_RMSErrors[epoch - 1] = best_RMSError
            RMSErrors[epoch - 1] = eval_RMSError

            time_elapsed = datetime.now() - start_time

            if run:
                run.log('MSELoss', eval_MSELoss)
                run.log('RMSLoss', eval_RMSError)
                run.log('best MSELoss', best_RMSError)
                run.log('epoch time', time_elapsed)

            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_RMSError': best_RMSError,
                    'is_best': is_best,
                    'train_MSELoss': train_MSELoss,
                    'train_RMSError': train_RMSError,
                    'eval_MSELoss': eval_MSELoss,
                    'eval_RMSError': eval_RMSError,
                    'lr': lr,
                    'time_elapsed': time_elapsed,
                },
                is_best,
                checkpointsPath,
                saveCheckpoints)

            print('')
            print('Epoch {epoch:5d} with RMSError {rms_error:.5f}'.format(epoch=epoch, rms_error=best_RMSError))
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
            print('')
            print('\'RMS_Errors\': {0},'.format(RMSErrors))
            print('\'Best_RMS_Errors\': {0}'.format(best_RMSErrors))
            print('')

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))


# Fast Gradient Sign Attack (FGSA)
def adversarialAttack(image, data_grad, epsilon=0.1):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def euclideanBatchError(output, target):
    """ For a batch of output and target returns corresponding batch of euclidean errors
    """
    # Batch Euclidean Distance sqrt(dx^2 + dy^2)
    return torch.sqrt(torch.sum(torch.pow(output - target, 2), 1))


def train(dataset, model, criterion, optimizer, scheduler, epoch, batch_size, device, dataset_limit=None, verbose=False,
          args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    MSELosses = AverageMeter()
    RMSErrors = AverageMeter()
    num_samples = 0

    if not verbose:
        progress_bar = ProgressBar(max_value=dataset.size, label=dataset.split)

    # switch to train mode
    model.train()

    end = time.time()

    # HSM Update - Every epoch
    if args.hsm:
        # Reset every few epoch (hsm_cycle)
        if epoch > 0 and epoch % args.hsm_cycle == 0:
            args.multinomial_weights = torch.ones(dataset.size, dtype=torch.double)
        # update dataloader and sampler
        sampler = torch.utils.data.WeightedRandomSampler(args.multinomial_weights, int(len(args.multinomial_weights)),
                                                         replacement=True)
        loader = torch.utils.data.DataLoader(
            dataset.loader.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.workers)
        # Line-space for HSM meter
        print('')
        if not verbose:
            args.sampling_bar.display(args.multinomial_weights)
    else:
        loader = dataset.loader

    lrs = []

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
            batch_loss[batch_loss > 1.0] = 1.0
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

        # Update LR
        scheduler.step()
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        lrs.append(lr_step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Epoch ({split:7s}): [{epoch:3d}][{batchNum:7d}/{dataset_size:7d}]\t'
                  'Time {batch_time.val:8.4f} ({batch_time.avg:8.4f})\t'
                  'Data {data_time.val:8.4f} ({data_time.avg:8.4f})\t'
                  'MSELoss {MSELosses.val:8.4f} ({MSELosses.avg:8.4f})\t'
                  'RMSError {RMSErrors.val:8.4f} ({RMSErrors.avg:8.4f})\t'.format(
                split=dataset.split,
                epoch=epoch,
                batchNum=batchNum,
                dataset_size=dataset.size,
                batch_time=batch_time,
                data_time=data_time,
                MSELosses=MSELosses,
                RMSErrors=RMSErrors))
        else:
            progress_bar.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    print('lrs={}'.format(lrs))

    return MSELosses.avg, RMSErrors.avg


def evaluate(dataset, model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit=None, verbose=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    MSELosses = AverageMeter()
    RMSErrors = AverageMeter()
    num_samples = 0

    if not verbose:
        progress_bar = ProgressBar(max_value=dataset.size, label=dataset.split)

    # switch to evaluate mode
    model.eval()

    end = time.time()

    results = []

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(dataset.loader):
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
            print('Epoch ({split:7s}): [{epoch:3d}][{batchNum:7d}/{dataset_size:7d}]\t'
                  'Time {batch_time.val:8.4f} ({batch_time.avg:8.4f})\t'
                  'MSELoss {MSELosses.val:8.4f} ({MSELosses.avg:8.4f})\t'
                  'RMSError {RMSErrors.val:8.4f} ({RMSErrors.avg:8.4f})\t'.format(
                split=dataset.split,
                epoch=epoch,
                batchNum=batchNum,
                dataset_size=dataset.size,
                batch_time=batch_time,
                MSELosses=MSELosses,
                RMSErrors=RMSErrors))
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
    return evaluate(datasets['val'], model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit,
                    verbose)


def test(datasets,
         model,
         criterion,
         epoch,
         checkpointsPath,
         batch_size,
         device,
         dataset_limit=None,
         verbose=False):
    return evaluate(datasets['test'], model, criterion, epoch, checkpointsPath, batch_size, device, dataset_limit,
                    verbose)


def export_onnx_model(model, device, verbose):
    # switch to evaluate mode
    model.eval()

    batch_size = 1
    color_depth = 3  # 3 bytes for RGB color space
    face_grid_size = GRID_SIZE * GRID_SIZE

    imFace = torch.randn(batch_size, color_depth, IMAGE_WIDTH, IMAGE_HEIGHT).to(device=device).float()
    imEyeL = torch.randn(batch_size, color_depth, IMAGE_WIDTH, IMAGE_HEIGHT).to(device=device).float()
    imEyeR = torch.randn(batch_size, color_depth, IMAGE_WIDTH, IMAGE_HEIGHT).to(device=device).float()
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
        shutil.copyfile('ITrackerModel.py', os.path.join(checkpointsPath, 'ITrackerModel.py'))

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


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


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
    parser.add_argument('--output_path',
                        help="Path to checkpoint",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints'))
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
    parser.add_argument('--mode', help="Multi-GPU mode: dp, ddp1, [ddp2]", default='ddp2')
    parser.add_argument('--local_rank', help="", nargs='+', default=[0])
    parser.add_argument('--hsm', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--hsm_cycle', type=int, default=8)
    parser.add_argument('--adv', type=str2bool, nargs='?', const=True, default=False, help="Enables Adversarial Attack")
    args = parser.parse_args()

    args.device = None
    usingCuda = False
    if not args.disable_cuda and torch.cuda.is_available() and len(args.local_rank) > 0:
        usingCuda = True
        # remove any device which doesn't exists
        args.local_rank = [int(d) for d in args.local_rank if 0 <= int(d) < torch.cuda.device_count()]
        # # set args.local_rank[0] (the master node) as the current device
        torch.cuda.set_device(args.local_rank[0])
        args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')

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
           usingCuda, workers, epochs, dataset_limit, verbose, device


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print('Thread is killed.')
        sys.exit()
    print('')
    print('DONE')
    print('')
