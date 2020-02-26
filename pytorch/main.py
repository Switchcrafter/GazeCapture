import json
import math
import os
import sys  # for command line argument dumping
import time
from datetime import datetime  # for timing
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from ITrackerModel import ITrackerModel
from ITrackerData import load_all_data
from Utilities import AverageMeter, ProgressBar, SamplingBar, Visualizations, resize, set_print_policy
import checkpoint_manager

import cyclical_learning_rate
import argument_parser

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
EPOCHS_PER_STEP = 4


def main():
    args = argument_parser.parse_commandline_arguments()

    initialize_visualization(args)

    # make sure checkpoints directory exists
    if not os.path.exists(args.output_path):
        print('{0} does not exist, creating...'.format(args.output_path))
        os.mkdir(args.output_path)

    RMSErrors, best_RMSError, best_RMSErrors, epoch, learning_rates, model = initialize_model(args)

    print('epoch = %d' % epoch)

    totalstart_time = datetime.now()

    datasets = load_all_data(args.data_path,
                             IMAGE_SIZE,
                             FACE_GRID_SIZE,
                             args.workers,
                             args.batch_size,
                             args.verbose,
                             args.color_space,
                             args.data_loader,
                             not args.disable_boost)

    criterion, optimizer, scheduler = initialize_hyper_parameters(args, datasets, model)

    if args.mode == 'Train':
        # resize variables to epochs size
        resize(learning_rates, args.epochs)
        resize(best_RMSErrors, args.epochs)
        resize(RMSErrors, args.epochs)

        if args.hsm:
            args.multinomial_weights = torch.ones(datasets['train'].size, dtype=torch.double)
            if not args.verbose:
                args.sampling_bar = SamplingBar('HSM')

        # Placeholder for overall (all epoch) visualizations
        args.vis.plotAll('LearningRate', 'lr', "LearningRate (Overall)", None, None)
        args.vis.plotAll('RMSError', 'train', "RMSError (Overall)", None, None)
        args.vis.plotAll('RMSError', 'val', "RMSError (Overall)", None, None)
        args.vis.plotAll('BestRMSError', 'val', "Best RMSError (Overall)", None, None)
        # Populate visualizations with checkpoint info
        for epoch_num in range(1, epoch):
            args.vis.plotAll('LearningRate', 'lr_history', "LearningRate (Overall)", epoch_num,
                             learning_rates[epoch_num], 'dot')
            args.vis.plotAll('RMSError', 'val_history', "RMSError (Overall)", epoch_num, RMSErrors[epoch_num], 'dot')
            args.vis.plotAll('BestRMSError', 'val_history', "Best RMSError (Overall)", epoch_num,
                             best_RMSErrors[epoch_num], 'dot')
            if epoch_num == epoch - 1:
                args.vis.plotAll('LearningRate', 'lr', "LearningRate (Overall)", epoch_num, learning_rates[epoch_num])
                args.vis.plotAll('RMSError', 'val', "RMSError (Overall)", epoch_num, RMSErrors[epoch_num])
                args.vis.plotAll('BestRMSError', 'val', "Best RMSError (Overall)", epoch_num, best_RMSErrors[epoch_num])

        # now start training from last best epoch
        for epoch in range(epoch, args.epochs + 1):
            print('Epoch %05d of %05d - adjust, train, validate' % (epoch, args.epochs))
            start_time = datetime.now()
            learning_rates[epoch - 1] = scheduler.get_last_lr()

            args.vis.reset()
            # train for one epoch
            print('\nEpoch:{} [device:{}, best_RMSError:{:2.4f}, hsm:{}, adv:{}]'.format(epoch,
                                                                                         args.device,
                                                                                         best_RMSError,
                                                                                         args.hsm,
                                                                                         args.adv))
            train_MSELoss, train_RMSError = train(datasets['train'],
                                                  model,
                                                  criterion,
                                                  optimizer,
                                                  scheduler,
                                                  epoch,
                                                  args.batch_size,
                                                  args.device,
                                                  args.dataset_limit,
                                                  args.verbose,
                                                  args)

            # evaluate on validation set
            eval_MSELoss, eval_RMSError = evaluate(datasets['val'],
                                                   model,
                                                   criterion,
                                                   epoch,
                                                   args.output_path,
                                                   args.device,
                                                   args.dataset_limit,
                                                   args.verbose,
                                                   args)

            # remember best RMSError and save checkpoint
            is_best = eval_RMSError < best_RMSError
            best_RMSError = min(eval_RMSError, best_RMSError)

            best_RMSErrors[epoch - 1] = best_RMSError
            RMSErrors[epoch - 1] = eval_RMSError

            args.vis.plotAll('LearningRate', 'lr', "LearningRate (Overall)", epoch, scheduler.get_last_lr())
            args.vis.plotAll('RMSError', 'train', "RMSError (Overall)", epoch, train_RMSError)
            args.vis.plotAll('RMSError', 'val', "RMSError (Overall)", epoch, eval_RMSError)
            args.vis.plotAll('BestRMSError', 'val', "Best RMSError (Overall)", epoch, best_RMSError)
            time_elapsed = datetime.now() - start_time

            if run:
                run.log('MSELoss', eval_MSELoss)
                run.log('RMSLoss', eval_RMSError)
                run.log('best MSELoss', best_RMSError)
                run.log('epoch time', time_elapsed)

            checkpoint_manager.save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_RMSError': best_RMSError,
                    'is_best': is_best,
                    'train_MSELoss': train_MSELoss,
                    'train_RMSError': train_RMSError,
                    'eval_MSELoss': eval_MSELoss,
                    'eval_RMSError': eval_RMSError,
                    'time_elapsed': time_elapsed,
                    'RMSErrors': RMSErrors,
                    'best_RMSErrors': best_RMSErrors,
                    'learning_rates': learning_rates,
                },
                is_best,
                args.output_path,
                args.save_checkpoints)

            print('')
            print('Epoch {epoch:5d} with RMSError {rms_error:.5f}'.format(epoch=epoch, rms_error=best_RMSError))
            print('Epoch Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
            print('')
            print('\'RMS_Errors\': {0},'.format(RMSErrors))
            print('\'Best_RMS_Errors\': {0}'.format(best_RMSErrors))
            print('')
    elif args.mode == 'Test':
        # Quick test
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = evaluate(datasets['test'],
                                               model,
                                               criterion,
                                               1,
                                               args.output_path,
                                               args.device,
                                               args.dataset_limit,
                                               args.verbose,
                                               args)
        time_elapsed = datetime.now() - start_time
        print('')
        print('Testing MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
        print('Testing Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif args.mode == 'Validate':
        start_time = datetime.now()
        eval_MSELoss, eval_RMSError = evaluate(datasets['val'],
                                               model,
                                               criterion,
                                               1,
                                               args.output_path,
                                               args.device,
                                               args.dataset_limit,
                                               args.verbose,
                                               args)
        time_elapsed = datetime.now() - start_time
        print('')  # print blank line after loading data
        print('Validation MSELoss: %.5f, RMSError: %.5f' % (eval_MSELoss, eval_RMSError))
        print('Validation Time elapsed(hh:mm:ss.ms) {}'.format(time_elapsed))
    elif args.mode == 'ExportONNX':
        # export the model for use in other frameworks
        export_onnx_model(model, args.device, args.verbose)

    totaltime_elapsed = datetime.now() - totalstart_time
    print('Total Time elapsed(hh:mm:ss.ms) {}'.format(totaltime_elapsed))


def initialize_visualization(args):
    # Initialize the visualization environment open => http://localhost:8097
    args.vis = Visualizations(args.name)
    args.vis.resetAll()


def initialize_model(args):
    if args.verbose:
        print('')
        if args.using_cuda:
            print('Using cuda devices:', args.local_rank)
            # print('CUDA DEVICE_COUNT {0}'.format(torch.cuda.device_count()))
        print('')

    # Retrieve model
    model = ITrackerModel().to(device=args.device)
    # GPU optimizations and modes
    cudnn.benchmark = True
    if args.using_cuda:
        if args.mode == 'dp':
            print('Using DataParallel Backend')
            if not args.disable_sync:
                from sync_batchnorm import convert_model
                # Convert batchNorm layers into synchronized batchNorm
                model = convert_model(model)
            model = torch.nn.DataParallel(model, device_ids=args.local_rank).to(device=args.device)
        elif args.mode == 'ddp1':
            # Single-Process Multiple-GPU: You'll observe all gpus running a single process (processes with same PID)
            print('Using DistributedDataParallel Backend - Single-Process Multi-GPU')
            torch.distributed.init_process_group(backend="nccl")
            model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.mode == 'ddp2':
            # Multi-Process Single-GPU : You'll observe multiple gpus running different processes (different PIDs)
            # OMP_NUM_THREADS = nb_cpu_threads / nproc_per_node
            torch.distributed.init_process_group(backend='nccl')
            if not args.disable_sync:
                # Convert batchNorm layers into synchronized batchNorm
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.local_rank,
                                                              output_device=args.local_rank[0])
            ###### code after this place runs in their own process #####
            set_print_policy(args.master, torch.distributed.get_rank())
            print('Using DistributedDataParallel Backend - Multi-Process Single-GPU')
        else:
            print("No Parallelization")
    else:
        print("Cuda disabled")
    RMSErrors, best_RMSError, best_RMSErrors, epoch, learning_rates = checkpoint_manager.extract_checkpoint_data(args,
                                                                                                                 model)
    return RMSErrors, best_RMSError, best_RMSErrors, epoch, learning_rates, model


def initialize_hyper_parameters(args, datasets, model):
    criterion = nn.MSELoss(reduction='mean').to(device=args.device)
    optimizer = torch.optim.SGD(model.parameters(), START_LR,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    batch_count = math.ceil(datasets['train'].size / args.batch_size)
    step_size = EPOCHS_PER_STEP * batch_count
    clr = cyclical_learning_rate.cyclical_lr(batch_count,
                                             shape=cyclical_learning_rate.shape_function(args.shape_type,
                                                                                         step_size),
                                             decay=cyclical_learning_rate.decay_function(args.decay_type,
                                                                                         EPOCHS_PER_STEP),
                                             min_lr=END_LR / LR_FACTOR,
                                             max_lr=END_LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    return criterion, optimizer, scheduler


# Fast Gradient Sign Attack (FGSA)
def adversarial_attack(image, data_grad, epsilon=0.1):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def euclidean_batch_error(output, target):
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
        if args.data_loader == "cpu":
            # Reset every few epoch (hsm_cycle)
            if epoch > 0 and epoch % args.hsm_cycle == 0:
                args.multinomial_weights = torch.ones(dataset.size, dtype=torch.double)
            # update dataloader and sampler
            sampler = torch.utils.data.WeightedRandomSampler(args.multinomial_weights,
                                                             int(len(args.multinomial_weights)),
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
        else:  # dali modes
            # todo: HSM support for DALI
            loader = dataset.loader
    else:
        loader = dataset.loader

    lrs = []

    # load data samples and train
    # for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(loader):
    for i, data in enumerate(dataset.loader):
        if args.data_loader == "cpu":
            (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) = data
        else:  # dali modes
            if args.data_loader == "dali_gpu_all":
                # TODO test with dp mode
                batch_data = data[int(args.local_rank[0])]
            else:  # dali_gpu, dali_cpu
                batch_data = data[0]
            row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices = batch_data["row"], batch_data["imFace"], \
                                                                          batch_data["imEyeL"], batch_data["imEyeR"], \
                                                                          batch_data["faceGrid"], \
                                                                          batch_data["gaze"], batch_data["frame"], \
                                                                          batch_data["indices"]

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
        error = euclidean_batch_error(output, gaze)

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
            perturbed_imFace = adversarial_attack(imFace, imFace_grad)
            perturbed_imEyeL = adversarial_attack(imEyeL, imEyeL_grad)
            perturbed_imEyeR = adversarial_attack(imEyeR, imEyeR_grad)
            perturbed_faceGrid = adversarial_attack(faceGrid, faceGrid_grad)

            # Regenerate output for the perturbed input
            output_adv = model(perturbed_imFace, perturbed_imEyeL, perturbed_imEyeR, perturbed_faceGrid)
            loss_adv = criterion(output_adv, gaze)

            # concatenate both real and adversarial loss functions
            loss = loss + loss_adv
            del loss_adv

        # backprop the loss
        loss.backward()
        del loss

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
            args.vis.plot("loss", dataset.split, "RMSError (epoch: {})".format(epoch), num_samples, RMSErrors.avg)
            progress_bar.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    # print('lrs={}'.format(lrs))

    return MSELosses.avg, RMSErrors.avg


def evaluate(dataset,
             model,
             criterion,
             epoch,
             checkpoints_path,
             device,
             dataset_limit=None,
             verbose=False,
             args=None):
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

    # for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) in enumerate(dataset.loader):
    for i, data in enumerate(dataset.loader):
        if args.data_loader == "cpu":
            (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) = data
        else:  # dali modes
            if args.data_loader == "dali_gpu_all":
                # TODO test with dp mode
                batch_data = data[int(args.local_rank[0])]
            else:  # dali_gpu, #dali_cpu
                batch_data = data[0]
            row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices = batch_data["row"], \
                                                                          batch_data["imFace"], \
                                                                          batch_data["imEyeL"], \
                                                                          batch_data["imEyeR"], \
                                                                          batch_data["faceGrid"], \
                                                                          batch_data["gaze"], \
                                                                          batch_data["frame"], \
                                                                          batch_data["indices"]

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
        error = euclidean_batch_error(output, gaze)

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
            args.vis.plot("loss", dataset.split, "RMSError (epoch: {})".format(epoch), num_samples, RMSErrors.avg)
            progress_bar.update(num_samples, MSELosses.avg, RMSErrors.avg)

        if dataset_limit and dataset_limit <= batchNum:
            break

    resultsFileName = os.path.join(checkpoints_path, 'results.json')
    with open(resultsFileName, 'w+') as outfile:
        json.dump(results, outfile)

    return MSELosses.avg, RMSErrors.avg


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


if __name__ == "__main__":
    try:
        FNULL = open(os.devnull, 'w')
        visdomProcess = subprocess.Popen(["python", "-m", "visdom.server"], stdout=FNULL, stderr=FNULL)
        while visdomProcess.poll() is not None:
            pass
        time.sleep(4)
        main()
    except (KeyboardInterrupt, SystemExit):
        visdomProcess.wait()
        print('Thread is killed.')
        sys.exit()
    print('')
    print('DONE')
    print('')
