import argparse
import os
import sys

import torch


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
    parser.add_argument('--data_path',
                        help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.",
                        default='/data/gc-data-prepped-rc')
    parser.add_argument('--output_path',
                        help="Path to checkpoint",
                        default="")
    parser.add_argument('--save_checkpoints', action='store_true', default=False,
                        help="Save each of the checkpoints as the run progresses.")
    parser.add_argument('--test', action='store_true', default=False, help="Just test and terminate.")
    parser.add_argument('--validate', action='store_true', default=False,
                        help="Just validate and terminate.")
    parser.add_argument('--reset', action='store_true', default=False,
                        help="Start from scratch (do not load).")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dataset_limit', type=int, default=0, help="Limits the dataset size, useful for debugging")
    parser.add_argument('--exportONNX', action='store_true', default=False)
    parser.add_argument('--disable-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="verbose mode - print details every batch")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mode', help="Parallelization mode: [none], dp, ddp1, ddp2", default='none')
    parser.add_argument('--disable_sync', action='store_true', default=False, help='Disable Sync BN')
    parser.add_argument('--disable_boost', action='store_true', default=False, help='Disable eval boost')
    parser.add_argument('--name', help="Provide a name to the experiment", default='main')
    parser.add_argument('--local_rank', help="", nargs='+', default=list(range(torch.cuda.device_count())))
    parser.add_argument('--master', type=int, default=0)
    parser.add_argument('--hsm', action='store_true', default=False, help="")
    parser.add_argument('--hsm_cycle', type=int, default=8)
    parser.add_argument('--adv', action='store_true', default=False, help="Enables Adversarial Attack")
    parser.add_argument('--color_space', default='YCbCr', help='Image color space - RGB, YCbCr, L, HSV, LAB')
    parser.add_argument('--decay_type', default='none', help='none, step_decay, exp_decay, time_decay')
    parser.add_argument('--shape_type', default='triangular', help='triangular, flat')
    parser.add_argument('--data_loader', default="cpu", help="cpu, dali_cpu, dali_gpu, dali_gpu_all")
    parser.add_argument('--visdom', action='store_true', default=False, help="Enables Visdom Server")
    args = parser.parse_args()

    args.device_group = "".join([str(device) for device in args.local_rank])
    # Create a checkpoint directory per device (or device group) for multiple executions
    if args.output_path == "":
        args.output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints', 'gpu' + args.device_group)

    args.device = None
    args.using_cuda = False
    if torch.cuda.device_count() > 1 and args.mode == "none":
        print("##################################################################################################")
        print("You're running in non-Parallelization mode ['none'] on a multi-GPU machine.\n \
        Make sure you're not using `-m torch.distributed.launch --nproc_per_node=<num devices>`\n \
        which would just create multiple parallel runs on different GPUs without any synchronization.\n \
        For no parallelization use              : --local_rank 0 --mode 'none' \n \
        For data parallelization use            : --local_rank 0 1 2 3 --mode 'dp' \n \
        For distributed DP1 use                 : -m torch.distributed.launch --nproc_per_node=1 --mode 'ddp1' \n \
        For distributed DP2 use                 : -m torch.distributed.launch --nproc_per_node=4 --mode 'ddp2' \n \
        ")
        print("##################################################################################################")
    args.master = args.local_rank[0] if args.mode == 'dp' else args.master
    args.batch_size = args.batch_size * torch.cuda.device_count() if args.mode == 'ddp1' else args.batch_size
    if not args.disable_cuda and torch.cuda.is_available() and len(args.local_rank) > 0:
        args.using_cuda = True
        # remove any device which doesn't exists
        args.local_rank = [int(d) for d in args.local_rank if 0 <= int(d) < torch.cuda.device_count()]
        # set args.local_rank[0] as the current device
        torch.cuda.set_device(args.local_rank[0])
        args.device = torch.device("cuda")
        if args.mode != 'dp' and args.mode != 'ddp1' and torch.cuda.device_count() > 1:
            args.name = args.name + "_" + str(args.local_rank[0])
    else:
        args.device = torch.device('cpu')

    if args.using_cuda and torch.cuda.device_count() > 0:
        # Change batch_size in commandLine args if out of cuda memory
        args.batch_size = len(args.local_rank) * args.batch_size
    else:
        args.batch_size = 1

    args.mode = 'Train'
    if args.test:
        args.mode = 'Test'
    elif args.validate:
        args.mode = 'Validate'
    elif args.exportONNX:
        args.mode = 'ExportONNX'

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
        print('args.color_space      = %s' % args.color_space)
        print('args.using_cuda       = %s' % args.using_cuda)
        print('===================================================')

    return args
