import argparse
import math

import matplotlib.pyplot as plt


def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='cyclical_lr-test-visualize.py')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset_size', type=int, default=1251983)  # size of 'train' MIT GazeCapture dataset
    parser.add_argument('--step_scalar',
                        type=float,
                        default=4.,
                        help="How many epochs in one steps (ie: one-half triangle cycle)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--plot_columns',
                        type=int,
                        default=4,
                        help='Number of columns in output plot')
    parser.add_argument('--decay_type',
                        default='no_decay',
                        help='no_decay, step_decay, exp_decay, time_decay')
    parser.add_argument('--shape_type',
                        default='triangular',
                        help='triangular, flat')
    parser.add_argument('--single_graph', type=str2bool, nargs='?', const=True, default=False,
                        help="Display the graph as one graph, with no sub-plots.")
    args = parser.parse_args()

    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def decay_function(decay_type, epochs_per_step):
    if decay_type == 'no_decay':
        decay = lambda current_epoch: 1.
    elif decay_type == 'step_decay':
        drop = 0.5
        decay = lambda current_epoch: math.pow(drop, math.floor(1 + current_epoch / (2 * epochs_per_step)))
    elif decay_type == 'exp_decay':
        k = 0.1
        decay = lambda current_epoch: math.exp(-k * current_epoch)
    elif decay_type == 'time_decay':
        decay_time = 0.1
        decay = lambda current_epoch: 1. / (1. + decay_time * current_epoch)

    return decay


def shape_function(shape_type, step_size):
    if shape_type == 'flat':
        shape = lambda it: 1.
    elif shape_type == 'triangular':
        # for a given iteration, determines which cycle it belongs to
        # note that a cycle is 2x steps in the triangular waveform
        cycle = lambda it: math.floor(1 + it / (2 * step_size))

        shape = lambda it: max(0, (1 - abs(it / step_size - 2 * cycle(it) + 1)))

    return shape


# Based on https://www.jeremyjordan.me/nn-learning-rate/
def cyclical_lr(batch_count, shape, decay, min_lr=3e-4, max_lr=3e-3):
    epoch = lambda it: math.floor(it / batch_count)

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * shape(it) * decay(epoch(it))

    return lr_lambda


def main():
    args = parse_commandline_arguments()

    START_LR = 1
    END_LR = 3E-3
    LR_FACTOR = 6
    EPOCHS_PER_STEP = args.step_scalar  # epochs per step

    min_lr = END_LR / LR_FACTOR
    max_lr = END_LR

    shape_type = args.shape_type
    decay_type = args.decay_type

    batch_size = args.batch_size  # datapoints per batch (limited by GPU memory)
    dataset_size = args.dataset_size

    batch_count = math.ceil(dataset_size / batch_size)  # batches per epoch
    step_size = EPOCHS_PER_STEP * batch_count  # batches per step
    clr = cyclical_lr(batch_count,
                      shape=shape_function(shape_type, step_size),
                      decay=decay_function(decay_type, EPOCHS_PER_STEP),
                      min_lr=min_lr,
                      max_lr=max_lr,
                      )

    batch_num = 0
    all_lrs = []

    for current_epoch in range(args.epochs):
        lrs = []
        for j in range(batch_count):
            # this is equivalent to calling scheduler.step() once per batch
            lrs.append(clr(batch_num))
            batch_num += 1

        all_lrs.append(lrs)

    if args.single_graph:
        combined_lrs = []
        for lrs in all_lrs:
            combined_lrs.extend(lrs)
        plt.plot(combined_lrs)
    else:
        plot_columns = args.plot_columns
        plot_rows = math.ceil(args.epochs / plot_columns)

        fig, axs = plt.subplots(plot_rows, plot_columns, sharex=True, sharey=True)

        for current_epoch in range(args.epochs):
            row = math.floor(current_epoch / plot_columns)
            column = current_epoch % plot_columns
            axs[row, column].plot(all_lrs[current_epoch])
            axs[row, column].set_title(f'Epoch {current_epoch + 1}')

    plt.show()


main()
