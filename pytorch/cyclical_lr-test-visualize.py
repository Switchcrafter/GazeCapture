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
    parser.add_argument('--decay_factor',
                        type=float,
                        default=1.,
                        help="How much decay should occur. 1 = no decay. 0.5-0.8 are common values")
    parser.add_argument('--plot_columns',
                        type=int,
                        default=6,
                        help='Number of columns in output plot')
    args = parser.parse_args()

    return args


# Based on https://www.jeremyjordan.me/nn-learning-rate/
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay_factor=1):
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Scaler: 1 = no decay, less than 1 is decay (ie: 0.8)
    def scaler(it, decay_factor):
        return decay_factor ** cycle(it, stepsize)

    # for a given iteration, determines which cycle it belongs to
    # note that a cycle is 2x steps in the triangular waveform
    def cycle(it, stepsize):
        return math.floor(1 + it / (2 * stepsize))

    # determine, for a given point it, what the resulting f(it) value is for the triangular function
    def relative(it, stepsize):
        x = abs(it / stepsize - 2 * cycle(it, stepsize) + 1)
        return max(0, (1 - x)) * scaler(it, decay_factor)

    return lr_lambda


def main():
    args = parse_commandline_arguments()

    START_LR = 1
    END_LR = 3E-3
    LR_FACTOR = 6
    STEP_SCALAR = args.step_scalar

    batch_size = args.batch_size
    dataset_size = args.dataset_size
    epochs = args.epochs
    decay_factor = args.decay_factor

    plot_columns = args.plot_columns
    plot_rows = math.ceil(epochs / plot_columns)

    batch_count = math.ceil(dataset_size / batch_size)
    step_size = STEP_SCALAR * batch_count
    clr = cyclical_lr(step_size, min_lr=END_LR / LR_FACTOR, max_lr=END_LR, decay_factor=decay_factor)

    batch_num = 0
    fig, axs = plt.subplots(plot_rows, plot_columns, sharex=True, sharey=True)
    for epoch in range(epochs):
        lrs = []
        for j in range(batch_count):
            # this is equivalent to calling scheduler.step() once per batch
            lrs.append(clr(batch_num))
            batch_num += 1

        row = math.floor(epoch / plot_columns)
        column = epoch % plot_columns
        axs[row, column].plot(lrs)
        axs[row, column].set_title(f'Epoch {epoch + 1}')
    plt.show()


main()
