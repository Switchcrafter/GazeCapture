import math


def decay_function(decay_type, epochs_per_step):
    if decay_type == 'none':
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