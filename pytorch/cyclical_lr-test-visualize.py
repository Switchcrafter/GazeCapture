import math
import matplotlib.pyplot as plt


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


START_LR = 1
END_LR = 3E-3
LR_FACTOR = 6
STEP_SCALAR = 4

batch_size = 128                        # typical batch size on Alienware 51m with RTX 2080Ti
dataset_size = 1251983                  # size of 'train' dataset
epochs = 8                              # how many epochs to plot the CLR

batch_count = math.ceil(dataset_size / batch_size)
step_size = STEP_SCALAR * batch_count
clr = cyclical_lr(step_size, min_lr=END_LR / LR_FACTOR, max_lr=END_LR)

lrs = []
for i in range(epochs * batch_count):
    lrs.append(clr(i))
plt.plot(lrs)
plt.show()
