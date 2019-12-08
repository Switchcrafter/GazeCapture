import math
import matplotlib.pyplot as plt


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


START_LR = 1
END_LR = 3E-3
LR_FACTOR = 6
STEP_SCALAR = 4                         # This results in one steps (ie: one-half triangle cycle) over 4 epochs

batch_size = 128                        # typical batch size on Alienware 51m with RTX 2080Ti
dataset_size = 1251983                  # size of 'train' dataset
epochs = 30                             # how many epochs to plot the CLR
decay_factor = 1                        # 1 = no decay, less than 1 is decay (ie: 0.8)

plot_columns = 6
plot_rows = math.ceil(epochs / plot_columns)

batch_count = math.ceil(dataset_size / batch_size)
step_size = STEP_SCALAR * batch_count
clr = cyclical_lr(step_size, min_lr=END_LR / LR_FACTOR, max_lr=END_LR, decay_factor=decay_factor)

i = 0
fig, axs = plt.subplots(plot_rows, plot_columns, sharex=True, sharey=True)
for epoch in range(epochs):
    lrs = []
    for j in range(batch_count):
        lrs.append(clr(i))
        i += 1

    row = math.floor(epoch / plot_columns)
    column = epoch % plot_columns
    axs[row, column].plot(lrs)
    axs[row, column].set_title(f'Epoch {epoch + 1}')

plt.show()
