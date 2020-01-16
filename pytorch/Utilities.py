import shutil
from datetime import datetime

import torch
import visdom
import numpy as np

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


class Bar(object):
    def __init__(self):
        pass

    def getTerminalWidth(self):
        default_width = 80
        default_height = 20
        size_tuple = shutil.get_terminal_size((default_width, default_height))  # pass fallback
        return size_tuple.columns


class ProgressBar(Bar):
    '''A progress bar which stretches to fill the line.'''

    def __init__(self, max_value=100, label='', marker='=', left='|', right='|', arrow='>', fill='-'):
        '''Creates a customizable progress bar.
        max_value - max possible value for the progressbar
        label - title for the progressbar as prefix
        marker - string or callable object to use as a marker
        left - string or callable object to use as a left border
        right - string or callable object to use as a right border
        fill - character to use for the empty part of the progress bar
        '''
        self.label = '{:5}'.format(label)
        self.left = left
        self.marker = marker
        self.arrow = arrow
        self.right = right
        self.fill = fill
        self.max_value = max_value
        # self.start_time = None
        # self.sample_value = None
        # self.sample_time = None
        self.start_time = self.sample_time = datetime.now()
        self.sample_value = 0

    def create_marker(self, value, width):
        if self.max_value > 0:
            length = int(value / self.max_value * width)
            if length == width:
                return (self.marker * length)
            elif length == 0:
                return ''
            else:
                marker = (self.marker * (length - 1)) + self.arrow
        else:
            marker = self.marker
        return marker

    def update(self, value, metric, error):
        '''Updates the progress bar and its subcomponents'''
        metric = '[{metric:.4f}]'.format(metric=metric) if metric else ''
        error = '[{error:.4f}]'.format(error=error) if error else ''
        time = datetime.now() - self.start_time

        # Overall
        speed = int(value / time.total_seconds())
        # Instantaneous
        vel = int( (value-self.sample_value) / (datetime.now() - self.sample_time).total_seconds())
        self.sample_value = value
        self.sample_time = datetime.now()
            
        time_eta = '[ETA : ' + str((time / value) * self.max_value) + ']'
        assert (value <= self.max_value), 'ProgressBar value (' + str(value) + ') can not exceed max_value (' + str(
            self.max_value) + ').'
        width = self.getTerminalWidth() - (
                    len(self.label) + len(self.left) + len(self.right) + len(metric) + len(error) + len(time_eta))
        marker = self.create_marker(value, width).ljust(width, self.fill)
        marker = self.left + marker + self.right
        # append infoString at the center
        infoString = ' {val:d}/{max:d} {speed:d}@{vel:d}/s ({percent:d}%) '.format(val=value, max=self.max_value,
                                                                             speed=speed, vel=vel,
                                                                             percent=int(value / self.max_value * 100))
        index = (len(marker) - len(infoString)) // 2
        marker = marker[:index] + infoString + marker[index + len(infoString):]
        if value < self.max_value:
            print(self.label + marker + metric + error + time_eta, end='\r')
        else:
            time_elapsed = '[Time: ' + str(time) + ']'
            print(self.label + marker + metric + error + time_elapsed, end='\n')


class SamplingBar(Bar):
    '''A sampling hotness bar which stretches to fill the line.'''

    def __init__(self, label='', left='|', right='|'):
        '''Creates a multinomial sampling hotness bar.
        '''
        self.label = '{:5}'.format(label)
        self.left = left
        self.right = right

    #  colorCodes = {black, VIBGYOR, White}
    def getCode(self, value=0.1, max=1.0, s='█'):
        colorCodes = ["\033[30m", "\033[1;30m", "\033[35m", "\033[1;35m", "\033[34m", "\033[1;34m", "\033[36m",
                      "\033[32m", "\033[1;32m", "\033[1;33m", "\033[33m", "\033[1;31m", "\033[31m", "\033[37m",
                      "\033[1;37m"]
        index = int((len(colorCodes) - 1) * (value / max))
        return colorCodes[index] + s + "\033[0m"

    # creates numBins of equal length
    # (except last bin which contains remaining items)
    # returns max val in each bucket
    def bucket(self, data, numBins):
        dataLength = len(data)
        if dataLength <= numBins:
            return data
        windowLength = dataLength // numBins
        limit = numBins * windowLength
        output = torch.Tensor(numBins)
        for i in range(0, numBins):
            start = i * windowLength
            stop = (i + 1) * windowLength
            if (stop == limit):
                stop = dataLength
            output[i] = torch.max(data[start:stop])
        return output

    def display(self, data):
        barLength = self.getTerminalWidth() - 40 - len(self.label)
        normalizedData = torch.floor((1.0 * data * barLength) / torch.max(data))
        bucketData = self.bucket(normalizedData, barLength)
        maxValue = torch.max(bucketData)
        code = ''
        for i in range(1, len(bucketData)):
            code = code + self.getCode(bucketData[i], maxValue, '█')
        # For Live Heatmap: print in previous line and comeback
        print('\033[F' + self.label + self.left + code + self.right, end='\n')


def centeredText(infoString, marker='-', length=40):
    marker = marker * length
    index = (len(marker) - len(infoString)) // 2
    return marker[:index] + infoString + marker[index + len(infoString):]

class Visualizations(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        try:
            self.viz = visdom.Visdom()
            # wait until visdom connection is up
            while self.viz.check_connection() is not True:
                pass
        except:
            print("Can't initialize visdom")
        # env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env = env_name
        self.split_plots = {}
        self.epoch_plots = {}
        self.closed_windows = []

    def getColor(self, split_name):
        if split_name == "train":
            return np.array([[0, 0, 255],]) # Blue
        elif split_name == "val" or split_name == "val_history" :
            return np.array([[255, 0, 0],]) # Red
        elif split_name == "test":
            return np.array([[255, 0, 0],]) # Green
        else:
            return np.array([[0, 0, 0],]) # Black

    def getStyle(self, style='solid'):
        if style == "dash":
            return np.array(['dash'])
        elif style == "dashdot":
            return np.array(['dashdot'])
        elif style == "dot":
            return np.array(['dot'])
        else:
            return np.array(['solid'])

    def resetAll(self):
        # Close all windows in the given environment
        self.viz.close(None, self.env)

    def reset(self):
        for var_name in self.split_plots:
            self.closed_windows.append(self.split_plots[var_name])
        self.split_plots = {}
        for window in self.closed_windows:
            # make sure that the window is closed before moving on
            while self.viz.win_exists(window, self.env):
                self.viz.close(window, self.env)
            # remove the closed window
            self.closed_windows.remove(window)

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.split_plots:
            self.split_plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                linecolor=self.getColor(split_name),
                xlabel='Samples',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.split_plots[var_name], name=split_name,
            update = 'append', opts=dict(linecolor=self.getColor(split_name)))

    def plotAll(self, var_name, split_name, title_name, x, y, style='solid'):
        ytype = 'log' if split_name == "lr" else 'linear'
        if var_name not in self.epoch_plots:
            self.epoch_plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                linecolor=self.getColor(split_name),
                dash=self.getStyle(style),
                xlabel='Epoch',
                ytickmin=0,
                ytickmax=None,
                ytype=ytype,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.epoch_plots[var_name], name=split_name,
            update = 'append', opts=dict(linecolor=self.getColor(split_name), dash=self.getStyle(style)))

