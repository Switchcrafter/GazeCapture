import sys
import shutil
import os
from datetime import datetime

import torch
import visdom
import numpy as np
import math


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

class SimpleProgressBar(Bar):
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

    def update(self, value):
        if value == 0:
            value = 1
        '''Updates the progress bar and its subcomponents'''
        time = datetime.now() - self.start_time

        # Overall
        speed = int(value / time.total_seconds())
        # Instantaneous
        vel = int( (value-self.sample_value) / (datetime.now() - self.sample_time).total_seconds())
        self.sample_value = value
        self.sample_time = datetime.now()

        time_eta = '[ETA : ' + str((time / value) * (self.max_value-value)) + ']'
        assert (value <= self.max_value), 'ProgressBar value (' + str(value) + ') can not exceed max_value (' + str(
            self.max_value) + ').'
        width = self.getTerminalWidth() - (
                    len(self.label) + len(self.left) + len(self.right) + len(time_eta))
        marker = self.create_marker(value, width).ljust(width, self.fill)
        marker = self.left + marker + self.right
        # append infoString at the center
        infoString = ' {val:d}/{max:d} {speed:d}@{vel:d}/s ({percent:d}%) '.format(val=value, max=self.max_value,
                                                                             speed=speed, vel=vel,
                                                                             percent=int(value / self.max_value * 100))
        index = (len(marker) - len(infoString)) // 2
        marker = marker[:index] + infoString + marker[index + len(infoString):]
        if value < self.max_value:
            print(self.label + marker + time_eta, end='\r')
        else:
            time_elapsed = '[Time: ' + str(time) + ']'
            print(self.label + marker + time_elapsed, end='\n')

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

        time_eta = '[ETA : ' + str((time / value) * (self.max_value-value)) + ']'
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


class MultiProgressBar(Bar):
    def __init__(self, max_value=100, label='Progress', marker='=', left='|', right='|', arrow='>', fill='-', boundary = False):
        self.label = '{:5}'.format(label)
        self.left = left
        self.marker = marker
        self.arrow = arrow
        self.right = right
        self.fill = fill
        self.max_value = max_value
        self.boundary = boundary
        self.processValue = [0] * max_value
        self.adjProcessValue = []
        self.processMax = [0] * max_value
        self.adjProcessMax = []
        self.codeLength = 1
        self.countEmptyTask = 0 
        self.start_time = self.sample_time = datetime.now()
    
    def get_status(self):
        complete = sum(self.processValue)
        max = sum(self.processMax)
        count = self.max_value - self.processMax.count(0) + self.countEmptyTask
        total_work = ((max/count)*self.max_value)
        return complete, total_work
        
    def create_marker(self, width):
        self.adjProcessValue = [0] * width
        self.adjProcessMax = [0] * width
        # codeLength must be calculated through flooring
        self.codeLength = math.floor(width / self.max_value)
        for index in range(self.max_value):
            if self.codeLength > 1:
                adjIndex = (index * self.codeLength)
            else:
                v = math.ceil(self.max_value/width)+1
                adjIndex = math.ceil(index / v)

            self.adjProcessValue[adjIndex] += self.processValue[index]
            self.adjProcessMax[adjIndex] += self.processMax[index]
        
        # create marker string
        code = ''
        if self.codeLength >= 1:
            limit = self.codeLength*self.max_value
            for i in range(0, limit, self.codeLength):
                code = code + self.getCode(i)
        else:
            v = math.ceil(self.max_value/width)+1
            limit = math.ceil(self.max_value / v)
            for i in range(0, limit, 1):
                code = code + self.getCode(i)
        
        # center justify filing remainder of the bar with empty string
        code = code.center(width, ' ')
        return code
    
    def getCode(self, i):
        length = max(self.codeLength,1)
        if self.adjProcessMax[i] > 0:
            # Scheduled tasks 
            if self.boundary:
                marker = "≠" + self.marker * math.floor(self.adjProcessValue[i] / self.adjProcessMax[i] * (length-1))
            else:
                marker = self.marker * math.floor(self.adjProcessValue[i] / self.adjProcessMax[i] * length)
        else:
            marker = self.fill
        return marker.ljust(length, self.fill)
        
    def addSubProcess(self, index, max_value):
        self.processMax[index] = max_value
        if max_value == 0:
            self.countEmptyTask += 1
            # force update for empty processes here
            self.update(index, 0)

    def update(self, index, value):
        self.processValue[index] = value
        remaining = [max - val for max, val in zip(self.processMax, self.processValue ) if max != 0 ]
        completedProcesses = self.countEmptyTask + remaining.count(0)

        # display 
        complete, total = self.get_status()
        time = datetime.now() - self.start_time
        if complete > 0 and complete < total:
            time_info = '[ETA : ' + str((time / complete) * (total - complete)) + ']'
        else:
            time_info = '[Time: ' + str(time) + ']'
        
        width = self.getTerminalWidth() - (len(self.label) + len(self.left) + len(self.right) + len(time_info))
        code = self.create_marker(width)

        # append infoString at the center
        percentage = int(complete / total * 100) if total > 0 else 0
        infoString = ' {val:d}/{max:d} ({percent:d}%) '.format(val=completedProcesses, max=self.max_value, percent=percentage)
        index = (len(code) - len(infoString)) // 2
        code = code[:index] + infoString + code[index + len(infoString):]
        
        print(self.label + self.left + code + self.right + time_info, end='\r')


def centered_text(infoString, marker='-', length=40):
    marker = marker * length
    index = (len(marker) - len(infoString)) // 2
    return marker[:index] + infoString + marker[index + len(infoString):]


class Visualizations(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', active=False):
        self.active = active
        if self.active:
            try:
                self.viz = visdom.Visdom()
                # wait until visdom connection is up
                while self.viz.check_connection() is not True:
                    pass
            except:
                print("Can't initialize visdom")
        else:
            self.viz = None
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
        if self.active:
            self.viz.close(None, self.env)

    def reset(self):
        if self.active:
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
        if self.active:
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
        if self.active:
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


def resize(arr, new_size, filling=None):
    if new_size > len(arr):
        arr.extend([filling for x in range(len(arr), new_size)])
    else:
        del arr[new_size:]


def set_print_policy(master, local_rank):
    print("[PrintPolicy]", master, local_rank, "Verbatim" if local_rank == master else "Silent")
    if local_rank == master:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(os.devnull, 'w')