import shutil
from datetime import datetime

import torch


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
    def __init__(self, max_value=100, label='', marker='=', left='|', right='|', arrow = '>', fill='-'):
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
        self.start_time = None

    def create_marker(self, value, width):
        if self.max_value > 0:
            length = int(value / self.max_value * width)
            if length == width:
                return (self.marker * length)
            elif length == 0:
                return ''
            else:
                marker = (self.marker * (length-1)) + self.arrow
        else:
            marker = self.marker
        return marker

    def update(self, value, metric, error):
        '''Updates the progress bar and its subcomponents'''
        self.start_time = self.start_time or datetime.now()
        metric = '[{metric:.4f}]'.format(metric=metric) if metric else ''
        error = '[{error:.4f}]'.format(error=error) if error else ''
        time = datetime.now() - self.start_time
        eta = (time/value)*self.max_value
        time_eta = '[ETA : '+str(eta)+']'
        assert( value <= self.max_value), 'ProgressBar value (' + str(value) + ') can not exceed max_value ('+ str(self.max_value)+').'
        width = self.getTerminalWidth() - (len(self.label) + len(self.left) + len(self.right) + len(metric) + len(error) + len(time_eta))
        marker = self.create_marker(value, width).ljust(width, self.fill)
        marker = self.left + marker + self.right
        # append infoString at the center
        infoString = ' {val:d}/{max:d} @{speed:d}/s ({percent:d}%) '.format(val = value, max = self.max_value, speed=int(value/time.total_seconds()), percent = int(value/self.max_value*100))
        index = (len(marker)-len(infoString))//2
        marker = marker[:index] + infoString + marker[index + len(infoString):]
        if value < self.max_value:
            print(self.label + marker + metric + error + time_eta, end = '\r')
        else:
            time_elapsed = ' [Time: '+str(time)+']'
            print(self.label + marker + metric + error + time_elapsed, end = '\n')


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
        colorCodes = ["\033[30m", "\033[1;30m", "\033[35m", "\033[1;35m", "\033[34m", "\033[1;34m", "\033[36m", "\033[32m", "\033[1;32m", "\033[1;33m", "\033[33m",  "\033[1;31m", "\033[31m", "\033[37m", "\033[1;37m"]
        index = int((len(colorCodes)-1) * (value/max))
        return colorCodes[index] + s + "\033[0m"

    # creates numBins of equal length
    # (except last bin which contains remaining items)
    # returns max val in each bucket
    def bucket(self, data, numBins):
        dataLength = len(data)
        if dataLength <= numBins:
            return data
        windowLength = dataLength//numBins
        limit = numBins * windowLength
        output = torch.Tensor(numBins)
        for i in range(0, numBins):
            start = i*windowLength
            stop = (i+1)*windowLength
            if (stop == limit):
                stop = dataLength
            output[i] = torch.max(data[start:stop])
        return output

    def display(self, data):
        barLength = self.getTerminalWidth() - 40 - len(self.label)
        normalizedData = torch.floor((1.0*data*barLength)/torch.max(data))
        bucketData = self.bucket(normalizedData, barLength)
        maxValue = torch.max(bucketData)
        code = ''
        for i in range(1, len(bucketData)):
            code = code + self.getCode(bucketData[i], maxValue, '█')
        # For Live Heatmap: print in previous line and comeback
        print('\033[F'+self.label + self.left + code + self.right, end='\n')


def centeredText(infoString, marker='-', length=40):
    marker = marker*length
    index = (len(marker)-len(infoString))//2
    return marker[:index] + infoString + marker[index + len(infoString):]
