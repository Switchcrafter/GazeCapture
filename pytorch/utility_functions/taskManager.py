import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager, SyncManager
import os
import platform
import inspect
from utility_functions.Utilities import SimpleProgressBar, MultiProgressBar
import time

def print_sysinfo():
    print('\nPython version  :', platform.python_version())
    print('compiler        :', platform.python_compiler())
    print('system     :', platform.system())
    print('release    :', platform.release())
    print('machine    :', platform.machine())
    print('processor  :', platform.processor())
    print('CPU count  :', multiprocessing.cpu_count())
    print('interpreter:', platform.architecture()[0])
    print('\n\n')


def worker(taskFunction, dataSample, workerId, jobId, progressBar):
    result = taskFunction(dataSample, jobId, progressBar)
    return (jobId, result)

def job(taskFunction, taskData, dataLoader, numWorkers = multiprocessing.cpu_count()):
    # single process tasks
    if dataLoader == None:
        taskFunction(taskData)
    else: # Multiprocess tasks
        # create a worker pool
        workerPool = multiprocessing.Pool(processes=numWorkers)

        # assign tasks to workers
        workerProcesses = []

        # synchronization manager for shared MultiProgressBar between processes
        BaseManager.register('MultiProgressBar', MultiProgressBar)
        manager = BaseManager()
        manager.start()
        progressBar = manager.MultiProgressBar(len(taskData), "Progress ")

        scheduleBar = SimpleProgressBar(len(taskData), "Scheduled")
        for jobId in range(len(taskData)):
            dataSample, workerId = dataLoader(taskData, numWorkers, jobId)
            workerProcesses.append(workerPool.apply_async(worker, args=(taskFunction, dataSample, workerId, jobId, progressBar)))
            scheduleBar.update(jobId+1)

        # collect results
        results = []
        completeBar = SimpleProgressBar(len(taskData), "Completed")
        for p in workerProcesses:
            (jobId,result) = p.get()
            completeBar.update(jobId+1)
            results.append(result)
        
        # leave one blank line
        print()
        return results


