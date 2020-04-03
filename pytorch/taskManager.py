import multiprocessing
from multiprocessing import Process
import os
import platform
import inspect
from Utilities import SimpleProgressBar
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


def worker(taskFunction, dataSample, workerId, jobId):
    result = taskFunction(dataSample)
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
        progressBar = SimpleProgressBar(len(taskData), "Scheduled")
        for jobId in range(len(taskData)):
            dataSample, workerId = dataLoader(taskData, numWorkers, jobId)
            workerProcesses.append(workerPool.apply_async(worker, args=(taskFunction, dataSample, workerId, jobId)))
            progressBar.update(jobId+1)

        # collect results
        results = []
        progressBar = SimpleProgressBar(len(taskData), "Completed")
        for p in workerProcesses:
            (jobId,result) = p.get()
            results.append(result)
            progressBar.update(jobId+1)

        # leave one blank line
        print()
        return results


