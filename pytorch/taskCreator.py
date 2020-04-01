import os
import argparse
import taskManager
import shutil


################################################################################
## Dataloaders
################################################################################
def getDirNameExt(filepath):
    dir, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    return dir, name, ext

def getRelativePath(filepath, input):
    return filepath.replace(input,"")

def isExtension(fileName, extensionList):
    fileName, ext = os.path.splitext(fileName)
    if ext in extensionList:
        return True
    else:
        return False

def getFileList(path, extensionList):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if isExtension(file, extensionList):
                files.append(os.path.join(r, file))
    return files

def preparePath(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

################################################################################
## Dataloaders
################################################################################

def ListLoader(listData, numWorkers, i):
  return listData[i], i % numWorkers


################################################################################
## Task Definitions
################################################################################

# Tasks
def noneTask(fileName):
    return fileName

def cubeTask(x):
    return x**3

def copyTask(filepath):
    from_dir, from_filename, from_ext = getDirNameExt(filepath)
    relative_path = getRelativePath(filepath, args.input)
    to_file = os.path.join(args.output, relative_path)
    # print(filepath + "-->" + to_file)
    preparePath(to_file)
    shutil.copy(filepath, to_file)
    return to_file


def resizeTask(filePath):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
    parser.add_argument('--input', help="input directory path", default="./gc-data/")
    parser.add_argument('--output', help="output directory path", default="./gc-data2/")
    parser.add_argument('--task', help="task name", default="copyTask")
    args = parser.parse_args()

    if args.task == "noneTask":
        taskFunction = noneTask
        extensionList = [".jpg", ".jpeg", ".JPG", ".JPEG"]
        taskData = getFileList("./gc-data/", extensionList)
        dataLoader = ListLoader
    elif args.task == "cubeTask":
        taskFunction = cubeTask
        taskData = [0,1,3,4,6,7]
        dataLoader = ListLoader
    elif args.task == "copyTask":
        taskFunction = copyTask
        extensionList = [".jpg", ".jpeg", ".JPG", ".JPEG"]
        taskData = getFileList(args.input, extensionList)
        dataLoader = ListLoader

    output = taskManager.job(taskFunction, taskData, dataLoader)











