import os
import re
import json
import sys
import argparse
import taskManager
import shutil
import numpy as np
import scipy.io as sio
from PIL import Image as PILImage
from face_utilities import *


################################################################################
## Utility functions
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

def preparePath(path, clear=False):
    if not os.path.isdir(path):
        try:
            os.makedirs(path, 0o777)
        except FileExistsError:
            pass
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path


def logError(msg, critical=False):
    print(msg)
    if critical:
        sys.exit(1)

def json_read(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data

def json_write(filename, data):
    with open(filename, "w") as write_file:
        json.dump(data, write_file)


def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
    res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]

    return res

def RC_cropImage(img, bbox):
    # print(bbox)
    bbox = np.array(bbox, int)
    # rect = ((bbox[0],bbox[1]), (bbox[2],bbox[3]), bbox[4])
    return crop_rect(img, bbox)

################################################################################
## DataIndexers
################################################################################

def getFileList(path, extensionList):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if isExtension(file, extensionList):
                files.append(os.path.join(r, file))
    return files

def getDirList(path, regexString):
    nameFormat = re.compile(regexString)
    folders = [dirName for dirName in os.listdir(path) if nameFormat.match(dirName)]
    return folders

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


# Equivalent: generate_faces
def ROIDetectionTask(directory):
    recording_path = os.path.join(args.input, directory)
    output_path = os.path.join(args.output, directory)
    # Read information for valid frames
    filenames = json_read(os.path.join(recording_path, "frames.json"))

    faceInfoDict = newFaceInfoDict()
    for idx, filename in enumerate(filenames):
        # load image
        image_path = os.path.join(recording_path, "frames", filename)
        image = PILImage.open(image_path)
        image = np.array(image.convert('RGB'))

        # ROI detection
        shape_np, isValid = find_face_dlib(image)
        if args.rc:
            face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)
            faceInfoDict, faceInfoIdx = rc_faceEyeRectsToFaceInfoDict(faceInfoDict,
                                                                face_rect,
                                                                left_eye_rect,
                                                                right_eye_rect,
                                                                isValid)
        else:
            face_rect, left_eye_rect, right_eye_rect, isValid = landmarksToRects(shape_np, isValid)
            faceInfoDict, faceInfoIdx = faceEyeRectsToFaceInfoDict(faceInfoDict,
                                                                face_rect,
                                                                left_eye_rect,
                                                                right_eye_rect,
                                                                isValid)
    # ensure the output directory exists
    preparePath(output_path)
    # write the Face, LeftEye and RightEye
    json_write(os.path.join(output_path, 'dlibFace.json'), faceInfoDict["Face"])
    json_write(os.path.join(output_path, 'dlibLeftEye.json'), faceInfoDict["LeftEye"])
    json_write(os.path.join(output_path, 'dlibRightEye.json'), faceInfoDict["RightEye"])
    return


# Equivalent: prepareDataset
def ROIExtractionTask(directory):

    recDir = os.path.join(args.input, directory)
    dlibDir = os.path.join(args.metapath, directory)
    recDirOut = os.path.join(args.output, directory)

    # Output structure
    meta = {
        'labelRecNum': [],
        'frameIndex': [],
        'labelDotXCam': [],
        'labelDotYCam': [],
        'labelFaceGrid': [],
    }

    # Read metadata JSONs from metapath
    appleFace = json_read(os.path.join(dlibDir, 'dlibFace.json'))
    appleLeftEye = json_read(os.path.join(dlibDir, 'dlibLeftEye.json'))
    appleRightEye = json_read(os.path.join(dlibDir, 'dlibRightEye.json'))

    # Read input JSONs from inputpath
    dotInfo = json_read(os.path.join(recDir, 'dotInfo.json'))
    faceGrid = json_read(os.path.join(recDir, 'faceGrid.json'))
    frames = json_read(os.path.join(recDir, 'frames.json'))
    # info = json_read(os.path.join(recDir, 'info.json'))
    # screen = json_read(os.path.join(recDir, 'screen.json'))

    # prepape output paths
    facePath = preparePath(os.path.join(recDirOut, 'appleFace'))
    leftEyePath = preparePath(os.path.join(recDirOut, 'appleLeftEye'))
    rightEyePath = preparePath(os.path.join(recDirOut, 'appleRightEye'))

    # Preprocess
    allValid = np.logical_and(np.logical_and(appleFace['IsValid'], appleLeftEye['IsValid']),
                                np.logical_and(appleRightEye['IsValid'], faceGrid['IsValid']))

    frames = np.array([int(re.match('(\d{5})\.jpg$', x).group(1)) for x in frames])

    if args.rc:
        bboxFromJson = lambda data: np.stack((data['X'], data['Y'], data['W'], data['H'], data['Theta']), axis=1).astype(int)
        # handle original face_grid data separately
        bboxFromJsonFaceGrid = lambda data: np.stack((data['X'], data['Y'], data['W'], data['H']), axis=1).astype(int)
        faceBbox = bboxFromJson(appleFace) + [-1, -1, 1, 1, 0]  # for compatibility with matlab code
        leftEyeBbox = bboxFromJson(appleLeftEye) + [0, -1, 0, 0, 0]
        rightEyeBbox = bboxFromJson(appleRightEye) + [0, -1, 0, 0, 0]
        faceGridBbox = bboxFromJsonFaceGrid(faceGrid)
    else:
        bboxFromJson = lambda data: np.stack((data['X'], data['Y'], data['W'], data['H']), axis=1).astype(int)
        faceBbox = bboxFromJson(appleFace) + [-1, -1, 1, 1]  # for compatibility with matlab code
        leftEyeBbox = bboxFromJson(appleLeftEye) + [0, -1, 0, 0]
        rightEyeBbox = bboxFromJson(appleRightEye) + [0, -1, 0, 0]
        leftEyeBbox[:, :2] += faceBbox[:, :2]  # relative to face
        rightEyeBbox[:, :2] += faceBbox[:, :2]
        faceGridBbox = bboxFromJson(faceGrid)

    for j, frame in enumerate(frames):
        # Can we use it?
        if not allValid[j]:
            continue

        # Load image
        imgFile = os.path.join(recDir, 'frames', '%05d.jpg' % frame)
        if not os.path.isfile(imgFile):
            logError('Warning: Could not read image file %s!' % imgFile)
            continue
        img = PILImage.open(imgFile)
        if img is None:
            logError('Warning: Could not read image file %s!' % imgFile)
            continue
        img = np.array(img.convert('RGB'))

        # Crop images
        if args.rc:
            imFace = RC_cropImage(img, faceBbox[j, :])
            imEyeL = RC_cropImage(img, leftEyeBbox[j, :])
            imEyeR = RC_cropImage(img, rightEyeBbox[j, :])
        else:
            imFace = cropImage(img, faceBbox[j, :])
            imEyeL = cropImage(img, leftEyeBbox[j, :])
            imEyeR = cropImage(img, rightEyeBbox[j, :])

        # Save images
        PILImage.fromarray(imFace).save(os.path.join(facePath, '%05d.jpg' % frame), quality=95)
        PILImage.fromarray(imEyeL).save(os.path.join(leftEyePath, '%05d.jpg' % frame), quality=95)
        PILImage.fromarray(imEyeR).save(os.path.join(rightEyePath, '%05d.jpg' % frame), quality=95)

        # Collect metadata
        meta['labelRecNum'] += [int(directory)]
        meta['frameIndex'] += [frame]
        meta['labelDotXCam'] += [dotInfo['XCam'][j]]
        meta['labelDotYCam'] += [dotInfo['YCam'][j]]
        meta['labelFaceGrid'] += [faceGridBbox[j, :]]

    return meta


def compareTask(meta):
    # Load reference metadata
    print('Will compare to the reference GitHub dataset metadata.mat...')
    reference = sio.loadmat('./reference_metadata.mat', struct_as_record=False)
    reference['labelRecNum'] = reference['labelRecNum'].flatten()
    reference['frameIndex'] = reference['frameIndex'].flatten()
    reference['labelDotXCam'] = reference['labelDotXCam'].flatten()
    reference['labelDotYCam'] = reference['labelDotYCam'].flatten()
    reference['labelTrain'] = reference['labelTrain'].flatten()
    reference['labelVal'] = reference['labelVal'].flatten()
    reference['labelTest'] = reference['labelTest'].flatten()

    # Find mapping
    mKey = np.array(['%05d_%05d' % (rec, frame) for rec, frame in zip(meta['labelRecNum'], meta['frameIndex'])],
                    np.object)
    rKey = np.array(
        ['%05d_%05d' % (rec, frame) for rec, frame in zip(reference['labelRecNum'], reference['frameIndex'])],
        np.object)
    mIndex = {k: i for i, k in enumerate(mKey)}
    rIndex = {k: i for i, k in enumerate(rKey)}
    mToR = np.zeros((len(mKey, )), int) - 1
    for i, k in enumerate(mKey):
        if k in rIndex:
            mToR[i] = rIndex[k]
        else:
            logError('Did not find rec_frame %s from the new dataset in the reference dataset!' % k)
    rToM = np.zeros((len(rKey, )), int) - 1
    for i, k in enumerate(rKey):
        if k in mIndex:
            rToM[i] = mIndex[k]
        else:
            logError('Did not find rec_frame %s from the reference dataset in the new dataset!' % k, critical=False)
            # break

    # Copy split from reference
    meta['labelTrain'] = np.zeros((len(meta['labelRecNum'], )), np.bool)
    meta['labelVal'] = np.ones((len(meta['labelRecNum'], )), np.bool)  # default choice
    meta['labelTest'] = np.zeros((len(meta['labelRecNum'], )), np.bool)

    validMappingMask = mToR >= 0
    meta['labelTrain'][validMappingMask] = reference['labelTrain'][mToR[validMappingMask]]
    meta['labelVal'][validMappingMask] = reference['labelVal'][mToR[validMappingMask]]
    meta['labelTest'][validMappingMask] = reference['labelTest'][mToR[validMappingMask]]

    # Write out metadata
    metaFile = os.path.join(args.output, 'metadata.mat')
    print('Writing out the metadata.mat to %s...' % metaFile)
    sio.savemat(metaFile, meta)

    # Statistics
    nMissing = np.sum(rToM < 0)
    nExtra = np.sum(mToR < 0)
    totalMatch = len(mKey) == len(rKey) and np.all(np.equal(mKey, rKey))
    print('======================\n\tSummary\n======================')
    print('Total added %d frames from %d recordings.' % (len(meta['frameIndex']), len(np.unique(meta['labelRecNum']))))
    if nMissing > 0:
        print(
            'There are %d frames missing in the new dataset. This may affect the results. Check the log to see which files are missing.' % nMissing)
    else:
        print('There are no missing files.')
    if nExtra > 0:
        print(
            'There are %d extra frames in the new dataset. This is generally ok as they were marked for validation split only.' % nExtra)
    else:
        print('There are no extra files that were not in the reference dataset.')
    if totalMatch:
        print('The new metadata.mat is an exact match to the reference from GitHub (including ordering)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
    parser.add_argument('--input', help="input directory path", default="./gc-data/")
    parser.add_argument('--output', help="output directory path", default="./gc-data-meta-rc")
    parser.add_argument('--metapath', help="metadata path", default="./gc-data-meta/")
    parser.add_argument('--task', help="task name: copyTask, ROIDetectionTask, ROIExtractionTask", default="ROIDetectionTask")
    parser.add_argument('--rc', action='store_true', help="apply rotation correction", default=False)
    parser.add_argument('--source_compare', action='store_true', help="compare against source", default=False)
    args = parser.parse_args()

    # pre-processing for the task
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
    elif args.task == "ROIDetectionTask":
        taskFunction = ROIDetectionTask
        sessionRegex = '([0-9]){5}'
        taskData = getDirList(args.input, sessionRegex)
        dataLoader = ListLoader
    elif args.task == "ROIExtractionTask":
        taskFunction = ROIExtractionTask
        taskData = getDirList(args.input, '([0-9]){5}')
        dataLoader = ListLoader

    # run the job
    output = taskManager.job(taskFunction, taskData, dataLoader)

    # post-processing after the task has completed
    if args.task == "ROIExtractionTask":
        # Output structure
        meta = {
            'labelRecNum': [],
            'frameIndex': [],
            'labelDotXCam': [],
            'labelDotYCam': [],
            'labelFaceGrid': [],
        }
        for m in output:
            meta['labelRecNum'] += m['labelRecNum']
            meta['frameIndex'] += m['frameIndex']
            meta['labelDotXCam'] += m['labelDotXCam']
            meta['labelDotYCam'] += m['labelDotYCam']
            meta['labelFaceGrid'] += m['labelFaceGrid']

        # Integrate
        meta['labelRecNum'] = np.stack(meta['labelRecNum'], axis=0).astype(np.int16)
        meta['frameIndex'] = np.stack(meta['frameIndex'], axis=0).astype(np.int32)
        meta['labelDotXCam'] = np.stack(meta['labelDotXCam'], axis=0)
        meta['labelDotYCam'] = np.stack(meta['labelDotYCam'], axis=0)
        meta['labelFaceGrid'] = np.stack(meta['labelFaceGrid'], axis=0).astype(np.uint8)

        # print(meta)
        # compareTask(meta)











