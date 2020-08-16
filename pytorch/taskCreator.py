import os
import re
import json
import sys
import csv
import math
import shutil
import argparse
import torch
import random
import subprocess
from utility_functions import taskManager
import pandas as pd
import scipy.io as sio
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from utility_functions.face_utilities import *
import dateutil.parser
from utility_functions.cam2screen import screen2cam
from utility_functions.Utilities import MultiProgressBar


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
    if len(bbox) == 5:
        return crop_rect(img, bbox)
    else:
        aSrc = np.maximum(bbox[:2], 0)
        bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

        aDst = aSrc - bbox[:2]
        bDst = aDst + (bSrc - aSrc)

        res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
        res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]

        return res

def RC_cropImage(img, bbox):
    bbox = np.array(bbox, int)
    return crop_rect(img, bbox)

def marker(num):
    markerList = ['','.','^','s','+','D','o','p','P','X','$f$']
    return markerList[num%len(markerList)]

def getScreenOrientation(capture_data):
    orientation = 0

    # Camera Offset and Screen Orientation compensation
    # if capture_data['NativeOrientation'] == "Landscape":
    if capture_data['Orientation'] == "Landscape":
        # Camera above screen
        # - Landscape on Surface devices
        orientation = 1
    elif capture_data['Orientation'] == "LandscapeFlipped":
        # Camera below screen
        # - Landscape inverted on Surface devices
        orientation = 2
    elif capture_data['Orientation'] == "PortraitFlipped":
        # Camera left of screen
        # - Portrait with camera on left on Surface devices
        orientation = 3
    elif capture_data['Orientation'] == "Portrait":
        # Camera right of screen
        # - Portrait with camera on right on Surface devices
        orientation = 4

    return orientation

def getCaptureTimeString(capture_data):
    sessiontime = dateutil.parser.parse(capture_data["SessionTimestamp"])
    currenttime = dateutil.parser.parse(capture_data["Timestamp"])
    timedelta = sessiontime - currenttime
    return str(timedelta.total_seconds())


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
    return sorted(folders)

# used by prepareEyeCatcherTask
# input path is to the folder containing data from various devices e.g. /data/Source/EyeCapture/200407/
def getCaptureSessionDirList(path):
    session_paths = []
    devices = os.listdir(path)
    for device in devices:
        users = os.listdir(os.path.join(path, device))
        for user in users:
            sessions = sorted(os.listdir(os.path.join(path, device, user)), key=str)
            for session in sessions:
                session_paths.append(os.path.join(device, user, session))
    return session_paths

# used by prepareEyeCatcherTask
def getCaptureSessionFileList(path):
    files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "frames")) if f.endswith('.json') and not f == "session.json"]
    return files
################################################################################
## Dataloaders
################################################################################

def ListLoader(listData, numWorkers, i):
  return listData[i], i % numWorkers


################################################################################
## Task Definitions
################################################################################

# Multi-process Tasks

def noneTask(fileName, jobId, progressbar):
    return fileName

def cubeTask(x, jobId, progressbar):
    return x**3

def copyTask(filepath, jobId, progressbar):
    from_dir, from_filename, from_ext = getDirNameExt(filepath)
    relative_path = getRelativePath(from_dir, args.input)
    to_dir = os.path.join(args.output, relative_path)
    to_file = os.path.join(to_dir, from_filename+from_ext)
    # print(filepath + "-->" + to_file)
    preparePath(to_dir)
    shutil.copy(filepath, to_file)
    return to_file

def resizeTask(filePath, jobId, progressbar):
    pass

# Equivalent: prepare_EyeCatcher
def prepareEyeCatcherTask(directory, directory_idx, progressbar):
    captures = sorted(getCaptureSessionFileList(os.path.join(args.input, directory)), key=str)
    total_captures = len(captures)

    # Read directory level json
    deviceMetrics_data = json_read(os.path.join(args.input, directory, "deviceMetrics.json"))
    info_data = json_read(os.path.join(args.input, directory, "info.json"))
    screen_data = json_read(os.path.join(args.input, directory, "screen.json"))

    # dotinfo.json - { "DotNum": [ 0, 0, ... ],
    #                  "XPts": [ 160, 160, ... ],
    #                  "YPts": [ 284, 284, ... ],
    #                  "XCam": [ 1.064, 1.064, ... ],
    #                  "YCam": [ -6.0055, -6.0055, ... ],
    #                  "Confidence": [ 59.3, 94.2, ... ],
    #                  "Time": [ 0.205642, 0.288975, ... ] }
    #
    # PositionIndex == DotNum
    # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
    dotinfo = {
        "DotNum": [],
        "XPts": [],
        "YPts": [],
        "XCam": [],
        "YCam": [],
        "Confidence": [],
        "Time": []
    }

    output_path = os.path.join(args.output, f"{directory_idx:05d}")
    output_frame_path = os.path.join(output_path, "frames")

    faceInfoDict = newFaceInfoDict()

    # frames.json - ["00000.jpg","00001.jpg"]
    frames = []

    facegrid = {
        "X": [],
        "Y": [],
        "W": [],
        "H": [],
        "IsValid": []
    }

    if directory_idx % 10 < 8:
        dataset_split = "train"
    elif directory_idx % 10 < 9:
        dataset_split = "val"
    else:
        dataset_split = "test"

    # info.json - {"TotalFrames":99,"NumFaceDetections":97,"NumEyeDetections":56,"Dataset":"train","DeviceName":"iPhone 6"}
    info = {
        "TotalFrames": total_captures,
        "NumFaceDetections": 0,
        "NumEyeDetections": 0,
        "Dataset": dataset_split,
        "DeviceName": info_data["DeviceName"]
    }

    # screen.json - { "H": [ 568, 568, ... ], "W": [ 320, 320, ... ], "Orientation": [ 1, 1, ... ] }
    screen = {
        "H": [],
        "W": [],
        "Orientation": []
    }

    # ensure the output directories exist
    preparePath(args.output)
    preparePath(output_path)
    preparePath(output_frame_path)

    screen_orientation = getScreenOrientation(screen_data)
    progressbar.addSubProcess(directory_idx, len(captures))
    for capture_idx, capture in enumerate(captures):
        progressbar.update(directory_idx, capture_idx+1)

        capture_json_path = os.path.join(args.input, directory, "frames", capture + ".json")
        capture_jpg_path = os.path.join(args.input, directory, "frames", capture + ".jpg")

        if os.path.isfile(capture_json_path) and os.path.isfile(capture_jpg_path):
            capture_data = json_read(capture_json_path)
            capture_image = PILImage.open(capture_jpg_path)
            capture_image_np = np.array(capture_image)  # dlib wants images in numpy array format

            shape_np, isValid = find_face_dlib(capture_image_np)
            info["NumFaceDetections"] = info["NumFaceDetections"] + 1
            if args.rc:
                face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)
                faceInfoDict, faceInfoIdx = rc_faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect,
                                                                        right_eye_rect, isValid)
            else:
                face_rect, left_eye_rect, right_eye_rect, isValid = landmarksToRects(shape_np, isValid)
                faceInfoDict, faceInfoIdx = faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect,
                                                                        right_eye_rect, isValid)

            # facegrid.json - { "X": [ 6, 6, ... ], "Y": [ 10, 10, ... ], "W": [ 13, 13, ... ], "H": [ 13, 13, ... ], "IsValid": [ 1, 1, ... ] }
            if isValid:
                faceGridX, faceGridY, faceGridW, faceGridH = generate_face_grid_rect(face_rect, capture_image.width,
                                                                                        capture_image.height)
            else:
                faceGridX = 0
                faceGridY = 0
                faceGridW = 0
                faceGridH = 0

            facegrid["X"].append(faceGridX)
            facegrid["Y"].append(faceGridY)
            facegrid["W"].append(faceGridW)
            facegrid["H"].append(faceGridH)
            facegrid["IsValid"].append(isValid)


            info["NumEyeDetections"] = info["NumEyeDetections"] + 1
            # screen.json - { "H": [ 568, 568, ... ], "W": [ 320, 320, ... ], "Orientation": [ 1, 1, ... ] }
            screen["H"].append(screen_data['H'])
            screen["W"].append(screen_data['W'])
            screen["Orientation"].append(screen_orientation)

            # dotinfo.json - { "DotNum": [ 0, 0, ... ],
            #                  "XPts": [ 160, 160, ... ],
            #                  "YPts": [ 284, 284, ... ],
            #                  "XCam": [ 1.064, 1.064, ... ],
            #                  "YCam": [ -6.0055, -6.0055, ... ],
            #                  "Confidence": [ 59.3, 94.2, ... ],
            #                  "Time": [ 0.205642, 0.288975, ... ] }
            #
            # PositionIndex == DotNum
            # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
            x_raw = capture_data["XRaw"]
            y_raw = capture_data["YRaw"]
            x_cam, y_cam = screen2cam(x_raw,  # xScreenInPoints
                                        y_raw,  # yScreenInPoints
                                        screen_orientation,  # orientation,
                                        screen_data["W"],  # widthScreenInPoints
                                        screen_data["H"],  # heightScreenInPoints
                                        deviceName=info_data["DeviceName"])
            confidence = capture_data["Confidence"]

            dotinfo["DotNum"].append(0)  # TODO replace with dot number as needed
            dotinfo["XPts"].append(x_raw)
            dotinfo["YPts"].append(y_raw)
            dotinfo["XCam"].append(x_cam)
            dotinfo["YCam"].append(y_cam)
            dotinfo["Confidence"].append(confidence)
            dotinfo["Time"].append(0)  # TODO replace with timestamp as needed

            # Convert image from PNG to JPG
            frame_name = str(f"{capture_idx:05d}.jpg")
            frames.append(frame_name)

            shutil.copyfile(capture_jpg_path, os.path.join(output_frame_path, frame_name))
        else:
            print(f"Error processing capture {capture}")

    # write json files
    json_write(os.path.join(output_path, 'frames.json'), frames)
    json_write(os.path.join(output_path, 'screen.json'), screen)
    json_write(os.path.join(output_path, 'info.json'), info)
    json_write(os.path.join(output_path, 'dotInfo.json'), dotinfo)
    json_write(os.path.join(output_path, 'faceGrid.json'), facegrid)
    # write the Face, LeftEye and RightEye
    json_write(os.path.join(output_path, 'dlibFace.json'), faceInfoDict["Face"])
    json_write(os.path.join(output_path, 'dlibLeftEye.json'), faceInfoDict["LeftEye"])
    json_write(os.path.join(output_path, 'dlibRightEye.json'), faceInfoDict["RightEye"])

# Equivalent: generate_faces
def ROIDetectionTask(directory, directory_idx, progressbar):
    if args.apple:
        return

    recording_path = os.path.join(args.input, directory)
    output_path = os.path.join(args.output, directory)
    # Read information for valid frames
    filenames = json_read(os.path.join(recording_path, "frames.json"))

    faceInfoDict = newFaceInfoDict()
    progressbar.addSubProcess(directory_idx, len(filenames))
    for idx, filename in enumerate(filenames):
        progressbar.update(directory_idx, idx+1)
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

# Equivalent: prepareDataset_dlib
def ROIExtractionTask(directory, directory_idx, progressbar):

    recDir = os.path.join(args.input, directory)
    recDirOut = os.path.join(args.output, directory)

    # Output structure
    meta = {
        'labelRecNum': [],
        'frameIndex': [],
        'labelDotXCam': [],
        'labelDotYCam': [],
        'labelFaceGrid': [],
        'labelTrain': [],
        'labelVal': [],
        'labelTest': []
    }

    if args.apple:
        # Read metadata JSONs from recDir
        appleFace = json_read(os.path.join(recDir, 'appleFace.json'))
        appleLeftEye = json_read(os.path.join(recDir, 'appleLeftEye.json'))
        appleRightEye = json_read(os.path.join(recDir, 'appleRightEye.json'))
    else: # use Dlib crops generated from ROIDetection phase
        # Read metadata JSONs from metapath
        dlibDir = os.path.join(args.metapath, directory)
        appleFace = json_read(os.path.join(dlibDir, 'dlibFace.json'))
        appleLeftEye = json_read(os.path.join(dlibDir, 'dlibLeftEye.json'))
        appleRightEye = json_read(os.path.join(dlibDir, 'dlibRightEye.json'))

    # prepape output paths
    facePath = preparePath(os.path.join(recDirOut, 'appleFace'))
    leftEyePath = preparePath(os.path.join(recDirOut, 'appleLeftEye'))
    rightEyePath = preparePath(os.path.join(recDirOut, 'appleRightEye'))

    # directory, frame, dotInfo, faceGridBbox, info
    # Read input JSONs from inputpath
    dotInfo = json_read(os.path.join(recDir, 'dotInfo.json'))
    faceGrid = json_read(os.path.join(recDir, 'faceGrid.json'))
    frames = json_read(os.path.join(recDir, 'frames.json'))
    info = json_read(os.path.join(recDir, 'info.json'))
    screen = json_read(os.path.join(recDir, 'screen.json'))

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

    progressbar.addSubProcess(directory_idx, len(frames))
    for j, frame in enumerate(frames):
        progressbar.update(directory_idx, j+1)

        # Can we use it?
        if not allValid[j]:
            continue

        if args.portraitOnly:
            # Is upright data?
            if screen['Orientation'][j] != 1:
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
        imFace = cropImage(img, faceBbox[j, :])
        imEyeL = cropImage(img, leftEyeBbox[j, :])
        imEyeR = cropImage(img, rightEyeBbox[j, :])

        # Rotation Correction FaceGrid
        if args.rc:
            faceGridPath = preparePath(os.path.join(recDirOut, 'faceGrid'))
            imFaceGrid = generate_grid2(faceBbox[j, :], img)

            imFace = cv2.resize(imFace, (256, 256), cv2.INTER_AREA)
            imEyeL = cv2.resize(imEyeL, (256, 256), cv2.INTER_AREA)
            imEyeR = cv2.resize(imEyeR, (256, 256), cv2.INTER_AREA)
            imFaceGrid = cv2.resize(imFaceGrid, (256, 256), cv2.INTER_AREA)

        # Save images
        PILImage.fromarray(imFace).save(os.path.join(facePath, '%05d.jpg' % frame), quality=95)
        PILImage.fromarray(imEyeL).save(os.path.join(leftEyePath, '%05d.jpg' % frame), quality=95)
        PILImage.fromarray(imEyeR).save(os.path.join(rightEyePath, '%05d.jpg' % frame), quality=95)
        if args.rc:
            PILImage.fromarray(imFaceGrid).save(os.path.join(faceGridPath, '%05d.jpg' % frame), quality=95)

        # Collect metadata
        meta['labelRecNum'] += [int(directory)]
        meta['frameIndex'] += [frame]
        meta['labelDotXCam'] += [dotInfo['XCam'][j]]
        meta['labelDotYCam'] += [dotInfo['YCam'][j]]
        meta['labelFaceGrid'] += [faceGridBbox[j, :]]
        split = info["Dataset"]
        meta['labelTrain'] += [split == "train"]
        meta['labelVal'] += [split == "val"]
        meta['labelTest'] += [split == "test"]

        # # Data Mirroring
        # if args.mirror:
        #     imFace_mirror = cv2.flip(imFace, 1)
        #     imEyeL_mirror = cv2.flip(imEyeL, 1)
        #     imEyeR_mirror = cv2.flip(imEyeR, 1)
        #     PILImage.fromarray(imFace_mirror).save(os.path.join(facePath, '%05d_mirror.jpg' % frame), quality=95)
        #     PILImage.fromarray(imEyeL_mirror).save(os.path.join(leftEyePath, '%05d_mirror.jpg' % frame), quality=95)
        #     PILImage.fromarray(imEyeR_mirror).save(os.path.join(rightEyePath, '%05d_mirror.jpg' % frame), quality=95)
        #     if args.rc:
        #         imFaceGrid_mirror = cv2.flip(imFaceGrid, 1)
        #         PILImage.fromarray(imFaceGrid_mirror).save(os.path.join(faceGridPath, '%05d_mirror.jpg' % frame), quality=95)

        #     (XFactor, YFactor) = (-1.0, 1.0) if screen['Orientation'][j] <= 2 else (1.0, -1.0)
        #     # mirror faceGridBbox
        #     f = [24-faceGridBbox[j, 0]-faceGridBbox[j, 2], faceGridBbox[j, 1], faceGridBbox[j, 2], faceGridBbox[j, 3]]
        #     # Mirror metadata - Assuming Camera is on the top center
        #     meta['labelRecNum'] += [int(directory)]
        #     meta['frameIndex'] += [frame]
        #     meta['labelDotXCam'] += [XFactor * dotInfo['XCam'][j]]
        #     meta['labelDotYCam'] += [YFactor * dotInfo['YCam'][j]]
        #     meta['labelFaceGrid'] += [f]
        #     meta['labelTrain'] += [split == "train"]
        #     meta['labelVal'] += [split == "val"]
        #     meta['labelTest'] += [split == "test"]

    return meta


# Single process Tasks
def compareTask(meta):
    if args.reference != "":
        # Load reference metadata
        print('Will compare to the reference GitHub dataset metadata.mat...')
        reference = sio.loadmat(args.reference, struct_as_record=False)
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
    print('======================\n\tSummary\n======================')
    print('Total added %d frames from %d recordings.' % (len(meta['frameIndex']), len(np.unique(meta['labelRecNum']))))

    if args.reference != "":
        nMissing = np.sum(rToM < 0)
        nExtra = np.sum(mToR < 0)
        totalMatch = len(mKey) == len(rKey) and np.all(np.equal(mKey, rKey))
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

def plotErrorTask(All_RMS_Errors):
    # Make a data frame
    rms_object = {'x': range(1, 31)}
    for key in All_RMS_Errors.keys():
        if All_RMS_Errors[key]["Plot"]:
            rms_object[key] = np.array((All_RMS_Errors[key])['RMS_Errors'])

    df_rms = pd.DataFrame(rms_object)

    # style
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    # multiple line plot
    num = 0
    for column in df_rms.drop('x', axis=1):
        num += 1
        ax1.plot(df_rms['x'], df_rms[column], marker=marker(num), color=palette(num), linewidth=2, alpha=0.9, label=column)

    # Add titles
    ax1.set_title("RMS Errors by Epoch", loc='left', fontsize=12, fontweight=0, color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMS Error")

    best_rms_object = {'x': range(1, 31)}
    for key in All_RMS_Errors.keys():
        if All_RMS_Errors[key]["Plot"]:
            best_rms_object[key] = np.array((All_RMS_Errors[key])['Best_RMS_Errors'])

    # Make a data frame
    df_best_rms = pd.DataFrame(best_rms_object)

    # multiple line plot
    num = 0
    for column in df_best_rms.drop('x', axis=1):
        num += 1
        ax2.plot(df_best_rms['x'], df_best_rms[column], marker=marker(num), color=palette(num), linewidth=2, alpha=0.9, label=column)

    # Add titles
    ax2.set_title("Best RMS Errors by Epoch", loc='left', fontsize=12, fontweight=0, color='red')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMS Error")

    # Add legend
    # plt.legend(loc=2, ncol=2)
    plt.legend(bbox_to_anchor=(0.6, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def plotErrorHeatmapTask(results_path):
    jsondata = json_read(results_path)
    x, y, c = [], [], []

    for datapoint in jsondata:
        x.append(datapoint["gazePoint"][0])
        y.append(datapoint["gazePoint"][1])
        c.append(math.sqrt(((datapoint["gazePoint"][0]-datapoint["gazePrediction"][0])**2)+((datapoint["gazePoint"][1]-datapoint["gazePrediction"][1])**2)))

    cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=min(c), vmax=max(c))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    plt.scatter(x, y, c=scalarMap.to_rgba(c))
    #plt.colorbar(scalarMap)
    #cb.set_label('Error (cm)')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Heatmap of Error')
    plt.show()

def plotGazePointHeatmapTask(results_path):
    jsondata = json_read(results_path)
    x, y, c = [], [], []
    gazepoints = {}

    for datapoint in jsondata:
        # create unique gazepoint key
        key = ('%.5f' % datapoint["gazePoint"][0]) + ('%.5f' % datapoint["gazePoint"][1])

        # find key
        if key in gazepoints:
            index = gazepoints[key]
        else:
            index = x.__len__()
            x.append(datapoint["gazePoint"][0])
            y.append(datapoint["gazePoint"][1])
            c.append(0)
            gazepoints[key] = index

        c[index] += 1

    cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=min(c), vmax=max(c))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    plt.scatter(x, y, c=scalarMap.to_rgba(c))
    #plt.colorbar(scalarMap)
    #cb.set_label('Count')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.show()

def plotErrorHistogramTask(results_path):
    jsondata = json_read(results_path)
    x = []
    for datapoint in jsondata:
        x.append(math.sqrt(((datapoint["gazePoint"][0]-datapoint["gazePrediction"][0])**2)+((datapoint["gazePoint"][1]-datapoint["gazePrediction"][1])**2)))

    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5, density=1)
    plt.xlabel('Error (cm)')
    plt.ylabel('Probability')
    plt.title('Histogram of Error')
    plt.show()

def parseResultsTask(results_path):
    jsondata = json_read(results_path)
    csvwriter = csv.writer(open("best_results.csv", "w", newline=''))
    csvwriter.writerow(
        ['frameId', 'frame0', 'frame1', 'gazePointX', 'gazePointY', 'gazePreditionX', 'gazePredictionY', 'distance'])

    for datapoint in jsondata:
        distance = math.sqrt(((datapoint["gazePoint"][0] - datapoint["gazePrediction"][0]) ** 2) + (
                    (datapoint["gazePoint"][1] - datapoint["gazePrediction"][1]) ** 2))
        csvwriter.writerow(
            [f'{datapoint["frame"][0]}_{datapoint["frame"][1]}', datapoint["frame"][0], datapoint["frame"][1],
            datapoint["gazePoint"][0], datapoint["gazePoint"][1], datapoint["gazePrediction"][0],
            datapoint["gazePrediction"][1], distance])

def dataStatsTask(filepath):
    if not os.path.isfile(filepath):
        print(filepath + " doesn't exists.")
        return
    data = sio.loadmat(filepath, struct_as_record=False)
    trainSize = data['labelTrain'].flatten().tolist().count(1)
    validSize = data['labelVal'].flatten().tolist().count(1)
    testSize = data['labelTest'].flatten().tolist().count(1)
    total = trainSize + validSize + testSize
    total2 = len(data['labelTrain'].flatten().tolist())
    print('{:11s}: {:8d} {:6.2f}%'.format('totalSize', total, 100*total/total))
    print('{:11s}: {:8d} {:6.2f}%'.format('trainSize', trainSize, 100*trainSize/total))
    print('{:11s}: {:8d} {:6.2f}%'.format('validSize', validSize, 100*validSize/total))
    print('{:11s}: {:8d} {:6.2f}%'.format('testSize', testSize, 100*testSize/total))
    print('{:11s}: {:8d} {:6.2f}%'.format('total2Size', total2, 100*total2/total))

def countFilesTaskSerial(filepath):
    count = 0
    sessionRegex = '([0-9]){5}'
    directories = getDirList(args.input, sessionRegex)
    # directories = directories[:10]
    bar = MultiProgressBar(max_value=len(directories), boundary=True)
    for directory_idx, directory in enumerate(directories):
        recording_path = os.path.join(args.input, directory)
        filenames = json_read(os.path.join(recording_path, "frames.json"))
        bar.addSubProcess(directory_idx, len(filenames))
        for idx, filename in enumerate(filenames):
            image_path = os.path.join(recording_path, "frames", filename)
            image = PILImage.open(image_path)
            image = np.array(image.convert('RGB'))
            shape_np, isValid = find_face_dlib(image)
            bar.update(directory_idx, idx+1)
            if isValid:
                count += 1
    print(count)

def countFilesTaskParallel(directory, directory_idx, bar):
    recording_path = os.path.join(args.input, directory)
    filenames = json_read(os.path.join(recording_path, "frames.json"))
    count = 0
    bar.addSubProcess(directory_idx, len(filenames))
    for idx, filename in enumerate(filenames):
        image_path = os.path.join(recording_path, "frames", filename)
        image = PILImage.open(image_path)
        image = np.array(image.convert('RGB'))
        shape_np, isValid = find_face_dlib(image)
        bar.update(directory_idx, idx+1)
        if isValid:
            count += 1
    return count

def countValidTask(directory, directory_idx, bar):
    recording_path = os.path.join(args.input, directory)
    # print(recording_path)
    frames = json_read(os.path.join('/data/gc-data', directory, "frames.json"))
    
    # reference
    faceData1 = json_read(os.path.join("/data/gc-output-dlib", directory, "dlibFace.json"))
    count1 = faceData1["IsValid"].count(1)

    # test subject "/data/gc-data-meta-rc", "/data/old_data/gc-data-meta-rc", "/data/gc-rc-meta
    faceData2 = json_read(os.path.join("/data/gc-data-meta-rc", directory, "dlibFace.json"))
    count2 = faceData2["IsValid"].count(1)

    if count1 != count2:
        # print(faceData1["IsValid"], faceData2["IsValid"])
        match = np.asarray(faceData1["IsValid"]) - np.asarray(faceData2["IsValid"])
        # more frames in ref
        idx = np.argwhere(match > 0).flatten()
        for i in idx:
            from_file = os.path.join('/data/gc-data', directory, "frames", frames[i])
            to_file = os.path.join('/data/gc-data-extra/ref', directory, frames[i])
            # print(from_file + "-->" + to_file)
            preparePath(os.path.join('/data/gc-data-extra/ref', directory))
            shutil.copy(from_file, to_file)
        
        # more frames in test
        idx = np.argwhere(match < 0).flatten()
        for i in idx:
            from_file = os.path.join('/data/gc-data', directory, "frames", frames[i])
            to_file = os.path.join('/data/gc-data-extra/test', directory, frames[i])
            # print(from_file + "-->" + to_file)
            preparePath(os.path.join('/data/gc-data-extra/test', directory))
            shutil.copy(from_file, to_file)
            
    return directory, len(frames), count1, count2

def plotRotationHistogramTask(directory, directory_idx, bar):
    faceData = json_read(os.path.join(args.input, directory, "dlibFace.json"))
    Theta = faceData["Theta"]
    return Theta

def modelStatsTask(checkpoint_dirpath):
    filepath = os.path.join(checkpoint_dirpath, 'checkpoint.pth.tar')
    print(filepath)
    saved = torch.load(filepath)
    if saved:
        epoch = saved.get('epoch', None)
        # for backward compatibility
        val_rms_errors = saved.get('val_RMSErrors', saved.get('RMSErrors', None)) 
        test_rms_errors = saved.get('test_RMSErrors', None)
        train_rms_errors = saved.get('train_RMSErrors', None)
        best_rms_error = saved.get('best_RMSError', None)
        best_rms_errors = saved.get('best_RMSErrors', None)
        learning_rates = saved.get('learning_rates', None)

        print('')
        print('Epoch {epoch:5d} with RMSError {rms_error:.5f}'.format(epoch=epoch, rms_error=best_rms_error))
        print('')
        print('\'RMS_Errors\': {0},'.format(val_rms_errors))
        print('\'Best_RMS_Errors\': {0}'.format(best_rms_errors))
        print('')

def splitInfoTask(directory, directory_idx, progressbar):
    recDir = os.path.join(args.input, directory)
    info = json_read(os.path.join(recDir, 'info.json'))
    print(info)
    info["Dataset"] = getSplit()
    print(info)
    outDir = preparePath(os.path.join(args.output, directory))
    json_write(os.path.join(outDir, 'info.json'), info)

def syncTask(src):
    HOME = os.environ['HOME']
    USER = os.environ['USER']
    HOST = os.uname()[1]
    dst = f"{USER}@{HOST}:/nfs/deepstore/Reference/{USER}"
    cmd = f"rsync -e 'ssh -i ~/.ssh/{USER}_deepthink_rsa' -rlptz --exclude=.git --exclude=.vscode {src} {dst}/"
    p = subprocess.Popen(cmd, shell=True)
    return

def checkpointInfoTask(filepath):
    if not os.path.isfile(filepath):
        return None
    saved = torch.load(filepath)

    if saved:
        epoch = saved.get('epoch')
        # for backward compatibility
        val_rms_errors = saved.get('RMSErrors', saved.get('val_RMSErrors')) 
        test_rms_errors = saved.get('test_RMSErrors')
        train_rms_errors = saved.get('train_RMSErrors')
        best_rms_error = saved.get('best_RMSError')
        best_rms_errors = saved.get('best_RMSErrors')
        learning_rates = saved.get('learning_rates')
        print('\'RMS_Errors\': {0},'.format(val_rms_errors))
        print('\'Best_RMS_Errors\': {0}'.format(best_rms_errors))
    return 

# all tasks are handled here
if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
    parser.add_argument('--input', help="input directory path", default="./gc-data/")
    parser.add_argument('--output', help="output directory path", default="./gc-data-meta-rc")
    parser.add_argument('--metapath', help="metadata path", default="./gc-data-meta/")
    parser.add_argument('--task', help="task name: copyTask, ROIDetectionTask, ROIExtractionTask", default="")
    parser.add_argument('--rc', action='store_true', help="apply rotation correction", default=False)
    parser.add_argument('--apple', action='store_true', help="apply original apple face crops", default=False)
    parser.add_argument('--mirror', action='store_true', help="apply data mirroring", default=False)
    parser.add_argument('--portraitOnly', action='store_true', help="use portrait data only", default=False)
    parser.add_argument('--source_compare', action='store_true', help="compare against source", default=False)
    parser.add_argument('--reference', default="", help="reference .mat path")
    parser.add_argument('--ext', help="", nargs='+', default=[".jpg", ".jpeg", ".JPG", ".JPEG"])
    args = parser.parse_args()

    if args.task == "":
        print("================= Task Menu =================")
        print("demoTask", 0)
        print("plotErrorTask", 1)
        print("plotErrorHeatmapTask", 2)
        print("plotGazePointHeatmapTask", 3)
        print("plotErrorHistogramTask", 4)
        print("parseResultsTask", 5)

        task = input('Input:')
        if task == '0':
            args.task = "demoTask"
        elif task == '1':
            args.task = "plotErrorTask"
        elif task == '2':
            args.task = "plotErrorHeatmapTask"
        elif task == '3':
            args.task = "plotGazePointHeatmapTask"
        elif task == '4':
            args.task = "plotErrorHistogramTask"
        elif task == '5':
            args.task = "parseResultsTask"

    # pre-processing for the task
    if args.task == "noneTask":
        taskFunction = noneTask
        taskData = getFileList("./gc-data/", args.ext)
        dataLoader = ListLoader
    elif args.task == "cubeTask":
        taskFunction = cubeTask
        taskData = [0,1,3,4,6,7]
        dataLoader = ListLoader
    elif args.task == "copyTask":
        taskFunction = copyTask
        taskData = getFileList(args.input, args.ext)
        dataLoader = ListLoader
    ######### Data Preparation Tasks #########
    elif args.task == "prepareEyeCatcherTask":
        taskFunction = prepareEyeCatcherTask
        taskData = getCaptureSessionDirList(args.input)
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
    ######### Data Visualization Tasks #########
    elif args.task == "plotErrorTask":
        # from RMS_errors import All_RMS_Errors
        script_directory = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(script_directory, "metadata/all_rms_errors.json")
        All_RMS_Errors = json_read(json_path)
        taskData = All_RMS_Errors
        dataLoader = None
        taskFunction = plotErrorTask
    elif args.task == "plotErrorHeatmapTask":
        taskData = "best_results.json"
        dataLoader = None
        taskFunction = plotErrorHeatmapTask
    elif args.task == "plotGazePointHeatmapTask":
        taskData = "best_results.json"
        dataLoader = None
        taskFunction = plotGazePointHeatmapTask
    elif args.task == "plotErrorHistogramTask":
        taskData = "best_results.json"
        dataLoader = None
        taskFunction = plotErrorHistogramTask
    elif args.task == "plotRotationHistogramTask":
        sessionRegex = '([0-9]){5}'
        taskData = getDirList(args.input, sessionRegex)
        dataLoader = ListLoader
        taskFunction = plotRotationHistogramTask
    elif args.task == "modelStatsTask":
        taskData = args.input
        dataLoader = None
        taskFunction = modelStatsTask
    ######### Data Parsing Tasks #########
    elif args.task == "parseResultsTask":
        taskData = "best_results.json"
        dataLoader = None
        taskFunction = parseResultsTask
    ######### Demo Tasks #########
    elif args.task == "demoTask":
        sys.path.append(".")
        from iTrackerGUITool import live_demo
        taskData = 0
        dataLoader = None
        taskFunction = live_demo
    ######### Data Statistics Tasks #########
    elif args.task == "dataStatsTask":
        # e.g. "/data/gc-data-prepped-dlib/metadata.mat"
        taskData = args.input
        dataLoader = None
        taskFunction = dataStatsTask
    ######### Test Tasks #########
    elif args.task == "countFilesTaskSerial":
        taskData = args.input
        dataLoader = None
        taskFunction = countFilesTaskSerial
    elif args.task == "countFilesTaskParallel":
        sessionRegex = '([0-9]){5}'
        taskData = getDirList(args.input, sessionRegex)
        dataLoader = ListLoader
        taskFunction = countFilesTaskParallel
    elif args.task == "countValidTask":
        sessionRegex = '([0-9]){5}'
        taskData = getDirList(args.input, sessionRegex)
        dataLoader = ListLoader
        taskFunction = countValidTask
    elif args.task == "splitInfoTask":
        taskFunction = splitInfoTask
        taskData = getDirList(args.input, '([0-9]){5}')
        dataLoader = ListLoader
    elif args.task == "syncTask":
        taskFunction = syncTask
        taskData = args.input
        dataLoader = None
    elif args.task == "checkpointInfoTask":
        taskFunction = checkpointInfoTask
        taskData = args.input
        dataLoader = None

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
            'labelTrain': [],
            'labelVal': [],
            'labelTest': []
        }
        # Combine results from various workers
        for m in output:
            meta['labelRecNum'] += m['labelRecNum']
            meta['frameIndex'] += m['frameIndex']
            meta['labelDotXCam'] += m['labelDotXCam']
            meta['labelDotYCam'] += m['labelDotYCam']
            meta['labelFaceGrid'] += m['labelFaceGrid']
            meta['labelTrain'] += m['labelTrain']
            meta['labelVal'] += m['labelVal']
            meta['labelTest'] += m['labelTest']

        # Integrate
        meta['labelRecNum'] = np.stack(meta['labelRecNum'], axis=0).astype(np.int16)
        meta['frameIndex'] = np.stack(meta['frameIndex'], axis=0).astype(np.int32)
        meta['labelDotXCam'] = np.stack(meta['labelDotXCam'], axis=0)
        meta['labelDotYCam'] = np.stack(meta['labelDotYCam'], axis=0)
        meta['labelFaceGrid'] = np.stack(meta['labelFaceGrid'], axis=0).astype(np.uint8)
        # print(meta)
        compareTask(meta)
    elif args.task == "countFilesTaskParallel":
        # Combine results from various workers
        print(sum(output))
    elif args.task == "countValidTask":
        # Combine results from various workers
        # print(output)
        sum = 0
        valid_ref = 0
        valid_test = 0
        issues = []
        for item in output:
            sum += item[1] 
            valid_ref += item[2]
            valid_test += item[3]
            if item[2] != item[3]:
                issues.append(item)

        print(issues)
        print(sum, valid_ref, valid_test)
    elif args.task == "plotRotationHistogramTask":
        # Combine results from various workers
        x = [theta for theta_list in output for theta in theta_list]
        num_bins = 60
        # the histogram of the data
        n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5, density=True)
        plt.axis([-45, 45, None, None]) 
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Probability Density')
        plt.title('Histogram of Rotation')
        plt.savefig('process_results/MIT_plotRotationHistogram.png')
        plt.show()

        
        











