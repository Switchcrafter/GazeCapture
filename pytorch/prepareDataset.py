import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import scipy.io as sio
from PIL import Image

from utility_functions.Utilities import MultiProgressBar

'''
Prepares the GazeCapture dataset for use with the pytorch code. Crops images, compiles JSONs into metadata.mat

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


def parse_arguments():
    parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
    parser.add_argument('--dataset_path', help="Path to data directories.",
                        default=None)
    parser.add_argument('--ignore_reference', default=False, action='store_true',
                        help="Set to true when parsing non-MIT data. Will ignore all reference metadata")
    parser.add_argument('--use_reference_split', default=False, action='store_true',
                        help="Set to true to to use the reference splits. Implicitly sets --ingnore_reference")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if args.dataset_path is None or not os.path.isdir(args.dataset_path):
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    reference_data_split = None
    if args.use_reference_split:
        reference_data_split = readJson('./reference_data_split.json')
        args.ignore_reference = True

    # list recordings
    # TODO: Modify to do a recursive search, finding any subdirectory which contains a file "dotInfo.json"
    #   Store the recording directory path for later use
    recordingDirs = []
    for (root, dirs, files) in os.walk(args.dataset_path):
        if (os.path.isfile(os.path.join(root, "dotInfo.json"))):
            recordingDirs.append(root)

    # recordings = os.listdir(args.dataset_path)
    recordings = np.array(recordingDirs, np.object)
    recordings = recordings[[os.path.isdir(r) for r in recordings]]
    recordings.sort()

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

    total_recordings = len(recordings)

    multi_progress_bar = MultiProgressBar(max_value=total_recordings, boundary=True)

    for recording_idx, recording in enumerate(recordings):

        print("Processing %s" % recording)

        appleFace = readJson(os.path.join(recording, 'dlibFace.json'))
        if appleFace is None:
            logError('Skipping: Could not read dlibFace for recording %s!' % recording)
            continue
        appleLeftEye = readJson(os.path.join(recording, 'dlibLeftEye.json'))
        if appleLeftEye is None:
            logError('Skipping: Could not read dlibLeftEye for recording %s!' % recording)
            continue
        appleRightEye = readJson(os.path.join(recording, 'dlibRightEye.json'))
        if appleRightEye is None:
            logError('Skipping: Could not read dlibRightEye for recording %s!' % recording)
            continue
        dotInfo = readJson(os.path.join(recording, 'dotInfo.json'))
        if dotInfo is None:
            logError('Skipping: Could not read dotInfo for recording %s!' % recording)
            continue
        faceGrid = readJson(os.path.join(recording, 'faceGrid.json'))
        if faceGrid is None:
            logError('Skipping: Could not read faceGrid for recording %s!' % recording)
            continue
        frames = readJson(os.path.join(recording, 'frames.json'))
        if frames is None:
            logError('Skipping: Could not read frames for recording %s!' % recording)
            continue
        info = readJson(os.path.join(recording, 'info.json'))
        if info is None:
            logError('Skipping: Could not read info for recording %s!' % recording)
            continue

        facePath = preparePath(os.path.join(recording, 'appleFace'))
        leftEyePath = preparePath(os.path.join(recording, 'appleLeftEye'))
        rightEyePath = preparePath(os.path.join(recording, 'appleRightEye'))

        # Preprocess
        allValid = np.logical_and(np.logical_and(appleFace['IsValid'], appleLeftEye['IsValid']),
                                  np.logical_and(appleRightEye['IsValid'], faceGrid['IsValid']))
        if not np.any(allValid):
            logError('Skipping: Invalid face or eyes for recording %s!' % recording)
            continue

        frames = np.array([re.match('(.+)\.jpg$', x).group(1) for x in frames])

        bboxFromJson = lambda data: np.stack((data['X'], data['Y'], data['W'], data['H']), axis=1).astype(int)
        faceBbox = bboxFromJson(appleFace) + [-1, -1, 1, 1]  # for compatibility with matlab code
        leftEyeBbox = bboxFromJson(appleLeftEye) + [0, -1, 0, 0]
        rightEyeBbox = bboxFromJson(appleRightEye) + [0, -1, 0, 0]
        leftEyeBbox[:, :2] += faceBbox[:, :2]  # relative to face
        rightEyeBbox[:, :2] += faceBbox[:, :2]
        faceGridBbox = bboxFromJson(faceGrid)

        total_frames = len(frames)

        multi_progress_bar.addSubProcess(index=recording_idx, max_value=total_frames)

        for frame_idx, frame in enumerate(frames):

            # Load image
            imgFile = os.path.join(recording, 'frames', '%s.jpg' % frame)
            if not os.path.isfile(imgFile):
                logError('Warning: Could not find image file %s!' % imgFile)
                continue

            img = Image.open(imgFile)
            if img is None:
                logError('Warning: Could not open image file %s!' % imgFile)
                continue

            img = np.array(img.convert('RGB'))

            # Crop images
            imFace = cropImage(img, faceBbox[frame_idx, :])
            imEyeL = cropImage(img, leftEyeBbox[frame_idx, :])
            imEyeR = cropImage(img, rightEyeBbox[frame_idx, :])

            # Save images
            Image.fromarray(imFace).save(os.path.join(facePath, '%s.jpg' % frame), quality=95)
            Image.fromarray(imEyeL).save(os.path.join(leftEyePath, '%s.jpg' % frame), quality=95)
            Image.fromarray(imEyeR).save(os.path.join(rightEyePath, '%s.jpg' % frame), quality=95)

            # Collect metadata
            meta['labelRecNum'] += [recording]
            meta['frameIndex'] += [frame]
            meta['labelDotXCam'] += [dotInfo['XCam'][frame_idx]]
            meta['labelDotYCam'] += [dotInfo['YCam'][frame_idx]]
            meta['labelFaceGrid'] += [faceGridBbox[frame_idx, :]]

            split = info["Dataset"]
            meta['labelTrain'] += [split == "train"]
            meta['labelVal'] += [split == "val"]
            meta['labelTest'] += [split == "test"]

            multi_progress_bar.update(index=recording_idx, value=frame_idx+1)

    # Integrate
    meta['labelRecNum'] = np.stack(meta['labelRecNum'], axis=0)
    meta['frameIndex'] = np.stack(meta['frameIndex'], axis=0)
    meta['labelDotXCam'] = np.stack(meta['labelDotXCam'], axis=0)
    meta['labelDotYCam'] = np.stack(meta['labelDotYCam'], axis=0)
    meta['labelFaceGrid'] = np.stack(meta['labelFaceGrid'], axis=0).astype(np.uint8)

    if not args.ignore_reference:
        # Load reference metadata
        print('Will compare to the reference GitHub dataset metadata.mat...')
        reference = sio.loadmat('metadata/reference_metadata.mat', struct_as_record=False)
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
    metaFile = os.path.join(args.dataset_path, 'metadata.mat')
    print('Writing out the metadata.mat to %s...' % metaFile)
    sio.savemat(metaFile, meta)

    # Statistics
    print('======================\n\tSummary\n======================')
    print('Total added %d frames from %d recordings.' % (len(meta['frameIndex']), len(np.unique(meta['labelRecNum']))))

    if not args.ignore_reference:
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

    # Output statistics for frames

    count_train = meta['labelTrain'].count(1)
    count_val = meta['labelVal'].count(1)
    count_test = meta['labelTest'].count(1)
    count_total = len(meta['frameIndex'])

    print("")
    print(f"Train      {count_train:10d} frames - {(count_train / count_total) * 100:6.2f}%")
    print(f"Validation {count_val:10d} frames - {(count_val / count_total) * 100:6.2f}%")
    print(f"Test       {count_test:10d} frames - {(count_test / count_total) * 100:6.2f}%")
    print("")


    # import pdb; pdb.set_trace()
    input("Press Enter to continue...")


def readJson(filename):
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


def preparePath(path, clear=False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
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


def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
    res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]

    return res


if __name__ == "__main__":
    main()
