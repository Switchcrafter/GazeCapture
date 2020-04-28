import json

import os
import re

from utility_functions.face_utilities import faceEyeRectsToFaceInfoDict, newFaceInfoDict, find_face_dlib, landmarksToRects
from PIL import Image as PILImage
import numpy as np


def findCaptureSessionDirs(path):
    reg = re.compile('([0-9]){5}')
    folders = [x for x in os.listdir(path) if reg.match(x)]

    return folders


def loadJsonData(filename):
    data = None

    with open(filename) as f:
        data = json.load(f)

    return data


data_directory = "gc-data"
output_directory = "gc-output-dlib"

directories = sorted(findCaptureSessionDirs(data_directory))
total_directories = len(directories)

print(f"Found {total_directories} directories")

for directory_idx, directory in enumerate(directories):
    print(f"Processing {directory_idx + 1}/{total_directories}")

    recording_path = os.path.join(data_directory, directory)
    output_path = os.path.join(output_directory, directory)
    filenames = loadJsonData(os.path.join(recording_path, "frames.json"))

    faceInfoDict = newFaceInfoDict()
    for idx, filename in enumerate(filenames):
        image_path = os.path.join(recording_path, "frames", filename)
        image = PILImage.open(image_path)
        image = np.array(image.convert('RGB'))
        shape_np, isValid = find_face_dlib(image)
        face_rect, left_eye_rect, right_eye_rect, isValid = landmarksToRects(shape_np, isValid)

        faceInfoDict, faceInfoIdx = faceEyeRectsToFaceInfoDict(faceInfoDict,
                                                               face_rect,
                                                               left_eye_rect,
                                                               right_eye_rect,
                                                               isValid)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, 'dlibFace.json'), "w") as write_file:
        json.dump(faceInfoDict["Face"], write_file)
    with open(os.path.join(output_path, 'dlibLeftEye.json'), "w") as write_file:
        json.dump(faceInfoDict["LeftEye"], write_file)
    with open(os.path.join(output_path, 'dlibRightEye.json'), "w") as write_file:
        json.dump(faceInfoDict["RightEye"], write_file)
