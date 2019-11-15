import json

import cv2
import os

import dlib
import numpy as np
from PIL import Image
from imutils import face_utils

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def main():
    dataset_path = "/data/gc-data"
    recordings = os.listdir(dataset_path)

    for recording in sorted(recordings):
        recording_path = os.path.join(dataset_path, recording)
        print('Processing recording %s' % recording_path)

        face_dict = {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'IsValid': []
        }

        left_eye_dict = {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'IsValid': []
        }

        right_eye_dict = {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'IsValid': []
        }

        for image_name in sorted(os.listdir(os.path.join(recording_path, 'frames'))):
            print(image_name)

            face_rect, left_eye_rect, right_eye_rect, isValid = \
                find_face_dlib(os.path.join(recording_path, 'frames', image_name))
            face_dict['X'].append(face_rect[0])
            face_dict['Y'].append(face_rect[1])
            face_dict['W'].append(face_rect[2])
            face_dict['H'].append(face_rect[3])
            face_dict['IsValid'].append(isValid)

            left_eye_dict['X'].append(left_eye_rect[0])
            left_eye_dict['Y'].append(left_eye_rect[1])
            left_eye_dict['W'].append(left_eye_rect[2])
            left_eye_dict['H'].append(left_eye_rect[3])
            left_eye_dict['IsValid'].append(isValid)

            right_eye_dict['X'].append(right_eye_rect[0])
            right_eye_dict['Y'].append(right_eye_rect[1])
            right_eye_dict['W'].append(right_eye_rect[2])
            right_eye_dict['H'].append(right_eye_rect[3])
            right_eye_dict['IsValid'].append(isValid)

        with open(os.path.join(recording_path, 'dlibFace.json'), "w") as write_file:
            json.dump(face_dict, write_file)
        with open(os.path.join(recording_path, 'dlibLeftEye.json'), "w") as write_file:
            json.dump(left_eye_dict, write_file)
        with open(os.path.join(recording_path, 'dlibRightEye.json'), "w") as write_file:
            json.dump(right_eye_dict, write_file)


def find_face_dlib(image_path):
    face_rect = (0, 0, 0, 0)
    left_eye_rect = (0, 0, 0, 0)
    right_eye_rect = (0, 0, 0, 0)
    isValid = 0

    image = Image.open(image_path)
    if image is None:
        print('Warning: Could not read image file %s!' % image_path)
    else:
        image = np.array(image.convert('RGB'))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rectangles = detector(gray, 0)

        if len(face_rectangles) == 1:
            isValid = 1
            rect = face_rectangles[0]

            shape = predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)

            (leftEyeLandmarksStart, leftEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rightEyeLandmarksStart, rightEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            face_rect = cv2.boundingRect(shape_np)
            left_eye_rect = cv2.boundingRect(shape_np[leftEyeLandmarksStart:leftEyeLandmarksEnd])
            right_eye_rect = cv2.boundingRect(shape_np[rightEyeLandmarksStart:rightEyeLandmarksEnd])

            # need to pad y by 15 and x by 5 for right and left eye

    return face_rect, left_eye_rect, right_eye_rect, isValid


main()
