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
        recording_path = os.path.join(dataset_path, recording, 'frames')

        face_dict = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }

        left_eye_dict = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }

        right_eye_dict = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }

        for image_name in sorted(os.listdir(recording_path)):
            face_rect, left_eye_rect, right_eye_rect, isValid = \
                find_face_dlib(os.path.join(recording_path, image_name))
            face_dict['X'].append(face_rect['x'])
            face_dict['Y'].append(face_rect['y'])
            face_dict['W'].append(face_rect['w'])
            face_dict['H'].append(face_rect['h'])
            face_dict['isValid'].append(isValid)

            left_eye_dict['X'].append(left_eye_rect['x'])
            left_eye_dict['Y'].append(left_eye_rect['y'])
            left_eye_dict['W'].append(left_eye_rect['w'])
            left_eye_dict['H'].append(left_eye_rect['h'])
            left_eye_dict['isValid'].append(isValid)

            right_eye_dict['X'].append(right_eye_rect['x'])
            right_eye_dict['Y'].append(right_eye_rect['y'])
            right_eye_dict['W'].append(right_eye_rect['w'])
            right_eye_dict['H'].append(right_eye_rect['h'])
            right_eye_dict['isValid'].append(isValid)

        # do something with dictionaries


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
