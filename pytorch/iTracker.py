from collections import OrderedDict
from datetime import datetime  # for timing

import cv2
import dlib
import imutils
import numpy as np
import torch
from PIL import Image
from imutils import face_utils
from screeninfo import get_monitors

from ITrackerData import NormalizeImage
from ITrackerModel import ITrackerModel

MEAN_PATH = '.'


def main():
    model = ITrackerModel().to(device='cpu')
    saved = torch.load('checkpoint.pth.tar', map_location='cpu')

    state = remove_module_from_state(saved)

    model.load_state_dict(state)
    model.eval()

    monitor = get_monitors()[0]

    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv2.VideoCapture(0)

    normalize_image = NormalizeImage()

    while True:
        _, image = cap.read()

        faceImage = None
        rightEyeImage = None
        leftEyeImage = None
        faceGrid = None
        isValid = False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rectangles = detector(gray, 0)

        # if face_rectangles length != 1, then not valid

        for (i, rect) in enumerate(face_rectangles):
            shape = predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)

            (leftEyeLandmarksStart, leftEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rightEyeLandmarksStart, rightEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            (x, y, w, h) = cv2.boundingRect(shape_np)

            if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                # print('Face (%4d, %4d, %4d, %4d)' % (x, y, w, h))
                isValid = True
                faceImage = image.copy()
                faceImage = faceImage[rect.tl_corner().y:rect.br_corner().y, rect.tl_corner().x:rect.br_corner().x]
                faceImage = imutils.resize(faceImage, width=225, inter=cv2.INTER_CUBIC)

                (x, y, w, h) = cv2.boundingRect(shape_np[leftEyeLandmarksStart:leftEyeLandmarksEnd])
                leftEyeImage = image.copy()
                leftEyeImage = leftEyeImage[y - 15:y + h + 15, x - 5:x + w + 5]
                leftEyeImage = imutils.resize(leftEyeImage, width=61, inter=cv2.INTER_CUBIC)

                (x, y, w, h) = cv2.boundingRect(shape_np[rightEyeLandmarksStart:rightEyeLandmarksEnd])
                rightEyeImage = image.copy()
                rightEyeImage = rightEyeImage[y - 15:y + h + 15, x - 5:x + w + 5]
                rightEyeImage = imutils.resize(rightEyeImage, width=61, inter=cv2.INTER_CUBIC)

                if rect.tl_corner().x < 0 or rect.tl_corner().y < 0:
                    isValid = False

                cv2.rectangle(image, (rect.tl_corner().x, rect.tl_corner().y), (rect.br_corner().x, rect.br_corner().y),
                              (255, 255, 00), 2)

                imageWidth = image.shape[1]
                imageHeight = image.shape[0]

                faceGridX = int((rect.tl_corner().x / imageWidth) * 25)
                faceGridY = int((rect.tl_corner().y / imageHeight) * 25)
                faceGridW = int((rect.br_corner().x / imageWidth) * 25) - faceGridX
                faceGridH = int((rect.br_corner().y / imageHeight) * 25) - faceGridY

                faceGridImage = np.zeros((25, 25, 1), dtype=np.uint8)
                faceGrid = np.zeros((25, 25, 1), dtype=np.uint8)
                faceGridImage.fill(255)
                for m in range(faceGridW):
                    for n in range(faceGridH):
                        faceGridImage[faceGridY + n, faceGridX + m] = 0
                        faceGrid[faceGridY + n, faceGridX + m] = 1

                # Draw on our image, all the found coordinate points (x,y)
                for (x, y) in shape_np:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            else:
                print('Face (%4d, %4d, %4d, %4d) Not Valid' % (x, y, w, h))

        cv2.imshow("display", np.zeros((monitor.height, monitor.width, 1), dtype=np.uint8))
        cv2.imshow("WebCam", image)
        if isValid:
            cv2.imshow("face", faceImage)
            cv2.imshow("rightEye", rightEyeImage)
            cv2.imshow("leftEye", leftEyeImage)
            cv2.imshow("faceGrid", faceGridImage)

            imFace = Image.fromarray(faceImage, 'RGB')
            imEyeL = Image.fromarray(leftEyeImage, 'RGB')
            imEyeR = Image.fromarray(rightEyeImage, 'RGB')

            imFace = normalize_image.face(image=imFace)
            imEyeL = normalize_image.eye_left(image=imEyeL)
            imEyeR = normalize_image.eye_right(image=imEyeR)
            faceGrid = torch.FloatTensor(faceGrid)

            # convert the 3 dimensional array into a 4 dimensional array, making it a batch size of 1
            imFace.unsqueeze_(0)
            imEyeL.unsqueeze_(0)
            imEyeR.unsqueeze_(0)
            faceGrid.unsqueeze_(0)

            imFace = torch.autograd.Variable(imFace, requires_grad=False)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad=False)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad=False)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad=False)

            # compute output
            with torch.no_grad():
                start_time = datetime.now()

                output = model(imFace, imEyeL, imEyeR, faceGrid)
                gazePredictionNp = output.numpy()[0]

                time_elapsed = datetime.now() - start_time
                print('GazePrediction - {} - time elapsed {}'.format(gazePredictionNp, time_elapsed))

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def remove_module_from_state(saved_state):
    # when using Cuda for training we use DataParallel. When using DataParallel, there is a
    # 'module.' added to the namespace of the item in the dictionary.
    # remove 'module.' from the front of the name to make it compatible with cpu only
    state = OrderedDict()

    for key, value in saved_state['state_dict'].items():
        state[key[7:]] = value.to(device='cpu')

    return state


if __name__ == "__main__":
    main()
    print('')
    print('DONE')
    print('')
