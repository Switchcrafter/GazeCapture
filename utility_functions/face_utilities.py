import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from PIL import Image
from skimage import exposure
from skimage import feature

import os

file_dir_path = os.path.dirname(os.path.realpath(__file__))
landmarks_path = os.path.join(file_dir_path, '../metadata/shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_path)

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
GRID_SIZE = 25
FACE_GRID_SIZE = (GRID_SIZE, GRID_SIZE)


def find_face_dlib(image):
    isValid = 0
    shape_np = None

    face_rectangles = detector(image, 0)

    if len(face_rectangles) == 1:
        isValid = 1
        rect = face_rectangles[0]

        shape = predictor(image, rect)
        shape_np = face_utils.shape_to_np(shape)

    return shape_np, isValid


def landmarksToRects(shape_np, isValid):
    face_rect = (0, 0, 0, 0)
    left_eye_rect_relative = (0, 0, 0, 0)
    right_eye_rect_relative = (0, 0, 0, 0)

    if isValid:
        (leftEyeLandmarksStart, leftEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rightEyeLandmarksStart, rightEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        left_eye_shape_np = shape_np[leftEyeLandmarksStart:leftEyeLandmarksEnd]
        right_eye_shape_np = shape_np[rightEyeLandmarksStart:rightEyeLandmarksEnd]

        face_rect = cv2.boundingRect(shape_np)
        left_eye_rect = cv2.boundingRect(left_eye_shape_np)
        right_eye_rect = cv2.boundingRect(right_eye_shape_np)

        isValid = check_negative_coordinates(face_rect) and \
                  check_negative_coordinates(left_eye_rect) and \
                  check_negative_coordinates(right_eye_rect)

        left_eye_rect_relative = getEyeRectRelative(face_rect, left_eye_rect)
        right_eye_rect_relative = getEyeRectRelative(face_rect, right_eye_rect)

    return face_rect, left_eye_rect_relative, right_eye_rect_relative, isValid


def check_negative_coordinates(tup):
    isValid = True
    for idx in range(0, len(tup)):
        if tup[idx] < 0:
            isValid = False

    return isValid


def getFaceBox(faceDict, idx):
    x = faceDict['X'][idx]
    y = faceDict['Y'][idx]
    w = faceDict['W'][idx]
    h = faceDict['H'][idx]

    shape = [(x, y), (x + w, y + h)]

    return shape


def getEyeBox(faceDict, eyeDict, idx):
    faceX = faceDict['X'][idx]
    faceY = faceDict['Y'][idx]

    eyeX = eyeDict['X'][idx]
    eyeY = eyeDict['Y'][idx]
    eyeW = eyeDict['W'][idx]
    eyeH = eyeDict['H'][idx]

    # the eye box is referenced from the TL of the face box
    shape = [(faceX + eyeX, faceY + eyeY), (faceX + eyeX + eyeW, faceY + eyeY + eyeH)]

    return shape


def drawEyeBox(draw, eyeBox, offset, color):
    draw.rectangle(eyeBox, outline=color)

    tl = eyeBox[0]  # TL
    br = eyeBox[1]  # BR
    x_center = (br[0] - tl[0]) / 2 + tl[0]
    y_center = (br[1] - tl[1]) / 2 + tl[1]

    tinyEyeBox = [(x_center - offset, y_center - offset),
                  (x_center + offset, y_center + offset)]
    draw.rectangle(tinyEyeBox, outline=color)


def invertBLtoTL(box, height):
    # the boxes need oriended to the TL instead of the BR, so flip them along Y axis
    return [(box[0][0], height - box[0][1]), (box[1][0], height - box[1][1])]


def drawBoundingBoxes(draw, faceInfoDict, frameImageSize, idx, inverted=False, offset=0):
    faceDict = faceInfoDict["Face"]
    leftEyeDict = faceInfoDict["LeftEye"]
    rightEyeDict = faceInfoDict["RightEye"]
    color = faceInfoDict["Color"]

    height = frameImageSize[1]

    faceBox = getFaceBox(faceDict, idx)
    leftEyeBox = getEyeBox(faceDict, leftEyeDict, idx)
    rightEyeBox = getEyeBox(faceDict, rightEyeDict, idx)

    if inverted:
        faceBox = invertBLtoTL(faceBox, height)
        leftEyeBox = invertBLtoTL(leftEyeBox, height)
        rightEyeBox = invertBLtoTL(rightEyeBox, height)

    if faceDict["IsValid"][idx] == 1:
        draw.rectangle(faceBox, outline=color)

    if leftEyeDict["IsValid"][idx] == 1:
        drawEyeBox(draw, leftEyeBox, offset, color)
    if rightEyeDict["IsValid"][idx] == 1:
        drawEyeBox(draw, rightEyeBox, offset, color)


def getEyeRectRelative(face_rect, eye_rect):
    # find center of eye
    eye_center = (eye_rect[0] + int(eye_rect[2] / 2), eye_rect[1] + int(eye_rect[3] / 2))

    # eye box is 3/10 of the face width
    eye_side = int(3 * face_rect[2] / 10)
    face_top_left = (face_rect[0], face_rect[1])

    # eye positions are face Top Left relative. ie: eye is at (120, 120) absolute, but stored (20,20)
    # for a face located at (100,100)

    # take eye center & expand to a square with sides eye_side
    eye_tl = (eye_center[0] - int(eye_side / 2), eye_center[1] - int(eye_side / 2))

    # adjust coordinates to be relative to TL of face, converting the lower right to width and height
    eye_rect_relative = (eye_tl[0] - face_top_left[0], eye_tl[1] - face_top_left[1], eye_side, eye_side)

    return eye_rect_relative


def getRect(data):
    # get the parameter of the small rectangle
    center, size, angle = data[0], data[1], data[2]

    # The function minAreaRect seems to give angles ranging in (-90, 0].
    # This is based on the long edge of the rectangle
    if angle < -45:
        angle = 90 + angle
        size = (size[1], size[0])

    return int(center[0]), int(center[1]), int(size[0]), int(size[1]), int(angle)

def calibrate(data):
    # get the parameter of the small rectangle
    center, size, angle = data[0], data[1], data[2]

    # The function minAreaRect seems to give angles ranging in (-90, 0].
    # This is based on the long edge of the rectangle
    if angle < -45:
        angle = 90 + angle
        size = (size[1], size[0])

    return int(center[0]), int(center[1]), int(size[0]), int(size[1]), int(angle)

def makeSquare(size):
    length = max(size[0], size[1])
    return (length, length)

def getSquareBoundingRect(shape_vector):
    boundingRect = cv2.minAreaRect(shape_vector)
    return boundingRect[0], makeSquare(boundingRect[1]), boundingRect[2]

def rc_landmarksToRects(shape_np, isValid):
    face_rect = (0, 0, 0, 0, 0)
    left_eye_rect = (0, 0, 0, 0, 0)
    right_eye_rect = (0, 0, 0, 0, 0)

    if isValid:
        (leftEyeLandmarksStart, leftEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rightEyeLandmarksStart, rightEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        left_eye_shape_np = shape_np[leftEyeLandmarksStart:leftEyeLandmarksEnd]
        right_eye_shape_np = shape_np[rightEyeLandmarksStart:rightEyeLandmarksEnd]

        # face_rect = getRect(cv2.minAreaRect(shape_np))
        # left_eye_rect = getRect(cv2.minAreaRect(left_eye_shape_np))
        # right_eye_rect = getRect(cv2.minAreaRect(right_eye_shape_np))

        face_rect = getSquareBoundingRect(shape_np)
        left_eye_rect = cv2.minAreaRect(left_eye_shape_np)
        right_eye_rect = cv2.minAreaRect(right_eye_shape_np)

        face_rect = calibrate(face_rect)
        left_eye_rect = calibrate(left_eye_rect)
        right_eye_rect = calibrate(right_eye_rect)

        # ToDo enable negative coordinate check. Last value is theta which can be negative.
        isValid = check_negative_coordinates(face_rect[:-1]) and \
                  check_negative_coordinates(left_eye_rect[:-1]) and \
                  check_negative_coordinates(right_eye_rect[:-1])

    return face_rect, left_eye_rect, right_eye_rect, isValid


def rc_faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect, right_eye_rect, isValid):
    face_dict = faceInfoDict["Face"]
    left_eye_dict = faceInfoDict["LeftEye"]
    right_eye_dict = faceInfoDict["RightEye"]

    face_dict['X'].append(face_rect[0])
    face_dict['Y'].append(face_rect[1])
    face_dict['W'].append(face_rect[2])
    face_dict['H'].append(face_rect[3])
    face_dict['Theta'].append(face_rect[4])
    face_dict['IsValid'].append(isValid)

    left_eye_dict['X'].append(left_eye_rect[0])
    left_eye_dict['Y'].append(left_eye_rect[1])
    left_eye_dict['W'].append(left_eye_rect[2])
    left_eye_dict['H'].append(left_eye_rect[3])
    left_eye_dict['Theta'].append(left_eye_rect[4])
    left_eye_dict['IsValid'].append(isValid)

    right_eye_dict['X'].append(right_eye_rect[0])
    right_eye_dict['Y'].append(right_eye_rect[1])
    right_eye_dict['W'].append(right_eye_rect[2])
    right_eye_dict['H'].append(right_eye_rect[3])
    right_eye_dict['Theta'].append(right_eye_rect[4])
    right_eye_dict['IsValid'].append(isValid)

    idx = len(face_dict['X']) - 1

    return faceInfoDict, idx


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = (rect[0], rect[1]), (rect[2], rect[3]), rect[4]

    center, size = tuple(map(int, center)), tuple(map(int, size))
    # get a square crop of the detected region with 10px padding
    # size = (max(size) + 10, max(size) + 10)
    
    # get a square crop of the detected region
    size = makeSquare(size)

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop


def rc_generate_face_eye_images(face_rect, left_eye_rect, right_eye_rect, webcam_image):
    face_image = crop_rect(webcam_image.copy(), face_rect)
    face_image = imutils.resize(face_image, width=IMAGE_WIDTH)

    left_eye_image = crop_rect(webcam_image.copy(), left_eye_rect)
    left_eye_image = imutils.resize(left_eye_image, width=IMAGE_WIDTH)

    right_eye_image = crop_rect(webcam_image.copy(), right_eye_rect)
    right_eye_image = imutils.resize(right_eye_image, width=IMAGE_WIDTH)

    face_grid, face_grid_image = generate_grid(face_rect, webcam_image.copy())

    return face_image, left_eye_image, right_eye_image, face_grid, face_grid_image


def grid_generate_face_eye_images(face_rect, left_eye_rect, right_eye_rect, webcam_image):
    face_image = crop_rect(webcam_image.copy(), face_rect)
    face_image = imutils.resize(face_image, width=IMAGE_WIDTH)

    left_eye_image = crop_rect(webcam_image.copy(), left_eye_rect)
    left_eye_image = imutils.resize(left_eye_image, width=IMAGE_WIDTH)

    right_eye_image = crop_rect(webcam_image.copy(), right_eye_rect)
    right_eye_image = imutils.resize(right_eye_image, width=IMAGE_WIDTH)

    face_grid_image = generate_grid2(face_rect, webcam_image.copy())
    face_grid_image = imutils.resize(face_grid_image, width=IMAGE_WIDTH)

    return face_image, left_eye_image, right_eye_image, face_grid_image


def getBox(face_rect):
    return ((face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), face_rect[4])


def generate_grid2(rect, webcam_image):
    im = np.zeros(webcam_image.shape, np.uint8)
    im[:] = (255, 255, 255)

    box = np.int0(cv2.boxPoints(getBox(rect)))
    im = cv2.drawContours(im, [box], 0, (0, 0, 0), -1)  # 2 for line, -1 for filled
    return im


def generate_grid(face_rect, im):
    box = np.int0((cv2.boxPoints(getBox(face_rect))))
    im = im * 0 + 255
    face_grid_image = cv2.drawContours(im, [box], 0, (0, 0, 0), -1)  # 2 for line, -1 for filled
    face_grid, _, _ = cv2.split(face_grid_image)
    face_grid = cv2.resize(face_grid, (GRID_SIZE, GRID_SIZE), cv2.INTER_AREA)
    face_grid_flat = face_grid.flatten()  # flatten from 2d (25, 25) to 1d (625)

    face_grid_stacked = np.stack((face_grid,) * 3, axis=-1)
    face_grid_image = Image.fromarray(face_grid_stacked).convert("RGB")

    return face_grid_flat, face_grid_image


def newFaceInfoDict(color="blue"):
    faceInfoDict = {
        "Face": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'Theta': [],
            'IsValid': []
        },
        "LeftEye": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'Theta': [],
            'IsValid': []
        },
        "RightEye": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'Theta': [],
            'IsValid': []
        },
        "Color": color,
    }

    return faceInfoDict


def faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect, right_eye_rect, isValid):
    face_dict = faceInfoDict["Face"]
    left_eye_dict = faceInfoDict["LeftEye"]
    right_eye_dict = faceInfoDict["RightEye"]

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

    idx = len(face_dict['X']) - 1

    return faceInfoDict, idx


def generate_face_eye_images(face_rect, left_eye_rect_relative, right_eye_rect_relative, webcam_image):
    face_image = webcam_image.copy()

    face_image = face_image[face_rect[1]:face_rect[1] + face_rect[3], face_rect[0]:face_rect[0] + face_rect[2]]
    face_image = imutils.resize(face_image, width=IMAGE_WIDTH)

    left_eye_image = webcam_image.copy()
    left_eye_image = left_eye_image[face_rect[1] + left_eye_rect_relative[1]:face_rect[1] + left_eye_rect_relative[1] +
                                                                             left_eye_rect_relative[3],
                     face_rect[0] + left_eye_rect_relative[0]:face_rect[0] + left_eye_rect_relative[0] +
                                                              left_eye_rect_relative[2]]
    left_eye_image = imutils.resize(left_eye_image, width=IMAGE_WIDTH)

    right_eye_image = webcam_image.copy()
    right_eye_image = right_eye_image[
                      face_rect[1] + right_eye_rect_relative[1]:face_rect[1] + right_eye_rect_relative[1] +
                                                                right_eye_rect_relative[3],
                      face_rect[0] + right_eye_rect_relative[0]:face_rect[0] + right_eye_rect_relative[0] +
                                                                right_eye_rect_relative[2]]
    right_eye_image = imutils.resize(right_eye_image, width=IMAGE_WIDTH)

    return face_image, left_eye_image, right_eye_image


def generate_face_grid_rect(face_rect, image_width, image_height):
    faceGridX = int((face_rect[0] / image_width) * GRID_SIZE)
    faceGridY = int((face_rect[1] / image_height) * GRID_SIZE)
    faceGridW = int(((face_rect[0] + face_rect[2]) / image_width) * GRID_SIZE) - faceGridX
    faceGridH = int(((face_rect[1] + face_rect[3]) / image_height) * GRID_SIZE) - faceGridY

    return faceGridX, faceGridY, faceGridW, faceGridH


def generate_face_grid(face_rect, image_width, image_height):
    faceGridX, faceGridY, faceGridW, faceGridH = generate_face_grid_rect(face_rect, image_width, image_height)

    face_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for m in range(faceGridW):
        for n in range(faceGridH):
            x = min(GRID_SIZE - 1, faceGridX + m)
            y = min(GRID_SIZE - 1, faceGridY + n)
            face_grid[y, x] = 1
    face_grid_flat = face_grid.flatten()  # flatten from 2d (25, 25) to 1d (625)

    # generate an image suitable for display, not used by ML
    face_grid_inverted = (255 - (255 * face_grid))
    face_grid_stacked = np.stack((face_grid_inverted,) * 3, axis=-1)
    face_grid_image = Image.fromarray(face_grid_stacked).convert("RGB")

    return face_grid_flat, face_grid_image


def prepare_image_inputs(face_image, left_eye_image, right_eye_image):
    imFace = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), 'RGB')
    imEyeL = Image.fromarray(cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2RGB), 'RGB')
    imEyeR = Image.fromarray(cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2RGB), 'RGB')

    return imEyeL, imEyeR, imFace


def grid_prepare_image_inputs(face_image, left_eye_image, right_eye_image, face_grid_image):
    imFace = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), 'RGB')
    imEyeL = Image.fromarray(cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2RGB), 'RGB')
    imEyeR = Image.fromarray(cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2RGB), 'RGB')
    imFaceGrid = Image.fromarray(cv2.cvtColor(face_grid_image, cv2.COLOR_BGR2RGB), 'RGB')

    return imEyeL, imEyeR, imFace, imFaceGrid


def hogImage(image):
    H, hogImage = feature.hog(image,
                              orientations=8,
                              pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1),
                              visualize=True,
                              multichannel=True)
    hogImage = exposure.rescale_intensity(hogImage, in_range=(0, 10), out_range=(0, 255))

    return hogImage
