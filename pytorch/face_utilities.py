import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from PIL import Image
from skimage import exposure
from skimage import feature

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

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
        face_rect = cv2.boundingRect(shape_np)

    return face_rect, isValid


def getFaceBox(faceDict, idx):
    x = faceDict['X'][idx]
    y = faceDict['Y'][idx]
    w = faceDict['W'][idx]
    h = faceDict['H'][idx]

    shape = [(x, y), (x + w, y + h)]

    return shape


def invertBLtoTL(box, height):
    # the boxes need oriended to the TL instead of the BR, so flip them along Y axis
    return [(box[0][0], height - box[0][1]), (box[1][0], height - box[1][1])]


def drawBoundingBoxes(draw, faceInfoDict, frameImageSize, idx, inverted=False, offset=0):
    faceDict = faceInfoDict["Face"]
    color = faceInfoDict["Color"]

    height = frameImageSize[1]

    faceBox = getFaceBox(faceDict, idx)

    if inverted:
        faceBox = invertBLtoTL(faceBox, height)

    if faceDict["IsValid"][idx] == 1:
        draw.rectangle(faceBox, outline=color)


def newFaceInfoDict(color="blue"):
    faceInfoDict = {
        "Face": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'IsValid': []
        },
        "LeftEye": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'IsValid': []
        },
        "RightEye": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'IsValid': []
        },
        "Color": color,
    }

    return faceInfoDict


def faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, isValid):
    face_dict = faceInfoDict["Face"]

    face_dict['X'].append(face_rect[0])
    face_dict['Y'].append(face_rect[1])
    face_dict['W'].append(face_rect[2])
    face_dict['H'].append(face_rect[3])
    face_dict['IsValid'].append(isValid)

    idx = len(face_dict['X']) - 1

    return faceInfoDict, idx


def generate_face_eye_images(face_rect, webcam_image):
    face_image = webcam_image.copy()

    face_image = face_image[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2]]
    face_image = imutils.resize(face_image, width=IMAGE_WIDTH)

    return face_image


def generate_face_grid(face_rect, webcam_image):
    image_width = webcam_image.shape[1]
    image_height = webcam_image.shape[0]
    faceGridX = int((face_rect[0] / image_width) * GRID_SIZE)
    faceGridY = int((face_rect[1] / image_height) * GRID_SIZE)
    faceGridW = int(((face_rect[0] + face_rect[2]) / image_width) * GRID_SIZE) - faceGridX
    faceGridH = int(((face_rect[1] + face_rect[3]) / image_height) * GRID_SIZE) - faceGridY
    faceGridImage = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    face_grid = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.uint8)
    faceGridImage.fill(255)
    for m in range(faceGridW):
        for n in range(faceGridH):
            faceGridImage[faceGridY + n, faceGridX + m] = (0, 0, 0)
            face_grid[faceGridY + n, faceGridX + m] = 1
    face_grid = face_grid.flatten()  # flatten from 2d (25, 25) to 1d (625)

    return faceGridImage, face_grid


def prepare_image_inputs(face_grid_image, face_image):
    imFace = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), 'RGB')

    return imFace


def hogImage(image):
    (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(3, 3), transform_sqrt=True, block_norm="L1",
                                visualize=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")

    return hogImage