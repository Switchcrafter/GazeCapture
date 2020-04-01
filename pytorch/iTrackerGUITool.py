import sys
from datetime import datetime  # for timing

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image
from imutils import face_utils

from screeninfo import get_monitors

from ITrackerData import normalize_image_transform
from ITrackerModel import ITrackerModel
from cam2screen import cam2screen

from face_utilities import find_face_dlib,\
                           rc_landmarksToRects,\
                           rc_generate_face_eye_images,\
                           prepare_image_inputs, \
                           hogImage


class InferenceEngine:
    def __init__(self, mode, color_space):
        self.mode = mode
        self.color_space = color_space
        if self.mode == "torch":
            self.modelSession = ITrackerModel(self.color_space).to(device='cpu')
            saved = torch.load('best_checkpoint.pth.tar', map_location='cpu')
            self.modelSession.load_state_dict(saved['state_dict'])
            self.modelSession.eval()
        elif self.mode == "onnx":
            self.modelSession = onnxruntime.InferenceSession('itracker.onnx')

    def run_inference(self, normalize_image, image_face, image_eye_left, image_eye_right, face_grid):
        # compute output
        if self.mode == "torch":
            with torch.no_grad():
                output = self.modelSession(image_face, image_eye_left, image_eye_right, face_grid)
                gaze_prediction_np = output.numpy()[0]
        elif self.mode == "onnx":
            # compute output
            output = self.modelSession.run(None,
                                {"face": image_face.numpy(),
                                "eyesLeft": image_eye_left.numpy(),
                                "eyesRight": image_eye_right.numpy(),
                                "faceGrid": face_grid.numpy()})
            gaze_prediction_np = (output[0])[0]

        return gaze_prediction_np


MEAN_PATH = '.'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
GRID_SIZE = 25
FACE_GRID_SIZE = (GRID_SIZE, GRID_SIZE)

DEVICE_NAME = "Alienware 51m"


# targets on the screen
TARGETS = [(-10., -3.),
           (-10., -6.),
           (-10., -9.),
           (-10., -12.),
           (-10., -15.),
           (0., -3.),
           (0., -6.),
           (0., -9.),
           (0., -12.),
           (0., -15.),
           (10., -3.),
           (10., -6.),
           (10., -9.),
           (10., -12.),
           (10., -15.),
           ]

# various command-based actions

def live_demo():
    color_space = 'YCbCr'
    mode = 'rc'

    # initialize inference engine - torch or onnx
    inferenceEngine = InferenceEngine("torch", color_space)

    # get screen monitor and video capture stream
    monitor = get_monitors()[0]
    cap = cv2.VideoCapture(0)

    # transform to normalize images
    normalize_image = normalize_image_transform(image_size=IMAGE_SIZE, split='test', jitter=False, color_space=color_space)

    # ititial target - it will keep changing
    target = 0

    screenOffsetX = 0
    screenOffsetY = 0

    while True:
        # read a new frame
        _, webcam_image = cap.read()
        # create a display object
        display = np.zeros((monitor.height - screenOffsetY, monitor.width - screenOffsetX, 3), dtype=np.uint8)

        # find face landmarks/keypoints
        shape_np, isValid = find_face_dlib(webcam_image)
        if mode == "pc":
            webcam_image, anchor_indices = perspectiveCorrection(webcam_image, shape_np)
            shape_np, isValid = find_face_dlib(webcam_image)
        else:
            anchor_indices = range(68)

        # basic display
        live_image = webcam_image.copy()
        if isValid:
            draw_landmarks(live_image, shape_np, anchor_indices)
            # live_image = draw_landmarks2(live_image, shape_np, anchor_indices)
            # delaunay_correction(live_image, shape_np, delaunay_color=(255, 255, 255))
            # draw_delaunay(live_image, shape_np, delaunay_color=(255, 255, 255))
            # draw_outline(live_image, shape_np, color=(255, 255, 255))
        live_image = Image.fromarray(live_image)
        live_image = transforms.functional.hflip(live_image)
        live_image = transforms.functional.resize(live_image, (monitor.height, monitor.width), interpolation=2)
        live_image = transforms.functional.adjust_brightness(live_image, 0.4)
        live_image = np.asarray(live_image)
        generate_baseline_display_data(display, screenOffsetX, screenOffsetY, monitor, live_image)

         # do only for valid face objects
        if isValid:
            face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)
            face_image, left_eye_image, right_eye_image, face_grid, face_grid_image = rc_generate_face_eye_images(face_rect,
                                                                                                                left_eye_rect,
                                                                                                                right_eye_rect,
                                                                                                                webcam_image)

            # OpenCV BGR -> PIL RGB conversion
            image_eye_left, image_eye_right, image_face = prepare_image_inputs(face_image,
                                                                            left_eye_image,
                                                                            right_eye_image)

            # PIL RGB -> PIL YCBCr. Then Convert images into tensors
            imEyeL, imEyeR, imFace, imFaceGrid = prepare_image_tensors(color_space,
                                                                    image_face,
                                                                    image_eye_left,
                                                                    image_eye_right,
                                                                    face_grid,
                                                                    normalize_image)
            start_time = datetime.now()
            gaze_prediction_np = inferenceEngine.run_inference(normalize_image,
                                                                imFace,
                                                                imEyeL,
                                                                imEyeR,
                                                                imFaceGrid)
            time_elapsed = datetime.now() - start_time

            display = generate_display_data(display, face_grid_image, face_image, gaze_prediction_np, left_eye_image,
                                monitor, right_eye_image, time_elapsed, target)


        # show default or updated display object on the screen
        cv2.imshow("display", display)

        # keystroke detection
        k = cv2.waitKey(5) & 0xFF
        #d=100, g=103, m=109
        if k == 27: # ESC
            break
        if k == 32: # Space
            target = (target + 1) % len(TARGETS)
        # if k == 100: # d
        #     delauny = ~delauny
        # if k == 103: # g
        #     grid = ~grid
        # if k == 109: # m
        #     mask = ~mask
        # if k == 108: # l
        #     landmarks = ~landmarks

    cv2.destroyAllWindows()
    cap.release()

# main script here
def main():
    live_demo()


def generate_baseline_display_data(display, screenOffsetX, screenOffsetY, monitor, webcam_image):
    display = draw_overlay(display, screenOffsetX, screenOffsetY, webcam_image)
    # draw reference grid
    draw_reference_grid(display, monitor.height, monitor.width)
    return display

def draw_landmarks(im, shape_np, anchor_indices):
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    idx = 0
    for idx in range(len(shape_np)):
        (x,y) = shape_np[idx]
        if idx in anchor_indices:
            draw_text(im, x, y, str(idx), scale=0.3, fill=(255, 255, 255), thickness=1)
        cv2.circle(im, (x, y), 1, (255, 255, 255), -1)


# # Driver code to test above function
# a = -1.0
# b = 1.0
# c = 0.0
# x1 = 1.0
# y1 = 0.0

# x, y = mirrorImage(a, b, c, x1, y1);
# def mirrorImage( a, b, c, x1, y1):
#     temp = -2 * (a * x1 + b * y1 + c) /(a * a + b * b)
#     x = temp * a + x1
#     y = temp * b + y1
#     return (x, y)

def draw_landmarks2(im, shape_np, anchor_indices):
    im2 = cv2.flip(im, 1)
    shape2_np = shape_np.copy()
    # lp2 = shape_np
    h,w,c = im.shape
    for point in shape2_np:
        # print('before', point)
        point = [point[0], w-point[1]]
        # point = [w-point[0], point[1]]
        # print('after', point)

    draw_landmarks(im2, shape2_np, anchor_indices)
    draw_landmarks(im, shape_np, anchor_indices)
    im = cv2.add(im, im2)
    return im


def generate_display_data(display, face_grid_image, face_image, gaze_prediction_np, left_eye_image, monitor,
                          right_eye_image, time_elapsed, target):

    disp_offset_x, disp_offset_y = 40, 40
    tx, ty = (TARGETS[target])[0], (TARGETS[target])[1]
    # targetX/Y are target coordinates in screen coordinate system
    (targetX, targetY) = cam2screen(tx,
                                    ty,
                                    1,
                                    monitor.width,
                                    monitor.height,
                                    deviceName=DEVICE_NAME
                                )

    (predictionX, predictionY) = cam2screen(gaze_prediction_np[0],
                                            gaze_prediction_np[1],
                                            1,
                                            monitor.width,
                                            monitor.height,
                                            deviceName=DEVICE_NAME
                                        )

    input_images = np.concatenate((face_image,
                                   right_eye_image,
                                   left_eye_image,
                                   face_grid_image.resize((224, 224))),
                                  axis=0)

    hog_images = np.concatenate((hogImage(face_image),
                                hogImage(right_eye_image),
                                hogImage(left_eye_image)),
                                axis=0)

    # draw input images
    display = draw_overlay(display, monitor.width - 300, 0, input_images)
    # draw hog images
    display = draw_overlay_hog(display, monitor.width - 525, 0, hog_images)
    # Draw prediction
    display = draw_crosshair(display,
                             int(predictionX),  # Screen offset?
                             int(predictionY),
                             radius=25,
                             fill=(255, 0, 0),
                             width=3)
    # Draw target
    display = draw_circle(display,
                          int(targetX),
                          int(targetY),
                          radius=20,
                          fill=(0, 0, 255),
                          width=3)
    display = draw_circle(display,
                          int(targetX),
                          int(targetY),
                          radius=5,
                          fill=(0, 0, 255),
                          width=5)
    # elapsed time
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+20,
                        f'time elapsed {time_elapsed}',
                        fill=(255, 255, 255))

    # Camera coordinate system info
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+60,
                        'Camera coordinate-system',
                        fill=(255, 255, 255))
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+80,
                        f'Target    : ({tx:.4f},'
                        f' {ty:.4f})',
                        fill=(255, 255, 255))
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+100,
                        f'Prediction : ({gaze_prediction_np.item(0):.4f},'
                        f' {gaze_prediction_np.item(1):.4f})',
                        fill=(255, 255, 255))

    # Screen coordinate system info
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+140,
                        'Screen coordinates system',
                        fill=(255, 255, 255))
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+160,
                        f'Target    : ({targetX:.4f},'
                        f' {targetY:.4f})',
                        fill=(255, 255, 255))
    display = draw_text(display,
                        disp_offset_x+20,
                        disp_offset_y+180,
                        f'Prediction : ({predictionX:.4f},'
                        f' {predictionY:.4f})',
                        fill=(255, 255, 255))
    return display


def prepare_image_tensors(color_space, image_face, image_eye_left, image_eye_right, face_grid, normalize_image):
    # Convert to the desired color space
    image_face = image_face.convert(color_space)
    image_eye_left = image_eye_left.convert(color_space)
    image_eye_right = image_eye_right.convert(color_space)

    # normalize the image, results in tensors
    tensor_face = normalize_image(image_face)
    tensor_eye_left = normalize_image(image_eye_left)
    tensor_eye_right = normalize_image(image_eye_right)
    tensor_face_grid = torch.FloatTensor(face_grid)

    # convert the 3 dimensional array into a 4 dimensional array, making it a batch size of 1
    tensor_face.unsqueeze_(0)
    tensor_eye_left.unsqueeze_(0)
    tensor_eye_right.unsqueeze_(0)
    tensor_face_grid.unsqueeze_(0)

    # Convert the tensors into
    tensor_face = torch.autograd.Variable(tensor_face, requires_grad=False)
    tensor_eye_left = torch.autograd.Variable(tensor_eye_left, requires_grad=False)
    tensor_eye_right = torch.autograd.Variable(tensor_eye_right, requires_grad=False)
    tensor_face_grid = torch.autograd.Variable(tensor_face_grid, requires_grad=False)

    return tensor_face, tensor_eye_left, tensor_eye_right, tensor_face_grid



def perspectiveCorrection(im, shape_np):
    # print(shape_np)
    dst_pts = np.float32([[226, 217],
                        [226, 248],
                        [230, 279],
                        [236, 308],
                        [246, 337],
                        [260, 363],
                        [280, 384],
                        [304, 400],
                        [334, 405],
                        [365, 401],
                        [390, 386],
                        [410, 366],
                        [424, 342],
                        [434, 314],
                        [439, 284],
                        [443, 253],
                        [445, 221],
                        [237, 195],
                        [249, 174],
                        [273, 168],
                        [296, 172],
                        [319, 182],
                        [348, 184],
                        [371, 176],
                        [395, 172],
                        [419, 179],
                        [431, 198],
                        [334, 206],
                        [334, 227],
                        [334, 247],
                        [334, 269],
                        [306, 285],
                        [319, 288],
                        [333, 291],
                        [347, 288],
                        [361, 285],
                        [259, 213],
                        [272, 205],
                        [288, 206],
                        [303, 215],
                        [287, 219],
                        [271, 219],
                        [365, 215],
                        [379, 205],
                        [396, 206],
                        [410, 214],
                        [397, 219],
                        [380, 219],
                        [289, 327],
                        [306, 319],
                        [322, 313],
                        [333, 317],
                        [345, 314],
                        [361, 321],
                        [377, 329],
                        [361, 341],
                        [346, 346],
                        [332, 347],
                        [320, 345],
                        [305, 341],
                        [297, 328],
                        [321, 326],
                        [333, 328],
                        [346, 326],
                        [370, 329],
                        [345, 329],
                        [332, 330],
                        [321, 329]])

    # # # selected landmarks
    # homography_indices = [36,37,38,39,40,41, 42,43,44,45,46,47]
    # src_pts = np.float32([[shape_np[i][0],shape_np[i][1]] for i in homography_indices])
    # dst_pts = np.float32([[dst_pts[i][0],dst_pts[i][1]] for i in homography_indices])
    # all landmarks
    # src_pts = shape_np[::3]
    # dst_pts = dst_pts[::3]

    # homography_indices = [6, 10, 19, 24]
    # src_pts = np.float32([[shape_np[i][0],shape_np[i][1]] for i in homography_indices])
    # # dst_pts = np.float32([[dst_pts[i][0],dst_pts[i][1]] for i in homography_indices])
    # h,w = 640,480
    # dst_pts = np.float32([[0,h],[h,w],[0,0],[w,0]])
    # M = cv2.getPerspectiveTransform(src_pts, dst_pts)


    # least distortion : stable 4-point face square
    homography_indices = [6, 10, 19, 24]
    # #most aligned :
    # homography_indices = [6, 10, 19, 24, 27, 33, 51]
    # homography_indices = [6,10,19,24,39,42,29,33,51]
    # homography_indices = [6,10,19,24,27,28,29,30,33,8]
    src_pts = np.float32([[shape_np[i][0],shape_np[i][1]] for i in homography_indices])
    dst_pts = np.float32([[dst_pts[i][0],dst_pts[i][1]] for i in homography_indices])
    M, mask = cv2.findHomography(src_pts, dst_pts)

    h,w,c = im.shape
    # do perspective correction
    im2 = cv2.warpPerspective(im, M, (w, h))
    # im2 = cv2.warpPerspective(im, M, (w, h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # print(M)

    # # show perspective
    # # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # pts = np.float32([[240, 180], [240, 340], [400, 340], [400, 180]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # im2 = cv2.polylines(im,[np.int32(dst)],True, (255,0,0), 3, cv2.LINE_AA)

    return im2, homography_indices


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw delaunay triangles
def draw_delaunay(img, landmarks, delaunay_color=(255, 255, 255)):
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for x,y in landmarks:
        # keep the coordinates within the rectangle limits
        x = max(min(size[1]-1, x), 0)
        y = max(min(size[0]-1, y), 0)
        subdiv.insert((int(x), int(y)))

    # Draw delaunay triangles
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# # Calculate delanauy triangle
# def calculateDelaunayTriangles(rect, points):
#     # Create subdiv
#     subdiv = cv2.Subdiv2D(rect);

#     # Insert points into subdiv
#     for p in points:
#         subdiv.insert((p[0], p[1]));


#     # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
#     triangleList = subdiv.getTriangleList();

#     # Find the indices of triangles in the points array

#     delaunayTri = []

#     for t in triangleList:
#         pt = []
#         pt.append((t[0], t[1]))
#         pt.append((t[2], t[3]))
#         pt.append((t[4], t[5]))

#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])

#         if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
#             ind = []
#             for j in range(0, 3):
#                 for k in range(0, len(points)):
#                     if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
#                         ind.append(k)
#             if len(ind) == 3:
#                 delaunayTri.append((ind[0], ind[1], ind[2]))

#     return delaunayTri

# def constrainPoint(p, w, h) :
#     p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
#     return p;


# def delaunay_correction(img, landmarks, delaunay_color=(255, 255, 255)):
#     avg_landmarks = np.float32([[226, 217],
#                     [226, 248],
#                     [230, 279],
#                     [236, 308],
#                     [246, 337],
#                     [260, 363],
#                     [280, 384],
#                     [304, 400],
#                     [334, 405],
#                     [365, 401],
#                     [390, 386],
#                     [410, 366],
#                     [424, 342],
#                     [434, 314],
#                     [439, 284],
#                     [443, 253],
#                     [445, 221],
#                     [237, 195],
#                     [249, 174],
#                     [273, 168],
#                     [296, 172],
#                     [319, 182],
#                     [348, 184],
#                     [371, 176],
#                     [395, 172],
#                     [419, 179],
#                     [431, 198],
#                     [334, 206],
#                     [334, 227],
#                     [334, 247],
#                     [334, 269],
#                     [306, 285],
#                     [319, 288],
#                     [333, 291],
#                     [347, 288],
#                     [361, 285],
#                     [259, 213],
#                     [272, 205],
#                     [288, 206],
#                     [303, 215],
#                     [287, 219],
#                     [271, 219],
#                     [365, 215],
#                     [379, 205],
#                     [396, 206],
#                     [410, 214],
#                     [397, 219],
#                     [380, 219],
#                     [289, 327],
#                     [306, 319],
#                     [322, 313],
#                     [333, 317],
#                     [345, 314],
#                     [361, 321],
#                     [377, 329],
#                     [361, 341],
#                     [346, 346],
#                     [332, 347],
#                     [320, 345],
#                     [305, 341],
#                     [297, 328],
#                     [321, 326],
#                     [333, 328],
#                     [346, 326],
#                     [370, 329],
#                     [345, 329],
#                     [332, 330],
#                     [321, 329]])

#     # Delaunay triangulation
#     h,w,c = img.shape
#     rect = (0, 0, w, h);
#     dt = calculateDelaunayTriangles(rect, avg_landmarks);

#     # Output image
#     output = np.zeros((h,w,3), np.float32());

#     # Warp input images to average image landmarks
#     for i in range(0, len(imagesNorm)) :
#         img = np.zeros((h,w,3), np.float32());
#         # Transform triangles one by one
#         for j in range(0, len(dt)) :
#             tin = [];
#             tout = [];

#             for k in range(0, 3) :
#                 pIn = pointsNorm[i][dt[j][k]];
#                 pIn = constrainPoint(pIn, w, h);

#                 pOut = pointsAvg[dt[j][k]];
#                 pOut = constrainPoint(pOut, w, h);

#                 tin.append(pIn);
#                 tout.append(pOut);


#             warpTriangle(imagesNorm[i], img, tin, tout);


#         # Add image intensities for averaging
#         output = output + img;

#     return output



def draw_outline(img, landmarks, color=(255, 255, 255)):
    for key in face_utils.FACIAL_LANDMARKS_IDXS:
        # print(face_utils.FACIAL_LANDMARKS_IDXS[key])
        start, end = face_utils.FACIAL_LANDMARKS_IDXS[key]
        for idx in range(start+1, end):
            (x1,y1) = landmarks[idx-1]
            (x2,y2) = landmarks[idx]
            draw_line(img, (x1, y1), (x2, y2), color, 1)

# drawing helper methods
def draw_overlay(image, x_offset, y_offset, s_img):
    height = min(s_img.shape[0], image.shape[0] - y_offset)
    width = min(s_img.shape[1], image.shape[1] - x_offset)
    image[y_offset:y_offset + height, x_offset:x_offset + width] = s_img[0:height, 0:width, :]
    return image

def draw_overlay_hog(image, x_offset, y_offset, s_img):
    height = min(s_img.shape[0], image.shape[0] - y_offset)
    width = min(s_img.shape[1], image.shape[1] - x_offset)
    image[y_offset:y_offset + height, x_offset:x_offset + width, 2] = s_img[0:height, 0:width]
    return image

def draw_crosshair(image, centerX, centerY, radius=25, fill=(0, 0, 0), width=5):
    cv2.line(image,
             (centerX, centerY - radius),
             (centerX, centerY + radius),
             fill,
             width)
    cv2.line(image,
             (centerX - radius, centerY),
             (centerX + radius, centerY),
             fill,
             width)

    return image

def draw_reference_grid(image, monitor_height, monitor_width, fill=(16,16,16), width=1):

    # draw axes
    draw_line(image, (10, 0), (10, 200), (255,0,0), 2, "arrowed")
    draw_text(image, 10+20, 200, "Y", scale=0.5, fill=(255, 0, 0), thickness=2)

    draw_line(image, (0, 10), (200, 10), (0,255,0), 2, "arrowed")
    draw_text(image, 200, 10+20, "X", scale=0.5, fill=(0, 255, 0), thickness=2)

    # draw grid
    for x in range(0, monitor_width, int(monitor_width/25)):
        draw_line(image, (x, 0), (x, monitor_height), fill, width)

    for y in range(0, monitor_height, int(monitor_height/25)):
        draw_line(image, (0, y), (monitor_width, y), fill, width)

    return image

def get_random_target():
    x = random.randrange(-10, 10, 1)
    y = random.randrange(-3, -15, 1)
    return x,y


def draw_line(image, src, dst, fill=(0, 0, 0), width=5, type='normal'):
    if type == "normal":
        cv2.line(image, src, dst, fill, width)
    elif type == "arrowed":
        cv2.arrowedLine(image, src, dst, fill, width)
    return image

def draw_circle(image, centerX, centerY, radius=25, fill=(0, 0, 0), width=5):
    cv2.circle(image,
               (centerX, centerY),
               radius,
               fill,
               width)
    return image

def draw_text(image, x, y, string, scale=0.5, fill=(0, 0, 0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, string, (x, y), font, scale, fill, thickness, cv2.LINE_AA)
    return image

if __name__ == "__main__":
    main()
    print('')
    print('DONE')
    print('')
