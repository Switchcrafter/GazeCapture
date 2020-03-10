from datetime import datetime  # for timing

import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from screeninfo import get_monitors

from ITrackerData import normalize_image_transform
from ITrackerModel import ITrackerModel
from cam2screen import cam2screen

from face_utilities import find_face_dlib,\
                           landmarksToRects,\
                           generate_face_eye_images,\
                           generate_face_grid, \
                           prepare_image_inputs, \
                           hogImage

import onnxruntime


class InferenceEngine():
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

        # # basic display
        # live_image = Image.fromarray(webcam_image)
        # draw_landmarks(live_image, face_rect, shape_np)
        # live_image = transforms.functional.hflip(live_image)
        # live_image = transforms.functional.resize(live_image, (monitor.height, monitor.width), interpolation=2)
        # live_image = transforms.functional.adjust_brightness(live_image, 0.05)
        # live_image = np.asarray(live_image)
        # display = generate_baseline_display_data(display, screenOffsetX, screenOffsetY, monitor, live_image)

        # find face landmarks/keypoints
        shape_np, isValid = find_face_dlib(webcam_image)


        # basic display
        live_image = webcam_image.copy()
        if isValid:
            draw_landmarks(live_image, shape_np)
            # draw_delaunay(live_image, shape_np, delaunay_color=(255, 255, 255))
        live_image = Image.fromarray(live_image)
        live_image = transforms.functional.hflip(live_image)
        live_image = transforms.functional.resize(live_image, (monitor.height, monitor.width), interpolation=2)
        live_image = transforms.functional.adjust_brightness(live_image, 0.1)
        live_image = np.asarray(live_image)
        generate_baseline_display_data(display, screenOffsetX, screenOffsetY, monitor, live_image)

         # do only for valid face objects
        if isValid:
            try:
                # rotation correction
                webcam_image = perspectiveCorrection(webcam_image, shape_np)
                shape_np, isValid = find_face_dlib(webcam_image)

                # convert landmarks into bounding-box rectangles
                face_rect, left_eye_rect_relative, right_eye_rect_relative, isValid = landmarksToRects(shape_np, isValid)

                # crop to get face and eye images
                face_image, left_eye_image, right_eye_image = generate_face_eye_images(face_rect,
                                                                                    left_eye_rect_relative,
                                                                                    right_eye_rect_relative,
                                                                                    webcam_image)
                input_images = np.concatenate((face_image,
                                            right_eye_image,
                                            left_eye_image),
                                            axis=0)
                # draw input images
                draw_overlay(display, monitor.width - 324, 0, input_images)
            except:
                print("Unexpected error:", sys.exc_info()[0])

        # # do only for valid face objects
        # if isValid:
        #     try:
        #         # convert landmarks into bounding-box rectangles
        #         face_rect, left_eye_rect_relative, right_eye_rect_relative, isValid = landmarksToRects(shape_np, isValid)

        #         # crop to get face and eye images
        #         face_image, left_eye_image, right_eye_image = generate_face_eye_images(face_rect,
        #                                                                             left_eye_rect_relative,
        #                                                                             right_eye_rect_relative,
        #                                                                             webcam_image)
        #         # create a face grid
        #         face_grid_image, face_grid = generate_face_grid(face_rect, webcam_image)
        #         imEyeL, imEyeR, imFace = prepare_image_inputs(face_grid_image,
        #                                                     face_image,
        #                                                     left_eye_image,
        #                                                     right_eye_image)
        #         # convert images into tensors
        #         face_grid, imEyeL, imEyeR, imFace = prepare_image_tensors(color_space,
        #                                                                 face_grid,
        #                                                                 imEyeL,
        #                                                                 imEyeR,
        #                                                                 imFace,
        #                                                                 normalize_image)

        #         start_time = datetime.now()
        #         gaze_prediction_np = inferenceEngine.run_inference(normalize_image,
        #                                                             imFace,
        #                                                             imEyeL,
        #                                                             imEyeR,
        #                                                             face_grid)

        #         time_elapsed = datetime.now() - start_time

        #         face_image = Image.fromarray(face_image)
        #         face_image = transforms.functional.hflip(face_image)
        #         face_image = np.asarray(face_image)

        #         left_eye_image = Image.fromarray(left_eye_image)
        #         left_eye_image = transforms.functional.hflip(left_eye_image)
        #         left_eye_image = np.asarray(left_eye_image)

        #         right_eye_image = Image.fromarray(right_eye_image)
        #         right_eye_image = transforms.functional.hflip(right_eye_image)
        #         right_eye_image = np.asarray(right_eye_image)

        #         display = generate_display_data(display, face_grid_image, face_image, gaze_prediction_np, left_eye_image,
        #                                         monitor, right_eye_image, time_elapsed, target)
        #     except:
        #         print("Unexpected error:", sys.exc_info()[0])

        # show default or updated display object on the screen
        cv2.imshow("display", display)

        # keystroke detection
        k = cv2.waitKey(5) & 0xFF
        if k == 27: # ESC
            break
        if k == 32: # Space
            target = (target + 1) % len(TARGETS)

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

def draw_landmarks(im, shape_np):
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    idx = 0
    for (x, y) in shape_np:
        cv2.circle(im, (x, y), 1, (255, 0, 0), -1)
        draw_text(im, x, y, str(idx), scale=0.5, fill=(255, 255, 255), thickness=1)
        idx = idx + 1


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
                                   left_eye_image),
                                  axis=0)

    hog_images = np.concatenate((hogImage(face_image),
                                hogImage(right_eye_image),
                                hogImage(left_eye_image)),
                                axis=0)

    # draw input images
    display = draw_overlay(display, monitor.width - 324, 0, input_images)
    # draw hog images
    display = draw_overlay_hog(display, monitor.width - 550, 0, hog_images)
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


def prepare_image_tensors(color_space, face_grid, image_eye_left, image_eye_right, image_face, normalize_image):
    # Convert to the desired color space
    image_face = image_face.convert(color_space)
    image_eye_left = image_eye_left.convert(color_space)
    image_eye_right = image_eye_right.convert(color_space)

    # normalize the image, results in tensors
    image_face = normalize_image(image_face)
    image_eye_left = normalize_image(image_eye_left)
    image_eye_right = normalize_image(image_eye_right)
    face_grid = torch.FloatTensor(face_grid)

    # convert the 3 dimensional array into a 4 dimensional array, making it a batch size of 1
    image_face.unsqueeze_(0)
    image_eye_left.unsqueeze_(0)
    image_eye_right.unsqueeze_(0)
    face_grid.unsqueeze_(0)

    # Convert the tensors into
    image_face = torch.autograd.Variable(image_face, requires_grad=False)
    image_eye_left = torch.autograd.Variable(image_eye_left, requires_grad=False)
    image_eye_right = torch.autograd.Variable(image_eye_right, requires_grad=False)
    face_grid = torch.autograd.Variable(face_grid, requires_grad=False)

    return face_grid, image_eye_left, image_eye_right, image_face

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

    # # selected landmarks
    # homography_indices = [0,4,8,12,16,19,24,30]
    # src_pts = np.float32([[shape_np[i][0],shape_np[i][1]] for i in homography_indices])
    # dst_pts = np.float32([[dst_pts[i][0],dst_pts[i][1]] for i in homography_indices])
    # all landmarks
    src_pts = shape_np[::3]
    dst_pts = dst_pts[::3]

    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    M, mask = cv2.findHomography(src_pts, dst_pts)
    # matchesMask = mask.ravel().tolist()
    h,w,c = im.shape

    im2 = cv2.warpPerspective(im, M, (w, h))

    # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # im2 = cv2.polylines(im,[np.int32(dst)],True, (255,0,0), 3, cv2.LINE_AA)
    return im2


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
    for x in range(0, monitor_width, 100):
        draw_line(image, (x, 0), (x, monitor_height), fill, width)

    for y in range(0, monitor_height, 100):
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
