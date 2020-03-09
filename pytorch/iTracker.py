import argparse
from datetime import datetime  # for timing

import cv2
import numpy as np
import torch

from screeninfo import get_monitors

from ITrackerData import normalize_image_transform
from ITrackerModel import ITrackerModel
from cam2screen import cam2screen

from face_utilities import find_face_dlib,\
                           landmarksToRects,\
                           generate_face_eye_images,\
                           generate_face_grid, \
                           prepare_image_inputs

import onnxruntime

MEAN_PATH = '.'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
GRID_SIZE = 25
FACE_GRID_SIZE = (GRID_SIZE, GRID_SIZE)

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


def main():
    args = parse_arguments()

    device_name = args.device_name

    if device_name is None:
        print(f"Invalid argument - must specify device_name: {args.device_name}")
        return

    color_space = args.color_space

    use_torch = False
    use_onnx = False

    if args.mode == "torch":
        use_torch = True
    elif args.mode == "onnx":
        use_onnx = True
    else:
        print(f"Invalid argument - must specify valid mode: {args.mode}")
        return

    if use_torch:
        model = initialize_torch(args.torch_model_path, args.device)
    elif use_onnx:
        session = initialize_onnx(args.onnx_model_path, args.device)

    monitor = get_monitors()[0]  # Assume only one monitor

    cap = cv2.VideoCapture(0)

    normalize_image = normalize_image_transform(image_size=IMAGE_SIZE, jitter=False, split='test')

    target = 0

    stimulusX, stimulusY = change_target(target, monitor, device_name)

    screenOffsetX = 0
    screenOffsetY = 100

    while True:
        _, webcam_image = cap.read()

        display = np.zeros((monitor.height - screenOffsetY, monitor.width - screenOffsetX, 3), dtype=np.uint8)

        shape_np, isValid = find_face_dlib(webcam_image)

        if isValid:
            face_rect, left_eye_rect_relative, right_eye_rect_relative, isValid = landmarksToRects(shape_np, isValid)

            display = generate_baseline_display_data(display,
                                                     screenOffsetX,
                                                     screenOffsetY,
                                                     webcam_image,
                                                     face_rect)

            face_image, left_eye_image, right_eye_image = generate_face_eye_images(face_rect,
                                                                                   left_eye_rect_relative,
                                                                                   right_eye_rect_relative,
                                                                                   webcam_image)

            face_grid_image, face_grid = generate_face_grid(face_rect, webcam_image)
            image_eye_left, image_eye_right, image_face = prepare_image_inputs(face_grid_image,
                                                                               face_image,
                                                                               left_eye_image,
                                                                               right_eye_image)

            tensor_face, tensor_eye_left, tensor_eye_right, tensor_face_grid = prepare_image_tensors(color_space,
                                                                                                     image_face,
                                                                                                     image_eye_left,
                                                                                                     image_eye_right,
                                                                                                     face_grid,
                                                                                                     normalize_image,
                                                                                                     args.device)

            start_time = datetime.now()
            if use_torch:
                gaze_prediction_np = run_torch_inference(model,
                                                         tensor_face,
                                                         tensor_eye_left,
                                                         tensor_eye_right,
                                                         tensor_face_grid)
            elif use_onnx:
                gaze_prediction_np = run_onnx_inference(session,
                                                        tensor_face,
                                                        tensor_eye_left,
                                                        tensor_eye_right,
                                                        tensor_face_grid)

            time_elapsed = datetime.now() - start_time

            display = generate_display_data(display,
                                            face_image,
                                            left_eye_image,
                                            right_eye_image,
                                            gaze_prediction_np,
                                            monitor,
                                            stimulusX,
                                            stimulusY,
                                            time_elapsed,
                                            device_name)

        cv2.imshow("display", display)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:  # ESC
            break
        if k == 32:
            target = target + 1
            if target >= len(TARGETS):
                target = 0
            stimulusX, stimulusY = change_target(target, monitor)

    cv2.destroyAllWindows()
    cap.release()


def generate_baseline_display_data(display,
                                   screenOffsetX,
                                   screenOffsetY,
                                   webcam_image,
                                   face_rect):
    display = draw_overlay(display, screenOffsetX, screenOffsetY, webcam_image)
    # display = draw_text(display,
    #                     20,
    #                     20,
    #                     f'Face ({x:4d}, {y:4d}, {w:4d}, {h:4d})',
    #                     fill=(255, 255, 255))
    return display


def generate_display_data(display,
                          face_image,
                          left_eye_image,
                          right_eye_image,
                          gaze_prediction_np,
                          monitor,
                          stimulus_x,
                          stimulus_y,
                          time_elapsed,
                          device_name):
    (gazePredictionScreenPixelXFromCamera, gazePredictionScreenPixelYFromCamera) = cam2screen(
        gaze_prediction_np[0],
        gaze_prediction_np[1],
        1,
        monitor.width,
        monitor.height,
        deviceName=device_name
    )
    input_images = np.concatenate((face_image,
                                   right_eye_image,
                                   left_eye_image),
                                  axis=0)
    display = draw_overlay(display, monitor.width - 324, 0, input_images)
    display = draw_crosshair(display,
                             int(gazePredictionScreenPixelXFromCamera),  # Screen offset?
                             int(gazePredictionScreenPixelYFromCamera),
                             radius=25,
                             fill=(255, 0, 0),
                             width=3)
    display = draw_circle(display,
                          int(stimulus_x),
                          int(stimulus_y),
                          radius=20,
                          fill=(0, 0, 255),
                          width=3)
    display = draw_circle(display,
                          int(stimulus_x),
                          int(stimulus_y),
                          radius=5,
                          fill=(0, 0, 255),
                          width=5)
    display = draw_text(display,
                        20,
                        40,
                        f'time elapsed {time_elapsed}',
                        fill=(255, 255, 255))
    display = draw_text(display,
                        20,
                        60,
                        f'GazePrediction(cam) - ({gaze_prediction_np.item(0):.4f},'
                        f' {gaze_prediction_np.item(1):.4f})',
                        fill=(255, 255, 255))
    display = draw_text(display,
                        20,
                        80,
                        f'GazePrediction(screen) - ({gazePredictionScreenPixelXFromCamera:.4f},'
                        f' {gazePredictionScreenPixelYFromCamera:.4f})',
                        fill=(255, 255, 255))
    return display


def initialize_torch(path, device):
    model = ITrackerModel().to(device=device)
    saved = torch.load(path, map_location=device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model


def initialize_onnx(path, device):
    # if device == 'cuda':
    #     session = onnxruntime-gpu.InferenceSession(path)
    # elif device == 'cpu':
    session = onnxruntime.InferenceSession(path)

    return session


def run_torch_inference(model, image_face, image_eye_left, image_eye_right, face_grid):
    # compute output
    with torch.no_grad():
        output = model(image_face, image_eye_left, image_eye_right, face_grid)
        gaze_prediction_np = output.cpu().numpy()[0]
    return gaze_prediction_np


def run_onnx_inference(session, image_face, image_eye_left, image_eye_right, face_grid):
    # compute output
    output = session.run(None,
                         {"face": image_face.numpy(),
                          "eyesLeft": image_eye_left.numpy(),
                          "eyesRight": image_eye_right.numpy(),
                          "faceGrid": face_grid.numpy()})

    gaze_prediction_np = (output[0])[0]

    return gaze_prediction_np


def prepare_image_tensors(color_space, image_face, image_eye_left, image_eye_right, face_grid, normalize_image, device):
    # Convert to the desired color space
    image_face = image_face.convert(color_space)
    image_eye_left = image_eye_left.convert(color_space)
    image_eye_right = image_eye_right.convert(color_space)

    # normalize the image, results in tensors
    tensor_face = normalize_image(image_face).to(device)
    tensor_eye_left = normalize_image(image_eye_left).to(device)
    tensor_eye_right = normalize_image(image_eye_right).to(device)
    tensor_face_grid = torch.FloatTensor(face_grid).to(device)

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


def change_target(target, monitor, device_name):
    (stimulusX, stimulusY) = cam2screen(
        (TARGETS[target])[0],
        (TARGETS[target])[1],
        1,
        monitor.width,
        monitor.height,
        deviceName=device_name
    )

    return stimulusX, stimulusY


def draw_overlay(image, x_offset, y_offset, s_img):
    height = min(s_img.shape[0], image.shape[0] - y_offset)
    width = min(s_img.shape[1], image.shape[1] - x_offset)

    image[y_offset:y_offset + height, x_offset:x_offset + width] = s_img[0:height, 0:width, :]

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='iTracker realtime inference.')
    parser.add_argument('--mode', default='torch', help='Inference Engine - torch, onnx')
    parser.add_argument('--torch_model_path',
                        help="Path to torch model (best_checkpoint.pth.tar).",
                        default='best_checkpoint.pth.tar')
    parser.add_argument('--onnx_model_path',
                        help="Path to onnx model (itracker.onnx).",
                        default='itracker.onnx')
    parser.add_argument('--color_space',
                        default='YCbCr',
                        help='Model\'s color space - RGB, YCbCr, HSV, LAB')
    parser.add_argument('--device_name',
                        default=None,
                        help='from device_metrics.json - Alienware 51m, Surface Pro 6, etc.')
    parser.add_argument('--device', default='cpu', help='Select either cpu or cuda')
    args = parser.parse_args()
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    main()
    print('')
    print('DONE')
    print('')
