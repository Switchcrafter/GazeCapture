import argparse
import os
import json
import shutil

from utility_functions.cam2screen import screen2cam, isSupportedDevice
from utility_functions.face_utilities import faceEyeRectsToFaceInfoDict, newFaceInfoDict, find_face_dlib, \
    landmarksToRects, generate_face_grid_rect
from PIL import Image as PILImage  # Pillow
import numpy as np
import dateutil.parser
from utility_functions.Utilities import MultiProgressBar

# Example path is Surface_Pro_4/someuser/00000
def findCaptureSessionDirs(path):
    session_paths = []
    devices = os.listdir(path)
    
    for device in devices:
        if not os.path.isdir(os.path.join(path, device)):
            continue
        users = os.listdir(os.path.join(path, device))
        for user in users:
            sessions = sorted(os.listdir(os.path.join(path, device, user)), key=str)

            for session in sessions:
                session_paths.append(os.path.join(device, user, session))

    return session_paths

def findCapturesInSession(path):
    return [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "frames")) if f.endswith('.jpg')]

def loadJsonData(filename):
    with open(filename) as f:
        return json.load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
    parser.add_argument('--data_path',
                        help="Path to captured files.",
                        default=None)
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    data_directory = args.data_path

    if data_directory is None:
        os.error("Error: must specify --data_path, like /data/EyeCapture/200407")
        return

    directories = sorted(findCaptureSessionDirs(data_directory))
    total_directories = len(directories)

    # print(f"Found {total_directories} directories")

    multi_progress_bar = MultiProgressBar(max_value=total_directories, boundary=True)

    for directory_idx, directory in enumerate(directories):
        captures = sorted(findCapturesInSession(os.path.join(data_directory, directory)), key=str)
        total_captures = len(captures)

        info_data = loadJsonData(os.path.join(data_directory, directory, "info.json"))
        if not isSupportedDevice(info_data["DeviceName"]):
            # If the device is not supported in device_metrics_sku.json skip it
            # print('%s, %s, %s'%(directory_idx, directory, 'Unsupported SKU'))
            multi_progress_bar.addSubProcess(index=directory_idx, max_value=0)
            continue

        screen_data = loadJsonData(os.path.join(data_directory, directory, "screen.json"))

        # dotinfo.json - { "DotNum": [ 0, 0, ... ],
        #                  "XPts": [ 160, 160, ... ],
        #                  "YPts": [ 284, 284, ... ],
        #                  "XCam": [ 1.064, 1.064, ... ],
        #                  "YCam": [ -6.0055, -6.0055, ... ],
        #                  "Time": [ 0.205642, 0.288975, ... ] }
        #
        # PositionIndex == DotNum
        # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
        dotinfo = {
            "DotNum": [],
            "XPts": [],
            "YPts": [],
            "XCam": [],
            "YCam": [],
            "Confidence": [],
            "Time": []
        }

        output_path = os.path.join(data_directory, directory)

        faceInfoDict = newFaceInfoDict()

        # frames.json - ["00000.jpg","00001.jpg"]
        frames = []

        facegrid = {
            "X": [],
            "Y": [],
            "W": [],
            "H": [],
            "IsValid": []
        }

        if directory_idx % 10 < 8:
            dataset_split = "train"
        elif directory_idx % 10 < 9:
            dataset_split = "val"
        else:
            dataset_split = "test"

        # info.json - {"TotalFrames":99,"NumFaceDetections":97,"NumEyeDetections":56,"Dataset":"train","DeviceName":"iPhone 6"}
        info = {
            "TotalFrames": total_captures,
            "NumFaceDetections": 0,
            "NumEyeDetections": 0,
            "Dataset": dataset_split,
            "DeviceName": info_data["DeviceName"]
        }

        # screen.json - { "H": [ 568, 568, ... ], "W": [ 320, 320, ... ], "Orientation": [ 1, 1, ... ] }
        screen = {
            "H": [],
            "W": [],
            "Orientation": []
        }

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        multi_progress_bar.addSubProcess(index=directory_idx, max_value=total_captures)

        for capture_idx, capture in enumerate(captures):
            capture_json_path = os.path.join(data_directory, directory, "frames", capture + ".json")
            capture_jpg_path = os.path.join(data_directory, directory, "frames", capture + ".jpg")

            try:
                if os.path.isfile(capture_json_path) and os.path.isfile(capture_jpg_path):
                    capture_data = loadJsonData(capture_json_path)

                    capture_image = PILImage.open(capture_jpg_path)
                    capture_image_np = np.array(capture_image)  # dlib wants images in numpy array format

                    shape_np, isValid = find_face_dlib(capture_image_np)

                    info["NumFaceDetections"] = info["NumFaceDetections"] + 1

                    face_rect, left_eye_rect, right_eye_rect, isValid = landmarksToRects(shape_np, isValid)

                    # facegrid.json - { "X": [ 6, 6, ... ], "Y": [ 10, 10, ... ], "W": [ 13, 13, ... ], "H": [ 13, 13, ... ], "IsValid": [ 1, 1, ... ] }
                    if isValid:
                        faceGridX, faceGridY, faceGridW, faceGridH = generate_face_grid_rect(face_rect, capture_image.width,
                                                                                            capture_image.height)
                    else:
                        faceGridX = 0
                        faceGridY = 0
                        faceGridW = 0
                        faceGridH = 0

                    facegrid["X"].append(faceGridX)
                    facegrid["Y"].append(faceGridY)
                    facegrid["W"].append(faceGridW)
                    facegrid["H"].append(faceGridH)
                    facegrid["IsValid"].append(isValid)

                    faceInfoDict, faceInfoIdx = faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect,
                                                                        right_eye_rect, isValid)
                    info["NumEyeDetections"] = info["NumEyeDetections"] + 1

                    # screen.json - { "H": [ 568, 568, ... ], "W": [ 320, 320, ... ], "Orientation": [ 1, 1, ... ] }
                    screen["H"].append(screen_data['H'][capture_idx])
                    screen["W"].append(screen_data['W'][capture_idx])
                    screen["Orientation"].append(screen_data['Orientation'][capture_idx])

                    # dotinfo.json - { "DotNum": [ 0, 0, ... ],
                    #                  "XPts": [ 160, 160, ... ],
                    #                  "YPts": [ 284, 284, ... ],
                    #                  "XCam": [ 1.064, 1.064, ... ],
                    #                  "YCam": [ -6.0055, -6.0055, ... ],
                    #                  "Confidence": [ 59.3, 94.2, ... ],
                    #                  "Time": [ 0.205642, 0.288975, ... ] }
                    #
                    # PositionIndex == DotNum
                    # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
                    x_raw = capture_data["XRaw"]
                    y_raw = capture_data["YRaw"]
                    x_cam, y_cam = screen2cam(x_raw,
                                            y_raw,
                                            screen_data['Orientation'][capture_idx],
                                            screen_data["W"][capture_idx],
                                            screen_data["H"][capture_idx],
                                            info_data["DeviceName"])
                    confidence = capture_data["Confidence"]

                    dotinfo["DotNum"].append(capture_idx)
                    dotinfo["XPts"].append(x_raw)
                    dotinfo["YPts"].append(y_raw)
                    dotinfo["XCam"].append(x_cam)
                    dotinfo["YCam"].append(y_cam)
                    dotinfo["Confidence"].append(confidence)
                    dotinfo["Time"].append(0)  # TODO replace with timestamp as needed

                    frame_name = str(f"{capture}.jpg")
                    frames.append(frame_name)
                else:
                    print(f"Error file doesn't exists: {directory}/{capture}")
            except json.decoder.JSONDecodeError:
                print(f"Error processing file: {directory}/{capture}")

            multi_progress_bar.update(index=directory_idx, value=capture_idx+1)

        with open(os.path.join(output_path, 'frames.json'), "w") as write_file:
            json.dump(frames, write_file)
        with open(os.path.join(output_path, 'screen.json'), "w") as write_file:
            json.dump(screen, write_file)
        with open(os.path.join(output_path, 'info.json'), "w") as write_file:
            json.dump(info, write_file)
        with open(os.path.join(output_path, 'dotInfo.json'), "w") as write_file:
            json.dump(dotinfo, write_file)
        with open(os.path.join(output_path, 'faceGrid.json'), "w") as write_file:
            json.dump(facegrid, write_file)
        with open(os.path.join(output_path, 'dlibFace.json'), "w") as write_file:
            json.dump(faceInfoDict["Face"], write_file)
        with open(os.path.join(output_path, 'dlibLeftEye.json'), "w") as write_file:
            json.dump(faceInfoDict["LeftEye"], write_file)
        with open(os.path.join(output_path, 'dlibRightEye.json'), "w") as write_file:
            json.dump(faceInfoDict["RightEye"], write_file)

main()