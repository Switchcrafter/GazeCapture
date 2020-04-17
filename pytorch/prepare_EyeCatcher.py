import argparse
import os
import json
import shutil

from cam2screen import screen2cam
from face_utilities import faceEyeRectsToFaceInfoDict, newFaceInfoDict, find_face_dlib, \
    landmarksToRects, generate_face_grid_rect
from PIL import Image as PILImage  # Pillow
import numpy as np
import dateutil.parser
from Utilities import SimpleProgressBar


# Example path is Surface_Pro_4/someuser/00000
def findCaptureSessionDirs(path):
    session_paths = []
    devices = os.listdir(path)

    for device in devices:
        users = os.listdir(os.path.join(path, device))
        for user in users:
            sessions = sorted(os.listdir(os.path.join(path, device, user)), key=str)

            for session in sessions:
                session_paths.append(os.path.join(device, user, session))

    return session_paths


def findCapturesInSession(path):
    files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "frames")) if f.endswith('.json') and not f == "session.json"]

    return files


def loadJsonData(filename):
    with open(filename) as f:
        data = json.load(f)

    return data


def getScreenOrientation(capture_data):
    orientation = 0

    # Camera Offset and Screen Orientation compensation
    # if capture_data['NativeOrientation'] == "Landscape":
    if capture_data['Orientation'] == "Landscape":
        # Camera above screen
        # - Landscape on Surface devices
        orientation = 1
    elif capture_data['Orientation'] == "LandscapeFlipped":
        # Camera below screen
        # - Landscape inverted on Surface devices
        orientation = 2
    elif capture_data['Orientation'] == "PortraitFlipped":
        # Camera left of screen
        # - Portrait with camera on left on Surface devices
        orientation = 3
    elif capture_data['Orientation'] == "Portrait":
        # Camera right of screen
        # - Portrait with camera on right on Surface devices
        orientation = 4
    # if capture_data['NativeOrientation'] == "Portrait":
    #     if capture_data['CurrentOrientation'] == "Portrait":
    #         # Camera above screen
    #         # - Portrait on iOS devices
    #         orientation = 1
    #     elif capture_data['CurrentOrientation'] == "PortraitFlipped":
    #         # Camera below screen
    #         # - Portrait Inverted on iOS devices
    #         orientation = 2
    #     elif capture_data['CurrentOrientation'] == "Landscape":
    #         # Camera left of screen
    #         # - Landscape home button on right on iOS devices
    #         orientation = 3
    #     elif capture_data['CurrentOrientation'] == "LandscapeFlipped":
    #         # Camera right of screen
    #         # - Landscape home button on left on iOS devices
    #         orientation = 4

    return orientation


def getCaptureTimeString(capture_data):
    sessiontime = dateutil.parser.parse(capture_data["SessionTimestamp"])
    currenttime = dateutil.parser.parse(capture_data["Timestamp"])
    timedelta = sessiontime - currenttime
    return str(timedelta.total_seconds())


def parse_arguments():
    parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
    parser.add_argument('--data_path',
                        help="Path to captured files.",
                        default=None)
    parser.add_argument('--output_path',
                        default=None,
                        help="Where to write the output.")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    data_directory = args.data_path
    output_directory = args.output_path

    if data_directory is None:
        os.error("Error: must specify --data_dir")
        return

    if output_directory is None:
        os.error("Error: must specify --output_dir")
        return

    directories = sorted(findCaptureSessionDirs(data_directory))
    total_directories = len(directories)

    print(f"Found {total_directories} directories")

    for directory_idx, directory in enumerate(directories):
        print(f"Processing {directory_idx + 1}/{total_directories} - {directory}")

        captures = sorted(findCapturesInSession(os.path.join(data_directory, directory)), key=str)
        total_captures = len(captures)

        deviceMetrics_data = loadJsonData(os.path.join(data_directory, directory, "deviceMetrics.json"))
        info_data = loadJsonData(os.path.join(data_directory, directory, "info.json"))
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
            "Time": []
        }

        output_path = os.path.join(output_directory, f"{directory_idx:05d}")
        output_frame_path = os.path.join(output_path, "frames")

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

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(output_frame_path):
            os.mkdir(output_frame_path)

        screen_orientation = getScreenOrientation(screen_data)

        capture_progress_bar = SimpleProgressBar(max_value=total_captures, label=f"{directory_idx:05d}")

        for capture_idx, capture in enumerate(captures):
            capture_progress_bar.update(capture_idx + 1)

            capture_json_path = os.path.join(data_directory, directory, "frames", capture + ".json")
            capture_jpg_path = os.path.join(data_directory, directory, "frames", capture + ".jpg")

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
                screen["H"].append(screen_data['H'])
                screen["W"].append(screen_data['W'])
                screen["Orientation"].append(screen_orientation)

                # dotinfo.json - { "DotNum": [ 0, 0, ... ],
                #                  "XPts": [ 160, 160, ... ],
                #                  "YPts": [ 284, 284, ... ],
                #                  "XCam": [ 1.064, 1.064, ... ],
                #                  "YCam": [ -6.0055, -6.0055, ... ],
                #                  "Time": [ 0.205642, 0.288975, ... ] }
                #
                # PositionIndex == DotNum
                # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
                x_raw = capture_data["XRaw"]
                y_raw = capture_data["YRaw"]
                x_cam, y_cam = screen2cam(x_raw,  # xScreenInPoints
                                          y_raw,  # yScreenInPoints
                                          screen_orientation,  # orientation,
                                          screen_data["W"],  # widthScreenInPoints
                                          screen_data["H"],  # heightScreenInPoints
                                          deviceName=info_data["DeviceName"])

                dotinfo["DotNum"].append(0)  # TODO replace with dot number as needed
                dotinfo["XPts"].append(x_raw)
                dotinfo["YPts"].append(y_raw)
                dotinfo["XCam"].append(x_cam)
                dotinfo["YCam"].append(y_cam)
                dotinfo["Time"].append(0)  # TODO replace with timestamp as needed

                # Convert image from PNG to JPG
                frame_name = str(f"{capture_idx:05d}.jpg")
                frames.append(frame_name)

                shutil.copyfile(capture_jpg_path, os.path.join(output_frame_path, frame_name))
            else:
                print(f"Error processing capture {capture}")

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

        print("")

    print("DONE")


main()
