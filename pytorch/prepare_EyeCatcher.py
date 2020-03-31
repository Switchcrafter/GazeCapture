import os
import json
from face_utilities import faceEyeRectsToFaceInfoDict, newFaceInfoDict, find_face_dlib, \
    landmarksToRects, generate_face_grid_rect
from PIL import Image as PILImage  # Pillow
import numpy as np
import dateutil.parser


def findCaptureSessionDirs(path):
    session_paths = []
    devices = os.listdir(path)

    for device in devices:
        sessions = os.listdir(os.path.join(path, device))
        for session in sessions:
            session_paths.append(os.path.join(device, session))

    return session_paths


def findCapturesInSession(path):
    files = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.json')]

    return files


def loadJsonData(filename):
    data = None

    with open(filename) as f:
        data = json.load(f)

    return data


def getScreenOrientation(capture_data):
    orientation = 0

    # Camera Offset and Screen Orientation compensation
    if capture_data['NativeOrientation'] == "Landscape":
        if capture_data['CurrentOrientation'] == "Landscape":
            # Camera above screen
            # - Landscape on Surface devices
            orientation = 1
        elif capture_data['CurrentOrientation'] == "LandscapeFlipped":
            # Camera below screen
            # - Landscape inverted on Surface devices
            orientation = 2
        elif capture_data['CurrentOrientation'] == "PortraitFlipped":
            # Camera left of screen
            # - Portrait with camera on left on Surface devices
            orientation = 3
        elif capture_data['CurrentOrientation'] == "Portrait":
            # Camera right of screen
            # - Portrait with camera on right on Surface devices
            orientation = 4
    if capture_data['NativeOrientation'] == "Portrait":
        if capture_data['CurrentOrientation'] == "Portrait":
            # Camera above screen
            # - Portrait on iOS devices
            orientation = 1
        elif capture_data['CurrentOrientation'] == "PortraitFlipped":
            # Camera below screen
            # - Portrait Inverted on iOS devices
            orientation = 2
        elif capture_data['CurrentOrientation'] == "Landscape":
            # Camera left of screen
            # - Landscape home button on right on iOS devices
            orientation = 3
        elif capture_data['CurrentOrientation'] == "LandscapeFlipped":
            # Camera right of screen
            # - Landscape home button on left on iOS devices
            orientation = 4

    return orientation


def getCaptureTimeString(capture_data):
    sessiontime = dateutil.parser.parse(capture_data["SessionTimestamp"])
    currenttime = dateutil.parser.parse(capture_data["Timestamp"])
    timedelta = sessiontime - currenttime
    return str(timedelta.total_seconds())


data_directory = "EyeCaptures"
output_directory = "EyeCaptures-dlib"

directories = sorted(findCaptureSessionDirs(data_directory))
total_directories = len(directories)

print(f"Found {total_directories} directories")


for directory_idx, directory in enumerate(directories):
    print(f"Processing {directory_idx + 1}/{total_directories} - {directory}")

    captures = findCapturesInSession(os.path.join(data_directory, directory))
    total_captures = len(captures)

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

    recording_path = os.path.join(data_directory, directory)
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

    # info.json - {"TotalFrames":99,"NumFaceDetections":97,"NumEyeDetections":56,"Dataset":"train","DeviceName":"iPhone 6"}
    info = {
        "TotalFrames": total_captures,
        "NumFaceDetections": 0,
        "NumEyeDetections": 0,
        "Dataset": "train",  # For now put all data into training dataset
        "DeviceName": None
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

    for capture_idx, capture in enumerate(captures):
        print(f"Processing {capture_idx + 1}/{total_captures} - {capture}")

        capture_json_path = os.path.join(data_directory, directory, capture + ".json")
        capture_png_path = os.path.join(data_directory, directory, capture + ".jpg")

        if os.path.isfile(capture_json_path) and os.path.isfile(capture_png_path):
            capture_data = loadJsonData(capture_json_path)

            if info["DeviceName"] == None:
                info["DeviceName"] = capture_data["HostModel"]
            elif info["DeviceName"] != capture_data["HostModel"]:
                error(
                    f"Device name changed during session, expected \'{info['DeviceName']}\' but got \'{capture_data['HostModel']}\'")

            capture_image = PILImage.open(capture_png_path).convert(
                'RGB')  # dlib wants images in RGB or 8-bit grayscale format
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
            screen["H"].append(capture_data['ScreenHeightInRawPixels'])
            screen["W"].append(capture_data['ScreenWidthInRawPixels'])
            screen["Orientation"].append(getScreenOrientation(capture_data))

            # dotinfo.json - { "DotNum": [ 0, 0, ... ],
            #                  "XPts": [ 160, 160, ... ],
            #                  "YPts": [ 284, 284, ... ],
            #                  "XCam": [ 1.064, 1.064, ... ],
            #                  "YCam": [ -6.0055, -6.0055, ... ],
            #                  "Time": [ 0.205642, 0.288975, ... ] }
            #
            # PositionIndex == DotNum
            # Timestamp == Time, but no guarantee on order. Unclear if that is an issue or not
            dotinfo["DotNum"].append(capture_data["PositionIndex"])
            dotinfo["XPts"].append(capture_data["RawX"])
            dotinfo["YPts"].append(capture_data["RawY"])
            dotinfo["XCam"].append(0)
            dotinfo["YCam"].append(0)
            dotinfo["Time"].append(getCaptureTimeString(capture_data))

            # Convert image from PNG to JPG
            frame_name = str(f"{capture_idx:05d}.jpg")
            frames.append(frame_name)
            capture_img = PILImage.open(capture_png_path).convert('RGB')
            capture_img.save(os.path.join(output_frame_path, frame_name))
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

print("DONE")