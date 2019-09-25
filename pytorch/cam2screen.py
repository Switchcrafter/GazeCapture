import json
import os

file_dir_path = os.path.dirname(os.path.realpath(__file__))

devicesJson = open(os.path.join(file_dir_path, 'apple_device_data.json'))
devices = json.load(devicesJson)

def cam2screen(xDisplacementFromCameraInCm, yDisplacementFromCameraInCm, orientation, deviceName, widthScreenInPoints, heightScreenInPoints):

    # Need to pass in widthScreenInPoints and heightScreenInPoints because it can vary for the same deviceName due to zoom
    # xDisplacementFromCameraInCm, yDisplacementFromCameraInCm are "prediction space" coordinates

    deviceMetrics = getDeviceMetrics(deviceName)

    if deviceMetrics is None:
        return None

    # Camera Offset and Screen Orientation compensation
    if orientation == 1: # Portrait
        xScreenInCm = xDisplacementFromCameraInCm + deviceMetrics["xCameraToScreenDisplacementInCm"]
        yScreenInCm = -yDisplacementFromCameraInCm - deviceMetrics["yCameraToScreenDisplacementInCm"]
        xScreenInPoints = xScreenInCm / deviceMetrics["widthScreenInCm"] * widthScreenInPoints
        yScreenInPoints = yScreenInCm / deviceMetrics["heightScreenInCm"] * heightScreenInPoints
    elif orientation == 2: # Portrait Inverted
        xScreenInCm = xDisplacementFromCameraInCm - deviceMetrics["xCameraToScreenDisplacementInCm"] + deviceMetrics["widthScreenInCm"]
        yScreenInCm = -yDisplacementFromCameraInCm + deviceMetrics["yCameraToScreenDisplacementInCm"] + deviceMetrics["heightScreenInCm"]
        xScreenInPoints = xScreenInCm / deviceMetrics["widthScreenInCm"] * widthScreenInPoints
        yScreenInPoints = yScreenInCm / deviceMetrics["heightScreenInCm"] * heightScreenInPoints
    elif orientation == 3:
        xScreenInCm = xDisplacementFromCameraInCm - deviceMetrics["yCameraToScreenDisplacementInCm"]
        yScreenInCm = -yDisplacementFromCameraInCm - deviceMetrics["xCameraToScreenDisplacementInCm"] + deviceMetrics["widthScreenInCm"]
        xScreenInPoints = xScreenInCm / deviceMetrics["widthScreenInCm"] * heightScreenInPoints
        yScreenInPoints = yScreenInCm / deviceMetrics["heightScreenInCm"] * widthScreenInPoints
    elif orientation == 4:
        xScreenInCm = xDisplacementFromCameraInCm + deviceMetrics["yCameraToScreenDisplacementInCm"] + deviceMetrics["heightScreenInCm"]
        yScreenInCm = -yDisplacementFromCameraInCm + deviceMetrics["xCameraToScreenDisplacementInCm"]
        xScreenInPoints = xScreenInCm / deviceMetrics["widthScreenInCm"] * heightScreenInPoints
        yScreenInPoints = yScreenInCm / deviceMetrics["heightScreenInCm"] * widthScreenInPoints
    else:
        xScreenInPoints = 0
        yScreenInPoints = 0
        print("Unexpected orientation: {orientation}")

    return (xScreenInPoints, yScreenInPoints)


def getDeviceMetrics(deviceName):

    if not deviceName in devices:
        print(f"Device not found: {deviceName}")
        return None

    return devices[deviceName]