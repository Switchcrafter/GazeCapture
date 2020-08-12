import json
import os

file_dir_path = os.path.dirname(os.path.realpath(__file__))

devicesJson = open(os.path.join(file_dir_path, '../metadata/device_metrics_sku.json'))
devices = json.load(devicesJson)


def cam2screen(xDisplacementFromCameraInCm,
               yDisplacementFromCameraInCm,
               orientation,
               widthScreenInPoints,
               heightScreenInPoints,
               deviceName = "",
               xCameraToScreenDisplacementInCm = None,
               yCameraToScreenDisplacementInCm = None,
               widthScreenInCm = None,
               heightScreenInCm = None
               ):

    # Need to pass in widthScreenInPoints and heightScreenInPoints because it can vary for
    # the same deviceName due to zoom
    # xDisplacementFromCameraInCm, yDisplacementFromCameraInCm are "prediction space" coordinates

    deviceMetrics = getDeviceMetrics(deviceName)

    if deviceName is not None:
        xCameraToScreenDisplacementInCm = deviceMetrics["xCameraToScreenDisplacementInCm"]
        yCameraToScreenDisplacementInCm = deviceMetrics["yCameraToScreenDisplacementInCm"]
        widthScreenInCm = deviceMetrics["widthScreenInCm"]
        heightScreenInCm = deviceMetrics["heightScreenInCm"]

    if xCameraToScreenDisplacementInCm is None or \
            yCameraToScreenDisplacementInCm is None or \
            widthScreenInCm is None or \
            heightScreenInCm is None:
        return None

    # Camera Offset and Screen Orientation compensation
    if orientation == 1:
        # Camera above screen
        # - Portrait on iOS devices
        # - Landscape on Surface devices
        xScreenInCm = xDisplacementFromCameraInCm + xCameraToScreenDisplacementInCm
        yScreenInCm = -yDisplacementFromCameraInCm - yCameraToScreenDisplacementInCm
        xScreenInPoints = xScreenInCm / widthScreenInCm * widthScreenInPoints
        yScreenInPoints = yScreenInCm / heightScreenInCm * heightScreenInPoints
    elif orientation == 2:
        # Camera below screen
        # - Portrait Inverted on iOS devices
        # - Landscape inverted on Surface devices
        xScreenInCm = xDisplacementFromCameraInCm - xCameraToScreenDisplacementInCm + widthScreenInCm
        yScreenInCm = -yDisplacementFromCameraInCm + yCameraToScreenDisplacementInCm + heightScreenInCm
        xScreenInPoints = xScreenInCm / widthScreenInCm * widthScreenInPoints
        yScreenInPoints = yScreenInCm / heightScreenInCm * heightScreenInPoints
    elif orientation == 3:
        # Camera left of screen
        # - Landscape home button on right on iOS devices
        # - Portrait with camera on left on Surface devices
        xScreenInCm = xDisplacementFromCameraInCm - yCameraToScreenDisplacementInCm
        yScreenInCm = -yDisplacementFromCameraInCm - xCameraToScreenDisplacementInCm + widthScreenInCm
        xScreenInPoints = xScreenInCm / widthScreenInCm * heightScreenInPoints
        yScreenInPoints = yScreenInCm / heightScreenInCm * widthScreenInPoints
    elif orientation == 4:
        # Camera right of screen
        # - Landscape home button on left on iOS devices
        # - Portrait with camera on right on Surface devices
        xScreenInCm = xDisplacementFromCameraInCm + yCameraToScreenDisplacementInCm + heightScreenInCm
        yScreenInCm = -yDisplacementFromCameraInCm + xCameraToScreenDisplacementInCm
        xScreenInPoints = xScreenInCm / widthScreenInCm * heightScreenInPoints
        yScreenInPoints = yScreenInCm / heightScreenInCm * widthScreenInPoints
    else:
        xScreenInPoints = 0
        yScreenInPoints = 0
        print(f"Unexpected orientation: {orientation}")

    return xScreenInPoints, yScreenInPoints


def screen2cam(xScreenInPoints,
               yScreenInPoints,
               orientation,
               widthScreenInPoints,
               heightScreenInPoints,
               deviceName = "",
               xCameraToScreenDisplacementInCm = None,
               yCameraToScreenDisplacementInCm = None,
               widthScreenInCm = None,
               heightScreenInCm = None
               ):

    # Need to pass in widthScreenInPoints and heightScreenInPoints because it can vary for
    # the same deviceName due to zoom
    # xDisplacementFromCameraInCm, yDisplacementFromCameraInCm are "prediction space" coordinates

    deviceMetrics = getDeviceMetrics(deviceName)

    if deviceName is not None:
        xCameraToScreenDisplacementInCm = deviceMetrics["xCameraToScreenDisplacementInCm"]
        yCameraToScreenDisplacementInCm = deviceMetrics["yCameraToScreenDisplacementInCm"]
        widthScreenInCm = deviceMetrics["widthScreenInCm"]
        heightScreenInCm = deviceMetrics["heightScreenInCm"]

    if xCameraToScreenDisplacementInCm is None or \
            yCameraToScreenDisplacementInCm is None or \
            widthScreenInCm is None or \
            heightScreenInCm is None:
        return None

    # Camera Offset and Screen Orientation compensation
    if orientation == 1:
        # Camera above screen
        # - Portrait on iOS devices
        # - Landscape on Surface devices
        xScreenInCm = xScreenInPoints * widthScreenInCm / widthScreenInPoints
        yScreenInCm = yScreenInPoints * heightScreenInCm / heightScreenInPoints

        xDisplacementFromCameraInCm = xScreenInCm - xCameraToScreenDisplacementInCm
        yDisplacementFromCameraInCm = -yScreenInCm - yCameraToScreenDisplacementInCm
    elif orientation == 2:
        # Camera below screen
        # - Portrait Inverted on iOS devices
        # - Landscape inverted on Surface devices
        xScreenInCm = xScreenInPoints * widthScreenInCm / widthScreenInPoints
        yScreenInCm = yScreenInPoints * heightScreenInCm / heightScreenInPoints

        xDisplacementFromCameraInCm = xScreenInCm + xCameraToScreenDisplacementInCm - widthScreenInCm
        yDisplacementFromCameraInCm = -yScreenInCm + yCameraToScreenDisplacementInCm + heightScreenInCm
    elif orientation == 3:
        # Camera left of screen
        # - Landscape home button on right on iOS devices
        # - Portrait with camera on left on Surface devices
        xScreenInCm = xScreenInPoints * widthScreenInCm / heightScreenInPoints
        yScreenInCm = yScreenInPoints * heightScreenInCm / widthScreenInPoints

        xDisplacementFromCameraInCm = xScreenInCm + yCameraToScreenDisplacementInCm
        yDisplacementFromCameraInCm = -yScreenInCm - xCameraToScreenDisplacementInCm + widthScreenInCm
    elif orientation == 4:
        # Camera right of screen
        # - Landscape home button on left on iOS devices
        # - Portrait with camera on right on Surface devices
        xScreenInCm = xScreenInPoints * widthScreenInCm / heightScreenInPoints
        yScreenInCm = yScreenInPoints * heightScreenInCm / widthScreenInPoints

        xDisplacementFromCameraInCm = xScreenInCm - yCameraToScreenDisplacementInCm - heightScreenInCm
        yDisplacementFromCameraInCm = -yScreenInCm + xCameraToScreenDisplacementInCm
    else:
        xDisplacementFromCameraInCm = 0
        yDisplacementFromCameraInCm = 0
        print(f"Unexpected orientation: {orientation}")

    return xDisplacementFromCameraInCm, yDisplacementFromCameraInCm


def getDeviceMetrics(deviceName):

    if not deviceName in devices:
        print(f"Device not found: {deviceName}")
        return None

    return devices[deviceName]