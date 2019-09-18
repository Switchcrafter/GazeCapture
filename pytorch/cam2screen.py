def cam2screen(xDisplacementFromCameraInCm, yDisplacementFromCameraInCm, orientation, deviceName, widthScreenInPoints, heightScreenInPoints):

    # Need to pass in widthScreenInPoints and scheightScreenInPointsreenH because it can vary for the same deviceName due to zoom
    # xDisplacementFromCameraInCm, yDisplacementFromCameraInCm are "prediction space" coordinates

    deviceMetrics = getDeviceMetrics(deviceName)

    if deviceMetrics is None:
        return None

    # Camera Offset and Screen Orientation compensation
    if orientation == 1:
        # Orientation == 1, Portrait
        xScreenInCm = xDisplacementFromCameraInCm + deviceMetrics["xCameraToScreenDisplacementInCm"]
        yScreenInCm = -yDisplacementFromCameraInCm - deviceMetrics["yCameraToScreenDisplacementInCm"]

        # For Orientation == 1,2
        xScreenInPoints = xScreenInCm / deviceMetrics["widthScreenInCm"] * widthScreenInPoints
        yScreenInPoints = yScreenInCm / deviceMetrics["heightScreenInCm"] * heightScreenInPoints
    else:
        # TODO
        xScreenInPoints = 0
        yScreenInPoints = 0


    return (xScreenInPoints, yScreenInPoints)


def getDeviceMetrics(deviceName):

    # TODO: Load from JSON
    devices = {
        "iPhone 6s Plus": {
            "xCameraToScreenDisplacementInCm": 2.354,
            "yCameraToScreenDisplacementInCm": 0.866,
            "widthScreenInCm": 6.836,
            "heightScreenInCm": 12.154
        },
        "iPhone 6s": {
            "xCameraToScreenDisplacementInCm": 1.861,
            "yCameraToScreenDisplacementInCm": 0.804,
            "widthScreenInCm": 5.849,
            "heightScreenInCm": 10.405
        },
        "iPhone 6 Plus": {
            "xCameraToScreenDisplacementInCm": 2.354,
            "yCameraToScreenDisplacementInCm": 0.865,
            "widthScreenInCm": 6.836,
            "heightScreenInCm": 12.154
        },
        "iPhone 6": {
            "xCameraToScreenDisplacementInCm": 1.861,
            "yCameraToScreenDisplacementInCm": 0.803,
            "widthScreenInCm": 5.85,
            "heightScreenInCm": 10.405
        },
        "iPhone 5s": {
            "xCameraToScreenDisplacementInCm": 2.585,
            "yCameraToScreenDisplacementInCm": 1.065,
            "widthScreenInCm": 5.17,
            "heightScreenInCm": 9.039
        },
        "iPhone 5c": {
            "xCameraToScreenDisplacementInCm": 2.585,
            "yCameraToScreenDisplacementInCm": 1.064,
            "widthScreenInCm": 5.17,
            "heightScreenInCm": 9.039
        },
        "iPhone 5": {
            "xCameraToScreenDisplacementInCm": 2.585,
            "yCameraToScreenDisplacementInCm": 1.065,
            "widthScreenInCm": 5.17,
            "heightScreenInCm": 9.039
        },
        "iPhone 4s": {
            "xCameraToScreenDisplacementInCm": 1.496,
            "yCameraToScreenDisplacementInCm": 0.978,
            "widthScreenInCm": 4.992,
            "heightScreenInCm": 7.488
        },
        "iPad Mini": {
            "xCameraToScreenDisplacementInCm": 6.07,
            "yCameraToScreenDisplacementInCm": 0.87,
            "widthScreenInCm": 12.13,
            "heightScreenInCm": 16.12
        },
        "iPad Air 2": {
            "xCameraToScreenDisplacementInCm": 7.686,
            "yCameraToScreenDisplacementInCm": 0.737,
            "widthScreenInCm": 15.371,
            "heightScreenInCm": 20.311
        },
        "iPad Air": {
            "xCameraToScreenDisplacementInCm": 7.44,
            "yCameraToScreenDisplacementInCm": 0.99,
            "widthScreenInCm": 14.9,
            "heightScreenInCm": 19.81
        },
        "iPad 4": {
            "xCameraToScreenDisplacementInCm": 7.45,
            "yCameraToScreenDisplacementInCm": 1.05,
            "widthScreenInCm": 14.9,
            "heightScreenInCm": 19.81
        },
        "iPad 3": {
            "xCameraToScreenDisplacementInCm": 7.45,
            "yCameraToScreenDisplacementInCm": 1.05,
            "widthScreenInCm": 14.9,
            "heightScreenInCm": 19.81
        },
        "iPad 2": {
            "xCameraToScreenDisplacementInCm": 7.45,
            "yCameraToScreenDisplacementInCm": 1.05,
            "widthScreenInCm": 14.9,
            "heightScreenInCm": 19.81
        },
        "iPad Pro": {
            "xCameraToScreenDisplacementInCm": 9.831,
            "yCameraToScreenDisplacementInCm": 1.069,
            "widthScreenInCm": 19.661,
            "heightScreenInCm": 26.215
        }
    }

    if not deviceName in devices:
        return None

    # TODO: Hardcoded to metrics for the iPhone 6s Plus
    return devices[deviceName]