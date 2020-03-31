# Capture Data Format for GazeCapture pytorch

Version 200331

## Overview

Each capture session creates a new subdirectory for data capture named as an increasing int with five digits (e.g. 00005).  There are a number of metadata files:

 screen.json: { H: ..., W: ..., Orientation: ...}
 
 info.json: { DeviceName: ... }

 deviceMetrics.json {
     xCameraToScreenDisplacementInCm: ...,
     yCameraToScreenDisplacementInCm: ...,
     widthScreenInCm: ...,
     heightScreenInCm: ...,
     ppi: ...
     }

Frame data is captured into a subdirectory of the session directory named 'frames'.  two files from an app on the target device; a png compressed image and a json metadata file.  The frame json file contains:

 image.jpg (close to lossless compression as possible)
 dotInfo.json: { XRaw: ..., YRaw: ... }

This data is then post-processed to add metadata files:

 dotCam.json: { XCam: ..., YCam: ... }
 faceGrid: { X: ..., Y: ..., W: ..., H: ..., IsValid: ... }
 leftEyeGrid: { X: ..., Y: ..., W: ..., H: ..., IsValid: ... }
 rightEyeGrid: { X: ..., Y: ..., W: ..., H: ..., IsValid: ... }

And extracted images:

 face.jpg
 leftEye.jpg
 rightEye.jpg

## Changes to schema

Use DeviceSku in place of DeviceName (e.g. different metrics for different Surface Book 2's)

Capturing raw pixels vs. scaled pixels
    e.g. dotInfo.json should contain unscaled device pixels on screen (e.g. zoom mode independent)

## Design Summary

Capture -> Prepare -> ML

### Capture

00005\frames\*.jpg\json

{dataHome}\200331\{deviceSku}\{userName}\

{frame}.jpg     Camera Images in JPG Lossless
{frame}.json    XRaw, YRaw

### Prepare

dotCam.json
faceGrid.json
leftEyeGrid.json
rightEyeGrid.json

face.jpg
leftEye.jpg
rightEye.jpg