# Capture Data Format for GazeCapture pytorch

Schema Version 200407

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
 dotInfo.json: { XRaw: ..., YRaw: ..., Confidence: ... }
 
 (dotInfo.json Confidence will be positive is measured, negative otherwise)

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

Use DeviceSku in place of DeviceName (e.g. different metrics for different Surface Book 2s with different screen sizes)

Capturing raw pixels vs. scaled pixels
    e.g. dotInfo.json should contain unscaled device pixels on screen (e.g. zoom mode independent)

## Design Summary

Capture -> Prepare -> ML

### Capture

#### This is how MIT GazeCapture stores the data:

{dataHome}\{sessionId}\frames\{frameId}.jpg
{dataHome}\{sessionId}\frames.json              JSON array of jpg file names
{dataHome}\{sessionId}\dotInfo.json             JSON array of X/Y target point for each frame
    Arrays: DotNum, XPts, YPts, XCam, YCam, Time
{dataHome}\{sessionId}\info.json                Facial feature recognition metadata &amp; device type
    TotalFrames, NumFaceDetections, NumEyeDetections, Dataset (train/validate/test), DeviceName
{dataHome}\{sessionId}\screen.json              Screen W/H/Orientation for frames
    Arrays: H, W, Orientation

#### This is how we will store the data from EyeCapture

    /{dataHome}/{schemaVersion}/{deviceSku}/{userName}/sessionId/*.json
    /{dataHome}/{schemaVersion}/{deviceSku}/{userName}/sessionId/frames/*.json & *.jpg

e.g.

    /data/200407/Surface_Pro_6_1796_Commercial/sha256hashof(jbeavers)/guid/frames/00000.json & 00000.jpg

{frameId}.jpg       Camera Images in JPG Lossless
{frameId}.json      { "XRaw":..., "YRaw":..., "Confidence":... }
frames.json         JSON array of jpg file names
dotInfo.json        JSON arrays: DotNum, XPts, YPts, XCam, YCam, Time
                    XPts/YPts are in device dependent pixel coordinates, unaffected by display zoom.
info.json
                    TotalFrames, NumFaceDetections, NumEyeDetections, Dataset (train/validate/test), DeviceName
screen.json         JSON arrays: H, W, Orientation
                    Since we only support capturing in the default landscape orientation, these values are just duplicates

### Prepare Dataset step

This step using facial feature recognition to identify the face and eye bounding boxes and extract the face and eyes images.  It also calculates the camera distance offsets (dotCam) using screen metrics and device + orientation to camera position lookup table.

Since we are only going to support capture and playback on 'identical' devices for now in a singular orientation, we can optionally skip the dotCam calculation step.

info.json       Updates NumFaceDetections and NumEyeDetections based on dlib results

faceGrid.json
dlibFace.json
dlibLeftEyeGrid.json
dlibRightEyeGrid.json

appleFace/{frameId}.jpg
appleLeftEye/{frameId}.jpg
appleRightEye/{frameId}.jpg
