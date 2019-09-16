# Read ~/gc-data/{sampleId}/screen.json[frameId] H, W, Orientation
# Read ~/gc-data/{sampleId}/frames/{frameId}/jpg width height
# Read ~/gc-data/{sampleId}/dotInfo.json[frameId] XCam YCam

# Generate ~/gc-data-prepped/{sampleId}/frames/{frameId}.input.json if it doesn't exist

import os
import sys
import json
import time
from PIL import Image
from PIL import ImageDraw


if len(sys.argv) > 1:
    dataPath = sys.argv[1]
else:
    dataPath = ""

if len(sys.argv) > 2:
    resultsPath = sys.argv[2]
else:
    resultsPath = "best_results.json"

if not os.path.isdir(dataPath):
    print("Invalid arguments: must pass the data directory as first argument and results file as the second argument.")
    sys.exit(-1)

if not os.path.exists(os.path.join(dataPath, "README.md")):
    print(f"{dataPath} does not look like a valid data directory, missing README.md.")
    sys.exit(-1)

if not os.path.exists(resultsPath):
    print(f"Does not look like a valid results file: {resultsPath}")
    sys.exit(-1)

# Metrics collection
resultFrames = 0
skippedFrames = 0
predictedFrames = 0
gazeSamplesCount = 0
gazeSamplesWithPredictionsCount = 0
averageGazeSamplesFrameCount = 0


resultsJson = open(resultsPath)
results = json.load(resultsJson)

# Parse the results into an array of [sampleId, frameId] containing (gazePoint: (x, y), gazePrediction: (x, y))
print("Parsing Results")
gazeSamples = {}
for result in results:
    gazeSample = gazeSamples.pop(result['frame'][0], {})
    gazeSample[result['frame'][1]] = {"gazePointCamera": result['gazePoint'], "gazePredictionCamera": result['gazePrediction']}
    gazeSamples.update({result['frame'][0]: gazeSample })
    resultFrames += 1

gazeSamplesWithPredictionsCount = len(gazeSamples)
gazeSamplesFrameCount = list(map(lambda _: len(_.keys()), gazeSamples.values()))
averageGazeSamplesFrameCount = sum(gazeSamplesFrameCount) / len(gazeSamplesFrameCount)

print("Parsing Results Finished")

with os.scandir(dataPath) as sampleDirEntries:

    for sampleDirEntry in sampleDirEntries:

        if sampleDirEntry.is_dir():  

            gazeSamplesCount += 1
            sampleId = sampleDirEntry.name
            sampleIndex = int(sampleId)
            framesPath = os.path.join(sampleDirEntry.path, "frames")

            screenJson = open(f'{sampleDirEntry.path}/screen.json')
            screen = json.load(screenJson)
            dotInfoJson = open(f'{sampleDirEntry.path}/dotInfo.json')
            dotInfo = json.load(dotInfoJson)

            with os.scandir(framesPath) as framesDirEntries:

                for frameDirEntry in framesDirEntries:

                    frameId = os.path.splitext(frameDirEntry.name)[0]
                    frameIndex = int(frameId)

                    framePrediction = gazeSamples.get(sampleIndex, {}).get(frameIndex)
                    if framePrediction is None:
                        skippedFrames += 1
                        continue
                    else:
                        predictedFrames += 1

                    framePredictionJson = json.dumps(gazeSamples[int(sampleId)][int(frameId)])

                    # Check for existence of frame.input.json, don't regenerate if present
                    if os.path.isfile(f'{sampleDirEntry.path}/frames/{frameId}.input.json'):
                        continue

                    frameImage = Image.open(f'{sampleDirEntry.path}/frames/{frameId}.jpg')
                    frameImageSize = frameImage.size

                    frameInput = {
                        "image": {
                            "width": frameImageSize[0],
                            "height": frameImageSize[1]
                        },
                        "screen": {
                            "width": screen['W'][frameIndex],
                            "height": screen['H'][frameIndex],
                            "orientation": screen["Orientation"][frameIndex]
                        },
                        "gazePoint": {
                            "xcam": dotInfo["XCam"][frameIndex],
                            "ycam": dotInfo["YCam"][frameIndex]
                        }
                    }

                    frameInputJson = json.dumps(frameInput)

                    xScale = frameInput["image"]["width"] / frameInput["screen"]["width"]
                    yScale = frameInput["image"]["height"] / frameInput["screen"]["height"]
                    cameraXPoint = frameInput["screen"]["width"] / 2 # only for frameInput.screen.orientation == 1, 2, 0 for orientation == 3, width for orientation == 4
                    cameraYPoint = 0 # for frameInput.screen.orientation == 1,frameInput.screen.height for orientation == 2, height / 2 for orientation == 3, 4

                    gazeTargetXScreenPixel = xScale * (cameraXPoint + framePrediction["gazePointCamera"][0])
                    gazeTargetYScreenPixel = yScale * (cameraYPoint + framePrediction["gazePointCamera"][1])
                    gazePredictionXScreenPixel = xScale * (cameraXPoint + framePrediction["gazePredictionCamera"][0])
                    gazePredictionYScreenPixel = yScale * (cameraYPoint + framePrediction["gazePredictionCamera"][1])

                    frameOutput = {
                        "gazeTarget" : {
                            "x": gazeTargetXScreenPixel,
                            "y": gazeTargetYScreenPixel
                        },
                        "gazePrediction" : {
                            "x": gazePredictionXScreenPixel,
                            "y": gazePredictionYScreenPixel
                        }
                    }

                    print(f"Raw  {sampleId} {frameId} {xScale:.2f} {yScale:.2f} {framePrediction['gazePointCamera'][0]:.2f} {framePrediction['gazePointCamera'][1]:.2f} {framePrediction['gazePredictionCamera'][0]:.2f} {framePrediction['gazePredictionCamera'][1]:.2f}")
                    # print(f"Calc {gazeTargetXScreenPixel:.2f} {gazeTargetYScreenPixel:.2f} {gazePredictionXScreenPixel:.2f} {gazePredictionYScreenPixel:.2f}")

                    # print(frameInputJson)
                    # print(framePredictionJson)
                    # print(frameOutput)

                    draw = ImageDraw.Draw(frameImage)

                    draw.line( (gazeTargetXScreenPixel, gazeTargetYScreenPixel-25) + (gazeTargetXScreenPixel, gazeTargetYScreenPixel+25), fill=(64, 192, 64), width=2)
                    draw.line( (gazeTargetXScreenPixel-25, gazeTargetYScreenPixel) + (gazeTargetXScreenPixel+25, gazeTargetYScreenPixel), fill=(64, 192, 64), width=2)

                    draw.arc( (gazePredictionXScreenPixel-25, gazePredictionYScreenPixel-25) + (gazePredictionXScreenPixel+25, gazePredictionYScreenPixel+25), 0, 360, fill=(00, 128, 00), width=3)
                    draw.arc( (gazePredictionXScreenPixel-3, gazePredictionYScreenPixel-3) + (gazePredictionXScreenPixel+3, gazePredictionYScreenPixel+3), 0, 360, fill=(00, 128, 00), width=2)

                    filename = f"/data/gc-output/deepthink/2019-01-01.00/{sampleId}/{frameId}_overlay.jpg"

                    # directory = os.path.dirname(filename)
                    # if not os.path.exists(directory):
                    #     os.makedirs(directory)

                    # frameImage.save(filename)



print(f"Total Samples: {gazeSamplesCount}")
print(f"Samples with Predictions: {gazeSamplesWithPredictionsCount}")
print(f"Result Frames: {resultFrames}")
print(f"Predicted Frames: {predictedFrames}")
print(f"Skipped Frames: {skippedFrames}")
print(f"Result Average Frames per Sample: {averageGazeSamplesFrameCount}")
