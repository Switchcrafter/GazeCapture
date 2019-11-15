import json
import math
import csv

jsondata = json.load(open("best_results.json", "r"))

csvwriter = csv.writer(open("best_results.csv", "w", newline=''))
csvwriter.writerow(
    ['frameId', 'frame0', 'frame1', 'gazePointX', 'gazePointY', 'gazePreditionX', 'gazePredictionY', 'distance'])

for datapoint in jsondata:
    distance = math.sqrt(((datapoint["gazePoint"][0] - datapoint["gazePrediction"][0]) ** 2) + (
                (datapoint["gazePoint"][1] - datapoint["gazePrediction"][1]) ** 2))
    csvwriter.writerow(
        [f'{datapoint["frame"][0]}_{datapoint["frame"][1]}', datapoint["frame"][0], datapoint["frame"][1],
         datapoint["gazePoint"][0], datapoint["gazePoint"][1], datapoint["gazePrediction"][0],
         datapoint["gazePrediction"][1], distance])
