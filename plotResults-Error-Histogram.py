import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

jsondata = json.load(open("best_results.json", "r"))

x = []

for datapoint in jsondata:
    x.append(math.sqrt(((datapoint["gazePoint"][0]-datapoint["gazePrediction"][0])**2)+((datapoint["gazePoint"][1]-datapoint["gazePrediction"][1])**2)))

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5, density=1)
plt.xlabel('Error (cm)')
plt.ylabel('Probability')
plt.title('Histogram of Error')
plt.show()
