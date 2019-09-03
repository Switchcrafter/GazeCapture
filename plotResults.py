import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

jsondata = json.load(open("best_results.json", "r"))
for datapoint in jsondata:
    datapoint["distance"] = math.sqrt(((datapoint["gazePoint"][0]-datapoint["gazePrediction"][0])**2)+((datapoint["gazePoint"][1]-datapoint["gazePrediction"][1])**2))

x = []
y = []
c = []

for datapoint in jsondata:
    x.append(datapoint["gazePoint"][0])
    y.append(datapoint["gazePoint"][1])
    c.append(datapoint["distance"])

cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=min(c), vmax=max(c))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y, c=scalarMap.to_rgba(c))
fig.colorbar(scalarMap)
plt.show()