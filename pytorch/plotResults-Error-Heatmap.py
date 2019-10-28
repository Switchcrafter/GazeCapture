import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

jsondata = json.load(open("best_results.json", "r"))

x = []
y = []
c = []

for datapoint in jsondata:
    x.append(datapoint["gazePoint"][0])
    y.append(datapoint["gazePoint"][1])
    c.append(math.sqrt(((datapoint["gazePoint"][0]-datapoint["gazePrediction"][0])**2)+((datapoint["gazePoint"][1]-datapoint["gazePrediction"][1])**2)))

cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=min(c), vmax=max(c))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

plt.scatter(x, y, c=scalarMap.to_rgba(c))
#plt.colorbar(scalarMap)
#cb.set_label('Error (cm)')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Heatmap of Error')
plt.show()
