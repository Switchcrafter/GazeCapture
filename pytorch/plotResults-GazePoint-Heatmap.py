import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

jsondata = json.load(open("best_results.json", "r"))

gazepoints = {}
x = []
y = []
c = []

for datapoint in jsondata:
    # create unique gazepoint key
    key = ('%.5f' % datapoint["gazePoint"][0]) + ('%.5f' % datapoint["gazePoint"][1])

    # find key
    if key in gazepoints:
        index = gazepoints[key]
    else:
        index = x.__len__()
        x.append(datapoint["gazePoint"][0])
        y.append(datapoint["gazePoint"][1])
        c.append(0)
        gazepoints[key] = index
    
    c[index] += 1

cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=min(c), vmax=max(c))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

plt.scatter(x, y, c=scalarMap.to_rgba(c))
#plt.colorbar(scalarMap)
#cb.set_label('Count')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.show()
