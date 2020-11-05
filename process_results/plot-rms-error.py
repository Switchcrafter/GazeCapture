import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
json_path = os.path.join(script_directory, "../metadata/all_rms_errors.json")

All_RMS_Errors = json.load(open(json_path, "r"))

# Make a data frame
rms_object = {'x': range(1, 31)}
for key in All_RMS_Errors.keys():
    if All_RMS_Errors[key]["Plot"]:
        rms_object[key] = np.array((All_RMS_Errors[key])['RMS_Errors'])

df_rms = pd.DataFrame(rms_object)

# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df_rms.drop('x', axis=1):
    num += 1
    plt.plot(df_rms['x'], df_rms[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("RMS Errors by Epoch", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Epoch")
plt.ylabel("RMS Error")
plt.show()

best_rms_object = {'x': range(1, 31)}
for key in All_RMS_Errors.keys():
    if All_RMS_Errors[key]["Plot"]:
        best_rms_object[key] = np.array((All_RMS_Errors[key])['Best_RMS_Errors'])

# Make a data frame
df_best_rms = pd.DataFrame(best_rms_object)

# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df_best_rms.drop('x', axis=1):
    num += 1
    plt.plot(df_best_rms['x'], df_best_rms[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("Best RMS Errors by Epoch", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Epoch")
plt.ylabel("RMS Error")
plt.show()
