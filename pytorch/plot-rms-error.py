import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

All_RMS_Errors = {
    '224': {
        'RMS_Errors': [2.2792, 2.1564, 2.1008, 2.0834, 2.0686, 2.0394, 2.0582, 2.0767, 2.0929, 2.0364, 2.0401, 2.0394,
                       2.0723, 2.0292, 2.0341, 2.0799, 2.0749, 2.0886, 2.0446, 2.1027, 2.0727, 2.1012, 2.0383, 2.0473,
                       2.0619, 2.0223, 2.0581, 2.0354, 2.0375, 2.0400],
        'Best_RMS_Errors': [2.2792, 2.1564, 2.1008, 2.0834, 2.0686, 2.0394, 2.0394, 2.0394, 2.0394, 2.0364, 2.0364,
                            2.0364, 2.0364, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292,
                            2.0292, 2.0292, 2.0292, 2.0223, 2.0223, 2.0223, 2.0223, 2.0223],
    },
    '224-jitter-nomean': {
        'RMS_Errors': [2.3187, 2.2637, 2.2018, 2.1459, 2.1140, 2.0455, 2.0689, 2.0829, 2.0368, 2.0665, 2.0654, 2.0710,
                       2.0413, 2.0709, 2.0352, 2.0208, 2.0746, 2.0568, 2.0309, 2.0578, 2.0200, 2.0097, 2.0214, 2.0078,
                       2.0808, 2.0371, 2.0536, 2.0175, 2.0212, 2.0055],
        'Best_RMS_Errors': [2.3187, 2.2637, 2.2018, 2.1459, 2.1140, 2.0455, 2.0455, 2.0455, 2.0368, 2.0368, 2.0368,
                            2.0368, 2.0368, 2.0368, 2.0352, 2.0208, 2.0208, 2.0208, 2.0208, 2.0208, 2.0200, 2.0097,
                            2.0097, 2.0078, 2.0078, 2.0078, 2.0078, 2.0078, 2.0078, 2.0055],
    },
    '227-jitter-nomean': {
        'RMS_Errors': [2.3007, 2.1911, 2.1377, 2.0939, 2.1228, 2.0545, 2.0816, 2.0431, 2.0513, 2.0681, 2.0387, 2.0061,
                       2.0101, 2.0254, 2.0373, 2.0231, 2.0822, 2.0425, 2.0579, 2.0423, 2.0890, 2.0180, 2.0271, 2.0425,
                       2.0212, 2.0383, 2.0417, 2.0725, 2.0528, 2.0300],
        'Best_RMS_Errors': [2.3007, 2.1911, 2.1377, 2.0939, 2.0939, 2.0545, 2.0545, 2.0431, 2.0431, 2.0431, 2.0387,
                            2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061,
                            2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061, 2.0061],
    },
}

# Make a data frame
df_rms = pd.DataFrame({'x': range(1, 31),
                       '224': np.array((All_RMS_Errors['224'])['RMS_Errors']),
                       '224-jitter-nomean': np.array((All_RMS_Errors['224-jitter-nomean'])['RMS_Errors']),
                       '227-jitter-nomean': np.array((All_RMS_Errors['227-jitter-nomean'])['RMS_Errors']),
                       })

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

# Make a data frame
df_best_rms = pd.DataFrame({'x': range(1, 31),
                       '224': np.array((All_RMS_Errors['224'])['Best_RMS_Errors']),
                       '224-jitter-nomean': np.array((All_RMS_Errors['224-jitter-nomean'])['Best_RMS_Errors']),
                       '227-jitter-nomean': np.array((All_RMS_Errors['227-jitter-nomean'])['Best_RMS_Errors']),
                       })

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
