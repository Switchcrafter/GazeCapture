import scipy.io as sio
import json

from Utilities import SimpleProgressBar

data = sio.loadmat('./reference_metadata.mat', struct_as_record=False)

recordingDict = {
    "recNum": [],
    "label": []
}

total_frames = len(data['frameIndex'])
frames_bar = SimpleProgressBar(max_value=total_frames, label="frame")

for i, frame in enumerate(data['frameIndex']):
    frames_bar.update(i + 1)

    recNum = data['labelRecNum'][i][0]
    labelTest = data['labelTest'][i][0]
    labelTrain = data['labelTrain'][i][0]
    labelVal = data['labelVal'][i][0]

    label = None
    if labelTest:
        label = 'test'
    elif labelTrain:
        label = 'train'
    elif labelVal:
        label = 'val'
    else:
        print(f"Error: No valid dataset for recording {recNum}")

    try:
        recNum_index = recordingDict['recNum'].index(recNum)
    except ValueError:
        # expected when item not in
        recNum_index = None

    if recNum_index is None:
        recordingDict['recNum'].append(int(recNum))
        recordingDict['label'].append(label)

with open('reference_data_split.json', 'w') as outfile:
    json.dump(recordingDict, outfile)
