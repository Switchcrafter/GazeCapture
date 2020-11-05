import json

import cv2
import os


def main():
    dataset_path = "/data/gc-data"
    recordings = os.listdir(dataset_path)

    for recording in sorted(recordings):
        recording_path = os.path.join(dataset_path, recording)
        print('Processing recording %s' % recording_path)

        with open(os.path.join(recording_path, 'frames.json'), "r") as read_file:
            frames = json.load(read_file)

        with open(os.path.join(recording_path, 'appleFace.json'), "r") as read_file:
            appleFace = json.load(read_file)

        with open(os.path.join(recording_path, 'dlibFace.json'), "r") as read_file:
            dlibFace = json.load(read_file)

        for i in range(len(frames)):
            image_name = frames[i]
            image_path = os.path.join(recording_path, 'frames', image_name)
            image = cv2.imread(image_path)

            cv2.rectangle(image,
                          (int(appleFace['X'][i]), int(appleFace['Y'][i])),
                          (int(appleFace['X'][i] + appleFace['W'][i]), int(appleFace['Y'][i] + appleFace['H'][i])),
                          (255, 0, 0),
                          2)
            cv2.rectangle(image,
                          (dlibFace['X'][i], dlibFace['Y'][i]),
                          (dlibFace['X'][i] + dlibFace['W'][i], dlibFace['Y'][i] + dlibFace['H'][i]),
                          (0, 0, 255),
                          2)

            cv2.imshow(image_path, image)


main()
