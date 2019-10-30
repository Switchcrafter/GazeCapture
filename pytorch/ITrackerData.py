import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

MEAN_PATH = os.path.dirname(os.path.realpath(__file__))


def loadMetadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, mean_image):
        self.meanImg = transforms.ToTensor()(mean_image / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.meanImg)


class NormalizeImage:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

        self.mean_face = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'), silent=True)['image_mean']
        self.mean_left = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'), silent=True)['image_mean']
        self.mean_right = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'), silent=True)['image_mean']

        self.transform_face = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            SubtractMean(mean_image=self.mean_face),
        ])
        self.transform_eye_left = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            SubtractMean(mean_image=self.mean_left),
        ])
        self.transform_eye_right = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            SubtractMean(mean_image=self.mean_right),
        ])

    def face(self, image):
        return self.transform_face(image)

    def eye_left(self, image):
        return self.transform_eye_left(image)

    def eye_right(self, image):
        return self.transform_eye_right(image)


class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split='train', imSize=(224, 224), gridSize=(25, 25), silent=False):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')
        metadata_file = os.path.join(dataPath, 'metadata.mat')

        if metadata_file is None or not os.path.isfile(metadata_file):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metadata_file)
        self.metadata = loadMetadata(metadata_file, silent=True)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metadata_file)

        self.normalize_image = NormalizeImage(image_size=self.imSize)

        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        elif split == 'train':
            mask = self.metadata['labelTrain']
        else:
            raise Exception('split should be test, val or train. The value of split was: {}'.format(split))

        self.indices = np.argwhere(mask)[:, 0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            # im = Image.new("RGB", self.imSize, "white")

        return im

    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        imFacePath = os.path.join(self.dataPath,
                                  '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index],
                                                               self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath,
                                  '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index],
                                                                  self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath,
                                  '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index],
                                                                   self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.normalize_image.face(image=imFace)
        imEyeL = self.normalize_image.eye_left(image=imEyeL)
        imEyeR = self.normalize_image.eye_right(image=imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
        frame = np.array([self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]])

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index, :])

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame

    def __len__(self):
        return len(self.indices)
