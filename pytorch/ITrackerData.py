import torch

import os
import os.path
import scipy.io as sio
import numpy as np

from random import shuffle

# CPU data loader
from PIL import Image
import torchvision.transforms as transforms
from face_utilities import hogImage

try:
    # GPU data loader
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except ImportError:
    # If running on a non-CUDA system, stub out Pipeline to prevent code crash
    class Pipeline:
        def __init__(self, *args):
            return


    # If running on a non-CUDA system, stub out DALIGenericIterator to prevent code crash
    class DALIGenericIterator:
        def __init__(self, *args):
            return

from Utilities import centered_text


def normalize_image_transform(image_size, split, jitter, color_space):
    normalize_image = []
    
    # Only for training
    if split == 'train':
        normalize_image.append(transforms.Resize(240))
        if jitter:
            normalize_image.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
        normalize_image.append(transforms.RandomCrop(image_size))
    
    # For training and Eval
    normalize_image.append(transforms.Resize(image_size))
    normalize_image.append(transforms.ToTensor())
    if color_space == 'RGB':
        normalize_image.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # Well known ImageNet values

    return transforms.Compose(normalize_image)


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data, batch_size, image_size, split, silent, num_threads, device_id, data_loader, color_space, shuffle=False):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12)

        self.split = split
        if shuffle:
            data.shuffle()
        self.sourceIterator = iter(data)
        self.rowBatch = ops.ExternalSource()
        self.imFaceBatch = ops.ExternalSource()
        self.imEyeLBatch = ops.ExternalSource()
        self.imEyeRBatch = ops.ExternalSource()
        self.faceGridBatch = ops.ExternalSource()
        self.gazeBatch = ops.ExternalSource()
        self.frameBatch = ops.ExternalSource()
        self.indexBatch = ops.ExternalSource()

        mean = None
        std = None
        if color_space == 'RGB':
            output_type = types.RGB
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255]
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        elif color_space == 'YCbCr':
            output_type = types.YCbCr
        elif color_space == 'L':
            output_type = types.GRAY
        elif color_space == 'BGR':
            output_type = types.BGR
        else:
            print("Unsupported color_space:", color_space)

        # Variation range for Saturation, Contrast, Brightness and Hue
        self.dSaturation = ops.Uniform(range=[0.9, 1.1])
        self.dContrast = ops.Uniform(range=[0.9, 1.1])
        self.dBright = ops.Uniform(range=[0.9, 1.1])
        self.dHue = ops.Uniform(range=[-0.1, 0.1])

        if data_loader == "cpu":
            print("Error: cpu data loader shouldn't be handled by DALI")
        elif data_loader == "dali_cpu":
            self.decode = ops.ImageDecoder(device = "cpu", output_type = output_type)
            self.resize_big = ops.Resize(device="cpu", resize_x=240, resize_y=240)
            # depreciated replace with HSV and ops.BrightnessContrast soon
            self.color_jitter = ops.ColorTwist(device="cpu")
            # random area 0.93-1.0 corresponds to croping randomly from an image of size between (224-240)
            self.crop = ops.RandomResizedCrop(device="cpu", random_area=[0.93, 1.00], size=image_size)
            self.resize = ops.Resize(device="cpu", resize_x=image_size[0], resize_y=image_size[1])
            self.norm = ops.CropMirrorNormalize(device="cpu",
                                                output_dtype=types.FLOAT,
                                                output_layout='CHW',
                                                image_type=output_type,
                                                mean=mean,
                                                std=std)
        else:
            # data_loader == "dali_gpu" or data_loader == "dali_gpu_all":
            # ImageDecoder below accepts CPU inputs, but returns GPU outputs (hence device = "mixed"), HWC ordering
            self.decode = ops.ImageDecoder(device = "mixed", output_type = output_type)
            self.resize_big = ops.Resize(device="gpu", resize_x=240, resize_y=240)
            # depreciated replace with HSV and ops.BrightnessContrast soon
            self.color_jitter = ops.ColorTwist(device="gpu")
            # random area 0.93-1.0 corresponds to croping randomly from an image of size between (224-240)
            self.crop = ops.RandomResizedCrop(device="gpu", random_area=[0.93, 1.00], size=image_size)
            self.resize = ops.Resize(device="gpu", resize_x=image_size[0], resize_y=image_size[1])
            self.norm = ops.CropMirrorNormalize(device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout='CHW',
                                                image_type=output_type,
                                                mean=mean,
                                                std=std)

    def define_graph(self):
        self.row = self.rowBatch()
        self.imFace = self.imFaceBatch()
        self.imEyeL = self.imEyeLBatch()
        self.imEyeR = self.imEyeRBatch()
        self.faceGrid = self.faceGridBatch()
        self.gaze = self.gazeBatch()
        self.frame = self.frameBatch()
        self.index = self.indexBatch()

        sat, con, bri, hue = self.dSaturation(), self.dContrast(), self.dBright(), self.dHue()
        if self.split == 'train':
            imFaceD = self.norm(self.resize(self.crop(self.color_jitter(self.resize_big(self.decode(self.imFace)), saturation=sat, contrast=con, brightness=bri, hue=hue))))
            imEyeLD = self.norm(self.resize(self.crop(self.color_jitter(self.resize_big(self.decode(self.imEyeL)), saturation=sat, contrast=con, brightness=bri, hue=hue))))
            imEyeRD = self.norm(self.resize(self.crop(self.color_jitter(self.resize_big(self.decode(self.imEyeR)), saturation=sat, contrast=con, brightness=bri, hue=hue))))
        else:
            imFaceD = self.norm(self.resize(self.decode(self.imFace)))
            imEyeLD = self.norm(self.resize(self.decode(self.imEyeL)))
            imEyeRD = self.norm(self.resize(self.decode(self.imEyeR)))
            
        return (self.row, imFaceD, imEyeLD, imEyeRD, self.faceGrid, self.gaze, self.frame, self.index)

    @property
    def size(self):
        return len(self.sourceIterator)

    def iter_setup(self):
        (rowBatch, imFaceBatch, imEyeLBatch, imEyeRBatch, faceGridBatch, gazeBatch, frameBatch,
         indexBatch) = self.sourceIterator.next()
        self.feed_input(self.row, rowBatch)
        self.feed_input(self.imFace, imFaceBatch)
        self.feed_input(self.imEyeL, imEyeLBatch)
        self.feed_input(self.imEyeR, imEyeRBatch)
        self.feed_input(self.faceGrid, faceGridBatch)
        self.feed_input(self.gaze, gazeBatch)
        self.feed_input(self.frame, frameBatch)
        self.feed_input(self.index, indexBatch)


class ITrackerMetadata(object):
    def __init__(self, dataPath, silent=True):
        if not silent:
            print('Loading iTracker dataset')
        metadata_file = os.path.join(dataPath, 'metadata.mat')
        self.metadata = self.loadMetadata(metadata_file, silent)

    def loadMetadata(self, filename, silent):
        if filename is None or not os.path.isfile(filename):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % filename)
        try:
            # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
            if not silent:
                print('\tReading metadata from %s' % filename)
            metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
        except:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % filename)
        return metadata


class Dataset:
    def __init__(self, split, data, size, loader):
        self.split = split
        self.data = data
        self.size = size
        self.loader = loader


class ITrackerData(object):
    def __init__(self,
                 dataPath,
                 metadata,
                 batch_size,
                 imSize,
                 gridSize,
                 split,
                 silent=True,
                 jitter=True,
                 color_space='YCbCr',
                 data_loader='cpu'):
        self.dataPath = dataPath
        self.metadata = metadata
        self.batch_size = batch_size
        self.imSize = imSize
        self.gridSize = gridSize
        self.color_space = color_space
        self.data_loader = data_loader

        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        elif split == 'train':
            mask = self.metadata['labelTrain']
        elif split == 'all':
            mask = np.ones[len(self.metadata)]
        else:
            raise Exception('split should be test, val or train. The value of split was: {}'.format(split))

        self.indices = np.argwhere(mask)[:, 0]

        if not silent:
            print('Loaded iTracker dataset split "%s" with %d records.' % (split, len(self.indices)))

        if self.data_loader == 'cpu':
            self.normalize_image = normalize_image_transform(image_size=self.imSize, jitter=jitter, split=split, color_space=self.color_space)

    def __len__(self):
        return len(self.indices)

    def loadImage(self, path):
        try:
            im = Image.open(path).convert(self.color_space)
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
        return im
    
    def get_hog_descriptor(self, im):
        # im = Image.fromarray(hogImage(im), im.mode)
        # hog is failing below (20,20) so this should fix
        if im.size[0] < 20:
            im = transforms.functional.resize(im, (20,20), interpolation=2)
        try:
            hog = hogImage(im)
            im = Image.fromarray(hog, im.mode)
        except:
            # print(im.size)
            pass
        return im

    # merge two
    def __getitem__(self, index):
        rowIndex = self.indices[index]
        imFacePath = os.path.join(self.dataPath,
                                  '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][rowIndex],
                                                               self.metadata['frameIndex'][rowIndex]))
        imEyeLPath = os.path.join(self.dataPath,
                                  '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][rowIndex],
                                                                  self.metadata['frameIndex'][rowIndex]))
        imEyeRPath = os.path.join(self.dataPath,
                                  '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][rowIndex],
                                                                   self.metadata['frameIndex'][rowIndex]))
        gaze = np.array([self.metadata['labelDotXCam'][rowIndex], self.metadata['labelDotYCam'][rowIndex]], np.float32)
        frame = np.array([self.metadata['labelRecNum'][rowIndex], self.metadata['frameIndex'][rowIndex]])
        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][rowIndex, :])
        row = np.array([int(rowIndex)], dtype=np.uint8)
        index = np.array([int(index)], dtype=np.uint8)

        if self.data_loader == 'cpu':
            # Image loading, transformation and normalization happen here
            imFace = self.loadImage(imFacePath)
            imEyeL = self.loadImage(imEyeLPath)
            imEyeR = self.loadImage(imEyeRPath)

            # for hog experiments
            # imFace = self.get_hog_descriptor(imFace)
            # imEyeL = self.get_hog_descriptor(imEyeL)
            # imEyeR = self.get_hog_descriptor(imEyeR)

            imFace = self.normalize_image(imFace)
            imEyeL = self.normalize_image(imEyeL)
            imEyeR = self.normalize_image(imEyeR)

            # to tensor
            row = torch.LongTensor([int(index)])
            faceGrid = torch.FloatTensor(faceGrid)
            gaze = torch.FloatTensor(gaze)

            return row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, index
        else:
            # image loading, transformation and normalization happen in ExternalDataPipeline
            # we just pass imagePaths
            return row, imFacePath, imEyeLPath, imEyeRPath, faceGrid, gaze, frame, index

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

    def __iter__(self):
        self.index = 0
        self.size = len(self.indices)
        return self

    def shuffle(self):
        shuffle(self.indices)

    def __next__(self):
        rowBatch = []
        imFaceBatch = []
        imEyeLBatch = []
        imEyeRBatch = []
        faceGridBatch = []
        gazeBatch = []
        frameBatch = []
        indexBatch = []
        labels = []
        for _ in range(self.batch_size):
            row, imFacePath, imEyeLPath, imEyeRPath, faceGrid, gaze, frame, index = self.__getitem__(self.index)
            # print(row, index, self.index, self.size)
            imFace = open(imFacePath, 'rb')
            imEyeL = open(imEyeLPath, 'rb')
            imEyeR = open(imEyeRPath, 'rb')

            rowBatch.append(row)
            imFaceBatch.append(np.frombuffer(imFace.read(), dtype=np.uint8))
            imEyeLBatch.append(np.frombuffer(imEyeL.read(), dtype=np.uint8))
            imEyeRBatch.append(np.frombuffer(imEyeR.read(), dtype=np.uint8))
            faceGridBatch.append(faceGrid)
            gazeBatch.append(gaze)
            frameBatch.append(frame)
            indexBatch.append(index)

            imFace.close()
            imEyeL.close()
            imEyeR.close()

            self.index = (self.index + 1) % self.size
        return rowBatch, imFaceBatch, imEyeLBatch, imEyeRBatch, faceGridBatch, gazeBatch, frameBatch, indexBatch

    next = __next__


def load_data(split,
              dataPath,
              metadata,
              image_size,
              grid_size,
              workers,
              batch_size,
              verbose,
              color_space,
              data_loader,
              eval_boost):
    shuffle = True if split == 'train' else False
    data = ITrackerData(dataPath,
                        metadata,
                        batch_size,
                        image_size,
                        grid_size,
                        split,
                        silent=not verbose,
                        jitter=True,
                        color_space=color_space,
                        data_loader=data_loader)
    size = len(data)

    if data_loader == "cpu":
        if eval_boost:
            batch_size = batch_size if split == 'train' else batch_size * 2
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=False)
    else:
        num_gpus = torch.cuda.device_count()
        if data_loader == "dali_gpu" or data_loader == "dali_cpu":
            pipes = [ExternalSourcePipeline(data,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            split=split,
                                            silent=not verbose,
                                            num_threads=8,
                                            device_id=num_gpus - 1,
                                            data_loader=data_loader,
                                            color_space=color_space,
                                            shuffle=shuffle)]
        elif data_loader == "dali_gpu_all":
            pipes = [ExternalSourcePipeline(data,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            split=split,
                                            silent=not verbose,
                                            num_threads=1,
                                            device_id=i,
                                            data_loader=data_loader,
                                            color_space=color_space,
                                            shuffle=shuffle) for i in range(num_gpus)]
        else:
            # todo raise error
            print("Invalid data_loader mode", data_loader)
        # Todo: pin memory
        # auto_reset=True resets the iterator after each epoch
        # DALIGenericIterator has inbuilt build for all pipelines
        loader = DALIGenericIterator(pipes,
                                     ['row', 'imFace', 'imEyeL', 'imEyeR', 'faceGrid', 'gaze', 'frame', 'indices'],
                                     size=len(data),
                                     fill_last_batch=False,
                                     last_batch_padded=True, auto_reset=True)

    return Dataset(split, data, size, loader)


def load_all_data(path,
                  image_size,
                  grid_size,
                  workers,
                  batch_size,
                  verbose,
                  color_space='YCbCr',
                  data_loader='cpu',
                  eval_boost=False):
    print(centered_text('Loading Data'))
    metadata = ITrackerMetadata(path, silent=not verbose).metadata
    splits = ['train', 'val', 'test']
    all_data = {
        split: load_data(split,
                         path,
                         metadata,
                         image_size,
                         grid_size,
                         workers,
                         batch_size,
                         verbose,
                         color_space,
                         data_loader,
                         eval_boost)
        for split in splits}
    return all_data
