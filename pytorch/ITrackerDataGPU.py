from __future__ import division
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# %matplotlib inline

# import collections
import numpy as np
# from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import os
import os.path
import scipy.io as sio
import torch
from Utilities import centeredText

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

class ITrackerDataGPU(object):
    def __init__(self, batch_size, dataPath, metadata, split, gridSize, silent=True):
        self.batch_size = batch_size
        self.dataPath = dataPath
        self.gridSize = gridSize
        self.metadata = metadata

        # if not silent:
        #     print('Loading iTracker dataset')
        # metadata_file = os.path.join(dataPath, 'metadata.mat')
        # self.metadata = self.loadMetadata(metadata_file, silent)
        # self.metadata = ITrackerMetadata(metadata_file, silent)

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

    def __len__(self):
        return len(self.indices)

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
        row = np.array([int(rowIndex)], dtype = np.uint8)
        index = np.array([int(index)], dtype = np.uint8)

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
            imFaceBatch.append(np.frombuffer(imFace.read(), dtype = np.uint8))
            imEyeLBatch.append(np.frombuffer(imEyeL.read(), dtype = np.uint8))
            imEyeRBatch.append(np.frombuffer(imEyeR.read(), dtype = np.uint8))
            faceGridBatch.append(faceGrid)
            gazeBatch.append(gaze)
            frameBatch.append(frame)
            indexBatch.append(index)

            imFace.close()
            imEyeL.close()
            imEyeR.close()

            self.index = (self.index + 1) % self.size
        return (rowBatch, imFaceBatch, imEyeLBatch, imEyeRBatch, faceGridBatch, gazeBatch, frameBatch, indexBatch)

    next = __next__

class ExternalSourcePipeline(Pipeline):
    def __init__(self, data, batch_size, imageSize, split, silent, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)

        self.sourceIterator = iter(data)
        self.rowBatch = ops.ExternalSource()
        self.imFaceBatch = ops.ExternalSource()
        self.imEyeLBatch = ops.ExternalSource()
        self.imEyeRBatch = ops.ExternalSource()
        self.faceGridBatch = ops.ExternalSource()
        self.gazeBatch = ops.ExternalSource()
        self.frameBatch = ops.ExternalSource()
        self.indexBatch = ops.ExternalSource()

        GPU = True
        if GPU:
            # ImageDecoder below accepts  CPU inputs, but returns GPU outputs (hence device = "mixed"), HWC ordering
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
            # The rest of pre-processing is done on the GPU
            self.resize = ops.Resize(device="gpu", resize_x=256, resize_y=256)
            # self.res = ops.RandomResizedCrop(device="gpu", size =(224,224))
            # HWC->CHW, crop (224,224), normalize
            self.norm = ops.CropMirrorNormalize(device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout='CHW',
                                                crop=(224, 224),
                                                image_type=types.RGB,
                                                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                std=[0.229 * 255,0.224 * 255,0.225 * 255])
            self.cast = ops.Cast(device='gpu', dtype=types.FLOAT)#types.INT32,types.UINT8,types.FLOAT
        else:
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.resize = ops.Resize(device="cpu", resize_x=256, resize_y=256)
            self.norm = ops.CropMirrorNormalize(device="cpu",
                                                output_dtype=types.FLOAT,
                                                output_layout='CHW',
                                                crop=(224, 224),
                                                image_type=types.RGB,
                                                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            # self.cast = ops.Cast(device='cpu', dtype=types.FLOAT)


    def define_graph(self):
        self.row = self.rowBatch()
        self.imFace = self.imFaceBatch()
        self.imEyeL = self.imEyeLBatch()
        self.imEyeR = self.imEyeRBatch()
        self.faceGrid = self.faceGridBatch()
        self.gaze = self.gazeBatch()
        self.frame = self.frameBatch()
        self.index = self.indexBatch()

        imFaceD = self.norm(self.resize(self.decode(self.imFace)))
        imEyeLD = self.norm(self.resize(self.decode(self.imEyeL)))
        imEyeRD = self.norm(self.resize(self.decode(self.imEyeR)))

        return (self.row, imFaceD, imEyeLD, imEyeRD, self.faceGrid, self.gaze, self.frame, self.index)

    @property
    def size(self):
        return len(self.sourceIterator)

    def iter_setup(self):
        (rowBatch, imFaceBatch, imEyeLBatch, imEyeRBatch, faceGridBatch, gazeBatch, frameBatch, indexBatch) = self.sourceIterator.next()
        self.feed_input(self.row, rowBatch)
        self.feed_input(self.imFace, imFaceBatch)
        self.feed_input(self.imEyeL, imEyeLBatch)
        self.feed_input(self.imEyeR, imEyeRBatch)
        self.feed_input(self.faceGrid, faceGridBatch)
        self.feed_input(self.gaze, gazeBatch)
        self.feed_input(self.frame, frameBatch)
        self.feed_input(self.index, indexBatch)

class Dataset:
    def __init__(self, split, data, size, loader):
        self.split = split
        self.data = data
        self.size = size
        self.loader = loader

def load_data(split, dataPath, metadata, image_size, grid_size, workers, batch_size, verbose, color_space):
    num_gpus = torch.cuda.device_count()
    # shuffle = True if split == 'train' else False
    distributed = False # distributed=True is currently unstable/experimental mode
    if not distributed:
        data = ITrackerDataGPU(batch_size, dataPath, metadata, split, grid_size, silent=not verbose)
        size = len(data)
        #todo: shuffle, deviceId, color_space
        pipe = ExternalSourcePipeline(data, batch_size=batch_size, imageSize=image_size, split=split, silent=not verbose, num_threads=8, device_id = num_gpus-1)
        pipe.build()
        # Todo: pin memory, PyTorchIterator
        loader = DALIGenericIterator([pipe], ['row', 'imFace', 'imEyeL', 'imEyeR', 'faceGrid', 'gaze', 'frame', 'indices'], size=pipe.size, fill_last_batch=False)
    else: # experimental - unstable (distributed mode)
        data = ITrackerDataGPU(batch_size, dataPath, metadata, split, grid_size, silent=not verbose)
        size = len(data)
        # pipes = [ExternalSourcePipeline(data, batch_size=batch_size//num_gpus, imageSize=image_size, split=split, silent=not verbose, num_threads=8//num_gpus, device_id = i) for i in range(num_gpus)]
        pipes = [ExternalSourcePipeline(data, batch_size=batch_size, imageSize=image_size, split=split, silent=not verbose, num_threads=8, device_id = i) for i in range(num_gpus)]
        pipes[0].build()
        # Todo: pin memory
        loader = DALIGenericIterator(pipes, ['row', 'imFace', 'imEyeL', 'imEyeR', 'faceGrid', 'gaze', 'frame', 'indices'], size=pipes[0].size, fill_last_batch=False)
    return Dataset(split, data, size, loader)

def load_all_data(path, image_size, grid_size, workers, batch_size, verbose, color_space='YCbCr'):
    print(centeredText('Loading Data'))
    eval_boost=True
    metadata = ITrackerMetadata(path, silent=not verbose).metadata
    all_data = {
        # training data : model sees and learns from this data
        'train': load_data('train', path, metadata, image_size, grid_size, workers, batch_size, verbose, color_space),
        # validation data : model sees but never learns from this data
        'val': load_data('val', path, metadata, image_size, grid_size, workers, batch_size, verbose, color_space),
        # test data : model never sees or learns from this data
        'test': load_data('test', path, metadata, image_size, grid_size, workers, batch_size, verbose, color_space)
    }
    return all_data

def show_images(image_batch, batch_size):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(0, figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        # CHW -> HWC
        plt.imshow(np.transpose(image_batch[j], (1,2,0)))
        # # HWC format
        # plt.imshow(image_batch[j])
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    print("Running")
    batch_size = 8
    # dataPath='/home/jatin/data/gc-data-prepped/'
    dataPath='/data/gc-data-prepped/'
    IMAGE_SIZE=(256,256)
    verbose=True
    workers=2
    FACE_GRID_SIZE=(25,25)
    verbose=True
    color_space='RGB'
    datasets = load_all_data(dataPath, IMAGE_SIZE, FACE_GRID_SIZE, workers, batch_size, verbose, color_space)

    for i, data in enumerate(datasets['val'].loader):
        batch_data = data[0]
        row = batch_data["row"]
        imFace = batch_data["imFace"]
        imEyeL = batch_data["imEyeL"]
        imEyeR = batch_data["imEyeR"]
        faceGrid = batch_data["faceGrid"]
        gaze = batch_data["gaze"]
        frame = batch_data["frame"]
        indices = batch_data["indices"]
        # imFace.type('torch.FloatTensor').to(device)
        print(i, row[0], indices[1])
        # print(imFace.to('cpu'))
        # plt.ion()
        # plt.show()
        # show_images(imFace.to('cpu'), batch_size=batch_size)

    # (row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, indices) = pipe.run()
    # print(imFace.as_cpu().at(3).shape)
    # print(imFace.as_cpu().at(3))
    # print(imFace)#<nvidia.dali.backend_impl.TensorListGPU
    # show_images(imFace.as_cpu(), batch_size=batch_size)
    # input('')



