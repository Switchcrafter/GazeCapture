import os
import os.path
import scipy.io as sio
import numpy as np
import torch
import math
from random import shuffle

# CPU data loader
from PIL import Image
import torchvision.transforms as transforms
from utility_functions.face_utilities import hogImage
from utility_functions.Utilities import centered_text

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


def resize_image_transform(image_size):
    normalize_image = []
    normalize_image.append(transforms.Resize(image_size))
    normalize_image.append(transforms.ToTensor())
    return transforms.Compose(normalize_image)


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data, batch_size, image_size, split, silent, num_threads, device_id, data_loader, color_space, shuffle=False):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=-1)

        self.split = split
        self.color_space = color_space
        self.data_loader = data_loader
        if shuffle:
            data.shuffle()
        self.sourceIterator = iter(data)
        self.rowBatch = ops.ExternalSource()
        self.imFaceBatch = ops.ExternalSource()
        self.imEyeLBatch = ops.ExternalSource()
        self.imEyeRBatch = ops.ExternalSource()
        self.imFaceGridBatch = ops.ExternalSource()
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
        else:
            # ---------- Decoding Operations --------- #
            # ImageDecoder in mixed mode doesn't support YCbCr 
            # Ref: https://github.com/NVIDIA/DALI/pull/582/files
            self.decode = ops.ImageDecoder(device="cpu", output_type=output_type)

            # ---------- Augmentation Operations --------- #
            # execute rest of the operations on the target device based upon the mode
            device = "cpu" if data_loader == "dali_cpu" else "gpu"
            self.resize_big = ops.Resize(device=device, resize_x=240, resize_y=240)
            # depreciated replace with HSV and ops.BrightnessContrast soon
            self.color_jitter = ops.ColorTwist(device=device, image_type=output_type)
            # random area 0.93-1.0 corresponds to croping randomly from an image of size between (224-240)
            self.crop = ops.RandomResizedCrop(device=device, random_area=[0.93, 1.00], size=image_size)

            # ---------- Normalization Operations --------- #
            self.resize = ops.Resize(device=device, resize_x=image_size[0], resize_y=image_size[1])
            self.norm = ops.CropMirrorNormalize(device=device,
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
        self.imFaceGrid = self.imFaceGridBatch()
        self.gaze = self.gazeBatch()
        self.frame = self.frameBatch()
        self.index = self.indexBatch()
        sat, con, bri, hue = self.dSaturation(), self.dContrast(), self.dBright(), self.dHue()

        def stream(image, augment=True):
            # Decoding
            image = self.decode(image)
            if self.data_loader == "dali_gpu":
                image = image.gpu()
            # Augmentations (for training only)
            if self.split == 'train' and augment:
                image = self.resize_big(image)
                image = self.color_jitter(image, saturation=sat, contrast=con, brightness=bri, hue=hue)
            # Normalize
            image = self.resize(image)
            image = self.norm(image)
            return image
    
        # pass the input through dali stream
        imFaceD = stream(self.imFace)
        imEyeLD = stream(self.imEyeL)
        imEyeRD = stream(self.imEyeR)
        imFaceGridD = stream(self.imFaceGrid, False)
        return (self.row, imFaceD, imEyeLD, imEyeRD, imFaceGridD, self.gaze, self.frame, self.index)

    @property
    def size(self):
        return len(self.sourceIterator)

    def iter_setup(self):
        (rowBatch, imFaceBatch, imEyeLBatch, imEyeRBatch, imFaceGridBatch, gazeBatch, frameBatch,
         indexBatch) = self.sourceIterator.next()
        self.feed_input(self.row, rowBatch)
        self.feed_input(self.imFace, imFaceBatch)
        self.feed_input(self.imEyeL, imEyeLBatch)
        self.feed_input(self.imEyeR, imEyeRBatch)
        self.feed_input(self.imFaceGrid, imFaceGridBatch)
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
                 data_loader='cpu',
                 shard_id=0,
                 num_shards=1):
        self.dataPath = dataPath
        self.metadata = metadata
        self.batch_size = batch_size
        self.imSize = imSize
        self.gridSize = gridSize
        self.color_space = color_space
        self.data_loader = data_loader
        self.index = 0

        # ======= Sharding configuration variables  ========
        if num_shards > 0:
            self.num_shards = num_shards
        else:
            raise ValueError("num_shards cannot be negative")

        if shard_id >= 0 and shard_id < self.num_shards: 
            self.shard_id = shard_id
        else:
            raise ValueError(f"shard_id should be between 0 and %d i.e. 0 <= shard_id < num_shards."%(num_shards))
        # ====================================================

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
            self.resize_transform = resize_image_transform(image_size=self.imSize)
            

    def __len__(self):
        return math.floor(len(self.indices)/self.num_shards)

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

    def __getitem__(self, shard_index):
        # mapping for shards: shard index to absolute index
        index = self.shard_id * self.__len__() + shard_index
        
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
        imFaceGridPath = os.path.join(self.dataPath,
                                  '%05d/faceGrid/%05d.jpg' % (self.metadata['labelRecNum'][rowIndex],
                                                                   self.metadata['frameIndex'][rowIndex]))
        # Note: Converted from double (float64) to float (float32) as pipeline output is float in MSE calculation
        gaze = np.array([self.metadata['labelDotXCam'][rowIndex], self.metadata['labelDotYCam'][rowIndex]], np.float32)
        frame = np.array([self.metadata['labelRecNum'][rowIndex], self.metadata['frameIndex'][rowIndex]])
        # faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][rowIndex, :])
        row = np.array([int(rowIndex)])
        index = np.array([int(index)])

        if self.data_loader == 'cpu':
            # Image loading, transformation and normalization happen here
            imFace = self.loadImage(imFacePath)
            imEyeL = self.loadImage(imEyeLPath)
            imEyeR = self.loadImage(imEyeRPath)
            imfaceGrid = self.loadImage(imFaceGridPath)

            imFace = self.normalize_image(imFace)
            imEyeL = self.normalize_image(imEyeL)
            imEyeR = self.normalize_image(imEyeR)
            imfaceGrid = self.resize_transform(imfaceGrid)

            # to tensor
            row = torch.LongTensor([int(index)])
            # faceGrid = torch.FloatTensor(faceGrid)
            gaze = torch.FloatTensor(gaze)

            return row, imFace, imEyeL, imEyeR, imfaceGrid, gaze, frame, index
        else:
            # image loading, transformation and normalization happen in ExternalDataPipeline
            # we just pass imagePaths
            return row, imFacePath, imEyeLPath, imEyeRPath, imFaceGridPath, gaze, frame, index

    # TODO: Not in use anymore due to RC. Should eventually be removed
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

    # used by dali
    def __iter__(self):
        self.size = self.__len__()
        return self

    def shuffle(self):
        shuffle(self.indices)

    def __next__(self):
        rowBatch = []
        imFaceBatch = []
        imEyeLBatch = []
        imEyeRBatch = []
        imFaceGridBatch = []
        gazeBatch = []
        frameBatch = []
        indexBatch = []
        labels = []

        for local_index in range(self.batch_size):
            row, imFacePath, imEyeLPath, imEyeRPath, imFaceGridPath, gaze, frame, index = self.__getitem__(self.index)
            self.index = (self.index + 1) % self.__len__()

            imFace = open(imFacePath, 'rb')
            imEyeL = open(imEyeLPath, 'rb')
            imEyeR = open(imEyeRPath, 'rb')
            imFaceGrid = open(imFaceGridPath, 'rb')

            rowBatch.append(row)
            imFaceBatch.append(np.frombuffer(imFace.read(), dtype=np.uint8))
            imEyeLBatch.append(np.frombuffer(imEyeL.read(), dtype=np.uint8))
            imEyeRBatch.append(np.frombuffer(imEyeR.read(), dtype=np.uint8))
            imFaceGridBatch.append(np.frombuffer(imFaceGrid.read(), dtype=np.uint8))
            gazeBatch.append(gaze)
            frameBatch.append(frame)
            indexBatch.append(index)

            imFace.close()
            imEyeL.close()
            imEyeR.close()
            imFaceGrid.close()
        
        return rowBatch, imFaceBatch, imEyeLBatch, imEyeRBatch, imFaceGridBatch, gazeBatch, frameBatch, indexBatch

    # For compatibiity with Python 2
    def next(self):
        return self.__next__()


def load_data(split,
              dataPath,
              metadata,
              image_size,
              grid_size,
              workers,
              batch_size,
              verbose,
              local_rank,
              color_space,
              data_loader,
              eval_boost,
              mode):
    
    shuffle = True if split == 'train' else False
    
    # Enable shading here for ddp2 mode only
    if mode == "ddp2":
        shard_id, num_shards = local_rank[0], torch.cuda.device_count()
    else:
        shard_id, num_shards = 0, 1

    if eval_boost:
        batch_size = batch_size if split == 'train' else batch_size * 2
    data = ITrackerData(dataPath,
                        metadata,
                        batch_size,
                        image_size,
                        grid_size,
                        split,
                        silent=not verbose,
                        jitter=True,
                        color_space=color_space,
                        data_loader=data_loader,
                        shard_id=shard_id,
                        num_shards=num_shards)
    size = len(data)

    # DALI implementation would do a cross-shard shuffle
    # CPU implementation would do a in-shard shuffle
    if data_loader == "cpu":
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True) 
    elif data_loader == "dali_gpu" or data_loader == "dali_cpu":
        pipes = [ExternalSourcePipeline(data,
                                        batch_size=batch_size,
                                        image_size=image_size,
                                        split=split,
                                        silent=not verbose,
                                        num_threads=8,
                                        device_id=local_rank[0],
                                        data_loader=data_loader,
                                        color_space=color_space,
                                        shuffle=True)]

        # DALI automatically allocates Pinned memory whereever possible
        # auto_reset=True resets the iterator after each epoch
        # DALIGenericIterator has inbuilt build for all pipelines
        loader = DALIGenericIterator(pipes,
                                    ['row', 'imFace', 'imEyeL', 'imEyeR', 'imFaceGrid', 'gaze', 'frame', 'indices'],
                                    size=len(data),
                                    fill_last_batch=False,
                                    last_batch_padded=True, auto_reset=True)
    else:
        raise ValueError(f"Invalid data_loader mode: %s"%(data_loader))

    return Dataset(split, data, size, loader)


def load_all_data(path,
                  image_size,
                  grid_size,
                  workers,
                  batch_size,
                  verbose,
                  local_rank,
                  color_space='YCbCr',
                  data_loader='cpu',
                  eval_boost=False,
                  mode='none'):
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
                         local_rank,
                         color_space,
                         data_loader,
                         eval_boost,
                         mode)
        for split in splits}
    return all_data
