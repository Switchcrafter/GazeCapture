# Towards Hardware-Agnostic Gaze-Trackers

## Introduction
This is the official repository for the code, models and datasets associated with the 2020 ArXiv paper, [“Towards Hardware-Agnostic Gaze-Trackers”](https://arxiv.org/abs/2010.05123).

We express our sincere thanks to Krakfa, Khosla, Kellnhofer et al &ndash; authors of the 2016 CVPR paper, ["Eye Tracking for Everyone"](https://people.csail.mit.edu/khosla/papers/cvpr2016_Khosla.pdf) &ndash; who contributed the iTracker architecture and [GazeCapture](https://github.com/CSAILVision/GazeCapture) dataset to the entire research community. Their dataset is by far one of the biggest and most diverse publicly available data for constrained gaze-tracking on portable devices and their work on the iTracker architecture is an inspiration to this work. We reproduced the iTracker network architecture using Python as a baseline to our incremental enhancements and utilized the original GazeCapture dataset along with its restructured variant GazeCapture* for the proposed experiments in the paper. If you are interested in the original iTracker paper and GazeCapture dataset please refer to their paper and project website as mentioned above.

We propose a series of incremental enhancements over the iTracker architecture to reduce the overall RMSError. We do not employ any calibration or device-specific fine-tuning during evaluation. Data Augmentation is only used during the training phase and not for test phase.
As a part of our experiments, we relax the uniform data distribution constraint in the GazeCapture dataset by allowing subjects who did not look at the full set of points for evaluation and explore its impact on performance.



## How to use:

### Dataset preparation

1. Download the GazeCapture dataset from http://gazecapture.csail.mit.edu/download.php
2. Extract the files (including the sub-archives) to a folder S (Source folder). The resulting structure should be something like this:
```
Source
\--00002
    \--frames
    \--appleFace.json
    \--appleLeftEye.json
    \--appleRightEye.json
    \--dotInfo.json
    \--faceGrid.json
    \--frames.json
    \--info.json
    \--motion.json
    \--screen.json
\--00003
...
```

3. Run ROI Detection Task to generate metadata:
```
python taskCreator.py --task 'ROIDetectionTask' --input <SourcePath> --output <MetadataPath> --type <DetectionType>
```
Valid DetectionTypes are circa, dlib and rc. For the circa detection, this step won't perform any action and use the original landmarks, therefore you can skip this step too. The resulting structure should be something like this:
```
Metadata
\--00002
    \--dlibFace.json
    \--dlibLeftEye.json
    \--dlibRightEye.json
\--00003
...
```

4. Run ROI Extraction Task to generate processed dataset:
```
python taskCreator.py --task 'ROIExtractionTask' --input <SourcePath> --metapath <MetadataPath> --output <DestinationPath> --type <DetectionType>
```
Valid DetectionTypes are circa, dlib and rc. For the circa detection, you should skip the metapath field. The resulting structure should be something like this:

```
Destination
\---00002
    \---appleFace
        \---00000.jpg
        ...
    \---appleLeftEye
        ...
    \---appleRightEye
        ...
\---00003
...
\---metadata.mat
```

### Training
```
python main.py --data_path [D: Destination Path] --reset
```

### Validation
```
python main.py --data_path [D: Destination Path] --validate
```

### Testing
```
python main.py --data_path [D: Destination Path] --test
```

### Arguments
```
Frequently used args:
--local_rank  : gpu id {0 to max_gpus-1}
--batch_size  : batch size (e.g. 64, 100, 128, 256)
--data_path   : directory path to the data (i.e. SourcePath)
--base_lr     : lower bound on learning rate (e.g. 1E-7)
--max_lr      : upper bound on learning rate (e.g. 1E-2)
--reset       : starts from a new untrained model

Optional args:
--color_space : colorspace (e.g. 'RGB, 'YCbCr')
--epochs     : maximum number of training epochs (e.g. 30)
--name       : experiment tag (e.g. "resnet-rc")
--visdom     : visdom server path (e.g. "self", "127.0.0.1")
--force_test : run test after train and validation cycles
```

## Dependencies & Dockerization
## Code

Code dependencies are listed in requirements.txt. CUDA 10.0 and Python 3.7+ with following packages (exact version may not be necessary):

* torch (1.6.0)
* torchvision (0.7.0)
* scipy (1.5.2)
* jsonpatch (1.26)
* visdom (0.1.8.9)
* torchtoolbox (0.1.5)
* matplotlib (3.3.2)
* imutils (0.5.3)
* scikit-image (0.17.2)
* nvidia-dali-cuda10.0
* opencv-python-headless (4.4.0.44)
* cmake (3.18.2)
* dlib (19.19.0)
* onnxruntime (1.5.1)
* docker (4.3.1)
* pandas (1.1.3)
* screeninfo (0.6.5)
* numpy (1.19.2)
* pillow (8.0.0)
* torchfile (0.1.0)


If you prefer running the experiments in a sandbox environment, you could use the Dockerfile to build an image and then run experiments in a container. Dockefile uses the dependencies listed in requirements.txt. To build an image please run:
```
sudo docker build -t gazepy .
```

Once the image is built successfully, go inside the pytorch directory and use the following syntax:

**sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v \$(pwd):\$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy** `main.py --data_path [Source Path] --reset`


## History
Any necessary changes to the dataset will be documented here.

* **October 2020**: ArXiv announcement of “Towards Hardware-Agnostic Gaze-Trackers” and release of updated instructions.
* **January 2019**: A dataset preprocessing code for an easier deployment. A conversion to pytorch 0.4.1.
* **March 2018**: Original code release.

## Terms
Usage of this dataset (including all data, models, and code) is subject to the associated license, found in [LICENSE.md](LICENSE.md). The license permits the use of released code, dataset and models for research purposes only.

We also ask that you cite the associated paper if you make use of the GazeCapture dataset; following is the BibTeX entry:

```
@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```

If you use the Enhanced iTracker architecture or any other work presented in the paper [“Towards Hardware-Agnostic Gaze-Trackers”](https://arxiv.org/abs/2010.05123) then please cite the paper following the BibTeX entry:

```
@misc{sharma2020,
      title={Towards Hardware-Agnostic Gaze-Trackers},
      author={Jatin Sharma and Jon Campbell and Pete Ansell and Jay Beavers and Christopher O'Dowd},
      year={2020},
      eprint={2010.05123},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url = {https://arxiv.org/abs/2010.05123}
}
```


## Contact

Please email any questions or comments concerning the paper [“Towards Hardware-Agnostic Gaze-Trackers”](https://arxiv.org/abs/2010.05123) to [the authors](mailto:jatin.sharma@microsoft.com).
