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


## Dockerization

Alternatively, you can use the [gazepy docker image]() to run your experiments in a sandbox environment. In this case, please go inside the pytorch directory inside the repo and use the following syntax -

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

## Code

Requires CUDA and Python 3.6+ with following packages (exact version may not be necessary):

* numpy (1.15.4)
* Pillow (5.3.0)
* torch (0.4.1)
* torchfile (0.1.0)
* torchvision (0.2.1)
* scipy (1.1.0)


## Contact

Please email any questions or comments concerning the paper [“Towards Hardware-Agnostic Gaze-Trackers”](https://arxiv.org/abs/2010.05123) to [the authors](mailto:jatin.sharma@microsoft.com).
