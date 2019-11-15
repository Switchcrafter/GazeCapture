import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

'''
Pytorch model for the iTracker.

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


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unique weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            # The shape of the layers below is heavily influenced by AlexNet, discussed in the paper
            # "ImageNet Classification with Deep Convolutional Neural Networks"
            # https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
            # The comments for the convolutional layers below are based on the descriptions from the AlexNet paper,
            # with adjustments based on the "Eye Gaze for Everyone" paper.
            # https://people.csail.mit.edu/khosla/papers/cvpr2016_Khosla.pdf

            # CONV-1
            # The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a
            # stride of 4 pixels (this is the distance between the receptive field centers of neighboring neurons in a
            # kernel map).
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Added based on best practices
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),   # should be CrossMapLRN2d, but swapping for
                                                                            # LocalResponseNorm for ONNX export

            # CONV-2
            # The second convolutional layer takes as input the (response-normalized and pooled) output of the first
            # convolutional layer and filters it with 256 kernels of size 5×5×48.
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),                                                # Added based on best practices
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),   # should be CrossMapLRN2d, but swapping for
                                                                            # LocalResponseNorm for ONNX export

            # CONV-3
            # The third and fourth convolutional layers are connected to one another without any intervening pooling or
            # normalization layers. The third convolutional layer has 384 kernels of size 3×3×256 connected to the
            # (normalized, pooled) outputs of the second convolutional layer.
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),                                                # Added based on best practices


            # CONV-4
            # The fourth convolutional layer has 384 kernels of size 1×1×64. This layer is differs from the AlexNet
            # paper (which an additional 5th layer, where layers 4 and 5 were 3x3x192)
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),                                                # Added based on best practices
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            # FC-F1
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            # FC-F2
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            # FC-FG1
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            # FC-FG2
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()

        # Joining both eyes
        self.eyesFC = nn.Sequential(
            # FC-E1
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Joining everything
        self.fc = nn.Sequential(
            # FC1
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # FC2
            nn.Linear(128, 2),
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)     # CONV-E1 -> ... -> CONV-E4
        xEyeR = self.eyeModel(eyesRight)    # CONV-E1 -> ... -> CONV-E4

        # Cat Eyes and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)          # FC-E1

        # Face net
        xFace = self.faceModel(faces)       # CONV-F1 -> ... -> CONV-E4 -> FC-F1 -> FC-F2
        xGrid = self.gridModel(faceGrids)   # FC-FG1 -> FC-FG2

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)                      # FC1 -> FC2

        return x
