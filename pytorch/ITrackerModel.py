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
    # output = (input-k+2p)/s + 1
    # ZeroPad = (k-1)/2
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
            #
            # Note that per https://www.learnopencv.com/understanding-alexnet/, perhaps we should feed in 227x227 images
            # Which would make the math work out better
            # 3C x 224H x 224W
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            # (224 - 11) / 4 + 1 ~= 54
            # 96C x 54H x 54W
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (54 - 3) / 2 + 1 ~= 26
            # 96C x 26H x 26W
            nn.ReLU(inplace=True),

            # CONV-2
            # The second convolutional layer takes as input the (response-normalized and pooled) output of the first
            # convolutional layer and filters it with 256 kernels of size 5×5×48.
            # 96C x 26H x 26W
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.1),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            # (26 + 2 * 2 - 5) / 1 + 1 = 26
            # 256C x 26H x 26W
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (26 - 3) / 2 + 1 ~= 12
            # 256C x 12H x 12W
            nn.ReLU(inplace=True),

            # CONV-3
            # The third and fourth convolutional layers are connected to one another without any intervening pooling or
            # normalization layers. The third convolutional layer has 384 kernels of size 3×3×256 connected to the
            # (normalized, pooled) outputs of the second convolutional layer.
            # 256C x 12H x 12W
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            # (12 + 2 * 1 - 3) / 1 + 1 ~= 12
            # 384C x 12H x 12W
            nn.ReLU(inplace=True),

            # CONV-4
            # The fourth convolutional layer has 384 kernels of size 1×1×64. This layer is differs from the AlexNet
            # paper (which an additional 5th layer, where layers 4 and 5 were 3x3x192)
            # 384C x 12H x 12W
            nn.BatchNorm2d(384),
            nn.Dropout2d(0.1),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            # (12 + 2 * 1 - 3) / 1 + 1 ~= 12
            # 64C x 12H x 12W
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        # 64C x 12H x 12W
        x = x.view(x.size(0), -1)
        # 9216 (64x12x12)
        return x

class FaceImageModel(nn.Module):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            # FC-F1
            # 9216 (64x12x12)
            nn.Dropout(0.1),
            nn.Linear(12 * 12 * 64, 128),
            # 128
            nn.ReLU(inplace=True),

            # FC-F2
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            # 64
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # 3C x 224H x 224W
        x = self.conv(x)
        # 9216 (64x12x12)
        x = self.fc(x)
        # 64
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            # FC-FG1
            # 625 (25x25)
            nn.Linear(gridSize * gridSize, 256),
            # 256
            nn.ReLU(inplace=True),

            # FC-FG2
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            # 128
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 25x25
        x = x.view(x.size(0), -1)
        # 128
        x = self.fc(x)
        # 128
        return x


class ITrackerModel(nn.Module):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        # 3Cx224Hx224W --> 9216 (64x12x12)
        self.eyeModel = ItrackerImageModel()
        # 3Cx224Hx224W --> 64
        self.faceModel = FaceImageModel()
        # 1Cx25Hx25W --> 128
        self.gridModel = FaceGridModel()

        # Joining both eyes
        self.eyesFC = nn.Sequential(
            # FC-E1
            nn.Dropout(0.1),
            # 18,432‬ (64x12x12)*2
            nn.Linear(2 * 12 * 12 * 64, 128),
            # 128
            nn.ReLU(inplace=True),
        )

        # Joining everything
        self.fc = nn.Sequential(
            # FC1
            nn.Dropout(0.1),
            # 384 FC-E1 (128) + FC-F2(64) + FC-FG2(128)
            nn.Linear(128 + 64 + 128, 128),
            # 128
            nn.ReLU(inplace=True),

            # FC2
            nn.Dropout(0.1),
            # 128
            nn.Linear(128, 2),
            # 2
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
