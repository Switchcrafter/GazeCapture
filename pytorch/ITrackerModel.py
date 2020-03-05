import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import models

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
        self.model = models.resnet18(pretrained=True)
        self.conv = nn.Sequential(*list(self.model.children())[:-2])

        # TODO Try fine tuning using RGB color space rather than YCbCr
        #      Fine tuning might be more successful in the same color space
        #      A large error from color space issues is a reasonable outcome
        # # Freeze the parameters
        # for param in self.conv.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # 25088 (512×7×7)
        return x


class FaceImageModel(nn.Module):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            # FC-F1
            # 25088
            nn.Dropout(0.1),
            nn.Linear(25088, 128),
            # 128
            nn.ReLU(inplace=True),

            # FC-F2
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            # 64
            nn.ReLU(inplace=True),
            # 64
        )
        
    def forward(self, x):
        # 3C x 224H x 224W
        x = self.conv(x)
        # 25088
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
            # 256

            # FC-FG2
            # 256
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
        # 3Cx224Hx224W --> 64
        self.faceModel = FaceImageModel()
        # 1Cx25Hx25W --> 128
        self.gridModel = FaceGridModel()

        # Joining everything
        self.fc = nn.Sequential(
            # FC1
            nn.Dropout(0.1),
            # 192 FC-F2(64) + FC-FG2(128)
            nn.Linear(64 + 128, 128),
            # 128
            nn.ReLU(inplace=True),
            # 128

            # FC2
            # 128
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            # 2
        )

    def forward(self, faces, faceGrids):
        # Face net
        xFace = self.faceModel(faces)       # CONV-F1 -> ... -> CONV-E4 -> FC-F1 -> FC-F2
        xGrid = self.gridModel(faceGrids)   # FC-FG1 -> FC-FG2

        # Cat all
        x = torch.cat((xFace, xGrid), 1)
        x = self.fc(x)                      # FC1 -> FC2

        return x
