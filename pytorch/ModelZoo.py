import torch
import torch.nn as nn

# def ConvBNReLu(in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1):
#     """Convolution, BatchNorm, ReLU"""
#     convBNReLu = nn.Sequential(
#                     nn.Dropout2d(0.1),
#                     nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
#                     nn.BatchNorm2d(out_planes),
#                     nn.ReLU6(inplace=True),
#                 )
#     return convBNReLu

# def InverseResidual(in_planes, mid_planes, out_planes):
#     """Inverse Residual Layer"""
#     inverseResidual = nn.Sequential(
#                     ConvBNReLu(in_planes, mid_planes, kernel_size=1, stride=1, padding=0),
#                     ConvBNReLu(mid_planes, mid_planes, kernel_size=3, stride=2, padding=1, groups=mid_planes),
#                     nn.Dropout2d(0.1),
#                     nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0),
#                     nn.BatchNorm2d(out_planes)
#                 )
#     return inverseResidual

def DCB(in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, dropout=0.1):
    """Dropout, Convolution, BatchNorm"""
    dcb = nn.Sequential(
                    nn.Dropout2d(dropout),
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
                    nn.BatchNorm2d(out_planes),
                )
    return dcb
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, 
                    groups=1, base_width=64, dilation=1, dropout=0.1):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dcb1 = DCB(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, dropout=dropout)
        self.dcb2 = DCB(out_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=groups, dropout=dropout)
        if type(downsample) == str and downsample == 'nin':
            self.downsample = DCB(in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=groups, dropout=dropout)
            # self.downsample = DCB(in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=in_planes, dropout=dropout)
        else:
            self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.dcb1(x)
        out = self.relu(out)
        out = self.dcb2(out)

        if self.downsample:
            out = out + self.downsample(x)
        else:
            out = out + x 

        out = self.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, 
                    groups=1, base_width=64, dilation=1, dropout=0.1):
        super(BottleneckBlock, self).__init__()
        mid_planes = int(out_planes * (base_width / 64.)) * groups
        self.relu = nn.ReLU(inplace=True)
        self.dcb1 = DCB(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, dropout=dropout)
        self.dcb2 = DCB(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=groups, dropout=dropout)
        self.dcb3 = DCB(mid_planes, out_planes * self.expansion, kernel_size=1, stride=1, padding=0, dropout=dropout)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.dcb1(x)
        out = self.relu(out)
        out = self.dcb2(out)
        out = self.relu(out)
        out = self.dcb3(out)

        if self.downsample:
            out = out + self.downsample(x)
        else:
            out = out + x 

        out = self.relu(out)
        return out

class FeaturesNet(nn.Module):
    def __init__(self):
        super(FeaturesNet, self).__init__()
        in_planes, out_planes = 3, 32
        # 3x(256x256)   | 3x(224x224)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        # 32x(128x128)   | 8x(112x112)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 32x(64X64)    | 8x(56x56)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

def DCB(in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, dropout=0.1):
    """Dropout, Convolution, BatchNorm"""
    dcb = nn.Sequential(
                    nn.Dropout2d(dropout),
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
                    nn.BatchNorm2d(out_planes),
                )
    return dcb

# for every ResNet Layer - first conv doubles the channels and halves the size
# this function controls the dropout values in the entire network
def make_layer(block, in_planes, planes, num_blocks, stride=1, downsample=None, dropout=0.1):
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    layers.append(block(in_planes, planes, stride, downsample, dropout=dropout))
    for _ in range(1, num_blocks):
        layers.append(block(planes, planes))
    return nn.Sequential(*layers)


class ImportanceNet(nn.Module):
    def __init__(self, num_blocks):
        super(ImportanceNet, self).__init__()
        block = BasicBlock
        # block = BottleneckBlock
        # 128x(56x56)   | 16x(56x56)
        self.layer1 = make_layer(block, 32+32, 32, num_blocks[0], stride=1)
        # 128x(56x56)   | 16x(56x56)
        self.layer2 = make_layer(block, 32, 64, num_blocks[1], stride=2)
        # 32x(28x28)    | 32x(28x28)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class CombinationNet(nn.Module):
    def __init__(self, num_blocks):
        super(CombinationNet, self).__init__()
        block = BasicBlock
        # 64x(16x16)    | 32x(28x28)
        self.layer3 = make_layer(block, 64+64, 128, num_blocks[2], stride=2)
        # 32x(8x8)      | 64x(14x14)
        self.layer4 = make_layer(block, 128, 256, num_blocks[3], stride=2)
        # 2x(4x4)       | 128x(7x7)
        # self.avgPool = nn.AdaptiveAvgPool2d((7,7))
        self.avgPool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Linear(256 * block.expansion, 2)
        # self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgPool(x)
        # x = self.flatten(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # x = x.view(x.size(0), -1)
        # 2 (X and Y featureMaps of size 1)
        return x


class DeepEyeModel(nn.Module):
    def __init__(self):
        super(DeepEyeModel, self).__init__()

        num_blocks = [1,2,1,2]
        # 3C x (256H x 256W) --> 8C x (128H x 128W)
        self.featuresNetGrid = FeaturesNet()
        self.featuresNetFace = FeaturesNet()
        self.featuresNetEye = FeaturesNet()

        # 16C x (128H x 128W) --> 8C x (8H x 8W)
        self.importanceNetFaceGrid = ImportanceNet(num_blocks)
        self.importanceNetBothEyes = ImportanceNet(num_blocks)

        # 16C x (8H x 8W) --> 2
        self.combinationNet = CombinationNet(num_blocks)


    def forward(self, faces, eyesLeft, eyesRight, grids):
        
        # Grid and Face
        f_grid = self.featuresNetGrid(grids) 
        f_face = self.featuresNetFace(faces) 
        f_face_grid = torch.cat((f_grid, f_face), 1)
        # f_face_grid = f_grid
        i_face_grid = self.importanceNetFaceGrid(f_face_grid)

        # Left and Right Eye
        f_eyeL = self.featuresNetEye(eyesLeft) 
        f_eyeR = self.featuresNetEye(eyesRight) 
        f_eyeLR = torch.cat((f_eyeL, f_eyeR), 1)
        # f_eyeLR = f_eyeL
        i_eyeLR = self.importanceNetBothEyes(f_eyeLR)

        # Cat all
        i_all = torch.cat((i_face_grid, i_eyeLR), 1)
        # i_all = i_face_grid
        x = self.combinationNet(i_all)
        return x
