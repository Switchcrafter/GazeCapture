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

# class FeaturesNet(nn.Module):
#     def __init__(self):
#         super(FeaturesNet, self).__init__()
#         # 3x(256x256)   | 3x(224x224)
#         self.features = ConvBNReLu(3, 8, kernel_size=3, stride=2, padding=1)
#         # 8x(128x128)   | 3x(112x112)

#     def forward(self, x):
#         x = self.features(x)
#         return x

# class ImportanceNet(nn.Module):
#     def __init__(self):
#         super(ImportanceNet, self).__init__()
#         # 16x(128x128)  | 3x(112x112)
#         self.ir1 = InverseResidual(16, 24, 32)
#         # 32x(64X64)    | 3x(56x56)
#         self.ir2 = InverseResidual(32, 48, 64)
#         # 64x(32x32)    | 3x(28x28)

#     def forward(self, x):
#         x = self.ir1(x)
#         x = self.ir2(x)
#         return x


# class CombinationNet(nn.Module):
#     def __init__(self):
#         super(CombinationNet, self).__init__()

#         # 128x(32x32)   | 128x(28x28)
#         self.ir1 = InverseResidual(128, 32, 64)
#         # 64x(16x16)    | 64x(14x14)
#         self.ir2 = InverseResidual(64, 16, 32)
#         # 32x(8x8)      | 32x(7x7)
#         self.ir3 = InverseResidual(32, 8, 2)
#         # 2x(4x4)       | 2x(3x3)
#         ##### Change kernel_size and stride here per image size ##### 
#         self.avgPool = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
#         # 2x(1x1)       | 2x(1x1)

#     def forward(self, x):
#         x = self.ir1(x)
#         x = self.ir2(x)
#         x = self.ir3(x)
#         x = self.avgPool(x)
#         x = x.view(x.size(0), -1)
#         # 2 (X and Y featureMaps of size 1)
#         return x


# class DeepEyeModel(nn.Module):
#     def __init__(self):
#         super(DeepEyeModel, self).__init__()
#         # 3C x (256H x 256W) --> 8C x (128H x 128W)
#         self.featuresNetGrid = FeaturesNet()
#         self.featuresNetFace = FeaturesNet()
#         self.featuresNetEye = FeaturesNet()

#         # 16C x (128H x 128W) --> 8C x (8H x 8W)
#         self.importanceNetFaceGrid = ImportanceNet()
#         self.importanceNetBothEyes = ImportanceNet()

#         # 16C x (8H x 8W) --> 2
#         self.combinationNet = CombinationNet()


#     def forward(self, faces, eyesLeft, eyesRight, grids):
        
#         # Grid and Face
#         f_grid = self.featuresNetGrid(grids) 
#         f_face = self.featuresNetFace(faces) 
#         f_face_grid = torch.cat((f_grid, f_face), 1)
#         i_face_grid = self.importanceNetFaceGrid(f_face_grid)

#         # Left and Right Eye
#         f_eyeL = self.featuresNetEye(eyesLeft) 
#         f_eyeR = self.featuresNetEye(eyesRight) 
#         f_eyeLR = torch.cat((f_eyeL, f_eyeR), 1)
#         i_eyeLR = self.importanceNetBothEyes(f_eyeLR)

#         # Cat all
#         i_all = torch.cat((i_face_grid, i_eyeLR), 1)
#         x = self.combinationNet(i_all)
#         return x


############################## ResNet ###############################

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
    layers.append(block(in_planes, planes, stride, downsample, dropout))
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




############################## ResNet ###############################

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, out_planes, stride=1, downsample=None, 
#                     groups=1, base_width=64, dilation=1):
#         super(BasicBlock, self).__init__()
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(in_planes, out_planes, stride)
#         self.bn1 = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = conv3x3(out_planes, out_planes)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = out + identity
#         out = self.relu(out)
#         return out

# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#     expansion = 4
#     def __init__(self, in_planes, out_planes, stride=1, downsample=None, 
#                     groups=1, base_width=64, dilation=1):
#         super(Bottleneck, self).__init__()

#         width = int(out_planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(in_planes, width)
#         self.bn1 = nn.BatchNorm2d(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = nn.BatchNorm2d(width)
#         self.conv3 = conv1x1(width, out_planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None):
#         super(ResNet, self).__init__()

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation)
#                         )

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

#     def forward(self, x):
#         return self._forward_impl(x)


# def _resnet(arch, block, layers, num_classes, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, num_classes, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

# def resnet18(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 2, pretrained, progress,
#                    **kwargs)


# def resnet34(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], 2, pretrained, progress,
#                    **kwargs)


# def resnet50(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], 2, pretrained, progress,
#                    **kwargs)

# def resnetDeepEye(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnetDeepEye', Bottleneck, [1, 2, 1, 2], 2, pretrained, progress,
#                    **kwargs)