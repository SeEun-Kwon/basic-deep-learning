import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model = models.vgg16_bn(weights='DEFAULT')

class FCN_8(nn.Module):
    def __init__(self):
        super(FCN_8, self).__init__()
        # feature extractor
        self.conv123 = model.features[:24]      # (conv,conv,max) x 2, (conv,conv,conv,max) -> 64, 128, 256
        self.conv4 = model.features[24:34]      # conv,conv,conv,max    -> 512
        self.conv5 = model.features[34:44]      # conv,conv,conv,max    -> 512

        self.conv123_1x1 = nn.Conv2d(256, 21, kernel_size=1)
        self.conv4_1x1 = nn.Conv2d(512, 21, kernel_size=1)

        # classifier to conv layer
        self.conv6_1 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv6_3 = nn.Conv2d(2048, 21, kernel_size=1, stride=1, padding=0)
        self.bn6_1 = nn.BatchNorm2d(2048)
        self.bn6_2 = nn.BatchNorm2d(2048)
        self.bn6_3 = nn.BatchNorm2d(21)
        self.relu6_1 = nn.ReLU()
        self.relu6_2 = nn.ReLU()

        # upsampling
        self.upsample = nn.ConvTranspose2d(21, 21, kernel_size=4, stride=2, padding=1)

        self.dropout = nn.Dropout()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # feature extractor
        x = self.conv123(x)         # 256 -> 32
        x1 = self.conv123_1x1(x)
        x = self.conv4(x)           # 32 -> 16
        x2 = self.conv4_1x1(x)
        x = self.conv5(x)           # 16 -> 8

        # classifier
        x = self.dropout(self.relu6_1(self.bn6_1(self.conv6_1(x))))
        x = self.dropout(self.relu6_2(self.bn6_2(self.conv6_2(x))))
        x = self.conv6_3(x)

        # upsampling
        x = x2 + self.upsample(x)       # 8 -> 16
        x = x1 + self.upsample(x)       # 16 -> 32
        x = F.interpolate(x, size=(256, 256), mode='bilinear')  # 32 -> 256

        return x
