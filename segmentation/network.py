import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model = models.vgg16_bn(weights='DEFAULT')

#VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (9): ReLU(inplace=True)
#     (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (12): ReLU(inplace=True)
#     (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (16): ReLU(inplace=True)
#     (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (19): ReLU(inplace=True)
#     (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (26): ReLU(inplace=True)
#     (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (29): ReLU(inplace=True)
#     (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (32): ReLU(inplace=True)
#     (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (36): ReLU(inplace=True)
#     (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (39): ReLU(inplace=True)
#     (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (42): ReLU(inplace=True)
#     (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )

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
