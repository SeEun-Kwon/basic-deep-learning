import torch
import torch.nn as nn

# class BasicBlock(nn.Module):
#     def __init__(self, c_in, c_out, downsample=False):
#         super(BasicBlock, self).__init__()
#         if downsample == True:
#             self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=2, padding=1)
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0),
#                 nn.BatchNorm2d(c_out)
#             )
#         else:
#             self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)
#             self.shortcut = nn.Identity()
#         self.bn1 = nn.BatchNorm2d(c_in)
#         self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(c_out)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(x))
#         out = self.shortcut(x) + out
#         out = self.relu(out)
#         return out

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(Bottleneck, self).__init__()
        self.c_mid = c_out // 4
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
        else:
            self.shortcut = nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, self.c_mid, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(self.c_mid),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.c_mid, self.c_mid, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.c_mid),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.c_mid, c_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c_out)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 256, 1),
            self.make_layer(256, num_blocks[0] - 1)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 512, 2),
            self.make_layer(512, num_blocks[1] - 1)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 1024, 2),
            self.make_layer(1024, num_blocks[2] - 1)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 2048, 2),
            self.make_layer(2048, num_blocks[3] - 1)
        )
        self.classifying = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 200)
        )
    def make_layer(self, c, num_block):
        blocks = []
        for i in range(num_block - 1):
            blocks.append(Bottleneck(c, c, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifying(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])
