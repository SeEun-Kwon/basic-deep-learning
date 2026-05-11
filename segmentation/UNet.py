import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, padding=1):
        super(ConvBnReLU, self).__init__()
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.convbnrelu(x)
        return x

class ConvDown(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, padding=1):
        super(ConvDown, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel, padding)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel, padding)

        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.avgpool(x)
        return x, y

class ConvUp(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, padding=1):
        super(ConvUp, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel, padding)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel, padding)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = F.interpolate(x, size=(2 * x.shape[2], 2 * x.shape[3]), mode='bilinear')
        x = torch.cat([x, y], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.downsampling = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.prediction = nn.Conv2d(64, 21, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        connection1 = x               # size=256, ch=64
        x = self.conv2(self.downsampling(x))
        connection2 = x               # size=128, ch=128
        x = self.conv3(self.downsampling(x))
        connection3 = x               # size=64, ch=256
        x = self.conv4(self.downsampling(x))
        connection4 = x               # size=32, ch=512
        x = self.conv5(self.downsampling(x))
        connection5 = x               # size=16, ch=512
        x = self.conv6(self.downsampling(x))    # size=16, ch=512

        x = F.interpolate(x, (16, 16), mode='bilinear')
        x = self.conv7(torch.cat([x, connection5], dim=1))
        x = F.interpolate(x, (32, 32), mode='bilinear')
        x = self.conv8(torch.cat([x, connection4], dim=1))
        x = F.interpolate(x, (64, 64), mode='bilinear')
        x = self.conv9(torch.cat([x, connection3], dim=1))
        x = F.interpolate(x, (128, 128), mode='bilinear')
        x = self.conv10(torch.cat([x, connection2], dim=1))
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.conv11(torch.cat([x, connection1], dim=1))
        x = self.prediction(x)

        return x
