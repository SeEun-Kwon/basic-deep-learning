import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        self.resnet = pretrained_dilres()
        self.aux_cls = nn.Sequential(
            nn.Conv2d(1024, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 21, kernel_size=1, stride=1, padding=0)
        )

        self.psp = PSPmodule()
        self.cls = nn.Sequential(
            nn.Conv2d(4096, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 21, kernel_size=1, stride=1, padding=0))
        self.up = nn.Upsample(scale_factor=8, mode='bilinear')


    def forward(self, x, AL):
        if AL:
            x, aux = self.resnet(x, AL=True)
            aux = self.aux_cls(aux)
            aux = self.up(aux)
            x = self.psp(x)
            x = self.cls(x)
            x = self.up(x)
            return x, aux
        else:
            x = self.resnet(x, AL=False)
            x = self.psp(x)
            x = self.cls(x)
            x = self.up(x)
            return x

class PSPmodule(nn.Module):
    def __init__(self):
        super(PSPmodule, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool4 = nn.AdaptiveAvgPool2d((6, 6))

        self.cbr1 = cbr(2048, 512, 1, 1, 0)
        self.cbr2 = cbr(2048, 512, 1, 1, 0)
        self.cbr3 = cbr(2048, 512, 1, 1, 0)
        self.cbr4 = cbr(2048, 512, 1, 1, 0)

        self.up = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, x):
        up1 = self.up(self.cbr1(self.pool1(x)))
        up2 = self.up(self.cbr2(self.pool2(x)))
        up3 = self.up(self.cbr3(self.pool3(x)))
        up4 = self.up(self.cbr4(self.pool4(x)))
        x = torch.cat((up1, up2, up3, up4, x), dim=1)
        return x

class pretrained_dilres(nn.Module):
    def __init__(self):
        super(pretrained_dilres, self).__init__()

        self.head = nn.ModuleList(models.resnet152(weights='IMAGENET1K_V2').children())[:6]
        self.head = nn.Sequential(*self.head)

        self.layer3 = nn.ModuleList(models.resnet152(weights='IMAGENET1K_V2').children())[6]
        self.layer3[0].downsample[0].stride = (1, 1)
        for m in self.layer3:
            for l in m.children():
                if isinstance(l, nn.Conv2d):
                    l.stride = (1, 1)
                    if l.kernel_size == (3, 3):
                        l.dilation = (2, 2)
                        l.padding = (2, 2)

        self.layer4 = nn.ModuleList(models.resnet152(weights='IMAGENET1K_V2').children())[7]
        self.layer4[0].downsample[0].stride = (1, 1)
        for m in self.layer4:
            for l in m.children():
                if isinstance(l, nn.Conv2d):
                    l.stride = (1, 1)
                    if l.kernel_size == (3, 3):
                        l.dilation = (4, 4)
                        l.padding = (4, 4)

        self.layer3_1 = self.layer3[:3]
        self.layer3_2 = self.layer3[3:]

        self.layer3_1 = nn.Sequential(*self.layer3_1)
        self.layer3_2 = nn.Sequential(*self.layer3_2)
        self.layer4 = nn.Sequential(*self.layer4)

    def forward(self, x, AL):
        x = self.head(x)
        if AL:
            x = checkpoint(self.layer3_1, x)
            aux = x
            x = self.layer3_2(x)
            x = self.layer4(x)
            return x, aux
        else:
            x = checkpoint(self.layer3_1, x)
            x = self.layer3_2(x)
            x = self.layer4(x)
            return x

class cbr(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        super(cbr, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
