import torch.nn as nn
import torch.nn.functional as F

import binarized_modules

class CNN_medium(nn.Module):

    def __init__(self, full=False, binary=True, conv_thres=0.7, fc_thres=0.9, align=False, pad=0):
        super(CNN_medium, self).__init__()

        self.pruned = False
        self.full = full
        self.binary = binary
        self.pad = pad

        self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.htanh1 = nn.Hardtanh(inplace=True)

        if full:
            self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.fc1 = nn.Linear(512*4*4, 10, bias=False)
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv2 = binarized_modules.BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv3 = binarized_modules.BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv4 = binarized_modules.BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv5 = binarized_modules.BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv6 = binarized_modules.BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv7 = binarized_modules.BinarizeConv2d(512, 10, kernel_size=4, padding=0, bias=False)
            self.fc1 = binarized_modules.BinarizeLinear(1024, 1024, bias=False)
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 128, 128, kernel_size=3, padding=0, bias=False, align=align)
            self.conv3 = binarized_modules.TernarizeConv2d(conv_thres, 128, 256, kernel_size=3, padding=0, bias=False, align=align)
            self.conv4 = binarized_modules.TernarizeConv2d(conv_thres, 256, 256, kernel_size=3, padding=0, bias=False, align=align)
            self.conv5 = binarized_modules.TernarizeConv2d(conv_thres, 256, 512, kernel_size=3, padding=0, bias=False, align=align)
            self.conv6 = binarized_modules.TernarizeConv2d(conv_thres, 512, 512, kernel_size=3, padding=0, bias=False, align=align)
            self.conv7 = binarized_modules.TernarizeConv2d(0.49, 512, 10, kernel_size=4, padding=0, bias=False, align=align)
            self.fc1 = binarized_modules.BinarizeLinear(1024, 1024, bias=False)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.htanh2 = nn.Hardtanh(inplace=True)


        self.bn3 = nn.BatchNorm2d(256)
        self.htanh3 = nn.Hardtanh(inplace=True)


        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.htanh4 = nn.Hardtanh(inplace=True)


        self.bn5 = nn.BatchNorm2d(512)
        self.htanh5 = nn.Hardtanh(inplace=True)


        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.htanh6 = nn.Hardtanh(inplace=True)


        self.bnfc1 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.bn1(x)

            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.bn2(x)

            x = F.relu(self.conv3(x))
            x = self.bn3(x)

            x = F.relu(self.conv4(x))
            x = self.pool4(x)
            x = self.bn4(x)

            x = F.relu(self.conv5(x))
            x = self.bn5(x)

            x = F.relu(self.conv6(x))
            x = self.pool6(x)
            x = self.bn6(x)

            x = x.view(-1, 512*4*4)
            x = F.relu(self.fc1(x))
            self.fc1_result = x.data.clone()
        else:
            x = F.pad(x, (1,1,1,1), value=self.pad)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.htanh1(x)

            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv2(x)
            else:
                x = self.conv2(x, self.pruned)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.htanh2(x)

            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv3(x)
            else:
                x = self.conv3(x, self.pruned)
            x = self.bn3(x)
            x = self.htanh3(x)

            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv4(x)
            else:
                x = self.conv4(x, self.pruned)
            x = self.pool4(x)
            x = self.bn4(x)
            x = self.htanh4(x)

            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv5(x)
            else:
                x = self.conv5(x, self.pruned)
            x = self.bn5(x)
            x = self.htanh5(x)

            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv6(x)
            else:
                x = self.conv6(x, self.pruned)
            x = self.pool6(x)
            x = self.bn6(x)
            x = self.htanh6(x)

            if self.binary:
                x = self.conv7(x)
            else:
                x = self.conv7(x, self.pruned)
            x = x.view(-1, 10)
            x = self.bnfc1(x)
        return self.logsoftmax(x)
