import torch
import torch.nn as nn
import torch.nn.functional as F

import binarized_modules

class IEGMNet(nn.Module):
    def __init__(self):
        super(IEGMNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=740, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1,740)

        fc1_output = F.relu(self.fc1(conv5_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class IEGMNetXNOR(nn.Module):
    def __init__(self, in_bw=1, out_bw=1, weight_bw=1):
        super(IEGMNetXNOR, self).__init__()
        self.conv1 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, binarize_input=False, in_channels=1, out_channels=32, kernel_size=(6, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        self.conv2 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=32, out_channels=32, kernel_size=(5, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        self.conv3 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        self.conv4 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        self.conv5 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        self.fc1 = nn.Sequential(
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=1120, out_features=32, bias=False),
            nn.BatchNorm1d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )
        self.fc2 = nn.Sequential(
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=32, out_features=2, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.pool = nn.MaxPool2d((2, 1), stride=(2, 1))

    def forward(self, x): #(1, 1, 1248, 1)
        x = self.conv1(x)
        #x = self.pool(x)
        x = self.conv2(x)
        #x = self.pool(x)
        x = self.conv3(x)
        #x = self.pool(x)
        x = self.conv4(x)
        #x = self.pool(x)
        x = self.conv5(x).permute((0, 2, 1, 3)).reshape((x.shape[0], -1))
        #x = self.pool(x)

        fc1_output = self.fc1(x)
        fc2_output = self.fc2(fc1_output)
        return fc2_output