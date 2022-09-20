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
    def __init__(self, in_bw=4, out_bw=4, weight_bw=4):
        super(IEGMNetXNOR, self).__init__()
        self.conv1 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )

        self.conv2 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )

        self.conv3 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )

        self.conv4 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )

        self.conv5 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=740, out_features=10, bias=False),
            nn.BatchNorm1d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )
        self.fc2 = nn.Sequential(
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=10, out_features=2, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
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

class IEGMNetXNORNew(nn.Module):
    def __init__(self, in_bw=4, out_bw=4, weight_bw=4):
        super(IEGMNetXNORNew, self).__init__()
        self.conv1 = binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        self.htanh1 = nn.Hardtanh(inplace=True)

        self.conv2 = binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        self.htanh2 = nn.Hardtanh(inplace=True)

        self.conv3 = binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        self.htanh3 = nn.Hardtanh(inplace=True)

        self.conv4 = binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        self.htanh4 = nn.Hardtanh(inplace=True)

        self.conv5 = binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        self.htanh5 = nn.Hardtanh(inplace=True)

        self.dropout1 = nn.Dropout(0.5)

        self.fc1 = binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=740, out_features=10, bias=False)
        self.bn6 = nn.BatchNorm1d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        self.htanh6 = nn.Hardtanh(inplace=True)

        self.fc2 = binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=10, out_features=2, bias=False)
        self.bn7 = nn.BatchNorm1d(2, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.htanh2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.htanh3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.htanh4(x)

        x = self.conv5(x)
