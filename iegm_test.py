import torch
from torch import nn
import binarized_modules

from compile_utils import compile_conv_block, compile_fc_block, convert_fc_act, convert_conv_act

torch.manual_seed(0)

class IEGMNetXNOR(nn.Module):
    def __init__(self, in_bw=1, out_bw=1, weight_bw=1):
        super(IEGMNetXNOR, self).__init__()
        self.conv1 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, binarize_input=False, in_channels=1, out_channels=32, kernel_size=(6, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
            #nn.MaxPool2d((2, 1), stride=(2, 1))
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
        x = self.pool(x)
        x = self.conv2(x)
        #x = self.pool(x)
        x = self.conv3(x)
        #x = self.pool(x)
        x = self.conv4(x)
        #x = self.pool(x)
        x = self.conv5(x)
        #x = self.pool(x)
        x = x.view(-1,1120)

        fc1_output = self.fc1(x)
        fc2_output = self.fc2(fc1_output)
        return fc2_output

net = IEGMNetXNOR()
net.eval()

def randomize_bn_layer(bn_layer):
    bn_layer.running_mean.copy_(torch.rand(bn_layer.running_mean.shape)*2)
    with torch.no_grad():
        bn_layer.weight.data.copy_(torch.rand(bn_layer.weight.data.shape)-0.5)

dummy_input = torch.round(torch.rand((1, 1, 1248, 1))*255)
torch.onnx.export(net, dummy_input, "/home/ravit/Documents/NanoCAD/TinyMLContest/tinyml_contest2022_demo_example/toy.onnx")

torch.manual_seed(0)
x = torch.round(torch.rand((1, 1, 1248, 1))*255)
#x = torch.ones((1, 1, 1248, 1))
conv2_in = net.pool(net.conv1(x))
conv3_in = net.conv2(conv2_in)
conv4_in = net.conv3(conv3_in)
conv5_in = net.conv4(conv4_in)
fc1_in = net.conv5(conv5_in).permute((0, 2, 1, 3)).reshape((conv5_in.shape[0], -1))
fc2_in = net.fc1(fc1_in)
y = net.fc2(fc2_in)

print(x.flatten().tolist())
print(y)

#compile_conv_block(net.conv1, x)
#compile_conv_block(net.conv2, conv2_in)
#compile_conv_block(net.conv3, conv3_in)
#compile_conv_block(net.conv4, conv4_in)
#compile_conv_block(net.conv5, conv5_in)
#compile_fc_block(net.fc1, fc1_in)
#compile_fc_block(net.fc2, fc2_in, binarize_output=False)