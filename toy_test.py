import torch
from torch import nn
import binarized_modules

from compile_utils import compile_conv_block, compile_fc_block

torch.manual_seed(0)

class ToyModel(nn.Module):
    def __init__(self, in_bw=1, out_bw=1, weight_bw=1):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, binarize_input=False, in_channels=1, out_channels=32, kernel_size=(3, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )
        self.conv2 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )
        self.fc1 = nn.Sequential(
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=32, out_features=32, bias=False),
            nn.BatchNorm1d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.Hardtanh(inplace=True),
        )
        self.fc2 = nn.Sequential(
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=32, out_features=10, bias=False),
            nn.BatchNorm1d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((-1, 32))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = ToyModel()
net.eval()

def randomize_bn_layer(bn_layer):
    bn_layer.running_mean.copy_(torch.rand(bn_layer.running_mean.shape)*2)
    with torch.no_grad():
        bn_layer.weight.data.copy_(torch.rand(bn_layer.weight.data.shape)-0.5)

dummy_input = torch.round(torch.rand((1, 1, 5, 1))*5)
torch.onnx.export(net, dummy_input, "/home/ravit/Documents/NanoCAD/TinyMLContest/tinyml_contest2022_demo_example/toy.onnx")

#x = torch.round(torch.rand((1, 1, 5, 1))*5)
x = torch.reshape(torch.tensor([1, 2, 3, 4, 5.]), (1, 1, 5, 1))
conv2_in = net.conv1(x)
fc1_in = net.conv2(conv2_in).view((conv2_in.shape[0], -1))
fc2_in = net.fc1(fc1_in)
y = net.fc2(fc2_in)

#compile_conv_block(net.conv1, x)
#compile_conv_block(net.conv2, conv2_in)
#compile_fc_block(net.fc1, fc1_in)
print("FC2 start")
compile_fc_block(net.fc2, fc2_in, binarize_output=False)