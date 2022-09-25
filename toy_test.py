import torch
from torch import nn
import binarized_modules

torch.manual_seed(0)

class ToyModel(nn.Module):
    def __init__(self, in_bw=1, out_bw=1, weight_bw=1):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, binarize_input=False, in_channels=1, out_channels=32, kernel_size=(3, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            #nn.Hardtanh(inplace=True),
        )
        self.conv2 = nn.Sequential(
            binarized_modules.BinarizeConv2d(in_bw, out_bw, weight_bw, in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            #nn.Hardtanh(inplace=True),
        )
        self.fc1 = nn.Sequential(
            binarized_modules.BinarizeLinear(in_bw, out_bw, weight_bw, in_features=32, out_features=32, bias=False),
            nn.BatchNorm1d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )
    def forward_0(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((x.shape[0], -1))
        x = self.fc1(x)
        return x

    def forward_1(self, x):
        x = self.conv2(x)
        return x

    def forward(self, x):
        return self.forward_0(x)

def pack(arr, pckWdt=32):
    assert len(arr)%pckWdt == 0
    print("pack arr", arr)
    packs = []
    for i in range(int(len(arr)/pckWdt)):
        pack_dec = 0
        for j in range(pckWdt):
            pack_dec += arr[i*pckWdt + j]*pow(2, pckWdt-1-j) if arr[i*pckWdt + j] == 1 else 0
        pack = str(hex(int(pack_dec)))
        packs.append(pack)
    return ", ".join(packs)

def convert_fc_act(x):
    x = x[0].flatten().tolist()
    x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_conv_act(x, binarize=True):
    #x = x[0].permute((0, 2, 1)).flatten().tolist() #CHW -> CWH
    x = x[0].permute((1, 2, 0)).flatten().tolist() #CHW -> HWC
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_conv_weight(w):
    w = w.permute((0, 3, 2, 1)).flatten().tolist() #NCHW -> NWHC
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_bn(mu, sigma, gamma, beta):
    thr = ((beta*sigma / torch.sqrt(torch.pow(gamma, 2) + 1e-4)) + mu).tolist()
    sign = [0 if elem<0 else 1 for elem in (torch.sign(gamma)).tolist()]
    return [e*2 for e in thr], sign


net = ToyModel()
net.eval()
bn_layer = list(net.modules())[3]
bn_layer.running_mean.copy_(torch.rand(bn_layer.running_mean.shape)*2)
with torch.no_grad():
    bn_layer.weight.data.copy_(torch.rand(bn_layer.weight.data.shape)-0.5)

x = torch.round(torch.rand((1, 1, 3, 1))*5)
y = net.forward_1(x)
torch.onnx.export(net, x, "/home/ravit/Documents/NanoCAD/TinyMLContest/tinyml_contest2022_demo_example/toy.onnx")

x_inf = convert_conv_act(x)
conv_layer = list(net.modules())[2]
w_inf = convert_conv_weight(conv_layer.weight)
bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
bn_sign_inf_pack = pack(bn_sign_inf)
y_inf = convert_fc_act(y)

print("x", x_inf)
print("w", w_inf)
print("thr", bn_th_inf)
print("sign", bn_sign_inf_pack, bn_sign_inf)
print("y", y_inf)
