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
        x = x.view((-1, 32))
        x = self.fc1(x)
        return x

    def forward_1(self, x):
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.forward_0(x)

def pack(arr, pckWdt=32):
    assert len(arr)%pckWdt == 0
    packs = []
    for i in range(int(len(arr)/pckWdt)):
        pack_dec = 0
        for j in range(pckWdt):
            pack_dec += arr[i*pckWdt + j]*pow(2, pckWdt-1-j) if arr[i*pckWdt + j] == 1 else 0
        pack = str(hex(int(pack_dec)))
        packs.append(pack)
    return ", ".join(packs)

def convert_fc_act(x, binarize=True):
    x = x[0].flatten().tolist()
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_conv_act(x, binarize=True):
    #x = x[0].permute((0, 2, 1)).flatten().tolist() #CHW -> CWH
    x = x[0].permute((1, 2, 0)) #CHW -> HWC
    x = x.flatten().tolist()
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_fc_weight(w):
    #w = w.permute((1, 0))
    w = w.flatten().tolist()
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_conv_weight(w):
    w = w.permute((0, 3, 2, 1)) #NCHW -> NWHC
    #w = w.permute((0, 2, 3, 1)) #NCHW -> NHWC
    w = w.flatten().tolist()
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_bn(mu, sigma, gamma, beta):
    thr = ((beta*sigma / torch.sqrt(torch.pow(gamma, 2) + 1e-4)) + mu).tolist()
    sign = [0 if elem<0 else 1 for elem in (torch.sign(gamma)).tolist()]
    return [e*2 for e in thr], sign

def convert_bn_float(mu, sigma, gamma, beta):
    return (mu*4).tolist(), (sigma*4).tolist(), gamma.tolist(), beta.tolist()

net = ToyModel()
net.eval()
#conv_layer = list(net.modules())[5]
#bn_layer = list(net.modules())[6]

#conv_layer = list(net.modules())[5]
#bn_layer = list(net.modules())[6]

fc_layer = list(net.modules())[8]
bn_layer = list(net.modules())[9]

"""
bn_layer.running_mean.copy_(torch.rand(bn_layer.running_mean.shape)*2)
"""
with torch.no_grad():
    bn_layer.weight.data.copy_(torch.rand(bn_layer.weight.data.shape)-0.5)

dummy_input = torch.round(torch.rand((1, 1, 5, 1))*5)
torch.onnx.export(net, dummy_input, "/home/ravit/Documents/NanoCAD/TinyMLContest/tinyml_contest2022_demo_example/toy.onnx")


"""
x = torch.round(torch.rand((1, 32, 3, 1))*5)
y = net.forward_1(x)

x_inf = convert_conv_act(x)
w_inf = convert_conv_weight(conv_layer.weight)
bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
bn_sign_inf_pack = pack(bn_sign_inf)
y_inf = convert_fc_act(y)

print("x", x_inf)
print("w", w_inf)
print("thr", bn_th_inf)
print("sign", bn_sign_inf_pack, bn_sign_inf)
print("y", y_inf)
"""

"""
x = torch.rand((1, 32, 3, 1))-0.5
y = net.forward_1(x)

x_inf = convert_conv_act(x)
w_inf = convert_conv_weight(conv_layer.weight)
bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
bn_sign_inf_pack = pack(bn_sign_inf)
y_inf = convert_conv_act(y, binarize=True)

x_inf_pack = pack(x_inf)
w_inf_pack = pack(w_inf)

print("x", x_inf_pack, x_inf)
print("w", w_inf_pack, w_inf)
print("thr", bn_th_inf)
print("sign", bn_sign_inf_pack, bn_sign_inf)
print("y", y_inf)
"""

x = torch.rand((1, 32))-0.5
y = net.forward_1(x)

x_inf = convert_fc_act(x)
w_inf = convert_fc_weight(fc_layer.weight)
mu, sigma, gamma, beta = convert_bn_float(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
bn_sign_inf_pack = pack(bn_sign_inf)
y_inf = convert_fc_act(y, binarize=True)

x_inf_pack = pack(x_inf)
w_inf_pack = pack(w_inf)

print("x", x_inf_pack, x_inf)
print("w", w_inf_pack, w_inf)
print("mu", mu)
print("sigma", sigma)
print("gamma", gamma)
print("beta", beta)
print("thr", bn_th_inf)
print("sign", bn_sign_inf_pack, bn_sign_inf)
print("y", y_inf)
