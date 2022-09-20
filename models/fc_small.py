import torch.nn as nn
import torch.nn.functional as F

import binarized_modules

class FC_small(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, hid=512, ind=784, align=False):
        super(FC_small, self).__init__()
        self.align = align
        self.pruned = False
        self.hid = hid
        self.ind = ind

        self.full = full
        self.binary = binary

        if full:
            self.fc1 = nn.Linear(ind, hid, bias=False)
            self.fc2 = nn.Linear(hid, 10, bias=False)
        elif binary:
            self.fc1 = binarized_modules.BinarizeLinear(ind, hid, bias=False)
            self.fc2 = binarized_modules.BinarizeLinear(hid, 10, bias=False)
        else:
            self.fc1 = binarized_modules.TernarizeLinear(first_sparsity, ind, hid, bias=False, align=align)
            self.fc2 = binarized_modules.TernarizeLinear(rest_sparsity, hid, 10, bias=False, align=align)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(hid)

        self.bn2 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax=nn.LogSoftmax(dim=1)
    def forward(self, x):
        if self.full:
            x = x.view(-1, 784)
            if x.size(1)==784:
                x = x[:,:768]
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = self.fc2(x)
        else:
            x = x.view(-1, 784)
            if x.size(1)==784:
                x = x[:,:768]
            if self.binary:
                x = self.fc1(x)
            else:
                x = self.fc1(x, self.pruned)
            x = self.bn1(x)
            x = self.htanh1(x)
            if self.binary:
                x = self.fc2(x)
            else:
                x = self.fc2(x, self.pruned)
            x = self.bn2(x)
        return x
