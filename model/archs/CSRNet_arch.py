import torch
import torch.nn as nn
import torch.nn.functional as F


def get_CSRNet(in_nc=1, out_nc=1):
    return CSRNet(in_nc=in_nc, out_nc=out_nc)

def get_ALMBNet():
    return ALMBNet()

class Condition(nn.Module):
    def __init__(self, in_nc=1, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.model = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(nf, nf, 3, stride, pad, bias=True),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(nf, nf, 3, stride, pad, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = torch.mean(self.model(x), dim=[2, 3], keepdim=False)  
        return out



# 3 layers with control
class CSRNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, base_nf=64, cond_nf=32):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)


        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf,  bias=True)
        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)

        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True) 
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)


    def forward(self, x):
        cond = self.cond_net(x) 

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)  
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out  # 调制 GFN
        out = self.act(out)  
        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out
    
# net = CSRNet()
# x = torch.randn(1, 1, 1024, 1024)
# y = net(x)
# print(y.shape)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        cond = self.avg_pool(x).view(b, c)
        scale = self.fc1(cond)
        shift = self.fc2(cond)
        out = x * scale.view(b, c, 1, 1) + shift.view(b, c, 1, 1) + x
        return out

class ALMBNet(nn.Module):
    def __init__(self, in_nc=1, nf=32):
        super(ALMBNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1, bias=True),
            SELayer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, 1, bias=True),
            SELayer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, bias=True),
            SELayer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, bias=True),
            SELayer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, bias=True),
            SELayer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, 1, bias=True),
        )

    def forward(self, x):
        out = self.model(x)
        return out
