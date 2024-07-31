import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis


def get_net(model_name):
    """
    model_name: 
                LETBNet
                ALMBNet
                TSTMNet
                TSTMNet_reverse
    """
    if model_name == 'LETBNet':
        model = LETBNet(embedd_dim=64, depth=10)
    elif model_name == 'ALMBNet': 
        model = ALMBNet(embedd_dim=64, depth=10)
    elif model_name == 'TSTMNet':
        model = TSTMNet(embedd_dim=64, depth=10, num_head=8)
    elif model_name == 'TSTMNet_reverse':
        model = TSTMNet_reverse(embedd_dim=64, depth=10, num_head=8)
    else:
        raise RuntimeError('Error: no network name ' + model_name)
    return model

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fK' % (total / 1e3))

def print_model_flops(model, input):
    flops = FlopCountAnalysis(model, input)
    print("FLOPs: %.2fG" % (flops.total()/1e9))

def print_model_complexity(model, input):
    print_model_parm_nums(model)
    print_model_flops(model, input)
# ------------------------------------------------------------------------
#                                   ALMB
# ------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, need_learnable=True):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=need_learnable)
        self.rg = Rearrange('b c h w -> b (h w) c')

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        _, _, H, W = x.shape
        x = self.norm(self.rg(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)  # 中间一个卷积的通道数
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        cond = self.avg_pool(x)
        # cond_max = self.max_pool(x)
        scale = self.fc1(cond)
        shift = self.fc2(cond)
        # shift = self.fc2(cond_max)
        out = x * scale + shift + x
        return out


class ALMBBlock(nn.Module):
    def __init__(self, in_chs):
        super(ALMBBlock, self).__init__()
        self.norm1 = nn.Sequential()
        self.norm2 = nn.Sequential()
        self.conv1 = nn.Conv2d(in_chs, in_chs, 1, groups=in_chs)  # dwconv
        self.cam = SELayer(in_chs)
        self.ffn = FeedForward(in_chs, 1)

    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        x = self.cam(self.conv1(self.norm1(x))) + x
        x = self.ffn(self.norm2(x)) + x
        return x

# ------------------------------------------------------------------------
#                              LETB
# ------------------------------------------------------------------------
class Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """ Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Channel_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Input: x: [B, C, H, W]
        Output: x: [B, C, H, W]
        """
        _, _, H, W= x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) 
        qkv = qkv.permute(2, 0, 3, 1, 4)    
        q, k, v = qkv[0], qkv[1], qkv[2]   

        q = q.transpose(-2, -1)    
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)    
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 
        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)  
        x = self.proj(attened_x)
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        return x

class LocalFFN(nn.Module):
    def __init__(self, dim, hidden_dim=64, act_layer=nn.GELU):
        super(LocalFFN, self).__init__()
        self.project_in = nn.Conv2d(dim, hidden_dim*2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1, groups=hidden_dim*2)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LocalEnhanceModule(nn.Module):
    def __init__(self, dim, num_heads):
        super(LocalEnhanceModule, self).__init__()
        
        self.dwconv_path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )
        self.attn_path = Channel_Attention(dim, num_heads)
    
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        # ---------------------------------------------------------

    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        attened_x = self.attn_path(x)   # [B, C, H, W]
        conv_x = self.dwconv_path(x)  # [B, C, H, W]

        spatial_map = self.spatial_interaction(attened_x)  # [B, 1, H, W]
        conv_x = conv_x * torch.sigmoid(spatial_map)  # [B, C, H, W]

        x = attened_x + conv_x   # [B, C, H, W]
        
        return x

# local enhance tranformer block
class LETB(nn.Module):
    def __init__(self, dim, num_heads):
        super(LETB, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = LocalEnhanceModule(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = LocalFFN(dim)

    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    



# ------------------------------------------------------------------------
#                               Networks
# ------------------------------------------------------------------------
class LETBNet(nn.Module):
    def __init__(self, embedd_dim=64, depth=6):
        super(LETBNet, self).__init__()
        self.first_conv = nn.Conv2d(1, embedd_dim, 1, 1)

        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = LETB(embedd_dim, 8)
            self.layers.append(layer)

        self.last_conv = nn.Conv2d(embedd_dim, 1, 1, 1)
        
     
    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        shortcut = x
        x = self.first_conv(x)
        for layer in self.layers:
            x = layer(x)  
        x = shortcut + self.last_conv(x)
        return x
    


class ALMBNet(nn.Module):
    def __init__(self, embedd_dim, depth):
        super(ALMBNet, self).__init__()
        self.first_conv = nn.Conv2d(1, embedd_dim, 1, 1)
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ALMBBlock(embedd_dim))
        
        self.last_conv = nn.Conv2d(embedd_dim, 1, 1, 1)

    def forward(self, x):
        shortcut = x
        x = self.first_conv(x)
        for layer in self.layers:
            x = layer(x)  
        x = shortcut + self.last_conv(x)
        return x


    
class TSTMNet(nn.Module):
    def __init__(self, embedd_dim, depth, num_head=8):
        super(TSTMNet, self).__init__()
        self.first_conv = nn.Conv2d(1, embedd_dim, 1, 1)

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        for i in range(depth//2):
            self.layers1.append(ALMBBlock(embedd_dim))
            self.layers2.append(LETB(embedd_dim, num_head))

        self.last_conv = nn.Conv2d(embedd_dim, 1, 1, 1)

    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        shortcut = x
        x = self.first_conv(x)
        for layer in self.layers1:
            x = layer(x)
        for layer in self.layers2:
            x = layer(x)  
        x = shortcut + self.last_conv(x)
        return x
    



class TSTMNet_reverse(nn.Module):
    def __init__(self, embedd_dim, depth, num_head=8):
        super(TSTMNet_reverse, self).__init__()
        self.first_conv = nn.Conv2d(1, embedd_dim, 1, 1)

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        for i in range(depth//2):
            self.layers1.append(ALMBBlock(embedd_dim))
            self.layers2.append(LETB(embedd_dim, num_head))

        self.last_conv = nn.Conv2d(embedd_dim, 1, 1, 1)

    def forward(self, x):
        """
        input: B, C, H, W
        output: B, C, H, W
        """
        shortcut = x
        x = self.first_conv(x)
        for layer in self.layers2:
            x = layer(x)
        for layer in self.layers1:
            x = layer(x)  
        x = shortcut + self.last_conv(x)
        return x


##########################################################################
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
##########################################################################
