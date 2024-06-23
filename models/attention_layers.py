import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import OptMultiConfig
from mmcv.cnn import ConvModule
from mmyolo.models.backbones.axial_attention import AxialAttention
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule
from einops import rearrange
from torch.nn import Softmax


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, localFlag=False):
        super(SpatialAttentionModule, self).__init__()
        self.localFlag = localFlag
        if localFlag:
            self.local = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
        else:
            self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        if self.localFlag:
            local = self.local(x)
            avgout = torch.mean(x, dim=1, keepdim=True)
            maxout, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avgout, maxout, local], dim=1)
        else:
            avgout = torch.mean(x, dim=1, keepdim=True)
            maxout, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class Axial(BaseModule):
    def __init__(self,
                 in_dim,
                 ratio=2,
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden = in_dim // ratio
        self.row = RowAttention(
            in_dim=in_dim, q_k_dim=hidden)
        self.col = ColAttention(
            in_dim=in_dim, q_k_dim=hidden)
        self.lsam = SpatialAttentionModule(
            in_channels=in_dim,
            localFlag=True)

    def forward(self, x):
        identity = x
        sp = self.lsam(x)
        r1 = self.row(x)
        c1 = self.col(x)
        x = r1 + c1
        x = x * sp + identity
        return x

class FocusedLinearAttention(BaseModule):
    def __init__(self,
                 dim,
                 nums_heads=8,
                 kernel_size=5,
                 focusing_factor=3,
                 proj_drop=0.,
                 attn_drop=0.,
                 qkv_bias=True
                 ):
        super().__init__()
        self.act = nn.ReLU()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        head_dim = dim // nums_heads
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim,
                             kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.focusing_factor = focusing_factor
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        qkv = self.qkv(x).reshape(B, H*W, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        q = self.act(q) + 1e-6
        k = self.act(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (
            rearrange(x, "b n (h c) -> (b h) n c", h=8)
            for x in [q, k, v])
        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        qk = torch.einsum("b i c, b j c -> b i j", q, k)
        x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (w h) c -> b c w h", w=W, h=H)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=8)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

	
class LastLayer(BaseModule):
    def __init__(self,
                 in_channels,
                 ratio=4,
                 msa=True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        hidden_dim = in_channels // ratio
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True))

        if msa:
            self.att = MsaBlock(
                dim=hidden_dim)
        else:
            self.att = FocusedLinearAttention(
                dim=hidden_dim,
                nums_heads=8,
                kernel_size=5,
                qkv_bias=True)

        self.conv2 = ConvModule(
            in_channels=hidden_dim,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True))
        
        self.lsam = SpatialAttentionModule(
            in_channels=in_channels,
            localFlag=True)

    def forward(self, x):
        identity = x
        sp = self.lsam(x)
        x = self.conv1(x)
        x = self.att(x)
        x = self.conv2(x)
        x = x * sp + identity
        return x