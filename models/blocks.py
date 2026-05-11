# -*- coding: utf-8 -*-
"""
Attention and Convolution Blocks for CTM-PolypNet.

Contains:
  - CBAM  (Convolutional Block Attention Module)
  - CoordAtt (Coordinate Attention)
  - SEBlock (Squeeze-and-Excitation)
  - PASPP  (Parallel Atrous Spatial Pyramid Pooling)
  - SKConv / SKUnit (Selective Kernel)
  - conv_layer (multi-scale axial depthwise conv block)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


# ---------------------------------------------------------------------------
# CBAM — Convolutional Block Attention Module
# Woo et al., ECCV 2018
# ---------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lp':
                pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lse':
                pool = _logsumexp_2d(x)
            raw = self.mlp(pool)
            channel_att_sum = raw if channel_att_sum is None else channel_att_sum + raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def _logsumexp_2d(tensor):
    flat = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(flat, dim=2, keepdim=True)
    return s + (flat - s).exp().sum(dim=2, keepdim=True).log()


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg', 'max'), no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# ---------------------------------------------------------------------------
# Coordinate Attention
# Hou et al., CVPR 2021
# ---------------------------------------------------------------------------

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention module."""

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation recalibration."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.global_avg_pool(x)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se


# ---------------------------------------------------------------------------
# PASPP — Parallel Atrous Spatial Pyramid Pooling
# ---------------------------------------------------------------------------

class PASPP(nn.Module):
    """Parallel Atrous Spatial Pyramid Pooling block."""

    def __init__(self, inplanes, outplanes, output_stride=4,
                 BatchNorm=nn.BatchNorm2d):
        super().__init__()
        dilation_map = {
            1:  [1, 16, 32, 48],
            2:  [1, 12, 24, 36],
            4:  [1,  6, 12, 18],
            8:  [1,  4,  6, 10],
            16: [1,  2,  3,  4],
        }
        if output_stride not in dilation_map:
            raise NotImplementedError(f"output_stride={output_stride} not supported.")
        dilations = dilation_map[output_stride]

        self._norm_layer = BatchNorm
        self.silu = nn.SiLU(inplace=True)

        q = inplanes // 4
        self.conv1 = self._make_layer(inplanes, q)
        self.conv2 = self._make_layer(inplanes, q)
        self.conv3 = self._make_layer(inplanes, q)
        self.conv4 = self._make_layer(inplanes, q)

        self.atrous_conv1 = nn.Conv2d(q, q, 3, dilation=dilations[0], padding=dilations[0])
        self.atrous_conv2 = nn.Conv2d(q, q, 3, dilation=dilations[1], padding=dilations[1])
        self.atrous_conv3 = nn.Conv2d(q, q, 3, dilation=dilations[2], padding=dilations[2])
        self.atrous_conv4 = nn.Conv2d(q, q, 3, dilation=dilations[3], padding=dilations[3])

        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, inplanes)

    def _make_layer(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            self._norm_layer(outplanes),
            self.silu)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.conv2(X)
        x3 = self.conv3(X)
        x4 = self.conv4(X)

        x12 = torch.add(x1, x2)
        x34 = torch.add(x3, x4)

        x1 = torch.add(self.atrous_conv1(x1), x12)
        x2 = torch.add(self.atrous_conv2(x2), x12)
        x3 = torch.add(self.atrous_conv3(x3), x34)
        x4 = torch.add(self.atrous_conv4(x4), x34)

        x12 = self.conv5(torch.cat([x1, x2], dim=1))
        x34 = self.conv5(torch.cat([x3, x4], dim=1))
        return self.convout(torch.cat([x12, x34], dim=1))


# ---------------------------------------------------------------------------
# Selective Kernel Convolution
# ---------------------------------------------------------------------------

class SKConv(nn.Module):
    """Selective Kernel Convolution (3×3 kernels with M branches)."""

    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, 3, stride=stride,
                          padding='same', dilation=i + 1, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True))
            for i in range(M)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([nn.Conv2d(d, features, 1) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = torch.cat([conv(x) for conv in self.convs], dim=1)
        feats = feats.view(B, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = feats.sum(dim=1)
        feats_Z = self.fc(self.gap(feats_U))
        attn = torch.cat([fc(feats_Z) for fc in self.fcs], dim=1)
        attn = self.softmax(attn.view(B, self.M, self.features, 1, 1))
        return (feats * attn).sum(dim=1)


class SKUnit(nn.Module):
    """Residual block using SKConv in the middle layer."""

    def __init__(self, in_features, mid_features, out_features,
                 M=2, G=32, r=16, stride=1, L=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True))
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features))
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv3(self.conv2_sk(self.conv1(x)))
        return self.relu(out + self.shortcut(x))


# ---------------------------------------------------------------------------
# Multi-scale Axial Depthwise Convolution Block
# ---------------------------------------------------------------------------

class AxialDW(nn.Module):
    """Axial depth-wise convolution (height + width separable)."""

    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, (h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, (1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        return x + self.dw_h(self.dw_w(x))


class PCA(nn.Module):
    """Priority Channel Attention."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding='same')
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        c = reduce(x, 'b c w h -> b c', 'mean')
        x = self.dw(x)
        c_ = reduce(x, 'b c w h -> b c', 'mean')
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))
        return torch.einsum('bchw, bc -> bchw', x, att_score)


class PSA(nn.Module):
    """Priority Spatial Attention."""

    def __init__(self, dim):
        super().__init__()
        self.prob = nn.Softmax2d()

    def forward(self, x):
        B, C, H, W = x.shape
        pw_h = nn.Conv2d(H, H, (1, 1)).to(x.device)
        pw_w = nn.Conv2d(W, W, (1, 1)).to(x.device)

        s = reduce(x, 'b c w h -> b w h', 'mean')
        x_h = pw_h(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = pw_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        s_ = reduce(x, 'b c w h -> b w h', 'mean')
        s_h = reduce(x_h, 'b c w h -> b w h', 'mean')
        s_w = reduce(x_w, 'b c w h -> b w h', 'mean')

        raise_sp = self.prob(s_ - s)
        raise_h = self.prob(s_h - s)
        raise_w = self.prob(s_w - s)

        att_score = torch.sigmoid(
            s_ * (1 + raise_sp) + s_h * (1 + raise_h) + s_w * (1 + raise_w))
        return torch.einsum('bcwh, bwh -> bcwh', x, att_score)


class ConvMixer(nn.Module):
    """Depthwise (PCA) + pointwise (PSA) mixer."""

    def __init__(self, inplane, outplane):
        super().__init__()
        self.depthwise = PCA(inplane)
        self.pointwise = PSA(outplane)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class plugin_conv(nn.Module):
    """Channel-split feature mixing plugin."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim // 2, dim, kernel_size=1, padding='same', bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottle = ConvMixer(dim // 2, dim // 2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        combined = (self.avgpool(x1) + self.avgpool(x2)) / 2
        main = self.conv(self.bottle(combined))
        x1_2, x2_2 = torch.chunk(main, 2, dim=1)
        return torch.cat([x1 * x1_2, x2 * x2_2], dim=1)


class conv_layer(nn.Module):
    """Multi-scale axial DW conv with plugin mixing (skip connection block)."""

    def __init__(self, dim):
        super().__init__()
        q = dim // 4
        self.dw_3 = AxialDW(q, (3, 3))
        self.dw_5 = AxialDW(q, (5, 5))
        self.dw_7 = AxialDW(q, (7, 7))
        self.dw_9 = AxialDW(q, (9, 9))
        self.plugin = plugin_conv(q)
        self.conv_3 = nn.Conv2d(q, q, 3, padding='same', groups=q, dilation=3)
        self.conv_1 = nn.Conv2d(dim, dim, 1, padding='same', groups=dim, bias=False)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.conv_3(self.plugin(self.dw_3(x1)))
        x2 = self.conv_3(self.plugin(self.dw_5(x2)))
        x3 = self.conv_3(self.plugin(self.dw_7(x3)))
        x4 = self.conv_3(self.plugin(self.dw_9(x4)))
        return self.conv_1(torch.cat([x1, x2, x3, x4], dim=1))
