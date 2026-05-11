# -*- coding: utf-8 -*-
"""
Mamba-based Building Blocks for CTM-PolypNet.

Contains:
  - SS2D        : 2D Selective State Space (4-direction scanning)
  - VSSBlock    : Visual State Space block (standard, with conv)
  - FreeConvSS2D: SS2D variant without the internal conv (used in MEVSS)
  - MEVSSBlock  : Mamba Enhanced Vision State Space block
  - ResMambaBlock: Residual wrapper around MEVSSBlock
  - Mlp2        : MetaFormer-style channel MLP
  - DepthwiseBlock: Multi-kernel depthwise conv mixer
  - MambaMLPMixer : Proposed Mamba-MLP Fusion block (Algorithm 2 in paper)
  - SpatialMLPAttention: Proposed Spatial MLP Attention (Algorithm 1 in paper)
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath, to_2tuple

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None   # will raise at runtime if Mamba is not installed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Mlp2(nn.Module):
    """MetaFormer-style channel MLP (no depth-wise conv)."""

    def __init__(self, dim, mlp_ratio=4, out_features=None,
                 act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        hidden = int(mlp_ratio * dim)
        out = out_features or dim
        self.fc1 = nn.Linear(dim, hidden, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, out, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class DepthwiseBlock(nn.Module):
    """Multi-kernel depthwise convolution mixer."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.dwconv3 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.act(self.dwconv1(x) + self.dwconv2(x) + self.dwconv3(x))


# ---------------------------------------------------------------------------
# SS2D — 2D Selective State Space (standard, with depthwise conv)
# ---------------------------------------------------------------------------

class SS2D(nn.Module):
    """
    2D visual Selective State Space scanning in 4 directions.
    Used inside VSSBlock (standard Mamba visual block).
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv,
                                padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs).weight
            for _ in range(4)], dim=0))

        self.dt_projs_weight, self.dt_projs_bias = self._build_dt_projs(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
            dt_init_floor, factory_kwargs)

        self.A_logs = SS2D._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = SS2D._D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self._forward_core
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # -- init helpers -------------------------------------------------------

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max,
                 dt_init_floor, **factory_kwargs):
        proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(proj.weight, std)
        else:
            nn.init.uniform_(proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) *
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        proj.bias._no_reinit = True
        return proj

    def _build_dt_projs(self, dt_rank, d_inner, dt_scale, dt_init,
                        dt_min, dt_max, dt_init_floor, factory_kwargs):
        projs = [self._dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max,
                               dt_init_floor, **factory_kwargs) for _ in range(4)]
        weight = nn.Parameter(torch.stack([p.weight for p in projs], dim=0))
        bias = nn.Parameter(torch.stack([p.bias for p in projs], dim=0))
        return weight, bias

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # -- core scan ----------------------------------------------------------

    def _forward_core(self, x: torch.Tensor):
        scan = selective_scan_fn
        B, C, H, W = x.shape
        L, K = H * W, 4

        x_hwwh = torch.stack([x.view(B, -1, L),
                               x.transpose(2, 3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = scan(xs, dts, As, Bs, Cs, Ds, z=None,
                     delta_bias=dt_bias, delta_softplus=True,
                     return_last_state=False).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = out_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        invwh_y = inv_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = self.out_norm(y.transpose(1, 2).contiguous().view(B, H, W, -1))
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """Standard Visual State Space block (with conv2d inside SS2D)."""

    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                                   d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


# ---------------------------------------------------------------------------
# FreeConvSS2D / MEVSSBlock — variant without the internal conv
# (used in ResMambaBlock to avoid double conv with the external DW conv)
# ---------------------------------------------------------------------------

class FreeConvSS2D(SS2D):
    """SS2D without the internal depthwise conv (conv2d step is bypassed)."""

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        # Skip conv2d — the ResMambaBlock applies DW conv externally
        y1, y2, y3, y4 = self.forward_core(x.permute(0, 3, 1, 2).contiguous())
        y = y1 + y2 + y3 + y4
        y = self.out_norm(y.transpose(1, 2).contiguous().view(B, H, W, -1))
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class MEVSSBlock(nn.Module):
    """Mamba Enhanced Vision State Space block (uses FreeConvSS2D)."""

    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = FreeConvSS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                                            d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class ResMambaBlock(nn.Module):
    """
    Residual Mamba block: external DW conv + MEVSSBlock + InstanceNorm + LeakyReLU.
    Referred to as 'ResMEVSS' in the paper (LiteMamba-Bound style).
    """

    def __init__(self, in_c, k_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same',
                              groups=in_c, bias=True)
        self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.block = MEVSSBlock(hidden_dim=in_c)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        skip = x
        x = self.conv(x)
        x = self.block(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act(self.ins_norm(x))
        return x + skip * self.scale


# ---------------------------------------------------------------------------
# Spatial MLP Attention (Algorithm 1 in paper)
# ---------------------------------------------------------------------------

class SpatialMLPAttention(nn.Module):
    """
    Spatial MLP Attention: learns separate attention weights along height
    and width dimensions using MLPs rather than convolution kernels.
    """

    def __init__(self, dim, reduction_ratio=4, mlp_ratio=2, drop=0.):
        super().__init__()
        reduced = dim // reduction_ratio
        self.conv_reduce = nn.Conv2d(dim, reduced, 1)
        hidden = int(reduced * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(reduced, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, reduced),
            nn.Dropout(drop))
        self.conv_h = nn.Conv2d(reduced, dim, 1)
        self.conv_w = nn.Conv2d(reduced, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pool along each spatial axis
        xh = F.adaptive_avg_pool2d(x, (H, 1))          # B×C×H×1
        xw = F.adaptive_avg_pool2d(x, (1, W))           # B×C×1×W

        # Concatenate H and W features
        xh = xh.transpose(2, 3)                          # B×C×1×H
        xcat = torch.cat([xh, xw], dim=3)                # B×C×1×(H+W)
        xcat = self.conv_reduce(xcat)                     # B×reduced×1×(H+W)

        # MLP over sequence dimension
        xcat = xcat.squeeze(2).permute(0, 2, 1)          # B×(H+W)×reduced
        xcat = self.mlp(xcat)
        xcat = xcat.permute(0, 2, 1).unsqueeze(2)        # B×reduced×1×(H+W)

        # Split back and generate attention maps
        xh_out, xw_out = torch.split(xcat, [H, W], dim=3)
        xh_out = xh_out.transpose(2, 3)                  # B×reduced×H×1
        sh = torch.sigmoid(self.conv_h(xh_out))          # B×C×H×1
        sw = torch.sigmoid(self.conv_w(xw_out))          # B×C×1×W

        return x * sh * sw


# ---------------------------------------------------------------------------
# Mamba-MLP Fusion Block (Algorithm 2 in paper — proposed module)
# ---------------------------------------------------------------------------

class MambaMLPMixer(nn.Module):
    """
    Proposed Mamba-MLP Fusion Block.

    Pipeline (per Algorithm 2):
        x → ResMambaBlock (long-range deps) → SpatialMLPAttention →
        LayerNorm → ChannelMLP (residual) → output
    """

    def __init__(self, dim, mlp=Mlp2, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.1):
        super().__init__()
        self.mamba = ResMambaBlock(dim)
        self.spatial_attn = SpatialMLPAttention(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ins_norm = nn.InstanceNorm2d(dim, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: ResMEVSS (long-range spatial dependencies)
        x = self.mamba(x)
        # Step 2: Spatial MLP Attention (fine-grained spatial focus)
        x = self.spatial_attn(x)
        # Step 3: Channel MLP with LayerNorm and residual DropPath
        x = x.permute(0, 2, 3, 1)                         # B×H×W×C
        x = x + self.drop_path(self.mlp(self.norm(x)))
        x = x.permute(0, 3, 1, 2)                         # B×C×H×W
        return x
