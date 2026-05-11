# -*- coding: utf-8 -*-
"""
CTM-PolypNet: Main model definition.

Architecture overview (see Fig. 1 in paper):
  Encoder : PVT-v2 backbone → 4 feature pyramids
  Bottleneck: PASPP (dilated multi-scale context)
  Decoder : 4-stage U-Net decoder with
            - conv_layer (multi-scale DW conv on skip)
            - SEBlock / CoordAtt / CBAM on skip
            - MambaMLPMixer (Mamba-MLP Fusion block)
            - RAFEB (Reverse Attention Feature Enhancement)
            - MapReduce + bilinear upsampling (deep supervision heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (CBAM, PASPP, SEBlock, CoordAtt, conv_layer)
from .mamba_blocks import MambaMLPMixer
from .pvt_v2 import pvt_v2_b2


# ---------------------------------------------------------------------------
# Decoder helpers
# ---------------------------------------------------------------------------

class conv_block(nn.Module):
    """2×(Conv-BN-ReLU) block."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class up_conv(nn.Module):
    """Bilinear upsample × 2 then reduce channels."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class MapReduce(nn.Module):
    """Reduce C channels → 1 channel (deep supervision head)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# RAFE — Reverse Attention Feature Enhancement
# ---------------------------------------------------------------------------

class BFEB(nn.Module):
    """
    Boundary-aware Foreground Extraction Block (base).
    Computes F_att and B_att from a predicted map, fuses with input.
    """

    def __init__(self, in_c: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.in_c = in_c

    def forward(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        residual = x
        score = self.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        B_att = 1 - (dist / 0.5)
        F_att = torch.abs(0.5 - score)
        att = F.interpolate(F_att - B_att, size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        att_x = att.expand(-1, self.in_c, -1, -1) * x
        return att_x + residual


class RAFEB(nn.Module):
    """
    Reverse Attention Feature Enhancement Block (proposed).

    Applies a reversed sigmoid to the decoder feature, then uses BFEB
    to fuse boundary-aware attention from the previous prediction head.
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.bfeb = BFEB(in_c)
        self.in_c = in_c
        self.sigmoid = nn.Sigmoid()

    def forward(self, xd: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        xd = 1 - self.sigmoid(xd)                           # reverse attention
        x_pred = self.bfeb(xd, x_pred)                      # boundary fusion
        return xd.expand(-1, self.in_c, -1, -1) * x_pred   # modulate


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CTMPolypNet(nn.Module):
    """
    CTM-PolypNet: Convolution-Transformer-Mamba network for polyp segmentation.

    Returns four deep-supervision outputs (sigmoid-activated):
        y4 (finest, 1/1), y3, y2, y1 (coarsest, 1/8)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()

        # -- Backbone (PVT-v2) --
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # -- Bottleneck --
        self.paspp = PASPP(512, 512)

        # -- Skip-connection refinement (multi-scale DW conv) --
        self.conv_layer1 = conv_layer(512)
        self.conv_layer2 = conv_layer(320)
        self.conv_layer3 = conv_layer(128)
        self.conv_layer4 = conv_layer(64)

        # -- Squeeze-and-Excitation on each skip --
        self.se1 = SEBlock(512)
        self.se2 = SEBlock(320)
        self.se3 = SEBlock(128)
        self.se4 = SEBlock(64)

        # -- Coordinate Attention on each skip --
        self.ca1 = CoordAtt(512, 512)
        self.ca2 = CoordAtt(320, 320)
        self.ca3 = CoordAtt(128, 128)
        self.ca4 = CoordAtt(64, 64)

        # -- CBAM on each decoder feature --
        self.cbam1 = CBAM(512)
        self.cbam2 = CBAM(320)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(64)

        # -- Upsamplers --
        self.up1 = up_conv(512, 320)
        self.up2 = up_conv(320, 128)
        self.up3 = up_conv(128, 64)

        # -- Decoder conv blocks (cat → reduce) --
        self.upconv1 = conv_block(1024, 512)
        self.upconv2 = conv_block(640, 320)
        self.upconv3 = conv_block(256, 128)
        self.upconv4 = conv_block(128, 64)

        # -- Deep supervision heads --
        self.mapreduce1 = MapReduce(512)
        self.mapreduce2 = MapReduce(320)
        self.mapreduce3 = MapReduce(128)
        self.mapreduce4 = MapReduce(64)

        # -- Mamba-MLP Fusion blocks (proposed) --
        self.mamba1 = MambaMLPMixer(dim=512)
        self.mamba2 = MambaMLPMixer(dim=320)
        self.mamba3 = MambaMLPMixer(dim=128)
        self.mamba4 = MambaMLPMixer(dim=64)

        # -- RAFE blocks (proposed) --
        self.deep1 = RAFEB(320, 1)
        self.deep2 = RAFEB(128, 1)
        self.deep3 = RAFEB(64, 1)

        self.sigmoid = nn.Sigmoid()

    # -- Backbone feature extraction ----------------------------------------

    def _get_pyramid(self, x: torch.Tensor):
        """Extract 4-level feature pyramid from PVT-v2 backbone."""
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub in module:
                    x = sub(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)
        return pyramid

    # -- Forward pass -------------------------------------------------------

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2:]
        p = self._get_pyramid(x)

        # ---------- Stage 1: bottleneck (8×8) ----------
        p3_aspp = self.paspp(p[3])                              # (512, 8, 8)
        p[3] = self.se1(self.conv_layer1(p[3]))
        p[3] = torch.cat([p[3], p3_aspp], dim=1)               # (1024, 8, 8)
        p[3] = self.cbam1(self.mamba1(self.ca1(self.upconv1(p[3]))))

        x1 = F.interpolate(self.mapreduce1(p[3]), (H, W), mode='bilinear', align_corners=False)

        # ---------- Stage 2: 16×16 ----------
        p[2] = self.se2(self.conv_layer2(p[2]))
        p[3] = self.up1(p[3])                                   # (320, 16, 16)
        p[2] = torch.cat([p[3], p[2]], dim=1)                   # (640, 16, 16)
        p[2] = self.cbam2(self.mamba2(self.ca2(self.upconv2(p[2]))))
        p[2] = self.deep1(p[2], x1)

        x2 = F.interpolate(self.mapreduce2(p[2]), (H, W), mode='bilinear', align_corners=False)

        # ---------- Stage 3: 32×32 ----------
        p[1] = self.se3(self.conv_layer3(p[1]))
        p[2] = self.up2(p[2])                                   # (128, 32, 32)
        p[1] = torch.cat([p[2], p[1]], dim=1)                   # (256, 32, 32)
        p[1] = self.cbam3(self.mamba3(self.ca3(self.upconv3(p[1]))))
        p[1] = self.deep2(p[1], x2)

        x3 = F.interpolate(self.mapreduce3(p[1]), (H, W), mode='bilinear', align_corners=False)

        # ---------- Stage 4: 64×64 ----------
        p[0] = self.se4(self.conv_layer4(p[0]))
        p[1] = self.up3(p[1])                                   # (64, 64, 64)
        p[0] = torch.cat([p[1], p[0]], dim=1)                   # (128, 64, 64)
        p[0] = self.cbam4(self.mamba4(self.ca4(self.upconv4(p[0]))))
        p[0] = self.deep3(p[0], x3)

        x4 = F.interpolate(self.mapreduce4(p[0]), (H, W), mode='bilinear', align_corners=False)

        return (self.sigmoid(x4), self.sigmoid(x3),
                self.sigmoid(x2), self.sigmoid(x1))


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

def build_ctm_polypnet(pretrained_backbone_path: str = None) -> CTMPolypNet:
    """
    Instantiate CTM-PolypNet with a PVT-v2-B2 backbone.

    Args:
        pretrained_backbone_path: Path to `pvt_v2_b2.pth` pretrained weights.
                                  If None, backbone is randomly initialised.
    Returns:
        CTMPolypNet model (on CPU).
    """
    backbone = pvt_v2_b2()
    if pretrained_backbone_path is not None:
        state = torch.load(pretrained_backbone_path, map_location='cpu')
        backbone.load_state_dict(state)
        print(f"[CTMPolypNet] Loaded PVT-v2-B2 weights from {pretrained_backbone_path}")
    return CTMPolypNet(backbone=backbone)
