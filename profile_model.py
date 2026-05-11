# -*- coding: utf-8 -*-
"""
Model complexity analysis: parameter count and FLOPs.

Usage:
    python profile_model.py --pretrained_backbone /path/to/pvt_v2_b2.pth
"""

import argparse

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from models import build_ctm_polypnet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_backbone", type=str, default=None)
    p.add_argument("--img_size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_ctm_polypnet(args.pretrained_backbone).to(device)
    model.eval()

    dummy = torch.rand(1, 3, args.img_size, args.img_size).to(device)

    flops = FlopCountAnalysis(model, dummy)
    print(flop_count_table(flops))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {n_params / 1e6:.2f} M")
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")


if __name__ == "__main__":
    main()
