# -*- coding: utf-8 -*-
"""
Evaluation script for CTM-PolypNet.

Runs the model on all five test splits and reports DSC / IoU for each.

Usage:
    python evaluate.py \
        --data_path /path/to/polypData.npz \
        --ckpt_path /path/to/best.ckpt \
        --pretrained_backbone /path/to/pvt_v2_b2.pth
"""

import argparse

import pytorch_lightning as pl
import torch

from data import build_dataloaders
from models import build_ctm_polypnet
from train import CTMPolypNetModule


def parse_args():
    p = argparse.ArgumentParser(description="CTM-PolypNet evaluation")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--pretrained_backbone", type=str, default=None)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    model = build_ctm_polypnet(pretrained_backbone_path=args.pretrained_backbone)
    module = CTMPolypNetModule.load_from_checkpoint(
        args.ckpt_path, model=model)
    model.eval()

    _, _, testloaders = build_dataloaders(
        args.data_path,
        img_size=args.img_size,
        batch_size_val=args.batch_size,
        num_workers=args.num_workers)

    trainer = pl.Trainer(enable_progress_bar=True, logger=False)

    print("\n" + "=" * 55)
    print(f"{'Dataset':<15} {'DSC':>10} {'IoU':>10}")
    print("=" * 55)

    for name, loader in testloaders.items():
        results = trainer.test(module, loader, verbose=False)
        dice = results[0].get("test_dice", float("nan"))
        iou  = results[0].get("test_iou",  float("nan"))
        print(f"{name.upper():<15} {dice:>10.4f} {iou:>10.4f}")

    print("=" * 55)


if __name__ == "__main__":
    main()
