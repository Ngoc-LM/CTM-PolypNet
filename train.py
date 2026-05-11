# -*- coding: utf-8 -*-
"""
Training script for CTM-PolypNet using PyTorch Lightning.

Usage:
    python train.py \
        --data_path /path/to/polypData.npz \
        --pretrained_backbone /path/to/pvt_v2_b2.pth \
        --ckpt_dir ./checkpoints \
        --max_epochs 100

Outputs:
    Best checkpoint saved to  <ckpt_dir>/best_<val_dice>.ckpt
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from data import build_dataloaders
from models import build_ctm_polypnet
from utils.losses import deep_supervision_loss, dice_coef, iou_score


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class CTMPolypNetModule(pl.LightningModule):
    """PyTorch Lightning wrapper for CTM-PolypNet."""

    def __init__(self, model, lr: float = 2e-4, patience: int = 4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.patience = patience

    # -- forward ------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # -- shared step --------------------------------------------------------

    def _step(self, batch):
        imgs, masks = batch
        preds = self.model(imgs)               # (y4, y3, y2, y1)
        loss = deep_supervision_loss(preds, masks)
        dice = dice_coef(preds[0], masks)      # finest prediction
        iou = iou_score(preds[0], masks)
        return loss, dice, iou

    # -- training / validation / test steps ---------------------------------

    def training_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        self.log_dict({"loss": loss, "train_dice": dice, "train_iou": iou},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        self.log_dict({"val_loss": loss, "val_dice": dice, "val_iou": iou},
                      prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        self.log_dict({"test_loss": loss, "test_dice": dice, "test_iou": iou},
                      prog_bar=True)

    # -- optimiser ----------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_dice"}]

    # -- Lightning meta -----------------------------------------------------

    def get_metrics(self):
        items = super().get_metrics()
        items.pop("v_num", None)
        return items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CTM-PolypNet training")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to polypData.npz")
    p.add_argument("--pretrained_backbone", type=str, default=None,
                   help="Path to pvt_v2_b2.pth pre-trained weights")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Resume training from an existing checkpoint")
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--precision", type=int, default=16, choices=[16, 32])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # -- Data ----------------------------------------------------------------
    trainloader, valloader, _ = build_dataloaders(
        args.data_path,
        img_size=args.img_size,
        batch_size_train=args.batch_size,
        num_workers=args.num_workers)

    # -- Model ---------------------------------------------------------------
    model = build_ctm_polypnet(pretrained_backbone_path=args.pretrained_backbone)
    module = CTMPolypNetModule(model, lr=args.lr)

    if args.ckpt_path is not None:
        module = CTMPolypNetModule.load_from_checkpoint(
            args.ckpt_path, model=model, lr=args.lr)
        print(f"Resumed from checkpoint: {args.ckpt_path}")

    # -- Callbacks -----------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="ckpt{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        auto_insert_metric_name=False,
        verbose=True)
    progress_cb = TQDMProgressBar()

    # -- Trainer -------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[checkpoint_cb, progress_cb],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_progress_bar=True)

    trainer.fit(module, trainloader, valloader)
    print(f"\nBest checkpoint saved at: {checkpoint_cb.best_model_path}")
    print(f"Best val_dice: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
