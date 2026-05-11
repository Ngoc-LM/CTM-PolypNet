# -*- coding: utf-8 -*-
"""
Loss functions and evaluation metrics for CTM-PolypNet.

Loss functions:
  - DiceLoss
  - TverskyLoss  (α = 0.7)
  - DiceTverskyLoss  (combined, equal weights)
  - deep_supervision_loss  (weighted sum over 4 decoder outputs)

Metrics:
  - dice_coef  (differentiable, for logging)
  - iou_score  (threshold-based, numpy)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss (no sigmoid inside — expects raw logits or probabilities)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky loss.  α controls the FN penalty (α=0.5 → Dice; α→1 → recall).
    Applies sigmoid internally.
    """

    def __init__(self, alpha: float = 0.7, smooth: float = 1e-3):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def _tversky(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(y_pred).view(-1)
        y_true = y_true.view(-1)
        tp = (y_true * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        fp = ((1 - y_true) * y_pred).sum()
        return (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1 - self._tversky(y_pred, y_true)


class DiceTverskyLoss(nn.Module):
    """
    Combined Dice + Tversky loss (equal 0.5 / 0.5 weighting).
    Used as the per-head loss in CTM-PolypNet's deep supervision.
    """

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(pred, target) +
                (1 - self.dice_weight) * self.tversky(pred, target))


# ---------------------------------------------------------------------------
# Deep supervision loss (Eq. 3 in paper)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = (0.4, 0.3, 0.2, 0.1)   # λ4, λ3, λ2, λ1 (finest → coarsest)

_criterion = DiceTverskyLoss()


def deep_supervision_loss(preds, target: torch.Tensor,
                          weights=_DEFAULT_WEIGHTS) -> torch.Tensor:
    """
    Compute the weighted deep supervision loss.

    Args:
        preds  : Tuple/list of 4 sigmoid-activated predictions,
                 ordered (finest, ..., coarsest) — i.e. (y4, y3, y2, y1).
        target : Ground-truth mask  (B, 1, H, W).
        weights: Per-head λ values summing to 1.
    Returns:
        Scalar loss.
    """
    assert len(preds) == len(weights), "Number of predictions must match weights."
    return sum(w * _criterion(p, target) for p, w in zip(preds, weights))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def dice_coef(output: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Differentiable Dice coefficient (for logging during training)."""
    output = output.view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def iou_score(output: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Threshold-based IoU (numpy, for evaluation).

    Args:
        output    : Model prediction tensor (sigmoid-activated).
        target    : Ground-truth mask tensor.
        threshold : Binarisation threshold.
        smooth    : Smoothing term.
    Returns:
        IoU value as a Python float.
    """
    out = (output.detach().cpu().numpy() > threshold)
    tgt = (target.detach().cpu().numpy() > threshold)
    intersection = (out & tgt).sum()
    union = (out | tgt).sum()
    return float((intersection + smooth) / (union + smooth))
