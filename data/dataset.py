# -*- coding: utf-8 -*-
"""
Polyp segmentation dataset utilities.

Supports loading from:
  1. A single .npz file with keys  {split}_img  /  {split}_msk
     (Kvasir-SEG + CVC-ClinicDB combined format used in this project)
  2. A directory of raw images and masks (PNG/JPG)

Splits available in the .npz format:
  train, val,
  test_kvasir, test_etis, test_cvc300, test_clinic, test_colon
"""

import os
from typing import Optional, Callable

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transform(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_val_transform(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ---------------------------------------------------------------------------
# Dataset — .npz format
# ---------------------------------------------------------------------------

class PolypNpzDataset(Dataset):
    """
    Loads polyp images and masks from a pre-packed .npz file.

    Expected keys: ``{split}_img``  (N, H, W, 3) uint8
                   ``{split}_msk``  (N, H, W, 1) or (N, H, W) uint8

    Args:
        data_path  : Path to the .npz file.
        split      : One of 'train', 'val', 'test_kvasir', 'test_etis',
                     'test_cvc300', 'test_clinic', 'test_colon'.
        transform  : albumentations Compose pipeline (or None).
    """

    def __init__(self, data_path: str, split: str,
                 transform: Optional[A.Compose] = None):
        super().__init__()
        data = np.load(data_path)
        self.images = data[f"{split}_img"]
        self.masks = data[f"{split}_msk"].squeeze(-1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        msk = self.masks[idx]

        if self.transform is not None:
            out = self.transform(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        img = transforms.ToTensor()(img)
        msk = transforms.ToTensor()(np.expand_dims(msk, axis=-1))
        return img, msk


# ---------------------------------------------------------------------------
# Dataset — directory format
# ---------------------------------------------------------------------------

class PolypDirDataset(Dataset):
    """
    Loads polyp images and masks from two directories (image_dir / mask_dir).

    Masks are expected to be single-channel PNG files with the same stem name
    as the corresponding image.

    Args:
        image_dir  : Directory containing RGB images.
        mask_dir   : Directory containing binary mask images.
        transform  : albumentations Compose pipeline (or None).
        img_size   : Target resize (default 256).
    """

    def __init__(self, image_dir: str, mask_dir: str,
                 transform: Optional[A.Compose] = None,
                 img_size: int = 256):
        super().__init__()
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_dir = mask_dir
        self.transform = transform or get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        msk_path = os.path.join(self.mask_dir, stem + '.png')

        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.array(Image.open(msk_path).convert('L'))

        out = self.transform(image=img, mask=msk)
        img, msk = out["image"], out["mask"]

        img = transforms.ToTensor()(img)
        msk = transforms.ToTensor()(np.expand_dims(msk.astype(np.float32) / 255., axis=-1))
        return img, msk


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(data_path: str, img_size: int = 256,
                      batch_size_train: int = 16,
                      batch_size_val: int = 8,
                      num_workers: int = 2):
    """
    Build all dataloaders from a single .npz file.

    Returns:
        trainloader, valloader, and a dict of test loaders keyed by dataset name.
    """
    train_tf = get_train_transform(img_size)
    val_tf = get_val_transform(img_size)

    train_ds = PolypNpzDataset(data_path, 'train', transform=train_tf)
    val_ds = PolypNpzDataset(data_path, 'val', transform=val_tf)

    test_splits = {
        'kvasir':  'test_kvasir',
        'etis':    'test_etis',
        'cvc300':  'test_cvc300',
        'clinic':  'test_clinic',
        'colon':   'test_colon',
    }
    test_datasets = {
        name: PolypNpzDataset(data_path, split, transform=val_tf)
        for name, split in test_splits.items()
    }

    trainloader = DataLoader(train_ds, batch_size=batch_size_train,
                             num_workers=num_workers, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size_val,
                           num_workers=num_workers, shuffle=False)
    testloaders = {
        name: DataLoader(ds, batch_size=batch_size_val,
                         num_workers=num_workers, shuffle=False)
        for name, ds in test_datasets.items()
    }

    return trainloader, valloader, testloaders
