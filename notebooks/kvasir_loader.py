"""
kvasir_loader.py — Kvasir-SEG dataset for polyp segmentation.

Dataset: https://datasets.simula.no/kvasir-seg/
  1000 colonoscopy images with binary polyp masks.

Setup (one-time):
  1. Download Kvasir-SEG.zip from https://datasets.simula.no/kvasir-seg/
     (or from the alternate official mirror)
  2. Unzip to <data_root>/kvasir-seg/ such that you end up with:
       <data_root>/kvasir-seg/images/*.jpg
       <data_root>/kvasir-seg/masks/*.jpg

  Then call get_kvasir_seg(data_root) — it splits 800/100/100 (train/val/test)
  using a deterministic seed.

Usage:
    train_loader, val_loader, test_loader = get_kvasir_seg(
        data_root="./data", batch_size=8, image_size=224, seed=42
    )
"""

import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10


class KvasirSegDataset(Dataset):
    """Binary polyp segmentation. Returns (image, mask) tensors.

    image: (3, H, W) float, ImageNet-normalized
    mask:  (1, H, W) float in {0, 1}
    """
    def __init__(self, image_paths, mask_paths, image_size: int = 224,
                 augment: bool = False):
        assert len(image_paths) == len(mask_paths), \
            f"image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment

        # Normalization for the image
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load PIL images
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both to image_size using same interpolation type per channel
        image = TF.resize(image, [self.image_size, self.image_size],
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        # Augmentation: synchronized flips, color jitter only on image
        if self.augment:
            if np.random.rand() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if np.random.rand() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Random rotation (90° steps) preserves binary mask shapes
            if np.random.rand() < 0.5:
                k = np.random.choice([1, 2, 3])
                image = TF.rotate(image, 90 * k)
                mask = TF.rotate(mask, 90 * k)
            # Color jitter on image only
            cj = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            image = cj(image)

        # To tensor + normalize image
        image = TF.to_tensor(image)
        image = self.normalize(image)

        # Mask → binary float (0 or 1)
        mask = TF.to_tensor(mask)                     # (1, H, W) in [0, 1]
        mask = (mask > 0.5).float()

        return image, mask


def get_kvasir_seg(data_root: str = "./data", batch_size: int = 8,
                    image_size: int = 224, seed: int = 42,
                    num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """Loads Kvasir-SEG with deterministic 800/100/100 split.

    Returns (train_loader, val_loader, test_loader, num_classes, h, w).
    For binary segmentation, num_classes = 1 (foreground vs background via sigmoid).
    """
    root = Path(data_root) / "kvasir-seg"
    img_dir = root / "images"
    mask_dir = root / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Kvasir-SEG not found at {root}. "
            f"Download from https://datasets.simula.no/kvasir-seg/ and "
            f"unzip so that {img_dir} and {mask_dir} exist."
        )

    # Collect paired files (same stem in both folders)
    img_files = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    mask_files_by_stem = {p.stem: p for p in mask_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".png", ".jpeg")}

    paired_imgs, paired_masks = [], []
    for img_path in img_files:
        mask_path = mask_files_by_stem.get(img_path.stem)
        if mask_path is not None:
            paired_imgs.append(img_path)
            paired_masks.append(mask_path)

    n = len(paired_imgs)
    if n == 0:
        raise RuntimeError(f"No image/mask pairs found in {root}")
    print(f"  Found {n} image/mask pairs in {root}")

    # Deterministic shuffle then split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * DEFAULT_VAL_FRACTION))
    n_test = int(round(n * DEFAULT_TEST_FRACTION))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_imgs = [paired_imgs[i] for i in train_idx]
    train_masks = [paired_masks[i] for i in train_idx]
    val_imgs = [paired_imgs[i] for i in val_idx]
    val_masks = [paired_masks[i] for i in val_idx]
    test_imgs = [paired_imgs[i] for i in test_idx]
    test_masks = [paired_masks[i] for i in test_idx]

    train_ds = KvasirSegDataset(train_imgs, train_masks, image_size, augment=True)
    val_ds = KvasirSegDataset(val_imgs, val_masks, image_size, augment=False)
    test_ds = KvasirSegDataset(test_imgs, test_masks, image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, 1, image_size, image_size


if __name__ == "__main__":
    # Quick sanity check
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"
    tr, va, te, nc, h, w = get_kvasir_seg(data_root, batch_size=4, image_size=224)
    print(f"Kvasir-SEG: {nc} class(es), {h}×{w}")
    print(f"  train={len(tr.dataset)}  val={len(va.dataset)}  test={len(te.dataset)}")
    imgs, masks = next(iter(tr))
    print(f"  batch: images {tuple(imgs.shape)} dtype={imgs.dtype}")
    print(f"  batch: masks  {tuple(masks.shape)} dtype={masks.dtype} "
          f"min={masks.min().item():.2f} max={masks.max().item():.2f}")
