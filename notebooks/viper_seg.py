"""
viper_seg.py — Wavelet-conditioned segmentation transformer (Kvasir-SEG).

Architecture
------------

  Image (B, 3, 224, 224)
   │
   ├──→ DeiT-Tiny encoder (patch=16, pretrained ImageNet)
   │      → patch tokens (B, 197, 192)
   │      → drop CLS, reshape patches → spatial map (B, 192, 14, 14)
   │
   └──→ Wavelet branch (DyWPE-faithful)
          → modulated subbands at native resolutions
          → multi-scale features w_LL, w_L3, w_L2, w_L1

  Decoder (U-Net style, 4 upsampling stages):
     Stage  In         Out         Wavelet skip
     1      14×14×192  → 28×28×96   ←  LL (28×28×d_pe at J=3 for 224)
     2      28×28×96   → 56×56×48   ←  level-3 details (28×28)  [upsampled]
     3      56×56×48   → 112×112×24 ←  level-2 details (56×56)
     4      112×112×24 → 224×224×K  ←  level-1 details (112×112)

  Output: per-pixel logits (B, K, 224, 224)

Wavelet fusion modes (ablatable via cfg.fusion):
  • add:    decoder_feat + project(w)          (cleanest, U-Net style)
  • film:   (1+γ(w)) ⊙ decoder_feat + β(w)     (most expressive)
  • concat: cat([decoder_feat, project(w)]) → 1×1 conv

Run examples
------------

# Smoke test (3 epochs, just ViPER, FiLM fusion):
python viper_seg.py --suite smoke --epochs 3

# Main ablation across fusion modes (single seed):
python viper_seg.py --suite ablation --epochs 60

# Multi-seed final run:
python viper_seg.py --suite contenders --epochs 60 --seeds 42 123 7
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")

from pytorch_wavelets import DWTForward

from kvasir_loader import get_kvasir_seg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ViPERSegConfig:
    # Wavelet feature extractor
    n_levels:     int = 3
    wavelet:      str = "db4"
    channel_mode: str = "gray"
    d_pe:         int = 32             # internal dim of wavelet branch

    # Fusion mode for wavelet features into decoder
    fusion:       str = "film"         # add | film | concat

    # Architecture
    use_pretrained_encoder: bool = True
    encoder_lr_mult: float = 0.1        # encoder LR = decoder_LR * this


# =============================================================================
# Wavelet feature extractor (reused logic from viper_v5)
# =============================================================================

class ChannelProjection(nn.Module):
    def __init__(self, in_channels: int, mode: str = "gray"):
        super().__init__()
        self.mode = mode
        if mode == "gray":
            self.register_buffer(
                "rgb2gray",
                torch.tensor([0.299, 0.587, 0.114]).view(1, in_channels, 1, 1)
                if in_channels == 3 else
                torch.ones(1, in_channels, 1, 1) / in_channels
            )
        elif mode == "learnable_proj":
            self.proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
            if in_channels == 3:
                with torch.no_grad():
                    self.proj.weight.copy_(
                        torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                    )
    def forward(self, x):
        if self.mode == "gray":
            return (x * self.rgb2gray).sum(dim=1, keepdim=True)
        return self.proj(x)


class ViPERWaveletBranch(nn.Module):
    """Extracts modulated wavelet subbands at MULTIPLE scales for the decoder.

    Unlike v5 (which collapsed everything to a single patch-grid feature map),
    this version preserves each scale as a separate feature map so the decoder
    can use them as skip connections.
    """
    def __init__(self, cfg: ViPERSegConfig, in_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        self.d_pe = cfg.d_pe

        self.channel_proj = ChannelProjection(in_channels, cfg.channel_mode)
        self.dwt = DWTForward(J=cfg.n_levels, wave=cfg.wavelet, mode="symmetric")

        # DyWPE-style gating
        self.n_subbands = 1 + 3 * cfg.n_levels
        self.scale_embeddings = nn.Parameter(
            torch.randn(self.n_subbands, cfg.d_pe) * 0.02
        )
        self.W_g = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)
        self.W_v = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)

    def _compute_gate(self, sb_idx: int) -> torch.Tensor:
        e = self.scale_embeddings[sb_idx]
        g = torch.sigmoid(self.W_g(e))
        v = torch.tanh(self.W_v(e))
        return g * v

    def _modulate(self, coeffs: torch.Tensor, sb_idx: int) -> torch.Tensor:
        """coeffs: (B, 1, H, W) → (B, d_pe, H, W) via gate-broadcast."""
        gate = self._compute_gate(sb_idx).view(1, self.d_pe, 1, 1)
        return coeffs * gate

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of multi-scale feature maps, COARSE TO FINE:
            [w_LL, w_L_J, w_L_{J-1}, ..., w_L_1]
        Each w is shape (B, d_pe, h, w) at the native subband resolution.

        For J=3 on a 224×224 image, this gives:
            w_LL:  (B, d_pe, 28,  28)   — semantic, coarsest
            w_L3:  (B, d_pe, 28,  28)   — detail at level 3
            w_L2:  (B, d_pe, 56,  56)   — detail at level 2
            w_L1:  (B, d_pe, 112, 112)  — detail at level 1, finest
        """
        x = self.channel_proj(image)                          # (B, 1, H, W)
        Yl, Yh = self.dwt(x)

        # LL approximation (deepest level, semantic)
        w_LL = self._modulate(Yl, 0)                          # (B, d_pe, h_J, w_J)

        # For each level j, combine the 3 detail subbands (LH/HL/HH) into one
        # multi-channel feature map per scale.
        # We modulate each direction with its own gate, then sum (matches the
        # information content while reducing channel inflation).
        per_level_features = []
        for j in range(self.cfg.n_levels):
            details = Yh[j]                                    # (B, 1, 3, h, w)
            level_acc = torch.zeros_like(self._modulate(details[:, :, 0], 1))
            for d in range(3):
                sb_idx = 1 + j * 3 + d
                sb = details[:, :, d]                          # (B, 1, h, w)
                level_acc = level_acc + self._modulate(sb, sb_idx)
            per_level_features.append(level_acc)               # (B, d_pe, h_j, w_j)

        # pytorch_wavelets returns Yh[0] = finest level (largest), Yh[-1] = coarsest
        # We want decoder skips coarse → fine, so we reverse:
        # [w_LL, w_L_J (coarsest detail), ..., w_L_1 (finest)]
        per_level_features = list(reversed(per_level_features))

        return [w_LL] + per_level_features


# =============================================================================
# Decoder with wavelet fusion
# =============================================================================

class WaveletFusion(nn.Module):
    """Fuse wavelet feature `w` into decoder feature `x`.

    Supports three modes:
      • add:    x + project(w)
      • film:   (1+γ(w)) ⊙ x + β(w)
      • concat: 1×1 conv on cat([x, project(w)])
    """
    def __init__(self, x_channels: int, w_channels: int, mode: str = "film"):
        super().__init__()
        self.mode = mode
        if mode == "add":
            self.proj = nn.Conv2d(w_channels, x_channels, kernel_size=1)
        elif mode == "film":
            self.film_proj = nn.Conv2d(w_channels, 2 * x_channels, kernel_size=1)
            # Zero init so γ=β=0 → x passes through unchanged at init
            with torch.no_grad():
                self.film_proj.weight.zero_()
                if self.film_proj.bias is not None:
                    self.film_proj.bias.zero_()
        elif mode == "concat":
            self.fuse = nn.Conv2d(x_channels + w_channels, x_channels,
                                   kernel_size=1)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Resize w to match x's spatial dims if necessary
        if w.shape[-2:] != x.shape[-2:]:
            w = F.interpolate(w, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
        if self.mode == "add":
            return x + self.proj(w)
        elif self.mode == "film":
            params = self.film_proj(w)
            gamma, beta = params.chunk(2, dim=1)
            return (1.0 + gamma) * x + beta
        elif self.mode == "concat":
            return self.fuse(torch.cat([x, w], dim=1))


class DecoderBlock(nn.Module):
    """One upsampling stage: 2× upsample → conv → fuse with wavelet skip → conv."""
    def __init__(self, in_channels: int, out_channels: int, w_channels: int,
                 fusion: str = "film"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.fuse = WaveletFusion(out_channels, w_channels, mode=fusion)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x, w_skip):
        x = self.up(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.fuse(x, w_skip)
        x = self.act(self.norm2(self.conv2(x)))
        return x


# =============================================================================
# Full segmentation model
# =============================================================================

class ViPERSeg(nn.Module):
    """DeiT-Tiny encoder + wavelet-conditioned decoder for binary segmentation."""

    def __init__(self, num_classes: int = 1, image_size: int = 224,
                 patch_size: int = 16, cfg: Optional[ViPERSegConfig] = None):
        super().__init__()
        self.cfg = cfg or ViPERSegConfig()
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size

        # Encoder
        self.encoder = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=self.cfg.use_pretrained_encoder,
            num_classes=0,                                # remove classifier head
            global_pool="",                                # keep all tokens
            img_size=image_size,
        )
        d_enc = self.encoder.embed_dim                     # 192
        # Drop the encoder's classification head completely
        self.encoder.head = nn.Identity()

        # Wavelet branch
        self.wavelet_branch = ViPERWaveletBranch(self.cfg, in_channels=3)
        d_pe = self.cfg.d_pe

        # Decoder: 4 upsampling stages, halving channels each time
        # Channels chosen to roughly match U-Net's pyramid for tiny model
        self.dec_block1 = DecoderBlock(in_channels=d_enc, out_channels=96,
                                         w_channels=d_pe, fusion=self.cfg.fusion)
        self.dec_block2 = DecoderBlock(in_channels=96,  out_channels=48,
                                         w_channels=d_pe, fusion=self.cfg.fusion)
        self.dec_block3 = DecoderBlock(in_channels=48,  out_channels=24,
                                         w_channels=d_pe, fusion=self.cfg.fusion)
        self.dec_block4 = DecoderBlock(in_channels=24,  out_channels=24,
                                         w_channels=d_pe, fusion=self.cfg.fusion)

        # Final 1×1 to per-pixel logits
        self.final = nn.Conv2d(24, num_classes, kernel_size=1)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encoder forward, returns spatial feature map (B, d, H/16, W/16)."""
        # patch_embed + cls + pos_embed (using pretrained pos_embed here)
        x = self.encoder.patch_embed(image)                   # (B, N, d)
        B = x.shape[0]
        cls = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)

        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)

        # Drop CLS, reshape patches → spatial
        patches = x[:, 1:]                                    # (B, N, d)
        H_p = self.image_size // self.patch_size
        W_p = self.image_size // self.patch_size
        feat = patches.transpose(1, 2).reshape(B, -1, H_p, W_p)  # (B, d, 14, 14)
        return feat

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Encoder → spatial feature
        feat = self.encode(image)                              # (B, 192, 14, 14)

        # Wavelet branch → multi-scale features
        w_LL, w_L3, w_L2, w_L1 = self.wavelet_branch(image)
        # Shapes for 224 input:
        #   w_LL: (B, d_pe, 28, 28)
        #   w_L3: (B, d_pe, 28, 28)
        #   w_L2: (B, d_pe, 56, 56)
        #   w_L1: (B, d_pe, 112, 112)

        # Decoder: upsample step-by-step, each fusing the matching wavelet scale
        x = self.dec_block1(feat,  w_LL)                       # → 28×28×96
        x = self.dec_block2(x,     w_L3)                       # → 56×56×48
        x = self.dec_block3(x,     w_L2)                       # → 112×112×24
        x = self.dec_block4(x,     w_L1)                       # → 224×224×24

        logits = self.final(x)                                 # (B, K, 224, 224)
        return logits


# =============================================================================
# Loss + metrics
# =============================================================================

def dice_loss(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6
              ) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.
    logits: (B, 1, H, W)
    masks:  (B, 1, H, W) in {0, 1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.shape[0], -1)
    masks_flat = masks.view(masks.shape[0], -1)
    intersection = (probs * masks_flat).sum(dim=1)
    union = probs.sum(dim=1) + masks_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, masks):
    """Dice + BCE, equal weight."""
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    dl = dice_loss(logits, masks)
    return 0.5 * bce + 0.5 * dl


@torch.no_grad()
def compute_seg_metrics(logits: torch.Tensor, masks: torch.Tensor,
                         threshold: float = 0.5, eps: float = 1e-6
                         ) -> Dict[str, float]:
    """Per-batch Dice and IoU."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks_b = masks.float()

    intersection = (preds * masks_b).sum(dim=[1, 2, 3])
    pred_area = preds.sum(dim=[1, 2, 3])
    mask_area = masks_b.sum(dim=[1, 2, 3])

    dice = (2 * intersection + eps) / (pred_area + mask_area + eps)
    iou = (intersection + eps) / (pred_area + mask_area - intersection + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
    }


# =============================================================================
# Training / evaluation
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_seg(model, loader, device):
    model.eval()
    dice_total, iou_total, n_batches = 0.0, 0.0, 0
    loss_total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss_total += combined_loss(logits, masks).item()
        m = compute_seg_metrics(logits, masks)
        dice_total += m["dice"]
        iou_total += m["iou"]
        n_batches += 1
    return {
        "dice": dice_total / max(n_batches, 1),
        "iou":  iou_total / max(n_batches, 1),
        "loss": loss_total / max(n_batches, 1),
    }


def train_one_seg(cfg: ViPERSegConfig, train_loader, val_loader, test_loader,
                   image_size: int, n_epochs: int = 60, lr: float = 1e-4,
                   weight_decay: float = 1e-4, device=DEVICE, seed: int = 42,
                   verbose: bool = True, name: Optional[str] = None,
                   save_checkpoint_path: Optional[Path] = None):
    set_seed(seed)
    model = ViPERSeg(num_classes=1, image_size=image_size, patch_size=16, cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        tag = name or "viper_seg"
        print(f"[{tag}]  total_params={n_params:,}")
        print(f"  cfg: {cfg}")

    # Different LR for encoder (lower, fine-tuning pretrained weights) and decoder
    encoder_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr * cfg.encoder_lr_mult},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val": [], "epoch_times": []}
    best_val_dice, best_state = 0.0, None

    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        avg_loss = ep_loss / len(train_loader)
        val_m = evaluate_seg(model, val_loader, device)
        ep_t = time.time() - t0
        history["train_loss"].append(avg_loss)
        history["val"].append(val_m)
        history["epoch_times"].append(ep_t)

        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % max(1, n_epochs // 10) == 0:
            print(f"  ep {ep+1:3d}/{n_epochs}  tr={avg_loss:.4f}  "
                  f"val_dice={val_m['dice']:.4f}  val_iou={val_m['iou']:.4f}  "
                  f"t={ep_t:.1f}s")

    test_m = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_m = evaluate_seg(model, test_loader, device)
        if verbose:
            print(f"  TEST  dice={test_m['dice']:.4f}  iou={test_m['iou']:.4f}")

        if save_checkpoint_path is not None:
            torch.save({
                "state_dict": {k: v.cpu() for k, v in best_state.items()},
                "cfg": asdict(cfg),
                "image_size": image_size,
            }, save_checkpoint_path)

    return {
        "name":          name or "viper_seg",
        "cfg":           asdict(cfg),
        "seed":          seed,
        "n_params":      n_params,
        "best_val_dice": best_val_dice,
        "test":          test_m,
        "history":       history,
        "avg_epoch_t":   float(np.mean(history["epoch_times"])),
    }


# =============================================================================
# Suites
# =============================================================================

def get_suite(suite_name: str) -> List[Tuple[ViPERSegConfig, str]]:
    runs = []
    if suite_name == "smoke":
        cfg = ViPERSegConfig(fusion="film")
        runs = [(cfg, "viper_seg_smoke")]
    elif suite_name == "ablation":
        # Fusion mode ablation
        for fusion in ["add", "film", "concat"]:
            cfg = ViPERSegConfig(fusion=fusion)
            runs.append((cfg, f"fusion_{fusion}"))
    elif suite_name == "contenders":
        cfg = ViPERSegConfig(fusion="film")
        runs.append((cfg, "viper_seg_film"))
        cfg2 = ViPERSegConfig(fusion="add")
        runs.append((cfg2, "viper_seg_add"))
    else:
        raise ValueError(f"Unknown suite: {suite_name}")
    return runs


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke",
                        choices=["smoke", "ablation", "contenders"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="viper_seg_results")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_checkpoints", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if args.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Outputs: {out_dir.resolve()}")

    train_loader, val_loader, test_loader, nc, h, w = get_kvasir_seg(
        data_root=args.data_root, batch_size=args.batch_size,
        image_size=args.image_size, seed=args.seeds[0],
        num_workers=args.num_workers,
    )
    print(f"Kvasir-SEG: {nc} class(es), {h}×{w}")
    print(f"  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    runs = get_suite(args.suite)
    all_results = []
    for seed in args.seeds:
        for cfg, name in runs:
            run_id = f"{name}_seed{seed}"
            run_path = runs_dir / f"{run_id}.json"
            if run_path.exists():
                print(f"SKIP {run_id} (already done)")
                with open(run_path) as f:
                    all_results.append(json.load(f))
                continue
            print(f"\n{'='*70}\n{run_id}\n{'='*70}")
            ckpt_path = (ckpt_dir / f"{run_id}.pt") if args.save_checkpoints else None
            try:
                r = train_one_seg(
                    cfg, train_loader, val_loader, test_loader,
                    image_size=args.image_size,
                    n_epochs=args.epochs if args.suite != "smoke" else 3,
                    lr=args.lr, weight_decay=args.weight_decay,
                    device=DEVICE, seed=seed, name=run_id,
                    save_checkpoint_path=ckpt_path,
                )
                all_results.append(r)
                with open(run_path, "w") as f:
                    json.dump(r, f, indent=2, default=str)
            except Exception as e:
                import traceback
                print(f"[ERROR] {run_id}: {e}")
                traceback.print_exc()

    summary_path = out_dir / f"{args.suite}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}\nSaved: {summary_path}")
    print(f"{'='*70}")
    print(f"{'name':<35} {'test_dice':>10} {'test_iou':>10}")
    print("-" * 60)
    rows = [(r["name"],
             (r["test"]["dice"] if r["test"] else float("nan")),
             (r["test"]["iou"] if r["test"] else float("nan")))
            for r in all_results]
    rows.sort(key=lambda x: -x[1] if not math.isnan(x[1]) else 0)
    for name, dice, iou in rows:
        print(f"{name:<35} {dice:>10.4f} {iou:>10.4f}")


if __name__ == "__main__":
    main()
