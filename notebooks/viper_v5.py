"""
ViPER v5 — Pretrained DeiT-Tiny backbone with swappable Positional Encoding.

This experiment tests whether ViPER (and its competitors) show clearer
differentiation when starting from a strong pretrained backbone, rather
than training a small ViT from scratch on tiny datasets.

Setup:
  • Backbone: DeiT-Tiny (timm), 12 layers, d=192, ImageNet-1K pretrained
  • PE module: we strip DeiT's learned absolute PE and replace it with
    each candidate (or apply per-block for multipeg-style methods)
  • Fine-tune: 30 epochs at lr=3e-5 (recommended) with cosine schedule
  • Evaluation: standard test set + multi-resolution generalization

ViPER variant tested:
  This file uses ViPER v1 (DyWPE-faithful, input-only): wavelet feature
  map computed once, projected to d_model, added DIRECTLY to patch
  embeddings before the first transformer block. This is the closest 2D
  analog to DyWPE's 1D operation.

Run examples
------------

# Smoke test (3 epochs):
python viper_v5.py --dataset bloodmnist --suite smoke --epochs 3

# Main suite (all PEs, single seed):
python viper_v5.py --dataset bloodmnist --suite main --epochs 30

# Multi-seed contenders:
python viper_v5.py --dataset bloodmnist --suite contenders \\
    --epochs 30 --seeds 42 123 7

# Multi-resolution evaluation (uses checkpoints saved during training):
python viper_v5.py --dataset bloodmnist --suite eval_multires \\
    --checkpoint_dir viper_v5_results_bloodmnist/checkpoints
"""

import argparse
import importlib.util
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
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, roc_auc_score

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")

from pytorch_wavelets import DWTForward


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ViPERConfig:
    """ViPER v1 (DyWPE-faithful, input-only) configuration."""
    n_levels:        int = 3
    wavelet:         str = "db4"
    channel_mode:    str = "gray"
    d_pe:            int = 32           # internal dim of wavelet branch
    use_channel_proj: bool = True
    use_gating:       bool = True


# =============================================================================
# ViPER v1 — DyWPE-faithful input-only PE
# =============================================================================

class ChannelProjection(nn.Module):
    """RGB → 1 channel for DWT input."""
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
        else:
            raise ValueError(f"Unknown channel_mode: {mode}")

    def forward(self, x):
        if self.mode == "gray":
            return (x * self.rgb2gray).sum(dim=1, keepdim=True)
        return self.proj(x)


class ViPERFeatureExtractor(nn.Module):
    """ViPER v1: image → wavelet feature map → patch grid → projected to d_model.

    Pipeline (DyWPE-faithful):
       1. Channel projection (RGB → 1ch)
       2. 2D DWT, J levels
       3. For each subband: gate (σ⊙tanh) ⊗ broadcasted coefficients → (B, d_pe, h, w)
       4. Average-pool each subband to patch grid (H_p, W_p), sum across subbands
       5. Project d_pe → d_model via 1×1 conv

    Output: (B, N+1, d_model) — PE tokens ready to add to patch embeddings.
    """
    def __init__(self, cfg: ViPERConfig, in_channels: int, image_h: int,
                 image_w: int, patch_size: int, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.image_h, self.image_w = image_h, image_w
        self.H_p = image_h // patch_size
        self.W_p = image_w // patch_size
        self.d_pe = cfg.d_pe

        # Channel projection
        if cfg.use_channel_proj:
            self.channel_proj = ChannelProjection(in_channels, cfg.channel_mode)
        else:
            self.channel_proj = nn.Identity()

        # 2D DWT
        self.dwt = DWTForward(J=cfg.n_levels, wave=cfg.wavelet, mode="symmetric")

        # Gating (DyWPE-style scale embeddings + gate matrices)
        self.n_subbands = 1 + 3 * cfg.n_levels       # LL + (LH,HL,HH) per level
        if cfg.use_gating:
            self.scale_embeddings = nn.Parameter(
                torch.randn(self.n_subbands, cfg.d_pe) * 0.02
            )
            self.W_g = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)
            self.W_v = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)

        # Project d_pe → d_model after combining subbands at patch grid
        self.proj_to_dmodel = nn.Linear(cfg.d_pe, d_model)

        # CLS token PE (learnable, zero-initialized so CLS gets no positional info)
        self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))

    def _compute_gate(self, sb_idx: int) -> torch.Tensor:
        if not self.cfg.use_gating:
            device = next(self.parameters()).device
            return torch.ones(self.d_pe, device=device)
        e = self.scale_embeddings[sb_idx]
        g = torch.sigmoid(self.W_g(e))
        v = torch.tanh(self.W_v(e))
        return g * v

    def _modulate(self, coeffs: torch.Tensor, sb_idx: int) -> torch.Tensor:
        """coeffs: (B, 1, H, W) → (B, d_pe, H, W) via gate-broadcast."""
        gate = self._compute_gate(sb_idx).view(1, self.d_pe, 1, 1)
        return coeffs * gate

    def _pool_to_patch_grid(self, x):
        _, _, h, w = x.shape
        if h == self.H_p and w == self.W_p:
            return x
        if h >= self.H_p and w >= self.W_p:
            return F.adaptive_avg_pool2d(x, (self.H_p, self.W_p))
        return F.interpolate(x, size=(self.H_p, self.W_p),
                             mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, C, H, W) → pe_tokens: (B, N+1, d_model)"""
        if self.cfg.use_channel_proj:
            x = self.channel_proj(image)
        else:
            x = image.mean(dim=1, keepdim=True)

        Yl, Yh = self.dwt(x)

        # Modulate LL and pool to patch grid
        ll = self._modulate(Yl, 0)
        ll = self._pool_to_patch_grid(ll)
        accumulator = ll

        # Modulate detail subbands and accumulate
        for j in range(self.cfg.n_levels):
            details = Yh[j]
            for d in range(3):
                sb_idx = 1 + j * 3 + d
                sb = details[:, :, d]
                sb = self._modulate(sb, sb_idx)
                sb = self._pool_to_patch_grid(sb)
                accumulator = accumulator + sb

        # accumulator: (B, d_pe, H_p, W_p) → (B, N, d_pe) → (B, N, d_model)
        B = accumulator.shape[0]
        pe_seq = accumulator.permute(0, 2, 3, 1).reshape(B, -1, self.d_pe)
        pe_seq = self.proj_to_dmodel(pe_seq)                # (B, N, d_model)

        # Prepend CLS PE
        cls = self.cls_pe.expand(B, -1, -1)                  # (B, 1, d_model)
        return torch.cat([cls, pe_seq], dim=1)               # (B, N+1, d_model)


# =============================================================================
# Baseline PEs (all designed to output (B, N+1, d_model) for input-only addition)
# =============================================================================

class NoPE(nn.Module):
    """Returns zeros — no positional information."""
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
    def forward(self, x, image=None):
        return torch.zeros(x.shape[0], self.num_tokens, self.d_model,
                            device=x.device, dtype=x.dtype)


class LearnedPE(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
    def forward(self, x, image=None):
        return self.pe.expand(x.shape[0], -1, -1)


class SinCos2DPE(nn.Module):
    def __init__(self, H: int, W: int, d_model: int):
        super().__init__()
        assert d_model % 4 == 0
        d_q = d_model // 4
        y = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        xs = torch.arange(W, dtype=torch.float32).unsqueeze(1)
        wt = 1.0 / (10000 ** (torch.arange(d_q, dtype=torch.float32) / d_q))
        enc_h = torch.cat([torch.sin(y*wt), torch.cos(y*wt)], dim=-1)
        enc_w = torch.cat([torch.sin(xs*wt), torch.cos(xs*wt)], dim=-1)
        enc_h = enc_h.unsqueeze(1).expand(H, W, -1)
        enc_w = enc_w.unsqueeze(0).expand(H, W, -1)
        pe = torch.cat([enc_h, enc_w], dim=-1).reshape(1, H*W, d_model)
        cls_pe = torch.zeros(1, 1, d_model)
        self.register_buffer("pe", torch.cat([cls_pe, pe], dim=1))
    def forward(self, x, image=None):
        return self.pe.expand(x.shape[0], -1, -1)


class Relative2DPE(nn.Module):
    """Per-position embedding derived from average relative offset (input-only)."""
    def __init__(self, H: int, W: int, d_model: int):
        super().__init__()
        self.H, self.W = H, W
        self.rel_h = nn.Embedding(2*H - 1, d_model // 2)
        self.rel_w = nn.Embedding(2*W - 1, d_model // 2)
        self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))
    def forward(self, x, image=None):
        B = x.shape[0]; device = x.device
        rows = torch.arange(self.H, device=device).repeat_interleave(self.W)
        cols = torch.arange(self.W, device=device).repeat(self.H)
        rel_r = (rows.unsqueeze(1) - rows.unsqueeze(0)) + (self.H - 1)
        rel_c = (cols.unsqueeze(1) - cols.unsqueeze(0)) + (self.W - 1)
        emb_r = self.rel_h(rel_r).mean(dim=1)
        emb_c = self.rel_w(rel_c).mean(dim=1)
        pe = torch.cat([emb_r, emb_c], dim=-1).unsqueeze(0).expand(B, -1, -1)
        cls = self.cls_pe.expand(B, -1, -1)
        return torch.cat([cls, pe], dim=1)


# =============================================================================
# Per-block PE: CPE (single PEG at input) and MultiPEG (per-block PEG)
# =============================================================================

class CPE(nn.Module):
    """Single-PEG CPE: depthwise zero-padded conv applied once at input.
       Outputs the PE adjustment to ADD to tokens.
    """
    def __init__(self, H: int, W: int, d_model: int, k: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.proj = nn.Conv2d(d_model, d_model, k, padding=k//2,
                               groups=d_model, padding_mode='zeros')
        self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x_tokens):
        """x_tokens: (B, N+1, d_model). Returns same shape with PE applied."""
        cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)
        sp = self.proj(sp) + sp                     # CPE: conv output + residual
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls + self.cls_pe.expand(B, -1, -1), pts], dim=1)


class PerBlockPEG(nn.Module):
    """Single depthwise zero-padded conv to be applied at one block.
       Stored as a list inside the model for multipeg.
    """
    def __init__(self, H: int, W: int, d_model: int, k: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.conv = nn.Conv2d(d_model, d_model, k, padding=k//2,
                               groups=d_model, padding_mode='zeros')

    def forward(self, x_tokens):
        cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)
        sp = self.conv(sp) + sp
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls, pts], dim=1)


# =============================================================================
# DeiT-Tiny wrapper with swappable PE
# =============================================================================

class DeiTWithCustomPE(nn.Module):
    """Wraps timm DeiT-Tiny: strips its built-in absolute PE and applies
       a candidate PE module instead.

    Two modes:
       • Input-only PEs (none, learned, sincos2d, relative2d, cpe, viper_v1):
         PE adjustment is added to patch+cls embeddings before block 0.
       • Per-block PEs (multipeg): the model holds a list of PerBlockPEG
         modules and applies one before each transformer block.
    """
    def __init__(self, num_classes: int, image_size: int, patch_size: int,
                 pe_type: str, viper_cfg: Optional[ViPERConfig] = None,
                 num_blocks_for_peg: int = 12):
        super().__init__()
        self.pe_type = pe_type
        self.viper_cfg = viper_cfg
        self.image_size = image_size
        self.patch_size = patch_size

        # Load pretrained DeiT-Tiny
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=num_classes,
            img_size=image_size,
        )

        d_model = self.backbone.embed_dim                    # 192
        H_p = image_size // patch_size
        W_p = image_size // patch_size
        num_patches = H_p * W_p
        self.H_p, self.W_p = H_p, W_p
        self.d_model = d_model
        self.num_blocks = len(self.backbone.blocks)

        # KILL the built-in pos_embed by zero-initializing it and freezing it
        # (so all positional info comes from our PE module instead)
        with torch.no_grad():
            self.backbone.pos_embed.zero_()
        self.backbone.pos_embed.requires_grad = False

        # Build the candidate PE module
        if pe_type == "viper":
            assert viper_cfg is not None
            self.pe_module = ViPERFeatureExtractor(
                viper_cfg, in_channels=3, image_h=image_size, image_w=image_size,
                patch_size=patch_size, d_model=d_model,
            )
            self.per_block = None
        elif pe_type == "none":
            self.pe_module = NoPE(num_patches + 1, d_model)
            self.per_block = None
        elif pe_type == "learned":
            self.pe_module = LearnedPE(num_patches + 1, d_model)
            self.per_block = None
        elif pe_type == "sincos2d":
            self.pe_module = SinCos2DPE(H_p, W_p, d_model)
            self.per_block = None
        elif pe_type == "relative2d":
            self.pe_module = Relative2DPE(H_p, W_p, d_model)
            self.per_block = None
        elif pe_type == "cpe":
            self.pe_module = CPE(H_p, W_p, d_model)
            self.per_block = None
        elif pe_type == "multipeg":
            self.pe_module = None
            self.per_block = nn.ModuleList([
                PerBlockPEG(H_p, W_p, d_model) for _ in range(self.num_blocks)
            ])
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

    def _adjust_for_resolution(self, image_size: int):
        """Reconfigure for a different image resolution (multi-res eval)."""
        H_p = image_size // self.patch_size
        W_p = image_size // self.patch_size
        if H_p != self.H_p or W_p != self.W_p:
            self.H_p, self.W_p = H_p, W_p
            # PE modules that depend on resolution must be rebuilt
            # For simplicity we just warn here — multi-res eval handled separately

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]

        # Patch embed
        x = self.backbone.patch_embed(image)                 # (B, N, d)

        # Prepend CLS
        cls_tok = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x = torch.cat([cls_tok, x], dim=1)                   # (B, N+1, d)

        # Apply input-only PE (or pass through for multipeg)
        if self.pe_module is not None:
            if self.pe_type == "cpe":
                x = self.pe_module(x)                         # CPE returns (B, N+1, d)
            elif self.pe_type == "viper":
                pe_tokens = self.pe_module(image)              # (B, N+1, d)
                x = x + pe_tokens
            else:
                pe_tokens = self.pe_module(x)                 # (B, N+1, d)
                x = x + pe_tokens

        # Dropout (matches DeiT's pos_drop)
        x = self.backbone.pos_drop(x)

        # Pass through transformer blocks (with optional per-block PEG)
        for i, block in enumerate(self.backbone.blocks):
            if self.per_block is not None:
                x = self.per_block[i](x)
            x = block(x)

        # Norm + head
        x = self.backbone.norm(x)
        cls_out = x[:, 0]
        return self.backbone.head(cls_out)


# =============================================================================
# Training utilities (mostly unchanged from v4)
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(logits, labels, num_classes):
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    y = labels.cpu().numpy()
    acc = (preds == y).mean()
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y, probs, multi_class="ovr", average="macro") \
              if len(np.unique(y)) == num_classes else float("nan")
    except Exception:
        auc = float("nan")
    return {"acc": float(acc), "f1": float(f1), "auc": float(auc)}


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    logits_all, labels_all = [], []
    crit = nn.CrossEntropyLoss()
    loss_total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss_total += crit(out, labels).item()
        logits_all.append(out)
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    m = compute_metrics(logits, labels, num_classes)
    m["loss"] = loss_total / len(loader)
    return m


def train_one_v5(pe_type, viper_cfg, train_loader, val_loader, test_loader,
                 image_size, patch_size, num_classes,
                 n_epochs=30, lr=3e-5, weight_decay=0.05,
                 device=DEVICE, seed=42, verbose=True, name=None,
                 save_checkpoint_path: Optional[Path] = None):
    set_seed(seed)
    model = DeiTWithCustomPE(
        num_classes=num_classes, image_size=image_size,
        patch_size=patch_size, pe_type=pe_type, viper_cfg=viper_cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count PE-specific params (only what we added or that's the candidate PE)
    if pe_type == "viper":
        pe_params = sum(p.numel() for p in model.pe_module.parameters())
    elif pe_type == "multipeg":
        pe_params = sum(p.numel() for p in model.per_block.parameters())
    elif model.pe_module is not None:
        pe_params = sum(p.numel() for p in model.pe_module.parameters()
                        if p.requires_grad)
    else:
        pe_params = 0

    if verbose:
        tag = name or pe_type
        print(f"[{tag}]  trainable={n_params:,}  PE={pe_params:,}")

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val": [], "epoch_times": []}
    best_val_acc, best_state = 0.0, None

    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()
        avg_tr = tr_loss / len(train_loader)
        val_m = evaluate(model, val_loader, device, num_classes)
        ep_t = time.time() - t0
        history["train_loss"].append(avg_tr)
        history["val"].append(val_m)
        history["epoch_times"].append(ep_t)
        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % max(1, n_epochs // 5) == 0:
            print(f"  ep {ep+1:3d}/{n_epochs}  tr={avg_tr:.4f}  "
                  f"val_acc={val_m['acc']:.4f}  f1={val_m['f1']:.4f}  "
                  f"auc={val_m['auc']:.4f}  t={ep_t:.1f}s")

    test_m = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_m = evaluate(model, test_loader, device, num_classes)
        if verbose:
            print(f"  TEST  acc={test_m['acc']:.4f}  f1={test_m['f1']:.4f}  "
                  f"auc={test_m['auc']:.4f}")

        # Save checkpoint for later multi-res evaluation
        if save_checkpoint_path is not None:
            torch.save({
                "state_dict": {k: v.cpu() for k, v in best_state.items()},
                "pe_type": pe_type,
                "viper_cfg": asdict(viper_cfg) if viper_cfg else None,
                "image_size": image_size,
                "patch_size": patch_size,
                "num_classes": num_classes,
            }, save_checkpoint_path)

    return {
        "name":         name or pe_type,
        "pe_type":      pe_type,
        "viper_cfg":    asdict(viper_cfg) if viper_cfg else None,
        "seed":         seed,
        "n_params":     n_params,
        "pe_params":    pe_params,
        "best_val_acc": best_val_acc,
        "test":         test_m,
        "history":      history,
        "avg_epoch_t":  float(np.mean(history["epoch_times"])),
    }


# =============================================================================
# Suites
# =============================================================================

def get_default_viper_cfg():
    return ViPERConfig(n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32)


def get_suite(suite_name: str) -> List[Tuple[str, Optional[ViPERConfig], str]]:
    runs = []
    if suite_name == "smoke":
        cfg = get_default_viper_cfg()
        runs = [("viper", cfg, "viper_v5_smoke")]
    elif suite_name == "baselines":
        for pe in ["none", "learned", "sincos2d", "relative2d", "cpe", "multipeg"]:
            runs.append((pe, None, pe))
    elif suite_name == "main":
        for pe in ["none", "learned", "sincos2d", "relative2d", "cpe", "multipeg"]:
            runs.append((pe, None, pe))
        cfg = get_default_viper_cfg()
        runs.append(("viper", cfg, "viper_v5"))
    elif suite_name == "contenders":
        for pe in ["cpe", "multipeg", "learned", "none"]:
            runs.append((pe, None, pe))
        cfg = get_default_viper_cfg()
        runs.append(("viper", cfg, "viper_v5"))
    else:
        raise ValueError(f"Unknown suite: {suite_name}")
    return runs


# =============================================================================
# Dataset + main
# =============================================================================

def load_data_loader_module(data_loader_path: str):
    spec = importlib.util.spec_from_file_location("dl", data_loader_path)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bloodmnist",
                        choices=["eurosat", "bloodmnist", "dermamnist",
                                 "pathmnist", "resisc45", "dtd"])
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size (default 224 for DeiT-Tiny)")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size (DeiT-Tiny uses 16)")
    parser.add_argument("--suite", default="main",
                        choices=["smoke", "baselines", "main", "contenders"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_loader_path", default="../data/data-loader.py")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save best checkpoints for multi-res eval later")
    args = parser.parse_args()

    if args.image_size != 224:
        print(f"WARNING: DeiT-Tiny is pretrained at 224; using {args.image_size} "
              f"will require some interpolation/reinit")

    out_dir = Path(args.out_dir or f"viper_v5_results_{args.dataset}")
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
    print(f"Dataset: {args.dataset}  image_size={args.image_size}  patch_size={args.patch_size}")
    print(f"Outputs: {out_dir.resolve()}")

    dl = load_data_loader_module(args.data_loader_path)
    os.makedirs(args.data_root, exist_ok=True)
    train_loader, val_loader, test_loader, num_classes, img_h, img_w = \
        dl.get_dataset(args.dataset, data_root=args.data_root,
                       batch_size=args.batch_size, image_size=args.image_size,
                       seed=args.seeds[0], num_workers=args.num_workers)
    print(f"  {num_classes} classes, {img_h}×{img_w}")
    print(f"  train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  test={len(test_loader.dataset):,}")

    runs = get_suite(args.suite)
    all_results = []
    for seed in args.seeds:
        for pe_type, cfg, name in runs:
            run_id = f"{name}_seed{seed}"
            run_path = runs_dir / f"{run_id}.json"
            if run_path.exists():
                print(f"SKIP {run_id} (already done)")
                with open(run_path) as f:
                    all_results.append(json.load(f))
                continue
            print(f"\n{'='*70}\n{run_id}\n{'='*70}")
            if cfg:
                print(f"  cfg: {cfg}")
            ckpt_path = (ckpt_dir / f"{run_id}.pt") if args.save_checkpoints else None
            try:
                r = train_one_v5(
                    pe_type, cfg,
                    train_loader, val_loader, test_loader,
                    image_size=args.image_size, patch_size=args.patch_size,
                    num_classes=num_classes,
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
    print(f"\n{'='*70}\nSummary saved: {summary_path}")
    print(f"{'='*70}")
    print(f"{'name':<35} {'pe_type':<12} {'test_acc':>10} {'test_f1':>10} {'PE params':>10}")
    print("-" * 82)
    rows = [(r["name"], r["pe_type"],
             (r["test"]["acc"] if r["test"] else float("nan")),
             (r["test"]["f1"] if r["test"] else float("nan")),
             r["pe_params"])
            for r in all_results]
    rows.sort(key=lambda x: -x[2] if not math.isnan(x[2]) else 0)
    for name, pe_type, acc, f1, pe_p in rows:
        print(f"{name:<35} {pe_type:<12} {acc:>10.4f} {f1:>10.4f} {pe_p:>10,}")


if __name__ == "__main__":
    main()
