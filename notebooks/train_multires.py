"""
train_multires.py — Resolution-augmented training of DeiT-Tiny + ViPER/CPE.

Hypothesis: ViPER's wavelet decomposition is structurally scale-aware, so
training at multiple resolutions should let ViPER learn to handle unseen
resolutions better than CPE (whose conv is scale-agnostic but doesn't
exploit multi-scale structure).

This script:
  • Patches PE modules in DeiTWithCustomPE to handle dynamic input resolutions
  • Trains with per-batch random resolution sampling
  • Compares only PE types that natively support variable resolution:
      viper, cpe, multipeg, none
  • Saves checkpoints for downstream eval_multires.py

Usage:
    python train_multires.py --dataset bloodmnist \\
        --train_resolutions 160 192 224 256 288 \\
        --pe_types viper cpe multipeg none \\
        --epochs 30 --seeds 42 123 7 \\
        --data_loader_path ../data/data-loader.py
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

# Reuse model + helpers from viper_v5
from viper_v5 import (
    ViPERConfig, DeiTWithCustomPE, ViPERFeatureExtractor,
    CPE, PerBlockPEG, NoPE,
    set_seed, compute_metrics, evaluate, load_data_loader_module,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Make PE modules resolution-agnostic at forward time
# =============================================================================

def make_pe_dynamic(model: DeiTWithCustomPE):
    """Patch ViPER and CPE PE modules to handle dynamic input resolutions.

    The model was built for a specific image_size; this patch lets it
    actually run on different-sized inputs by computing grid dims from
    the actual image at forward time.
    """
    pe_type = model.pe_type

    # Disable timm PatchEmbed input-size assertion so any resolution works
    pe = model.backbone.patch_embed
    pe.strict_img_size = False
    if hasattr(pe, 'dynamic_img_pad'):
        pe.dynamic_img_pad = True

    if pe_type == "viper":
        ve = model.pe_module                            # ViPERFeatureExtractor

        def viper_forward(image: torch.Tensor) -> torch.Tensor:
            B, C, H, W = image.shape
            # Dynamic H_p, W_p from the actual image
            H_p = H // model.patch_size
            W_p = W // model.patch_size

            # Channel projection
            if ve.cfg.use_channel_proj:
                x = ve.channel_proj(image)
            else:
                x = image.mean(dim=1, keepdim=True)

            # DWT (works at any size)
            Yl, Yh = ve.dwt(x)

            # Modulate LL and pool to dynamic patch grid
            gate = ve._compute_gate(0).view(1, ve.d_pe, 1, 1)
            ll = Yl * gate
            if ll.shape[-1] != W_p or ll.shape[-2] != H_p:
                ll = F.adaptive_avg_pool2d(ll, (H_p, W_p))
            acc = ll

            for j in range(ve.cfg.n_levels):
                details = Yh[j]
                for d in range(3):
                    sb_idx = 1 + j * 3 + d
                    sb = details[:, :, d]
                    g = ve._compute_gate(sb_idx).view(1, ve.d_pe, 1, 1)
                    sb = sb * g
                    if sb.shape[-1] != W_p or sb.shape[-2] != H_p:
                        sb = F.adaptive_avg_pool2d(sb, (H_p, W_p))
                    acc = acc + sb

            pe_seq = acc.permute(0, 2, 3, 1).reshape(B, -1, ve.d_pe)
            pe_seq = ve.proj_to_dmodel(pe_seq)
            cls = ve.cls_pe.expand(B, -1, -1)
            return torch.cat([cls, pe_seq], dim=1)

        # Bind the new forward
        ve.forward = viper_forward

    elif pe_type == "cpe":
        cpe = model.pe_module

        def cpe_forward(x_tokens: torch.Tensor) -> torch.Tensor:
            cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
            B, N, C = pts.shape
            side = int(math.sqrt(N))
            sp = pts.transpose(1, 2).reshape(B, C, side, side)
            sp = cpe.proj(sp) + sp
            pts = sp.flatten(2).transpose(1, 2)
            return torch.cat([cls + cpe.cls_pe.expand(B, -1, -1), pts], dim=1)

        cpe.forward = cpe_forward

    elif pe_type == "multipeg":
        # Each PerBlockPEG needs dynamic H, W handling
        for peg in model.per_block:
            def make_peg_forward(peg_mod):
                def peg_forward(x_tokens):
                    cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
                    B, N, C = pts.shape
                    side = int(math.sqrt(N))
                    sp = pts.transpose(1, 2).reshape(B, C, side, side)
                    sp = peg_mod.conv(sp) + sp
                    pts = sp.flatten(2).transpose(1, 2)
                    return torch.cat([cls, pts], dim=1)
                return peg_forward
            peg.forward = make_peg_forward(peg)

    elif pe_type == "none":
        # NoPE returns a fixed-size zero tensor — patch it to return dynamic shape
        nope = model.pe_module

        def nope_forward(x, image=None):
            return torch.zeros(x.shape[0], x.shape[1], nope.d_model,
                                device=x.device, dtype=x.dtype)

        nope.forward = nope_forward

    else:
        raise ValueError(
            f"pe_type={pe_type} doesn't natively support multi-resolution. "
            f"Use one of: viper, cpe, multipeg, none."
        )


# =============================================================================
# Training with random per-batch resolution
# =============================================================================

def resize_batch(imgs: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize a batch of images to (target_size, target_size)."""
    return F.interpolate(imgs, size=(target_size, target_size),
                          mode="bilinear", align_corners=False)


def train_one_multires(pe_type, viper_cfg, train_loader, val_loader,
                        test_loader, base_image_size, patch_size, num_classes,
                        train_resolutions: List[int],
                        n_epochs=30, lr=3e-5, weight_decay=0.05,
                        device=DEVICE, seed=42, verbose=True, name=None,
                        save_checkpoint_path: Optional[Path] = None):
    set_seed(seed)

    # Build model at the LARGEST resolution to make sure DeiT's frozen pos_embed
    # buffer is sized adequately — but pos_embed is zeroed anyway.
    # Actually we build at base_image_size and patch the PE modules to be dynamic.
    model = DeiTWithCustomPE(
        num_classes=num_classes, image_size=base_image_size,
        patch_size=patch_size, pe_type=pe_type, viper_cfg=viper_cfg,
    ).to(device)
    make_pe_dynamic(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        print(f"  multi-res training: {train_resolutions}")

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val": [], "epoch_times": [], "res_used": []}
    best_val_acc, best_state = 0.0, None

    rng = random.Random(seed)

    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        ep_resolutions = []
        for imgs, labels in train_loader:
            # Sample resolution for this batch
            target_res = rng.choice(train_resolutions)
            ep_resolutions.append(target_res)
            imgs_resized = resize_batch(imgs.to(device), target_res)
            labels = labels.to(device)

            opt.zero_grad()
            loss = crit(model(imgs_resized), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        sched.step()
        avg_tr = tr_loss / len(train_loader)

        # Validate at base_image_size (the canonical evaluation resolution)
        val_m = evaluate(model, val_loader, device, num_classes)
        ep_t = time.time() - t0
        history["train_loss"].append(avg_tr)
        history["val"].append(val_m)
        history["epoch_times"].append(ep_t)
        history["res_used"].append(ep_resolutions)

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % max(1, n_epochs // 5) == 0:
            res_counts = {r: ep_resolutions.count(r) for r in train_resolutions}
            print(f"  ep {ep+1:3d}/{n_epochs}  tr={avg_tr:.4f}  "
                  f"val_acc={val_m['acc']:.4f}  f1={val_m['f1']:.4f}  "
                  f"t={ep_t:.1f}s  res={res_counts}")

    test_m = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_m = evaluate(model, test_loader, device, num_classes)
        if verbose:
            print(f"  TEST  acc={test_m['acc']:.4f}  f1={test_m['f1']:.4f}  "
                  f"auc={test_m['auc']:.4f}")

        if save_checkpoint_path is not None:
            torch.save({
                "state_dict": {k: v.cpu() for k, v in best_state.items()},
                "pe_type": pe_type,
                "viper_cfg": asdict(viper_cfg) if viper_cfg else None,
                "image_size": base_image_size,
                "patch_size": patch_size,
                "num_classes": num_classes,
                "multires_training": True,
                "train_resolutions": train_resolutions,
            }, save_checkpoint_path)

    return {
        "name":         name or pe_type,
        "pe_type":      pe_type,
        "viper_cfg":    asdict(viper_cfg) if viper_cfg else None,
        "seed":         seed,
        "n_params":     n_params,
        "pe_params":    pe_params,
        "train_resolutions": train_resolutions,
        "best_val_acc": best_val_acc,
        "test":         test_m,
        "history":      history,
        "avg_epoch_t":  float(np.mean(history["epoch_times"])),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bloodmnist",
                        choices=["eurosat", "bloodmnist", "dermamnist",
                                 "pathmnist", "resisc45", "dtd"])
    parser.add_argument("--base_image_size", type=int, default=224,
                        help="Canonical resolution for model construction and val")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--train_resolutions", type=int, nargs="+",
                        default=[160, 192, 224, 256, 288])
    parser.add_argument("--pe_types", nargs="+",
                        default=["viper", "cpe", "multipeg", "none"],
                        help="PE types to train (must support dynamic res)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Smaller default since the largest res is 288")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_loader_path", default="../data/data-loader.py")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"viper_v5_multires_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Base image size: {args.base_image_size}, patch_size: {args.patch_size}")
    print(f"Training resolutions: {args.train_resolutions}")
    print(f"PE types to train: {args.pe_types}")
    print(f"Outputs: {out_dir.resolve()}")

    # Load data at base_image_size — we'll resize per batch
    dl = load_data_loader_module(args.data_loader_path)
    os.makedirs(args.data_root, exist_ok=True)
    train_loader, val_loader, test_loader, num_classes, img_h, img_w = \
        dl.get_dataset(args.dataset, data_root=args.data_root,
                       batch_size=args.batch_size, image_size=args.base_image_size,
                       seed=args.seeds[0], num_workers=args.num_workers)
    print(f"  {num_classes} classes, {img_h}×{img_w}")
    print(f"  train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  test={len(test_loader.dataset):,}")

    # Build the run list
    runs = []
    for pe in args.pe_types:
        if pe == "viper":
            cfg = ViPERConfig(n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32)
            runs.append(("viper", cfg, "viper_multires"))
        elif pe in ["cpe", "multipeg", "none"]:
            runs.append((pe, None, f"{pe}_multires"))
        else:
            print(f"WARNING: skipping {pe} — not supported by multi-res training")

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
            ckpt_path = ckpt_dir / f"{run_id}.pt"
            try:
                r = train_one_multires(
                    pe_type, cfg,
                    train_loader, val_loader, test_loader,
                    base_image_size=args.base_image_size,
                    patch_size=args.patch_size,
                    num_classes=num_classes,
                    train_resolutions=args.train_resolutions,
                    n_epochs=args.epochs,
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

    summary_path = out_dir / "multires_train_summary.json"
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

    print("\n→ Next: run eval_multires.py on the checkpoints to test")
    print(f"  resolution generalization (96, 160, 224, 320, 448).")


if __name__ == "__main__":
    main()
