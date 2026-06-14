"""
train_extra_pes.py — Train DeiT-Tiny + RoPE-Mixed or ALiBi-2D.

These PEs operate inside attention (not on input tokens), so they need
their own launcher. This wraps viper_v5's training loop with the attention
patching logic.

Usage:
    python train_extra_pes.py --dataset bloodmnist --pe rope_mixed \\
        --epochs 30 --seeds 42 123 7

    python train_extra_pes.py --dataset bloodmnist --pe alibi2d \\
        --epochs 30 --seeds 42 123 7
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

# Reuse from viper_v5
from viper_v5 import (
    DeiTWithCustomPE, set_seed, evaluate, load_data_loader_module,
    compute_metrics,
)
from viper_v5_extra_pes import apply_internal_pe_to_deit


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_extra_pe(pe_type: str, train_loader, val_loader, test_loader,
                         image_size: int, patch_size: int, num_classes: int,
                         n_epochs: int = 30, lr: float = 3e-5,
                         weight_decay: float = 0.05, device=DEVICE,
                         seed: int = 42, verbose: bool = True,
                         name: Optional[str] = None,
                         distance: str = "l1"):
    """Train DeiT-Tiny + internal PE (RoPE-Mixed or ALiBi-2D)."""
    set_seed(seed)

    # Build base model with pe_type='none', then patch in internal PE
    model = DeiTWithCustomPE(
        num_classes=num_classes, image_size=image_size,
        patch_size=patch_size, pe_type="none", viper_cfg=None,
    ).to(device)

    # Determine patch grid + head configuration
    H_p = image_size // patch_size
    W_p = image_size // patch_size
    num_heads = 3                    # DeiT-Tiny
    head_dim = 64                    # DeiT-Tiny: 192/3

    apply_internal_pe_to_deit(
        model, pe_type=pe_type, H_p=H_p, W_p=W_p,
        num_heads=num_heads, head_dim=head_dim, distance=distance,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pe_params = sum(p.numel() for p in model.internal_pe.parameters())
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

    return {
        "name":         name or pe_type,
        "pe_type":      pe_type,
        "seed":         seed,
        "n_params":     n_params,
        "pe_params":    pe_params,
        "best_val_acc": best_val_acc,
        "test":         test_m,
        "history":      history,
        "avg_epoch_t":  float(np.mean(history["epoch_times"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bloodmnist",
                        choices=["eurosat", "bloodmnist", "dermamnist", "pathmnist", "resisc45", "dtd", "tissuemnist", "flowers102", "fgvc_aircraft"])
    parser.add_argument("--pe", choices=["rope_mixed", "alibi2d"], required=True,
                        help="Which internal PE to use")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_loader_path", default="../data/data-loader.py")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--distance", default="l1", choices=["l1", "l2"],
                        help="For alibi2d: distance metric between patches")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"viper_v5_results_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}, PE: {args.pe}")
    print(f"image_size={args.image_size}, patch_size={args.patch_size}")
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

    all_results = []
    for seed in args.seeds:
        run_id = f"{args.pe}_seed{seed}"
        run_path = runs_dir / f"{run_id}.json"
        if run_path.exists():
            print(f"SKIP {run_id} (already done)")
            with open(run_path) as f:
                all_results.append(json.load(f))
            continue
        print(f"\n{'='*70}\n{run_id}\n{'='*70}")
        try:
            r = train_one_extra_pe(
                pe_type=args.pe,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader,
                image_size=args.image_size, patch_size=args.patch_size,
                num_classes=num_classes, n_epochs=args.epochs,
                lr=args.lr, weight_decay=args.weight_decay,
                device=DEVICE, seed=seed, name=run_id,
                distance=args.distance,
            )
            all_results.append(r)
            with open(run_path, "w") as f:
                json.dump(r, f, indent=2, default=str)
        except Exception as e:
            import traceback
            print(f"[ERROR] {run_id}: {e}")
            traceback.print_exc()

    summary_path = out_dir / f"{args.pe}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}\nSummary saved: {summary_path}")
    print(f"{'='*70}")
    print(f"{'name':<30} {'test_acc':>10} {'test_f1':>10} {'PE params':>10}")
    print("-" * 70)
    for r in all_results:
        t = r["test"]
        acc = t["acc"] if t else float("nan")
        f1 = t["f1"] if t else float("nan")
        print(f"{r['name']:<30} {acc:>10.4f} {f1:>10.4f} {r['pe_params']:>10,}")


if __name__ == "__main__":
    main()
