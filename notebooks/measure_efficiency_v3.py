"""
measure_efficiency_v3.py — RELIABLE efficiency measurement.

Fixes v2 issues:
  - Multiple full passes through all methods (3x by default)
  - Random ordering within each pass to dilute warmup/cooldown effects
  - Longer warmup, more timed iterations
  - CUDA event timing (sub-ms accurate)
  - Reports median throughput across passes + std dev for honesty

Usage:
    # Requires SOLE access to the GPU — check `nvidia-smi` first.
    python measure_efficiency_v3.py --image_size 224 --batch_size 32 --n_passes 3
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("Install fvcore: pip install fvcore")
    raise

from viper_v5 import DeiTWithCustomPE, ViPERConfig
from viper_v5_extra_pes import apply_internal_pe_to_deit


def measure_throughput_robust(model, x, n_warmup=20, n_iter=200):
    """High-accuracy throughput via CUDA events."""
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    B = x.shape[0]

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_iter):
                _ = model(x)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                _ = model(x)
        elapsed = time.perf_counter() - t0

    return (B * n_iter) / elapsed


def estimate_dwt_flops(H, W, n_levels=3, filter_len=8):
    total = 0
    h, w = H, W
    for j in range(n_levels):
        row = h * (w // 2) * filter_len * 2 * 2
        col = 4 * (h // 2) * (w // 2) * filter_len * 2 * 2
        total += row + col
        h //= 2
        w //= 2
    return total


def measure_viper_extras_flops(image_size, n_levels=3, d_pe=32, d_model=192,
                                  patch_size=16):
    H_p = image_size // patch_size
    chproj = 3 * image_size * image_size * 2
    n_subbands = 1 + 3 * n_levels
    gate = n_subbands * (d_pe * d_pe * 2 + d_pe * 2)
    pool = n_subbands * H_p * H_p * d_pe
    proj = H_p * H_p * d_pe * d_model * 2
    return chproj + gate + pool + proj


def measure_flops(model, x, pe_type, image_size, backbone_flops=None):
    model.eval()
    try:
        flops_obj = FlopCountAnalysis(model, x)
        flops_obj.unsupported_ops_warnings(False)
        flops = flops_obj.total()
        if flops <= 0:
            raise RuntimeError("non-positive")
        # For ViPER, augment with manual DWT estimate (fvcore misses pytorch_wavelets)
        if pe_type == "viper":
            flops += estimate_dwt_flops(image_size, image_size)
        return flops
    except Exception:
        if pe_type == "viper" and backbone_flops:
            dwt = estimate_dwt_flops(image_size, image_size)
            extras = measure_viper_extras_flops(image_size)
            return backbone_flops + dwt + extras
        return backbone_flops if backbone_flops else 0


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_pe_params(model, pe_type):
    if pe_type in ("rope_mixed", "alibi2d"):
        return sum(p.numel() for p in model.internal_pe.parameters())
    if hasattr(model, "pe_module") and model.pe_module is not None:
        return sum(p.numel() for p in model.pe_module.parameters() if p.requires_grad)
    if hasattr(model, "per_block") and model.per_block is not None:
        return sum(p.numel() for p in model.per_block.parameters())
    return 0


def build_model(pe_type, image_size, patch_size, num_classes, device):
    if pe_type == "viper":
        cfg = ViPERConfig(n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32)
        return DeiTWithCustomPE(num_classes=num_classes, image_size=image_size,
                                  patch_size=patch_size, pe_type="viper",
                                  viper_cfg=cfg).to(device)
    elif pe_type in ("rope_mixed", "alibi2d"):
        model = DeiTWithCustomPE(num_classes=num_classes, image_size=image_size,
                                   patch_size=patch_size, pe_type="none",
                                   viper_cfg=None).to(device)
        H_p = image_size // patch_size
        apply_internal_pe_to_deit(model, pe_type=pe_type, H_p=H_p, W_p=H_p,
                                     num_heads=3, head_dim=64)
        return model
    else:
        return DeiTWithCustomPE(num_classes=num_classes, image_size=image_size,
                                  patch_size=patch_size, pe_type=pe_type,
                                  viper_cfg=None).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_passes", type=int, default=3,
                        help="Number of full passes through all methods")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="efficiency_results")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size}, Patch size: {args.patch_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of passes: {args.n_passes} (with randomized order)")
    print(f"Warmup: 20 iter | Timed: 200 iter\n")

    if args.device == "cuda":
        print("IMPORTANT: Make sure no other jobs are running on this GPU!")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()

    methods = ["none", "learned", "sincos2d", "relative2d", "cpe", "multipeg",
               "rope_mixed", "alibi2d", "viper"]

    # First measure 'none' to use as backbone reference for ViPER's manual DWT
    print("=== Computing 'none' backbone FLOPs reference ===")
    none_model = build_model("none", args.image_size, args.patch_size,
                              args.num_classes, args.device)
    x_single = torch.randn(1, 3, args.image_size, args.image_size,
                            device=args.device)
    backbone_flops_obj = FlopCountAnalysis(none_model, x_single)
    backbone_flops_obj.unsupported_ops_warnings(False)
    backbone_flops = backbone_flops_obj.total()
    print(f"Backbone FLOPs: {backbone_flops/1e9:.4f} G\n")
    del none_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Collect throughputs across passes
    throughputs = {m: [] for m in methods}
    metadata = {}  # params and FLOPs (don't change across passes)

    for pass_idx in range(args.n_passes):
        order = methods.copy()
        random.shuffle(order)
        print(f"\n{'='*70}")
        print(f"PASS {pass_idx + 1}/{args.n_passes} — order: {order}")
        print('='*70)
        for pe_type in order:
            try:
                model = build_model(pe_type, args.image_size, args.patch_size,
                                      args.num_classes, args.device)
                model.eval()

                # Measure FLOPs and params only on first pass
                if pe_type not in metadata:
                    flops = measure_flops(model, x_single, pe_type,
                                            args.image_size, backbone_flops)
                    total_p = count_params(model)
                    pe_p = count_pe_params(model, pe_type)
                    metadata[pe_type] = {
                        "total_params": total_p,
                        "pe_params": pe_p,
                        "flops_g": flops / 1e9,
                    }

                # Throughput
                x_batch = torch.randn(args.batch_size, 3, args.image_size,
                                      args.image_size, device=args.device)
                tput = measure_throughput_robust(model, x_batch)
                throughputs[pe_type].append(tput)
                print(f"  {pe_type:<14} {tput:>10.1f} img/s")

                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"  ERROR {pe_type}: {e}")
                traceback.print_exc()

    # Aggregate
    results = []
    for m in methods:
        if not throughputs[m]:
            continue
        tputs = throughputs[m]
        median_t = float(np.median(tputs))
        std_t = float(np.std(tputs))
        results.append({
            "method": m,
            **metadata[m],
            "throughput_median": median_t,
            "throughput_std": std_t,
            "throughput_all_passes": tputs,
            "batch_size": args.batch_size,
        })

    out_file = out_dir / "efficiency_v3_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}\n")

    # Final table
    print("=" * 110)
    print(f"{'Method':<14} {'Total Params':>14} {'PE Params':>12} "
          f"{'FLOPs (G)':>12} {'Throughput (median±std)':>26} {'Slowdown':>12}")
    print("-" * 110)
    none_med = next((r["throughput_median"] for r in results
                      if r["method"] == "none"), None)
    for r in results:
        if r["method"] == "none":
            slowdown_str = "(reference)"
        else:
            slow = 100 * (none_med - r["throughput_median"]) / none_med
            slowdown_str = f"{slow:+5.1f}%"
        tput_str = f"{r['throughput_median']:.1f} ± {r['throughput_std']:.1f}"
        print(f"{r['method']:<14} {r['total_params']:>14,} {r['pe_params']:>12,} "
              f"{r['flops_g']:>12.4f} {tput_str:>26} {slowdown_str:>12}")

    print("\nReading guide:")
    print("  Lower-is-better:  Total/PE Params, FLOPs, slowdown magnitude")
    print("  Higher-is-better: Throughput (median across passes)")
    print("  Std should be < ~5% of median for trustworthy numbers.")


if __name__ == "__main__":
    main()
