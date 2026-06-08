"""
measure_efficiency_v2.py — Efficiency measurement for ALL PE methods.

Improvements over v1:
  1. Manual DWT FLOPs estimate for ViPER (fvcore can't trace pytorch_wavelets)
  2. Support for RoPE-Mixed and ALiBi-2D (internal-PE methods that monkey-patch attention)
  3. Slowdown % column relative to 'none' baseline

Usage:
    python measure_efficiency_v2.py --image_size 224 --batch_size 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("Install fvcore: pip install fvcore")
    raise

from viper_v5 import DeiTWithCustomPE, ViPERConfig
from viper_v5_extra_pes import apply_internal_pe_to_deit


def measure_throughput(model, x, n_warmup=10, n_iter=50):
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    B = x.shape[0]

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (B * n_iter) / elapsed


def estimate_dwt_flops(H: int, W: int, n_levels: int = 3, filter_len: int = 8):
    """2D DWT FLOPs estimate.

    Per level: 2 passes (row, then col). Each pass is a 1D conv on each
    row/col. For db4, filter_len = 8.
    """
    total = 0
    h, w = H, W
    for j in range(n_levels):
        # Row decomposition: h rows of length w → low+high (length w/2 each)
        # FLOPs ≈ h * (w/2) * filter_len * 2 (mult+add) * 2 (low+high outputs)
        row = h * (w // 2) * filter_len * 2 * 2
        # Col decomposition on each of 4 row sub-bands
        col = 4 * (h // 2) * (w // 2) * filter_len * 2 * 2
        total += row + col
        h //= 2
        w //= 2
    return total


def measure_viper_extras_flops(image_size, n_levels=3, d_pe=32, d_model=192,
                                  patch_size=16):
    """ViPER's non-DWT FLOPs (channel proj, gating, pooling, projection)."""
    H_p = image_size // patch_size
    W_p = image_size // patch_size
    chproj = 3 * image_size * image_size * 2
    n_subbands = 1 + 3 * n_levels
    gate = n_subbands * (d_pe * d_pe * 2 + d_pe * 2)
    pool = n_subbands * H_p * W_p * d_pe
    proj = H_p * W_p * d_pe * d_model * 2
    return chproj + gate + pool + proj


def measure_flops_robust(model, x, pe_type, image_size, backbone_flops):
    """FLOPs with fallback for ops fvcore can't trace."""
    model.eval()
    try:
        flops_obj = FlopCountAnalysis(model, x)
        flops_obj.unsupported_ops_warnings(False)
        flops = flops_obj.total()
        if flops <= 0:
            raise RuntimeError("fvcore returned non-positive")
        return flops
    except Exception as e:
        print(f"  [fvcore fallback for {pe_type}: {e}]")
        if pe_type == "viper":
            dwt = estimate_dwt_flops(image_size, image_size, n_levels=3, filter_len=8)
            extras = measure_viper_extras_flops(image_size)
            return backbone_flops + dwt + extras
        else:
            # Internal PEs (RoPE-Mixed, ALiBi): PE compute is negligible
            return backbone_flops


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
    parser.add_argument("--out_dir", default="efficiency_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size}, Patch size: {args.patch_size}")
    print(f"Batch size: {args.batch_size}\n")

    methods = ["none", "learned", "sincos2d", "relative2d", "cpe", "multipeg",
               "rope_mixed", "alibi2d", "viper"]

    # First pass: measure the 'none' backbone FLOPs for fallback estimation
    backbone_flops = None

    results = []
    for pe_type in methods:
        print(f"=== {pe_type} ===")
        try:
            model = build_model(pe_type, args.image_size, args.patch_size,
                                  args.num_classes, args.device)
            model.eval()

            x_single = torch.randn(1, 3, args.image_size, args.image_size,
                                   device=args.device)
            # Use 'none' FLOPs as backbone reference for fallback cases
            if pe_type == "none":
                flops_obj = FlopCountAnalysis(model, x_single)
                flops_obj.unsupported_ops_warnings(False)
                backbone_flops = flops_obj.total()
                flops = backbone_flops
            else:
                flops = measure_flops_robust(model, x_single, pe_type,
                                                args.image_size, backbone_flops)

            x_batch = torch.randn(args.batch_size, 3, args.image_size,
                                  args.image_size, device=args.device)
            throughput = measure_throughput(model, x_batch)

            total_params = count_params(model)
            pe_params = count_pe_params(model, pe_type)

            print(f"  total_params: {total_params:>12,}")
            print(f"  PE_params:    {pe_params:>12,}")
            print(f"  FLOPs:        {flops / 1e9:>10.4f} G")
            print(f"  throughput:   {throughput:>10.1f} img/s @ bs={args.batch_size}")
            print()

            results.append({
                "method": pe_type,
                "total_params": total_params,
                "pe_params": pe_params,
                "flops_g": flops / 1e9,
                "throughput_img_per_sec": throughput,
                "batch_size": args.batch_size,
            })

            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    out_file = out_dir / "efficiency_v2_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}\n")

    print("=" * 108)
    print(f"{'Method':<14} {'Total Params':>14} {'PE Params':>12} "
          f"{'FLOPs (G)':>12} {'Throughput':>14} {'Slowdown vs none':>22}")
    print("-" * 108)
    none_throughput = next((r["throughput_img_per_sec"] for r in results
                             if r["method"] == "none"), None)
    for r in results:
        slowdown_str = "(reference)"
        if none_throughput is not None and r["method"] != "none":
            slowdown_pct = 100 * (none_throughput - r["throughput_img_per_sec"]) / none_throughput
            slowdown_str = f"{slowdown_pct:+6.1f}%"
        print(f"{r['method']:<14} {r['total_params']:>14,} {r['pe_params']:>12,} "
              f"{r['flops_g']:>12.4f} {r['throughput_img_per_sec']:>12.1f} {slowdown_str:>22}")
    print()
    print("Reading guide:")
    print("  Total Params: lower is better (deployment cost)")
    print("  PE Params:    lower is better (PE overhead)")
    print("  FLOPs:        lower is better (compute cost)")
    print("  Throughput:   higher is better (inference speed)")
    print("  Slowdown:     lower magnitude is better")


if __name__ == "__main__":
    main()
