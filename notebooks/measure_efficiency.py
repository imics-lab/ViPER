"""
measure_efficiency.py — Measure params, FLOPs, throughput for all PE methods.

Outputs a table:
  Method | Total Params | PE Params | FLOPs (G) | Throughput (img/s)

Usage:
    python measure_efficiency.py --backbone scratch_vit_tiny --image_size 224 --batch_size 32
    python measure_efficiency.py --backbone pretrained_deit_tiny --image_size 224 --batch_size 32
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


def measure_throughput(model, x, n_warmup=10, n_iter=50):
    """Measure inference throughput (img/s) on a fixed batch."""
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    B = x.shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed iterations
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    throughput = (B * n_iter) / elapsed
    return throughput


def measure_flops(model, x):
    """Measure FLOPs (multiply-adds counted as 1) using fvcore."""
    model.eval()
    try:
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        return flops.total()
    except Exception as e:
        print(f"  FLOPs measurement failed: {e}")
        return -1


def count_params(model):
    """Return total trainable params."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_pe_params(model, backbone):
    """Best-effort PE-specific param count for each backbone variant."""
    pe_p = 0
    if hasattr(model, "viper_features"):       # scratch v2/v3/v4/v5 ViPER
        pe_p = sum(p.numel() for p in model.viper_features.parameters())
    elif hasattr(model, "pe_module") and model.pe_module is not None:
        pe_p = sum(p.numel() for p in model.pe_module.parameters() if p.requires_grad)
    elif hasattr(model, "per_block") and model.per_block is not None:
        pe_p = sum(p.numel() for p in model.per_block.parameters())
    elif hasattr(model, "wavelet_extractor"):  # v4
        pe_p = sum(p.numel() for p in model.wavelet_extractor.parameters())
        for blk in getattr(model, "blocks", []):
            if hasattr(blk, "wavelet_cpe"):
                pe_p += sum(p.numel() for p in blk.wavelet_cpe.parameters())
    return pe_p


def build_model_scratch_v5(pe_type, viper_cfg, num_classes, image_size, patch_size):
    """Build a v5-style model with pretrained DeiT-Tiny backbone."""
    from viper_v5 import DeiTWithCustomPE
    return DeiTWithCustomPE(
        num_classes=num_classes, image_size=image_size,
        patch_size=patch_size, pe_type=pe_type, viper_cfg=viper_cfg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Just needed for head sizing; doesn't affect measurements meaningfully")
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
    print(f"Batch size: {args.batch_size}")
    print()

    # Methods to measure
    from viper_v5 import ViPERConfig
    viper_cfg = ViPERConfig(n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32)

    methods = [
        ("none",      None),
        ("learned",   None),
        ("sincos2d",  None),
        ("relative2d", None),
        ("cpe",       None),
        ("multipeg",  None),
        ("viper",     viper_cfg),
    ]

    results = []
    for pe_type, cfg in methods:
        print(f"=== {pe_type} ===")
        try:
            model = build_model_scratch_v5(
                pe_type, cfg, args.num_classes, args.image_size, args.patch_size
            ).to(args.device)
            model.eval()

            # Dummy input
            x = torch.randn(args.batch_size, 3, args.image_size, args.image_size,
                            device=args.device)

            total_params = count_params(model)
            pe_params = count_pe_params(model, "pretrained_deit_tiny")

            # Single-sample FLOPs (more standard reporting)
            x_single = torch.randn(1, 3, args.image_size, args.image_size,
                                   device=args.device)
            flops = measure_flops(model, x_single)

            # Throughput at batch size
            throughput = measure_throughput(model, x)

            print(f"  total_params: {total_params:>12,}")
            print(f"  PE_params:    {pe_params:>12,}")
            print(f"  FLOPs:        {flops / 1e9:>10.3f} G")
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

    # Save
    out_file = out_dir / "efficiency_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_file}")

    # Print summary
    print()
    print("=" * 100)
    print(f"{'Method':<14} {'Total params':>14} {'PE params':>12} {'FLOPs (G)':>11} {'Throughput':>14}")
    print("-" * 100)
    for r in results:
        print(f"{r['method']:<14} {r['total_params']:>14,} {r['pe_params']:>12,} "
              f"{r['flops_g']:>11.3f} {r['throughput_img_per_sec']:>12.1f} img/s")


if __name__ == "__main__":
    main()
def measure_viper_flops_manual(model, x):
    """Manual FLOPs for ViPER:
       - Backbone (DeiT-Tiny): get FLOPs from a NoPE counterpart
       - DWT: estimate as 2 * H * W * J * filter_length (Daubechies db4 = 8 taps)
       - Channel proj + gating + patch agg + linear: small constants
    """
    H, W = x.shape[-2:]
    # DWT FLOPs estimate: 2*H*W*J*8 for db4
    # (J levels, each does a 1D conv twice for rows + cols)
    J = 3
    filter_len = 8  # db4
    dwt_flops = 2 * H * W * J * filter_len * 4  # mult+add, 2 channels per pass
    # IDWT (if used) is symmetric: same cost
    # Gating + projections: ~ d_pe^2 * n_subbands ≈ 32*32*10 = 10k FLOPs (tiny)
    # Patch aggregation conv: d_pe * d_model * patch_size^2 = 32*192*256 = 1.5M FLOPs
    extras_flops = 32 * 192 * 256 + 32 * 32 * 10
    return dwt_flops + extras_flops
