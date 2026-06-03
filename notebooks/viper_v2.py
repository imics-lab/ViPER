"""
ViPER v2 — Vision Positional Encoding with Hierarchical Dynamic Wavelet Encoding.

This is a clean re-implementation grounded in the published DyWPE method
(Habib & Metsis, ICASSP 2026), extended to 2D images with vision-specific
components:

  1. 2D-DWT instead of 1D-DWT (mandatory for images)
  2. Region pooling instead of IDWT (correct 2D-equivalent operation)
  3. Direction-aware sub-band fusion (LH/HL/HH treated separately)
  4. Spatial Context Aggregation (depthwise conv before/after wavelet branch)
  5. Adaptive Level Weighting (ALW: gate over levels conditioned on LL)
  6. Layer-wise re-injection (PE re-added at every transformer layer)

Run examples
------------

# Quick smoke test (3 epochs, 1 config):
python viper_v2.py --dataset eurosat --suite smoke

# Main result: baselines + ViPER variants on EuroSAT, single seed:
python viper_v2.py --dataset eurosat --suite main --epochs 50

# Injection mode ablation on BloodMNIST:
python viper_v2.py --dataset bloodmnist --suite injection --epochs 20

# Multi-seed final run for confidence intervals:
python viper_v2.py --dataset eurosat --suite final --seeds 42 123 7 2024 99

# Available suites:
#   smoke       — 1 config, 3 epochs (sanity check)
#   baselines   — 7 baseline PEs
#   main        — best ViPER variant + 7 baselines
#   injection   — layer_reinject vs input_only vs concat vs rotary
#   components  — ablate channel_proj, gating, ALW, SCA
#   wavelets    — db1, db2, db4, sym4, coif1
#   levels      — n_levels = 1..max
#   d_pe        — internal PE dimension sweep
#   final       — winner config, multi-seed
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
from pytorch_wavelets import DWTForward


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ViPERConfig:
    """All ablation axes for the v2 ViPER module."""
    # Wavelet decomposition
    n_levels:        int = 3
    wavelet:         str = "db4"           # db1, db2, db4, sym4, coif1, haar
    channel_mode:    str = "gray"          # gray | learnable_proj | per_channel

    # PE pathway internal dimension
    d_pe:            int = 32              # smaller than d_model; projected up at end

    # Sub-band fusion (within a level)
    subband_fusion:  str = "gated"         # gated | concat | weighted_sum | sum

    # Level combination
    level_combine:   str = "alw"           # alw | sum | weighted | concat

    # Re-injection mode (where in the transformer block PE is added)
    injection_mode:  str = "layer_reinject"
    #   input_only      — add to embeddings before layer 0 (DyWPE-style)
    #   layer_reinject  — add at every layer entry (CPE-style)
    #   pre_attention   — add only before attention sub-layer
    #   pre_ffn         — add only before FFN sub-layer
    #   concat_input    — concat to input embeddings, project back to d_model
    #   rotary          — rotary modulation of Q,K inside attention

    # Rotary-specific (only used when injection_mode = "rotary")
    rotary_theta_base:  float = 10000.0    # coord rotary base frequency
    rotary_gamma_init:  float = 0.0        # γ_k init; 0 = pure RoPE, learns content
    rotary_gamma_shared: bool = False      # share γ across heads

    # Spatial Context Aggregation
    sca: bool = True                       # depthwise conv on patch embeddings

    # Component flags for ablation
    use_channel_proj: bool = True          # if False, skip channel projection
    use_gating:       bool = True          # if False, just sum sub-bands (no gates)


# =============================================================================
# Wavelet feature extraction (Stage 1)
# =============================================================================

class ChannelProjection(nn.Module):
    """RGB → 1 channel for DWT. Initialized to luminance weights."""
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
        elif mode == "per_channel":
            # No projection: keep all channels, will be averaged later
            pass
        else:
            raise ValueError(f"Unknown channel_mode: {mode}")

    def forward(self, x):
        if self.mode == "gray":
            return (x * self.rgb2gray).sum(dim=1, keepdim=True)
        elif self.mode == "learnable_proj":
            return self.proj(x)
        elif self.mode == "per_channel":
            return x  # downstream handles multi-channel


class HierarchicalGating(nn.Module):
    """
    DyWPE-style dynamic gating, extended to 2D direction-aware form.

    For each sub-band at level j with direction d (LL/LH/HL/HH), we have a
    learnable scale embedding e_{j,d} ∈ R^{d_pe}. The gate is:

        gate(e, c) = (σ(W_g e) ⊙ tanh(W_v e)) ⊗ c'

    matching DyWPE's Step 4 exactly. Sub-bands within a level are fused (gated,
    weighted_sum, or concat), then levels are combined (sum, ALW, etc.).
    """
    def __init__(self, n_levels: int, d_pe: int, subband_fusion: str,
                 level_combine: str, use_gating: bool = True):
        super().__init__()
        self.n_levels = n_levels
        self.d_pe = d_pe
        self.subband_fusion = subband_fusion
        self.level_combine = level_combine
        self.use_gating = use_gating

        # Number of sub-bands: 1 LL (at deepest level) + 3 details per level
        self.n_subbands = 1 + 3 * n_levels

        if use_gating:
            # Learnable scale embeddings — one per sub-band
            self.scale_embeddings = nn.Parameter(
                torch.randn(self.n_subbands, d_pe) * 0.02
            )
            self.W_g = nn.Linear(d_pe, d_pe, bias=False)
            self.W_v = nn.Linear(d_pe, d_pe, bias=False)

        # Sub-band fusion within a level (combines 3 detail sub-bands)
        if subband_fusion == "concat":
            self.subband_proj = nn.Conv2d(3 * d_pe, d_pe, kernel_size=1)
        elif subband_fusion == "weighted_sum":
            self.subband_weights = nn.Parameter(
                torch.ones(n_levels, 3) / 3.0
            )

        # Level combination
        if level_combine == "concat":
            self.level_proj = nn.Conv2d((n_levels + 1) * d_pe, d_pe, kernel_size=1)
        elif level_combine == "weighted":
            self.level_weights = nn.Parameter(
                torch.ones(n_levels + 1) / (n_levels + 1)
            )
        elif level_combine == "alw":
            # Adaptive Level Weighting: gate conditioned on LL features
            self.alw_mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d_pe, n_levels + 1),
            )

    def _gate(self, coeffs: torch.Tensor, sb_idx: int) -> torch.Tensor:
        """Apply DyWPE-style gating to one sub-band's coefficients.

        coeffs: (B, d_pe, H, W)  — sub-band map projected to d_pe channels
        sb_idx: index into self.scale_embeddings
        """
        if not self.use_gating:
            return coeffs
        e = self.scale_embeddings[sb_idx]                       # (d_pe,)
        g = torch.sigmoid(self.W_g(e))                          # (d_pe,)
        v = torch.tanh(self.W_v(e))                             # (d_pe,)
        modulation = (g * v).view(1, self.d_pe, 1, 1)           # (1, d_pe, 1, 1)
        return coeffs * modulation

    def fuse_subbands(self, ll_or_details: List[torch.Tensor],
                       level_idx: int, is_approx: bool = False) -> torch.Tensor:
        """Combine sub-bands within a single level.

        For LL (approx), the list has 1 tensor.
        For detail levels, the list has 3 tensors (LH, HL, HH).
        Each tensor is shape (B, d_pe, H_j, W_j).
        Returns: (B, d_pe, H_j, W_j).
        """
        if is_approx:
            # Single LL: gate (if enabled) and return
            return self._gate(ll_or_details[0], 0)

        # 3 detail sub-bands at this level
        sb_indices = [1 + level_idx * 3 + d for d in range(3)]

        if self.subband_fusion == "sum":
            return sum(self._gate(c, idx) for c, idx in zip(ll_or_details, sb_indices))

        elif self.subband_fusion == "gated":
            # DyWPE-style: gate each, then sum (gates carry per-band weighting)
            return sum(self._gate(c, idx) for c, idx in zip(ll_or_details, sb_indices))

        elif self.subband_fusion == "weighted_sum":
            w = torch.softmax(self.subband_weights[level_idx], dim=0)
            return sum(w[d] * self._gate(c, idx)
                       for d, (c, idx) in enumerate(zip(ll_or_details, sb_indices)))

        elif self.subband_fusion == "concat":
            gated = [self._gate(c, idx) for c, idx in zip(ll_or_details, sb_indices)]
            return self.subband_proj(torch.cat(gated, dim=1))

        else:
            raise ValueError(f"Unknown subband_fusion: {self.subband_fusion}")

    def combine_levels(self, level_pes: List[torch.Tensor]) -> torch.Tensor:
        """Combine per-level PE maps into a single PE map.

        level_pes: list of (B, d_pe, H_p, W_p) — already at patch-grid resolution
        Returns:   (B, d_pe, H_p, W_p)
        """
        if self.level_combine == "sum":
            return sum(level_pes)

        elif self.level_combine == "weighted":
            w = torch.softmax(self.level_weights, dim=0)
            return sum(wi * pi for wi, pi in zip(w, level_pes))

        elif self.level_combine == "concat":
            return self.level_proj(torch.cat(level_pes, dim=1))

        elif self.level_combine == "alw":
            # Gate conditioned on LL (level_pes[0])
            gate_input = level_pes[0]
            w = torch.softmax(self.alw_mlp(gate_input), dim=-1)     # (B, n_levels+1)
            B = w.shape[0]
            out = sum(
                w[:, i].view(B, 1, 1, 1) * pi
                for i, pi in enumerate(level_pes)
            )
            return out

        else:
            raise ValueError(f"Unknown level_combine: {self.level_combine}")


class ViPERFeatureExtractor(nn.Module):
    """Stage 1: Image → per-patch PE map.

    Output shape: (B, d_pe, H_p, W_p)
    """
    def __init__(self, cfg: ViPERConfig, in_channels: int, image_h: int,
                 image_w: int, patch_size: int):
        super().__init__()
        self.cfg = cfg
        self.image_h, self.image_w = image_h, image_w
        self.patch_size = patch_size
        self.H_p = image_h // patch_size
        self.W_p = image_w // patch_size

        # Sanity check on n_levels
        max_levels = int(math.log2(min(image_h, image_w) // patch_size))
        if cfg.n_levels > max_levels + 1:
            warnings.warn(
                f"n_levels={cfg.n_levels} exceeds meaningful max "
                f"{max_levels + 1} for image_size={image_h} and "
                f"patch_size={patch_size}. Sub-bands smaller than patch grid "
                f"will be upsampled, which is degenerate."
            )

        # Channel projection
        if cfg.use_channel_proj:
            self.channel_proj = ChannelProjection(in_channels, cfg.channel_mode)
            dwt_in_channels = in_channels if cfg.channel_mode == "per_channel" else 1
        else:
            self.channel_proj = nn.Identity()
            dwt_in_channels = in_channels

        # 2D DWT
        self.dwt = DWTForward(J=cfg.n_levels, wave=cfg.wavelet, mode="symmetric")

        # Sub-band → d_pe projection
        self.subband_proj = nn.Conv2d(dwt_in_channels, cfg.d_pe, kernel_size=1)

        # Gating + fusion
        self.gating = HierarchicalGating(
            cfg.n_levels, cfg.d_pe,
            cfg.subband_fusion, cfg.level_combine,
            use_gating=cfg.use_gating,
        )

        # Spatial Context Aggregation (depthwise conv refinement)
        if cfg.sca:
            self.sca = nn.Sequential(
                nn.Conv2d(cfg.d_pe, cfg.d_pe, kernel_size=3, padding=1,
                          groups=cfg.d_pe, bias=False),
                nn.GroupNorm(1, cfg.d_pe),
            )

    def _pool_to_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Region pooling: reduce sub-band spatial dims to (H_p, W_p).

        If sub-band is larger than patch grid → adaptive_avg_pool (region pool).
        If sub-band is exactly patch grid → identity.
        If sub-band is smaller (degenerate) → bilinear upsample with warning.
        """
        _, _, h, w = x.shape
        if h == self.H_p and w == self.W_p:
            return x
        if h >= self.H_p and w >= self.W_p:
            return F.adaptive_avg_pool2d(x, (self.H_p, self.W_p))
        return F.interpolate(x, size=(self.H_p, self.W_p),
                             mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, C, H, W) → P_viper: (B, d_pe, H_p, W_p)"""
        # 1. Channel projection
        x = self.channel_proj(image)

        # 2. DWT decomposition
        Yl, Yh = self.dwt(x)
        # Yl: (B, C_in, H/2^J, W/2^J)
        # Yh: list of length J, each (B, C_in, 3, H/2^j, W/2^j)

        # 3. Project each sub-band to d_pe channels, then pool to patch grid
        # Process LL (approximation at deepest level)
        ll_pe = self.subband_proj(Yl)                              # (B, d_pe, ...)
        ll_pe = self._pool_to_patch_grid(ll_pe)

        # Process detail sub-bands at each level
        per_level_pe = []
        # First entry: LL gated alone (treat as "level 0")
        ll_fused = self.gating.fuse_subbands([ll_pe], level_idx=-1, is_approx=True)
        per_level_pe.append(ll_fused)

        for j in range(self.cfg.n_levels):
            details = Yh[j]                                         # (B, C, 3, h, w)
            level_subbands = []
            for d in range(3):
                sb = details[:, :, d]                               # (B, C, h, w)
                sb = self.subband_proj(sb)                          # (B, d_pe, h, w)
                sb = self._pool_to_patch_grid(sb)
                level_subbands.append(sb)

            fused = self.gating.fuse_subbands(level_subbands,
                                              level_idx=j, is_approx=False)
            per_level_pe.append(fused)

        # 4. Combine levels
        pe = self.gating.combine_levels(per_level_pe)               # (B, d_pe, H_p, W_p)

        # 5. SCA refinement
        if self.cfg.sca:
            pe = pe + self.sca(pe)                                  # residual

        return pe


# =============================================================================
# Stage 2 — Injection into the transformer
# =============================================================================

def _build_2d_rope_freqs(H: int, W: int, head_dim: int, base: float = 10000.0):
    """Standard axial 2D RoPE: half head_dim encodes row, half encodes col."""
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    half = head_dim // 2
    quarter = head_dim // 4
    inv_freq = 1.0 / (base ** (torch.arange(0, quarter).float() / quarter))

    rows = torch.arange(H).float()
    cols = torch.arange(W).float()
    row_freqs = torch.outer(rows, inv_freq)                         # (H, quarter)
    col_freqs = torch.outer(cols, inv_freq)                         # (W, quarter)

    # Tile to (H*W, half)
    row_freqs_grid = row_freqs.unsqueeze(1).expand(H, W, quarter).reshape(H*W, quarter)
    col_freqs_grid = col_freqs.unsqueeze(0).expand(H, W, quarter).reshape(H*W, quarter)

    # Cat to (H*W, half) for row and (H*W, half) for col
    row_angles = torch.cat([row_freqs_grid, row_freqs_grid], dim=-1)  # duplicate for pairs
    col_angles = torch.cat([col_freqs_grid, col_freqs_grid], dim=-1)

    # Full angle tensor shape (H*W, head_dim): first half = row angles, second = col
    full_angles = torch.cat([row_angles, col_angles], dim=-1)
    return full_angles                                                # (N, head_dim)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding. x: (..., head_dim), cos/sin: (..., head_dim).

    Pairs (x_0, x_d/2), (x_1, x_d/2+1), ... are rotated.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos1, cos2 = cos[..., :half], cos[..., half:]
    sin1, sin2 = sin[..., :half], sin[..., half:]
    return torch.cat([x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2], dim=-1)


class ViPERAttention(nn.Module):
    """Multi-head self-attention with optional ViPER rotary modulation.

    When injection_mode = "rotary", Q and K are rotated by:
        θ_ij^(k) = θ_coord^(k)(i,j) + γ_k · g_ij
    where g_ij is the scalar wavelet feature averaged across the d_pe channels
    (or a projection of the d_pe vector to a scalar).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float,
                 cfg: ViPERConfig, H_p: int, W_p: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.cfg = cfg
        self.H_p, self.W_p = H_p, W_p

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

        if cfg.injection_mode == "rotary":
            # Precompute coord-only rotary angles
            full_angles = _build_2d_rope_freqs(H_p, W_p, self.head_dim,
                                                base=cfg.rotary_theta_base)
            self.register_buffer("coord_angles", full_angles)       # (N, head_dim)

            # Per-head γ scalars
            n_gamma = 1 if cfg.rotary_gamma_shared else num_heads
            self.gamma = nn.Parameter(
                torch.full((n_gamma,), cfg.rotary_gamma_init)
            )

            # Scalar projection from d_pe → 1 (to get g_ij)
            self.g_proj = nn.Linear(cfg.d_pe, 1)

    def forward(self, x: torch.Tensor, viper_pe_2d: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        x:           (B, N+1, d_model) — tokens (CLS + patches)
        viper_pe_2d: (B, d_pe, H_p, W_p) — ViPER spatial PE (only used for rotary)
        """
        B, N1, _ = x.shape
        qkv = self.qkv(x).reshape(B, N1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # (B, H, N1, hd)

        if self.cfg.injection_mode == "rotary" and viper_pe_2d is not None:
            # Extract scalar g_ij per patch
            pe_flat = viper_pe_2d.permute(0, 2, 3, 1).reshape(B, -1, self.cfg.d_pe)
            g = self.g_proj(pe_flat).squeeze(-1)                    # (B, N)

            # Build θ_ij^(k) = θ_coord^(k) + γ_k · g_ij
            coord = self.coord_angles                                # (N, head_dim)
            # Per-head γ: (n_gamma,) → broadcast
            n_gamma = self.gamma.shape[0]
            if n_gamma == 1:
                gamma_per_head = self.gamma.expand(self.num_heads)
            else:
                gamma_per_head = self.gamma
            # Modulation: (B, H, N, head_dim)
            # gamma_per_head: (H,) → (1, H, 1, 1)
            # g: (B, N) → (B, 1, N, 1)
            modulation = gamma_per_head.view(1, self.num_heads, 1, 1) \
                          * g.view(B, 1, -1, 1)
            angles = coord.view(1, 1, -1, self.head_dim) + modulation
            cos = torch.cos(angles)
            sin = torch.sin(angles)

            # Apply rotary to patch tokens only (skip CLS at index 0)
            q_cls, q_pts = q[:, :, :1], q[:, :, 1:]
            k_cls, k_pts = k[:, :, :1], k[:, :, 1:]
            q_pts = _apply_rotary(q_pts, cos, sin)
            k_pts = _apply_rotary(k_pts, cos, sin)
            q = torch.cat([q_cls, q_pts], dim=2)
            k = torch.cat([k_cls, k_pts], dim=2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = (attn @ v).transpose(1, 2).reshape(B, N1, self.d_model)
        return self.out(out)


class ViPERBlock(nn.Module):
    """Pre-norm transformer block with optional ViPER PE re-injection.

    The re-injection point depends on cfg.injection_mode:
      - input_only:     no per-block injection (PE added before block 0 only)
      - layer_reinject: add λ · P_viper before attention AND before FFN
      - pre_attention:  add λ · P_viper before attention only
      - pre_ffn:        add λ · P_viper before FFN only
      - rotary:         rotary inside attention (no additive injection)
      - concat_input:   concat to input, project; no per-block injection
    """
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int,
                 dropout: float, cfg: ViPERConfig, H_p: int, W_p: int):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ViPERAttention(d_model, num_heads, dropout, cfg, H_p, W_p)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

        # Learnable per-layer scaling for re-injection
        if cfg.injection_mode in ("layer_reinject", "pre_attention", "pre_ffn"):
            self.lambda_attn = nn.Parameter(torch.tensor(0.1))
            self.lambda_ffn = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor,
                pe_token_seq: Optional[torch.Tensor] = None,
                pe_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:            (B, N+1, d_model)
        pe_token_seq: (B, N+1, d_model) — ViPER PE projected to d_model, with CLS=0 prepended
        pe_2d:        (B, d_pe, H_p, W_p) — raw 2D PE for rotary modulation
        """
        mode = self.cfg.injection_mode

        # Pre-attention injection
        x_in = x
        if mode in ("layer_reinject", "pre_attention") and pe_token_seq is not None:
            x_in = x + self.lambda_attn * pe_token_seq

        x = x + self.attn(self.norm1(x_in), viper_pe_2d=pe_2d)

        # Pre-FFN injection
        x_in = x
        if mode in ("layer_reinject", "pre_ffn") and pe_token_seq is not None:
            x_in = x + self.lambda_ffn * pe_token_seq

        x = x + self.mlp(self.norm2(x_in))
        return x


# =============================================================================
# Baseline PEs (copied from v1 for fair head-to-head)
# =============================================================================

class NoPE(nn.Module):
    def forward(self, x, image=None): return x


class LearnedPE(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
    def forward(self, x, image=None):
        return x + self.pe[:, : x.shape[1]]


class SinCos2DPE(nn.Module):
    def __init__(self, H, W, d_model):
        super().__init__()
        assert d_model % 4 == 0
        d_q = d_model // 4
        y = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(W, dtype=torch.float32).unsqueeze(1)
        w = 1.0 / (10000 ** (torch.arange(d_q, dtype=torch.float32) / d_q))
        enc_h = torch.cat([torch.sin(y*w), torch.cos(y*w)], dim=-1)
        enc_w = torch.cat([torch.sin(x*w), torch.cos(x*w)], dim=-1)
        enc_h = enc_h.unsqueeze(1).expand(H, W, -1)
        enc_w = enc_w.unsqueeze(0).expand(H, W, -1)
        pe = torch.cat([enc_h, enc_w], dim=-1).reshape(1, H*W, d_model)
        cls_pe = torch.zeros(1, 1, d_model)
        self.register_buffer("pe", torch.cat([cls_pe, pe], dim=1))
    def forward(self, x, image=None):
        return x + self.pe[:, : x.shape[1]]


class CPE(nn.Module):
    """Conditional PE (Chu et al. 2021): depthwise conv injects position."""
    def __init__(self, H, W, d_model, k=3):
        super().__init__()
        self.H, self.W = H, W
        self.proj = nn.Conv2d(d_model, d_model, k, padding=k//2, groups=d_model)
        self.cls_pe = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    def forward(self, x, image=None):
        cls, pts = x[:, :1], x[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)
        sp = self.proj(sp) + sp
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls + self.cls_pe.expand(B, -1, -1), pts], dim=1)


class Relative2DPE(nn.Module):
    def __init__(self, H, W, d_model):
        super().__init__()
        self.H, self.W = H, W
        self.rel_h = nn.Embedding(2*H - 1, d_model // 2)
        self.rel_w = nn.Embedding(2*W - 1, d_model // 2)
        self.cls_pe = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    def forward(self, x, image=None):
        B, _, _ = x.shape
        device = x.device
        rows = torch.arange(self.H, device=device).repeat_interleave(self.W)
        cols = torch.arange(self.W, device=device).repeat(self.H)
        rel_r = (rows.unsqueeze(1) - rows.unsqueeze(0)) + (self.H - 1)
        rel_c = (cols.unsqueeze(1) - cols.unsqueeze(0)) + (self.W - 1)
        emb_r = self.rel_h(rel_r).mean(dim=1)
        emb_c = self.rel_w(rel_c).mean(dim=1)
        pe = torch.cat([emb_r, emb_c], dim=-1).unsqueeze(0).expand(B, -1, -1)
        cls = self.cls_pe.expand(B, -1, -1)
        return x + torch.cat([cls, pe], dim=1)


class RoPE2D(nn.Module):
    """Standard 2D RoPE — applied at input embedding for baseline comparison."""
    def __init__(self, H, W, d_model):
        super().__init__()
        assert d_model % 4 == 0
        self.H, self.W = H, W
        self.half = d_model // 2
        self.quarter = d_model // 4
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.quarter, dtype=torch.float32) / self.quarter))
        self.register_buffer("inv_freq", inv_freq)
    def _axis(self, pos):
        f = torch.outer(pos.float(), self.inv_freq)
        return f.cos(), f.sin()
    @staticmethod
    def _rotate(x, c, s):
        x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1)
    def forward(self, x, image=None):
        cls, pts = x[:, :1], x[:, 1:]
        device = x.device
        rows = torch.arange(self.H, device=device).repeat_interleave(self.W)
        cols = torch.arange(self.W, device=device).repeat(self.H)
        ch, sh = self._axis(rows); cw, sw = self._axis(cols)
        ph = self._rotate(pts[..., :self.half], ch, sh)
        pw = self._rotate(pts[..., self.half:], cw, sw)
        return torch.cat([cls, torch.cat([ph, pw], dim=-1)], dim=1)


class iRPE(nn.Module):
    """Bucketed 2D relative bias (simplified)."""
    def __init__(self, H, W, d_model, n_buckets=16):
        super().__init__()
        self.H, self.W = H, W
        self.n_buckets = n_buckets
        self.embed = nn.Embedding(n_buckets * n_buckets, d_model)
        self.cls_pe = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    def _bucket(self, x, n):
        sign = torch.sign(x); x = torch.abs(x).float()
        log_max = math.log(n/2 + 1)
        idx = (torch.log(x + 1) / log_max * (n/2)).long().clamp(max=n//2 - 1)
        return ((sign.long() + 1) * (n//2) + idx).clamp(0, n - 1)
    def forward(self, x, image=None):
        B = x.shape[0]; device = x.device
        rows = torch.arange(self.H, device=device).repeat_interleave(self.W)
        cols = torch.arange(self.W, device=device).repeat(self.H)
        br = self._bucket(rows - rows.float().mean(), self.n_buckets)
        bc = self._bucket(cols - cols.float().mean(), self.n_buckets)
        idx = br * self.n_buckets + bc
        pe = self.embed(idx).unsqueeze(0).expand(B, -1, -1)
        cls = self.cls_pe.expand(B, -1, -1)
        return x + torch.cat([cls, pe], dim=1)


# =============================================================================
# Patch embedding + full model
# =============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, P, d, H, W):
        super().__init__()
        self.P = P
        self.pad_h = (P - H % P) % P
        self.pad_w = (P - W % P) % P
        H_e, W_e = H + self.pad_h, W + self.pad_w
        self.H_p, self.W_p = H_e // P, W_e // P
        self.num_patches = self.H_p * self.W_p
        self.proj = nn.Conv2d(in_ch, d, P, P)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
    def forward(self, x):
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)


class ViPERViT(nn.Module):
    """ViT with ViPER positional encoding (or a baseline PE for comparison)."""
    def __init__(self, image_h, image_w, in_channels, patch_size, d_model,
                 num_layers, num_heads, mlp_dim, num_classes, dropout,
                 pe_type: str, viper_cfg: Optional[ViPERConfig] = None):
        super().__init__()
        self.pe_type = pe_type
        self.viper_cfg = viper_cfg
        self.patch_embed = PatchEmbed(in_channels, patch_size, d_model, image_h, image_w)
        H_p, W_p = self.patch_embed.H_p, self.patch_embed.W_p
        N = self.patch_embed.num_patches
        self.H_p, self.W_p = H_p, W_p
        self.d_model = d_model

        # Build PE module
        if pe_type == "viper":
            assert viper_cfg is not None
            self.viper_features = ViPERFeatureExtractor(
                viper_cfg, in_channels, image_h, image_w, patch_size
            )
            # Project d_pe → d_model for additive injection (not used for rotary)
            self.pe_to_dmodel = nn.Linear(viper_cfg.d_pe, d_model)
            # Concat mode needs a back-projection
            if viper_cfg.injection_mode == "concat_input":
                self.concat_proj = nn.Linear(2 * d_model, d_model)
            # Encoder blocks are ViPERBlock-aware
            self.blocks = nn.ModuleList([
                ViPERBlock(d_model, num_heads, mlp_dim, dropout, viper_cfg, H_p, W_p)
                for _ in range(num_layers)
            ])
        else:
            # Baseline path: use a standard transformer + classic PE module
            self.viper_features = None
            self.pe_module = self._build_baseline_pe(pe_type, d_model, H_p, W_p, N)
            layer = nn.TransformerEncoderLayer(
                d_model, num_heads, mlp_dim, dropout,
                batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _build_baseline_pe(self, t, d, H, W, N):
        if t == "none":       return NoPE()
        if t == "learned":    return LearnedPE(N + 1, d)
        if t == "sincos2d":   return SinCos2DPE(H, W, d)
        if t == "rope2d":     return RoPE2D(H, W, d)
        if t == "relative2d": return Relative2DPE(H, W, d)
        if t == "cpe":        return CPE(H, W, d)
        if t == "irpe":       return iRPE(H, W, d)
        raise ValueError(f"Unknown PE: {t}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, image):
        x = self.patch_embed(image)

        if self.pe_type == "viper":
            # 1. Extract ViPER 2D PE map (B, d_pe, H_p, W_p)
            pe_2d = self.viper_features(image)

            # 2. Project to d_model token sequence (with CLS=0)
            B = x.shape[0]
            pe_seq = pe_2d.permute(0, 2, 3, 1).reshape(B, -1, self.viper_cfg.d_pe)
            pe_seq_dmodel = self.pe_to_dmodel(pe_seq)                # (B, N, d_model)
            # Prepend zero for CLS
            zero_cls = torch.zeros(B, 1, self.d_model, device=x.device, dtype=x.dtype)
            pe_token_seq = torch.cat([zero_cls, pe_seq_dmodel], dim=1)

            # 3. Apply injection per mode
            mode = self.viper_cfg.injection_mode
            if mode == "input_only":
                x = x + pe_token_seq
                pe_for_blocks = None
                pe_2d_for_blocks = None
            elif mode == "concat_input":
                x = self.concat_proj(torch.cat([x, pe_token_seq], dim=-1))
                pe_for_blocks = None
                pe_2d_for_blocks = None
            elif mode in ("layer_reinject", "pre_attention", "pre_ffn"):
                pe_for_blocks = pe_token_seq
                pe_2d_for_blocks = None
            elif mode == "rotary":
                pe_for_blocks = None
                pe_2d_for_blocks = pe_2d
            else:
                raise ValueError(f"Unknown injection_mode: {mode}")

            for blk in self.blocks:
                x = blk(x, pe_token_seq=pe_for_blocks, pe_2d=pe_2d_for_blocks)
        else:
            # Baseline path
            x = self.pe_module(x)
            x = self.encoder(x)

        return self.head(self.norm(x[:, 0]))


# =============================================================================
# Training / evaluation utilities
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


def train_one_v2(pe_type, model_params, viper_cfg, train_loader, val_loader,
                 test_loader, n_epochs=50, lr=3e-4, weight_decay=0.05,
                 device=DEVICE, num_classes=10, seed=42, verbose=True,
                 name=None):
    set_seed(seed)
    model = ViPERViT(pe_type=pe_type, viper_cfg=viper_cfg, **model_params).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Count ViPER-specific params
    if pe_type == "viper":
        pe_params = sum(p.numel() for p in model.viper_features.parameters()) \
                    + sum(p.numel() for p in model.pe_to_dmodel.parameters())
        if hasattr(model, "concat_proj"):
            pe_params += sum(p.numel() for p in model.concat_proj.parameters())
        # Plus λ scalars from blocks
        for blk in model.blocks:
            if hasattr(blk, "lambda_attn"):
                pe_params += blk.lambda_attn.numel() + blk.lambda_ffn.numel()
            if hasattr(blk.attn, "gamma"):
                pe_params += blk.attn.gamma.numel()
                pe_params += sum(p.numel() for p in blk.attn.g_proj.parameters())
    else:
        pe_params = sum(p.numel() for p in model.pe_module.parameters()) \
                    if hasattr(model, "pe_module") else 0

    if verbose:
        tag = name or pe_type
        print(f"[{tag}]  total={n_params:,}  PE={pe_params:,}")

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
# Ablation suite definitions
# =============================================================================

def get_baseline_configs():
    return [(pe, None) for pe in
            ["none", "learned", "sincos2d", "rope2d", "relative2d", "cpe", "irpe"]]


def get_default_viper_cfg():
    """Best-guess winner config for ViPER (paper-spec)."""
    return ViPERConfig(
        n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32,
        subband_fusion="gated", level_combine="alw",
        injection_mode="layer_reinject", sca=True,
    )


def get_suite(suite_name: str) -> List[Tuple[str, Optional[ViPERConfig], str]]:
    """Return list of (pe_type, viper_cfg, name) tuples for a named suite."""
    runs = []

    if suite_name == "smoke":
        cfg = get_default_viper_cfg()
        runs = [("viper", cfg, "viper_smoke")]

    elif suite_name == "baselines":
        runs = [(pe, None, pe) for pe, _ in get_baseline_configs()]

    elif suite_name == "main":
        runs = [(pe, None, pe) for pe, _ in get_baseline_configs()]
        cfg = get_default_viper_cfg()
        runs.append(("viper", cfg, "viper_main"))

    elif suite_name == "injection":
        for mode in ["input_only", "layer_reinject", "pre_attention",
                     "pre_ffn", "concat_input", "rotary"]:
            cfg = get_default_viper_cfg()
            cfg.injection_mode = mode
            runs.append(("viper", cfg, f"inj_{mode}"))

    elif suite_name == "components":
        # Ablate each component independently
        base = get_default_viper_cfg()
        runs.append(("viper", base, "comp_full"))

        cfg = get_default_viper_cfg(); cfg.use_channel_proj = False
        runs.append(("viper", cfg, "comp_no_chproj"))

        cfg = get_default_viper_cfg(); cfg.use_gating = False
        runs.append(("viper", cfg, "comp_no_gating"))

        cfg = get_default_viper_cfg(); cfg.level_combine = "sum"
        runs.append(("viper", cfg, "comp_no_alw"))

        cfg = get_default_viper_cfg(); cfg.sca = False
        runs.append(("viper", cfg, "comp_no_sca"))

    elif suite_name == "wavelets":
        for wav in ["haar", "db1", "db2", "db4", "sym4", "coif1"]:
            cfg = get_default_viper_cfg()
            cfg.wavelet = wav
            runs.append(("viper", cfg, f"wav_{wav}"))

    elif suite_name == "levels":
        for j in [1, 2, 3, 4]:
            cfg = get_default_viper_cfg()
            cfg.n_levels = j
            runs.append(("viper", cfg, f"levels_{j}"))

    elif suite_name == "d_pe":
        for d in [16, 32, 64, 128]:
            cfg = get_default_viper_cfg()
            cfg.d_pe = d
            runs.append(("viper", cfg, f"dpe_{d}"))

    elif suite_name == "final":
        # Winner config — caller adds seeds outside
        cfg = get_default_viper_cfg()
        runs = [("viper", cfg, "viper_final")]
        for pe in ["cpe", "relative2d", "none"]:   # strongest 3 baselines
            runs.append((pe, None, pe))
    elif suite_name == "contenders":
        # Top 3 baselines (from single-seed results) + ViPER for multi-seed comparison
        runs.append(("cpe", None, "cpe"))
        runs.append(("sincos2d", None, "sincos2d"))
        runs.append(("none", None, "none"))
        cfg = get_default_viper_cfg()
        runs.append(("viper", cfg, "viper_pre_ffn"))

    else:
        raise ValueError(f"Unknown suite: {suite_name}")

    return runs


# =============================================================================
# Dataset loading (uses data-loader.py from the repo)
# =============================================================================

def load_data_loader_module(data_loader_path: str):
    spec = importlib.util.spec_from_file_location("dl", data_loader_path)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl


# =============================================================================
# Main runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eurosat",
                        choices=["eurosat", "bloodmnist", "dermamnist",
                                 "pathmnist", "resisc45"])
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override default image size for the dataset.")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Override patch size. Defaults: 8 for eurosat, 16 for 224x224.")
    parser.add_argument("--suite", default="main",
                        choices=["smoke", "baselines", "main", "injection",
                                 "components", "wavelets", "levels", "d_pe", "final"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data_loader_path", default="../data/data-loader.py")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # Per-dataset defaults
    DATASET_DEFAULTS = {
        "eurosat":     dict(image_size=64,  patch_size=8,  batch_size=64),
        "bloodmnist":  dict(image_size=224, patch_size=16, batch_size=32),
        "dermamnist":  dict(image_size=224, patch_size=16, batch_size=32),
        "pathmnist":   dict(image_size=224, patch_size=16, batch_size=32),
        "resisc45":    dict(image_size=224, patch_size=16, batch_size=32),
    }
    d = DATASET_DEFAULTS[args.dataset]
    image_size = args.image_size or d["image_size"]
    patch_size = args.patch_size or d["patch_size"]
    batch_size = args.batch_size or d["batch_size"]

    out_dir = Path(args.out_dir or f"viper_v2_results_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}  image_size={image_size}  patch_size={patch_size}")
    print(f"Outputs: {out_dir.resolve()}")

    # Load dataset
    dl = load_data_loader_module(args.data_loader_path)
    os.makedirs(args.data_root, exist_ok=True)
    train_loader, val_loader, test_loader, num_classes, img_h, img_w = \
        dl.get_dataset(args.dataset, data_root=args.data_root,
                       batch_size=batch_size, image_size=image_size,
                       seed=args.seeds[0], num_workers=args.num_workers)
    print(f"  {num_classes} classes, {img_h}×{img_w}")
    print(f"  train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  test={len(test_loader.dataset):,}")

    model_params = dict(
        image_h=img_h, image_w=img_w, in_channels=3,
        patch_size=patch_size, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads,
        mlp_dim=args.mlp_dim, num_classes=num_classes, dropout=args.dropout,
    )

    # Iterate over (seed × suite configs)
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
            try:
                r = train_one_v2(
                    pe_type, model_params, cfg,
                    train_loader, val_loader, test_loader,
                    n_epochs=args.epochs if args.suite != "smoke" else 3,
                    lr=args.lr, weight_decay=args.weight_decay,
                    device=DEVICE, num_classes=num_classes, seed=seed,
                    name=run_id,
                )
                all_results.append(r)
                with open(run_path, "w") as f:
                    json.dump(r, f, indent=2, default=str)
            except Exception as e:
                import traceback
                print(f"[ERROR] {run_id}: {e}")
                traceback.print_exc()

    # Summary
    summary_path = out_dir / f"{args.suite}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}\nSummary saved: {summary_path}")
    print(f"{'='*70}")
    print(f"{'name':<30} {'pe_type':<10} {'test_acc':>10} {'test_f1':>10} {'PE params':>12}")
    print("-" * 80)
    rows = [(r["name"], r["pe_type"],
             (r["test"]["acc"] if r["test"] else float("nan")),
             (r["test"]["f1"] if r["test"] else float("nan")),
             r["pe_params"])
            for r in all_results]
    rows.sort(key=lambda x: -x[2] if not math.isnan(x[2]) else 0)
    for name, pe_type, acc, f1, pe_p in rows:
        print(f"{name:<30} {pe_type:<10} {acc:>10.4f} {f1:>10.4f} {pe_p:>12,}")


if __name__ == "__main__":
    main()
