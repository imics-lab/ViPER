"""
ViPER v4 — Wavelet-Conditioned CPE (FiLM-modulated PEG).

This version pivots methodologically: rather than competing with CPE on
positional grounds, ViPER v4 INHERITS CPE's working mechanism (depthwise
zero-padded conv per block) and uses wavelet features to MODULATE that
conv via FiLM (Feature-wise Linear Modulation).

Init invariant: γ=0, β=0 ⇒ method reduces exactly to plain CPE at init.
The model can ONLY get better than CPE by learning to use wavelet info.

Architecture
------------

  Image
   ├──→ patch_embed → tokens (B, N+1, d)
   └──→ wavelet_extractor → w (B, d_w, H_p, W_p)   [computed once]

  For each block ℓ:
       tokens → WaveletCPE_ℓ(tokens, w) → tokens
           sp = reshape(tokens → grid)
           conv_out = DWConv3x3_zeropad(sp)         # CPE mechanism
           γ_ℓ, β_ℓ = FiLM_proj_ℓ(w)                # per-block conditioning
           sp = sp + (1+γ_ℓ) ⊙ conv_out + β_ℓ       # FiLM + residual
           tokens = flatten(sp)
       tokens → standard attn + FFN → tokens

Ablation axes
-------------
  • modulation:  film | scale | gate | none      (none ⇒ plain CPE baseline)
  • film_sharing: per_block | shared
  • wavelet_path: idwt | region_pool             (IDWT is default)
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
from pytorch_wavelets import DWTForward, DWTInverse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ViPERConfig:
    # Wavelet decomposition (for computing w)
    n_levels:        int = 3
    wavelet:         str = "db4"
    channel_mode:    str = "gray"           # gray | learnable_proj
    d_pe:            int = 32               # internal dim of wavelet branch

    # Wavelet feature path: how to go from coefficients → (B, d_pe, H_p, W_p)
    wavelet_path:    str = "idwt"           # idwt | region_pool

    # FiLM modulation type
    modulation:      str = "film"           # film | scale | gate | none
    film_sharing:    str = "per_block"      # per_block | shared

    # CPE kernel
    cpe_kernel:      int = 3                # 3x3 depthwise conv

    # Component flags
    use_channel_proj: bool = True
    use_gating:       bool = True


# =============================================================================
# Wavelet feature extraction (Stage 1) — produces w
# =============================================================================

class ChannelProjection(nn.Module):
    """RGB → 1 channel for DWT input. Luminance-initialized."""
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


class WaveletFeatureExtractor(nn.Module):
    """Image → wavelet feature map w of shape (B, d_pe, H_p, W_p).

    Two paths:
      • idwt: DWT → gate at native resolution → IDWT → strided conv to (H_p, W_p)
      • region_pool: DWT → 1×1 lift → adaptive avg pool to (H_p, W_p) → gate
    """
    def __init__(self, cfg: ViPERConfig, in_channels: int, image_h: int,
                 image_w: int, patch_size: int):
        super().__init__()
        self.cfg = cfg
        self.image_h, self.image_w = image_h, image_w
        self.patch_size = patch_size
        self.H_p = image_h // patch_size
        self.W_p = image_w // patch_size
        self.d_pe = cfg.d_pe

        # Channel projection
        if cfg.use_channel_proj:
            self.channel_proj = ChannelProjection(in_channels, cfg.channel_mode)
        else:
            self.channel_proj = nn.Identity()

        # Wavelet ops
        self.dwt = DWTForward(J=cfg.n_levels, wave=cfg.wavelet, mode="symmetric")
        if cfg.wavelet_path == "idwt":
            self.idwt = DWTInverse(wave=cfg.wavelet, mode="symmetric")

        # DyWPE-style gating
        self.n_subbands = 1 + 3 * cfg.n_levels
        if cfg.use_gating:
            self.scale_embeddings = nn.Parameter(
                torch.randn(self.n_subbands, cfg.d_pe) * 0.02
            )
            self.W_g = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)
            self.W_v = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)

        if cfg.wavelet_path == "idwt":
            # After IDWT we have (B, d_pe, H, W); aggregate via strided depthwise conv
            self.aggregate = nn.Conv2d(
                cfg.d_pe, cfg.d_pe, kernel_size=patch_size, stride=patch_size,
                groups=cfg.d_pe, bias=False,
            )
        elif cfg.wavelet_path == "region_pool":
            # We need a 1x1 conv to lift 1-ch coefficients to d_pe (info-poor, but matches v2)
            self.subband_proj = nn.Conv2d(1, cfg.d_pe, kernel_size=1)
        else:
            raise ValueError(f"Unknown wavelet_path: {cfg.wavelet_path}")

    def _compute_gate(self, sb_idx: int) -> torch.Tensor:
        if not self.cfg.use_gating:
            device = next(self.parameters()).device if any(self.parameters()) else 'cpu'
            return torch.ones(self.d_pe, device=device)
        e = self.scale_embeddings[sb_idx]
        g = torch.sigmoid(self.W_g(e))
        v = torch.tanh(self.W_v(e))
        return g * v

    def _modulate_idwt(self, coeffs: torch.Tensor, sb_idx: int) -> torch.Tensor:
        """coeffs: (B, 1, H, W), returns (B, d_pe, H, W) for IDWT path."""
        gate = self._compute_gate(sb_idx).view(1, self.d_pe, 1, 1)
        return coeffs * gate

    def _modulate_pool(self, coeffs: torch.Tensor, sb_idx: int) -> torch.Tensor:
        """For region_pool path: lift (B, 1, H, W) → (B, d_pe, H, W) then gate."""
        lifted = self.subband_proj(coeffs)
        gate = self._compute_gate(sb_idx).view(1, self.d_pe, 1, 1)
        return lifted * gate

    def _pool_to_patch_grid(self, x):
        _, _, h, w = x.shape
        if h == self.H_p and w == self.W_p:
            return x
        if h >= self.H_p and w >= self.W_p:
            return F.adaptive_avg_pool2d(x, (self.H_p, self.W_p))
        return F.interpolate(x, size=(self.H_p, self.W_p),
                             mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, C, H, W) → w: (B, d_pe, H_p, W_p)"""
        if self.cfg.use_channel_proj:
            x = self.channel_proj(image)
        else:
            x = image.mean(dim=1, keepdim=True)

        Yl, Yh = self.dwt(x)

        if self.cfg.wavelet_path == "idwt":
            # Gate at native resolution, then IDWT, then strided aggregation
            mod_Yl = self._modulate_idwt(Yl, 0)                  # (B, d_pe, H/2^J, W/2^J)
            mod_Yh = []
            for j in range(self.cfg.n_levels):
                details = Yh[j]
                per_dir = []
                for d in range(3):
                    sb_idx = 1 + j * 3 + d
                    sb = details[:, :, d]                         # (B, 1, h, w)
                    per_dir.append(self._modulate_idwt(sb, sb_idx))
                mod_Yh.append(torch.stack(per_dir, dim=2))        # (B, d_pe, 3, h, w)

            reconstructed = self.idwt((mod_Yl, mod_Yh))           # (B, d_pe, H, W)
            _, _, H_rec, W_rec = reconstructed.shape
            if H_rec != self.image_h or W_rec != self.image_w:
                reconstructed = F.interpolate(
                    reconstructed, size=(self.image_h, self.image_w),
                    mode="bilinear", align_corners=False
                )
            w = self.aggregate(reconstructed)                      # (B, d_pe, H_p, W_p)

        else:  # region_pool
            ll = self._modulate_pool(Yl, 0)
            ll = self._pool_to_patch_grid(ll)
            accumulator = ll
            for j in range(self.cfg.n_levels):
                details = Yh[j]
                for d in range(3):
                    sb_idx = 1 + j * 3 + d
                    sb = details[:, :, d]
                    sb = self._modulate_pool(sb, sb_idx)
                    sb = self._pool_to_patch_grid(sb)
                    accumulator = accumulator + sb
            w = accumulator                                        # (B, d_pe, H_p, W_p)

        return w


# =============================================================================
# Stage 2: Wavelet-Conditioned CPE (FiLM modulation)
# =============================================================================

class WaveletConditionedCPE(nn.Module):
    """CPE depthwise conv with FiLM modulation from wavelet features.

    Init invariant: when γ=0, β=0, this reduces exactly to plain CPE:
        out = DWConv(x) + x
    """
    def __init__(self, d_model: int, H: int, W: int, d_w: int,
                 modulation: str = "film", kernel: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.d_model = d_model
        self.modulation = modulation

        # Depthwise conv with zero padding (CPE's positional mechanism)
        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=kernel,
                                 padding=kernel // 2, groups=d_model,
                                 padding_mode='zeros')

        # FiLM/scale/gate projection from wavelet features
        if modulation == "film":
            self.film_proj = nn.Conv2d(d_w, 2 * d_model, kernel_size=1)
            self._zero_init(self.film_proj)
        elif modulation == "scale":
            self.film_proj = nn.Conv2d(d_w, d_model, kernel_size=1)
            self._zero_init(self.film_proj)
        elif modulation == "gate":
            # Gate must be near 1 at init (so conv passes through unchanged).
            # Use sigmoid(b) starting near 1 means b should be large; we
            # init bias high and weights to 0.
            self.film_proj = nn.Conv2d(d_w, d_model, kernel_size=1)
            with torch.no_grad():
                self.film_proj.weight.zero_()
                self.film_proj.bias.fill_(3.0)  # sigmoid(3) ≈ 0.95
        elif modulation == "none":
            self.film_proj = None
        else:
            raise ValueError(f"Unknown modulation: {modulation}")

        self.cls_pe = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    @staticmethod
    def _zero_init(layer):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: torch.Tensor, w: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        tokens: (B, N+1, d_model)  — CLS + N patch tokens
        w:      (B, d_w, H, W)     — wavelet conditioning features (per image)
        """
        cls, pts = tokens[:, :1], tokens[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)

        conv_out = self.dwconv(sp)

        if self.modulation == "none" or w is None:
            modulated = conv_out
        elif self.modulation == "film":
            params = self.film_proj(w)                  # (B, 2*C, H, W)
            gamma, beta = params.chunk(2, dim=1)
            modulated = (1.0 + gamma) * conv_out + beta
        elif self.modulation == "scale":
            gamma = self.film_proj(w)
            modulated = (1.0 + gamma) * conv_out
        elif self.modulation == "gate":
            g = torch.sigmoid(self.film_proj(w))
            modulated = g * conv_out

        sp = modulated + sp                              # residual (CPE-style)
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls + self.cls_pe.expand(B, -1, -1), pts], dim=1)


class WaveletCPEBlock(nn.Module):
    """Transformer block: WaveletCPE → MultiheadAttention → FFN."""
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int,
                 dropout: float, H_p: int, W_p: int, d_w: int,
                 cfg: ViPERConfig, shared_cpe: Optional[nn.Module] = None):
        super().__init__()
        if shared_cpe is not None:
            self.wavelet_cpe = shared_cpe
        else:
            self.wavelet_cpe = WaveletConditionedCPE(
                d_model, H_p, W_p, d_w,
                modulation=cfg.modulation, kernel=cfg.cpe_kernel,
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.wavelet_cpe(x, w)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Baseline PEs (copied from v3 for fair head-to-head)
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
        return x + self.pe[:, : x.shape[1]]


class CPE(nn.Module):
    """Single-PEG CPE baseline (the existing baseline)."""
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


class MultiPEG(nn.Module):
    """Multi-layer CPE: PEG applied between every encoder block.

    This is the 'proper' CPVT-style implementation. Used as a stronger baseline.
    Each block needs its own PEG instance, so this is just the marker class;
    actual implementation is in the model's forward pass.
    """
    def __init__(self, d_model, num_layers, H, W, k=3):
        super().__init__()
        self.H, self.W = H, W
        self.pegs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, k, padding=k//2, groups=d_model)
            for _ in range(num_layers)
        ])
        self.cls_pe = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def apply_peg(self, x, layer_idx):
        cls, pts = x[:, :1], x[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)
        sp = self.pegs[layer_idx](sp) + sp
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls + self.cls_pe.expand(B, -1, -1), pts], dim=1)


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
    """ViT supporting:
       • pe_type == 'viper_cpe':  Wavelet-Conditioned CPE per block (v4 method)
       • pe_type == 'multipeg':   Plain CPE applied per block (strong baseline)
       • pe_type == ...other:    Single-application baselines (input only)
    """
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
        self.num_layers = num_layers

        if pe_type == "viper_cpe":
            assert viper_cfg is not None
            self.wavelet_extractor = WaveletFeatureExtractor(
                viper_cfg, in_channels, image_h, image_w, patch_size
            )
            # Per-block or shared FiLM
            if viper_cfg.film_sharing == "shared":
                shared_cpe = WaveletConditionedCPE(
                    d_model, H_p, W_p, viper_cfg.d_pe,
                    modulation=viper_cfg.modulation, kernel=viper_cfg.cpe_kernel,
                )
                self.blocks = nn.ModuleList([
                    WaveletCPEBlock(d_model, num_heads, mlp_dim, dropout,
                                     H_p, W_p, viper_cfg.d_pe, viper_cfg,
                                     shared_cpe=shared_cpe)
                    for _ in range(num_layers)
                ])
            else:  # per_block
                self.blocks = nn.ModuleList([
                    WaveletCPEBlock(d_model, num_heads, mlp_dim, dropout,
                                     H_p, W_p, viper_cfg.d_pe, viper_cfg,
                                     shared_cpe=None)
                    for _ in range(num_layers)
                ])
        elif pe_type == "multipeg":
            # Per-block plain CPE (proper CPVT-style baseline)
            self.multipeg = MultiPEG(d_model, num_layers, H_p, W_p)
            # Standard transformer blocks
            self.blocks_std = nn.ModuleList([
                _PreNormBlock(d_model, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ])
        else:
            # Single-application baseline path
            self.pe_module = self._build_baseline_pe(pe_type, d_model, H_p, W_p, N)
            self.blocks_std = nn.ModuleList([
                _PreNormBlock(d_model, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ])

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

        if self.pe_type == "viper_cpe":
            w = self.wavelet_extractor(image)
            for blk in self.blocks:
                x = blk(x, w)
        elif self.pe_type == "multipeg":
            for i, blk in enumerate(self.blocks_std):
                x = self.multipeg.apply_peg(x, i)
                x = blk(x)
        else:
            x = self.pe_module(x)
            for blk in self.blocks_std:
                x = blk(x)

        return self.head(self.norm(x[:, 0]))


class _PreNormBlock(nn.Module):
    """Standard pre-norm transformer block (for baseline path)."""
    def __init__(self, d_model, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


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


def train_one_v4(pe_type, model_params, viper_cfg, train_loader, val_loader,
                 test_loader, n_epochs=50, lr=3e-4, weight_decay=0.05,
                 device=DEVICE, num_classes=10, seed=42, verbose=True,
                 name=None):
    set_seed(seed)
    model = ViPERViT(pe_type=pe_type, viper_cfg=viper_cfg, **model_params).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if pe_type == "viper_cpe":
        pe_params = sum(p.numel() for p in model.wavelet_extractor.parameters())
        for blk in model.blocks:
            pe_params += sum(p.numel() for p in blk.wavelet_cpe.parameters())
    elif pe_type == "multipeg":
        pe_params = sum(p.numel() for p in model.multipeg.parameters())
    elif hasattr(model, "pe_module"):
        pe_params = sum(p.numel() for p in model.pe_module.parameters())
    else:
        pe_params = 0

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
# Ablation suites
# =============================================================================

def get_default_viper_cfg():
    return ViPERConfig(
        n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32,
        wavelet_path="idwt", modulation="film", film_sharing="per_block",
    )


def get_suite(suite_name: str) -> List[Tuple[str, Optional[ViPERConfig], str]]:
    runs = []

    if suite_name == "smoke":
        cfg = get_default_viper_cfg()
        runs = [("viper_cpe", cfg, "viper_v4_smoke")]

    elif suite_name == "baselines":
        for pe in ["none", "learned", "sincos2d", "rope2d", "relative2d", "cpe", "irpe", "multipeg"]:
            runs.append((pe, None, pe))

    elif suite_name == "main":
        # All baselines + multi-PEG + ViPER v4
        for pe in ["none", "learned", "sincos2d", "rope2d", "relative2d", "cpe", "irpe", "multipeg"]:
            runs.append((pe, None, pe))
        cfg = get_default_viper_cfg()
        runs.append(("viper_cpe", cfg, "viper_v4"))

    elif suite_name == "modulation":
        for mod in ["film", "scale", "gate", "none"]:
            cfg = get_default_viper_cfg()
            cfg.modulation = mod
            runs.append(("viper_cpe", cfg, f"mod_{mod}"))

    elif suite_name == "sharing":
        for share in ["per_block", "shared"]:
            cfg = get_default_viper_cfg()
            cfg.film_sharing = share
            runs.append(("viper_cpe", cfg, f"share_{share}"))

    elif suite_name == "wavelet_path":
        for path in ["idwt", "region_pool"]:
            cfg = get_default_viper_cfg()
            cfg.wavelet_path = path
            runs.append(("viper_cpe", cfg, f"path_{path}"))

    elif suite_name == "wavelets":
        for wav in ["haar", "db1", "db2", "db4", "sym4", "coif1"]:
            cfg = get_default_viper_cfg()
            cfg.wavelet = wav
            runs.append(("viper_cpe", cfg, f"wav_{wav}"))

    elif suite_name == "levels":
        for j in [1, 2, 3, 4]:
            cfg = get_default_viper_cfg()
            cfg.n_levels = j
            runs.append(("viper_cpe", cfg, f"levels_{j}"))

    elif suite_name == "d_pe":
        for d in [8, 16, 32, 64]:
            cfg = get_default_viper_cfg()
            cfg.d_pe = d
            runs.append(("viper_cpe", cfg, f"dpe_{d}"))

    elif suite_name == "contenders":
        # Top baselines + multi-PEG + ViPER v4 for multi-seed comparison
        for pe in ["cpe", "multipeg", "sincos2d", "none"]:
            runs.append((pe, None, pe))
        cfg = get_default_viper_cfg()
        runs.append(("viper_cpe", cfg, "viper_v4"))

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
    parser.add_argument("--dataset", default="eurosat",
                        choices=["eurosat", "bloodmnist", "dermamnist",
                                 "pathmnist", "resisc45", "dtd", "tissuemnist"])
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--suite", default="main",
                        choices=["smoke", "baselines", "main", "modulation",
                                 "sharing", "wavelet_path", "wavelets",
                                 "levels", "d_pe", "contenders"])
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

    DATASET_DEFAULTS = {
        "eurosat":     dict(image_size=64,  patch_size=8,  batch_size=64),
        "bloodmnist":  dict(image_size=224, patch_size=16, batch_size=32),
        "dermamnist":  dict(image_size=224, patch_size=16, batch_size=32),
        "pathmnist":   dict(image_size=224, patch_size=16, batch_size=32),
        "resisc45":    dict(image_size=224, patch_size=16, batch_size=32),
        "dtd":         dict(image_size=224, patch_size=16, batch_size=32),
    }
    d = DATASET_DEFAULTS[args.dataset]
    image_size = args.image_size or d["image_size"]
    patch_size = args.patch_size or d["patch_size"]
    batch_size = args.batch_size or d["batch_size"]

    out_dir = Path(args.out_dir or f"viper_v4_results_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}  image_size={image_size}  patch_size={patch_size}")
    print(f"Outputs: {out_dir.resolve()}")

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
                r = train_one_v4(
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

    summary_path = out_dir / f"{args.suite}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}\nSummary saved: {summary_path}")
    print(f"{'='*70}")
    print(f"{'name':<30} {'pe_type':<12} {'test_acc':>10} {'test_f1':>10} {'PE params':>12}")
    print("-" * 82)
    rows = [(r["name"], r["pe_type"],
             (r["test"]["acc"] if r["test"] else float("nan")),
             (r["test"]["f1"] if r["test"] else float("nan")),
             r["pe_params"])
            for r in all_results]
    rows.sort(key=lambda x: -x[2] if not math.isnan(x[2]) else 0)
    for name, pe_type, acc, f1, pe_p in rows:
        print(f"{name:<30} {pe_type:<12} {acc:>10.4f} {f1:>10.4f} {pe_p:>12,}")


if __name__ == "__main__":
    main()
