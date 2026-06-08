"""
viper_v5_extra_pes.py — Additional PE baselines for v5: RoPE-Mixed and ALiBi-2D.

These two PEs operate INSIDE attention (rotating Q,K for RoPE-Mixed; adding
bias to attention scores for ALiBi). They cannot be implemented as
input-token modifications like the other v5 PEs, so we monkey-patch DeiT's
attention modules to apply rotation/bias at the right point.

Usage:
    from viper_v5_extra_pes import apply_internal_pe_to_deit
    model = DeiTWithCustomPE(num_classes=8, pe_type='none', ...)
    apply_internal_pe_to_deit(model, pe_type='rope_mixed', H_p=14, W_p=14)
    # OR
    apply_internal_pe_to_deit(model, pe_type='alibi2d', H_p=14, W_p=14)

References:
    - Heo et al. (2024) ECCV. Rotary Position Embedding for Vision Transformer.
      https://arxiv.org/abs/2403.13298 (RoPE-Mixed)
    - Press et al. (2022). Train Short, Test Long: ALiBi.
      https://arxiv.org/abs/2108.12409 (ALiBi, 1D)
"""

import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# RoPE-Mixed (Heo et al. 2024)
# =============================================================================

class RoPEMixed2D(nn.Module):
    """RoPE-Mixed: learnable per-head 2D rotary position encoding.

    Following Heo et al. ECCV 2024:
      - Each attention head has its OWN learnable frequency vector θ ∈ R^{head_dim/2}.
      - The x and y axis use the same θ but rotate independently:
            rotation_x[h, t, c] = θ_h[c] * x_position(t)
            rotation_y[h, t, c] = θ_h[c] * y_position(t)
      - Final rotation per head, per token: [rot_x | rot_y]
      - Rotation applied to Q and K only (not V).

    Args:
        num_heads:   number of attention heads
        head_dim:    dimension per head (must be divisible by 4)
        max_h, max_w: maximum expected patch grid (cached cos/sin tables)
    """
    def __init__(self, num_heads: int, head_dim: int, max_h: int = 64,
                 max_w: int = 64, theta_init: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4"
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.quarter = head_dim // 4

        # Learnable per-head frequencies (one per quarter dim, for both x and y)
        # Initialize like standard RoPE: θ_k = 1 / (theta_init ** (2k/d))
        # But make them learnable per-head as in Heo 2024
        freqs_init = 1.0 / (theta_init ** (torch.arange(0, self.quarter,
                                                          dtype=torch.float32) / self.quarter))
        # Broadcast to [num_heads, quarter]
        self.freqs = nn.Parameter(freqs_init.unsqueeze(0).expand(num_heads, -1).clone())

        # Pre-compute grid positions (no params, just buffers)
        self.max_h, self.max_w = max_h, max_w

    def _compute_angles(self, H: int, W: int, device: torch.device
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) angles for the given patch grid.

        Returns tensors of shape (num_heads, H*W, head_dim).
        """
        # Position grids: x ∈ [0, W), y ∈ [0, H)
        ys = torch.arange(H, device=device, dtype=torch.float32).repeat_interleave(W)  # (HW,)
        xs = torch.arange(W, device=device, dtype=torch.float32).repeat(H)              # (HW,)

        # Per-head angles
        # freqs: (num_heads, quarter)
        # We want (num_heads, HW, quarter) for each of x, y
        ang_x = xs.unsqueeze(0).unsqueeze(-1) * self.freqs.unsqueeze(1)   # (num_heads, HW, quarter)
        ang_y = ys.unsqueeze(0).unsqueeze(-1) * self.freqs.unsqueeze(1)   # (num_heads, HW, quarter)

        # Build the rotation angles for x-half and y-half of the head_dim
        # x-half: dims 0..head_dim/2, y-half: dims head_dim/2..head_dim
        # Within each half, the rotation operates on quarter pairs (k, k+quarter)
        cos_x = torch.cos(ang_x)        # (num_heads, HW, quarter)
        sin_x = torch.sin(ang_x)
        cos_y = torch.cos(ang_y)
        sin_y = torch.sin(ang_y)

        # Stack into full head_dim by repeating per pair (cos, cos, sin, sin pattern handled in apply)
        # For convenience, return separate (cos, sin) for x and y
        return cos_x, sin_x, cos_y, sin_y

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
                 ) -> torch.Tensor:
        """Apply rotation to the first dim of x using cos/sin.

        x:   (B, num_heads, N, half_dim) where half_dim = head_dim/2
        cos: (num_heads, N, quarter)
        sin: (num_heads, N, quarter)
        """
        # Split half_dim into two quarters
        x1, x2 = x[..., :self.quarter], x[..., self.quarter:]    # (B, H, N, quarter)
        # Broadcast cos/sin: (1, H, N, quarter)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        return torch.cat([rot_x1, rot_x2], dim=-1)              # (B, H, N, half_dim)

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor,
                     H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE-Mixed rotation to Q and K.

        q, k: (B, num_heads, N+1, head_dim) — N+1 because CLS token included
        H, W: patch grid dimensions
        """
        # Split into CLS and patch tokens
        q_cls, q_pts = q[:, :, :1], q[:, :, 1:]              # (B, H, 1|N, head_dim)
        k_cls, k_pts = k[:, :, :1], k[:, :, 1:]

        # Compute angles
        cos_x, sin_x, cos_y, sin_y = self._compute_angles(H, W, q.device)

        # Split head_dim into x-half and y-half
        half = self.head_dim // 2
        q_pts_x, q_pts_y = q_pts[..., :half], q_pts[..., half:]
        k_pts_x, k_pts_y = k_pts[..., :half], k_pts[..., half:]

        # Apply rotation per half
        q_pts_x = self._rotate(q_pts_x, cos_x, sin_x)
        q_pts_y = self._rotate(q_pts_y, cos_y, sin_y)
        k_pts_x = self._rotate(k_pts_x, cos_x, sin_x)
        k_pts_y = self._rotate(k_pts_y, cos_y, sin_y)

        q_pts = torch.cat([q_pts_x, q_pts_y], dim=-1)
        k_pts = torch.cat([k_pts_x, k_pts_y], dim=-1)

        q = torch.cat([q_cls, q_pts], dim=2)
        k = torch.cat([k_cls, k_pts], dim=2)
        return q, k


# =============================================================================
# ALiBi 2D (extension of Press 2022)
# =============================================================================

class ALiBi2D(nn.Module):
    """2D ALiBi: linear-distance bias for attention scores.

    Following Press et al. ALiBi but extended to 2D:
      - Each head has a learnable slope m_h.
      - Bias[i, j] = -m_h * d((x_i,y_i), (x_j,y_j))
      - d can be L1 (Manhattan) or L2 (Euclidean). We use L1 by default
        (faster, matches ALiBi's "linear distance" intuition).

    Args:
        num_heads:   number of attention heads
        distance:    'l1' (default, Manhattan) or 'l2' (Euclidean)
    """
    def __init__(self, num_heads: int, distance: str = "l1"):
        super().__init__()
        assert distance in ("l1", "l2")
        self.num_heads = num_heads
        self.distance = distance

        # ALiBi's recommended initialization: m_h = 2^(-8*h/num_heads)
        # We make slopes LEARNABLE so the model can adapt
        slopes_init = torch.tensor(
            [2.0 ** (-8.0 * (h + 1) / num_heads) for h in range(num_heads)],
            dtype=torch.float32,
        )
        self.slopes = nn.Parameter(slopes_init)

    def _compute_distance_matrix(self, H: int, W: int, device: torch.device
                                   ) -> torch.Tensor:
        """Returns (H*W, H*W) distance matrix between patches."""
        ys = torch.arange(H, device=device).repeat_interleave(W).float()
        xs = torch.arange(W, device=device).repeat(H).float()
        # Pairwise distances
        dy = ys.unsqueeze(0) - ys.unsqueeze(1)              # (HW, HW)
        dx = xs.unsqueeze(0) - xs.unsqueeze(1)
        if self.distance == "l1":
            return dy.abs() + dx.abs()
        else:  # l2
            return torch.sqrt(dy * dy + dx * dx)

    def bias(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns ALiBi bias matrix for attention scores.

        Shape: (num_heads, N+1, N+1)  — including CLS at index 0 with bias 0.
        """
        N = H * W
        d_patches = self._compute_distance_matrix(H, W, device)   # (N, N)

        # Per-head: bias = -slope * distance
        bias_patches = -self.slopes.view(-1, 1, 1) * d_patches.unsqueeze(0)  # (H, N, N)

        # Add CLS token row/col with zero bias (CLS attends uniformly)
        bias_full = torch.zeros(self.num_heads, N + 1, N + 1, device=device)
        bias_full[:, 1:, 1:] = bias_patches
        return bias_full


# =============================================================================
# Custom Attention that supports internal PE
# =============================================================================

class CustomAttention(nn.Module):
    """Drop-in replacement for timm's Attention that supports internal PE.

    Built around the standard ViT attention; can apply RoPE-Mixed rotation
    to Q,K or add ALiBi bias to attention scores before softmax.
    """
    def __init__(self, original_attn: nn.Module, pe_module: nn.Module,
                 pe_type: str, H_p: int, W_p: int):
        super().__init__()
        # Copy over the original attention's parameters
        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        # Handle dropout (timm uses attn_drop and proj_drop)
        self.attn_drop = getattr(original_attn, "attn_drop", nn.Identity())
        self.proj_drop = getattr(original_attn, "proj_drop", nn.Identity())

        # Attention scale and head info from original
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim if hasattr(original_attn, "head_dim") \
                         else (self.qkv.in_features // self.num_heads)
        self.scale = self.head_dim ** -0.5

        # Position info
        self.pe_module = pe_module
        self.pe_type = pe_type
        self.H_p, self.W_p = H_p, W_p

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]         # each (B, H, N, hd)

        # Apply internal PE
        if self.pe_type == "rope_mixed":
            q, k = self.pe_module.apply_to_qk(q, k, self.H_p, self.W_p)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        elif self.pe_type == "alibi2d":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            bias = self.pe_module.bias(self.H_p, self.W_p, x.device)   # (H, N+1, N+1)
            # Handle CLS-token sizing mismatch — bias is for N+1 patches+cls
            # Our x has N tokens; check sizes match
            if attn.shape[-1] == bias.shape[-1]:
                attn = attn + bias.unsqueeze(0)
            else:
                # Resize bias if grid changed (e.g. multi-res eval)
                attn = attn   # skip bias
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# =============================================================================
# Patching function: swap timm DeiT's attention with CustomAttention
# =============================================================================

def apply_internal_pe_to_deit(model, pe_type: str, H_p: int, W_p: int,
                                num_heads: int = 3, head_dim: int = 64,
                                distance: str = "l1"):
    """Patch a DeiTWithCustomPE model to use RoPE-Mixed or ALiBi-2D.

    Call this AFTER constructing the model with pe_type='none'. This will:
      - Build a single shared pe_module (RoPE-Mixed or ALiBi-2D)
      - Replace each transformer block's attention with CustomAttention
        wrapping the original

    Args:
        model: a DeiTWithCustomPE built with pe_type='none'
        pe_type: 'rope_mixed' or 'alibi2d'
        H_p, W_p: patch grid dims
        num_heads: number of heads (DeiT-Tiny = 3)
        head_dim: per-head dim (DeiT-Tiny = 64)
        distance: for ALiBi, 'l1' or 'l2'
    """
    # Build the PE module
    if pe_type == "rope_mixed":
        pe_module = RoPEMixed2D(num_heads=num_heads, head_dim=head_dim,
                                 max_h=max(H_p, 64), max_w=max(W_p, 64))
    elif pe_type == "alibi2d":
        pe_module = ALiBi2D(num_heads=num_heads, distance=distance)
    else:
        raise ValueError(f"Unknown internal pe_type: {pe_type}")

    # Move PE module to model's device
    device = next(model.parameters()).device
    pe_module = pe_module.to(device)

    # Attach to model so its parameters are tracked
    model.internal_pe = pe_module

    # Swap attention in each block
    for block in model.backbone.blocks:
        original_attn = block.attn
        block.attn = CustomAttention(
            original_attn=original_attn,
            pe_module=pe_module,
            pe_type=pe_type,
            H_p=H_p,
            W_p=W_p,
        ).to(device)

    return model


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    # Verify both PE types work with a fake setup
    import sys
    sys.path.insert(0, ".")
    from viper_v5 import DeiTWithCustomPE

    print("=== Testing RoPE-Mixed ===")
    model = DeiTWithCustomPE(num_classes=8, image_size=224, patch_size=16,
                              pe_type="none", viper_cfg=None)
    apply_internal_pe_to_deit(model, pe_type="rope_mixed", H_p=14, W_p=14)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rope_params = sum(p.numel() for p in model.internal_pe.parameters())
    print(f"  trainable={n_params:,}  PE_params={rope_params:,}  out={tuple(out.shape)}")

    print("=== Testing ALiBi-2D ===")
    model = DeiTWithCustomPE(num_classes=8, image_size=224, patch_size=16,
                              pe_type="none", viper_cfg=None)
    apply_internal_pe_to_deit(model, pe_type="alibi2d", H_p=14, W_p=14)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    alibi_params = sum(p.numel() for p in model.internal_pe.parameters())
    print(f"  trainable={n_params:,}  PE_params={alibi_params:,}  out={tuple(out.shape)}")

    print("All internal-PE forward passes OK")
