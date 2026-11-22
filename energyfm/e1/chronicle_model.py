"""Chronicle transformer for text and time-series inputs.

Adapted from InertialAI/Chronicle under the Apache License 2.0.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_LEN = 32  # length of one time-series patch


# -----------------------------------------------------------------------------
# Core transformer components (standalone, not dependent on gpt.py)
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale parameter (no bias)."""

    def __init__(self, size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        # RMS normalization
        norm_x = x.float()
        rms = torch.sqrt(torch.mean(norm_x**2, dim=-1, keepdim=True) + 1e-5)
        x_normed = norm_x / rms
        return (self.weight * x_normed).to(x.dtype)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to queries or keys."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


def norm(x):
    """Purely functional rmsnorm with no learnable params (for QK norm)."""
    return F.rms_norm(x, (x.size(-1),))


class CausalSelfAttention(nn.Module):
    """Multi-head or Group Query Attention with rotary embeddings."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings and QK norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm (functional, no params)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply KV cache if present
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=enable_gqa
            )
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )
        else:
            # Allow cached keys and use causal attention within the new chunk.
            prefix_len = Tk - Tq
            key_positions = torch.arange(Tk, device=q.device)
            query_positions = prefix_len + torch.arange(Tq, device=q.device)
            attn_mask = key_positions[None, :] <= query_positions[:, None]
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
            )

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SwiGLU(nn.Module):
    """SwiGLU activation function with 8/3 hidden dimension expansion."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Transformer block with attention and SwiGLU MLP."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = SwiGLU(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(self.attn_norm(x), cos_sin, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# -----------------------------------------------------------------------------
# Time-series specific components
# -----------------------------------------------------------------------------


class InstanceNorm(nn.Module):
    """
    Per-series instance normalization (Chronos-style).
    Computes mean/std per series over MASKED positions only.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None, loc_scale=None):
        """
        Args:
            x: (B, L) - flattened time series per batch item
            mask: (B, L) - 1 for valid, 0 for pad/nan
            loc_scale: Optional (B, 2) tensor with [loc, scale] to reuse

        Returns:
            x_norm: (B, L) - normalized series (masked positions only)
            loc_scale: (B, 2) - [loc, scale] used for normalization
        """
        if loc_scale is None:
            # Compute loc/scale only over masked positions
            if mask is not None:
                # Set NaN where mask is 0, compute nanmean
                x_masked = torch.where(mask > 0, x, torch.nan)
                loc = torch.nanmean(x_masked, dim=1, keepdim=True)  # (B, 1)
                demean = x_masked - loc
                var = torch.nanmean(demean**2, dim=1, keepdim=True)
                scale = torch.sqrt(var + 1e-8)
            else:
                # No mask - use all values
                loc = x.mean(dim=1, keepdim=True)
                scale = x.std(dim=1, keepdim=True) + 1e-8

            loc_scale = torch.cat([loc, scale], dim=1)  # (B, 2)
        else:
            loc = loc_scale[:, 0:1]
            scale = loc_scale[:, 1:2]

        # Normalize - zero out masked positions
        if mask is not None:
            x_norm = torch.where(mask > 0, (x - loc) / scale, 0.0)
        else:
            x_norm = (x - loc) / scale

        return x_norm, loc_scale

    def inverse(self, x_norm, loc_scale):
        """
        Inverse transform back to original scale.

        Args:
            x_norm: (B, L) - normalized values
            loc_scale: (B, 2) - [loc, scale] from forward pass

        Returns:
            x: (B, L) - values in original scale
        """
        loc = loc_scale[:, 0:1]  # (B, 1)
        scale = loc_scale[:, 1:2]  # (B, 1)

        # Denormalize
        x = x_norm * scale + loc

        return x


@dataclass
class ChronicleConfig:
    """Chronicle architecture configuration."""

    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 3  # number of key/value heads (for GQA) - default 2:1 ratio
    n_embd: int = 768
    patch_len: int = PATCH_LEN  # Length of each time series patch
    num_quantiles: int = 21  # Number of quantiles to predict
    tie_weights: bool = True  # Tie embedding and lm_head weights


class PatchProjection(nn.Module):
    """Projects [time_ramp | value_norm | mask] to embedding dimension."""

    def __init__(self, config):
        super().__init__()
        # Input: 4 * patch_len per step, as trained. The fourth channel is
        # reserved; end-to-end series inference ships with the transformers
        # port — the hosted API serves it today.
        self.proj = nn.Linear(4 * config.patch_len, config.n_embd)

    def forward(self, patches_norm, mask, time_ramp):
        """
        Args:
            patches_norm: (B, T, P) - normalized values
            mask: (B, T, P) - validity mask
            time_ramp: (B, T, P) - time positions

        Returns:
            (B, T, n_embd)
        """
        # Concatenate features: [time | value | mask | reserved]
        features = torch.cat(
            [time_ramp, patches_norm, mask, torch.zeros_like(patches_norm)], dim=-1
        )  # (B, T, 4*P)
        return norm(self.proj(features))


class QuantileHead(nn.Module):
    """Predicts quantiles for next patch. Simple."""

    def __init__(self, config):
        super().__init__()
        self.patch_len = config.patch_len
        self.num_quantiles = config.num_quantiles
        self.proj = nn.Linear(config.n_embd, config.patch_len * config.num_quantiles)

    def forward(self, x):
        """x: (B, T, n_embd) -> (B, T, patch_len, num_quantiles)"""
        h = norm(x)
        out = self.proj(h)  # (B, T, patch_len * num_quantiles)
        B, T = out.shape[:2]
        return out.view(B, T, self.patch_len, self.num_quantiles)


class Chronicle(nn.Module):
    """
    Standalone Multimodal GPT that handles both text tokens and time series patches.

    Features:
    - SwiGLU activation (8/3 ffn multiple)
    - Weight tying between embeddings and lm_head
    - Learnable RMSNorm (parametric, with scale but no bias)
    - Group Query Attention (GQA)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Core transformer components
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.embed_norm = RMSNorm(config.n_embd)  # Normalize after embedding
        self.final_norm = RMSNorm(config.n_embd)

        # Output projection (tied or untied with embeddings)
        if config.tie_weights:
            self.lm_head = None  # Will use tied weights
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Time-series specific components
        self.patch_proj = PatchProjection(config)
        self.quantile_head = QuantileHead(config)
        self.ts_instance_norm = InstanceNorm()

        # Rotary embeddings cache
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=500000, device=None
    ):
        """Precompute rotary embeddings."""
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        """Get the device of the model."""
        return self.transformer.wte.weight.device

    def forward(
        self,
        idx,
        targets=None,
        ts_patches=None,
        ts_targets=None,
        ts_mask_in=None,
        ts_mask_tgt=None,
        kv_cache=None,
        loss_reduction="mean",
        text_loss_weight=1.0,
        ts_loss_weight=1.0,
        return_text_logits=True,
    ):
        """
        Unified forward pass for multimodal GPT.

        Every sample has text tokens (even if just BOS/EOS for pure TS).
        Time-series is optional and appended to text embeddings when present.

        Args:
            idx: (B, T_text) - text token IDs (REQUIRED)
            targets: (B, T_text) - text targets for loss
            ts_patches: (B, T_ts, P) - optional TS patches (raw values)
            ts_targets: (B, T_ts, P) - optional TS targets
            ts_mask_in: (B, T_ts, P) - TS input validity mask
            ts_mask_tgt: (B, T_ts, P) - TS target validity mask
            text_loss_weight: Weight for text cross-entropy loss
            ts_loss_weight: Weight for time-series quantile loss
            return_text_logits: Include the vocabulary projection during inference

        Returns:
            If training: combined loss (text + TS)
            If inference: (text_logits, ts_quantiles) or just text_logits
        """
        device = self.get_device()
        B = idx.shape[0]

        # Embed text tokens
        text_embeds = self.transformer.wte(idx)  # (B, T_text, n_embd)

        # Optionally append TS embeddings
        if ts_patches is not None:
            B_ts, T_ts, P = ts_patches.shape
            assert B == B_ts, "Batch size mismatch"

            # Instance normalization (mask-aware)
            values_flat = ts_patches.view(B, -1)
            mask_flat = ts_mask_in.view(B, -1) if ts_mask_in is not None else None
            values_norm, loc_scale = self.ts_instance_norm(values_flat, mask_flat, None)
            values_norm = values_norm.view(B, T_ts, P)

            # Time ramp for positional info
            L = T_ts * P
            time_ramp = torch.arange(-L, 0, device=device, dtype=torch.float32)
            time_ramp = (time_ramp / L).view(1, T_ts, P).expand(B, -1, -1)

            # Project TS to embeddings
            mask_reshaped = (
                ts_mask_in.view(B, T_ts, P)
                if ts_mask_in is not None
                else torch.ones_like(values_norm)
            )
            ts_embeds = self.patch_proj(values_norm, mask_reshaped, time_ramp)

            # Concatenate: [text | TS]
            embeddings = torch.cat([text_embeds, ts_embeds], dim=1)
            T_text = text_embeds.shape[1]
        else:
            embeddings = text_embeds
            T_text = embeddings.shape[1]
            loc_scale = None

        # Transformer
        seq_len = embeddings.shape[1]
        assert seq_len <= self.cos.size(1), (
            f"Sequence length {seq_len} exceeds rotary cache {self.cos.size(1)}"
        )
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (self.cos[:, T0 : T0 + seq_len], self.sin[:, T0 : T0 + seq_len])

        x = self.embed_norm(embeddings)  # Normalize after embedding (like base GPT)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        x = self.final_norm(x)

        # Split outputs
        text_out = x[:, :T_text, :]
        ts_out = x[:, T_text:, :] if ts_patches is not None else None

        # Compute losses
        total_loss = 0.0
        num_losses = 0
        softcap = 15

        if targets is not None:
            # Use tied weights if configured
            if self.lm_head is not None:
                logits = self.lm_head(text_out)
            else:
                # Weight tying: use transposed embedding matrix
                logits = F.linear(text_out, self.transformer.wte.weight)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            logits = logits.float()  # use tf32/fp32 for logits
            text_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction=loss_reduction,
            )
            total_loss = total_loss + text_loss_weight * text_loss
            num_losses += 1

        if ts_targets is not None and ts_out is not None:
            quantiles = self.quantile_head(ts_out)

            # Normalize targets
            tgt_flat = ts_targets.view(B, -1)
            tgt_mask_flat = ts_mask_tgt.view(B, -1) if ts_mask_tgt is not None else None
            tgt_norm, _ = self.ts_instance_norm(tgt_flat, tgt_mask_flat, loc_scale)
            tgt_norm = tgt_norm.view(B, ts_out.shape[1], P)

            ts_loss = quantile_loss(quantiles, tgt_norm, mask=ts_mask_tgt)
            total_loss = total_loss + ts_loss_weight * ts_loss
            num_losses += 1

        # Return loss or predictions
        if num_losses > 0:
            return total_loss

        logits = None
        if return_text_logits:
            if self.lm_head is not None:
                logits = self.lm_head(text_out)
            else:
                logits = F.linear(text_out, self.transformer.wte.weight)
            logits = softcap * torch.tanh(logits / softcap)

        if ts_out is not None:
            quantiles = self.quantile_head(ts_out)
            # Denormalize
            B, T_ts, P, Q = quantiles.shape
            q_flat = quantiles.permute(0, 1, 3, 2).contiguous().view(B, -1)
            q_inv = self.ts_instance_norm.inverse(q_flat, loc_scale)
            q_inv = q_inv.view(B, T_ts, Q, P).permute(0, 1, 3, 2).contiguous()
            return logits, q_inv
        return logits


def quantile_loss(quantile_preds, targets, mask=None, quantiles=None, reduction="mean"):
    """
    Quantile regression loss (pinball loss) with optional masking.

    Args:
        quantile_preds: (B, T, P, Q) - predicted quantiles
        targets: (B, T, P) - actual values
        mask: (B, T, P) - validity mask (1=real, 0=pad)
        quantiles: List of quantile levels (default: 21 quantiles from 0.05 to 0.95)
        reduction: 'mean', 'none', or 'sum'

    Returns:
        loss: Quantile loss
    """
    if quantiles is None:
        quantiles = torch.linspace(0.05, 0.95, 21, device=quantile_preds.device)

    # Expand targets to match quantile predictions
    targets_expanded = targets.unsqueeze(-1)  # (B, T, P, 1)

    # Compute errors
    errors = targets_expanded - quantile_preds  # (B, T, P, Q)

    # Quantile loss (pinball loss)
    quantiles = quantiles.view(1, 1, 1, -1)  # Broadcast
    loss = torch.where(errors >= 0, quantiles * errors, (quantiles - 1) * errors)

    # Apply mask if provided
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)  # (B, T, P, 1)
        loss = loss * mask_expanded
        if reduction == "mean":
            return loss.sum() / (mask.sum() * quantile_preds.size(-1)).clamp(min=1)
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
