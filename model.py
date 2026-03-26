import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect as _inspect

# Optional: optimised Triton kernels (pip install flash-linear-attention)
try:
    from fla.ops.rwkv6 import chunk_rwkv6 as _fla_chunk_rwkv6
    from fla.ops.rwkv6 import fused_recurrent_rwkv6 as _fla_fused_recurrent_rwkv6
    _HAS_FLA_V6 = True
except ImportError:
    _HAS_FLA_V6 = False
try:
    from fla.ops.rwkv7 import chunk_rwkv7 as _fla_chunk_rwkv7
    from fla.ops.rwkv7 import fused_recurrent_rwkv7 as _fla_fused_recurrent_rwkv7
    _HAS_FLA_V7 = True
except ImportError:
    _HAS_FLA_V7 = False


# ======================================================================
#  Backbone blocks (LSTM / GRU / minGRU / Transformer)
# ======================================================================

AVAILABLE_BACKBONES = [
    "mamba", "mambav1", "mamba2",
    "lstm", "gru", "mingru",
    "transformer",
    "rwkv6", "rwkv7", "rwkv8", "griffin", "xlstm",
]


class _FeedForward(nn.Module):
    """Position-wise feed-forward shared across backbones."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ── LSTM ──────────────────────────────────────────────────────────

class LSTMBlock(nn.Module):
    """LSTM backbone block with pre-norm residual and feed-forward."""

    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _FeedForward(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        y, _ = self.lstm(y)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        d = self.lstm.hidden_size
        h0 = torch.zeros(1, batch_size, d, device=device)
        c0 = torch.zeros(1, batch_size, d, device=device)
        return (h0, c0)

    def step(self, x, cache):
        h, c = cache
        y = self.ln1(x).unsqueeze(1)
        y, (h, c) = self.lstm(y, (h, c))
        y = y.squeeze(1)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, (h, c)


# ── GRU ───────────────────────────────────────────────────────────

class GRUBlock(nn.Module):
    """GRU backbone block with pre-norm residual and feed-forward.

    Uses the full ``nn.GRU`` (with hidden-to-hidden weights W_hh)
    for maximum expressivity.  Training is sequential over L, but
    cuDNN fuses the kernels so wall-clock is reasonable.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _FeedForward(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        y, _ = self.gru(y)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        d = self.gru.hidden_size
        return torch.zeros(1, batch_size, d, device=device)

    def step(self, x, cache):
        h = cache
        y = self.ln1(x).unsqueeze(1)
        y, h = self.gru(y, h)
        y = y.squeeze(1)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, h


# ── minGRU  (Feng et al., 2024 — "Were RNNs All We Needed?") ─────
#
#   z_t  = σ(W_z · x_t + b_z)             gate (input-only)
#   h̃_t = W_h · x_t + b_h                candidate (input-only)
#   h_t  = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t
#
# No hidden-to-hidden weights ⇒ the linear recurrence is amenable to
# a parallel prefix (associative) scan during training.  The sequential
# fallback here is correct and simple; swap in a CUDA scan kernel for
# O(L / P) wall-clock on very long sequences.
# ──────────────────────────────────────────────────────────────────

from minGRU_pytorch.minGRU import (
    heinsen_associative_scan_log,
    log_g as _mingru_log_g,
    g as _mingru_g,
)


class _MinGRUCell(nn.Module):
    """Minimal GRU cell — log-space parallel scan (Feng et al., 2024).

    Training uses the numerically stable log-space formulation with
    Heinsen's associative scan (``cumsum`` + ``logcumsumexp`` — two CUDA
    kernels instead of L sequential Python iterations).

    Hidden states are constrained to be positive via ``g()`` (Appendix
    B.3 of the paper).  The single-step path (``step``) uses the same
    ``g()`` so training and inference are consistent.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear_z = nn.Linear(d_model, d_model)
        self.linear_h = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x, h_prev=None):
        """x: [B, L, D] → [B, L, D]  (parallel log-space scan)"""
        B, L, D = x.shape
        gate_logits = self.linear_z(x)                     # [B, L, D]
        h_candidate = self.linear_h(x)                     # [B, L, D]

        # Log-space parallel scan (Appendix B.3)
        log_coeffs = -F.softplus(gate_logits)              # log(1 − σ(z))
        log_z      = -F.softplus(-gate_logits)             # log(σ(z))
        log_values = log_z + _mingru_log_g(h_candidate)    # log(z · g(h̃))

        if h_prev is not None:
            log_values = torch.cat([h_prev.clamp(min=1e-8).log().unsqueeze(1),
                                    log_values], dim=1)
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

        h = heinsen_associative_scan_log(log_coeffs, log_values)
        return h[:, -L:]

    def step(self, x, h_prev):
        """Single-step recurrence.  x, h_prev: [B, D]"""
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = _mingru_g(self.linear_h(x))
        h = (1 - z) * h_prev + z * h_tilde
        return h, h


class MinGRUBlock(nn.Module):
    """minGRU backbone block with pre-norm residual and feed-forward."""

    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mingru = _MinGRUCell(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _FeedForward(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        y = self.mingru(y)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        return torch.zeros(batch_size, self.mingru.d_model, device=device)

    def step(self, x, cache):
        y = self.ln1(x)
        y, cache = self.mingru.step(y, cache)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, cache


# ── Transformer  (RoPE + GQA + SwiGLU + Attention-Sink Sliding-Window)
#
# Three targeted fixes over the vanilla version:
#
# 1. **No train/test mismatch** – windowed causal attention is used
#    during training too (when the sequence is longer than the window),
#    so the model never learns to rely on context it won't have during
#    streaming.  The first ``n_sink`` positions are always visible to
#    every query (attention sinks).
#
# 2. **SwiGLU FFN** – gated feed-forward (Shazeer 2020, used by LLaMA /
#    Mistral / DeepSeek) gives better parameter efficiency than the
#    plain GELU FFN.
#
# 3. **Grouped-Query Attention (GQA)** – shares KV heads across
#    multiple query heads (Ainslie et al. 2023), shrinking the KV
#    cache by ``n_heads / n_kv_heads`` and freeing capacity.
#
# Streaming: KV cache keeps the first ``n_sink`` tokens plus a sliding
# window of the most recent ``window_size`` tokens, bounding memory to
# O(n_sink + window_size) per layer — same as before but now matched to
# training.
#
# References
#   Xiao et al., "Efficient Streaming Language Models with Attention
#       Sinks", 2023.
#   Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
#       Models from Multi-Head Checkpoints", 2023.
#   Shazeer, "GLU Variants Improve Transformer", 2020.
# ──────────────────────────────────────────────────────────────────

class _RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (Zhang & Sennrich 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate(x) * V(x) mapped back to d_model.

    Uses 8/3·d_model hidden dim (≈ 2.67x), similar param count to 4x GELU
    FFN but with gating.
    """

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(8 * d_model / 3)
        # Round to nearest multiple of 8 for tensor-core efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _make_sliding_window_mask(L: int, window_size: int, n_sink: int,
                              device: torch.device) -> torch.Tensor:
    """Build a [L, L] bool attention mask for windowed + sink attention.

    Entry ``mask[i, j] = True`` means query-i **cannot** attend to key-j.
    Compatible with ``F.scaled_dot_product_attention(attn_mask=...)`` which
    treats True as "masked out" (−inf) when the mask dtype is bool.
    """
    # Start with everything masked
    mask = torch.ones(L, L, dtype=torch.bool, device=device)
    row = torch.arange(L, device=device).unsqueeze(1)
    col = torch.arange(L, device=device).unsqueeze(0)

    # (a) Causal: can only attend to past+self
    causal = col <= row
    # (b) Within sliding window
    in_window = (row - col) < window_size
    # (c) Sink positions (first n_sink tokens are always visible)
    is_sink = col < n_sink

    visible = causal & (in_window | is_sink)
    mask = ~visible  # True = masked-out for SDPA
    return mask


class TransformerBlock(nn.Module):
    """Transformer block: RoPE + GQA + SwiGLU + sliding-window attention.

    Parameters
    ----------
    d_model     : int   – hidden dimension.
    n_heads     : int   – number of **query** heads (default 4).
    n_kv_heads  : int   – number of KV heads for GQA.  Must divide n_heads
                          evenly.  Default = n_heads (= standard MHA).
                          Set to 1 for Multi-Query Attention.
    window_size : int   – sliding-window width (default 4096).
    n_sink      : int   – attention-sink positions (default 4).
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_kv_heads: int = 0,
                 window_size: int = 4096, n_sink: int = 4):
        super().__init__()
        if n_kv_heads <= 0:
            n_kv_heads = n_heads          # default: standard MHA
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # GQA repeat factor
        self.d_head     = d_model // n_heads
        self.d_model    = d_model
        self.window_size = window_size
        self.n_sink      = n_sink

        self.ln1 = _RMSNorm(d_model)
        # Separate Q and KV projections for GQA
        self.q_proj  = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = _RMSNorm(d_model)
        self.ff = _SwiGLU(d_model)

        # RoPE inverse frequencies (not a trained parameter)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.d_head, 2).float() / self.d_head)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    # ---- RoPE helpers ------------------------------------------------

    def _apply_rope(self, x, positions):
        """Apply rotary position embeddings.

        x         : [B, n_heads, L, d_head]
        positions : LongTensor [L]
        """
        freqs = torch.outer(positions.float(), self.rope_inv_freq)  # [L, dh/2]
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)                 # [1,1,L,dh/2]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        x1 = x[..., : self.d_head // 2]
        x2 = x[..., self.d_head // 2 :]
        return torch.cat([x1 * cos - x2 * sin,
                          x2 * cos + x1 * sin], dim=-1)

    def _expand_kv(self, kv):
        """Repeat KV heads to match the number of query heads (GQA)."""
        if self.n_rep == 1:
            return kv
        B, n_kv, L, D = kv.shape
        return kv[:, :, None, :, :].expand(B, n_kv, self.n_rep, L, D) \
                   .reshape(B, self.n_heads, L, D)

    # ---- full-sequence forward (training) ----------------------------
    # Uses the SAME sliding-window + sink mask as inference so there is
    # no distribution shift.

    def forward(self, x, inference_params=None):
        B, L, D = x.shape
        y = self.ln1(x)

        q = self.q_proj(y).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(y).reshape(B, L, 2, self.n_kv_heads, self.d_head)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)         # [B, n_kv, L, dh]
        v = v.transpose(1, 2)

        positions = torch.arange(L, device=x.device)
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        # Expand KV heads for GQA
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        # Build sliding-window + sink mask (matches inference behaviour)
        if L <= self.window_size + self.n_sink:
            # Short sequence — plain causal is equivalent & faster
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = _make_sliding_window_mask(L, self.window_size, self.n_sink,
                                             device=x.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).reshape(B, L, D)
        y = self.out_proj(y)
        x = x + y

        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    # ---- streaming helpers -------------------------------------------

    def init_cache(self, batch_size: int, device):
        """Return empty KV cache: (k, v, step_count)."""
        k = torch.zeros(batch_size, self.n_kv_heads, 0, self.d_head, device=device)
        v = torch.zeros(batch_size, self.n_kv_heads, 0, self.d_head, device=device)
        return (k, v, 0)

    def step(self, x, cache):
        """Single-token step with attention-sink + sliding-window eviction.

        x     : [B, D]
        cache : (k_cache, v_cache, step_count)
        """
        k_cache, v_cache, step_count = cache
        B = x.shape[0]

        y = self.ln1(x).unsqueeze(1)                                # [B, 1, D]
        q = self.q_proj(y).reshape(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(y).reshape(B, 1, 2, self.n_kv_heads, self.d_head)
        k_new, v_new = kv.unbind(dim=2)
        k_new = k_new.transpose(1, 2)                               # [B, n_kv, 1, dh]
        v_new = v_new.transpose(1, 2)

        # RoPE at the current absolute position
        pos_t = torch.tensor([step_count], device=x.device)
        q     = self._apply_rope(q, pos_t)
        k_new = self._apply_rope(k_new, pos_t)

        # Append to cache
        k_cache = torch.cat([k_cache, k_new], dim=2)
        v_cache = torch.cat([v_cache, v_new], dim=2)
        step_count += 1

        # Evict: keep first n_sink entries + last window_size entries
        max_cache = self.n_sink + self.window_size
        if k_cache.shape[2] > max_cache:
            k_cache = torch.cat([k_cache[:, :, :self.n_sink],
                                 k_cache[:, :, -self.window_size:]], dim=2)
            v_cache = torch.cat([v_cache[:, :, :self.n_sink],
                                 v_cache[:, :, -self.window_size:]], dim=2)

        # Expand KV for GQA and attend
        k_exp = self._expand_kv(k_cache)
        v_exp = self._expand_kv(v_cache)

        # Expand Q to match — already has n_heads
        scale = self.d_head ** -0.5
        attn = (q @ k_exp.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v_exp)                                          # [B,nh,1,dh]
        y = y.transpose(1, 2).reshape(B, 1, self.d_model)
        y = self.out_proj(y).squeeze(1)                              # [B, D]

        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, (k_cache, v_cache, step_count)


# ── RWKV-6  (Peng et al., 2024 — "Eagle and Finch") ──────────────
#
#   Linear-complexity RNN with data-dependent decay and matrix-valued
#   hidden state.  Parallelisable via a prefix-sum scan; sequential
#   fallback here for portability.
#
#   Time-mixing:
#     Token-shift interpolation → R, K, V, G projections
#     Data-dependent decay:  w_t = base_w + W_w(x_t)
#     Matrix state update:   S_t = diag(exp(w_t)) · S_{t-1} + k_t v_t^T
#     Output:                o_t = (S_t @ r_t) ⊙ gate_t
# ──────────────────────────────────────────────────────────────────

class _RWKV6TimeMix(nn.Module):
    """RWKV-6 time-mixing with optimised WKV computation.

    Uses FLA Triton kernels (``pip install flash-linear-attention``) when
    available on CUDA, otherwise falls back to a pure-PyTorch chunk-wise
    parallel implementation (O(C²) matmuls per chunk, O(L/C) sequential
    chunk iterations).  Both paths are numerically equivalent.
    """

    def __init__(self, d_model: int, n_heads: int = 4, chunk_size: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.chunk_size = chunk_size

        # Token-shift interpolation coefficients
        self.mix_r = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_g = nn.Parameter(torch.ones(d_model) * 0.5)

        # Projections
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Data-dependent decay  (official: -softplus(-x) - 0.5 ensures w < -0.5)
        self.base_decay = nn.Parameter(torch.zeros(n_heads, self.d_head) + 0.5)
        self.W_decay = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        nn.init.zeros_(self.W_decay.weight)

        # Bonus (u) — per-head position-0 bias, as in the official RWKV
        self.bonus = nn.Parameter(torch.zeros(n_heads, self.d_head))

        self.ln_out = nn.GroupNorm(n_heads, d_model)

    def _token_shift(self, x, last_x):
        """Shift x by one position, prepending last_x (or zeros)."""
        if last_x is None:
            last_x = torch.zeros_like(x[:, 0])
        return torch.cat([last_x.unsqueeze(1), x[:, :-1]], dim=1)

    @staticmethod
    def _decay(base, delta):
        """-softplus(-(base + delta)) - 0.5  →  always < -0.5."""
        return -F.softplus(-(base + delta)) - 0.5

    # ── Pure-PyTorch chunk-wise fallback ──────────────────────────────

    def _chunk_wkv(self, r, k, v, w, initial_state):
        """Chunk-wise parallel WKV (pure PyTorch, works on CPU & GPU).

        Args:
            r, k, v: [B, L, H, dh]
            w:       [B, L, H, dh]  (log-decay, negative)
            initial_state: [B, H, dh, dh] or None
        Returns:
            out: [B, L, H, dh],  state: [B, H, dh, dh]
        """
        B, L, H, dh = r.shape
        C = self.chunk_size

        # Pad to multiple of C
        pad = (C - L % C) % C
        if pad > 0:
            r = F.pad(r, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            w = F.pad(w, (0, 0, 0, 0, 0, pad))
        Lp = L + pad
        nc = Lp // C

        r = r.view(B, nc, C, H, dh)
        k = k.view(B, nc, C, H, dh)
        v = v.view(B, nc, C, H, dh)
        w = w.view(B, nc, C, H, dh)

        w_cumsum = w.cumsum(dim=2)

        # Intra-chunk: causal attention with decay
        rel_log = w_cumsum.unsqueeze(3) - w_cumsum.unsqueeze(2)
        causal = torch.tril(torch.ones(C, C, device=r.device, dtype=torch.bool))
        causal = causal.view(1, 1, C, C, 1, 1)

        attn = (r.unsqueeze(3) * k.unsqueeze(2) * torch.exp(rel_log))
        attn = attn.masked_fill(~causal, 0.0).sum(dim=-1)   # [B, nc, Ct, Cs, H]
        o_intra = torch.einsum('bncsh,bnshd->bnchd',
                               attn.permute(0, 1, 2, 3, 4), v)

        # Inter-chunk: propagate recurrent state
        state = initial_state if initial_state is not None else r.new_zeros(B, H, dh, dh)
        inter_list = []
        for c_idx in range(nc):
            r_c = r[:, c_idx]
            decay_c = torch.exp(w_cumsum[:, c_idx])
            o_inter_c = torch.einsum('bhde,bche->bchd', state, r_c * decay_c)
            inter_list.append(o_inter_c)

            total_decay = torch.exp(w_cumsum[:, c_idx, -1])
            state = state * total_decay.unsqueeze(-1)
            kv_decay = torch.exp(w_cumsum[:, c_idx, -1:] - w_cumsum[:, c_idx])
            state = state + torch.einsum('bchd,bche->bhde',
                                         k[:, c_idx] * kv_decay, v[:, c_idx])

        o_inter = torch.stack(inter_list, dim=1)
        out = (o_intra + o_inter).reshape(B, Lp, H, dh)
        if pad > 0:
            out = out[:, :L]
        return out, state

    # ── Forward (selects FLA kernels or PyTorch fallback) ─────────────

    def forward(self, x, last_x=None, state=None):
        """x: [B, L, D] → ([B, L, D], last_x_new, state_new)."""
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        x_prev = self._token_shift(x, last_x)

        # Interpolated inputs
        r = self.W_r(x * self.mix_r + x_prev * (1 - self.mix_r)).view(B, L, H, dh)
        k = self.W_k(x * self.mix_k + x_prev * (1 - self.mix_k)).view(B, L, H, dh)
        v = self.W_v(x * self.mix_v + x_prev * (1 - self.mix_v)).view(B, L, H, dh)
        g = torch.sigmoid(self.W_g(x * self.mix_g + x_prev * (1 - self.mix_g)))

        # L2-normalize k per head (prevents state blow-up)
        k = F.normalize(k, p=2, dim=-1)

        w = self._decay(self.base_decay, self.W_decay(x).view(B, L, H, dh))

        # Dispatch to FLA Triton kernels when on CUDA, else PyTorch fallback
        if _HAS_FLA_V6 and x.is_cuda:
            fn = _fla_fused_recurrent_rwkv6 if L <= 64 else _fla_chunk_rwkv6
            out, s = fn(
                r=r, k=k, v=v, w=w,
                u=self.bonus,
                scale=1.0,
                initial_state=state,
                output_final_state=True,
            )
        else:
            out, s = self._chunk_wkv(r, k, v, w, state)

        out = out.reshape(B, L, D)
        out = self.ln_out(out.transpose(1, 2)).transpose(1, 2)
        out = out * g
        return self.W_o(out), x[:, -1], s

    def step(self, x, last_x, state):
        """Single-token step.  x, last_x: [B, D], state: [B, H, dh, dh]."""
        B = x.shape[0]
        H, dh = self.n_heads, self.d_head

        r = self.W_r(x * self.mix_r + last_x * (1 - self.mix_r)).view(B, H, dh)
        k = self.W_k(x * self.mix_k + last_x * (1 - self.mix_k)).view(B, H, dh)
        v = self.W_v(x * self.mix_v + last_x * (1 - self.mix_v)).view(B, H, dh)
        g = torch.sigmoid(self.W_g(x * self.mix_g + last_x * (1 - self.mix_g)))

        k = F.normalize(k, p=2, dim=-1)

        w = self._decay(self.base_decay, self.W_decay(x).view(B, H, dh))
        decay = torch.exp(w)

        state = state * decay.unsqueeze(-1) + torch.einsum('bhd,bhe->bhde', k, v)
        o = torch.einsum('bhde,bhd->bhe', state, r).reshape(B, self.d_model)
        o = self.ln_out(o.unsqueeze(-1)).squeeze(-1)
        o = o * g
        return self.W_o(o), x, state


class RWKV6Block(nn.Module):
    """RWKV-6 backbone block with pre-norm residual and SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int = 4, chunk_size: int = 32):
        super().__init__()
        self.ln1 = _RMSNorm(d_model)
        self.time_mix = _RWKV6TimeMix(d_model, n_heads, chunk_size=chunk_size)
        self.ln2 = _RMSNorm(d_model)
        self.ff = _SwiGLU(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        y, _, _ = self.time_mix(y)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        tm = self.time_mix
        last_x = torch.zeros(batch_size, tm.d_model, device=device)
        state = torch.zeros(batch_size, tm.n_heads, tm.d_head, tm.d_head,
                            device=device)
        return (last_x, state)

    def step(self, x, cache):
        last_x, state = cache
        y = self.ln1(x)
        y, last_x, state = self.time_mix.step(y, last_x, state)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, (last_x, state)


# ── RWKV-7 "Goose"  (Peng et al., 2025) ─────────────────────────
#
#   Extends RWKV-6 with a *non-diagonal* state transition matrix,
#   enabling cross-dimension state interactions (beyond TC⁰).
#
#   State update (per head):
#     S_t = (diag(w_t) + a_t b_t^T) · S_{t-1}  +  k_t v_t^T
#     o_t = S_t @ r_t
#
#   The rank-1 term  a_t b_t^T  lets the model perform "copy"
#   state transitions and recognize all regular languages.
#
#   Uses FLA Triton kernels on CUDA, sequential fallback on CPU.
# ──────────────────────────────────────────────────────────────────

class _RWKV7TimeMix(nn.Module):
    """RWKV-7 time-mixing with non-diagonal state transition."""

    def __init__(self, d_model: int, n_heads: int = 4, chunk_size: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.chunk_size = chunk_size

        # Token-shift interpolation
        self.mix_r = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_g = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_a = nn.Parameter(torch.ones(d_model) * 0.5)

        # Projections
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Non-diagonal state transition: a, b projections (zero-init)
        self.W_a = nn.Linear(d_model, d_model, bias=False)
        self.W_b = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.W_a.weight)
        nn.init.zeros_(self.W_b.weight)

        # Data-dependent decay  (official: -softplus(-x) - 0.5)
        self.base_decay = nn.Parameter(torch.zeros(n_heads, self.d_head) + 0.5)
        self.W_decay = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        nn.init.zeros_(self.W_decay.weight)

        # Bonus (position-0 bias)
        self.bonus = nn.Parameter(torch.zeros(n_heads, self.d_head))

        self.ln_out = nn.GroupNorm(n_heads, d_model)

    def _token_shift(self, x, last_x):
        if last_x is None:
            last_x = torch.zeros_like(x[:, 0])
        return torch.cat([last_x.unsqueeze(1), x[:, :-1]], dim=1)

    @staticmethod
    def _decay(base, delta):
        """-softplus(-(base + delta)) - 0.5  →  always < -0.5."""
        return -F.softplus(-(base + delta)) - 0.5

    def _chunk_sequential_wkv(self, r, k, v, w, a, b, initial_state):
        """Chunk-sequential WKV with non-diagonal transition.

        Processes chunks of size C sequentially, but vectorises
        the per-token state readout within each chunk so the
        Python loop runs L/C times instead of L times.
        """
        B, L, H, dh = r.shape
        C = self.chunk_size
        s = r.new_zeros(B, H, dh, dh) if initial_state is None else initial_state

        # Pad to multiple of C
        pad = (C - L % C) % C
        if pad > 0:
            r = F.pad(r, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            w = F.pad(w, (0, 0, 0, 0, 0, pad))
            a = F.pad(a, (0, 0, 0, 0, 0, pad))
            b = F.pad(b, (0, 0, 0, 0, 0, pad))
        Lp = L + pad
        nc = Lp // C

        all_outputs = []
        for ci in range(nc):
            sl = slice(ci * C, (ci + 1) * C)
            r_c, k_c, v_c = r[:, sl], k[:, sl], v[:, sl]
            w_c, a_c, b_c = w[:, sl], a[:, sl], b[:, sl]

            chunk_out = []
            for t in range(C):
                decay = torch.exp(w_c[:, t])
                bS = torch.einsum('bhd,bhde->bhe', b_c[:, t], s)
                ab_S = torch.einsum('bhd,bhe->bhde', a_c[:, t], bS)
                s = s * decay.unsqueeze(-1) + ab_S + \
                    torch.einsum('bhd,bhe->bhde', k_c[:, t], v_c[:, t])
                chunk_out.append(torch.einsum('bhde,bhd->bhe', s, r_c[:, t]))
            all_outputs.append(torch.stack(chunk_out, dim=1))

        out = torch.cat(all_outputs, dim=1)
        if pad > 0:
            out = out[:, :L]
        return out, s

    def forward(self, x, last_x=None, state=None):
        """x: [B, L, D] → ([B, L, D], last_x_new, state_new)."""
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        x_prev = self._token_shift(x, last_x)

        r = self.W_r(x * self.mix_r + x_prev * (1 - self.mix_r)).view(B, L, H, dh)
        k = self.W_k(x * self.mix_k + x_prev * (1 - self.mix_k)).view(B, L, H, dh)
        v = self.W_v(x * self.mix_v + x_prev * (1 - self.mix_v)).view(B, L, H, dh)
        g = torch.sigmoid(self.W_g(x * self.mix_g + x_prev * (1 - self.mix_g)))

        # L2-normalize k per head (prevents state blow-up)
        k = F.normalize(k, p=2, dim=-1)

        # Non-diagonal state transition vectors
        x_a = x * self.mix_a + x_prev * (1 - self.mix_a)
        a = torch.sigmoid(self.W_a(x_a)).view(B, L, H, dh)
        b = self.W_b(x_a).view(B, L, H, dh)

        w = self._decay(self.base_decay, self.W_decay(x).view(B, L, H, dh))

        # Dispatch: FLA Triton kernels on CUDA, else PyTorch fallback
        if _HAS_FLA_V7 and x.is_cuda:
            fn = _fla_fused_recurrent_rwkv7 if L <= 64 else _fla_chunk_rwkv7
            out, s = fn(
                r=r, w=w, k=k, v=v,
                a=-a, b=a * b,   # FLA convention: a→ -kappa, b→ kappa*a
                scale=1.0,
                initial_state=state,
                output_final_state=True,
            )
        else:
            if not _HAS_FLA_V7 and x.is_cuda:
                import warnings
                warnings.warn(
                    "RWKV-7 on CUDA without FLA is slow. "
                    "Install: pip install flash-linear-attention",
                    stacklevel=2,
                )
            out, s = self._chunk_sequential_wkv(r, k, v, w, a, b, state)

        out = out.reshape(B, L, D)
        out = self.ln_out(out.transpose(1, 2)).transpose(1, 2)
        out = out * g
        return self.W_o(out), x[:, -1], s

    def step(self, x, last_x, state):
        """Single-token step.  x, last_x: [B, D], state: [B, H, dh, dh]."""
        B = x.shape[0]
        H, dh = self.n_heads, self.d_head

        r = self.W_r(x * self.mix_r + last_x * (1 - self.mix_r)).view(B, H, dh)
        k = self.W_k(x * self.mix_k + last_x * (1 - self.mix_k)).view(B, H, dh)
        v = self.W_v(x * self.mix_v + last_x * (1 - self.mix_v)).view(B, H, dh)
        g = torch.sigmoid(self.W_g(x * self.mix_g + last_x * (1 - self.mix_g)))

        k = F.normalize(k, p=2, dim=-1)

        x_a = x * self.mix_a + last_x * (1 - self.mix_a)
        a = torch.sigmoid(self.W_a(x_a)).view(B, H, dh)
        b = self.W_b(x_a).view(B, H, dh)

        w = self._decay(self.base_decay, self.W_decay(x).view(B, H, dh))
        decay = torch.exp(w)

        # S = diag(decay) * S + (a b^T) @ S + k v^T
        bS = torch.einsum('bhd,bhde->bhe', b, state)
        ab_S = torch.einsum('bhd,bhe->bhde', a, bS)
        state = state * decay.unsqueeze(-1) + ab_S + \
            torch.einsum('bhd,bhe->bhde', k, v)

        o = torch.einsum('bhde,bhd->bhe', state, r).reshape(B, self.d_model)
        o = self.ln_out(o.unsqueeze(-1)).squeeze(-1)
        o = o * g
        return self.W_o(o), x, state


class RWKV7Block(nn.Module):
    """RWKV-7 backbone block with pre-norm residual and SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int = 4, chunk_size: int = 32):
        super().__init__()
        self.ln1 = _RMSNorm(d_model)
        self.time_mix = _RWKV7TimeMix(d_model, n_heads, chunk_size=chunk_size)
        self.ln2 = _RMSNorm(d_model)
        self.ff = _SwiGLU(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        y, _, _ = self.time_mix(y)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        tm = self.time_mix
        last_x = torch.zeros(batch_size, tm.d_model, device=device)
        state = torch.zeros(batch_size, tm.n_heads, tm.d_head, tm.d_head,
                            device=device)
        return (last_x, state)

    def step(self, x, cache):
        last_x, state = cache
        y = self.ln1(x)
        y, last_x, state = self.time_mix.step(y, last_x, state)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, (last_x, state)


# ── RWKV-8 "Heron" — ROSA  (Peng, 2025 — experimental) ──────────
#
#   Replaces the linear-recurrence attention with ROSA (Rapid Online
#   Suffix Automaton): a neurosymbolic mechanism that finds the
#   longest matching suffix of quantised Q in the history of K,
#   returning the V symbol after the match.
#
#   ⚠  EXPERIMENTAL — the suffix-matching core is inherently
#      sequential (not parallelisable on GPU).  Bit-packing and
#      output scatter are vectorised with PyTorch tensor ops.
#      Streaming (step) recomputes on full context.
#
#   Reference:  github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8
# ──────────────────────────────────────────────────────────────────

def _rosa_match(q_syms, k_syms, v_syms, max_ctx=0):
    """Suffix matching (reference impl from official RWKV-8).

    For each position i, find the longest suffix of q[0:i+1] that
    appears in k[0:i], and return v[match_end].

    Args:
        max_ctx: if >0, only look back at most max_ctx positions.
    """
    n = len(q_syms)
    idx = [0] * n
    ln = [0] * n
    for i in range(n):
        found = False
        max_w = min(i + 1, max_ctx) if max_ctx > 0 else i + 1
        start_j = max(0, i - max_ctx) if max_ctx > 0 else 0
        for w in range(max_w, 0, -1):
            t = q_syms[i + 1 - w: i + 1]
            for j in range(i - w, start_j - 1, -1):
                if k_syms[j: j + w] == t:
                    s = j + w
                    if s < n:
                        idx[i] = v_syms[s]
                    ln[i] = w
                    found = True
                    break
            if found:
                break
    return idx, ln


def _rosa_match_batch(q_packed, k_packed, v_packed, max_ctx=0):
    """Run _rosa_match over all groups, returning (idx, ln) tensors."""
    B, T, G = q_packed.shape
    idx_out = torch.zeros(B, T, G, dtype=torch.long)
    ln_out = torch.zeros(B, T, G, dtype=torch.long)
    for b in range(B):
        for g in range(G):
            qs = q_packed[b, :, g].tolist()
            ks = k_packed[b, :, g].tolist()
            vs = v_packed[b, :, g].tolist()
            idx, ln = _rosa_match(qs, ks, vs, max_ctx=max_ctx)
            idx_out[b, :, g] = torch.tensor(idx)
            ln_out[b, :, g] = torch.tensor(ln)
    return idx_out, ln_out


class _ROSA(nn.Module):
    """ROSA attention with vectorised bit-packing (RWKV-8).

    Args:
        max_ctx: maximum lookback window for suffix matching.
                 0 = unlimited (slow for long sequences).
    """

    def __init__(self, d_model: int, bits: int = 4, max_ctx: int = 512):
        super().__init__()
        assert d_model % bits == 0
        self.bits = bits
        self.n_groups = d_model // bits
        self.max_ctx = max_ctx
        self.emb = nn.Parameter(torch.ones(1, 1, d_model))
        # Pre-compute bit-shift powers
        self.register_buffer('_powers', 1 << torch.arange(bits))

    def _pack_bits(self, x):
        """Quantise & pack: [B, T, D] → [B, T, G] int symbols."""
        B, T, _ = x.shape
        xb = (x > 0).long()                              # [B, T, D]
        xb = xb.view(B, T, self.n_groups, self.bits)     # [B, T, G, bits]
        return (xb * self._powers.to(x.device)).sum(-1)   # [B, T, G]

    def _unpack_to_sign(self, syms, matched):
        """Unpack symbols to signed embedding output.

        Args:
            syms:    [B, T, G] int symbols (matched v values)
            matched: [B, T, G] bool (whether a match was found)
        Returns: [B, T, D] float
        """
        B, T, G = syms.shape
        device = self.emb.device
        syms = syms.to(device)
        matched = matched.to(device)

        # Unpack each symbol into bits: [B, T, G, bits]
        bits_expanded = ((syms.unsqueeze(-1) >> self._powers.to(device)) & 1).float()
        signs = bits_expanded * 2 - 1   # 0 → -1, 1 → +1
        # [B, T, G, bits] → [B, T, D]
        signs = signs.view(B, T, -1)
        # Apply embedding magnitude
        out = signs * self.emb  # broadcast [1,1,D]
        # Zero out unmatched positions: matched [B,T,G] → [B,T,D]
        mask = matched.unsqueeze(-1).expand(B, T, G, self.bits).reshape(B, T, -1)
        return out * mask.float()

    def forward(self, q, k, v):
        """q, k, v: [B, T, D] → [B, T, D]."""
        device = q.device
        # Vectorised bit-packing (on GPU if available)
        q_packed = self._pack_bits(q).cpu()
        k_packed = self._pack_bits(k).cpu()
        v_packed = self._pack_bits(v).cpu()

        # Suffix matching (CPU — inherently sequential)
        idx, ln = _rosa_match_batch(q_packed, k_packed, v_packed,
                                    max_ctx=self.max_ctx)

        # Vectorised output construction (back on GPU)
        return self._unpack_to_sign(idx, ln > 0)


class _RWKV8ROSA(nn.Module):
    """RWKV-8 ROSA time-mixing layer."""

    def __init__(self, d_model: int, bits: int = 4, max_ctx: int = 512):
        super().__init__()
        self.d_model = d_model
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_k = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_v = nn.Parameter(torch.zeros(1, 1, d_model))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.rosa = _ROSA(d_model, bits=bits, max_ctx=max_ctx)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: [B, T, D] → [B, T, D]."""
        xx = self.time_shift(x) - x
        q = self.q_proj(x + xx * self.x_q)
        k = self.k_proj(x + xx * self.x_k)
        v = self.v_proj(x + xx * self.x_v)
        return self.o_proj(self.rosa(q, k, v))


class RWKV8Block(nn.Module):
    """RWKV-8 ROSA block (experimental).

    Args:
        max_ctx: suffix-matching lookback window (default 512).
                 Limits ROSA to O(max_ctx²) per token instead of O(T²).
    """

    def __init__(self, d_model: int, bits: int = 4, max_ctx: int = 512):
        super().__init__()
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.rosa_mix = _RWKV8ROSA(d_model, bits=bits, max_ctx=max_ctx)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _SwiGLU(d_model)

    def forward(self, x, inference_params=None):
        x = x + self.rosa_mix(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def init_cache(self, batch_size: int, device):
        # Cache stores the full hidden-state history for recomputation
        return {'history': torch.zeros(batch_size, 0, self.d_model, device=device)}

    def step(self, x, cache):
        """Single-token step — appends to history and recomputes.

        ⚠  O(n²) per step where n is context length so far.
        """
        history = cache['history']
        # x: [B, D] → [B, 1, D]
        x_seq = torch.cat([history, x.unsqueeze(1)], dim=1)
        # Full forward on accumulated context
        out = self.forward(x_seq)
        # Return only the last token's output
        return out[:, -1], {'history': x_seq}


# ── Griffin  (De et al., 2024 — "Griffin / RecurrentGemma") ──────
#
#   Hybrid architecture: Real-Gated Linear Recurrent Unit (RG-LRU)
#   for long-range memory + local sliding-window causal attention
#   for fine-grained byte patterns.  Each block has three residual
#   sub-layers:
#     1. RG-LRU  (linear recurrence, O(1) per step)
#     2. Local causal attention (small window, bounded KV cache)
#     3. SwiGLU FFN
#
#   References
#     De et al., "Griffin: Mixing Gated Linear Recurrences with
#         Local Attention for Efficient Language Models", 2024.
# ──────────────────────────────────────────────────────────────────

class _RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit — diagonal SSM with input-
    dependent gating.

    Recurrence:
        λ_t = exp(−softplus(ν) · σ(a(x_t)))       per-dim decay
        h_t = λ_t ⊙ h_{t-1} + (1 − λ_t) ⊙ gate(x_t) ⊙ proj(x_t)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.recurrence_gate = nn.Linear(d_model, d_model)
        self.log_base_decay = nn.Parameter(torch.ones(d_model) * -3.0)

    def _decay(self, x):
        a = torch.sigmoid(self.recurrence_gate(x))
        return torch.exp(-F.softplus(self.log_base_decay) * a)

    def forward(self, x):
        """x: [B, L, D] → [B, L, D]."""
        B, L, D = x.shape
        inp = self.input_proj(x) * torch.sigmoid(self.gate_proj(x))
        lam = self._decay(x)          # [B, L, D]
        h = x.new_zeros(B, D)
        outputs = []
        for t in range(L):
            h = lam[:, t] * h + (1 - lam[:, t]) * inp[:, t]
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def step(self, x, h):
        """x, h: [B, D] → ([B, D], h_new)."""
        inp = self.input_proj(x) * torch.sigmoid(self.gate_proj(x))
        lam = self._decay(x.unsqueeze(1)).squeeze(1)
        h = lam * h + (1 - lam) * inp
        return h, h


class GriffinBlock(nn.Module):
    """Griffin-style hybrid: RG-LRU + local attention + SwiGLU FFN.

    Parameters
    ----------
    d_model      : int  – hidden dimension.
    n_heads      : int  – number of attention heads for local attention.
    local_window : int  – sliding-window width for local attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, local_window: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.local_window = local_window

        # Sub-layer 1: RG-LRU
        self.ln1 = _RMSNorm(d_model)
        self.rg_lru = _RGLRU(d_model)

        # Sub-layer 2: local causal attention
        self.ln2 = _RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Sub-layer 3: FFN
        self.ln3 = _RMSNorm(d_model)
        self.ff = _SwiGLU(d_model)

    def _local_attention(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        if L <= self.local_window:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = _make_sliding_window_mask(L, self.local_window, 0, x.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(y.transpose(1, 2).reshape(B, L, D))

    def forward(self, x, inference_params=None):
        # Recurrence
        y = self.ln1(x)
        y = self.rg_lru(y)
        x = x + y
        # Local attention
        y = self.ln2(x)
        y = self._local_attention(y)
        x = x + y
        # FFN
        y = self.ln3(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        h_rec = torch.zeros(batch_size, self.d_model, device=device)
        k_cache = torch.zeros(batch_size, self.n_heads, 0, self.d_head,
                              device=device)
        v_cache = torch.zeros(batch_size, self.n_heads, 0, self.d_head,
                              device=device)
        return (h_rec, k_cache, v_cache)

    def step(self, x, cache):
        h_rec, k_cache, v_cache = cache
        B = x.shape[0]

        # Recurrence step
        y = self.ln1(x)
        y, h_rec = self.rg_lru.step(y, h_rec)
        x = x + y

        # Local attention step (rolling KV cache)
        y = self.ln2(x).unsqueeze(1)                                 # [B,1,D]
        q = self.q_proj(y).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k_new = self.k_proj(y).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        v_new = self.v_proj(y).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        k_cache = torch.cat([k_cache, k_new], dim=2)
        v_cache = torch.cat([v_cache, v_new], dim=2)
        if k_cache.shape[2] > self.local_window:
            k_cache = k_cache[:, :, -self.local_window:]
            v_cache = v_cache[:, :, -self.local_window:]

        scale = self.d_head ** -0.5
        attn = (q @ k_cache.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v_cache).transpose(1, 2).reshape(B, 1, self.d_model)
        y = self.out_proj(y).squeeze(1)
        x = x + y

        # FFN
        y = self.ln3(x)
        y = self.ff(y)
        return x + y, (h_rec, k_cache, v_cache)


# ── xLSTM  (Beck et al., 2024 — "xLSTM: Extended LSTM") ─────────
#
#   mLSTM variant with:
#     • Exponential gating (input gate can exceed 1 for amplification)
#     • Matrix-valued cell state  C ∈ R^{d_head × d_head}
#     • Covariance normaliser for numerical stability
#     • Log-space stabilisation of gates (max-trick)
#
#   Recurrence (per head, in log-space-stabilised form):
#     m_t = max(log_f_t + m_{t-1},  log_i_t)
#     f'  = exp(log_f + m_{t-1} − m_t)
#     i'  = exp(log_i − m_t)
#     C_t = f' · C_{t-1}  +  i' · (v_t ⊗ k_t)
#     n_t = f' · n_{t-1}  +  i' · k_t
#     h_t = C_t @ q_t  /  max(|n_t · q_t|, 1)
#
#   References
#     Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.
# ──────────────────────────────────────────────────────────────────

class _mLSTMCell(nn.Module):
    """mLSTM cell — matrix memory with exponential gating."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Scalar gates per head
        self.W_i = nn.Linear(d_model, n_heads)   # input gate  (exp)
        self.W_f = nn.Linear(d_model, n_heads)   # forget gate (sigmoid)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(n_heads, d_model)

    def forward(self, x, state=None):
        """x: [B, L, D] → ([B, L, D], new_state).

        state = (C, n, m)  with shapes
            C: [B, H, dh, dh]   matrix cell
            n: [B, H, dh]       normaliser
            m: [B, H]           log-space stabiliser
        """
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.W_q(x).view(B, L, H, dh)
        k = self.W_k(x).view(B, L, H, dh) * (dh ** -0.5)
        v = self.W_v(x).view(B, L, H, dh)

        log_f = -F.softplus(-self.W_f(x))         # log(σ(·)), always ≤ 0
        log_i = self.W_i(x)                        # can be > 0

        if state is None:
            C = x.new_zeros(B, H, dh, dh)
            n = x.new_zeros(B, H, dh)
            m = x.new_zeros(B, H)
        else:
            C, n, m = state

        outputs = []
        for t in range(L):
            lf = log_f[:, t]                       # [B, H]
            li = log_i[:, t]
            m_new = torch.max(lf + m, li)
            f_prime = torch.exp(lf + m - m_new)    # [B, H]
            i_prime = torch.exp(li - m_new)
            m = m_new

            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]

            C = f_prime.unsqueeze(-1).unsqueeze(-1) * C + \
                i_prime.unsqueeze(-1).unsqueeze(-1) * \
                torch.einsum('bhd,bhe->bhde', k_t, v_t)
            n = f_prime.unsqueeze(-1) * n + i_prime.unsqueeze(-1) * k_t

            h_t = torch.einsum('bhde,bhd->bhe', C, q_t)
            denom = torch.einsum('bhd,bhd->bh', n, q_t) \
                         .unsqueeze(-1).abs().clamp(min=1.0)
            outputs.append(h_t / denom)

        out = torch.stack(outputs, dim=1).reshape(B, L, D)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(out), (C, n, m)

    def step(self, x, state):
        """Single-token step.  x: [B, D], state: (C, n, m)."""
        B = x.shape[0]
        H, dh = self.n_heads, self.d_head
        C, n, m = state

        q = self.W_q(x).view(B, H, dh)
        k = self.W_k(x).view(B, H, dh) * (dh ** -0.5)
        v = self.W_v(x).view(B, H, dh)

        lf = -F.softplus(-self.W_f(x)).squeeze(-1) if self.n_heads == 1 \
             else -F.softplus(-self.W_f(x))         # [B, H]
        li = self.W_i(x) if self.n_heads > 1 else self.W_i(x)

        m_new = torch.max(lf + m, li)
        f_prime = torch.exp(lf + m - m_new)
        i_prime = torch.exp(li - m_new)
        m = m_new

        C = f_prime.unsqueeze(-1).unsqueeze(-1) * C + \
            i_prime.unsqueeze(-1).unsqueeze(-1) * \
            torch.einsum('bhd,bhe->bhde', k, v)
        n = f_prime.unsqueeze(-1) * n + i_prime.unsqueeze(-1) * k

        h = torch.einsum('bhde,bhd->bhe', C, q)
        denom = torch.einsum('bhd,bhd->bh', n, q) \
                     .unsqueeze(-1).abs().clamp(min=1.0)
        h = (h / denom).reshape(B, self.d_model)
        h = self.group_norm(h.unsqueeze(-1)).squeeze(-1)
        return self.out_proj(h), (C, n, m)


class xLSTMBlock(nn.Module):
    """xLSTM (mLSTM) backbone block with pre-norm residual and SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.ln1 = _RMSNorm(d_model)
        self.mlstm = _mLSTMCell(d_model, n_heads)
        self.ln2 = _RMSNorm(d_model)
        self.ff = _SwiGLU(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        y, _ = self.mlstm(y)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        H = self.mlstm.n_heads
        dh = self.mlstm.d_head
        C = torch.zeros(batch_size, H, dh, dh, device=device)
        n = torch.zeros(batch_size, H, dh, device=device)
        m = torch.zeros(batch_size, H, device=device)
        return (C, n, m)

    def step(self, x, cache):
        y = self.ln1(x)
        y, cache = self.mlstm.step(y, cache)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        return x + y, cache


# ======================================================================
#  Model factory
# ======================================================================

def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cuda",
                   backbone="mamba", **backbone_kwargs):
    """Construct a BoaBytePredictor with the specified backbone.

    Parameters
    ----------
    backbone : str
        One of: mamba, mambav1, mamba2, lstm, gru, mingru, transformer,
        rwkv6, griffin, xlstm  (default: mamba).
    **backbone_kwargs
        Extra keyword arguments forwarded to the backbone block constructor.
        For ``transformer``: n_heads, n_kv_heads, window_size, n_sink.
        For ``rwkv6``: n_heads.
        For ``griffin``: n_heads, local_window.
        For ``xlstm``: n_heads.
    """
    backbone = backbone.lower()
    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from {AVAILABLE_BACKBONES}"
        )

    # ── Non-Mamba backbones ───────────────────────────────────────
    if backbone not in ("mamba", "mambav1", "mamba2"):
        _BLOCK_MAP = {
            "lstm":        LSTMBlock,
            "gru":         GRUBlock,
            "mingru":      MinGRUBlock,
            "transformer": TransformerBlock,
            "rwkv6":       RWKV6Block,
            "rwkv7":       RWKV7Block,
            "rwkv8":       RWKV8Block,
            "griffin":      GriffinBlock,
            "xlstm":       xLSTMBlock,
        }
        BlockCls = _BLOCK_MAP[backbone]

        # Only forward kwargs the block actually accepts
        _valid = (
            set(_inspect.signature(BlockCls.__init__).parameters) - {"self", "d_model"}
        )
        _bk = {k: v for k, v in backbone_kwargs.items() if k in _valid}

        class BoaBytePredictor(nn.Module):
            """Byte predictor with a non-Mamba backbone."""

            def __init__(self, d_model, num_layers, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.blocks = nn.ModuleList(
                    [BlockCls(d_model, **_bk) for _ in range(num_layers)]
                )
                self.final_norm = _RMSNorm(d_model)
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, vocab_size),
                )
                self._backbone_name = backbone

            def forward(self, x, inference_params=None):
                h = self.embedding(x)                      # [B, L, D]
                for blk in self.blocks:
                    h = blk(h, inference_params=inference_params)
                h = self.final_norm(h)
                return self.head(h)                        # [B, L, V]

            @torch.inference_mode()
            def init_stream(self, max_len: int, batch_size: int = 1,
                            device=None, dtype=None):
                return [blk.init_cache(batch_size, device)
                        for blk in self.blocks]

            @torch.inference_mode()
            def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
                h = self.embedding(byte_t)                 # [B, D]
                for i, blk in enumerate(self.blocks):
                    h, caches[i] = blk.step(h, caches[i])
                h = self.final_norm(h)
                return self.head(h)                        # [B, V]

        return BoaBytePredictor(
            d_model=d_model, num_layers=num_layers, vocab_size=vocab_size,
        )

    # ── Mamba backbones ──────────────────────────────────────────
    IS_CUDA = torch.cuda.is_available() and device == "cuda"

    if IS_CUDA:
        device = "cuda"
        from mamba_ssm import Mamba
        from mamba_ssm.utils.generation import InferenceParams
    else:
        device = "cpu"
        from mambapy.mamba import MambaBlock as MambaCPU, MambaConfig

    def tag_mamba_layers_with_ids(model):
        """Give each Mamba layer a unique .layer_idx (0..N-1) for streaming cache."""
        i = 0
        for m in model.modules():
            if IS_CUDA:
                if isinstance(m, Mamba):
                    setattr(m, "layer_idx", i)
                    i += 1
            else:
                if isinstance(m, MambaCPU):
                    setattr(m, "layer_idx", i)
                    i += 1

    def bump_offset(inf, k: int = 1):
        # Most builds use seqlen_offset
        if hasattr(inf, "seqlen_offset"):
            inf.seqlen_offset += k
        elif hasattr(inf, "sequence_length_offset"):
            setattr(inf, "sequence_length_offset", getattr(inf, "sequence_length_offset") + k)
        else:
            # set a best-effort attribute for obscure builds
            setattr(inf, "seqlen_offset", getattr(inf, "seqlen_offset", 0) + k)

    # ── mambav1: original architecture from V1.1.0 ──────────────
    if backbone == "mambav1":
        class MambaBlockV1(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                if IS_CUDA:
                    self.mamba = Mamba(d_model=d_model)
                else:
                    config = MambaConfig(d_model=d_model, n_layers=0, use_cuda=False)
                    self.mamba = MambaCPU(config)
                self.ln2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                )
            def forward(self, x, inference_params=None):
                y = self.ln1(x)
                if IS_CUDA:
                    y = self.mamba(y, inference_params=inference_params)
                else:
                    y = self.mamba(y)
                y = self.ln2(y)
                y = self.ff(y)
                return x + y

            if not IS_CUDA:
                def init_cache(self, batch_size: int, device):
                    d_inner = self.mamba.config.d_inner
                    d_conv = self.mamba.config.d_conv
                    inputs = torch.zeros(batch_size, d_inner, d_conv - 1, device=device)
                    return (None, inputs)

                def step(self, x, cache):
                    y = self.ln1(x)
                    y, cache = self.mamba.step(y, cache)
                    y = self.ln2(y)
                    y = self.ff(y)
                    return x + y, cache

        class BoaBytePredictorV1(nn.Module):
            """ Original Mamba byte predictor (V1.1.0). """
            def __init__(self, d_model=256, num_layers=4, vocab_size=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.blocks = nn.ModuleList([MambaBlockV1(d_model) for _ in range(num_layers)])
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, vocab_size)
                )
                self._backbone_name = "mambav1"

            def forward(self, x, inference_params=None):
                h = self.embedding(x)
                for blk in self.blocks:
                    h = blk(h, inference_params=inference_params)
                return self.head(h)

            if IS_CUDA:
                @torch.inference_mode()
                def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
                    return InferenceParams(max_batch_size=batch_size, max_seqlen=max_len)

                @torch.inference_mode()
                def step(self, byte_t: torch.LongTensor, inf) -> torch.Tensor:
                    x = self.embedding(byte_t).unsqueeze(1)
                    h = x
                    for blk in self.blocks:
                        h = blk(h, inference_params=inf)
                    logits_next = self.head(h).squeeze(1)
                    bump_offset(inf, 1)
                    return logits_next
            else:
                @torch.inference_mode()
                def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
                    return [blk.init_cache(batch_size, device) for blk in self.blocks]

                @torch.inference_mode()
                def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
                    h = self.embedding(byte_t)
                    for i, blk in enumerate(self.blocks):
                        h, caches[i] = blk.step(h, caches[i])
                    logits_next = self.head(h)
                    return logits_next

        model = BoaBytePredictorV1(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
        tag_mamba_layers_with_ids(model)
        return model

    # ── mamba2: Structured State Space Duality (Dao & Gu, 2024) ──
    if backbone == "mamba2":
        if not IS_CUDA:
            raise RuntimeError(
                "Mamba2 requires CUDA (mamba_ssm.modules.mamba2). "
                "Use backbone='mamba' for CPU fallback."
            )
        from mamba_ssm.modules.mamba2 import Mamba2

        class Mamba2Block(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.ln1 = _RMSNorm(d_model)
                self.mamba2 = Mamba2(d_model=d_model)
                self.ln2 = _RMSNorm(d_model)
                self.ff = _SwiGLU(d_model)

            def forward(self, x, inference_params=None):
                y = self.ln1(x)
                y = self.mamba2(y, inference_params=inference_params)
                x = x + y
                y = self.ln2(x)
                y = self.ff(y)
                return x + y

        class BoaBytePredictorM2(nn.Module):
            """Mamba-2 byte predictor (SSD kernel, CUDA only)."""
            def __init__(self, d_model=256, num_layers=4, vocab_size=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.blocks = nn.ModuleList(
                    [Mamba2Block(d_model) for _ in range(num_layers)]
                )
                self.final_norm = _RMSNorm(d_model)
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, vocab_size),
                )
                self._backbone_name = "mamba2"

            def forward(self, x, inference_params=None):
                h = self.embedding(x)
                for blk in self.blocks:
                    h = blk(h, inference_params=inference_params)
                h = self.final_norm(h)
                return self.head(h)

            @torch.inference_mode()
            def init_stream(self, max_len: int, batch_size: int = 1,
                            device=None, dtype=None):
                return InferenceParams(max_batch_size=batch_size,
                                       max_seqlen=max_len)

            @torch.inference_mode()
            def step(self, byte_t: torch.LongTensor, inf) -> torch.Tensor:
                h = self.embedding(byte_t).unsqueeze(1)
                for blk in self.blocks:
                    h = blk(h, inference_params=inf)
                h = self.final_norm(h).squeeze(1)
                logits_next = self.head(h)
                bump_offset(inf, 1)
                return logits_next

        model = BoaBytePredictorM2(d_model=d_model, num_layers=num_layers,
                                    vocab_size=vocab_size)
        tag_mamba_layers_with_ids(model)
        return model

    # ── mamba (improved): dual residual, RMSNorm, SwiGLU ────────
    class MambaBlock(nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.ln1 = _RMSNorm(d_model)
            if IS_CUDA:
                self.mamba = Mamba(d_model=d_model)
            else:
                config = MambaConfig(d_model=d_model, n_layers=0, use_cuda=False)
                self.mamba = MambaCPU(config)
            self.ln2 = _RMSNorm(d_model)
            self.ff = _SwiGLU(d_model)

        def forward(self, x, inference_params=None):
            # Two independent pre-norm residual streams — much better
            # gradient flow than the old single-residual path.
            y = self.ln1(x)
            if IS_CUDA:
                y = self.mamba(y, inference_params=inference_params)
            else:
                y = self.mamba(y)
            x = x + y                 # residual 1: Mamba
            y = self.ln2(x)
            y = self.ff(y)
            return x + y              # residual 2: FFN
        
        if not IS_CUDA:
            def init_cache(self, batch_size: int, device):
                # cache for mambapy.MambaBlock.step: (h, inputs)
                d_inner = self.mamba.config.d_inner
                d_conv = self.mamba.config.d_conv
                inputs = torch.zeros(batch_size, d_inner, d_conv - 1, device=device)
                return (None, inputs)

            def step(self, x, cache):
                # x: [B, D] -> [B, D], cache passthrough
                y = self.ln1(x)
                y, cache = self.mamba.step(y, cache)
                x = x + y             # residual 1
                y = self.ln2(x)
                y = self.ff(y)
                return x + y, cache   # residual 2
        
    class BoaBytePredictor(nn.Module):
        """ Mamba byte predictor.

        Improvements over the original:
        • Dual pre-norm residual streams (Mamba + FFN each have their own)
        • RMSNorm instead of LayerNorm (faster, works better with gating)
        • SwiGLU FFN instead of GELU (better param efficiency)
        • Final RMSNorm before output head (stabilises training)
        """
        def __init__(self, d_model=256, num_layers=4, vocab_size=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(num_layers)])
            self.final_norm = _RMSNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, vocab_size),
            )
            self._backbone_name = "mamba"

        def forward(self, x, inference_params=None):
            h = self.embedding(x)                  # [B, L, D]
            for blk in self.blocks:
                h = blk(h, inference_params=inference_params)
            h = self.final_norm(h)
            return self.head(h)                    # [B, L, vocab_size]
        
        if IS_CUDA:
            @torch.inference_mode()
            def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
                return InferenceParams(max_batch_size=batch_size, max_seqlen=max_len)

            @torch.inference_mode()
            def step(self, byte_t: torch.LongTensor, inf) -> torch.Tensor:
                h = self.embedding(byte_t).unsqueeze(1)   # [B, 1, D]
                for blk in self.blocks:
                    h = blk(h, inference_params=inf)
                h = self.final_norm(h).squeeze(1)         # [B, D]
                logits_next = self.head(h)                 # [B, vocab_size]
                bump_offset(inf, 1)
                return logits_next
        else:
            @torch.inference_mode()
            def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
                return [blk.init_cache(batch_size, device) for blk in self.blocks]

            @torch.inference_mode()
            def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
                h = self.embedding(byte_t)                # [B, D]
                for i, blk in enumerate(self.blocks):
                    h, caches[i] = blk.step(h, caches[i])
                h = self.final_norm(h)
                return self.head(h)                       # [B, vocab_size]

    model = BoaBytePredictor(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
    tag_mamba_layers_with_ids(model)
    return model

def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    # number of usable bytes that fit whole (batch_size * seq_len) chunks
    block = seq_len * batch_size
    return (n_bytes // block) * block

def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    train_bytes = buf[i0:i1].tobytes()
    val_bytes   = buf[i1:i2].tobytes()
    test_bytes  = buf[i2:i2+n_test].tobytes()

    return train_bytes, val_bytes, test_bytes

class ByteDataloader:
    """ Simple dataloader that yields batches of bytes. """
    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0
        self.device = device
        # Pre-allocate a pinned CPU tensor and a GPU tensor to avoid
        # repeated allocation + async H2D copies each step.
        self._block = self.seq_len * self.batch_size
        if device == "cuda" and torch.cuda.is_available():
            self._cpu_buf = torch.empty(self.batch_size, self.seq_len,
                                        dtype=torch.long, pin_memory=True)
            self._gpu_buf = torch.empty(self.batch_size, self.seq_len,
                                        dtype=torch.long, device=device)
        else:
            self._cpu_buf = None
            self._gpu_buf = None
    def __len__(self):
        """ Returns the total number of batches in the dataset. """
        return len(self.data_bytes) // (self.seq_len * self.batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos + self._block > len(self.data_bytes):
            self.pos = 0  # reset for simplicity
            raise StopIteration
        
        chunk = self.data_bytes[self.pos : self.pos + self._block]
        self.pos += self._block

        if self._cpu_buf is not None:
            # Fast path: copy into pinned buffer, async transfer to GPU
            np.copyto(self._cpu_buf.numpy().ravel(), chunk)
            self._gpu_buf.copy_(self._cpu_buf, non_blocking=True)
            return self._gpu_buf
        else:
            batch = chunk.reshape(self.batch_size, self.seq_len)
            return torch.tensor(batch, dtype=torch.long).to(self.device)
    
