"""
HydraBOA — Multi-head chunked Mamba model for fast byte-level compression.

Predicts K bytes per backbone step (default K=4), cutting Mamba compute by K×.

Architecture
────────────
1. **Chunked Backbone**: groups K bytes into a single embedding vector and
   passes the down-sampled sequence through Mamba.  Sequence length ≡ L/K.
2. **Chained Heads**: K ultra-lightweight prediction heads.  Head *k*
   receives the backbone context H_t concatenated with embeddings of the
   *k* preceding bytes within the current chunk (teacher-forced at train
   time).

This file is completely self-contained and does NOT modify any existing
boa-constrictor source files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Mamba imports (CUDA or CPU fallback) ─────────────────────────
_HAS_CUDA_MAMBA = False
_HAS_CPU_MAMBA = False

if torch.cuda.is_available():
    try:
        from mamba_ssm import Mamba as _CudaMamba
        from mamba_ssm.utils.generation import InferenceParams
        _HAS_CUDA_MAMBA = True
    except ImportError:
        pass

if not _HAS_CUDA_MAMBA:
    try:
        from mambapy.mamba import MambaBlock as _CpuMamba, MambaConfig as _CpuMambaConfig
        _HAS_CPU_MAMBA = True
    except ImportError:
        pass

IS_CUDA = torch.cuda.is_available() and _HAS_CUDA_MAMBA


# ── Utility modules ──────────────────────────────────────────────

class _RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (Zhang & Sennrich 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020).  ≈ 2.67× d_model hidden."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = ((8 * d_model // 3 + 7) // 8) * 8
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ── Mamba block ──────────────────────────────────────────────────

class _HydraMambaBlock(nn.Module):
    """Original Mamba-v1 block: LN → Mamba → LN → FFN(4×, GELU) + residual.

    Matches the ``MambaBlockV1`` in model.py (backbone="mambav1").
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if IS_CUDA:
            self.mamba = _CudaMamba(d_model=d_model)
        else:
            cfg = _CpuMambaConfig(d_model=d_model, n_layers=0, use_cuda=False)
            self.mamba = _CpuMamba(cfg)
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

    # CPU-only streaming helpers (mambapy path)
    if not IS_CUDA and _HAS_CPU_MAMBA:
        def init_cache(self, batch_size: int, device):
            d_inner = self.mamba.config.d_inner
            d_conv  = self.mamba.config.d_conv
            inputs  = torch.zeros(batch_size, d_inner, d_conv - 1, device=device)
            return (None, inputs)

        def step(self, x, cache):
            y = self.ln1(x)
            y, cache = self.mamba.step(y, cache)
            y = self.ln2(y)
            y = self.ff(y)
            return x + y, cache


# ── Helpers ──────────────────────────────────────────────────────

def _bump_offset(inf, k: int = 1):
    if hasattr(inf, "seqlen_offset"):
        inf.seqlen_offset += k
    elif hasattr(inf, "sequence_length_offset"):
        inf.sequence_length_offset += k
    else:
        setattr(inf, "seqlen_offset", getattr(inf, "seqlen_offset", 0) + k)


def _tag_mamba_layers(model):
    """Assign unique ``layer_idx`` to each Mamba submodule (needed for cache)."""
    i = 0
    for m in model.modules():
        if IS_CUDA and _HAS_CUDA_MAMBA:
            if isinstance(m, _CudaMamba):
                m.layer_idx = i
                i += 1
        elif _HAS_CPU_MAMBA:
            if isinstance(m, _CpuMamba):
                m.layer_idx = i
                i += 1


# ═══════════════════════════════════════════════════════════════════
#  HydraBOA
# ═══════════════════════════════════════════════════════════════════

class HydraBOA(nn.Module):
    r"""Multi-head chunked Mamba byte predictor.

    Parameters
    ----------
    d_model : int
        Hidden dimension (default 256).
    num_layers : int
        Number of Mamba blocks in the backbone (default 1).
    vocab_size : int
        Byte alphabet size (default 256).
    K : int
        Chunk size — number of bytes predicted per backbone step (default 4).

    Training
    --------
    ``forward(x)`` accepts ``x`` of shape ``[B, L]`` with ``L`` divisible
    by *K* and returns logits ``[B, L, vocab_size]``.  The logit at
    position *i* predicts the byte at position *i* (not *i+1*).
    Causality is guaranteed by the backbone shift (BOS token) and the
    chained-head structure.

    Loss::

        loss = F.cross_entropy(logits.view(-1, V), x.view(-1))

    Streaming
    ---------
    For compression / decompression::

        cache = model.init_stream(max_chunks=T, batch_size=B, device=dev)

        # For each K-byte chunk:
        H_t = model.step_backbone(prev_chunk_bytes, cache)
        for k in range(K):
            logits_k = model.predict_head(k, H_t, prev_bytes_in_chunk)
            probs_k  = softmax(logits_k)
            # feed probs_k to the range coder
    """

    def __init__(self, d_model: int = 256, num_layers: int = 1,
                 vocab_size: int = 256, K: int = 4):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._backbone_name = "hydra_mambav1"

        # ── shared byte embedding ──
        self.byte_embed = nn.Embedding(vocab_size, d_model)

        # ── chunk embedding: concat K byte-embeddings → d_model ──
        self.chunk_proj = nn.Linear(K * d_model, d_model)

        # ── learned BOS embedding (bootstraps the first chunk) ──
        self.bos_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # ── Mamba backbone ──
        self.blocks = nn.ModuleList(
            [_HydraMambaBlock(d_model) for _ in range(num_layers)]
        )
        self.backbone_norm = nn.LayerNorm(d_model)

        # ── K chained prediction heads ──
        # Head k:  (1 + k) · d_model  →  d_model  →  vocab_size
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear((1 + k) * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size),
            )
            for k in range(K)
        ])

        _tag_mamba_layers(self)

    # ── internal ─────────────────────────────────────────────────

    def _make_chunk_embeds(self, chunks: torch.LongTensor) -> torch.Tensor:
        """chunks: [B, T, K] → [B, T, d_model]"""
        emb = self.byte_embed(chunks)                       # [B, T, K, D]
        B, T, K, D = emb.shape
        return self.chunk_proj(emb.reshape(B, T, K * D))    # [B, T, D]

    # ── training forward ─────────────────────────────────────────

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x : [B, L]  long tensor of byte values (L divisible by K)
        Returns logits [B, L, vocab_size].
        """
        B, L = x.shape
        K, D = self.K, self.d_model
        assert L % K == 0, f"L={L} must be divisible by K={K}"
        T = L // K

        chunks = x.reshape(B, T, K)                         # [B, T, K]

        # ── backbone ──
        chunk_embs = self._make_chunk_embeds(chunks)         # [B, T, D]
        bos = self.bos_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, D)
        # Shifted: [BOS, c_0, c_1, …, c_{T-2}]  so that output t only
        # sees chunks < t.
        backbone_in = torch.cat([bos, chunk_embs[:, :-1, :]], dim=1)  # [B, T, D]

        h = backbone_in
        for blk in self.blocks:
            h = blk(h)                                       # full parallel scan
        H = self.backbone_norm(h)                            # [B, T, D]

        # ── chained heads (teacher-forced) ──
        byte_embs = self.byte_embed(chunks)                  # [B, T, K, D]

        logits_list = []
        for k in range(K):
            if k == 0:
                inp = H                                      # [B, T, D]
            else:
                prev = byte_embs[:, :, :k, :].reshape(B, T, k * D)
                inp = torch.cat([H, prev], dim=-1)           # [B, T, (1+k)·D]
            logits_list.append(self.heads[k](inp))           # [B, T, V]

        # Interleave heads → [B, T, K, V] → [B, L, V]
        logits = torch.stack(logits_list, dim=2).reshape(B, L, self.vocab_size)
        return logits

    # ── streaming interface (for codec) ──────────────────────────

    @torch.inference_mode()
    def init_stream(self, max_chunks: int, batch_size: int = 1,
                    device=None, dtype=None):
        """Allocate streaming state for up to *max_chunks* backbone steps."""
        if IS_CUDA:
            return InferenceParams(max_batch_size=batch_size,
                                   max_seqlen=max_chunks)
        else:
            dev = device or "cpu"
            return [blk.init_cache(batch_size, dev) for blk in self.blocks]

    @torch.inference_mode()
    def step_backbone(self, prev_chunk_bytes, cache,
                      batch_size: int = 1) -> torch.Tensor:
        """Feed one chunk embedding through the backbone → H_t.

        Parameters
        ----------
        prev_chunk_bytes : [B, K] long tensor   (previous chunk)
                           *None* for the very first chunk (BOS used).
        cache : streaming state from ``init_stream``.
        batch_size : used only when prev_chunk_bytes is None.

        Returns
        -------
        H_t : Tensor [B, d_model]
        """
        dev = next(self.parameters()).device

        if prev_chunk_bytes is None:
            x = self.bos_embed.unsqueeze(0).expand(batch_size, -1)   # [B, D]
        else:
            emb = self.byte_embed(prev_chunk_bytes)                  # [B, K, D]
            B, K, D = emb.shape
            x = self.chunk_proj(emb.reshape(B, K * D))              # [B, D]

        if IS_CUDA:
            h = x.unsqueeze(1)                                       # [B, 1, D]
            for blk in self.blocks:
                h = blk(h, inference_params=cache)
            H = self.backbone_norm(h.squeeze(1))                     # [B, D]
            _bump_offset(cache, 1)
        else:
            h = x                                                    # [B, D]
            for i, blk in enumerate(self.blocks):
                h, cache[i] = blk.step(h, cache[i])
            H = self.backbone_norm(h)                                # [B, D]

        return H

    @torch.inference_mode()
    def predict_head(self, k: int, H_t: torch.Tensor,
                     prev_bytes: torch.LongTensor = None) -> torch.Tensor:
        """Run head *k* and return logits [B, vocab_size].

        Parameters
        ----------
        k : int  (0 … K-1)
        H_t : [B, d_model]
        prev_bytes : [B, k] long tensor of bytes already decoded in this
                     chunk.  *None* (or empty) for k=0.
        """
        if k == 0:
            inp = H_t
        else:
            assert prev_bytes is not None and prev_bytes.shape[-1] == k
            emb = self.byte_embed(prev_bytes)                        # [B, k, D]
            B = H_t.shape[0]
            prev_flat = emb.reshape(B, k * self.d_model)
            inp = torch.cat([H_t, prev_flat], dim=-1)               # [B, (1+k)·D]
        return self.heads[k](inp)                                    # [B, V]
