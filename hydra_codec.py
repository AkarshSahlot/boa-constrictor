"""
HydraBOA codec — compression & decompression using the multi-head chunked model.

Provides three backends:
  1. CPU constriction range coder  (compress_hydra / decompress_hydra)
  2. GPU batched range coder       (compress_hydra_gpu / decompress_hydra_gpu)

This file is self-contained and does NOT modify any existing boa-constrictor
source files.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

# Check for the GPU range coder
_HAS_GPU_RC = False
try:
    import gpu_range_coder as _gr
    _HAS_GPU_RC = True
except ImportError:
    pass


def compress_hydra(
    model,
    data_bytes: bytes,
    K: int = 4,
    device: str = "cpu",
    progress: bool = True,
) -> tuple:
    """Compress *data_bytes* with HydraBOA.

    Returns
    -------
    compressed : np.ndarray[uint32]   — range-coded bitstream
    metadata   : dict                 — needed for decompression
    """
    import constriction

    model.eval().to(device)
    data = np.frombuffer(data_bytes, dtype=np.uint8).copy()
    orig_len = len(data)

    # Pad to a multiple of K
    remainder = len(data) % K
    if remainder:
        pad = K - remainder
        data = np.concatenate([data, np.zeros(pad, dtype=np.uint8)])
    else:
        pad = 0

    L = len(data)
    T = L // K

    enc = constriction.stream.queue.RangeEncoder()
    fam = constriction.stream.model.Categorical(perfect=False)

    cache = model.init_stream(max_chunks=T, batch_size=1, device=device)
    prev_chunk: torch.Tensor | None = None

    pbar = tqdm(total=T, disable=not progress, desc="Compress (HydraBOA)",
                unit="chunk", mininterval=0.3)

    for t in range(T):
        # ── heavy step: backbone ──
        H_t = model.step_backbone(prev_chunk, cache)           # [1, D]

        chunk_bytes = data[t * K : (t + 1) * K]
        decoded_in_chunk: list[int] = []

        # ── light steps: chained heads ──
        for k in range(K):
            if k == 0:
                prev_bytes = None
            else:
                prev_bytes = torch.tensor(
                    [decoded_in_chunk], dtype=torch.long, device=device
                )

            logits = model.predict_head(k, H_t, prev_bytes)   # [1, V]
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            sym = int(chunk_bytes[k])
            enc.encode(np.array([sym], dtype=np.int32), fam, probs)
            decoded_in_chunk.append(sym)

        # Prepare the chunk we just encoded as input for the next backbone step
        prev_chunk = torch.tensor(
            [chunk_bytes.tolist()], dtype=torch.long, device=device
        )
        pbar.update(1)

    pbar.close()

    compressed = enc.get_compressed()
    metadata = {
        "orig_len": orig_len,
        "padded_len": L,
        "pad": pad,
        "K": K,
        "T": T,
    }
    return compressed, metadata


def decompress_hydra(
    model,
    compressed: np.ndarray,
    metadata: dict,
    device: str = "cpu",
    progress: bool = True,
) -> bytes:
    """Decompress a HydraBOA bitstream back to the original bytes."""
    import constriction

    model.eval().to(device)

    K = metadata["K"]
    T = metadata["T"]
    orig_len = metadata["orig_len"]

    dec = constriction.stream.queue.RangeDecoder(compressed)
    fam = constriction.stream.model.Categorical(perfect=False)

    cache = model.init_stream(max_chunks=T, batch_size=1, device=device)
    output = np.empty(T * K, dtype=np.uint8)
    prev_chunk: torch.Tensor | None = None

    pbar = tqdm(total=T, disable=not progress, desc="Decompress (HydraBOA)",
                unit="chunk", mininterval=0.3)

    for t in range(T):
        # ── heavy step ──
        H_t = model.step_backbone(prev_chunk, cache)

        decoded_in_chunk: list[int] = []

        # ── light steps ──
        for k in range(K):
            if k == 0:
                prev_bytes = None
            else:
                prev_bytes = torch.tensor(
                    [decoded_in_chunk], dtype=torch.long, device=device
                )

            logits = model.predict_head(k, H_t, prev_bytes)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            sym = int(dec.decode(fam, probs)[0])
            decoded_in_chunk.append(sym)
            output[t * K + k] = sym

        prev_chunk = torch.tensor(
            [decoded_in_chunk], dtype=torch.long, device=device
        )
        pbar.update(1)

    pbar.close()
    return bytes(output[:orig_len])


# ═══════════════════════════════════════════════════════════════════
#  GPU-batched codec (N chunks in parallel via gpu_range_coder)
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def compress_hydra_gpu(
    model,
    data_bytes: bytes,
    K: int = 4,
    device: str = "cuda",
    progress: bool = True,
    batch_streams: int = 5000,
) -> tuple:
    """Compress *data_bytes* with HydraBOA using the GPU range coder.

    Returns
    -------
    compressed_list : list[np.ndarray[uint32]]
    metadata        : dict
    """
    if not _HAS_GPU_RC:
        raise ImportError("gpu_range_coder is required for GPU compression")

    model.eval().to(device)
    data = np.frombuffer(data_bytes, dtype=np.uint8).copy()
    orig_len = len(data)

    remainder = len(data) % K
    pad = (K - remainder) if remainder else 0
    if pad:
        data = np.concatenate([data, np.zeros(pad, dtype=np.uint8)])

    L_total = len(data)
    T_total = L_total // K
    vocab_size = model.vocab_size

    # Split into N chunks of equal size for batched GPU processing.
    # Each "chunk" is one file-segment whose backbone timesteps are
    # processed in lockstep across all N streams.
    #
    # We pick chunk_size so that N ≈ batch_streams, but the backbone
    # sequence length stays reasonable (≤ ~4096 steps).
    max_backbone_steps = 2500  # T per stream
    chunk_bytes = max_backbone_steps * K  # bytes per stream
    N = max(1, (L_total + chunk_bytes - 1) // chunk_bytes)
    N = min(N, batch_streams)
    chunk_bytes = ((L_total + N - 1) // N)
    # Align to K
    chunk_bytes = ((chunk_bytes + K - 1) // K) * K
    # Recalculate N after alignment
    N = (L_total + chunk_bytes - 1) // chunk_bytes

    T_per_chunk = chunk_bytes // K

    # Pad data to N * chunk_bytes
    padded_len = N * chunk_bytes
    if len(data) < padded_len:
        data = np.concatenate([data, np.zeros(padded_len - len(data), dtype=np.uint8)])

    # Reshape into [N, T_per_chunk, K] and pack onto GPU
    chunks_np = data[:padded_len].reshape(N, T_per_chunk, K)
    chunks_gpu = torch.from_numpy(chunks_np.astype(np.int64)).to(device)  # [N, T, K]

    # Total symbols to encode = N * T_per_chunk * K
    total_syms = N * T_per_chunk * K
    # GPU range coder: N streams, vocab_size alphabet, total_syms max length
    batch_rc = _gr.gpu.queue.RangeCoderBatch(N, vocab_size, T_per_chunk * K)

    # Initialize backbone streaming state
    cache = model.init_stream(max_chunks=T_per_chunk, batch_size=N, device=device)
    prev_chunks = None  # will be [N, K] for subsequent steps

    pbar = tqdm(total=T_per_chunk * K, disable=not progress,
                desc=f"Compress (HydraBOA GPU x{N})", unit="sym", mininterval=0.3)

    for t in range(T_per_chunk):
        # ── heavy step: backbone across all N streams ──
        H_t = model.step_backbone(prev_chunks, cache, batch_size=N)  # [N, D]

        current_chunk = chunks_gpu[:, t, :]  # [N, K]

        # ── light steps: K chained heads ──
        for k in range(K):
            if k == 0:
                prev_bytes = None
            else:
                prev_bytes = current_chunk[:, :k]  # [N, k] — teacher-forced (known)

            logits = model.predict_head(k, H_t, prev_bytes)  # [N, V]
            probs = torch.softmax(logits, dim=-1).to(torch.float32)

            syms = current_chunk[:, k].to(torch.int32)  # [N]
            batch_rc.encode_step(syms, probs)

            pbar.update(N)

        prev_chunks = current_chunk  # [N, K]

    pbar.close()
    batch_rc.finalize()
    compressed_list = batch_rc.get_compressed_list()

    metadata = {
        "orig_len": orig_len,
        "padded_len": padded_len,
        "pad": padded_len - orig_len,
        "K": K,
        "T_per_chunk": T_per_chunk,
        "N": N,
        "chunk_bytes": chunk_bytes,
    }
    return compressed_list, metadata


@torch.inference_mode()
def decompress_hydra_gpu(
    model,
    compressed_list: list,
    metadata: dict,
    device: str = "cuda",
    progress: bool = True,
) -> bytes:
    """Decompress HydraBOA bitstreams using the GPU range coder."""
    if not _HAS_GPU_RC:
        raise ImportError("gpu_range_coder is required for GPU decompression")

    model.eval().to(device)

    K = metadata["K"]
    T_per_chunk = metadata["T_per_chunk"]
    N = metadata["N"]
    orig_len = metadata["orig_len"]
    chunk_bytes = metadata["chunk_bytes"]
    vocab_size = model.vocab_size

    batch_rc = _gr.gpu.queue.RangeCoderBatch(N, vocab_size, T_per_chunk * K)
    batch_rc.load_compressed_list(compressed_list)
    batch_rc.init_decoder()

    # Output buffer on GPU
    output = torch.empty((N, T_per_chunk, K), dtype=torch.uint8, device=device)

    cache = model.init_stream(max_chunks=T_per_chunk, batch_size=N, device=device)
    prev_chunks = None
    out_syms = torch.empty((N,), dtype=torch.int32, device=device)

    pbar = tqdm(total=T_per_chunk * K, disable=not progress,
                desc=f"Decompress (HydraBOA GPU x{N})", unit="sym", mininterval=0.3)

    for t in range(T_per_chunk):
        H_t = model.step_backbone(prev_chunks, cache, batch_size=N)

        decoded_k = []  # accumulate decoded bytes for this chunk-step

        for k in range(K):
            if k == 0:
                prev_bytes = None
            else:
                prev_bytes = torch.stack(decoded_k, dim=1)  # [N, k]

            logits = model.predict_head(k, H_t, prev_bytes)
            probs = torch.softmax(logits, dim=-1).to(torch.float32)

            batch_rc.decode_step(probs, out_syms)
            output[:, t, k] = out_syms.to(torch.uint8)
            decoded_k.append(out_syms.to(torch.long))

            pbar.update(N)

        prev_chunks = torch.stack(decoded_k, dim=1)  # [N, K]

    pbar.close()

    # Flatten and trim
    flat = output.reshape(-1).cpu().numpy()
    return bytes(flat[:orig_len])
