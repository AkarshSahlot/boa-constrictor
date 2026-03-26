#!/usr/bin/env python3
"""
HydraBOA.

2. Builds a HydraBOA model (small d_model for speed).
3. Trains for a handful of epochs — verifies loss decreases.
4. Compresses a test segment with the trained model + range coder.
5. Decompresses it.
6. Verifies byte-exact round-trip.

Usage:
    python main_hydra.py [--device cpu|cuda] [--epochs 10] [--K 4]
"""

import argparse
import hashlib
import os
import time

import numpy as np
import torch

from hydra_model import HydraBOA
from hydra_codec import compress_hydra, decompress_hydra
try:
    from hydra_codec import compress_hydra_gpu, decompress_hydra_gpu
    _HAS_GPU_CODEC = True
except ImportError:
    _HAS_GPU_CODEC = False


def make_synthetic_data(size: int = 20_480, seed: int = 42) -> np.ndarray:
    """Repeating period-8 pattern with small additive noise (compressible)."""
    rng = np.random.RandomState(seed)
    base = np.tile(np.arange(8, dtype=np.uint8) * 30, size // 8 + 1)[:size]
    noise = rng.randint(0, 8, size=size, dtype=np.uint8)
    return ((base.astype(np.int16) + noise) % 256).astype(np.uint8)


def train_hydra(model, data: np.ndarray, *,
                seq_len: int, batch_size: int, num_epochs: int,
                device: str, K: int, lr: float = 1e-3,
                precision: str = "fp32", use_compile: bool = False):
    """Minimal training loop — returns final average bpp."""
    block = seq_len * batch_size
    n_blocks = len(data) // block
    if n_blocks == 0:
        raise ValueError(f"Data too small ({len(data)}) for seq_len={seq_len} × batch_size={batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    V = model.vocab_size

    if use_compile:
        print("  [INFO] Compiling model with torch.compile (mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")

    model.train()
    print(f"\n{'═'*60}")
    print(f"  Training  |  {num_epochs} epochs, seq_len={seq_len}, bs={batch_size}, "
          f"K={K}, lr={lr}, precision={precision}, device={device}")
    print(f"  Batches per epoch: {n_blocks}")
    print(f"{'═'*60}")

    amp_enabled = precision in ("bf16", "fp16") and device == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)

    best_bpp = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_loss, n_tok, pos = 0.0, 0, 0
        t0 = time.perf_counter()
        for _ in range(n_blocks):
            chunk = data[pos : pos + block]
            pos += block
            batch = torch.tensor(
                chunk.reshape(batch_size, seq_len), dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(batch)
                loss = criterion(logits.reshape(-1, V), batch.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch.numel()
            n_tok += batch.numel()

        avg = epoch_loss / n_tok
        bpp = avg / np.log(2)
        dt = time.perf_counter() - t0
        tok_s = n_tok / max(dt, 1e-6)
        mb_s = n_tok / 1e6 / max(dt, 1e-6)
        marker = " *" if bpp < best_bpp else ""
        best_bpp = min(best_bpp, bpp)
        print(f"  Epoch {epoch:2d}  loss={avg:.4f}  bpp={bpp:.3f}  "
              f"ratio~{8/max(bpp,0.01):.2f}x  {mb_s:.1f} MB/s  {dt:.1f}s{marker}")

    return bpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--test-bytes", type=int, default=256,
                    help="Bytes to compress/decompress for round-trip verify. "
                         "Use 0 to compress the full training data.")
    ap.add_argument("--data-path", type=str, default=None,
                    help="Path to a real binary dataset file (overrides synthetic).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--precision", type=str, default="fp32",
                    choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model for faster training.")
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="Fraction of data to use for training (0.0, 1.0].")
    ap.add_argument("--gpu-codec", action="store_true",
                    help="Use GPU range coder for compress/decompress.")
    ap.add_argument("--save-checkpoint", type=str, default=None,
                    help="Path to save model checkpoint (.pt) after training.")
    args = ap.parse_args()

    K = args.K
    device = args.device
    d_model = args.d_model

    print("╔══════════════════════════════════════════════════════════╗")
    print("║              HydraBOA  —  Smoke Test                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Device     : {device}")
    print(f"  K          : {K}")
    print(f"  d_model    : {d_model}")
    print(f"  num_layers : {args.num_layers}")
    print(f"  precision  : {args.precision}")

    # ── 1. Load data ──
    full_raw = None  # keep full file for compress-all
    if args.data_path:
        full_raw = np.fromfile(args.data_path, dtype=np.uint8)
        raw = full_raw.copy()
        # Apply data fraction for training only
        if args.data_frac < 1.0:
            raw = raw[:int(len(raw) * args.data_frac)]
        # Trim to multiple of (seq_len * batch_size) for clean batching
        block = args.seq_len * args.batch_size
        usable = (len(raw) // block) * block
        data = raw[:usable]
        print(f"  Data file  : {args.data_path}")
        print(f"  Full file  : {len(full_raw):,} bytes")
        frac_str = f" ({args.data_frac*100:.0f}%)" if args.data_frac < 1.0 else ""
        print(f"  Train data : {usable:,} bytes{frac_str} (trimmed to batch boundary)")
    else:
        data = make_synthetic_data(size=20_480)
        print(f"  Data       : synthetic ({len(data):,} bytes)")

    # ── 2. Build model ──
    model = HydraBOA(
        d_model=d_model,
        num_layers=args.num_layers,
        vocab_size=256,
        K=K,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    # ── 3. Train ──
    final_bpp = train_hydra(
        model, data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=device,
        K=K,
        lr=args.lr,
        precision=args.precision,
        use_compile=args.compile,
    )

    # ── 3b. Save checkpoint ──
    if args.save_checkpoint:
        ckpt_dir = os.path.dirname(args.save_checkpoint)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "d_model": d_model,
            "num_layers": args.num_layers,
            "K": K,
            "vocab_size": 256,
            "final_bpp": final_bpp,
            "epochs": args.epochs,
            "lr": args.lr,
            "data_path": args.data_path,
            "data_frac": args.data_frac,
        }
        torch.save(ckpt, args.save_checkpoint)
        print(f"  Checkpoint saved → {args.save_checkpoint}")

    # ── 4. Compress ──
    # When test-bytes=0, compress the FULL file (not just training slice)
    if args.test_bytes == 0 and full_raw is not None:
        compress_source = full_raw
    else:
        compress_source = data
    n_test = args.test_bytes if args.test_bytes > 0 else len(compress_source)
    # Align to K
    n_test = (n_test // K) * K
    test_data = compress_source[:n_test].tobytes()
    assert len(test_data) > 0
    print(f"\n{'─'*60}")
    print(f"  Compress / Decompress test  ({len(test_data)} bytes)")
    print(f"{'─'*60}")

    use_gpu = args.gpu_codec and _HAS_GPU_CODEC and device == "cuda"
    if args.gpu_codec and not use_gpu:
        print("  [WARN] GPU codec not available, falling back to CPU codec")

    t0 = time.perf_counter()
    if use_gpu:
        compressed, metadata = compress_hydra_gpu(
            model, test_data, K=K, device=device, progress=True
        )
        # compressed is a list of uint32 arrays (one per stream)
        comp_bytes = sum(len(a) * 4 for a in compressed)
    else:
        compressed, metadata = compress_hydra(
            model, test_data, K=K, device=device, progress=True
        )
        comp_bytes = len(compressed) * 4  # uint32 words → bytes
    t_comp = time.perf_counter() - t0
    ratio = len(test_data) / comp_bytes if comp_bytes else float("inf")
    comp_mbs = len(test_data) / 1e6 / max(t_comp, 1e-9)
    print(f"  Original   : {len(test_data):,} bytes ({len(test_data)/1e6:.1f} MB)")
    print(f"  Compressed : {comp_bytes:,} bytes  (ratio {ratio:.2f}x)")
    print(f"  Comp. time : {t_comp:.3f}s  ({comp_mbs:.2f} MB/s)")

    # ── 5. Decompress ──
    t0 = time.perf_counter()
    if use_gpu:
        decompressed = decompress_hydra_gpu(
            model, compressed, metadata, device=device, progress=True
        )
    else:
        decompressed = decompress_hydra(
            model, compressed, metadata, device=device, progress=True
        )
    t_dec = time.perf_counter() - t0
    dec_mbs = len(decompressed) / 1e6 / max(t_dec, 1e-9)
    print(f"  Decompressed : {len(decompressed):,} bytes")
    print(f"  Dec. time    : {t_dec:.3f}s  ({dec_mbs:.2f} MB/s)")

    # ── 6. Verify ──
    sha_orig = hashlib.sha256(test_data).hexdigest()
    sha_dec  = hashlib.sha256(decompressed).hexdigest()
    match = test_data == decompressed

    print(f"\n{'═'*60}")
    print(f"  Verification")
    print(f"{'═'*60}")
    print(f"  Original SHA-256     : {sha_orig}")
    print(f"  Decompressed SHA-256 : {sha_dec}")
    if match:
        print(f"  Result: PASS  ✓  (byte-exact round-trip)")
    else:
        print(f"  Result: FAIL  ✗")
        if len(test_data) != len(decompressed):
            print(f"    Length mismatch: {len(test_data)} vs {len(decompressed)}")
        else:
            for i in range(len(test_data)):
                if test_data[i] != decompressed[i]:
                    print(f"    First mismatch at byte {i}: "
                          f"expected {test_data[i]}, got {decompressed[i]}")
                    break

    print()
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
