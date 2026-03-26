#!/usr/bin/env python3
"""
Convert HydraBOA PyTorch checkpoint (.pt) to binary format (.bin) for C++ inference.

Binary layout (all tensors fp32 little-endian unless --fp16):
────────────────────────────────────────────────────────────
Header (16 bytes):
    magic          4B  "HYD\x00" (fp32) or "HYD\x01" (fp16)
    d_model        4B  uint32
    n_layers       4B  uint32
    K              4B  uint32

Weights (in order):
    byte_embed.weight           [256, D]
    bos_embed                   [D]
    chunk_proj.weight           [D, K*D]
    chunk_proj.bias             [D]

    -- for each block i: --
      blocks.i.ln1.weight       [D]
      blocks.i.ln1.bias         [D]
      ... Mamba block weights (same layout as convert_boa_weights.py) ...
      blocks.i.ln2.weight       [D]
      blocks.i.ln2.bias         [D]
      blocks.i.ff.0.weight      [4D, D]
      blocks.i.ff.0.bias        [4D]
      blocks.i.ff.2.weight      [D, 4D]
      blocks.i.ff.2.bias        [D]

    backbone_norm.weight        [D]
    backbone_norm.bias          [D]

    -- for each head k (0 .. K-1): --
      heads.k.0.weight          [D, (1+k)*D]
      heads.k.0.bias            [D]
      heads.k.2.weight          [256, D]
      heads.k.2.bias            [256]

Usage:
    python convert_hydra_weights.py --model checkpoints/hydra_d64_L1_K4.pt --output hydra_d64_L1_K4.bin
"""

import argparse
import struct
import sys

import numpy as np
import torch

MAGIC_FP32 = b"HYD\x00"
MAGIC_FP16 = b"HYD\x01"

g_use_fp16 = False


def write_tensor(f, t):
    if g_use_fp16:
        f.write(t.half().cpu().numpy().tobytes())
    else:
        f.write(t.float().cpu().numpy().tobytes())


def convert(model_path, output_path):
    global g_use_fp16

    print(f"Loading {model_path} ...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    d_model = ckpt.get("d_model", None)
    n_layers = ckpt.get("num_layers", None)
    K = ckpt.get("K", None)

    # fallback: infer from weights
    if d_model is None:
        d_model = sd["byte_embed.weight"].shape[1]
    if K is None:
        K = sd["chunk_proj.weight"].shape[1] // d_model
    if n_layers is None:
        n_layers = 0
        while f"blocks.{n_layers}.ln1.weight" in sd:
            n_layers += 1

    print(f"  d_model={d_model}, n_layers={n_layers}, K={K}")

    # Detect fp16
    sample_key = next(k for k in sd if "weight" in k)
    if sd[sample_key].dtype == torch.float16:
        print("  Detected fp16 weights → saving fp16")
        g_use_fp16 = True

    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC_FP16 if g_use_fp16 else MAGIC_FP32)
        f.write(struct.pack("<III", d_model, n_layers, K))

        # byte_embed
        print("  byte_embed.weight", list(sd["byte_embed.weight"].shape))
        write_tensor(f, sd["byte_embed.weight"])

        # bos_embed
        print("  bos_embed", list(sd["bos_embed"].shape))
        write_tensor(f, sd["bos_embed"])

        # chunk_proj
        print("  chunk_proj.weight", list(sd["chunk_proj.weight"].shape))
        write_tensor(f, sd["chunk_proj.weight"])
        print("  chunk_proj.bias", list(sd["chunk_proj.bias"].shape))
        write_tensor(f, sd["chunk_proj.bias"])

        # Blocks
        for i in range(n_layers):
            prefix = f"blocks.{i}."
            print(f"  Block {i}:")

            # LN1
            write_tensor(f, sd[f"{prefix}ln1.weight"])
            write_tensor(f, sd[f"{prefix}ln1.bias"])

            # Mamba weights
            m = f"{prefix}mamba."
            print(f"    in_proj.weight {list(sd[f'{m}in_proj.weight'].shape)}")
            write_tensor(f, sd[f"{m}in_proj.weight"])
            if f"{m}in_proj.bias" in sd:
                write_tensor(f, sd[f"{m}in_proj.bias"])
            else:
                write_tensor(f, torch.zeros(sd[f"{m}in_proj.weight"].shape[0]))

            write_tensor(f, sd[f"{m}conv1d.weight"])
            write_tensor(f, sd[f"{m}conv1d.bias"])
            write_tensor(f, sd[f"{m}x_proj.weight"])
            write_tensor(f, sd[f"{m}dt_proj.weight"])
            write_tensor(f, sd[f"{m}dt_proj.bias"])
            write_tensor(f, sd[f"{m}A_log"])
            write_tensor(f, sd[f"{m}D"])
            write_tensor(f, sd[f"{m}out_proj.weight"])
            if f"{m}out_proj.bias" in sd:
                write_tensor(f, sd[f"{m}out_proj.bias"])
            else:
                write_tensor(f, torch.zeros(sd[f"{m}out_proj.weight"].shape[0]))

            # LN2
            write_tensor(f, sd[f"{prefix}ln2.weight"])
            write_tensor(f, sd[f"{prefix}ln2.bias"])

            # FF
            write_tensor(f, sd[f"{prefix}ff.0.weight"])
            write_tensor(f, sd[f"{prefix}ff.0.bias"])
            write_tensor(f, sd[f"{prefix}ff.2.weight"])
            write_tensor(f, sd[f"{prefix}ff.2.bias"])

        # backbone_norm
        print("  backbone_norm")
        write_tensor(f, sd["backbone_norm.weight"])
        write_tensor(f, sd["backbone_norm.bias"])

        # Heads
        for k in range(K):
            prefix = f"heads.{k}."
            w0 = sd[f"{prefix}0.weight"]
            b0 = sd[f"{prefix}0.bias"]
            w2 = sd[f"{prefix}2.weight"]
            b2 = sd[f"{prefix}2.bias"]
            print(f"  Head {k}: linear1 {list(w0.shape)}, linear2 {list(w2.shape)}")
            write_tensor(f, w0)
            write_tensor(f, b0)
            write_tensor(f, w2)
            write_tensor(f, b2)

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HydraBOA weights to C++ binary")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if args.fp16:
        g_use_fp16 = True

    convert(args.model, args.output)
