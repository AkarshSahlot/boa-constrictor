import torch
import argparse
import struct
import numpy as np
import sys
import os

# Binary format header:
# Bytes 0-3: "BOA\x00" (fp32) or "BOA\x01" (fp16)
# Bytes 4+: weight data
MAGIC_FP32 = b"BOA\x00"
MAGIC_FP16 = b"BOA\x01"

g_use_fp16 = False

def write_tensor(f, t):
    if g_use_fp16:
        f.write(t.half().cpu().numpy().tobytes())
    else:
        f.write(t.float().cpu().numpy().tobytes())


def detect_backbone(sd):
    """Auto-detect backbone type from state dict keys."""
    for key in sd:
        if ".mamba." in key:
            return "mamba"
        if ".lstm." in key:
            return "lstm"
        if ".gru." in key and ".mingru." not in key:
            return "gru"
        if ".mingru." in key:
            return "mingru"
    return "mamba"  # default


def export_mamba_block(f, sd, prefix):
    """Export Mamba backbone weights for one block."""
    m_pre = f"{prefix}mamba."
    # in_proj
    print(f"  Writing {m_pre}in_proj.weight {sd[f'{m_pre}in_proj.weight'].shape}")
    write_tensor(f, sd[f"{m_pre}in_proj.weight"])

    if f"{m_pre}in_proj.bias" in sd:
        print(f"  Writing {m_pre}in_proj.bias {sd[f'{m_pre}in_proj.bias'].shape}")
        write_tensor(f, sd[f"{m_pre}in_proj.bias"])
    else:
        print(f"  Writing {m_pre}in_proj.bias (ZEROS - not found)")
        w_shape = sd[f"{m_pre}in_proj.weight"].shape
        bias_sim = torch.zeros(w_shape[0], dtype=torch.float32)
        write_tensor(f, bias_sim)

    # conv1d
    print(f"  Writing {m_pre}conv1d.weight")
    write_tensor(f, sd[f"{m_pre}conv1d.weight"])
    if f"{m_pre}conv1d.bias" in sd:
         print(f"  Writing {m_pre}conv1d.bias")
         write_tensor(f, sd[f"{m_pre}conv1d.bias"])

    # x_proj
    print(f"  Writing {m_pre}x_proj.weight {sd[f'{m_pre}x_proj.weight'].shape}")
    write_tensor(f, sd[f"{m_pre}x_proj.weight"])

    # dt_proj
    print(f"  Writing {m_pre}dt_proj.weight {sd[f'{m_pre}dt_proj.weight'].shape}")
    write_tensor(f, sd[f"{m_pre}dt_proj.weight"])
    print(f"  Writing {m_pre}dt_proj.bias {sd[f'{m_pre}dt_proj.bias'].shape}")
    write_tensor(f, sd[f"{m_pre}dt_proj.bias"])

    # A_log
    print(f"  Writing {m_pre}A_log")
    write_tensor(f, sd[f"{m_pre}A_log"])
    # D
    print(f"  Writing {m_pre}D")
    write_tensor(f, sd[f"{m_pre}D"])

    # out_proj
    print(f"  Writing {m_pre}out_proj.weight")
    write_tensor(f, sd[f"{m_pre}out_proj.weight"])
    if f"{m_pre}out_proj.bias" in sd:
        print(f"  Writing {m_pre}out_proj.bias")
        write_tensor(f, sd[f"{m_pre}out_proj.bias"])
    else:
        print(f"  Writing {m_pre}out_proj.bias (ZEROS - not found)")
        w_shape = sd[f"{m_pre}out_proj.weight"].shape
        bias_sim = torch.zeros(w_shape[0], dtype=torch.float32)
        write_tensor(f, bias_sim)


def export_lstm_block(f, sd, prefix):
    """Export LSTM backbone weights for one block.

    PyTorch nn.LSTM keys:
      lstm.weight_ih_l0  [4*d, d]
      lstm.weight_hh_l0  [4*d, d]
      lstm.bias_ih_l0    [4*d]
      lstm.bias_hh_l0    [4*d]

    C++ load order: w_ih, b_ih, w_hh, b_hh
    """
    m_pre = f"{prefix}lstm."
    for name in ["weight_ih_l0", "bias_ih_l0", "weight_hh_l0", "bias_hh_l0"]:
        key = f"{m_pre}{name}"
        print(f"  Writing {key} {sd[key].shape}")
        write_tensor(f, sd[key])


def export_gru_block(f, sd, prefix):
    """Export GRU backbone weights for one block.

    PyTorch nn.GRU keys:
      gru.weight_ih_l0  [3*d, d]
      gru.weight_hh_l0  [3*d, d]
      gru.bias_ih_l0    [3*d]
      gru.bias_hh_l0    [3*d]

    C++ load order: w_ih, b_ih, w_hh, b_hh
    """
    m_pre = f"{prefix}gru."
    for name in ["weight_ih_l0", "bias_ih_l0", "weight_hh_l0", "bias_hh_l0"]:
        key = f"{m_pre}{name}"
        print(f"  Writing {key} {sd[key].shape}")
        write_tensor(f, sd[key])


def export_mingru_block(f, sd, prefix):
    """Export minGRU backbone weights for one block.

    Keys:
      mingru.linear_z.weight  [d, d]
      mingru.linear_z.bias    [d]
      mingru.linear_h.weight  [d, d]
      mingru.linear_h.bias    [d]

    C++ load order: w_z, b_z, w_h, b_h
    """
    m_pre = f"{prefix}mingru."
    for name in ["linear_z.weight", "linear_z.bias", "linear_h.weight", "linear_h.bias"]:
        key = f"{m_pre}{name}"
        print(f"  Writing {key} {sd[key].shape}")
        write_tensor(f, sd[key])


BACKBONE_EXPORTERS = {
    "mamba": export_mamba_block,
    "mambav1": export_mamba_block,
    "lstm": export_lstm_block,
    "gru": export_gru_block,
    "mingru": export_mingru_block,
}


def convert(model_path, output_path, backbone=None):
    global g_use_fp16
    print(f"Loading {model_path}...")
    sd = torch.load(model_path, map_location='cpu', weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif "state_dict" in sd:
        sd = sd["state_dict"]

    # Auto-detect backbone if not specified
    if backbone is None:
        backbone = detect_backbone(sd)
    print(f"Backbone: {backbone}")

    if backbone not in BACKBONE_EXPORTERS:
        print(f"ERROR: Unknown backbone '{backbone}'. Supported: {list(BACKBONE_EXPORTERS.keys())}")
        return

    export_block_fn = BACKBONE_EXPORTERS[backbone]

    # Auto-detect fp16 weights from the checkpoint
    sample_key = next(k for k in sd if 'weight' in k)
    if sd[sample_key].dtype == torch.float16:
        print("Detected fp16 weights in checkpoint — saving as fp16 binary")
        g_use_fp16 = True

    precision_str = "fp16" if g_use_fp16 else "fp32"
    print(f"Output precision: {precision_str}")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(MAGIC_FP16 if g_use_fp16 else MAGIC_FP32)

        # 1. Embedding
        print("Exporting Embedding...")
        if "embedding.weight" in sd:
            write_tensor(f, sd["embedding.weight"])
        else:
            print("ERROR: embedding.weight not found")
            return

        # 2. Blocks
        i = 0
        while True:
            prefix = f"blocks.{i}."
            if f"{prefix}ln1.weight" not in sd:
                break

            print(f"Exporting Block {i} ({backbone})...")

            # LN1
            write_tensor(f, sd[f"{prefix}ln1.weight"])
            write_tensor(f, sd[f"{prefix}ln1.bias"])

            # Backbone-specific weights
            export_block_fn(f, sd, prefix)

            # LN2
            write_tensor(f, sd[f"{prefix}ln2.weight"])
            write_tensor(f, sd[f"{prefix}ln2.bias"])

            # FF — handle both Sequential (ff.0, ff.2) and named (ff.net.0, ff.net.2)
            ff_w1_key = f"{prefix}ff.0.weight"
            if ff_w1_key not in sd:
                ff_w1_key = f"{prefix}ff.net.0.weight"
            ff_prefix = ff_w1_key.rsplit(".weight", 1)[0].rsplit(".0", 1)[0]

            write_tensor(f, sd[f"{ff_prefix}.0.weight"])
            write_tensor(f, sd[f"{ff_prefix}.0.bias"])
            write_tensor(f, sd[f"{ff_prefix}.2.weight"])
            write_tensor(f, sd[f"{ff_prefix}.2.bias"])

            i += 1

        print(f"Exported {i} blocks.")

        # 2b. Final RMSNorm (non-mamba backbones)
        if "final_norm.weight" in sd:
            print("Exporting final_norm (RMSNorm)...")
            write_tensor(f, sd["final_norm.weight"])

        # 3. Head
        print("Exporting Head...")
        write_tensor(f, sd["head.0.weight"])
        write_tensor(f, sd["head.0.bias"])
        write_tensor(f, sd["head.2.weight"])
        write_tensor(f, sd["head.2.bias"])

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BoaConstrictor PyTorch weights to C++ binary format")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--output", required=True, help="Path to .bin output")
    parser.add_argument("--fp16", action="store_true", help="Force fp16 output (auto-detected from checkpoint if not set)")
    parser.add_argument("--backbone", default=None,
                        choices=["mamba", "mambav1", "lstm", "gru", "mingru"],
                        help="Backbone type (auto-detected from state dict if not set)")
    args = parser.parse_args()

    if args.fp16:
        g_use_fp16 = True

    convert(args.model, args.output, backbone=args.backbone)
