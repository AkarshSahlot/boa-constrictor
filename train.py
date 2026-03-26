import torch
import torch.nn as nn
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
import time

# Optional wandb import — gracefully degrade if not installed
try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False

# --- eval loop (reports mean bpp across the loader) ---
@torch.inference_mode()
def evaluate_bpp(model, loader, criterion, device="cuda", vocab_size=256):
    model.eval().to(device)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(x)  # [B, L-1, vocab_size]
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    mean_nll = total_loss / max(1, total_tokens)
    bpp = mean_nll / np.log(2)  # bits per input byte
    return bpp

def train(model, train_loader, val_loader, test_loader, optimizer, criterion, device="cuda", name="BoaBytePredictor", NUM_EPOCHS=10, PRECISION="fp32", progress=True, start_epoch=1, vocab_size=256, patience=0, scheduler=None, grad_clip=0.0, position_weights=None, wandb_run=None):
    
    IS_CUDA = torch.cuda.is_available() and device == "cuda"

    # Pre-process position weights (if provided) for fast per-token weighting
    _pw = None
    if position_weights is not None:
        # position_weights: [seq_len] float tensor — weight for each target position
        _pw = torch.tensor(position_weights, dtype=torch.float32, device=device)
        print(f"[INFO] Position-weighted loss: coarse={position_weights[0]:.2f}, fine={position_weights[-1]:.2f}")
    
    print(f"[INFO] Using precision = {PRECISION}")
    if patience > 0:
        print(f"[INFO] Early stopping enabled: patience = {patience} epochs")
    if scheduler is not None:
        print(f"[INFO] LR scheduler: {scheduler.__class__.__name__}")
    if grad_clip > 0:
        print(f"[INFO] Gradient clipping: max_norm={grad_clip}")

    def get_autocast_dtype(precision):
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        elif precision == "fp8":
            try:
                return torch.float8_e5m2  # Hopper architecture only (H100 / RTX 5090)
            except AttributeError:
                print("[WARN] FP8 not supported on this PyTorch build, falling back to FP16")
                return torch.float16
        else:
            return torch.float32
        
    AUTODTYPE = get_autocast_dtype(PRECISION)
    amp_enabled = PRECISION in ["bf16", "fp16", "fp8"] and IS_CUDA
    save_half = PRECISION in ["fp16", "fp8"]  # save weights as fp16 when training in reduced precision (bf16 saves as fp32)
    # GradScaler is not needed/used for bf16 (no loss scaling required)
    use_scaler = PRECISION in ["fp16", "fp8"] and IS_CUDA
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_bpp = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    # Wandb logging helper
    _wb = wandb_run  # None when wandb is disabled

    # Helper to strip torch.compile's _orig_mod. prefix from state_dict keys
    def _clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    def _get_backbone_name(m):
        # torch.compile wraps model in OptimizedModule; unwrap to find _backbone_name
        inner = getattr(m, '_orig_mod', m)
        return getattr(inner, '_backbone_name', 'unknown')

    model.train().to(device)
    train_steps_per_epoch = len(train_loader)
    total_train_steps = max(1, train_steps_per_epoch)
    global_step = (start_epoch - 1) * total_train_steps
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()
        epoch_tokens = 0
        loader = tqdm(train_loader, total=total_train_steps, desc=f"Epoch {epoch} [{PRECISION}]", disable=not progress)
        for batch in loader:
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)
            epoch_tokens += y.numel()

            optimizer.zero_grad(set_to_none=True)

            # --- Automatic mixed precision block ---
            with torch.autocast(device_type=device, dtype=AUTODTYPE, enabled=amp_enabled):
                logits = model(x)
                if _pw is not None:
                    # Per-position weighted loss: tile weights across the sequence
                    # x has shape [B, seq_len-1]; positions repeat with period event_bytes
                    B, L = y.shape
                    pw_len = len(_pw)
                    # Tile position weights to cover L positions
                    reps = (L + pw_len - 1) // pw_len
                    weights = _pw.repeat(reps)[:L]  # [L]
                    weights = weights.unsqueeze(0).expand(B, -1)  # [B, L]
                    import torch.nn.functional as F_local
                    loss_tok = F_local.cross_entropy(
                        logits.reshape(-1, vocab_size), y.reshape(-1), reduction="none"
                    ).reshape(B, L)
                    loss = (loss_tok * weights).sum() / weights.sum()
                else:
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            # --- Scaled backward for FP16/FP8 ---
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # --- Progress ---
            bits_per_byte = loss.item() / np.log(2)
            global_step += 1
            if progress:
                loader.set_postfix(
                    loss=f"{loss.item():.4f}",
                    bits=f"{bits_per_byte:.3f}",
                    ratio=f"{(8 / bits_per_byte):.2f}x"
                )

            # --- Wandb step-level logging ---
            if _wb is not None:
                _wb.log({
                    "train/loss": loss.item(),
                    "train/bpp": bits_per_byte,
                    "train/compression_ratio": 8.0 / max(bits_per_byte, 1e-8),
                }, step=global_step)

        epoch_elapsed = time.perf_counter() - epoch_start
        tok_per_sec = epoch_tokens / max(epoch_elapsed, 1e-6)
        mb_per_sec = epoch_tokens / (1024 * 1024) / max(epoch_elapsed, 1e-6)
        print(f"  Epoch {epoch} throughput: {tok_per_sec:,.0f} tok/s ({mb_per_sec:.2f} MB/s), {epoch_elapsed:.1f}s")

        _sd = _clean_state_dict(model.state_dict())
        if save_half:
            _sd = {k: v.half() if v.is_floating_point() else v for k, v in _sd.items()}
        _ckpt = {
            'state_dict': _sd,
            'backbone': _get_backbone_name(model),
        }
        torch.save(_ckpt, f"{name}_{datetime.now().strftime('%dth%b')}_Checkpoint_epoch_{epoch}_{PRECISION}.pt")
        val_bpp = evaluate_bpp(model, val_loader, criterion, device=device, vocab_size=vocab_size)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] val bpp={val_bpp:.4f} (ratio ~ {8/val_bpp:.2f}x)  lr={current_lr:.2e}")

        # --- Wandb epoch-level logging ---
        if _wb is not None:
            _wb.log({
                "epoch": epoch,
                "val/bpp": val_bpp,
                "val/compression_ratio": 8.0 / max(val_bpp, 1e-8),
                "train/throughput_tok_s": tok_per_sec,
                "train/throughput_MB_s": mb_per_sec,
                "train/epoch_time_s": epoch_elapsed,
                "train/lr": current_lr,
                "best/val_bpp": best_val_bpp,
                "best/epoch": best_epoch,
            }, step=global_step)

        if scheduler is not None:
            scheduler.step()

        # --- Early stopping logic ---
        if val_bpp < best_val_bpp:
            best_val_bpp = val_bpp
            best_epoch = epoch
            epochs_without_improvement = 0
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if patience > 0:
                print(f"  [early stop] no improvement for {epochs_without_improvement}/{patience} epochs (best={best_val_bpp:.4f} @ epoch {best_epoch})")
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  [early stop] Stopping at epoch {epoch}. Restoring best model from epoch {best_epoch}.")
                break

    # Restore best model if early stopping was used
    if best_state_dict is not None and patience > 0:
        model.load_state_dict(best_state_dict)
        print(f"  [early stop] Restored best model (val bpp={best_val_bpp:.4f}, epoch {best_epoch})")

    _final_sd = _clean_state_dict(model.state_dict())
    if save_half:
        _final_sd = {k: v.half() if v.is_floating_point() else v for k, v in _final_sd.items()}
    _final_ckpt = {
        'state_dict': _final_sd,
        'backbone': _get_backbone_name(model),
    }
    torch.save(_final_ckpt, f"{name}_final_model_{PRECISION}.pt")
    test_bpp = evaluate_bpp(model, test_loader, criterion, device=device, vocab_size=vocab_size)
    print(f"[TEST] bpp={test_bpp:.4f}  ratio ~ {8/test_bpp:.2f}x")

    # --- Wandb final summary ---
    if _wb is not None:
        _wb.log({
            "test/bpp": test_bpp,
            "test/compression_ratio": 8.0 / max(test_bpp, 1e-8),
        }, step=global_step)
        _wb.summary["test_bpp"] = test_bpp
        _wb.summary["test_compression_ratio"] = 8.0 / max(test_bpp, 1e-8)
        _wb.summary["best_val_bpp"] = best_val_bpp
        _wb.summary["best_epoch"] = best_epoch
        _wb.summary["total_epochs"] = epoch

