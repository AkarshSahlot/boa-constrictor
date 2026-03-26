
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from collections import defaultdict
import json
from datetime import datetime
from mamba_ssm import Mamba
from matplotlib.lines import Line2D
import pandas as pd

class CompressionEvaluator:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.results = defaultdict(list)

    @staticmethod
    def verify_bitwise(original_file: str, decompressed_file: str,
                       report_path: str = None) -> dict:
        """Verify bit-exact equality between original and decompressed files.
        
        Returns a dict with:
          - bit_exact: bool
          - original_size: int
          - decompressed_size: int  
          - original_sha256: str
          - decompressed_sha256: str
          - mismatch_count: int (number of differing bytes, capped at 1M scan)
          - first_mismatch_offset: int or None
        
        [Referee Minor #6]: Provides a direct bitwise equality statement with
        checksum comparison for the lossless claim.
        """
        import os
        with open(original_file, 'rb') as f:
            orig_data = f.read()
        with open(decompressed_file, 'rb') as f:
            dec_data = f.read()

        orig_hash = hashlib.sha256(orig_data).hexdigest()
        dec_hash = hashlib.sha256(dec_data).hexdigest()
        bit_exact = (orig_data == dec_data)

        mismatch_count = 0
        first_mismatch = None
        if not bit_exact:
            cap = min(len(orig_data), len(dec_data), 1_000_000)
            for i in range(cap):
                if i < len(orig_data) and i < len(dec_data):
                    if orig_data[i] != dec_data[i]:
                        mismatch_count += 1
                        if first_mismatch is None:
                            first_mismatch = i
            # Count length difference as mismatches
            mismatch_count += abs(len(orig_data) - len(dec_data))

        result = {
            'bit_exact': bit_exact,
            'original_size': len(orig_data),
            'decompressed_size': len(dec_data),
            'original_sha256': orig_hash,
            'decompressed_sha256': dec_hash,
            'mismatch_count': mismatch_count,
            'first_mismatch_offset': first_mismatch,
        }

        # Print report
        print("\n  === Bitwise Verification Report ===")
        print(f"  Original:     {os.path.basename(original_file)} ({len(orig_data):,} bytes)")
        print(f"  Decompressed: {os.path.basename(decompressed_file)} ({len(dec_data):,} bytes)")
        print(f"  SHA-256 original:     {orig_hash}")
        print(f"  SHA-256 decompressed: {dec_hash}")
        if bit_exact:
            print(f"  Result: BIT-EXACT MATCH — lossless round-trip verified")
        else:
            print(f"  Result: MISMATCH — {mismatch_count} differing bytes")
            if first_mismatch is not None:
                print(f"  First mismatch at byte offset: {first_mismatch}")
        print()

        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True) if os.path.dirname(report_path) else None
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)

        return result
    
    def plot_bit_exact_columns(
        self,
        original_file: str,
        decompressed_file: str,
        num_cols: int = 4,
        dtype: str = "float32",
        max_rows: int = 2000,
        savepath: str = "bit_exact_columns.png",
    ):
        """
        Load two binary files (original and decompressed), interpret them as a matrix of the given dtype
        with num_cols columns, and overlay the first few columns to visually confirm bit-exactness.

        Notes:
        - Assumes row-major layout of fixed-size records with `num_cols` fields of the given dtype.
        - If the total element count isn't divisible by `num_cols`, the last partial row is dropped.
        - If arrays are identical at the byte level, curves will overlap perfectly; otherwise, a
          difference panel is shown per column.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if not os.path.exists(original_file):
            raise FileNotFoundError(f"Original file not found: {original_file}")
        if not os.path.exists(decompressed_file):
            raise FileNotFoundError(f"Decompressed file not found: {decompressed_file}")

        # Load as raw bytes then view as the requested dtype
        a = np.fromfile(original_file, dtype=dtype)
        b = np.fromfile(decompressed_file, dtype=dtype)

        n_elems = min(a.size, b.size)
        if a.size != b.size:
            print(f"[WARN] Element counts differ: original={a.size}, decompressed={b.size}. Truncating to {n_elems}.")
        if n_elems < num_cols:
            raise ValueError(f"Not enough elements ({n_elems}) to form even one row with num_cols={num_cols}")

        # Reshape into rows x cols, dropping any incomplete tail
        n_rows = (n_elems // num_cols)
        a_mat = a[: n_rows * num_cols].reshape(n_rows, num_cols)
        b_mat = b[: n_rows * num_cols].reshape(n_rows, num_cols)

        rows_to_plot = min(max_rows, n_rows)

        # Check bit-exact equality for the plotted slice
        bit_exact = np.array_equal(a_mat[:rows_to_plot], b_mat[:rows_to_plot])

        # Styling for paper-friendly figure
        plt.rcParams.update({
            "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
            "legend.fontsize": 12, "lines.linewidth": 1.5,
        })

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.15})
        ax_main, ax_diff = axes

        x = np.arange(rows_to_plot)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for j in range(min(num_cols, a_mat.shape[1])):
            ax_main.plot(x, a_mat[:rows_to_plot, j], color=colors[j % len(colors)], alpha=0.9, label=f"col{j} original")
            ax_main.plot(x, b_mat[:rows_to_plot, j], color=colors[j % len(colors)], alpha=0.5, linestyle='--', label=f"col{j} decompressed")

            diff = (b_mat[:rows_to_plot, j] - a_mat[:rows_to_plot, j])
            ax_diff.plot(x, diff, color=colors[j % len(colors)], alpha=0.8)

        title_note = "BIT-EXACT" if bit_exact else "MISMATCH"
        ax_main.set_title(f"Original vs Decompressed (first {rows_to_plot} rows, {num_cols} cols) — {title_note}")
        ax_main.set_ylabel("Value")
        ax_main.grid(True, alpha=0.25)
        # Make a compact legend
        handles, labels = ax_main.get_legend_handles_labels()
        if len(handles) > 0:
            # show at most one pair per column in legend: collapse duplicates
            seen = set()
            filt_handles, filt_labels = [], []
            for h, lab in zip(handles, labels):
                colkey = lab.split()[0]  # "col{j}"
                if colkey not in seen:
                    seen.add(colkey)
                    filt_handles.append(h)
                    filt_labels.append(colkey)
            ax_main.legend(filt_handles, filt_labels, ncol=min(num_cols, 4), frameon=False, loc="upper right")

        ax_diff.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0)
        ax_diff.set_xlabel("Row index")
        ax_diff.set_ylabel("Δ (dec - orig)")
        ax_diff.grid(True, alpha=0.25)

        plt.tight_layout()
        os.makedirs(os.path.dirname(savepath), exist_ok=True) if os.path.dirname(savepath) else None
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()
        return {"bit_exact": bool(bit_exact), "rows_plotted": int(rows_to_plot), "num_cols": int(num_cols)}
        
    @torch.no_grad()
    def evaluate_bpp(self, loader):
        """Calculate bits per byte on a dataloader"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(loader, desc="Evaluating BPP"):
            batch = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        
        mean_nll = total_loss / max(1, total_tokens)
        bpp = mean_nll / np.log(2)
        return bpp
    
    @torch.no_grad()
    def collect_predictions(self, loader, max_batches=50):
        """Collect predictions for analysis"""
        self.model.eval()
        
        all_logits = []
        all_targets = []
        all_probs = []
        
        for i, batch in enumerate(tqdm(loader, desc="Collecting predictions", total=max_batches)):
            if i >= max_batches:
                break
                
            batch = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())
        
        return {
            'logits': torch.cat(all_logits, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'probs': torch.cat(all_probs, dim=0)
        }
    
    def plot_calibration_curves_multi(
        self,
        split_loaders: dict,
        n_bins: int = 20,
        max_batches: int = 50,
        savepath: str = "calibration_all.png",
        ignore_index=None,
        quantile_bins: bool = False,
    ):
        """
        Plot reliability curves for multiple splits on the same figure.
        split_loaders: {"train": train_loader, "val": val_loader, "test": test_loader}
        Returns: dict {split: {"ece": float, "counts": np.array, "mean_conf": np.array, "acc": np.array}}
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self.model.eval()
        curves = {}

        # Collect confidences/accuracy for each split
        split_conf = {}
        split_corr = {}
        for name, loader in split_loaders.items():
            preds = self.collect_predictions(loader, max_batches=max_batches)
            probs = preds["probs"]           # [B, L, K]
            targets = preds["targets"]       # [B, L]

            # Optionally drop padding tokens
            if ignore_index is not None:
                mask = (targets != ignore_index)
            else:
                mask = torch.ones_like(targets, dtype=torch.bool)

            conf = probs.max(dim=-1).values[mask]           # confidence of predicted class
            pred = probs.argmax(dim=-1)[mask]
            targ = targets[mask]
            corr = (pred == targ).float()

            # to numpy
            split_conf[name] = conf.detach().cpu().numpy().ravel()
            split_corr[name] = corr.detach().cpu().numpy().ravel()

        # Make common bins (either uniform or quantiles based on ALL splits together)
        all_conf = np.concatenate(list(split_conf.values())) if len(split_conf) else np.array([0.0])
        if quantile_bins and all_conf.size > 0:
            qs = np.linspace(0, 1, n_bins + 1)
            bins = np.unique(np.quantile(all_conf, qs))
            # ensure at least 2 edges
            if bins.size < 2:
                bins = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            bins = np.linspace(0.0, 1.0, n_bins + 1)

        # Compute per-split curves + ECE (weighted)
        # --- styling for double-column papers ---
        plt.rcParams.update({
            "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
            "legend.fontsize": 11, "lines.linewidth": 2.0,
        })

        palette = {"train": "#1f77b4", "val": "#2ca02c", "test": "#d62728"}
        markers = {"train": "o", "val": "s", "test": "^"}

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(7.0, 4.6),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.2}
        )

        # Perfect calibration line (we’ll add to legend manually)
        ax_top.plot([0, 1], [0, 1], "--", color="#7f7f7f", alpha=0.6)

        max_abs_resid = 0.0
        ece_map = {}

        for name in split_conf:
            conf_np = split_conf[name]
            corr_np = split_corr[name]

            inds = np.digitize(conf_np, bins, right=True) - 1
            inds = np.clip(inds, 0, len(bins) - 2)

            counts   = np.bincount(inds, minlength=len(bins) - 1).astype(float)
            sum_conf = np.bincount(inds, weights=conf_np, minlength=len(bins) - 1)
            sum_corr = np.bincount(inds, weights=corr_np, minlength=len(bins) - 1)

            with np.errstate(divide="ignore", invalid="ignore"):
                mean_conf = np.divide(sum_conf, counts, out=np.zeros_like(sum_conf), where=counts > 0)
                acc       = np.divide(sum_corr, counts, out=np.zeros_like(sum_corr), where=counts > 0)

            total = counts.sum() if counts.sum() > 0 else 1.0
            ece = float(np.sum(np.abs(acc - mean_conf) * counts) / total)
            ece_map[name] = ece

            nonempty = counts > 0
            color = palette.get(name, None)
            marker = markers.get(name, "o")

            # Top: reliability curve
            ax_top.plot(
                mean_conf[nonempty], acc[nonempty],
                marker=marker, color=color, markersize=4, linewidth=1.0
            )

            # Bottom: residuals Δ = acc − conf
            resid = acc - mean_conf
            max_abs_resid = max(max_abs_resid, float(np.nanmax(np.abs(resid[nonempty]))))
            ax_bot.plot(
                mean_conf[nonempty], resid[nonempty],
                marker=marker, color=color, markersize=4, linewidth=1
            )

        # Cosmetics
        ax_top.set_ylabel("Accuracy")
        ax_top.grid(True, alpha=0.25, linewidth=0.7)

        ax_bot.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.2)
        ax_bot.set_xlabel("Mean predicted confidence")
        ax_bot.set_ylabel(r"$\Delta$")
        ax_bot.grid(True, alpha=0.25, linewidth=1)
        ax_bot.ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
        from matplotlib.ticker import ScalarFormatter

        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((-2, -2))          # force ×10^{-2}
        ax_bot.yaxis.set_major_formatter(sf)
        ax_bot.yaxis.get_offset_text().set_va('bottom')  # optional: tweak offset text position
        # Symmetric y-limits for residuals
        r = max(0.02, min(0.25, 1.1 * max_abs_resid))
        ax_bot.set_ylim(-r, r)
        # ax_bot.set_ylim(-2.5e-2, 2.5e-2)

        # Clean legend with markers + ECE per split
        legend_handles = [
            Line2D([0], [0], linestyle="--", color="#7f7f7f", lw=1.0, label="Perfect calibration")
        ]
        for name in ["train", "val", "test"]:
            if name in ece_map:
                legend_handles.append(
                    Line2D(
                        [0], [0], linestyle="-", color=palette[name], lw=1.0,
                        marker=markers[name], markersize=4,
                        label=f"{name.capitalize()} (ECE={ece_map[name]:.3f})"
                    )
                )
        ax_top.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper left")

        plt.tight_layout()
        plt.savefig(savepath, dpi=600, bbox_inches="tight")                 # PNG
        # plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches="tight")  # PDF
        plt.show()


        return curves
    @torch.inference_mode()
    def plot_topk_accuracy(
        self,
        loader,
        k_max: int = 20,              # up to 256 for bytes
        step: int = 1,                # plot every 'step' (e.g., 1,2,3,...)
        ignore_index=None,
        savepath: str = "top_k_accuracy.png",
        annotate_ks=(1, 5, 10),       # which K's to label on the curve
    ):
        """
        Plots Top-K accuracy vs K for a single split and returns a dict:
        {"k": np.array, "topk_acc": np.array}
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self.model.eval()
        all_hits_cum = None
        total_N = 0

        for batch in tqdm(loader, desc=f"Top-{k_max} accuracy"):
            batch = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:].long()             # [B, L]

            logits = self.model(x)               # [B, L, C]
            C = logits.size(-1)

            # mask padding if provided
            if ignore_index is not None:
                mask = (y != ignore_index)
            else:
                mask = torch.ones_like(y, dtype=torch.bool)

            # flatten masked tokens
            y_flat = y[mask]                    # [N]
            if y_flat.numel() == 0:
                continue
            logits_flat = logits[mask]          # [N, C]

            # get top-k_max indices once
            k_eff = min(k_max, C)
            top_idx = torch.topk(logits_flat, k=k_eff, dim=-1).indices  # [N, k_eff]

            # membership matrix: 1 if target in top-j (for each j), cumulative along j
            hit = (top_idx == y_flat.unsqueeze(1)).float()              # [N, k_eff]
            hit_cum = hit.cumsum(dim=1).clamp(max=1)                    # [N, k_eff]

            # accumulate sums to average later
            if all_hits_cum is None:
                all_hits_cum = hit_cum.sum(dim=0)      # [k_eff]
            else:
                all_hits_cum += hit_cum.sum(dim=0)
            total_N += y_flat.size(0)

        # final accuracies per K
        topk_acc = (all_hits_cum / max(1, total_N)).cpu().numpy() if all_hits_cum is not None else np.zeros(k_eff)
        ks = np.arange(1, len(topk_acc) + 1)

        # --- Styling for a paper-friendly plot ---
        plt.rcParams.update({
            "font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20,
            "legend.fontsize": 20, "lines.linewidth": 3.0,
        })

        plt.figure(figsize=(10, 7))
        sel = (ks % step == 0)
        plt.plot(ks[sel], topk_acc[sel], marker="o", markersize=6)

        # annotate a few K's
        for k in annotate_ks:
            if 1 <= k <= len(topk_acc):
                plt.annotate(f"{topk_acc[k-1]:.3f}", (k, topk_acc[k-1]),
                            textcoords="offset points", xytext=(0, 8), ha="center")

        plt.xlabel("k")
        plt.ylabel("Top-k Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()

        return {"k": ks, "topk_acc": topk_acc}


    def plot_confusion_top_bytes(
        self,
        loader,
        top_n: int = 20,                 # how many most frequent true bytes to show
        ignore_index=None,               # padding label, if any
        normalize: str = "true",         # {"none","true","pred","all"}
        savepath: str = "confusion_top_bytes.png",
    ):
        """
        Compute confusion matrix for the Top-N most frequent *true* bytes and plot it.

        normalize:
        - "none": raw counts
        - "true": row-normalised (per-true-class recall)  [DEFAULT]
        - "pred": column-normalised (per-pred-class precision)
        - "all":  divided by total count
        Returns:
        dict with:
            - "classes": list of byte values shown (length top_n)
            - "conf_mat": numpy array [top_n, top_n] (normalised per 'normalize')
            - "counts": raw counts (same shape) for reference
            - "per_class": dict with precision/recall/f1 for those classes
        """
        self.model.eval()
        device = self.device

        C = 256
        total_counts = torch.zeros(C, dtype=torch.long, device="cpu")       # true label counts
        # 256x256 confusion counts
        conf_full = torch.zeros(C, C, dtype=torch.long, device="cpu")

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                x = batch[:, :-1]
                y = batch[:, 1:].long()                   # [B, L-1]
                logits = self.model(x)                    # [B, L-1, 256]
                pred = logits.argmax(dim=-1)              # [B, L-1]

                if ignore_index is not None:
                    mask = (y != ignore_index)
                    if mask.sum() == 0:
                        continue
                    y = y[mask]
                    pred = pred[mask]
                else:
                    y = y.reshape(-1)
                    pred = pred.reshape(-1)

                # counts of true labels
                total_counts += torch.bincount(y.cpu(), minlength=C)

                # confusion via bincount on pairs
                idx = (y * C + pred).cpu()
                conf_counts = torch.bincount(idx, minlength=C*C).reshape(C, C)
                conf_full += conf_counts

        # pick top-N by true frequency
        top_n = min(top_n, C)
        top_classes = torch.topk(total_counts, k=top_n).indices.cpu().numpy()
        top_classes_sorted = top_classes[np.argsort(top_classes)]  # optional: sort by byte value
        # (If you'd rather sort rows by frequency descending, use indices from topk directly.)

        # slice rows/cols to top classes
        counts_top = conf_full[top_classes_sorted][:, top_classes_sorted].cpu().numpy().astype(np.float64)

        # normalisation
        counts = counts_top.copy()
        if normalize == "true":            # rows sum to 1
            row_sums = counts.sum(axis=1, keepdims=True)
            conf_norm = np.divide(counts, np.maximum(row_sums, 1), where=row_sums>0)
        elif normalize == "pred":          # cols sum to 1
            col_sums = counts.sum(axis=0, keepdims=True)
            conf_norm = np.divide(counts, np.maximum(col_sums, 1), where=col_sums>0)
        elif normalize == "all":
            total = counts.sum()
            conf_norm = counts / max(total, 1.0)
        elif normalize == "false":                              # "none"
            conf_norm = counts

        # per-class metrics (computed on the full confusion but reported for top classes)
        cf = conf_full.numpy().astype(np.float64)
        tp = np.diag(cf)
        pred_tot = cf.sum(axis=0)     # predicted positives (by column)
        true_tot = cf.sum(axis=1)     # actual positives   (by row)

        precision = np.divide(tp, np.maximum(pred_tot, 1.0))
        recall    = np.divide(tp, np.maximum(true_tot, 1.0))
        f1        = np.divide(2*precision*recall, np.maximum(precision+recall, 1e-12))

        # gather for selected classes
        per_class = {
            int(c): {
                "support": int(true_tot[c]),
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
            } for c in top_classes_sorted
        }

        # -------- plot --------
        plt.rcParams.update({
            "font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20,
            "legend.fontsize": 20, "lines.linewidth": 2,
        })
        fig, ax = plt.subplots(figsize=(min(10, 1.0 + 0.45*top_n), min(8, 1.0 + 0.45*top_n)))

        im = ax.imshow(conf_norm, interpolation="nearest", cmap="magma")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label({"true":"Row-normalised","pred":"Col-normalised","all":"Global","false":"Counts"}[normalize])

        # tick labels as byte values
        xt = [str(c) for c in top_classes_sorted]
        yt = [str(c) for c in top_classes_sorted]
        ax.set_xticks(np.arange(top_n), labels=xt, rotation=90)
        ax.set_yticks(np.arange(top_n), labels=yt)
        ax.set_xlabel("Predicted byte")
        ax.set_ylabel("True byte")

        # optional: annotate diagonal with accuracy per class if row-normalised
        # if normalize == "true":
        #     diag = np.diag(conf_norm)
        #     for i in range(top_n):
        #         ax.text(i, i, f"{diag[i]:.2f}", ha="center", va="center", color="white" if diag[i] < 0.6 else "black", fontsize=8)

        plt.tight_layout()
        plt.savefig(savepath, dpi=600, bbox_inches="tight")
        # plt.savefig(savepath.replace(".png", ".pdf"), bbox_inches="tight")
        plt.show()

        return {
            "classes": top_classes_sorted.tolist(),
            "conf_mat": conf_norm,
            "counts": counts_top,
            "per_class": per_class,
        }
