#!/usr/bin/env python3
"""
N1 Threshold Calibration Analysis.

Re-runs inference from best_model.pt for each completed experiment, collects
per-class softmax probabilities, then sweeps the N1 decision threshold to find
the precision/recall/F1 trade-off.

Default prediction: argmax(stage_logits)  →  high recall, low precision for N1
With threshold T:   predict N1 if P(N1) > T, else predict best non-N1 class

Usage:
    python scripts/analyze_n1_threshold.py --results-dir results_2026-03-03

    # Also works on a specific fold/experiment subset:
    python scripts/analyze_n1_threshold.py --results-dir results_2026-03-03 \\
        --experiments final_tasa no_transitions --folds 0

Output (inside results-dir):
    N1_thresholding_{timestamp}/
        threshold_results.json          — full threshold sweep data per experiment
        best_thresholds_summary.json    — optimal threshold + metrics per experiment
        {fold}_{exp}_threshold_curve.png — 3-panel figure per model
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.configurable import ConfigurableTASA, ContextTASA
from src.training.loops import get_device
from src.config import PROCESSED_DIR

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
CHANNEL_INDICES = {1: [0], 2: [0, 2], 3: [0, 1, 2]}


# ---------------------------------------------------------------------------
# Minimal inference datasets (no sampler, sequential order)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """Single-epoch inference dataset for ConfigurableTASA models."""

    def __init__(self, metadata_path, data_dir, subject_ids, num_channels):
        self.data_dir = Path(data_dir)
        self.ch_idx = CHANNEL_INDICES[num_channels]

        meta = pd.read_csv(metadata_path)
        meta = meta[meta["subject_id"].isin(subject_ids)].reset_index(drop=True)
        meta["epoch_index"] = meta["epoch_index"].astype(int)
        meta = meta.sort_values(["subject_id", "epoch_index"]).reset_index(drop=True)
        self.meta = meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        signal = np.load(self.data_dir / row["filename"])[self.ch_idx, :]
        signal = torch.from_numpy(signal).float()
        mean = signal.mean(dim=1, keepdim=True)
        std  = signal.std(dim=1, keepdim=True)
        signal = torch.clamp((signal - mean) / (std + 1e-6), -20.0, 20.0)
        return {"signal": signal, "stage_label": int(row["stage_label"])}


class ContextInferenceDataset(Dataset):
    """Context-window inference dataset for ContextTASA models."""

    def __init__(self, metadata_path, data_dir, subject_ids, num_channels, context_window):
        self.data_dir = Path(data_dir)
        self.ch_idx = CHANNEL_INDICES[num_channels]
        self.context_window = context_window

        meta = pd.read_csv(metadata_path)
        meta = meta[meta["subject_id"].isin(subject_ids)].reset_index(drop=True)
        meta["epoch_index"] = meta["epoch_index"].astype(int)
        meta = meta.sort_values(["subject_id", "epoch_index"]).reset_index(drop=True)
        self.meta = meta

        self._subj_bounds = {}
        for subj, grp in self.meta.groupby("subject_id", sort=False):
            self._subj_bounds[subj] = (int(grp.index.min()), int(grp.index.max()))

    def __len__(self):
        return len(self.meta)

    def _load_epoch(self, iloc_pos):
        row = self.meta.iloc[iloc_pos]
        sig = np.load(self.data_dir / row["filename"])[self.ch_idx, :]
        sig = torch.from_numpy(sig).float()
        mean = sig.mean(dim=1, keepdim=True)
        std  = sig.std(dim=1, keepdim=True)
        return torch.clamp((sig - mean) / (std + 1e-6), -20.0, 20.0)

    def __getitem__(self, idx):
        row  = self.meta.iloc[idx]
        subj = row["subject_id"]
        lo, hi = self._subj_bounds[subj]
        signals = []
        for offset in range(-self.context_window, self.context_window + 1):
            context_pos = max(lo, min(hi, idx + offset))
            signals.append(self._load_epoch(context_pos))
        return {
            "signal":      torch.stack(signals, dim=0),  # (2K+1, C, T)
            "stage_label": int(row["stage_label"]),
        }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_stage_probs(model, loader, device):
    """Return (N×5 softmax probs, N true labels)."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", leave=False):
            sigs   = batch["signal"].to(device)
            labels = batch["stage_label"]
            out    = model(sigs)
            probs  = torch.softmax(out["stage_logits"], dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.vstack(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------

def apply_n1_threshold(probs, threshold):
    """
    For each sample:
      - If P(N1) > threshold → predict N1
      - Else → predict argmax over all 5 classes (may still be N1 if N1 < threshold
        but N1 is still highest overall — this is intentional; we only suppress N1
        by raising the threshold, not by forcing non-N1 when P(N1) wins)

    This mirrors the way thresholding is used in practice: you raise the bar for
    predicting a class to reduce false positives.  At T=argmax-equivalent (~0.2),
    results match the plain argmax; lower T → more N1; higher T → less N1.
    """
    n1_probs  = probs[:, 1]
    base_preds = probs.argmax(axis=1).copy()

    # Force N1 for samples where P(N1) > T (may already be argmax, or may not)
    preds = base_preds.copy()
    preds[n1_probs > threshold] = 1

    # For samples where P(N1) <= T AND argmax would have chosen N1: pick next best
    was_argmax_n1 = (base_preds == 1) & (n1_probs <= threshold)
    if was_argmax_n1.any():
        non_n1 = probs[was_argmax_n1].copy()
        non_n1[:, 1] = -1.0          # mask N1 column
        preds[was_argmax_n1] = non_n1.argmax(axis=1)

    return preds


def n1_metrics(labels, preds):
    n1_true = labels == 1
    n1_pred = preds  == 1
    tp = int((n1_true & n1_pred).sum())
    fp = int((~n1_true & n1_pred).sum())
    fn = int((n1_true & ~n1_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def threshold_sweep(probs, labels, thresholds):
    rows = []
    for T in thresholds:
        preds = apply_n1_threshold(probs, T)
        p, r, f1 = n1_metrics(labels, preds)
        acc = float((preds == labels).mean())
        try:
            kappa = float(cohen_kappa_score(labels, preds))
        except Exception:
            kappa = 0.0
        rows.append({
            "threshold":       float(T),
            "n1_precision":    p,
            "n1_recall":       r,
            "n1_f1":           f1,
            "accuracy":        acc,
            "kappa":           kappa,
            "composite":       kappa + 0.3 * r,
            "n1_pred_count":   int((preds == 1).sum()),
        })
    return rows


def argmax_metrics(probs, labels):
    """Metrics for plain argmax — the current default."""
    preds = probs.argmax(axis=1)
    p, r, f1 = n1_metrics(labels, preds)
    acc = float((preds == labels).mean())
    try:
        kappa = float(cohen_kappa_score(labels, preds))
    except Exception:
        kappa = 0.0
    # Effective threshold ≈ min N1 probability where argmax = N1
    n1_win_probs = probs[preds == 1, 1]
    eff_T = float(n1_win_probs.min()) if len(n1_win_probs) > 0 else 0.2
    return {
        "precision": p, "recall": r, "f1": f1,
        "accuracy": acc, "kappa": kappa,
        "effective_threshold": eff_T,
        "n1_pred_count": int((preds == 1).sum()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(sweep_data, default, fold_name, exp_name, output_dir):
    thresholds = np.array([d["threshold"]    for d in sweep_data])
    precisions = np.array([d["n1_precision"] for d in sweep_data])
    recalls    = np.array([d["n1_recall"]    for d in sweep_data])
    f1s        = np.array([d["n1_f1"]        for d in sweep_data])
    kappas     = np.array([d["kappa"]        for d in sweep_data])

    best_idx = int(np.argmax(f1s))
    best_T   = thresholds[best_idx]
    best_f1  = f1s[best_idx]
    best_p   = precisions[best_idx]
    best_r   = recalls[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f"{fold_name} / {exp_name}  —  N1 Threshold Calibration"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: Precision / Recall / F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, precisions, color="steelblue",  lw=2, label="N1 Precision")
    ax.plot(thresholds, recalls,    color="darkorange", lw=2, label="N1 Recall")
    ax.plot(thresholds, f1s,        color="seagreen",   lw=2, label="N1 F1")
    ax.axvline(best_T, color="seagreen", ls="--", alpha=0.8,
               label=f"Best F1 T={best_T:.2f} (F1={best_f1:.3f})")
    ax.axvline(default["effective_threshold"], color="gray", ls=":",
               alpha=0.7, label=f"Argmax ≈ T={default['effective_threshold']:.2f}")
    ax.axhline(default["f1"],        color="seagreen",   ls=":", alpha=0.5)
    ax.axhline(default["precision"], color="steelblue",  ls=":", alpha=0.5)
    ax.axhline(default["recall"],    color="darkorange", ls=":", alpha=0.5)
    ax.set_xlabel("N1 Decision Threshold (T)")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Panel 2: Precision-Recall curve (parameterised by T)
    ax = axes[1]
    sc = ax.scatter(recalls, precisions, c=thresholds, cmap="viridis",
                    s=25, zorder=3, alpha=0.8)
    ax.plot(recalls, precisions, color="gray", alpha=0.3, lw=1)
    ax.scatter([best_r], [best_p], color="seagreen", s=160, zorder=5,
               marker="*", label=f"Best F1={best_f1:.3f} (T={best_T:.2f})")
    ax.scatter([default["recall"]], [default["precision"]], color="red",
               s=100, zorder=5, marker="D", label=f"Argmax  F1={default['f1']:.3f}")
    # F1 iso-curves
    for iso_f1 in [0.40, 0.45, 0.50, 0.55, 0.60]:
        r_pts = np.linspace(0.01, 1.0, 200)
        p_pts = iso_f1 * r_pts / (2*r_pts - iso_f1)
        mask  = (p_pts >= 0) & (p_pts <= 1)
        ax.plot(r_pts[mask], p_pts[mask], "--", color="lightgray", lw=0.8, alpha=0.7)
        idx = np.where(mask)[0]
        if len(idx):
            ax.annotate(f"F1={iso_f1}", xy=(r_pts[idx[-1]], p_pts[idx[-1]]),
                        fontsize=7, color="gray")
    plt.colorbar(sc, ax=ax, label="Threshold")
    ax.set_xlabel("N1 Recall")
    ax.set_ylabel("N1 Precision")
    ax.set_title("N1 Precision-Recall Curve")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Panel 3: Overall kappa vs threshold
    ax = axes[2]
    ax.plot(thresholds, kappas, color="mediumpurple", lw=2)
    ax.axvline(best_T, color="seagreen", ls="--", alpha=0.8,
               label=f"Best N1-F1 T={best_T:.2f}")
    ax.axvline(default["effective_threshold"], color="gray", ls=":", alpha=0.7,
               label="Argmax")
    ax.axhline(default["kappa"], color="mediumpurple", ls=":", alpha=0.5,
               label=f"Argmax kappa={default['kappa']:.4f}")
    ax.set_xlabel("N1 Decision Threshold (T)")
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("Overall Kappa vs N1 Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Footer annotation
    fig.text(
        0.5, 0.01,
        f"Default (argmax):  precision={default['precision']:.3f}  "
        f"recall={default['recall']:.3f}  F1={default['f1']:.3f}  "
        f"kappa={default['kappa']:.4f}     |     "
        f"Optimal T={best_T:.2f}:  precision={best_p:.3f}  "
        f"recall={best_r:.3f}  F1={best_f1:.3f}",
        ha="center", fontsize=8.5, style="italic", color="#444444",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = output_dir / f"{fold_name}_{exp_name}_threshold_curve.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname, best_T, best_f1, best_p, best_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="N1 threshold calibration analysis for completed CV experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results_2026-03-03"),
        help="Top-level CV output directory"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Limit to specific experiment names (default: all completed)"
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Limit to specific fold indices, e.g. --folds 0 1"
    )
    parser.add_argument(
        "--n-thresholds", type=int, default=99,
        help="Number of threshold values to sweep between 0.01 and 0.99 (default: 99)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        sys.exit(1)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir / f"N1_thresholding_{timestamp}"
    output_dir.mkdir(parents=True)

    print(f"\n{'='*65}")
    print(f"N1 Threshold Calibration Analysis")
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"{'='*65}\n")

    metadata_path = PROCESSED_DIR / "metadata.csv"
    device        = get_device()
    thresholds    = np.linspace(0.01, 0.99, args.n_thresholds)

    # Discover completed experiments
    completed = []
    for fold_dir in sorted(results_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold_idx = int(fold_dir.name.split("_")[1])
        if args.folds is not None and fold_idx not in args.folds:
            continue
        for exp_dir in sorted(fold_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if args.experiments and exp_dir.name not in args.experiments:
                continue
            if (exp_dir / "results.json").exists() and (exp_dir / "best_model.pt").exists():
                completed.append((fold_dir.name, exp_dir.name, exp_dir))

    if not completed:
        print("No completed experiments found. Run some training first.")
        sys.exit(0)

    print(f"Found {len(completed)} completed experiment(s):\n")
    for fn, en, _ in completed:
        print(f"  {fn}/{en}")
    print()

    all_results  = {}
    summary_rows = []

    for fold_name, exp_name, exp_dir in completed:
        print(f"{'─'*55}")
        print(f"  {fold_name} / {exp_name}")

        with open(exp_dir / "results.json") as f:
            res = json.load(f)

        channels       = res["config"]["channels"]
        window         = res["config"]["window"]
        alpha          = res["config"]["alpha"]
        context_window = res["config"].get("context_window", 0)
        test_subjects  = res["subjects"]["test"]

        # Test dataset (sequential, no sampler)
        if context_window > 0:
            test_ds = ContextInferenceDataset(
                metadata_path, PROCESSED_DIR, test_subjects, channels, context_window
            )
        else:
            test_ds = InferenceDataset(
                metadata_path, PROCESSED_DIR, test_subjects, channels
            )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Load model
        if context_window > 0:
            model = ContextTASA(
                input_channels=channels,
                d_model=64, n_layers=4, n_heads=4,
                window_size=window, num_classes=5,
                context_window=context_window,
            ).to(device)
        else:
            model = ConfigurableTASA(
                input_channels=channels,
                d_model=64, n_layers=4, n_heads=4,
                window_size=window, num_classes=5,
            ).to(device)
        ckpt = torch.load(
            exp_dir / "best_model.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])

        # Inference
        probs, labels = get_stage_probs(model, test_loader, device)

        # Default (argmax) metrics — sanity-check against saved results.json
        default = argmax_metrics(probs, labels)
        print(f"  Argmax check  → acc={default['accuracy']*100:.2f}%  "
              f"kappa={default['kappa']:.4f}  "
              f"N1: P={default['precision']:.3f} R={default['recall']:.3f} "
              f"F1={default['f1']:.3f}")
        saved_acc = res["test_accuracy"]
        if abs(default["accuracy"] - saved_acc) > 0.005:
            print(f"  ⚠ Accuracy mismatch vs results.json ({saved_acc:.4f}) — "
                  f"check model/data loading")

        # Threshold sweep
        sweep = threshold_sweep(probs, labels, thresholds)

        # Plot
        fig_path, best_T, best_f1, best_p, best_r = make_figure(
            sweep, default, fold_name, exp_name, output_dir
        )

        f1_gain = best_f1 - default["f1"]
        print(f"  Optimal T={best_T:.2f} → "
              f"P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.3f}  "
              f"(ΔF1={f1_gain:+.3f})")
        print(f"  Figure → {fig_path.name}")

        key = f"{fold_name}/{exp_name}"
        all_results[key] = {
            "fold":       fold_name,
            "experiment": exp_name,
            "config":     {"channels": channels, "window": window, "alpha": alpha},
            "n_test":     len(labels),
            "argmax":     default,
            "optimal": {
                "threshold":    float(best_T),
                "n1_f1":        float(best_f1),
                "n1_precision": float(best_p),
                "n1_recall":    float(best_r),
                "f1_gain":      float(f1_gain),
            },
            "sweep": sweep,
        }

        summary_rows.append({
            "experiment":         exp_name,
            "fold":               fold_name,
            "alpha":              alpha,
            "argmax_n1_precision": default["precision"],
            "argmax_n1_recall":    default["recall"],
            "argmax_n1_f1":        default["f1"],
            "argmax_kappa":        default["kappa"],
            "optimal_threshold":   float(best_T),
            "optimal_n1_f1":       float(best_f1),
            "optimal_n1_precision": float(best_p),
            "optimal_n1_recall":   float(best_r),
            "f1_gain":             float(f1_gain),
        })

    # Save outputs
    with open(output_dir / "threshold_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "best_thresholds_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY  —  Default (argmax) vs Optimal N1 Threshold")
    print(f"{'='*90}")
    hdr = f"{'Experiment':<28} {'Def P':>6} {'Def R':>6} {'Def F1':>7} {'Def κ':>7} │ {'Opt T':>6} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>7} {'ΔF1':>6}"
    print(hdr)
    print("─" * 90)
    for row in summary_rows:
        print(
            f"{row['fold']}/{row['experiment']:<22} "
            f"{row['argmax_n1_precision']:>6.3f} "
            f"{row['argmax_n1_recall']:>6.3f} "
            f"{row['argmax_n1_f1']:>7.3f} "
            f"{row['argmax_kappa']:>7.4f} │ "
            f"{row['optimal_threshold']:>6.2f} "
            f"{row['optimal_n1_precision']:>6.3f} "
            f"{row['optimal_n1_recall']:>6.3f} "
            f"{row['optimal_n1_f1']:>7.3f} "
            f"{row['f1_gain']:>+6.3f}"
        )

    print(f"\nSaved → {output_dir}/")
    print(f"  threshold_results.json      — full sweep data")
    print(f"  best_thresholds_summary.json — optimal threshold per model")
    print(f"  *_threshold_curve.png        — figures")


if __name__ == "__main__":
    main()
