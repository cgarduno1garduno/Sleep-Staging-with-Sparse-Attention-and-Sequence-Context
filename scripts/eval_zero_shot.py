#!/usr/bin/env python3
"""
Study 03 — Zero-shot cross-dataset evaluation on ISRUC-Sleep Subgroup 1.

Loads the best_model.pt checkpoint from each of the 12 trained experiments
(9 from Study 1, 3 from Study 2) and evaluates it on all available ISRUC
subjects. Each of the 5 fold-checkpoints is evaluated independently on the
full ISRUC set; results are reported as mean ± std across folds.

This mirrors how within-dataset results were reported in Studies 1 and 2,
making the tables directly comparable.

Experiments evaluated:
  Study 1 (ConfigurableTASA, no sequence context):
    no_transitions, alpha_1_0, final_tasa, alpha_10_0,
    window_32, window_128, 1ch_FpzCz, 2ch_FpzCz_EOG, full_attention
  Study 2 (ContextTASA, 3-epoch context window):
    seq_context_k1, seq_context_k1_notrans, seq_context_k1_notrans_2ch

Output:
  study_03/results/isruc_eval_summary.json  — machine-readable results
  study_03/results/isruc_eval_summary.md    — markdown table for the paper
  study_03/results/fold_results/            — per-fold per-experiment JSON

Usage (from project root):
  python study_03/scripts/run_isruc_eval.py
  python study_03/scripts/run_isruc_eval.py --experiments seq_context_k1_notrans no_transitions
  python study_03/scripts/run_isruc_eval.py --folds 0 1 2    # subset of folds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "study_03"))

from src.config import (
    ISRUC_PROC_DIR, STUDY01_RESULTS, STUDY02_RESULTS, STUDY03_RESULTS,
    CHANNEL_INDICES,
)
from src.models.configurable import ConfigurableTASA, ContextTASA
from src.training.loops import get_device
from src.dataloading.isruc_dataset import ISRUCEvalDataset

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Each entry mirrors the config used in training. The "source" key tells us
# which study's results directory to look in for the checkpoint.
EXPERIMENTS = [
    # --- Study 1: ConfigurableTASA (no context) ---
    {"name": "no_transitions",          "source": "study_01", "channels": 3, "window": 64,   "alpha": 0.0, "context_window": 0},
    {"name": "alpha_1_0",               "source": "study_01", "channels": 3, "window": 64,   "alpha": 1.0, "context_window": 0},
    {"name": "final_tasa",              "source": "study_01", "channels": 3, "window": 64,   "alpha": 5.0, "context_window": 0},
    {"name": "alpha_10_0",              "source": "study_01", "channels": 3, "window": 64,   "alpha": 10.0,"context_window": 0},
    {"name": "window_32",               "source": "study_01", "channels": 3, "window": 32,   "alpha": 5.0, "context_window": 0},
    {"name": "window_128",              "source": "study_01", "channels": 3, "window": 128,  "alpha": 5.0, "context_window": 0},
    {"name": "1ch_FpzCz",               "source": "study_01", "channels": 1, "window": 64,   "alpha": 5.0, "context_window": 0},
    {"name": "2ch_FpzCz_EOG",           "source": "study_01", "channels": 2, "window": 64,   "alpha": 5.0, "context_window": 0},
    {"name": "full_attention",          "source": "study_01", "channels": 3, "window": None, "alpha": 5.0, "context_window": 0},
    # --- Study 2: ContextTASA (3-epoch context window) ---
    {"name": "seq_context_k1",               "source": "study_02", "channels": 3, "window": 64, "alpha": 5.0, "context_window": 1},
    {"name": "seq_context_k1_notrans",       "source": "study_02", "channels": 3, "window": 64, "alpha": 0.0, "context_window": 1},
    {"name": "seq_context_k1_notrans_2ch",   "source": "study_02", "channels": 2, "window": 64, "alpha": 0.0, "context_window": 1},
]

N_FOLDS = 5


def checkpoint_path(exp: dict, fold_idx: int) -> Path:
    base = STUDY01_RESULTS if exp["source"] == "study_01" else STUDY02_RESULTS
    return base / f"fold_{fold_idx}" / exp["name"] / "best_model.pt"


def build_model(exp: dict, device: torch.device) -> nn.Module:
    channels = exp["channels"]
    window   = exp["window"]
    ctx      = exp["context_window"]

    if ctx > 0:
        model = ContextTASA(
            input_channels=channels,
            d_model=64, n_layers=4, n_heads=4,
            window_size=window, num_classes=5, context_window=ctx,
        )
    else:
        model = ConfigurableTASA(
            input_channels=channels,
            d_model=64, n_layers=4, n_heads=4,
            window_size=window, num_classes=5,
        )
    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_path: Path,
    exp: dict,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Load one checkpoint and evaluate on the full ISRUC dataset."""

    model = build_model(exp, device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = ISRUCEvalDataset(
        proc_dir       = ISRUC_PROC_DIR,
        num_channels   = exp["channels"],
        context_window = exp["context_window"],
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            sigs   = batch["signal"].to(device)
            labels = batch["stage_label"]

            out    = model(sigs)
            preds  = out["stage_logits"].argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = float((all_preds == all_labels).mean())
    kappa  = float(cohen_kappa_score(all_labels, all_preds))
    report = classification_report(
        all_labels, all_preds,
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "n_epochs":   len(all_labels),
        "n_subjects": dataset.n_subjects,
        "accuracy":   acc,
        "kappa":      kappa,
        "n1_recall":  float(report["N1"]["recall"]),
        "n1_precision": float(report["N1"]["precision"]),
        "n1_f1":      float(report["N1"]["f1-score"]),
        "per_class":  {s: report[s] for s in STAGE_NAMES},
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_folds(fold_results: list) -> dict:
    """Compute mean ± std across fold results."""
    keys = ["accuracy", "kappa", "n1_recall", "n1_precision", "n1_f1"]
    agg  = {}
    for key in keys:
        vals = [r[key] for r in fold_results]
        agg[key] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "folds":  vals,
            "n_folds": len(vals),
        }
    return agg


def print_summary_table(summary: dict):
    print("\n" + "#" * 90)
    print("ISRUC CROSS-DATASET EVAL  (mean ± std across 5 fold-checkpoints)")
    print("#" * 90)
    print(f"\n{'Experiment':<30} {'Source':>8} {'Accuracy':>14} {'Kappa':>14} "
          f"{'N1 Recall':>12} {'N1-F1':>10} {'Folds':>6}")
    print("-" * 90)
    for exp_name, entry in summary.items():
        agg = entry["aggregate"]
        src = entry["source"]
        acc   = agg["accuracy"]
        kappa = agg["kappa"]
        n1r   = agg["n1_recall"]
        n1f1  = agg["n1_f1"]
        n     = acc["n_folds"]
        print(
            f"{exp_name:<30} {src:>8} "
            f"{acc['mean']*100:>6.2f}%±{acc['std']*100:.2f}  "
            f"{kappa['mean']:>6.4f}±{kappa['std']:.4f}  "
            f"{n1r['mean']:>5.3f}±{n1r['std']:.3f}  "
            f"{n1f1['mean']:>5.3f}±{n1f1['std']:.3f}  "
            f"{n:>6}"
        )


def write_markdown_table(summary: dict, out_path: Path):
    lines = [
        "# Study 03 — ISRUC Cross-Dataset Evaluation",
        "",
        "Zero-shot transfer: models trained on Sleep-EDF-78, evaluated on ISRUC Subgroup 1 (100 subjects).",
        "Each row is **mean ± std across the 5 fold-checkpoints** evaluated on all ISRUC subjects.",
        "",
        "| Experiment | Source | Accuracy | κ | N1-F1 | N1-Recall | Folds |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp_name, entry in summary.items():
        agg = entry["aggregate"]
        src = entry["source"]
        acc   = agg["accuracy"]
        kappa = agg["kappa"]
        n1f1  = agg["n1_f1"]
        n1r   = agg["n1_recall"]
        n     = acc["n_folds"]
        lines.append(
            f"| `{exp_name}` | {src} "
            f"| {acc['mean']*100:.2f}%±{acc['std']*100:.2f} "
            f"| {kappa['mean']:.4f}±{kappa['std']:.4f} "
            f"| {n1f1['mean']:.3f}±{n1f1['std']:.3f} "
            f"| {n1r['mean']:.3f}±{n1r['std']:.3f} "
            f"| {n} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot cross-dataset evaluation on ISRUC Subgroup 1"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Evaluate only these experiments (default: all 12)"
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Evaluate only these fold indices (default: 0-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="DataLoader batch size (default: 64)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-evaluate even if fold result JSON already exists"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device    = get_device()
    fold_ids  = args.folds if args.folds is not None else list(range(N_FOLDS))
    exps      = EXPERIMENTS
    if args.experiments:
        exps = [e for e in EXPERIMENTS if e["name"] in args.experiments]

    STUDY03_RESULTS.mkdir(parents=True, exist_ok=True)
    fold_results_dir = STUDY03_RESULTS / "fold_results"
    fold_results_dir.mkdir(exist_ok=True)

    # Verify preprocessed data exists
    if not ISRUC_PROC_DIR.exists() or not any(ISRUC_PROC_DIR.glob("isruc_S*.npz")):
        print(
            f"ERROR: No preprocessed ISRUC data found in {ISRUC_PROC_DIR}.\n"
            "Run:  python study_03/scripts/preprocess_isruc.py"
        )
        sys.exit(1)

    print(f"\n{'#'*70}")
    print("Study 03 — ISRUC Cross-Dataset Evaluation")
    print(f"  Device       : {device}")
    print(f"  Experiments  : {len(exps)}")
    print(f"  Folds        : {fold_ids}")
    print(f"  Preprocessed : {ISRUC_PROC_DIR}")
    print(f"  Results      : {STUDY03_RESULTS}")
    print(f"{'#'*70}\n")

    full_summary = {}

    for exp in exps:
        exp_name = exp["name"]
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_name}  (source: {exp['source']}, "
              f"ch={exp['channels']}, ctx={exp['context_window']}, W={exp['window']})")
        print(f"{'='*60}")

        fold_results = []

        for fold_idx in fold_ids:
            ckpt = checkpoint_path(exp, fold_idx)
            result_path = fold_results_dir / f"{exp_name}_fold{fold_idx}.json"

            if result_path.exists() and not args.overwrite:
                print(f"  [fold_{fold_idx}] Loading cached result")
                with open(result_path) as f:
                    fold_result = json.load(f)
            elif not ckpt.exists():
                print(f"  [fold_{fold_idx}] SKIP — checkpoint not found: {ckpt}")
                continue
            else:
                print(f"  [fold_{fold_idx}] Evaluating {ckpt.name} ...")
                fold_result = evaluate_checkpoint(ckpt, exp, device, args.batch_size)
                fold_result["fold"]       = fold_idx
                fold_result["experiment"] = exp_name
                fold_result["checkpoint"] = str(ckpt)
                with open(result_path, "w") as f:
                    json.dump(fold_result, f, indent=2)

            fold_results.append(fold_result)
            print(
                f"  [fold_{fold_idx}]  kappa={fold_result['kappa']:.4f}  "
                f"acc={fold_result['accuracy']*100:.2f}%  "
                f"N1-F1={fold_result['n1_f1']:.3f}"
            )

        if not fold_results:
            print(f"  No fold results available for {exp_name} — skipping summary")
            continue

        agg = aggregate_folds(fold_results)
        full_summary[exp_name] = {
            "source":      exp["source"],
            "config":      {k: exp[k] for k in ("channels", "window", "alpha", "context_window")},
            "aggregate":   agg,
            "fold_results": fold_results,
        }

        print(
            f"\n  Summary  kappa={agg['kappa']['mean']:.4f}±{agg['kappa']['std']:.4f}  "
            f"acc={agg['accuracy']['mean']*100:.2f}%±{agg['accuracy']['std']*100:.2f}  "
            f"N1-F1={agg['n1_f1']['mean']:.3f}±{agg['n1_f1']['std']:.3f}"
        )

    # ---- Write outputs ----
    summary_json = STUDY03_RESULTS / "isruc_eval_summary.json"
    with open(summary_json, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nJSON summary → {summary_json}")

    if full_summary:
        print_summary_table(full_summary)
        md_path = STUDY03_RESULTS / "isruc_eval_summary.md"
        write_markdown_table(full_summary, md_path)
        print(f"Markdown table → {md_path}")
    else:
        print("No results to summarise.")


if __name__ == "__main__":
    main()
