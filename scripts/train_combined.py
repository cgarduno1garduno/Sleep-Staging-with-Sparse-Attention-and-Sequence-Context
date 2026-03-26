#!/usr/bin/env python3
"""
Study 04 — Combined Sleep-EDF + ISRUC training (2-channel only).

Trains on 80% of Sleep-EDF subjects + 80% of ISRUC subjects simultaneously,
then evaluates on each dataset's held-out 20% separately.

5-fold CV design:
  - Sleep-EDF folds: reuse study_01/results/folds.json (same fold assignments
    as Studies 1 & 2 for direct comparability — test subjects are identical)
  - ISRUC folds: created fresh here with the same seed=42 convention
  - For each fold k: train on Sleep-EDF_{train+val,k} + ISRUC_{train+val,k}
    Evaluate separately on Sleep-EDF_test,k and ISRUC_test,k

Metrics reported:
  - Sleep-EDF test: κ, accuracy, N1-F1  (compare to Study 2 results)
  - ISRUC test:     κ, accuracy, N1-F1  (compare to Study 3 zero-shot results)

Usage (from project root):
  python study_04/scripts/run_combined_training.py
  python study_04/scripts/run_combined_training.py --folds 0 1    # subset
  python study_04/scripts/run_combined_training.py --smoke-test
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "study_04"))

from src.config import (
    SLEEPEDF_FOLDS_FILE, ISRUC_PROC_DIR, SLEEPEDF_PROC_DIR,
    STUDY04_RESULTS, EXPERIMENTS,
)
from src.models.configurable import ConfigurableTASA, ContextTASA
from src.training.loops import get_device
from src.training.loss import UncertaintyLossWrapper
from src.dataloading.combined_dataset import CombinedDataset

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_FOLDS     = 5
SEED        = 42


# ---------------------------------------------------------------------------
# ISRUC fold creation (subject-level, same convention as Sleep-EDF)
# ---------------------------------------------------------------------------

def create_isruc_folds(n_folds: int = 5, seed: int = 42) -> dict:
    """Create subject-level CV folds for ISRUC Subgroup 1."""
    npz_files = sorted(ISRUC_PROC_DIR.glob("isruc_S*.npz"))
    subjects  = sorted([int(p.stem.replace("isruc_S", "")) for p in npz_files])

    rng = np.random.default_rng(seed)
    subjects = list(rng.permutation(subjects))
    groups   = np.array_split(subjects, n_folds)

    folds = {}
    for k in range(n_folds):
        test = list(groups[k])
        remaining = [s for j, grp in enumerate(groups) for s in grp if j != k]
        n_val = max(1, round(len(remaining) * 0.15))
        folds[f"fold_{k}"] = {
            "test":  test,
            "val":   remaining[:n_val],
            "train": remaining[n_val:],
        }
    return folds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, split_label="eval") -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split_label}", leave=False):
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
        all_labels, all_preds, target_names=STAGE_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy":     acc,
        "kappa":        kappa,
        "n1_recall":    float(report["N1"]["recall"]),
        "n1_precision": float(report["N1"]["precision"]),
        "n1_f1":        float(report["N1"]["f1-score"]),
        "per_class":    {s: report[s] for s in STAGE_NAMES},
    }


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    fold_name: str,
    exp: dict,
    sedf_fold: dict,
    isruc_fold: dict,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    max_grad_norm: float,
) -> dict:
    ctx             = exp["context_window"]
    channels        = exp["channels"]
    channel_indices = exp["channel_indices"]
    window          = exp["window"]
    alpha           = exp["alpha"]
    exp_name        = exp["name"]

    exp_dir = output_dir / fold_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  {fold_name} | {exp_name} | ctx={ctx} ch={channels} W={window} device={device}")

    # --- Datasets -----------------------------------------------------------
    train_ds = CombinedDataset(
        sleepedf_subjects=sedf_fold["train"],
        isruc_subjects=isruc_fold["train"],
        context_window=ctx,
        channel_indices=channel_indices,
    )
    val_sedf_ds = CombinedDataset(
        sleepedf_subjects=sedf_fold["val"],
        isruc_subjects=[],
        context_window=ctx,
        channel_indices=channel_indices,
    )
    val_isruc_ds = CombinedDataset(
        sleepedf_subjects=[],
        isruc_subjects=isruc_fold["val"],
        context_window=ctx,
        channel_indices=channel_indices,
    )
    test_sedf_ds = CombinedDataset(
        sleepedf_subjects=sedf_fold["test"],
        isruc_subjects=[],
        context_window=ctx,
        channel_indices=channel_indices,
    )
    test_isruc_ds = CombinedDataset(
        sleepedf_subjects=[],
        isruc_subjects=isruc_fold["test"],
        context_window=ctx,
        channel_indices=channel_indices,
    )

    print(f"  Train: {len(train_ds)} epochs  "
          f"(sedf {len(sedf_fold['train'])}S + isruc {len(isruc_fold['train'])}S)")
    print(f"  Val sedf: {len(val_sedf_ds)}  Val isruc: {len(val_isruc_ds)}  "
          f"Test sedf: {len(test_sedf_ds)}  Test isruc: {len(test_isruc_ds)}")

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), num_samples=len(train_ds), replacement=True
    )
    train_loader      = DataLoader(train_ds,      batch_size=batch_size, sampler=sampler,  num_workers=0)
    val_sedf_loader   = DataLoader(val_sedf_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    val_isruc_loader  = DataLoader(val_isruc_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_sedf_loader  = DataLoader(test_sedf_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_isruc_loader = DataLoader(test_isruc_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Model --------------------------------------------------------------
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if ctx > 0:
        model = ContextTASA(
            input_channels=channels, d_model=64, n_layers=4, n_heads=4,
            window_size=window, num_classes=5, context_window=ctx,
        ).to(device)
    else:
        model = ConfigurableTASA(
            input_channels=channels, d_model=64, n_layers=4, n_heads=4,
            window_size=window, num_classes=5,
        ).to(device)

    loss_wrapper   = UncertaintyLossWrapper(num_tasks=2).to(device)
    stage_weights  = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0], device=device)
    stage_crit     = nn.CrossEntropyLoss(weight=stage_weights)
    trans_crit     = nn.BCEWithLogitsLoss()
    all_params     = list(model.parameters()) + list(loss_wrapper.parameters())
    optimizer      = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-3)

    # --- Resume -------------------------------------------------------------
    last_ckpt = exp_dir / "last_checkpoint.pt"
    best_ckpt = exp_dir / "best_model.pt"
    LAMBDA    = 0.3

    start_epoch        = 0
    best_val_composite = -1.0
    early_stop_counter = 0
    history            = {"train": [], "val": []}

    if last_ckpt.exists():
        print(f"  Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_wrapper.load_state_dict(ckpt["loss_wrapper_state_dict"])
        start_epoch        = ckpt["epoch"] + 1
        best_val_composite = ckpt["best_val_composite"]
        early_stop_counter = ckpt["early_stop_counter"]
        history            = ckpt["history"]

    # --- Training loop ------------------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = correct = total = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False):
            sigs  = batch["signal"].to(device)
            slbls = batch["stage_label"].to(device)
            # Transition labels not stored in combined dataset — use zeros
            tlbls = torch.zeros(len(slbls), device=device)

            optimizer.zero_grad()
            out      = model(sigs)
            sl       = stage_crit(out["stage_logits"], slbls)
            tl_raw   = trans_crit(out["transition_logits"].squeeze(-1), tlbls)
            combined = loss_wrapper([sl, alpha * tl_raw])
            combined.backward()
            nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            optimizer.step()

            total_loss += combined.item()
            preds = out["stage_logits"].argmax(dim=1)
            correct += (preds == slbls).sum().item()
            total   += slbls.size(0)

        train_acc = correct / total if total > 0 else 0.0
        history["train"].append({"train_loss": total_loss / len(train_loader),
                                  "train_accuracy": train_acc})

        # Validation on both datasets — composite criterion for checkpoint selection
        val_sedf_m  = evaluate(model, val_sedf_loader,  device, "val_sedf")
        val_isruc_m = evaluate(model, val_isruc_loader, device, "val_isruc")
        val_composite = (
            0.5 * (val_sedf_m["kappa"] + val_isruc_m["kappa"])
            + LAMBDA * val_sedf_m["n1_recall"]
        )
        val_entry = {f"sedf_{k}": v for k, v in val_sedf_m.items() if k != "per_class"}
        val_entry.update({f"isruc_{k}": v for k, v in val_isruc_m.items() if k != "per_class"})
        history["val"].append(val_entry)

        print(
            f"  Ep {epoch+1:3d}/{epochs} | "
            f"train_acc={train_acc:.4f} | "
            f"sedf_κ={val_sedf_m['kappa']:.4f}  isruc_κ={val_isruc_m['kappa']:.4f}  "
            f"composite={val_composite:.4f}",
            flush=True,
        )

        if val_composite > best_val_composite:
            best_val_composite = val_composite
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                "val_sedf_kappa":  val_sedf_m["kappa"],
                "val_isruc_kappa": val_isruc_m["kappa"],
                "val_composite": best_val_composite,
            }, best_ckpt)
            print(f"    ↑ New best (composite={best_val_composite:.4f}  "
                  f"sedf_κ={val_sedf_m['kappa']:.4f}  isruc_κ={val_isruc_m['kappa']:.4f})")
        else:
            early_stop_counter += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_wrapper_state_dict": loss_wrapper.state_dict(),
            "best_val_composite": best_val_composite,
            "early_stop_counter": early_stop_counter,
            "history": history,
        }, last_ckpt)

        if early_stop_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - t0

    # --- Test evaluation ----------------------------------------------------
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_sedf_m  = evaluate(model, test_sedf_loader,  device, "test_sedf")
    test_isruc_m = evaluate(model, test_isruc_loader, device, "test_isruc")

    print(f"\n  Sleep-EDF test:  kappa={test_sedf_m['kappa']:.4f}  "
          f"N1-F1={test_sedf_m['n1_f1']:.3f}")
    print(f"  ISRUC test:      kappa={test_isruc_m['kappa']:.4f}  "
          f"N1-F1={test_isruc_m['n1_f1']:.3f}")

    results = {
        "fold": fold_name, "experiment": exp_name,
        "training_time_seconds": training_time,
        "sleepedf_subjects": sedf_fold,
        "isruc_subjects": isruc_fold,
        "test_sleepedf": test_sedf_m,
        "test_isruc":    test_isruc_m,
        "history": history,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(output_dir: Path, fold_names: list, exp_names: list) -> dict:
    summary = {}
    for exp_name in exp_names:
        sedf_folds, isruc_folds = [], []
        for fold_name in fold_names:
            rpath = output_dir / fold_name / exp_name / "results.json"
            if not rpath.exists():
                continue
            with open(rpath) as f:
                r = json.load(f)
            sedf_folds.append(r["test_sleepedf"])
            isruc_folds.append(r["test_isruc"])

        if not sedf_folds:
            continue

        def agg_metric(fold_results, key):
            vals = [m[key] for m in fold_results]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": vals}

        summary[exp_name] = {
            "sleepedf": {k: agg_metric(sedf_folds,  k)
                         for k in ["accuracy", "kappa", "n1_f1", "n1_recall"]},
            "isruc":    {k: agg_metric(isruc_folds, k)
                         for k in ["accuracy", "kappa", "n1_f1", "n1_recall"]},
        }
    return summary


def print_summary(summary: dict):
    print("\n" + "#" * 80)
    print("STUDY 04 SUMMARY — Combined training results (mean ± std across 5 folds)")
    print("#" * 80)
    for exp_name, entry in summary.items():
        print(f"\n  {exp_name}")
        for dataset in ["sleepedf", "isruc"]:
            agg = entry[dataset]
            print(
                f"    {dataset:<10}  kappa={agg['kappa']['mean']:.4f}±{agg['kappa']['std']:.4f}  "
                f"acc={agg['accuracy']['mean']*100:.2f}%  "
                f"N1-F1={agg['n1_f1']['mean']:.3f}±{agg['n1_f1']['std']:.3f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Study 04 combined training")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--folds",      nargs="+", type=int, default=None)
    p.add_argument("--experiments",nargs="+", default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs   = 2
        args.patience = 999

    output_dir = args.output_dir or (STUDY04_RESULTS / f"run_{date.today().isoformat()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Sleep-EDF folds (reuse study_01 assignments) ------------------
    with open(SLEEPEDF_FOLDS_FILE) as f:
        sedf_folds = json.load(f)

    # --- Create or load ISRUC folds -----------------------------------------
    isruc_folds_path = output_dir / "isruc_folds.json"
    if isruc_folds_path.exists():
        with open(isruc_folds_path) as f:
            isruc_folds = json.load(f)
        print(f"Loaded ISRUC folds from {isruc_folds_path}")
    else:
        isruc_folds = create_isruc_folds(n_folds=N_FOLDS, seed=SEED)
        # Convert numpy int64 → Python int for JSON serialisation
        isruc_folds = {
            fold: {split: [int(s) for s in subs]
                   for split, subs in splits.items()}
            for fold, splits in isruc_folds.items()
        }
        with open(isruc_folds_path, "w") as f:
            json.dump(isruc_folds, f, indent=2)
        print(f"Created ISRUC folds → {isruc_folds_path}")

    fold_ids  = args.folds if args.folds is not None else list(range(N_FOLDS))
    fold_names = [f"fold_{k}" for k in fold_ids]
    exps = EXPERIMENTS
    if args.experiments:
        exps = [e for e in EXPERIMENTS if e["name"] in args.experiments]

    if args.smoke_test:
        for folds in [sedf_folds, isruc_folds]:
            for fold_name in folds:
                folds[fold_name]["train"] = folds[fold_name]["train"][:2]
                folds[fold_name]["val"]   = folds[fold_name]["val"][:1]
                folds[fold_name]["test"]  = folds[fold_name]["test"][:1]

    # --- State tracking (resume support) ------------------------------------
    state_path = output_dir / "state.json"
    state      = json.loads(state_path.read_text()) if state_path.exists() else {"completed": []}
    completed  = set(state["completed"])

    print(f"\n{'#'*60}")
    print(f"Study 04 — Combined Sleep-EDF + ISRUC Training")
    print(f"Output dir  : {output_dir}")
    print(f"Experiments : {[e['name'] for e in exps]}")
    print(f"Folds       : {fold_ids}")
    print(f"{'#'*60}")

    for fold_name in fold_names:
        for exp in exps:
            key = f"{fold_name}/{exp['name']}"
            results_path = output_dir / fold_name / exp["name"] / "results.json"

            if key in completed or results_path.exists():
                print(f"  [SKIP] {key}")
                completed.add(key)
                continue

            run_experiment(
                fold_name=fold_name,
                exp=exp,
                sedf_fold=sedf_folds[fold_name],
                isruc_fold=isruc_folds[fold_name],
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                max_grad_norm=args.max_grad_norm,
            )

            if results_path.exists():
                completed.add(key)
                state["completed"] = sorted(completed)
                state_path.write_text(json.dumps(state, indent=2))

    # --- Aggregate ----------------------------------------------------------
    summary = aggregate(output_dir, fold_names, [e["name"] for e in exps])
    if summary:
        summary_path = output_dir / "cv_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_summary(summary)
        print(f"\nSummary saved → {summary_path}")
    else:
        print("No completed experiments to summarize yet.")


if __name__ == "__main__":
    main()
