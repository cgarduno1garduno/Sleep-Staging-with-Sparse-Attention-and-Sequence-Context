#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training Runner.

Trains all 9 experiments × 5 folds with:
  - Gradient clipping (max_norm=1.0) — prevents gradient explosion
  - Log_var clamping in UncertaintyLossWrapper — fixes α=0 instability
  - Early stopping (default patience=10)
  - Last checkpoint saved after every epoch — safe to Ctrl+C and resume
  - Best model saved by val kappa — used for final test evaluation
  - All folds use identical subject assignments — comparable test sets
  - Results aggregated across folds at the end (mean ± std per metric)

Experiments:
  Study 1 (9 configs, full 5-fold CV complete):
    no_transitions   α=0.0  W=64  3ch   (baseline: no transition supervision)
    alpha_1_0        α=1.0  W=64  3ch
    final_tasa       α=5.0  W=64  3ch   (proposed model)
    alpha_10_0       α=10.0 W=64  3ch
    window_32        α=5.0  W=32  3ch
    window_128       α=5.0  W=128 3ch
    1ch_FpzCz        α=5.0  W=64  1ch
    2ch_FpzCz_EOG    α=5.0  W=64  2ch
    full_attention   α=5.0  W=None 3ch  (O(L²) baseline)

  Study 2 ablations (fold_1 only):
    criterion_n1f1   α=5.0  W=64  3ch   no-op: same checkpoint as n1_recall
    focal_loss_g2    α=5.0  W=64  3ch   γ=2 focal loss: unstable, no precision gain

  Study 2 full run (3 experiments × 5 folds):
    seq_context_k1              α=5.0  W=64  3ch  context=1  (ablation: transitions + context)
    seq_context_k1_notrans      α=0.0  W=64  3ch  context=1  (proposed best model)
    seq_context_k1_notrans_2ch  α=0.0  W=64  2ch  context=1  (wearable: FpzCz+EOG only)

Usage:
  # Full run — creates output dir and trains everything:
  python scripts/run_cv_training.py --output-dir results_$(date +%Y-%m-%d)

  # Resume after interrupt — re-run the exact same command:
  python scripts/run_cv_training.py --output-dir results_2026-03-03

  # Quick smoke test (2 epochs, small data):
  python scripts/run_cv_training.py --output-dir results_smoke --smoke-test

Output structure:
  results_{date}/
    folds.json               — fixed subject assignments for all 5 folds
    state.json               — tracks completed experiments (auto-updated)
    fold_0/
      final_tasa/
        last_checkpoint.pt   — saved after every epoch (resume from here)
        best_model.pt        — best val-kappa checkpoint (used for test eval)
        results.json         — test metrics (written when experiment completes)
        predictions.npz      — test predictions + labels + transition_probs
      no_transitions/
        ...
    fold_1/
      ...
    cv_summary.json          — mean ± std across folds (written at the very end)
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.configurable import ConfigurableTASA, ContextTASA
from src.training.loss import UncertaintyLossWrapper, FocalLoss
from src.training.loops import get_device
from src.config import PROCESSED_DIR
from src.dataloading.samplers import create_weighted_sampler

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {"name": "no_transitions",  "channels": 3, "window": 64,  "alpha": 0.0},
    {"name": "alpha_1_0",       "channels": 3, "window": 64,  "alpha": 1.0},
    {"name": "final_tasa",      "channels": 3, "window": 64,  "alpha": 5.0},
    {"name": "alpha_10_0",      "channels": 3, "window": 64,  "alpha": 10.0},
    {"name": "window_32",       "channels": 3, "window": 32,  "alpha": 5.0},
    {"name": "window_128",      "channels": 3, "window": 128, "alpha": 5.0},
    {"name": "1ch_FpzCz",       "channels": 1, "window": 64,  "alpha": 5.0},
    {"name": "2ch_FpzCz_EOG",   "channels": 2, "window": 64,  "alpha": 5.0},
    {"name": "full_attention",  "channels": 3, "window": None, "alpha": 5.0},
    # Study 2 ablations (fold_1 only — kept for record, not re-run)
    {"name": "criterion_n1f1", "channels": 3, "window": 64, "alpha": 5.0, "composite": "n1_f1"},
    {"name": "focal_loss_g2",  "channels": 3, "window": 64, "alpha": 5.0, "loss": "focal", "gamma": 2.0},
    # Study 2c: neighboring-epoch context (α=5, with transition task)
    {"name": "seq_context_k1",        "channels": 3, "window": 64, "alpha": 5.0, "context_window": 1},
    # Study 2d: same context, no transition task (α=0) — ablation
    {"name": "seq_context_k1_notrans",     "channels": 3, "window": 64, "alpha": 0.0, "context_window": 1},
    # Study 2 full run: 2ch variant (FpzCz+EOG) — tests whether wearable claim holds with context
    {"name": "seq_context_k1_notrans_2ch", "channels": 2, "window": 64, "alpha": 0.0, "context_window": 1},
]

# Stage labels for reporting
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# Channel selection indices
CHANNEL_INDICES = {1: [0], 2: [0, 2], 3: [0, 1, 2]}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoldDataset(torch.utils.data.Dataset):
    """Dataset with channel selection for one CV fold split."""

    def __init__(
        self,
        metadata_path: Path,
        data_dir: Path,
        subject_ids: List[str],
        num_channels: int = 3,
    ):
        self.data_dir = Path(data_dir)
        self.ch_idx = CHANNEL_INDICES[num_channels]

        meta = pd.read_csv(metadata_path)
        meta = meta[meta["subject_id"].isin(subject_ids)].reset_index(drop=True)

        meta["epoch_index"] = meta["epoch_index"].astype(int)
        meta = meta.sort_values(["subject_id", "epoch_index"])

        next_stage = meta["stage_label"].shift(-1)
        next_subj = meta["subject_id"].shift(-1)
        meta["transition_label"] = (
            (meta["subject_id"] == next_subj) & (meta["stage_label"] != next_stage)
        ).fillna(False).astype(int)

        label_counts = meta["stage_label"].value_counts()
        total = len(meta)
        weights = {lbl: total / cnt for lbl, cnt in label_counts.items()}
        max_w = max(weights.values())
        self._weights = {k: v / max_w for k, v in weights.items()}

        self.meta = meta

    def get_sample_weights(self):
        return [self._weights[lbl] for lbl in self.meta["stage_label"]]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        signal = np.load(self.data_dir / row["filename"])[self.ch_idx, :]
        signal = torch.from_numpy(signal).float()
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True)
        signal = torch.clamp((signal - mean) / (std + 1e-6), -20.0, 20.0)
        return {
            "signal": signal,
            "stage_label": int(row["stage_label"]),
            "transition_label": int(row["transition_label"]),
            "subject_id": row["subject_id"],
        }


class ContextFoldDataset(torch.utils.data.Dataset):
    """
    Like FoldDataset but yields a window of (2*context_window + 1) consecutive
    epochs per sample. Boundary epochs are clamped within the same subject so
    the window never crosses a subject boundary.

    Returns signal: (2K+1, C, T) with the target epoch at index K.
    """

    def __init__(
        self,
        metadata_path: Path,
        data_dir: Path,
        subject_ids: List[str],
        num_channels: int = 3,
        context_window: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.ch_idx = CHANNEL_INDICES[num_channels]
        self.context_window = context_window

        meta = pd.read_csv(metadata_path)
        meta = meta[meta["subject_id"].isin(subject_ids)].reset_index(drop=True)
        meta["epoch_index"] = meta["epoch_index"].astype(int)
        meta = meta.sort_values(["subject_id", "epoch_index"]).reset_index(drop=True)

        next_stage = meta["stage_label"].shift(-1)
        next_subj  = meta["subject_id"].shift(-1)
        meta["transition_label"] = (
            (meta["subject_id"] == next_subj) & (meta["stage_label"] != next_stage)
        ).fillna(False).astype(int)

        label_counts = meta["stage_label"].value_counts()
        total = len(meta)
        weights = {lbl: total / cnt for lbl, cnt in label_counts.items()}
        max_w = max(weights.values())
        self._weights = {k: v / max_w for k, v in weights.items()}

        self.meta = meta

        # Per-subject iloc boundaries for context clamping.
        # After reset_index the DataFrame index equals the iloc position.
        self._subj_bounds: dict = {}
        for subj, grp in self.meta.groupby("subject_id", sort=False):
            self._subj_bounds[subj] = (int(grp.index.min()), int(grp.index.max()))

    def get_sample_weights(self):
        return [self._weights[lbl] for lbl in self.meta["stage_label"]]

    def __len__(self):
        return len(self.meta)

    def _load_epoch(self, iloc_pos: int) -> torch.Tensor:
        row = self.meta.iloc[iloc_pos]
        sig = np.load(self.data_dir / row["filename"])[self.ch_idx, :]
        sig = torch.from_numpy(sig).float()
        mean = sig.mean(dim=1, keepdim=True)
        std  = sig.std(dim=1, keepdim=True)
        return torch.clamp((sig - mean) / (std + 1e-6), -20.0, 20.0)

    def __getitem__(self, idx: int) -> dict:
        row  = self.meta.iloc[idx]
        subj = row["subject_id"]
        lo, hi = self._subj_bounds[subj]

        signals = []
        for offset in range(-self.context_window, self.context_window + 1):
            context_pos = max(lo, min(hi, idx + offset))
            signals.append(self._load_epoch(context_pos))

        return {
            "signal":           torch.stack(signals, dim=0),  # (2K+1, C, T)
            "stage_label":      int(row["stage_label"]),
            "transition_label": int(row["transition_label"]),
            "subject_id":       subj,
        }


# ---------------------------------------------------------------------------
# Fold creation
# ---------------------------------------------------------------------------

def create_folds(
    metadata_path: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, List]]:
    """
    Create n_folds subject splits.

    For each fold k:
      test  = subjects in group k
      train+val = remaining subjects, split 85/15
    """
    meta = pd.read_csv(metadata_path)
    subjects = list(meta["subject_id"].unique())

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    groups = np.array_split(subjects, n_folds)

    folds = {}
    for k in range(n_folds):
        test_subjects = list(groups[k])
        remaining = [s for j, grp in enumerate(groups) for s in grp if j != k]
        n_val = max(1, round(len(remaining) * 0.15))
        val_subjects = remaining[:n_val]
        train_subjects = remaining[n_val:]
        folds[f"fold_{k}"] = {
            "test": test_subjects,
            "val": val_subjects,
            "train": train_subjects,
        }

    return folds


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state(output_dir: Path) -> dict:
    state_path = output_dir / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed": []}


def save_state(state: dict, output_dir: Path):
    with open(output_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def exp_key(fold_name: str, exp_name: str) -> str:
    return f"{fold_name}/{exp_name}"


# ---------------------------------------------------------------------------
# Validation / test evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_wrapper: nn.Module,
    stage_criterion: nn.Module,
    transition_criterion: nn.Module,
    device: torch.device,
    alpha: float,
) -> tuple:
    model.eval()
    total_loss = stage_loss_sum = trans_loss_sum = 0.0
    all_preds, all_labels, all_trans_probs, all_trans_labels = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            sigs = batch["signal"].to(device)
            slbls = batch["stage_label"].to(device)
            tlbls = batch["transition_label"].float().to(device)

            out = model(sigs)
            sl = stage_criterion(out["stage_logits"], slbls)
            tl_raw = transition_criterion(out["transition_logits"].squeeze(-1), tlbls)
            combined = loss_wrapper([sl, alpha * tl_raw])

            total_loss += combined.item()
            stage_loss_sum += sl.item()
            trans_loss_sum += tl_raw.item()

            preds = out["stage_logits"].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(slbls.cpu().numpy())
            tp = torch.sigmoid(out["transition_logits"].squeeze(-1))
            all_trans_probs.extend(tp.cpu().numpy())
            all_trans_labels.extend(tlbls.cpu().numpy())

    n = len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_trans_probs = np.array(all_trans_probs)
    all_trans_labels = np.array(all_trans_labels)

    try:
        trans_auc = float(roc_auc_score(all_trans_labels, all_trans_probs))
    except ValueError:
        trans_auc = float("nan")  # only one class present in labels

    acc = float((all_preds == all_labels).mean())
    kappa = float(cohen_kappa_score(all_labels, all_preds))
    report = classification_report(
        all_labels, all_preds, target_names=STAGE_NAMES, output_dict=True, zero_division=0
    )

    metrics = {
        "loss": total_loss / n,
        "stage_loss": stage_loss_sum / n,
        "trans_loss": trans_loss_sum / n,
        "accuracy": acc,
        "kappa": kappa,
        "n1_recall": float(report["N1"]["recall"]),
        "n1_precision": float(report["N1"]["precision"]),
        "n1_f1": float(report["N1"]["f1-score"]),
        "per_class": {cls: report[cls] for cls in STAGE_NAMES},
        "transition_auc": trans_auc,
    }
    return metrics, all_preds, all_labels, all_trans_probs


# ---------------------------------------------------------------------------
# Single experiment runner (with resume support)
# ---------------------------------------------------------------------------

def run_experiment(
    fold_name: str,
    exp: dict,
    fold_subjects: dict,
    output_dir: Path,
    metadata_path: Path,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    early_stop_patience: int,
    max_grad_norm: float,
    seed: int,
) -> dict:
    """
    Train one experiment for one fold.

    Saves last_checkpoint.pt after every epoch so training can be interrupted
    and resumed. When done, evaluates on the test set and saves results.json.
    """
    exp_name       = exp["name"]
    alpha          = exp["alpha"]
    channels       = exp["channels"]
    window         = exp["window"]
    context_window = exp.get("context_window", 0)

    exp_dir = output_dir / fold_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Fold: {fold_name}  |  Experiment: {exp_name}")
    print(f"  α={alpha}  W={window}  ch={channels}  ctx={context_window}  device={device}")
    print(f"{'='*60}")

    # --- Datasets -------------------------------------------------------
    if context_window > 0:
        train_ds = ContextFoldDataset(metadata_path, data_dir, fold_subjects["train"], channels, context_window)
        val_ds   = ContextFoldDataset(metadata_path, data_dir, fold_subjects["val"],   channels, context_window)
        test_ds  = ContextFoldDataset(metadata_path, data_dir, fold_subjects["test"],  channels, context_window)
    else:
        train_ds = FoldDataset(metadata_path, data_dir, fold_subjects["train"], channels)
        val_ds   = FoldDataset(metadata_path, data_dir, fold_subjects["val"],   channels)
        test_ds  = FoldDataset(metadata_path, data_dir, fold_subjects["test"],  channels)

    print(f"  Subjects — train: {len(fold_subjects['train'])}  "
          f"val: {len(fold_subjects['val'])}  test: {len(fold_subjects['test'])}")
    print(f"  Samples  — train: {len(train_ds)}  "
          f"val: {len(val_ds)}  test: {len(test_ds)}")

    sampler = create_weighted_sampler(
        train_ds.get_sample_weights(), num_samples=len(train_ds)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)

    # --- Model / loss / optimizer ----------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    if context_window > 0:
        model = ContextTASA(
            input_channels=channels,
            d_model=64,
            n_layers=4,
            n_heads=4,
            window_size=window,
            num_classes=5,
            context_window=context_window,
        ).to(device)
    else:
        model = ConfigurableTASA(
            input_channels=channels,
            d_model=64,
            n_layers=4,
            n_heads=4,
            window_size=window,
            num_classes=5,
        ).to(device)

    loss_wrapper = UncertaintyLossWrapper(num_tasks=2).to(device)

    # Stage loss — selectable per experiment via exp["loss"].
    # Class weight [1,2,1,1,1]: N1 gets 2× penalty for spectral difficulty.
    # WeightedRandomSampler already equalises class frequency; the weight addresses difficulty.
    # Order: [Wake=0, N1=1, N2=2, N3=3, REM=4]
    stage_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0], device=device)
    loss_type = exp.get("loss", "cross_entropy")
    if loss_type == "focal":
        gamma = exp.get("gamma", 2.0)
        stage_crit = FocalLoss(gamma=gamma, weight=stage_weights)
    else:
        stage_crit = nn.CrossEntropyLoss(weight=stage_weights)
    trans_crit = nn.BCEWithLogitsLoss()

    all_params = list(model.parameters()) + list(loss_wrapper.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-3)

    # --- Resume from last checkpoint if available -------------------------
    last_ckpt_path = exp_dir / "last_checkpoint.pt"
    best_ckpt_path = exp_dir / "best_model.pt"

    # Composite best-model metric: kappa + COMPOSITE_LAMBDA * <n1_recall or n1_f1>
    # composite_metric is set per-experiment via the "composite" key in the experiment dict.
    #   "n1_recall" (default, Study 1 behaviour): rewards sensitivity, can over-boost recall
    #   "n1_f1"    (Study 2a): rewards precision-recall balance directly
    # Lambda=0.3: gaining 0.05 kappa but losing 0.17 in the composite metric is break-even.
    COMPOSITE_LAMBDA = 0.3
    composite_metric = exp.get("composite", "n1_recall")

    start_epoch = 0
    best_val_composite = -1.0
    early_stop_counter = 0
    history = {"train": [], "val": []}

    if last_ckpt_path.exists():
        print(f"  Resuming from {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_wrapper.load_state_dict(ckpt["loss_wrapper_state_dict"])
        start_epoch            = ckpt["epoch"] + 1
        best_val_composite     = ckpt["best_val_composite"]
        early_stop_counter     = ckpt["early_stop_counter"]
        history                = ckpt["history"]
        print(f"  Resuming at epoch {start_epoch + 1}/{epochs}  "
              f"(best composite so far: {best_val_composite:.4f})")

    # --- Training loop ----------------------------------------------------
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = stage_loss_sum = trans_loss_sum = 0.0
        correct = total = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs} train", leave=False):
            sigs  = batch["signal"].to(device)
            slbls = batch["stage_label"].to(device)
            tlbls = batch["transition_label"].float().to(device)

            optimizer.zero_grad()
            out = model(sigs)

            sl     = stage_crit(out["stage_logits"], slbls)
            tl_raw = trans_crit(out["transition_logits"].squeeze(-1), tlbls)
            combined = loss_wrapper([sl, alpha * tl_raw])

            combined.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)
            optimizer.step()

            total_loss    += combined.item()
            stage_loss_sum += sl.item()
            trans_loss_sum += tl_raw.item()
            preds = out["stage_logits"].argmax(dim=1)
            correct += (preds == slbls).sum().item()
            total   += slbls.size(0)

        nb = len(train_loader)
        train_metrics = {
            "train_loss":            total_loss     / nb,
            "train_stage_loss":      stage_loss_sum / nb,
            "train_transition_loss": trans_loss_sum / nb,
            "train_accuracy":        correct / total if total > 0 else 0.0,
        }
        history["train"].append(train_metrics)

        # Validation
        val_metrics, _, _, _ = evaluate(
            model, val_loader, loss_wrapper, stage_crit, trans_crit, device, alpha
        )
        # Rename keys to val_*
        val_entry = {f"val_{k}": v for k, v in val_metrics.items()
                     if k not in ("per_class",)}
        history["val"].append(val_entry)

        composite_val = val_metrics[composite_metric]
        val_composite = val_metrics["kappa"] + COMPOSITE_LAMBDA * composite_val
        print(
            f"  Ep {epoch+1:3d}/{epochs} | "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"acc={train_metrics['train_accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"kappa={val_metrics['kappa']:.4f} "
            f"{composite_metric}={composite_val:.4f} "
            f"composite={val_composite:.4f}",
            flush=True,
        )

        # Best model checkpoint (by composite metric)
        if val_composite > best_val_composite:
            best_val_composite = val_composite
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                    "val_kappa": val_metrics["kappa"],
                    "val_n1_recall": val_metrics["n1_recall"],
                    "val_composite": best_val_composite,
                },
                best_ckpt_path,
            )
            print(f"    ↑ New best model (composite={best_val_composite:.4f}  "
                  f"kappa={val_metrics['kappa']:.4f}  {composite_metric}={composite_val:.4f})",
                  flush=True)
        else:
            early_stop_counter += 1

        # Last checkpoint (every epoch — resume point)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                "best_val_composite": best_val_composite,
                "early_stop_counter": early_stop_counter,
                "history": history,
            },
            last_ckpt_path,
        )

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break

    training_time = time.time() - start_time

    # --- Test evaluation (uses best model checkpoint) --------------------
    if not best_ckpt_path.exists():
        print(f"  WARNING: no best model checkpoint found — training may have failed")
        return {"error": "no best model checkpoint"}

    print(f"\n  Evaluating test set with best model checkpoint...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_preds, test_labels, test_trans_probs = evaluate(
        model, test_loader, loss_wrapper, stage_crit, trans_crit, device, alpha
    )

    print(f"  Test accuracy: {test_metrics['accuracy']*100:.2f}%  "
          f"kappa: {test_metrics['kappa']:.4f}  "
          f"N1 recall: {test_metrics['n1_recall']:.4f}")

    # Save predictions
    np.savez(
        exp_dir / "predictions.npz",
        predictions=test_preds,
        labels=test_labels,
        transition_probs=test_trans_probs,
    )

    # Save results
    results = {
        "fold": fold_name,
        "experiment": exp_name,
        "config": {
            "channels": channels,
            "window": window,
            "alpha": alpha,
            "epochs_run": len(history["train"]),
            "epochs_max": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": 1e-3,
            "early_stop_patience": early_stop_patience,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "stage_loss_weights": [1.0, 2.0, 1.0, 1.0, 1.0],
            "stage_loss_type": loss_type,
            "stage_loss_gamma": exp.get("gamma", None),
            "context_window": context_window,
            "best_model_criterion": f"kappa + {COMPOSITE_LAMBDA} * {composite_metric}",
        },
        "subjects": {
            "train": fold_subjects["train"],
            "val":   fold_subjects["val"],
            "test":  fold_subjects["test"],
        },
        "best_val_composite": best_val_composite,
        "training_time_seconds": training_time,
        "test_accuracy":    test_metrics["accuracy"],
        "test_kappa":       test_metrics["kappa"],
        "test_n1_recall":   test_metrics["n1_recall"],
        "test_n1_precision": test_metrics["n1_precision"],
        "test_n1_f1":          test_metrics["n1_f1"],
        "test_transition_auc": test_metrics.get("transition_auc"),
        "test_per_class":      test_metrics["per_class"],
        "history": history,
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(output_dir: Path, folds: dict) -> dict:
    """Read all results.json files and compute mean ± std per experiment."""
    fold_names = sorted(folds.keys())
    exp_names  = [e["name"] for e in EXPERIMENTS]

    summary = {}

    for exp_name in exp_names:
        metrics_per_fold = []
        for fold_name in fold_names:
            rpath = output_dir / fold_name / exp_name / "results.json"
            if not rpath.exists():
                continue
            with open(rpath) as f:
                r = json.load(f)
            metrics_per_fold.append({
                "accuracy":       r["test_accuracy"],
                "kappa":          r["test_kappa"],
                "n1_recall":      r["test_n1_recall"],
                "n1_precision":   r["test_n1_precision"],
                "n1_f1":          r["test_n1_f1"],
                "transition_auc": r.get("test_transition_auc"),
            })

        if not metrics_per_fold:
            continue

        agg = {}
        for key in ["accuracy", "kappa", "n1_recall", "n1_precision", "n1_f1"]:
            vals = [m[key] for m in metrics_per_fold]
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "folds": vals,
                "n_folds": len(vals),
            }

        # Transition AUC — optional, only present in runs that logged it
        auc_vals = [m["transition_auc"] for m in metrics_per_fold
                    if m.get("transition_auc") is not None
                    and not np.isnan(m["transition_auc"])]
        if auc_vals:
            agg["transition_auc"] = {
                "mean": float(np.mean(auc_vals)),
                "std":  float(np.std(auc_vals)),
                "folds": auc_vals,
                "n_folds": len(auc_vals),
            }

        summary[exp_name] = agg

    return summary


def print_summary_table(summary: dict):
    print("\n" + "#" * 80)
    print("CV SUMMARY  (mean ± std across folds)")
    print("#" * 80)
    has_auc = any("transition_auc" in agg for agg in summary.values())
    header = (f"\n{'Experiment':<22} {'Accuracy':>14} {'Kappa':>14} "
              f"{'N1 Recall':>12} {'N1-F1':>10} {'Folds':>7}")
    if has_auc:
        header += f"  {'Trans AUC':>10}"
    print(header)
    print("-" * (85 + (13 if has_auc else 0)))
    for exp_name, agg in summary.items():
        acc   = agg["accuracy"]
        kappa = agg["kappa"]
        n1r   = agg["n1_recall"]
        n1f1  = agg["n1_f1"]
        n = acc["n_folds"]
        row = (f"{exp_name:<22} "
               f"{acc['mean']*100:>6.2f}%±{acc['std']*100:.2f}  "
               f"{kappa['mean']:>6.4f}±{kappa['std']:.4f}  "
               f"{n1r['mean']:>5.3f}±{n1r['std']:.3f}  "
               f"{n1f1['mean']:>5.3f}±{n1f1['std']:.3f}  "
               f"{n:>7}")
        if has_auc and "transition_auc" in agg:
            tauc = agg["transition_auc"]
            row += f"  {tauc['mean']:>5.3f}±{tauc['std']:.3f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    today = date.today().isoformat()
    parser = argparse.ArgumentParser(
        description="5-fold CV training runner for TASA paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"results_{today}"),
        help=f"Output directory (default: results_{today}). "
             "Re-run with same path to resume."
    )
    parser.add_argument("--n-folds",  type=int,   default=5,    help="Number of CV folds")
    parser.add_argument("--epochs",   type=int,   default=50,   help="Max epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=32,   help="Batch size")
    parser.add_argument("--lr",       type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--early-stop-patience", type=int, default=10,
        help="Early stopping patience (epochs without val kappa improvement)"
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--seed",     type=int,   default=42,   help="Random seed for fold creation")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test: 2 epochs, 1 subject per split"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Run only specific experiments by name (default: all)"
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Run only specific fold indices, e.g. --folds 0 1 2 (default: all)"
    )
    parser.add_argument(
        "--folds-file", type=Path, default=None,
        help="Path to an existing folds.json to reuse (e.g. study_01/results/folds.json). "
             "Required for direct comparability across studies. If the output dir already "
             "has a folds.json, that takes precedence over this flag."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.early_stop_patience = 999  # disable for smoke test

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = PROCESSED_DIR / "metadata.csv"

    print(f"\n{'#'*60}")
    print(f"TASA 5-Fold CV Training Runner")
    print(f"Output dir : {output_dir.resolve()}")
    print(f"Epochs     : {args.epochs}")
    print(f"Folds      : {args.n_folds}")
    print(f"Smoke test : {args.smoke_test}")
    print(f"{'#'*60}")

    # --- Create / load fold assignments ----------------------------------
    folds_path = output_dir / "folds.json"
    if folds_path.exists():
        with open(folds_path) as f:
            folds = json.load(f)
        print(f"\nLoaded existing fold assignments from {folds_path}")
    elif args.folds_file:
        with open(args.folds_file) as f:
            folds = json.load(f)
        with open(folds_path, "w") as f:
            json.dump(folds, f, indent=2)
        print(f"\nUsing fold assignments from {args.folds_file} (copied to {folds_path})")
    else:
        folds = create_folds(metadata_path, n_folds=args.n_folds, seed=args.seed)
        if args.smoke_test:
            for fold_name in folds:
                folds[fold_name]["train"] = folds[fold_name]["train"][:2]
                folds[fold_name]["val"]   = folds[fold_name]["val"][:1]
                folds[fold_name]["test"]  = folds[fold_name]["test"][:1]
        with open(folds_path, "w") as f:
            json.dump(folds, f, indent=2)
        print(f"\nCreated fold assignments → {folds_path}")

    for fold_name, splits in folds.items():
        print(f"  {fold_name}: train={len(splits['train'])}  "
              f"val={len(splits['val'])}  test={len(splits['test'])}")

    # --- Load state -------------------------------------------------------
    state = load_state(output_dir)
    completed = set(state["completed"])

    # --- Select experiments and folds to run ------------------------------
    experiments = EXPERIMENTS
    if args.experiments:
        experiments = [e for e in EXPERIMENTS if e["name"] in args.experiments]

    fold_names = sorted(folds.keys())
    if args.folds is not None:
        fold_names = [f"fold_{k}" for k in args.folds if f"fold_{k}" in folds]

    total = len(fold_names) * len(experiments)
    done  = sum(1 for fn in fold_names for ex in experiments
                if exp_key(fn, ex["name"]) in completed)

    print(f"\nProgress: {done}/{total} experiments complete")
    print("(Re-run this command at any time to resume.)\n")

    # --- Main loop --------------------------------------------------------
    cv_results = {}

    for fold_name in fold_names:
        fold_subjects = folds[fold_name]
        for exp in experiments:
            key = exp_key(fold_name, exp["name"])

            # Skip completed experiments
            if key in completed:
                print(f"  [SKIP] {key} (already complete)")
                continue

            # Also skip if results.json already exists (more robust check)
            results_path = output_dir / fold_name / exp["name"] / "results.json"
            if results_path.exists():
                print(f"  [SKIP] {key} (results.json exists)")
                completed.add(key)
                state["completed"] = sorted(completed)
                save_state(state, output_dir)
                continue

            # Run the experiment
            result = run_experiment(
                fold_name      = fold_name,
                exp            = exp,
                fold_subjects  = fold_subjects,
                output_dir     = output_dir,
                metadata_path  = metadata_path,
                data_dir       = PROCESSED_DIR,
                epochs         = args.epochs,
                batch_size     = args.batch_size,
                lr             = args.lr,
                early_stop_patience   = args.early_stop_patience,
                max_grad_norm  = args.max_grad_norm,
                seed           = args.seed,
            )

            # Mark complete only if results.json was written
            if results_path.exists():
                completed.add(key)
                state["completed"] = sorted(completed)
                save_state(state, output_dir)
                cv_results[key] = result
                done += 1
                print(f"\n  Progress: {done}/{total} experiments complete")
            else:
                print(f"\n  WARNING: {key} did not produce results.json — may have failed")

    # --- Aggregate --------------------------------------------------------
    summary = aggregate_results(output_dir, folds)

    if summary:
        cv_summary_path = output_dir / "cv_summary.json"
        with open(cv_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCV summary saved → {cv_summary_path}")
        print_summary_table(summary)
    else:
        print("\nNo completed experiments to summarize yet.")

    n_remaining = total - len(completed.intersection(
        {exp_key(fn, ex["name"]) for fn in fold_names for ex in experiments}
    ))
    if n_remaining > 0:
        print(f"\n{n_remaining} experiments still pending.")
        print(f"Re-run: python scripts/run_cv_training.py --output-dir {output_dir}")
    else:
        print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
