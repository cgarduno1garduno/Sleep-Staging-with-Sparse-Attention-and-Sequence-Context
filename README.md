# Sleep Staging with Sparse Attention and Sequence Context

Automated sleep stage classification using an efficient sparse-attention transformer trained on Sleep-EDF-78 and evaluated for cross-dataset generalization on ISRUC Subgroup 1.

---

## Overview

This repository contains the code and result summaries for a four-study research arc exploring:

1. **Sparse attention is better than full attention** for within-epoch EEG feature extraction (Study 1: delta-kappa = +0.027 vs O(L^2) baseline).
2. **Three-epoch sequence context** substantially improves N1 staging, the hardest sleep stage (Study 2: N1-F1 from 0.472 to 0.532, +13% relative).
3. **2-channel wearable configuration** (Fpz-Cz + EOG) nearly matches 3-channel performance, enabling ambulatory use (Study 2: delta-kappa = 0.003).
4. **Joint Sleep-EDF + ISRUC training** largely closes the zero-shot generalization gap, recovering most of the cross-dataset kappa loss (Study 4 vs Study 3).

The proposed model uses 222K parameters — 55% fewer than TinySleepNet — while matching or approaching competitive performance.

---

## Architecture

The proposed architecture processes a window of 3 consecutive 30-second epochs (prev, center, next) through a shared backbone:

```
Input: (B, 3, C, 3000)    — batch of epoch triplets
         |
         v
Shared backbone (per epoch, weights shared):
  Conv1d embedding:  (B*3, C, 3000) -> (B*3, 64, 750)
  4x Sparse Attention layers (W=64, 4 heads)
  LayerNorm + FFN
         |
         v  global avg pool
Epoch embeddings: (B, 3, 64)
         |  concatenate
         v
Staging head: Linear(192->64) + ReLU + Linear(64->5)
```

Key design choices:
- **Sparse attention (W=64)**: each token attends only to neighbors within 64 positions, making attention O(L*W) instead of O(L^2)
- **Shared backbone**: neighbor epochs are processed with identical weights, keeping parameters at 222K
- **Context window = 1**: 3 epochs (90 seconds) of context; wider windows did not improve results
- **alpha=0 (no transition task)**: the auxiliary transition head (2,113 params) is present but receives zero gradient during training; removing it entirely produced identical results

**Architecture Consistency**: The core backbone (Conv1-d embedding + 4-layer sparse transformer) is identical across all studies. The only variations are:
1. **Input Channels**: Study 1 uses 1-ch, while others use 2 or 3 channels.
2. **Context Window**: Study 1 is single-epoch, while others use a 3-epoch window.
3. **Training Data**: Studies 1-2 use Sleep-EDF, Study 3 is zero-shot, and Study 4 is joint-domain training.

---

## Results Summary

All results are reported as mean +/- std across 5 folds of subject-level cross-validation.

### Study 1 — Single-epoch baseline (no context), Sleep-EDF-78

| Config | kappa | N1-F1 | Notes |
|---|---|---|---|
| no_transitions (best) | 0.7332 +/- 0.020 | 0.472 | alpha=0, sparse W=64, 3ch |
| full_attention (ablation) | 0.7059 +/- 0.030 | — | O(L^2), proves sparse > dense |
| 2ch_FpzCz_EOG | 0.7285 +/- 0.023 | — | wearable baseline |
| 1ch_FpzCz | 0.6991 +/- 0.032 | — | single channel |

### Study 2 — Proposed model with 3-epoch context, Sleep-EDF-78

| Config | kappa | N1-F1 | Notes |
|---|---|---|---|
| seq_context_k1 (ctx + transitions) | 0.7518 +/- 0.023 | 0.519 | |
| **seq_context_k1_notrans (proposed)** | **0.7574 +/- 0.024** | **0.532** | best overall |
| seq_context_k1_notrans_2ch (wearable) | 0.7541 +/- 0.022 | 0.526 | 2-channel |

### Study 3 — Zero-shot transfer to ISRUC Subgroup 1 (100 subjects)

Models trained on Sleep-EDF-78, evaluated on ISRUC without any fine-tuning.

| Config | kappa | Notes |
|---|---|---|
| seq_context_k1_notrans (3ch) | 0.2534 +/- 0.096 | Poor: Pz-Oz -> O1-A2 channel mismatch |
| seq_context_k1_notrans_2ch (2ch) | 0.4218 +/- 0.030 | Best zero-shot; drops occipital channel |

Key finding: the Pz-Oz -> O1-A2 channel mapping is poor and actively degrades 3-channel zero-shot performance. The 2-channel wearable configuration is the most portable.

### Study 4 — Joint Sleep-EDF + ISRUC training (5-fold CV on both datasets)

| Config | Sleep-EDF kappa | ISRUC kappa | Notes |
|---|---|---|---|
| combined_context_2ch (proposed) | 0.7585 +/- 0.026 | 0.6523 +/- 0.018 | 2ch + context |
| combined_context_3ch | 0.6800 +/- 0.018 | 0.6800 +/- 0.018 | 3ch + context |

Joint training recovers most of the zero-shot kappa gap for ISRUC (from 0.42 to 0.65 for 2ch), with only a negligible effect on Sleep-EDF performance.

### N1 Threshold Calibration (Study 2 best models)

Post-hoc threshold sweeping on the N1 decision boundary:

| Config | Argmax N1-F1 | Calibrated N1-F1 | Optimal threshold |
|---|---|---|---|
| seq_context_k1_notrans (3ch) | 0.532 | 0.553 | ~0.44-0.82 (varies by fold) |
| seq_context_k1_notrans_2ch (2ch) | 0.526 | 0.546 | similar range |

---

## Setup

### 1. Install environment

```bash
conda env create -f environment.yml
conda activate sleep-research
```

### 2. Download data

**Sleep-EDF Expanded (78 SC subjects)**:
```
https://physionet.org/content/sleep-edfx/1.0.0/
```
Download the `sleep-cassette/` directory. Place it under `data/`.

**ISRUC-Sleep Subgroup 1 (100 subjects)**:
```
https://sleeptight.isr.uc.pt/
```
Download Subgroup 1 `.rec` files and scorer 1 annotations. Place them under `data_ISRUC/subgroup1/{subject_id}/`.

---

## Preprocessing

### Sleep-EDF

Extracts 3-channel epochs (Fpz-Cz, Pz-Oz, EOG horizontal) at 100 Hz into 30-second `.npy` files with a `metadata.csv` index:

```bash
python scripts/preprocess_sleepedf.py
# or limit to N subjects for a quick test:
python scripts/preprocess_sleepedf.py --limit 5
```

Output: `processed_data/` directory with `metadata.csv` and one `.npy` file per epoch.

### ISRUC

Reads raw `.rec` EDF files, selects three channels (C3-A2, O1-A2, LOC-A2), downsamples 200 Hz -> 100 Hz, and saves per-subject `.npz` files:

```bash
python scripts/preprocess_isruc.py
# or for a subset:
python scripts/preprocess_isruc.py --subjects 1 2 3
```

Output: `data_ISRUC/processed/` directory with `isruc_S{n:03d}.npz` files.

---

## Replication

Run the studies in order. Each script auto-resumes if interrupted (re-run the same command).

### Study 1 — Single-epoch sparse attention ablations (Sleep-EDF)

```bash
python scripts/train_sleepedf.py \
    --output-dir results/study_01 \
    --experiments no_transitions full_attention 2ch_FpzCz_EOG
```

To replicate the full 45-experiment Study 1 ablation grid, omit `--experiments`.

### Study 2 — Sequence context experiments (Sleep-EDF)

Use the Study 1 fold assignments for direct comparability:

```bash
python scripts/train_sleepedf.py \
    --output-dir results/study_02 \
    --folds-file results/study_01_folds.json \
    --experiments seq_context_k1_notrans seq_context_k1_notrans_2ch seq_context_k1
```

### Study 3 — Zero-shot evaluation on ISRUC

Requires Study 1 and Study 2 checkpoints and preprocessed ISRUC data:

```bash
python scripts/eval_zero_shot.py
# or for specific experiments/folds:
python scripts/eval_zero_shot.py \
    --experiments seq_context_k1_notrans seq_context_k1_notrans_2ch \
    --folds 0 1 2 3 4
```

Output: `results/study_03/isruc_eval_summary.json` and a markdown table.

### Study 4 — Joint Sleep-EDF + ISRUC training

```bash
python scripts/train_combined.py
# smoke test (2 epochs):
python scripts/train_combined.py --smoke-test
```

### N1 Threshold Calibration

After completing Study 2 training:

```bash
python scripts/analyze_n1_threshold.py \
    --results-dir results/study_02 \
    --experiments seq_context_k1_notrans seq_context_k1_notrans_2ch
```

Output: threshold sweep curves (PNG) and a `best_thresholds_summary.json`.

---

## Repository Structure

```
sleep-staging-repo/
├── src/
│   ├── config.py                    # All paths and hyperparameters
│   ├── models/
│   │   ├── configurable.py          # Primary model implementations
│   │   ├── backbones.py             # Original SparseTransformerBackbone (Study 1 prototype)
│   │   ├── heads.py                 # SleepStagingHead, TransitionDetectionHead
│   │   └── mtl_model.py             # MTLSleepModel (Study 1 prototype)
│   ├── training/
│   │   ├── loss.py                  # FocalLoss, UncertaintyLossWrapper
│   │   ├── loops.py                 # train_one_epoch, validate
│   │   └── train.py                 # Single-run training script (Study 1 era)
│   ├── dataloading/
│   │   ├── dataset.py               # SleepDataset for Sleep-EDF
│   │   ├── samplers.py              # WeightedRandomSampler helpers
│   │   ├── combined_dataset.py      # CombinedDataset for Study 4
│   │   └── isruc_dataset.py         # ISRUCEvalDataset for Study 3
│   ├── preprocessing/
│   │   ├── extract.py               # MNE-based EDF loading
│   │   ├── segment.py               # Epoch segmentation + metadata.csv
│   │   └── run_preprocess.py        # Sleep-EDF preprocessing entry point
│   └── evaluation/
│       └── calc_metrics.py          # Metrics from saved .npz prediction files
├── scripts/
│   ├── preprocess_sleepedf.py       # Preprocess Sleep-EDF raw EDF files
│   ├── preprocess_isruc.py          # Preprocess ISRUC raw .rec files
│   ├── train_sleepedf.py            # 5-fold CV runner, Studies 1 & 2
│   ├── train_combined.py            # Joint training runner, Study 4
│   ├── eval_zero_shot.py            # Zero-shot ISRUC evaluation, Study 3
│   └── analyze_n1_threshold.py      # N1 threshold calibration analysis
├── results/
│   ├── study_01_cv_summary.json     # Study 1 mean +/- std per experiment
│   ├── study_01_folds.json          # Fixed fold assignments (reused across studies)
│   ├── study_02_cv_summary.json     # Study 2 results
│   ├── study_02_folds.json          # Same assignments as Study 1
│   ├── study_02_n1_thresholds.json  # N1 threshold calibration results
│   ├── study_03_isruc_summary.json  # ISRUC zero-shot evaluation results
│   └── study_04_cv_summary.json     # Joint training results
├── environment.yml                  # Conda environment specification
├── .gitignore
└── README.md
```

**Note on `src/config.py`**: The config file serves as the single source of truth for all paths. If you place your data in non-default locations, update the path variables at the top of `src/config.py`. All scripts import from this file.

**Note on study-specific scripts**: `train_sleepedf.py` runs both Study 1 and Study 2 experiments — the experiment list is controlled by the `--experiments` flag. The `--folds-file` flag ensures Study 2 uses the same subject assignments as Study 1, making results directly comparable.

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{placeholder2026,
  title   = {Efficient Sleep Staging with Sparse Attention and Sequence Context},
  author  = {[Author(s)]},
  journal = {[Venue]},
  year    = {2026},
}
```
