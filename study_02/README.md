# Study 02 — Sequence Context

**Status:** Complete — 3 configurations × 5 folds
**Dataset:** Sleep-EDF Expanded (same fold assignments as Study 1)

---

## What This Study Does

Study 1 hit a hard N1 precision ceiling (~0.32–0.36) that no loss modification could break. Study 2 investigates whether epoch-level sequence context — giving the model visibility into neighboring epochs — resolves the ambiguity that single-epoch classification cannot.

The study progressed through single-fold pilots before committing to full 5-fold runs:

**2a — Criterion change (pilot, no effect):** Switching the checkpoint criterion from `κ + 0.3·N1-recall` to `κ + 0.3·N1-F1` had zero effect. The two criteria selected the same checkpoint every epoch. The ceiling is not a checkpoint-selection problem.

**2b — Focal loss γ=2 (pilot, failed):** Adding focal loss on top of existing N1 class weighting and composite criterion caused training instability (val loss oscillated 0.57–1.27) and lowered both κ and N1-F1. Three simultaneous N1-boosting mechanisms (class weight + composite criterion + focal loss) all push toward higher recall at the cost of precision — redundant and destabilizing.

**2c — 3-epoch context window (pilot, breakthrough):** The model now processes `(prev, center, next)` epochs through the shared backbone. Their embeddings are concatenated before the staging head. N1 precision jumped from 0.373 → 0.411 on fold_1.

**2d — Context + α=0 (pilot, further improvement):** Removing the transition auxiliary task while keeping context gave the best single-fold result. The transition gradient competes with staging refinement precisely at N1/N2 boundaries — the hardest examples for both tasks simultaneously.

---

## Configurations (Full 5-fold)

| Config | Channels | Context | Alpha | Description |
|--------|----------|---------|-------|-------------|
| `seq_context_k1` | 3 | ±1 epoch | 5.0 | Context + transition task |
| `seq_context_k1_notrans` | 3 | ±1 epoch | 0.0 | **Proposed best model** |
| `seq_context_k1_notrans_2ch` | 2 | ±1 epoch | 0.0 | Wearable config (Fpz-Cz + EOG) |

---

## Key Results

| Model | κ ± σ | N1-F1 | N1-prec | N1-rec | Acc |
|---|---|---|---|---|---|
| `seq_context_k1` | 0.7518 ± 0.023 | 0.519 | 0.390 | 0.787 | 81.9% |
| `seq_context_k1_notrans_2ch` | 0.7541 ± 0.022 | 0.526 | 0.408 | 0.748 | 82.3% |
| **`seq_context_k1_notrans`** | **0.7574 ± 0.024** | **0.532** | **0.420** | **0.745** | **82.6%** |

**2ch vs 3ch: Δκ = 0.003** — not directionally consistent across all folds. A wearable (Fpz-Cz + EOG) loses essentially nothing compared to full PSG once context is available.

**N1 threshold calibration** adds +0.021 N1-F1 at zero training cost:

| Model | Argmax N1-F1 | Calibrated N1-F1 | Gain |
|---|---|---|---|
| `seq_context_k1_notrans` (3ch) | 0.532 | **0.553** | +0.021 |
| `seq_context_k1_notrans_2ch` (2ch) | 0.526 | **0.546** | +0.020 |

Aggregated CV metrics: `results/study_02_cv_summary.json`
Fold assignments: `results/study_02_folds.json`
N1 calibration thresholds: `results/study_02_n1_thresholds.json`

---

## Why Context Worked

N1 is defined sequentially by AASM scoring rules — it follows Wake and precedes N2. A model classifying each 30-second epoch independently cannot observe this. A morphologically ambiguous epoch that follows a Wake epoch is almost certainly N1; the same epoch following N2 is almost certainly N2. Adding `(prev, center, next)` embeddings makes this distinction directly observable, which is why precision improved (+13% N1-F1) without sacrificing recall.

---

## Reproduction

```bash
conda activate sleep-research

# Full 5-fold run, reusing Study 1 fold assignments
python scripts/run_cv_training.py \
  --output-dir <your-output-dir> \
  --folds-file results/study_01_folds.json \
  --experiments seq_context_k1 seq_context_k1_notrans seq_context_k1_notrans_2ch

# Post-hoc N1 threshold calibration
python scripts/analyze_n1_threshold.py --results-dir <your-output-dir>
```
