# Sleep Staging with Sparse Attention and Sequence Context

This repository implements a lightweight 222K-parameter transformer for efficient sleep stage classification. It focuses on cross-dataset generalization and uses epoch-level sequence context to break the N1 staging precision ceiling that loss modifications and multitask learning alone cannot address.

The work evolved across four studies, each answering one question before the next emerged:

| Study | Question |
|---|---|
| **Study 1** | Does sparse within-epoch attention beat dense attention, and what is the best channel setup? |
| **Study 2** | Does epoch-level sequence context break the N1 precision ceiling that no loss modification could fix? |
| **Study 3** | How does the best model generalize zero-shot to an unseen clinical dataset (ISRUC)? |
| **Study 4** | Does joint training recover the cross-dataset gap, and what does the recovery pattern reveal? |

---

## Highlights

1. **Efficiency Wins**: Sparse attention outperforms O(L²) full attention by +0.027 kappa — consistently directional across all 5 folds, not a mean artifact. EEG tokens only need to talk to their neighbors.
2. **The Power of Context**: Adding just one neighboring epoch on each side (90s total) boosted N1-F1 by **13%** and broke the N1 precision ceiling that neither focal loss, class weighting, nor multitask learning could address. N1 is defined sequentially (Wake→N1→N2) — a per-epoch model cannot resolve the ambiguity intrinsic to N1's temporal definition.
3. **Wearable Ready**: We matched 3-channel performance using only **2-channels** (Fpz-Cz + EOG), with Δκ=0.003 — not directionally consistent across folds. A single frontal EEG electrode plus EOG is the minimum AASM-required setup for REM detection and loses nothing compared to full PSG once context is available.
4. **Closing the Gap**: Zero-shot transfer to ISRUC collapsed to κ=0.42 — driven by channel mismatch, not population shift. Joint training on both datasets recovered 60% of the gap (κ=0.68) while preserving within-dataset accuracy. The cross-dataset failure is a training-data problem, not an architectural limit.
5. **Free N1 Gain**: Post-hoc N1 threshold calibration adds +0.021 N1-F1 at zero training cost, confirming systematic model miscalibration that is correctable at inference time.

---

## Architecture

The model is designed to be lean and efficient, processing a sliding window of 3 epochs through a shared backbone.

### 1. Sparse (Local) Attention

Why attend to the whole 3000-sample sequence when you can just look at your neighbors? We use a window size $W=64$, which keeps things fast and lightweight:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

Where $M_{ij} = -\infty$ if $|i-j| > W$. For a sequence of length L=750 (after temporal pooling), full attention costs O(L²) = O(562,500) per layer; sparse windowed attention reduces this to O(L·W) = O(48,000) — a 12× reduction. Sleep EEG features (spindles, K-complexes, delta waves) are locally structured within 1–4 seconds; the long-range receptive field of full attention provides no benefit.

### 2. Epoch-Level Sequence Context

The base model (214K parameters) classifies each 30-second epoch independently. The proposed model extends this by processing a window of 3 consecutive epochs `(prev, center, next)` through the shared backbone. Each epoch's embedding is produced independently (same weights), then all three are concatenated before the staging head:

$$ \hat{y}_t = f_{\text{stage}}\left([e_{t-1} \,\|\, e_t \,\|\, e_{t+1}]\right) \in \mathbb{R}^5 $$

This 8K parameter increase (214K → 222K) yields a +0.024 κ improvement. The mechanism: a morphologically ambiguous epoch that follows a Wake epoch is almost certainly N1. The same epoch following N2 is almost certainly N2. Without context, both are treated identically.

### 3. Addressing N1 Staging (Focal Loss)

N1 is famously difficult to get right. To help the model stay focused on these tricky examples, we use **Multiclass Focal Loss**:

$$ \text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t) $$

With $\gamma=2.0$. Note: focal loss alone cannot break the N1 precision ceiling — stacking it on top of class weighting and composite checkpoint criteria is mathematically redundant and destabilizes training (Study 2b). Context is the fix.

### 4. Multi-Task Learning (Uncertainty Weighting)

We train for both Staging and Transition Detection. To keep the tasks in balance without manual tuning, we use **Homoscedastic Uncertainty Weighting**:

$$ \mathcal{L}_{total} = \sum_{i} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log(\sigma) $$

*(Note: We set $\alpha=0$ for all final staging runs. The transition head does learn real signal — AUC=0.785 — but the gradient it produces competes with staging refinement at N1/N2 boundaries rather than complementing it. The infrastructure is there if you need it, and the failure mechanism is documented in `study_02/README.md`.)*

---

## Results Summary

| Study | Config | Dataset | Kappa | N1-F1 | Notes |
|---|---|---|---|---|---|
| **Study 1** | Single-epoch | Sleep-EDF | 0.7332 | 0.472 | Sparse > Full (+0.027 κ) |
| **Study 2** | **3-Epoch Context (3ch)** | Sleep-EDF | **0.7574** | **0.532** | Proposed baseline |
| **Study 2** | 3-Epoch Context (2ch) | Sleep-EDF | 0.7541 | 0.526 | Wearable config; Δκ=0.003 from 3ch |
| **Study 2** | Context + N1 calibration | Sleep-EDF | — | **0.553** | +0.021 N1-F1 at zero training cost |
| **Study 3** | Zero-Shot (2ch) | ISRUC | 0.4218 | 0.300 | Channel mismatch, not architecture |
| **Study 4** | **Joint Training (3ch)** | Sleep-EDF | **0.7661** | **0.532** | No within-domain forgetting |
| **Study 4** | **Joint Training (3ch)** | ISRUC | **0.6800** | **0.532** | 60% gap recovery; N1-F1 matches |
| **Study 4** | Joint Training (2ch) | Sleep-EDF | 0.7585 | 0.521 | Portable joint config |
| **Study 4** | Joint Training (2ch) | ISRUC | 0.6523 | 0.511 | |

---

## Honest Competitive Positioning

| Model | κ | Params | Channels | Notes |
|---|---|---|---|---|
| **This work — best (3ch, S2)** | **0.7574±0.024** | **222K** | **3** | proposed model |
| **This work — wearable (2ch, S2)** | **0.7541±0.022** | **222K** | **2** | FpzCz+EOG |
| **This work — joint 3ch (S4)** | **0.7661±0.014** | **222K** | **3** | Sleep-EDF (joint training) |
| TinySleepNet | ~0.77 | ~500K | 1 | single EEG, CNN-LSTM |
| SleepTransformer | ~0.79 | large | 1 | full sequence Transformer |
| SPTESleepNet (SOTA) | ~0.87 | large | varies | strip patch embeddings |

This is a paper about **parameter efficiency, sparse attention, and the role of sequence context in N1 detection** — not a claim to SOTA accuracy. At 222K parameters (55% fewer than TinySleepNet), the model reaches within Δκ=0.013 of TinySleepNet while using a two-electrode wearable-compatible channel setup. The gap to SOTA (κ=0.87) is real and should not be minimized.

---

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate sleep-research
```

### 2. Data Preparation

You'll need **Sleep-EDF** (78 SC subjects) and **ISRUC-Sleep** (Subgroup 1). Download ISRUC automatically:

```bash
bash scripts/download_isruc.sh
```

Place Sleep-EDF in `data/` and run preprocessing:

```bash
python scripts/preprocess_sleepedf.py
python scripts/preprocess_isruc.py
```

### 3. Replication

```bash
# Study 1 — 5-fold CV, 9 configurations (45 experiments)
python scripts/run_cv_training.py --output-dir results/study_01

# Study 2 — Sequence context models (reuse Study 1 fold assignments)
python scripts/train_sleepedf.py --output-dir results/study_02 --folds-file results/study_01_folds.json

# Study 3 — Zero-shot ISRUC evaluation (no new training)
python scripts/eval_zero_shot.py

# Study 4 — Joint training on Sleep-EDF + ISRUC
python scripts/train_combined.py --output-dir results/study_04

# Generate publication figures
python scripts/generate_publication_figures.py --output-dir results/figures
```

### 4. N1 Threshold Calibration

Post-hoc calibration consistently adds +0.02 N1-F1 at zero training cost:

```bash
python scripts/analyze_n1_threshold.py --results-dir results/study_02
```

---

## Repository Structure

```text
sleep-staging-repo/
├── src/
│   ├── models/              # Model implementations
│   │   ├── configurable.py  # Primary model (ConfigurableTASA + ContextTASA)
│   │   ├── backbones.py     # Sparse Transformer backbone
│   │   ├── heads.py         # Staging & Transition heads
│   │   └── mtl_model.py     # Multi-Task Model wrapper
│   ├── training/            # Focal Loss & Uncertainty Weighting
│   ├── dataloading/         # Dataset & Sampler helpers (Sleep-EDF + ISRUC)
│   ├── preprocessing/       # Signal extraction & segmentation
│   ├── visualization/       # Confusion matrices, hypnograms, transition heatmaps
│   └── evaluation/          # Metrics and calculation logic
├── scripts/                 # Entry points for all experiments
│   ├── run_cv_training.py   # 5-fold CV runner (Studies 1 & 2)
│   ├── train_sleepedf.py    # Study 2 context model training
│   ├── train_combined.py    # Study 4 joint training
│   ├── eval_zero_shot.py    # Study 3 ISRUC zero-shot evaluation
│   ├── analyze_n1_threshold.py  # Post-hoc N1 calibration
│   ├── generate_publication_figures.py
│   ├── preprocess_sleepedf.py
│   ├── preprocess_isruc.py
│   └── download_isruc.sh
├── study_01/                # Study 1: frozen, complete (45 experiments)
│   ├── README.md            # Full experiment log and findings
│   └── results/
│       └── results_2026-03-08.md  # Complete per-fold analysis
├── study_02/                # Study 2: sequence context ablations
│   └── README.md            # Progression from pilot to 5-fold
├── study_03/                # Study 3: zero-shot ISRUC evaluation
│   ├── README.md
│   └── results/
│       └── isruc_eval_summary.md
├── study_04/                # Study 4: joint training
│   └── README.md
├── results/                 # JSON summaries for all 4 studies
├── environment.yml          # Conda recipe
└── README.md                # You are here
```

**Single Source of Truth**: All paths and hyperparameters live in `src/config.py`. Update it once, and the rest of the scripts will follow.

---

## Author

**Cris Garduno**
[cgarduno1garduno](https://github.com/cgarduno1garduno)

If you use this code or findings in your project, please credit this repository.

---

*Claude Code was used to assist throughout various parts of this work — including experimental design, debugging, analysis, and documentation.*
