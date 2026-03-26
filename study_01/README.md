# Study 01 — Sparse Attention Baseline (5-Fold CV)

**Status:** Complete — 45 experiments (9 configs × 5 folds)
**Dataset:** Sleep-EDF Expanded (78 SC subjects, 3 channels: Fpz-Cz, Pz-Oz, EOG)

---

## What This Study Does

Establishes the baseline architecture and answers two questions:
1. Does sparse windowed attention outperform full O(L²) attention?
2. What is the best channel configuration — 1ch, 2ch, or 3ch?

All 9 configurations use the same 5-fold subject-level CV split (`results/study_01_folds.json`, seed=42). Folds are ~41-42 train / 7 val / 12-13 test subjects each.

---

## Configurations

| Config | Channels | Window | Alpha | Description |
|--------|----------|--------|-------|-------------|
| `no_transitions` | 3 | 64 | 0.0 | Staging only — Study 1 best |
| `final_tasa` | 3 | 64 | 5.0 | Full model with transition task |
| `alpha_1_0` | 3 | 64 | 1.0 | Transition loss sensitivity |
| `alpha_10_0` | 3 | 64 | 10.0 | Transition loss sensitivity |
| `window_32` | 3 | 32 | 5.0 | Smaller attention window |
| `window_128` | 3 | 128 | 5.0 | Larger attention window |
| `2ch_FpzCz_EOG` | 2 | 64 | 5.0 | Fpz-Cz + EOG only |
| `1ch_FpzCz` | 1 | 64 | 5.0 | Single-channel baseline |
| `full_attention` | 3 | None | 5.0 | Dense O(L²) attention |

---

## Key Results

| Experiment | κ ± σ | Acc | N1-F1 |
|---|---|---|---|
| `no_transitions` | 0.7332 ± 0.0196 | 80.6% | 0.472 |
| `window_32` | 0.7285 ± 0.0216 | 80.2% | 0.476 |
| `2ch_FpzCz_EOG` | 0.7285 ± 0.0230 | 80.3% | 0.475 |
| `final_tasa` | 0.7227 ± **0.0086** | 79.8% | 0.472 |
| `full_attention` | 0.7059 ± 0.0299 | 78.3% | 0.440 |
| `1ch_FpzCz` | 0.6991 ± 0.0315 | 78.0% | 0.432 |

Full per-fold analysis: `study_01/results/results_2026-03-08.md`
Aggregated CV metrics: `results/study_01_cv_summary.json`

---

## Key Findings

**Sparse > full attention (Δκ = +0.027)** — Consistent across all 5 folds. Local windowed attention provides a useful inductive bias for sleep EEG structure; long-range attention adds gradient noise without benefit.

**Smaller windows are better** — window_32 > window_64 > window_128. Relevant temporal features (spindles, K-complexes, delta waves) are locally structured within a few seconds.

**2ch ≈ 3ch** — `2ch_FpzCz_EOG` (κ=0.729) nearly matches 3-channel (κ=0.723). Pz-Oz adds marginal value. A frontal EEG + EOG setup is sufficient.

**Transition task stabilizes but doesn't improve** — `final_tasa` has the lowest fold-to-fold variance (σ=0.009) but a lower mean κ than `no_transitions`. The transition head learns real signal (AUC=0.785, confirmed in Study 2) but the gradient competes with staging at N1/N2 boundaries.

**N1 precision ceiling ~0.32–0.36** — No loss modification, checkpoint criterion, or class weighting broke through. This set up the Study 2 hypothesis.

---

## Reproduction

```bash
conda activate sleep-research

# Full 5-fold CV run (Study 1 configurations)
python scripts/run_cv_training.py \
  --output-dir <your-output-dir> \
  --experiments no_transitions final_tasa full_attention window_32 window_128 \
                alpha_1_0 alpha_10_0 1ch_FpzCz 2ch_FpzCz_EOG

# Smoke test (2 epochs, fast)
python scripts/run_cv_training.py --output-dir /tmp/smoke --smoke-test
```

The runner is resumable — re-run the same command with the same `--output-dir` to pick up from where it left off.
