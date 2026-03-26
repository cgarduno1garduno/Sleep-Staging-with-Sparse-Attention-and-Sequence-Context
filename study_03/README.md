# Study 03 — Zero-Shot Cross-Dataset Transfer (ISRUC)

**Status:** Complete
**Dataset:** ISRUC-Sleep Subgroup 1 (100 subjects, sleep-disordered population)

---

## What This Study Does

No new training. The 5 fold-checkpoints from each of the 12 Study 1+2 experiments are evaluated directly on all 100 ISRUC subjects. Results are reported as **mean ± std across the 5 fold-checkpoints** — directly comparable to the within-dataset tables in Studies 1 and 2.

---

## Channel Mapping

ISRUC uses clinical bipolar derivations. The closest matches to the Sleep-EDF channels the model was trained on:

| Model channel | Sleep-EDF (training) | ISRUC equivalent | Match quality |
|---|---|---|---|
| ch 0 | EEG Fpz-Cz | C3-A2 | Reasonable |
| ch 1 | EEG Pz-Oz | O1-A2 | Poor — different anatomy |
| ch 2 | EOG horizontal | LOC-A2 | Reasonable |

Label mapping: ISRUC uses `{0:W, 1:N1, 2:N2, 3:N3, 5:REM}` → remap `5 → 4` to match Sleep-EDF ordering.

---

## Results

| Experiment | Source | κ ± σ | Accuracy | N1-F1 |
|---|---|---|---|---|
| `seq_context_k1_notrans_2ch` | Study 02 | **0.4218 ± 0.030** | 55.2% | 0.300 |
| `2ch_FpzCz_EOG` | Study 01 | 0.3748 ± 0.019 | 51.6% | 0.303 |
| `1ch_FpzCz` | Study 01 | 0.3321 ± 0.036 | 46.7% | 0.249 |
| `full_attention` | Study 01 | 0.2939 ± 0.032 | 46.4% | 0.190 |
| `seq_context_k1_notrans` | Study 02 | 0.2534 ± 0.096 | 44.9% | 0.136 |
| `no_transitions` | Study 01 | 0.2509 ± 0.074 | 43.2% | 0.177 |
| `seq_context_k1` | Study 02 | 0.1895 ± 0.089 | 39.0% | 0.161 |

Within-dataset reference: κ=0.7574 (3ch), κ=0.7541 (2ch). Full results: `study_03/results/isruc_eval_summary.md` and `results/study_03_isruc_summary.json`.

---

## Key Findings

**Channel mismatch is the dominant bottleneck, not population shift.** Every model that drops channel 1 (Pz-Oz → O1-A2) outperforms its 3ch counterpart. The 3ch best model (κ=0.2534) collapses while the 2ch best (κ=0.4218) transfers reasonably — a Δ of 0.17 kappa explained entirely by one bad channel.

**Context amplifies input quality, for better or worse.** For 2ch, context adds Δκ=+0.047 (0.4218 vs 0.3748). For 3ch, context provides no benefit. Sequence context propagates whatever is in the input — useful with clean channels, counterproductive with a mismatched one.

**The 2ch wearable is the most portable configuration.** Fpz-Cz + EOG are the most anatomically transferable signals across recording setups and avoid the problematic occipital channel.

**High variance across fold-checkpoints** (σ=0.03–0.13 vs within-dataset σ≈0.02–0.03) shows cross-dataset generalization depends heavily on which Sleep-EDF training subjects were seen. This is a distributional training problem, addressed in Study 4.

---

## Reproduction

```bash
conda activate sleep-research

# Download ISRUC (if not already done)
bash scripts/download_isruc.sh

# Preprocess
python scripts/preprocess_isruc.py

# Evaluate (requires Study 1 and Study 2 checkpoints from your training runs)
python scripts/eval_zero_shot.py
```
