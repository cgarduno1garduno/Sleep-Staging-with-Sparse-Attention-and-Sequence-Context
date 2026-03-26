# Study 03 — ISRUC Cross-Dataset Evaluation

Zero-shot transfer: models trained on Sleep-EDF-78, evaluated on ISRUC Subgroup 1 (100 subjects).
Each row is **mean ± std across the 5 fold-checkpoints** evaluated on all ISRUC subjects.

| Experiment | Source | Accuracy | κ | N1-F1 | N1-Recall | Folds |
|---|---|---|---|---|---|---|
| `no_transitions` | study_01 | 43.19%±7.18 | 0.2509±0.0739 | 0.177±0.055 | 0.184±0.093 | 5 |
| `alpha_1_0` | study_01 | 46.35%±4.15 | 0.2840±0.0535 | 0.216±0.027 | 0.212±0.034 | 5 |
| `final_tasa` | study_01 | 44.90%±5.47 | 0.2778±0.0756 | 0.223±0.104 | 0.316±0.226 | 5 |
| `alpha_10_0` | study_01 | 44.51%±10.19 | 0.2633±0.1261 | 0.164±0.068 | 0.156±0.090 | 5 |
| `window_32` | study_01 | 48.54%±1.16 | 0.3186±0.0115 | 0.171±0.056 | 0.157±0.082 | 5 |
| `window_128` | study_01 | 38.38%±8.14 | 0.1745±0.0921 | 0.113±0.082 | 0.109±0.116 | 5 |
| `1ch_FpzCz` | study_01 | 46.68%±3.10 | 0.3321±0.0358 | 0.249±0.044 | 0.439±0.145 | 5 |
| `2ch_FpzCz_EOG` | study_01 | 51.63%±2.24 | 0.3748±0.0194 | 0.303±0.036 | 0.497±0.147 | 5 |
| `full_attention` | study_01 | 46.38%±2.46 | 0.2939±0.0317 | 0.190±0.063 | 0.188±0.094 | 5 |
| `seq_context_k1` | study_02 | 38.97%±7.17 | 0.1895±0.0891 | 0.161±0.110 | 0.178±0.136 | 5 |
| `seq_context_k1_notrans` | study_02 | 44.87%±7.87 | 0.2534±0.0957 | 0.136±0.035 | 0.102±0.029 | 5 |
| `seq_context_k1_notrans_2ch` | study_02 | 55.22%±2.36 | 0.4218±0.0304 | 0.300±0.030 | 0.415±0.085 | 5 |
