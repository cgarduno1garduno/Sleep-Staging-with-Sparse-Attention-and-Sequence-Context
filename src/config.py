"""
Configuration for the sleep staging project.

Covers both Sleep-EDF (Studies 1 & 2) and ISRUC (Studies 3 & 4).
Paths are resolved relative to the project root so the repo works regardless
of where it is cloned.
"""

from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Sleep-EDF paths
# ---------------------------------------------------------------------------
DATA_DIR           = PROJECT_ROOT / "data"                  # raw .edf files
PROCESSED_DIR      = PROJECT_ROOT / "processed_data"        # preprocessed epochs
SLEEPEDF_PROC_DIR  = PROCESSED_DIR
SLEEPEDF_META      = PROCESSED_DIR / "metadata.csv"
RESULTS_DIR        = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# ISRUC paths
# ---------------------------------------------------------------------------
ISRUC_RAW_DIR  = PROJECT_ROOT / "data_ISRUC" / "subgroup1"
ISRUC_PROC_DIR = PROJECT_ROOT / "data_ISRUC" / "processed"

# ---------------------------------------------------------------------------
# Signal parameters (must match preprocessing)
# ---------------------------------------------------------------------------
SAMPLING_RATE  = 100          # Hz
EPOCH_LENGTH   = 30           # seconds
EPOCH_SAMPLES  = SAMPLING_RATE * EPOCH_LENGTH  # 3000

# ---------------------------------------------------------------------------
# Sleep-EDF channel configuration
# ---------------------------------------------------------------------------
# Channel names as they appear in the raw .edf files
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

# Channel index sets for selecting subsets:
#   [0] Fpz-Cz  (central EEG, best single-channel)
#   [1] Pz-Oz   (occipital EEG)
#   [2] EOG     (horizontal electrooculogram)
CHANNEL_INDICES = {
    1: [0],        # 1-channel: Fpz-Cz only
    2: [0, 2],     # 2-channel: Fpz-Cz + EOG  (wearable-friendly)
    3: [0, 1, 2],  # 3-channel: all three
}

# Convenience aliases used in combined training (Study 4)
CHANNEL_INDICES_2CH = [0, 2]
CHANNEL_INDICES_3CH = [0, 1, 2]

# ---------------------------------------------------------------------------
# ISRUC channel configuration
# ---------------------------------------------------------------------------
# ISRUC Subgroup 1 raw channel order (in the .rec files):
#   0: LOC-A2  (left EOG)
#   1: ROC-A1  (right EOG)
#   2: F3-A2
#   3: C3-A2   <- closest analog to Fpz-Cz (central EEG)
#   4: O1-A2   <- closest analog to Pz-Oz (occipital EEG, poor mapping)
#   5: C4-A1
#   6: O2-A1
#   ...
ISRUC_CHANNEL_INDICES = [3, 4, 0]   # C3-A2, O1-A2, LOC-A2
ISRUC_CHANNEL_NAMES   = ["C3-A2", "O1-A2", "LOC-A2"]
N_SUBJECTS            = 100

# ---------------------------------------------------------------------------
# Study 4 experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # 2ch: best Study-2 config - joint training without the problematic channel
    {"name": "combined_context_2ch",   "channels": 2, "channel_indices": CHANNEL_INDICES_2CH, "window": 64, "alpha": 0.0, "context_window": 1},
    # 2ch baseline: no context
    {"name": "combined_nocontext_2ch", "channels": 2, "channel_indices": CHANNEL_INDICES_2CH, "window": 64, "alpha": 0.0, "context_window": 0},
    # 3ch: includes the occipital channel - zero-shot failed (Study 3: kappa=0.25) but joint
    # training may allow the model to learn cross-dataset channel invariance
    {"name": "combined_context_3ch",   "channels": 3, "channel_indices": CHANNEL_INDICES_3CH, "window": 64, "alpha": 0.0, "context_window": 1},
]

# Fold assignments: Sleep-EDF reuses study_01 folds for direct comparability
SLEEPEDF_FOLDS_FILE = PROJECT_ROOT / "results" / "study_01_folds.json"

# Study 04 outputs
STUDY04_RESULTS = PROJECT_ROOT / "results" / "study_04"

# Legacy paths for study_03 eval script (resolves from study_03 src/config)
STUDY01_RESULTS = PROJECT_ROOT / "results" / "study_01"
STUDY02_RESULTS = PROJECT_ROOT / "results" / "study_02"
STUDY03_RESULTS = PROJECT_ROOT / "results" / "study_03"
