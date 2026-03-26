"""
Combined Sleep-EDF + ISRUC dataset for Study 04.

Both datasets are loaded in 2-channel mode [central_EEG, EOG]:
  Sleep-EDF: channels [0, 2] = [Fpz-Cz, EOG horizontal]
  ISRUC:     channels [0, 2] = [C3-A2,  LOC-A2]

Each subject's epochs are loaded lazily (memory-mapped via np.load).
The dataset supports both per-epoch (no context) and context-window modes,
matching the behaviour of FoldDataset / ContextFoldDataset from Studies 1 & 2.
"""

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    SLEEPEDF_PROC_DIR, ISRUC_PROC_DIR,
    CHANNEL_INDICES_2CH, EPOCH_SAMPLES,
)


def _zscore(sig: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score normalisation, clipped to +-20."""
    mean = sig.mean(dim=1, keepdim=True)
    std  = sig.std(dim=1, keepdim=True)
    return torch.clamp((sig - mean) / (std + 1e-6), -20.0, 20.0)


# ---------------------------------------------------------------------------
# Sleep-EDF subject loader
# ---------------------------------------------------------------------------

class _SleepEDFSubject:
    """Lazy loader for one Sleep-EDF subject (individual .npy epoch files)."""

    def __init__(self, subject_id: str, meta: pd.DataFrame, data_dir: Path,
                 channel_indices: List[int]):
        rows = meta[meta["subject_id"] == subject_id].sort_values("epoch_index")
        self.subject_id      = subject_id
        self.filepaths       = [data_dir / r["filename"] for _, r in rows.iterrows()]
        self.labels          = rows["stage_label"].values.astype(np.int64)
        self.source          = "sleepedf"
        self._channel_indices = channel_indices

    def __len__(self):
        return len(self.labels)

    def load_epoch(self, ep: int) -> torch.Tensor:
        sig = np.load(self.filepaths[ep])[self._channel_indices, :]
        return _zscore(torch.from_numpy(sig).float())


# ---------------------------------------------------------------------------
# ISRUC subject loader
# ---------------------------------------------------------------------------

class _ISRUCSubject:
    """Loader for one ISRUC subject (.npz with all epochs pre-stacked)."""

    def __init__(self, subject_id: int, proc_dir: Path, channel_indices: List[int]):
        self.subject_id       = subject_id
        self.source           = "isruc"
        self._channel_indices = channel_indices
        data = np.load(proc_dir / f"isruc_S{subject_id:03d}.npz")
        self._signals = data["signals"]   # (N, 3, 3000) float32
        self.labels   = data["labels"]    # (N,)

    def __len__(self):
        return len(self.labels)

    def load_epoch(self, ep: int) -> torch.Tensor:
        sig = torch.from_numpy(self._signals[ep][self._channel_indices, :].copy()).float()
        return _zscore(sig)


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):
    """
    Combined Sleep-EDF + ISRUC dataset, 2-channel mode.

    Args:
        sleepedf_subjects: List of Sleep-EDF subject_id strings to include.
        isruc_subjects:    List of ISRUC subject IDs (ints) to include.
        context_window:    0 = per-epoch; 1 = (prev, center, next) triplet.
        sleepedf_proc_dir: Override for Sleep-EDF processed data dir.
        isruc_proc_dir:    Override for ISRUC processed data dir.
    """

    def __init__(
        self,
        sleepedf_subjects: List[str],
        isruc_subjects: List[int],
        context_window: int = 0,
        channel_indices: Optional[List[int]] = None,
        sleepedf_proc_dir: Optional[Path] = None,
        isruc_proc_dir: Optional[Path] = None,
    ):
        sleepedf_dir = Path(sleepedf_proc_dir or SLEEPEDF_PROC_DIR)
        isruc_dir    = Path(isruc_proc_dir    or ISRUC_PROC_DIR)
        ch_idx       = channel_indices if channel_indices is not None else CHANNEL_INDICES_2CH

        self.context_window = context_window
        self._subjects: list = []
        self._epoch_map: List[tuple] = []   # (subj_idx, ep_within_subj)

        # --- Load Sleep-EDF subjects ---
        if sleepedf_subjects:
            meta = pd.read_csv(sleepedf_dir / "metadata.csv")
            for sid in sleepedf_subjects:
                subj = _SleepEDFSubject(sid, meta, sleepedf_dir, ch_idx)
                si = len(self._subjects)
                self._subjects.append(subj)
                for ep in range(len(subj)):
                    self._epoch_map.append((si, ep))

        # --- Load ISRUC subjects ---
        for sid in isruc_subjects:
            subj = _ISRUCSubject(sid, isruc_dir, ch_idx)
            si = len(self._subjects)
            self._subjects.append(subj)
            for ep in range(len(subj)):
                self._epoch_map.append((si, ep))

        # Per-subject iloc bounds for context clamping
        self._bounds = {}
        start = 0
        for si, subj in enumerate(self._subjects):
            n = len(subj)
            self._bounds[si] = (start, start + n - 1)
            start += n

        # Class weights for WeightedRandomSampler
        all_labels = np.array([self._subjects[si].labels[ep]
                                for si, ep in self._epoch_map])
        counts = np.bincount(all_labels, minlength=5).astype(float)
        counts = np.where(counts == 0, 1, counts)
        w = len(all_labels) / counts
        w /= w.max()
        self._sample_weights = [float(w[self._subjects[si].labels[ep]])
                                 for si, ep in self._epoch_map]

    def get_sample_weights(self) -> List[float]:
        return self._sample_weights

    def __len__(self) -> int:
        return len(self._epoch_map)

    def _load(self, si: int, ep: int) -> torch.Tensor:
        return self._subjects[si].load_epoch(ep)

    def __getitem__(self, idx: int) -> dict:
        si, ep = self._epoch_map[idx]
        lo, hi = self._bounds[si]
        label  = int(self._subjects[si].labels[ep])

        if self.context_window == 0:
            signal = self._load(si, ep)   # (2, 3000)
        else:
            frames = []
            for offset in range(-self.context_window, self.context_window + 1):
                ctx_ep = max(0, min(hi - lo, ep + offset))
                frames.append(self._load(si, ctx_ep))
            signal = torch.stack(frames, dim=0)   # (2K+1, 2, 3000)

        return {
            "signal":      signal,
            "stage_label": label,
            "source":      self._subjects[si].source,
        }
