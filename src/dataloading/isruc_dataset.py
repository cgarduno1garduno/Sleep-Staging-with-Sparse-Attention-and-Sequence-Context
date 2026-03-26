"""
ISRUC-Sleep Subgroup 1 dataset loader for cross-dataset evaluation.

Loads preprocessed per-subject .npz files produced by scripts/preprocess_isruc.py.
Supports both per-epoch (ConfigurableTASA) and context-window (ContextTASA) modes,
using the same z-score normalization and channel-selection logic as the training code.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional

from src.config import ISRUC_PROC_DIR, CHANNEL_INDICES, EPOCH_SAMPLES


class ISRUCEvalDataset(Dataset):
    """
    Dataset for evaluating trained models on all ISRUC Subgroup 1 subjects.

    Each item is one 30-second epoch. For context models the item is a window
    of (2*context_window + 1) consecutive epochs, clamped at subject boundaries
    (same behaviour as ContextFoldDataset in training).

    Args:
        proc_dir:       Directory containing isruc_S{n:03d}.npz files.
        num_channels:   1, 2, or 3 - selects channels via CHANNEL_INDICES.
        context_window: 0 = single epoch; 1 = (prev, center, next) triplet, etc.
        subject_ids:    Optional list of subject IDs (ints) to restrict to.
                        None = all available subjects.
    """

    def __init__(
        self,
        proc_dir: Path = None,
        num_channels: int = 3,
        context_window: int = 0,
        subject_ids: Optional[List[int]] = None,
    ):
        if proc_dir is None:
            proc_dir = ISRUC_PROC_DIR

        self.proc_dir       = Path(proc_dir)
        self.ch_idx         = CHANNEL_INDICES[num_channels]
        self.context_window = context_window

        # Discover available subjects
        npz_files = sorted(self.proc_dir.glob("isruc_S*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No isruc_S*.npz files found in {self.proc_dir}. "
                "Run scripts/preprocess_isruc.py first."
            )

        # Load all subjects into memory (they're small after preprocessing:
        # ~900 epochs x 3 channels x 3000 samples x 4 bytes ~= 32 MB per subject,
        # 100 subjects ~= 3.2 GB -- too large for full preload.
        # Instead we memory-map the arrays and build an epoch index.)
        self._subjects: List[dict] = []   # {signals: (N,3,T), labels: (N,)}
        self._epoch_map: List[tuple] = []  # (subj_idx, epoch_within_subj)

        for path in npz_files:
            sid_str = path.stem.replace("isruc_S", "")
            try:
                sid = int(sid_str)
            except ValueError:
                continue

            if subject_ids is not None and sid not in subject_ids:
                continue

            data = np.load(path)
            signals = data["signals"]   # (N, 3, 3000) float32
            labels  = data["labels"]    # (N,)         int64

            subj_idx = len(self._subjects)
            self._subjects.append({"id": sid, "signals": signals, "labels": labels})

            for ep in range(len(labels)):
                self._epoch_map.append((subj_idx, ep))

        if not self._subjects:
            raise ValueError(
                f"No subjects loaded from {self.proc_dir}. "
                f"subject_ids filter={subject_ids}"
            )

    def __len__(self) -> int:
        return len(self._epoch_map)

    def _load_epoch(self, subj_idx: int, ep_idx: int) -> torch.Tensor:
        """Load one epoch, apply channel selection and z-score normalisation."""
        sig = self._subjects[subj_idx]["signals"][ep_idx]  # (3, 3000) float32
        sig = torch.from_numpy(sig[self.ch_idx, :])         # (C, 3000)
        mean = sig.mean(dim=1, keepdim=True)
        std  = sig.std(dim=1, keepdim=True)
        return torch.clamp((sig - mean) / (std + 1e-6), -20.0, 20.0)

    def __getitem__(self, idx: int) -> dict:
        subj_idx, ep_idx = self._epoch_map[idx]
        n_epochs = len(self._subjects[subj_idx]["labels"])
        label    = int(self._subjects[subj_idx]["labels"][ep_idx])
        subj_id  = self._subjects[subj_idx]["id"]

        if self.context_window == 0:
            signal = self._load_epoch(subj_idx, ep_idx)   # (C, T)
        else:
            # Context window: clamp at subject boundaries (never cross to another subject)
            frames = []
            for offset in range(-self.context_window, self.context_window + 1):
                ctx_ep = max(0, min(n_epochs - 1, ep_idx + offset))
                frames.append(self._load_epoch(subj_idx, ctx_ep))
            signal = torch.stack(frames, dim=0)  # (2K+1, C, T)

        return {
            "signal":      signal,
            "stage_label": label,
            "subject_id":  subj_id,
        }

    @property
    def subject_ids(self) -> List[int]:
        return [s["id"] for s in self._subjects]

    @property
    def n_subjects(self) -> int:
        return len(self._subjects)
