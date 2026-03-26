"""
PyTorch Dataset definition.

Classes:
- SleepDataset(Dataset): Loads preprocessed NumPy files. Handles signal + target.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional

from src.config import PROCESSED_DIR


class SleepDataset(Dataset):
    """
    Dataset for loading preprocessed sleep EEG epochs.

    Reads metadata.csv to map filenames to subject IDs and labels.
    """

    def __init__(
        self,
        metadata_path: Path = None,
        data_dir: Path = None,
        subject_ids: Optional[List[str]] = None,
        transform=None
    ):
        """
        Args:
            metadata_path: Path to metadata.csv file.
            data_dir: Directory containing .npy files.
            subject_ids: Optional list of subject IDs to include (for train/val/test split).
            transform: Optional transform to apply to data.
        """
        if metadata_path is None:
            metadata_path = PROCESSED_DIR / "metadata.csv"
        if data_dir is None:
            data_dir = PROCESSED_DIR

        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # Filter by subject IDs if provided
        if subject_ids is not None:
            self.metadata = self.metadata[
                self.metadata["subject_id"].isin(subject_ids)
            ].reset_index(drop=True)

        # DYNAMICALLY COMPUTE TRANSITION LABELS
        # Ensure proper sorting
        self.metadata["epoch_index"] = self.metadata["epoch_index"].astype(int)
        self.metadata = self.metadata.sort_values(["subject_id", "epoch_index"])

        # Calculate transitions: Change in stage between t and t+1 within the same subject
        # Shift -1 gets the NEXT row's value
        next_stage = self.metadata["stage_label"].shift(-1)
        next_subject = self.metadata["subject_id"].shift(-1)

        # Logic: It IS a transition if:
        # 1. Subject is the same (don't detect transition across subject boundary)
        # 2. Stage is different
        is_transition = (self.metadata["subject_id"] == next_subject) & (self.metadata["stage_label"] != next_stage)

        # Determine filling for last element (False)
        is_transition = is_transition.fillna(False)

        # Assign to metadata (overwriting any existing column)
        self.metadata["transition_label"] = is_transition.astype(int)

        # Compute class weights for balancing
        self._compute_class_weights()

    def _compute_class_weights(self):
        """Compute inverse frequency weights for each class."""
        label_counts = self.metadata["stage_label"].value_counts()
        total = len(self.metadata)
        self.class_weights = {
            label: total / count for label, count in label_counts.items()
        }
        # Normalize weights
        max_weight = max(self.class_weights.values())
        self.class_weights = {
            k: v / max_weight for k, v in self.class_weights.items()
        }

    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        return [
            self.class_weights[label]
            for label in self.metadata["stage_label"]
        ]

    def get_labels(self) -> np.ndarray:
        """Get all labels for the dataset."""
        return self.metadata["stage_label"].values

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - 'signal': torch.Tensor of shape (C, T)
                - 'stage_label': int (0-4 for sleep stages)
                - 'transition_label': int (0 or 1)
                - 'subject_id': str
        """
        row = self.metadata.iloc[idx]

        # Load signal
        filepath = self.data_dir / row["filename"]
        signal = np.load(filepath)

        # Convert to tensor
        signal = torch.from_numpy(signal).float()

        if self.transform is not None:
            signal = self.transform(signal)

        # Per-channel Z-score normalization
        # signal shape: (C, T) - normalize across time dimension (dim=1)
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True)
        signal = (signal - mean) / (std + 1e-6)

        # Clip to handle outliers
        signal = torch.clamp(signal, min=-20.0, max=20.0)

        # Precomputed in __init__
        transition_label = row["transition_label"]

        return {
            "signal": signal,
            "stage_label": int(row["stage_label"]),
            "transition_label": int(transition_label),
            "subject_id": row["subject_id"]
        }


def get_subject_splits(
    metadata_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Split subjects into train/val/test sets (subject-level split).

    Args:
        metadata_path: Path to metadata.csv
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    metadata = pd.read_csv(metadata_path)
    subjects = metadata["subject_id"].unique()

    np.random.seed(seed)
    np.random.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = list(subjects[:n_train])
    val_subjects = list(subjects[n_train:n_train + n_val])
    test_subjects = list(subjects[n_train + n_val:])

    return train_subjects, val_subjects, test_subjects
