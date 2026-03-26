"""
Custom samplers for imbalanced data.

Classes:
- BalancedSampler: Oversamples minority classes (N1, REM) or undersamples majority.
"""

from typing import List
import numpy as np
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(
    sample_weights: List[float],
    num_samples: int = None,
    replacement: bool = True
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for handling class imbalance.

    Args:
        sample_weights: Per-sample weights (higher = more likely to be sampled).
        num_samples: Number of samples to draw per epoch. Defaults to dataset size.
        replacement: Whether to sample with replacement.

    Returns:
        WeightedRandomSampler instance.
    """
    if num_samples is None:
        num_samples = len(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement
    )


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute inverse frequency weights for each class.

    Args:
        labels: Array of class labels.

    Returns:
        Dictionary mapping class label to weight.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    weights = {label: total / count for label, count in zip(unique, counts)}

    # Normalize so max weight is 1.0
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}

    return weights
