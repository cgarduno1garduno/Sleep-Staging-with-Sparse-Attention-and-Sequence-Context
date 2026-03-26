import mne
import numpy as np
from typing import List, Optional
from src.config import CHANNELS, SAMPLING_RATE

def load_edf(path: str) -> mne.io.Raw:
    """
    Load EDF file using MNE.

    Args:
        path: Path to .edf file.

    Returns:
        mne.io.Raw object.
    """
    # preload=True to load into memory
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return raw

def filter_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Select and filter specific channels (EEG, EOG).
    Resamples to global SAMPLING_RATE.

    Args:
        raw: mne.io.Raw object.

    Returns:
        Modified raw object.
    """
    # Sleep-EDF Expanded channel names often vary slightly or include other signals.
    # Common names: 'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'
    # Sometimes 'FPZ-CZ', 'PZ-OZ', etc.

    # We will pick channels primarily by mapping known variants to our CONFIG names.
    # This is a robust way to handle Sleep-EDF inconsistencies.

    target_channels = CHANNELS # ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

    # Find matching channels in raw
    found_channels = []
    missing_channels = []

    # Simple case-insensitive matching logic or exact match
    for target in target_channels:
        if target in raw.ch_names:
            found_channels.append(target)
        else:
            missing_channels.append(target)

    if missing_channels:
        raise ValueError(f"Missing channels {missing_channels} in {raw.filenames[0]}. Available: {raw.ch_names}")

    # Pick channels
    raw.pick_channels(found_channels)

    # Reorder to match config order
    raw.reorder_channels(target_channels)

    # Resample if necessary
    if raw.info['sfreq'] != SAMPLING_RATE:
        raw.resample(SAMPLING_RATE, npad="auto")

    return raw
