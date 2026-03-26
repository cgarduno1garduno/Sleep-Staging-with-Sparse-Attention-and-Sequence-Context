"""
Module for segmenting continuous signals into epochs.

Functions:
- segment_signals(raw, hypnogram_path): Cut signals into 30s epochs aligned with stage labels.
- save_segments_with_metadata(): Save .npy files with accompanying metadata.csv.
"""

import csv
import mne
import numpy as np
from pathlib import Path
from typing import List, Tuple
from src.config import PROCESSED_DIR, EPOCH_LENGTH

# Sleep Stage Mapping (AASM)
# Sleep-EDF uses:
# 'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R', 'Movement time', 'Sleep stage ?'
STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3, # Merge N3 and N4
    "Sleep stage R": 4,
    "Movement time": -1, # Exclude
    "Sleep stage ?": -1  # Exclude
}

def segment_signals(raw: mne.io.Raw, hypnogram_path: str) -> List[Tuple[np.ndarray, int]]:
    """
    Segment raw signal into epochs based on hypnogram annotations.

    Args:
        raw: Preprocessed MNE Raw object (Fs=100Hz).
        hypnogram_path: Path to hypnogram .edf file.

    Returns:
        List of tuples (epoch_data, label).
        epoch_data: (C, T) numpy array.
        label: int (0-4).
    """
    # 1. Read Annotations
    annot = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annot, emit_warning=False)

    # 2. Event Extraction
    # MNE events_from_annotations maps unique descriptions to integers (event_id)
    events, event_id = mne.events_from_annotations(raw, event_id=None, chunk_duration=30.)

    # 3. Create Epochs
    # We want to map the annotation descriptions to our classes
    # event_id is like {'Sleep stage W': 1, 'Sleep stage 1': 2, ...}

    # Create a mapping from MNE event_id to OUR labels
    # Inverse event_id map: int -> description
    id_to_desc = {v: k for k, v in event_id.items()}

    # Filter valid epochs
    epochs_data = [] # List of (data, label)

    # Create MNE Epochs object
    # tmin=0, tmax=30 - 1/Fs
    tmax = 30. - 1. / raw.info['sfreq']

    # We can use mne.Epochs to cut data
    # IMPORTANT: Sleep-EDF hypnograms cover a specific range.
    # MNE's chunk_duration handles the repetition of annotations for us.

    mne_epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Iterate and map
    for i, event_code in enumerate(mne_epochs.events[:, 2]):
        desc = id_to_desc[event_code]

        # Check mapping
        if desc in STAGE_MAPPING:
            label = STAGE_MAPPING[desc]
            if label != -1: # Valid class
                # Get data: (Channels, Time)
                # copy() is important
                data = mne_epochs[i].get_data(copy=True)[0]

                # Check shape correctness (should be 3000 pts)
                expected_pts = int(EPOCH_LENGTH * raw.info['sfreq'])
                if data.shape[1] == expected_pts:
                     epochs_data.append((data.astype(np.float32), label))

    return epochs_data


def save_segments_with_metadata(
    segments: List[Tuple[np.ndarray, int]],
    subject_id: str,
    output_dir: Path = None
) -> Path:
    """
    Save segmented epochs as .npy files and generate metadata.csv for traceability.

    Args:
        segments: List of (epoch_array, stage_label) tuples.
        subject_id: Hyphenated subject ID (e.g., "SC4001E0").
        output_dir: Directory to save files. Defaults to PROCESSED_DIR.

    Returns:
        Path to the generated metadata.csv file.
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    file_exists = metadata_path.exists()

    with open(metadata_path, "a", newline="") as csvfile:
        fieldnames = ["filename", "subject_id", "epoch_index", "stage_label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for idx, (epoch_data, stage_label) in enumerate(segments):
            filename = f"{subject_id}_epoch_{idx:04d}.npy"
            filepath = output_dir / filename

            np.save(filepath, epoch_data)

            writer.writerow({
                "filename": filename,
                "subject_id": subject_id,
                "epoch_index": idx,
                "stage_label": stage_label
            })

    return metadata_path
