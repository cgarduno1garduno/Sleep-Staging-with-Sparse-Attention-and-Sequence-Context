#!/usr/bin/env python3
"""
Preprocess ISRUC-Sleep Subgroup 1 for cross-dataset evaluation.

Reads raw .rec EDF files, selects the three channels closest to those the
model was trained on, downsamples 200 Hz → 100 Hz, segments into 30-second
epochs, and remaps sleep stage labels to match the training scheme.

Channel mapping (ISRUC raw → model channel order):
  Model ch 0 (Fpz-Cz analog) → C3-A2  (raw channel index 3)
  Model ch 1 (Pz-Oz analog)  → O1-A2  (raw channel index 4)
  Model ch 2 (EOG analog)    → LOC-A2 (raw channel index 0)

Label mapping:
  ISRUC uses R&K legacy numbering: {0:W, 1:N1, 2:N2, 3:N3, 5:REM}
  (label 4 = N4/S4 is absent; it was merged into N3 under AASM 2007)
  Model was trained on: {0:W, 1:N1, 2:N2, 3:N3, 4:REM}
  Remap: 5 → 4

Output:
  data_ISRUC/processed/isruc_S{n:03d}.npz  — one file per subject
    signals: (N_epochs, 3, 3000) float32
    labels:  (N_epochs,)         int64
  data_ISRUC/processed/metadata.csv

Usage (from project root):
  python study_03/scripts/preprocess_isruc.py
  python study_03/scripts/preprocess_isruc.py --subjects 1 2 3   # subset
  python study_03/scripts/preprocess_isruc.py --overwrite         # redo all
"""

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np
import scipy.signal

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "study_03"))

from src.config import (
    ISRUC_RAW_DIR, ISRUC_PROC_DIR, ISRUC_CHANNEL_INDICES, ISRUC_CHANNEL_NAMES,
    N_SUBJECTS, SAMPLING_RATE, EPOCH_SAMPLES
)

# Raw sampling rate in ISRUC subgroup 1
SRC_FS = 200
# Expected samples per EDF record at 200 Hz, 2-second records
EXPECTED_SPR = SRC_FS * 2   # 400

# Label remap: ISRUC R&K → our 0-4 scheme
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}


# ---------------------------------------------------------------------------
# Minimal EDF reader (handles .rec extension that MNE rejects)
# ---------------------------------------------------------------------------

def _read_edf_channels(path: Path, channel_indices: list) -> tuple:
    """
    Read selected channels from an EDF/REC file without external EDF libraries.

    Returns:
        channels: dict {raw_ch_idx: np.ndarray shape (n_total_samples,) float32}
        n_records: int
        spr: list of int — samples per record for each signal
    """
    with open(path, "rb") as f:
        raw_header = f.read(256)
        n_signals  = int(raw_header[252:256].decode("ascii").strip())
        sig_hdr    = f.read(n_signals * 256)
        data_bytes = f.read()

    # ---- Parse signal header fields ----
    # EDF signal header layout (each field is N × width bytes):
    #   label         : N × 16    offset 0
    #   transducer    : N × 80    offset N*16
    #   phys_dim      : N × 8     offset N*96
    #   phys_min      : N × 8     offset N*104
    #   phys_max      : N × 8     offset N*112
    #   dig_min       : N × 8     offset N*120
    #   dig_max       : N × 8     offset N*128
    #   prefiltering  : N × 80    offset N*136
    #   spr           : N × 8     offset N*216
    #   reserved      : N × 32    offset N*224
    N = n_signals

    def _parse(off, width, dtype):
        return [
            dtype(sig_hdr[off + i * width: off + (i + 1) * width].decode("ascii").strip())
            for i in range(N)
        ]

    spr      = _parse(N * 216, 8, int)
    phys_min = _parse(N * 104, 8, float)
    phys_max = _parse(N * 112, 8, float)
    dig_min  = _parse(N * 120, 8, int)
    dig_max  = _parse(N * 128, 8, int)

    n_records    = int(raw_header[236:244].decode("ascii").strip())
    record_size  = sum(spr)

    # Parse entire data section as int16, reshape to (n_records, record_size)
    all_data = np.frombuffer(data_bytes, dtype="<i2")
    if len(all_data) < n_records * record_size:
        # Truncated file — use what we have
        n_records = len(all_data) // record_size
    data_2d = all_data[: n_records * record_size].reshape(n_records, record_size)

    # Cumulative channel start positions within each record
    ch_starts = np.cumsum([0] + spr[:-1])

    channels = {}
    for ci in channel_indices:
        start   = int(ch_starts[ci])
        end     = start + spr[ci]
        digital = data_2d[:, start:end].flatten().astype(np.float32)

        # Scale digital → physical (μV, etc.)
        d_range = dig_max[ci] - dig_min[ci]
        if d_range != 0:
            gain   = (phys_max[ci] - phys_min[ci]) / d_range
            offset = phys_max[ci] - gain * dig_max[ci]
            digital = digital * gain + offset

        channels[ci] = digital

    return channels, n_records, spr


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def preprocess_subject(subj_id: int, out_dir: Path) -> dict:
    """
    Process one ISRUC subject.

    Returns a metadata dict (or raises on error).
    """
    rec_path = ISRUC_RAW_DIR / str(subj_id) / f"{subj_id}.rec"
    ann_path = ISRUC_RAW_DIR / str(subj_id) / f"{subj_id}_1.txt"   # scorer 1

    if not rec_path.exists():
        raise FileNotFoundError(f"Recording not found: {rec_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    # ---- Read EDF channels ----
    channels, n_records, spr = _read_edf_channels(rec_path, ISRUC_CHANNEL_INDICES)

    # Verify sampling rate assumption
    for ci in ISRUC_CHANNEL_INDICES:
        if spr[ci] != EXPECTED_SPR:
            raise ValueError(
                f"Subject {subj_id} channel {ci}: expected {EXPECTED_SPR} "
                f"samples/record, got {spr[ci]}"
            )

    # ---- Downsample 200 Hz → 100 Hz (factor 2) ----
    # resample_poly applies an anti-aliasing FIR filter before downsampling.
    downsampled = {}
    for ci in ISRUC_CHANNEL_INDICES:
        downsampled[ci] = scipy.signal.resample_poly(
            channels[ci], up=1, down=2
        ).astype(np.float32)

    # ---- Stack channels in canonical order [C3-A2, O1-A2, LOC-A2] ----
    # ISRUC_CHANNEL_INDICES = [3, 4, 0] → stack preserves that order
    signal = np.stack(
        [downsampled[ci] for ci in ISRUC_CHANNEL_INDICES], axis=0
    )  # (3, N_samples_100hz)

    # ---- Load annotations (scorer 1, integer per line) ----
    raw_labels = np.loadtxt(ann_path, dtype=int)

    # ---- Label remap: 5 (R&K REM) → 4 (our REM) ----
    labels = np.array(
        [LABEL_REMAP.get(int(l), -1) for l in raw_labels], dtype=np.int64
    )

    # ---- Align signal and annotations to 30-second epochs ----
    n_epochs_signal = signal.shape[1] // EPOCH_SAMPLES
    n_epochs_ann    = len(labels)
    n_epochs        = min(n_epochs_signal, n_epochs_ann)

    signal  = signal[:, : n_epochs * EPOCH_SAMPLES]
    labels  = labels[:n_epochs]

    # ---- Segment into epochs: (3, N*T) → (N, 3, T) ----
    epochs = signal.reshape(3, n_epochs, EPOCH_SAMPLES).transpose(1, 0, 2)
    # epochs: (N_epochs, 3, 3000)

    # ---- Drop epochs with invalid labels (e.g. movement artefacts if any) ----
    valid_mask = labels >= 0
    epochs = epochs[valid_mask]
    labels = labels[valid_mask]
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise ValueError(f"Subject {subj_id}: no valid epochs after label filtering")

    # ---- Save ----
    out_path = out_dir / f"isruc_S{subj_id:03d}.npz"
    np.savez_compressed(
        out_path,
        signals=epochs,   # (N, 3, 3000) float32
        labels=labels,    # (N,)          int64
    )

    # Stage counts for reporting
    stage_counts = {int(k): int((labels == k).sum()) for k in sorted(set(labels.tolist()))}

    return {
        "subject_id":   subj_id,
        "n_epochs":     n_valid,
        "stage_counts": stage_counts,
        "output_file":  str(out_path.name),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ISRUC-Sleep Subgroup 1 for cross-dataset evaluation"
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=None,
        help="Subject IDs to process (default: all 1-100)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess subjects that already have output files"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ISRUC_PROC_DIR.mkdir(parents=True, exist_ok=True)

    subject_ids = args.subjects if args.subjects else list(range(1, N_SUBJECTS + 1))

    print(f"\nISRUC Preprocessing — Subgroup 1")
    print(f"  Raw data : {ISRUC_RAW_DIR}")
    print(f"  Output   : {ISRUC_PROC_DIR}")
    print(f"  Subjects : {len(subject_ids)}")
    print(f"  Channels : {list(zip(ISRUC_CHANNEL_NAMES, ISRUC_CHANNEL_INDICES))}")
    print(f"  Resample : {SRC_FS} Hz → {SAMPLING_RATE} Hz")
    print(f"  Epochs   : 30 s × {SAMPLING_RATE} Hz = {EPOCH_SAMPLES} samples\n")

    metadata_rows = []
    ok = skip = fail = 0
    failed = []

    for subj_id in subject_ids:
        out_path = ISRUC_PROC_DIR / f"isruc_S{subj_id:03d}.npz"

        if out_path.exists() and not args.overwrite:
            print(f"  [{subj_id:3d}/{N_SUBJECTS}] Already processed — skip")
            skip += 1
            # Re-read metadata for summary
            d = np.load(out_path)
            stage_counts = {int(k): int((d["labels"] == k).sum())
                            for k in sorted(set(d["labels"].tolist()))}
            metadata_rows.append({
                "subject_id":   subj_id,
                "n_epochs":     len(d["labels"]),
                "stage_counts": str(stage_counts),
                "output_file":  out_path.name,
            })
            continue

        print(f"  [{subj_id:3d}/{N_SUBJECTS}] Processing ...", end=" ", flush=True)
        try:
            meta = preprocess_subject(subj_id, ISRUC_PROC_DIR)
            metadata_rows.append({
                "subject_id":   meta["subject_id"],
                "n_epochs":     meta["n_epochs"],
                "stage_counts": str(meta["stage_counts"]),
                "output_file":  meta["output_file"],
            })
            print(f"done  ({meta['n_epochs']} epochs, stages={meta['stage_counts']})")
            ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            fail += 1
            failed.append(subj_id)

    # ---- Write metadata CSV ----
    csv_path = ISRUC_PROC_DIR / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject_id", "n_epochs", "stage_counts", "output_file"]
        )
        writer.writeheader()
        writer.writerows(sorted(metadata_rows, key=lambda r: r["subject_id"]))

    print(f"\nDone — processed: {ok}, skipped: {skip}, failed: {fail}")
    if failed:
        print(f"Failed subjects: {failed}")
    print(f"Metadata saved → {csv_path}")


if __name__ == "__main__":
    main()
