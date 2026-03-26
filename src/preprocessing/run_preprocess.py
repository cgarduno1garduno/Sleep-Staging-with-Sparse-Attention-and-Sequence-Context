import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, PROCESSED_DIR
from src.preprocessing.extract import load_edf, filter_channels
from src.preprocessing.segment import segment_signals, save_segments_with_metadata

def setup_logger():
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_subject_pairs(data_dir: Path):
    """
    Find matching PSG and Hypnogram files.
    Returns list of (subject_id, psg_path, hypno_path)
    """
    pairs = []

    # Search recursively for *PSG.edf
    # Sleep-EDF structure:
    # sleep-cassette/SC4001E0-PSG.edf
    # sleep-cassette/SC4001EC-Hypnogram.edf

    psg_files = sorted(list(data_dir.rglob("*PSG.edf")))

    for psg_path in psg_files:
        subject_id = psg_path.name.split("-")[0] # SC4001E0

        # Hypnogram naming convention:
        # SC4... -> SC...-Hypnogram.edf
        # BUT the suffix changes (EC, E0, etc).
        # Actually pattern is SC4001E0-PSG -> SC4001EC-Hypnogram
        # The key is strict: [SubjectID excluding last 2 chars] match isn't enough,
        # Usually they share the first 6-7 chars.
        # EASIER STRATEGY: Look for file in same dir starting with same code.

        parent_dir = psg_path.parent
        # Try to find corresponding hypnogram
        # Pattern match: subject_id is first 8 chars usually, but sometimes they differ slightly.
        # Safe bet: startswith(subject_id[:7])?

        # SC4001E0 -> SC4001EC
        # SC4001E -> SC4001E

        prefix = subject_id[:7] # "SC4001E"
        hypno_candidates = list(parent_dir.glob(f"{prefix}*-Hypnogram.edf"))

        if len(hypno_candidates) == 1:
            pairs.append((subject_id, psg_path, hypno_candidates[0]))
        else:
            print(f"Warning: Could not find unique hypnogram for {subject_id}. Candidates: {hypno_candidates}")

    return pairs

def main():
    logger = setup_logger()
    logger.info(f"Starting Preprocessing. Data Dir: {DATA_DIR}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subjects")
    args = parser.parse_args()

    pairs = get_subject_pairs(DATA_DIR)
    logger.info(f"Found {len(pairs)} subject pairs.")

    if args.limit:
        pairs = pairs[:args.limit]
        logger.info(f"Limiting to first {args.limit} pairs.")

    unique_subjects = set()
    total_epochs = 0

    for i, (sub_id, psg_path, hypno_path) in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] Processing {sub_id}...")

        try:
            # 1. Load
            raw = load_edf(str(psg_path))

            # 2. Filter & Resample
            raw = filter_channels(raw)

            # 3. Segment
            segments = segment_signals(raw, str(hypno_path))

            if not segments:
                logger.warning(f"No valid segments found for {sub_id}")
                continue

            # 4. Save
            save_segments_with_metadata(segments, sub_id)

            total_epochs += len(segments)
            unique_subjects.add(sub_id)
            logger.info(f"  -> Saved {len(segments)} epochs.")

        except Exception as e:
            logger.error(f"Failed to process {sub_id}: {e}")

    logger.info("Preprocessing Complete.")
    logger.info(f"Total Subjects: {len(unique_subjects)}")
    logger.info(f"Total Epochs: {total_epochs}")

if __name__ == "__main__":
    main()
