"""
Retroactive evaluation script for calculating additional metrics.

Calculates Cohen's Kappa and other metrics from saved prediction files.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    classification_report
)


# Sleep stage labels
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def load_predictions(predictions_path: Path) -> tuple:
    """
    Load predictions from a .npz file.

    Args:
        predictions_path: Path to the .npz file.

    Returns:
        Tuple of (predictions, labels) arrays.
    """
    data = np.load(predictions_path)

    # Handle different key naming conventions
    if "predicted_stages" in data:
        predictions = data["predicted_stages"]
    elif "predictions" in data:
        predictions = data["predictions"]
    else:
        raise KeyError(f"No predictions found. Available keys: {list(data.keys())}")

    if "targets" in data:
        labels = data["targets"]
    elif "labels" in data:
        labels = data["labels"]
    else:
        raise KeyError(f"No labels found. Available keys: {list(data.keys())}")

    return predictions, labels


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        predictions: Predicted class labels.
        labels: True class labels.

    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "cohen_kappa": cohen_kappa_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
    }

    # Per-class F1 scores
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    for i, name in enumerate(STAGE_NAMES):
        if i < len(f1_per_class):
            metrics[f"f1_{name}"] = f1_per_class[i]

    return metrics


def evaluate_predictions(
    predictions_path: Path,
    output_path: Path = None,
    verbose: bool = True
) -> dict:
    """
    Load predictions and calculate all metrics.

    Args:
        predictions_path: Path to .npz predictions file.
        output_path: Optional path to save metrics JSON.
        verbose: Whether to print results.

    Returns:
        Dictionary of metrics.
    """
    predictions_path = Path(predictions_path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    # Load predictions
    predictions, labels = load_predictions(predictions_path)

    if verbose:
        print(f"\nLoaded predictions from: {predictions_path}")
        print(f"  Samples: {len(predictions)}")
        print(f"  Unique predicted classes: {np.unique(predictions)}")
        print(f"  Unique true classes: {np.unique(labels)}")

    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        kappa_label = "Cohen's Kappa"
        print(f"\n{kappa_label:.<30} {metrics['cohen_kappa']:.4f}")
        print(f"{'Accuracy':.<30} {metrics['accuracy']:.4f}")
        print(f"{'F1 Score (Macro)':.<30} {metrics['f1_macro']:.4f}")
        print(f"{'F1 Score (Weighted)':.<30} {metrics['f1_weighted']:.4f}")

        print("\nPer-Class F1 Scores:")
        for name in STAGE_NAMES:
            key = f"f1_{name}"
            if key in metrics:
                print(f"  {name:.<20} {metrics[key]:.4f}")

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            labels, predictions,
            target_names=STAGE_NAMES,
            digits=4,
            zero_division=0
        ))

    # Save metrics if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if verbose:
            print(f"\nMetrics saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics from saved predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions .npz file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save metrics JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    metrics = evaluate_predictions(
        predictions_path=args.predictions,
        output_path=args.output,
        verbose=not args.quiet
    )

    # Return kappa for easy scripting
    print(f"\n>>> COHEN'S KAPPA: {metrics['cohen_kappa']:.4f} <<<\n")


if __name__ == "__main__":
    main()
