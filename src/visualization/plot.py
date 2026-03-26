"""
Visualization utilities for sleep staging results.

Functions:
- plot_confusion_matrix: Generate and save confusion matrix from predictions.
- plot_training_history: Plot loss and accuracy curves.
- plot_hypnogram: Compare ground truth vs predicted hypnogram.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report

from src.config import RESULTS_DIR


# Sleep stage labels
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def _find_sleep_boundaries(labels: np.ndarray, buffer: int = 60) -> Tuple[int, int]:
    """Find first and last non-wake epoch indices, with a buffer on each side."""
    non_wake = np.where(labels != 0)[0]
    if len(non_wake) == 0:
        return 0, len(labels)
    start = max(0, non_wake[0] - buffer)
    end = min(len(labels), non_wake[-1] + buffer + 1)
    return start, end


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    normalize: bool = True,
    title: str = "Sleep Stage Confusion Matrix",
    class_names: List[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = "Blues"
) -> Path:
    """
    Generate and save a confusion matrix visualization.

    Args:
        predictions: Array of predicted class labels.
        labels: Array of true class labels.
        output_path: Path to save the figure. If None, saves to results/.
        normalize: Whether to normalize by true labels (row-wise).
        title: Plot title.
        class_names: List of class names. Defaults to sleep stage names.
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        Path to the saved figure.
    """
    if class_names is None:
        class_names = STAGE_NAMES

    if output_path is None:
        output_path = RESULTS_DIR / "confusion_matrix.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)

    if normalize:
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    else:
        cm_normalized = cm

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap — vmin/vmax anchored to 0-1 for consistent cross-model comparison
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_normalized.max() / 2.0

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_normalized[i, j]
            count = cm[i, j]

            if normalize:
                text = f"{value:.2f}\n({count})"
            else:
                text = f"{count}"

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")

    return output_path


def plot_training_history(
    history: dict,
    output_path: Path = None,
    figsize: tuple = (12, 5)
) -> Path:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with 'train' and 'val' keys containing metric dicts.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Path to the saved figure.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "training_history.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_history = history.get("train", [])
    val_history = history.get("val", [])

    if not train_history:
        print("No training history to plot")
        return None

    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    ax1 = axes[0]
    train_loss = [m["train_loss"] for m in train_history]
    ax1.plot(epochs, train_loss, "b-", label="Train Loss", marker="o")

    if val_history:
        val_loss = [m["val_loss"] for m in val_history]
        ax1.plot(epochs, val_loss, "r-", label="Val Loss", marker="s")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2 = axes[1]
    train_acc = [m["train_accuracy"] for m in train_history]
    ax2.plot(epochs, train_acc, "b-", label="Train Accuracy", marker="o")

    if val_history:
        val_acc = [m["val_accuracy"] for m in val_history]
        ax2.plot(epochs, val_acc, "r-", label="Val Accuracy", marker="s")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training history saved to: {output_path}")

    return output_path


def plot_hypnogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = None,
    figsize: tuple = (14, 6)
) -> Path:
    """
    Plot ground truth vs predicted hypnogram.

    Args:
        y_true: True sleep stage labels.
        y_pred: Predicted sleep stage labels.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Path to the saved figure.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "hypnogram.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Ground truth
    axes[0].step(epochs, y_true, where="mid", color="blue", linewidth=1.5)
    axes[0].set_ylabel("Sleep Stage")
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(STAGE_NAMES)
    axes[0].set_title("Ground Truth Hypnogram")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)

    # Predicted
    axes[1].step(epochs, y_pred, where="mid", color="red", linewidth=1.5)
    axes[1].set_ylabel("Sleep Stage")
    axes[1].set_xlabel("Epoch (30s)")
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].set_yticklabels(STAGE_NAMES)
    axes[1].set_title("Predicted Hypnogram")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Hypnogram saved to: {output_path}")

    return output_path


def generate_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path = None,
    class_names: List[str] = None
) -> str:
    """
    Generate and save a classification report.

    Args:
        predictions: Array of predicted class labels.
        labels: Array of true class labels.
        output_path: Path to save the report. If None, saves to results/.
        class_names: List of class names.

    Returns:
        Classification report string.
    """
    if class_names is None:
        class_names = STAGE_NAMES

    if output_path is None:
        output_path = RESULTS_DIR / "classification_report.txt"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        labels, predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    with open(output_path, "w") as f:
        f.write("Sleep Stage Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"Classification report saved to: {output_path}")

    return report


def plot_transition_heatmap(
    stage_labels: np.ndarray,
    transition_probs: np.ndarray,
    output_path: Path = None,
    threshold: float = 0.5,
    figsize: tuple = (16, 10),
) -> Path:
    """
    Visualize transition probabilities alongside stage hypnogram.

    Shows where the model predicted transitions (prob > threshold) compared
    to where actual stage changes occurred. Data is trimmed to the actual sleep
    period (first/last non-Wake epoch ± 30-min buffer) so the hypnogram is
    never a flat Wake line.

    All panels share the same x-axis and plotting width; the colorbar lives in
    a dedicated narrow column so it never compresses any panel.

    Args:
        stage_labels: True sleep stage labels.
        transition_probs: Model's transition probability predictions (0-1).
        output_path: Path to save the figure.
        threshold: Probability threshold for "confident" transition.
        figsize: Figure size.

    Returns:
        Path to the saved figure.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "transition_heatmap.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Trim to sleep period so hypnogram is not a flat Wake line
    start_idx, end_idx = _find_sleep_boundaries(stage_labels)
    stage_labels = stage_labels[start_idx:end_idx]
    transition_probs = transition_probs[start_idx:end_idx]

    n_epochs = len(stage_labels)

    # Normalize probabilities to min/max scale for better visualization
    prob_min = transition_probs.min()
    prob_max = transition_probs.max()
    probs_norm = (transition_probs - prob_min) / (prob_max - prob_min + 1e-6)

    # Calculate actual transitions (where stage changes)
    actual_transitions = np.zeros(n_epochs)
    for i in range(1, n_epochs):
        if stage_labels[i] != stage_labels[i - 1]:
            actual_transitions[i] = 1

    # Predicted transitions (where prob > threshold)
    predicted_transitions = (transition_probs > threshold).astype(float)

    epochs = np.arange(n_epochs)

    # GridSpec: 4 data rows + 1 narrow colorbar column so all panels are same width
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        4, 2,
        width_ratios=[50, 1],
        height_ratios=[2, 1, 1, 1],
        hspace=0.15,
        wspace=0.02,
    )

    # 1. Hypnogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.step(epochs, stage_labels, where="mid", color="blue", linewidth=1.5)
    ax1.set_ylabel("Sleep Stage")
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(STAGE_NAMES)
    ax1.set_title("Sleep Hypnogram with Transition Analysis")
    ax1.invert_yaxis()
    ax1.set_xlim(0, n_epochs)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # Mark actual transitions on hypnogram
    transition_epochs = np.where(actual_transitions == 1)[0]
    for te in transition_epochs:
        ax1.axvline(x=te, color="green", alpha=0.3, linestyle="--", linewidth=0.8)

    # Invisible colorbar placeholder for row 0 to keep alignment
    fig.add_subplot(gs[0, 1]).set_visible(False)

    # 2. Transition probability heatmap (normalized for contrast)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    prob_matrix = probs_norm.reshape(1, -1)
    im = ax2.imshow(prob_matrix, aspect="auto", cmap="Reds",
                    vmin=0, vmax=1, extent=[0, n_epochs, 0, 1])
    ax2.set_ylabel("Prob")
    ax2.set_yticks([])
    ax2.set_title(
        f"Model Transition Probability (normalized, raw range: {prob_min:.3f}–{prob_max:.3f})"
    )
    ax2.tick_params(labelbottom=False)

    # Colorbar in its own column — does NOT steal width from ax2
    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cbar)
    ax_cbar.set_ylabel("Prob", fontsize=8)

    # 3. Actual transitions
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.bar(epochs, actual_transitions, width=1.0, color="green", alpha=0.7)
    ax3.set_ylabel("Actual")
    ax3.set_ylim(0, 1.2)
    ax3.set_title("Actual Stage Transitions")
    ax3.tick_params(labelbottom=False)
    fig.add_subplot(gs[2, 1]).set_visible(False)

    # 4. Predicted transitions (thresholded)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.bar(epochs, predicted_transitions, width=1.0, color="red", alpha=0.7)
    ax4.set_ylabel("Predicted")
    ax4.set_xlabel("Epoch (30s)")
    ax4.set_ylim(0, 1.2)
    ax4.set_title(f"Predicted Transitions (prob > {threshold})")
    fig.add_subplot(gs[3, 1]).set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Calculate transition detection metrics
    tp = np.sum((predicted_transitions == 1) & (actual_transitions == 1))
    fp = np.sum((predicted_transitions == 1) & (actual_transitions == 0))
    fn = np.sum((predicted_transitions == 0) & (actual_transitions == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Transition heatmap saved to: {output_path}")
    print(f"  Transition Detection - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    return output_path


def plot_from_results(
    run_name: str,
    results_dir: Path = None,
    threshold: float = 0.5
) -> None:
    """
    Generate all plots from a training run's saved results.

    Args:
        run_name: Name of the training run.
        results_dir: Results directory. Defaults to RESULTS_DIR.
        threshold: Probability threshold for transition detection.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    results_dir = Path(results_dir)

    # Load predictions
    predictions_path = results_dir / f"predictions_{run_name}.npz"
    if predictions_path.exists():
        data = np.load(predictions_path)
        predictions = data["predictions"]
        labels = data["labels"]

        # Generate confusion matrix
        plot_confusion_matrix(
            predictions, labels,
            output_path=results_dir / f"confusion_matrix_{run_name}.png",
            title=f"Confusion Matrix - {run_name}"
        )

        # Generate classification report
        generate_classification_report(
            predictions, labels,
            output_path=results_dir / f"classification_report_{run_name}.txt"
        )

        # Generate transition heatmap if transition_probs available
        if "transition_probs" in data:
            transition_probs = data["transition_probs"]
            plot_transition_heatmap(
                stage_labels=labels,
                transition_probs=transition_probs,
                output_path=results_dir / f"transition_heatmap_{run_name}.png",
                threshold=threshold
            )
        else:
            print(f"No transition_probs in {predictions_path}, skipping heatmap")
    else:
        print(f"Predictions file not found: {predictions_path}")

    # Load and plot training history
    results_path = results_dir / f"results_{run_name}.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        if "history" in results:
            plot_training_history(
                results["history"],
                output_path=results_dir / f"training_history_{run_name}.png"
            )
    else:
        print(f"Results file not found: {results_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the training run")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Results directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for transition detection (default: 0.5)"
    )

    args = parser.parse_args()

    plot_from_results(args.run_name, args.results_dir, threshold=args.threshold)
