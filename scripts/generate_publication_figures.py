#!/usr/bin/env python3
"""
Generate Publication-Ready Figures for TASA Paper.

Creates properly aligned figures with:
- Correct axis scaling between panels
- Publication-quality fonts and sizing
- Proper colorbar placement that doesn't compress plots
- 300 DPI output for print

Figures generated:
1. Confusion matrices (proposed model and ablation)
2. Transition analysis figures (hypnogram + heatmap, properly aligned)
3. Ablation comparison plots
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR

# =============================================================================
# Configuration
# =============================================================================

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
BUFFER_EPOCHS = 60  # 30 minutes buffer
DPI = 300

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


# =============================================================================
# Helper Functions
# =============================================================================

def find_sleep_boundaries(labels: np.ndarray, buffer: int = BUFFER_EPOCHS) -> Tuple[int, int]:
    """Find first and last non-wake indices with buffer."""
    non_wake_indices = np.where(labels != 0)[0]

    if len(non_wake_indices) == 0:
        return 0, len(labels)

    first_sleep = non_wake_indices[0]
    last_sleep = non_wake_indices[-1]

    start_idx = max(0, first_sleep - buffer)
    end_idx = min(len(labels), last_sleep + buffer + 1)

    return start_idx, end_idx


# =============================================================================
# Confusion Matrix Figures
# =============================================================================

def plot_confusion_matrix_publication(
    labels: np.ndarray,
    predictions: np.ndarray,
    title: str,
    output_path: Path,
    figsize: Tuple[float, float] = (6, 5)
):
    """
    Generate publication-quality confusion matrix.

    Shows both normalized values (percentages) and raw counts.
    """
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=1)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontweight='bold')

    # Ticks and labels
    ax.set_xticks(np.arange(len(STAGE_NAMES)))
    ax.set_yticks(np.arange(len(STAGE_NAMES)))
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_yticklabels(STAGE_NAMES)

    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)

    # Text annotations
    thresh = 0.5
    for i in range(len(STAGE_NAMES)):
        for j in range(len(STAGE_NAMES)):
            value = cm_normalized[i, j]
            count = cm[i, j]
            text = f'{value:.2f}\n({count})'
            color = 'white' if value > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved confusion matrix: {output_path}")


# =============================================================================
# Transition Analysis Figures (FIXED ALIGNMENT)
# =============================================================================

def plot_transition_analysis_aligned(
    labels: np.ndarray,
    transition_probs: np.ndarray,
    title: str,
    subtitle: str,
    output_path: Path,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Generate transition analysis figure with PROPER ALIGNMENT.

    Uses GridSpec with explicit width ratios to ensure hypnogram and heatmap
    align correctly, with colorbar in separate column.
    """
    # Trim data to sleep period
    start_idx, end_idx = find_sleep_boundaries(labels)
    labels_trimmed = labels[start_idx:end_idx]
    probs_trimmed = transition_probs[start_idx:end_idx]

    n_epochs = len(labels_trimmed)
    epochs = np.arange(n_epochs)

    # Create figure with GridSpec for proper alignment
    # Main plot area gets most of width, colorbar gets small portion
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[50, 1], height_ratios=[2, 1],
                           hspace=0.08, wspace=0.02)

    # Top panel: Hypnogram
    ax_hyp = fig.add_subplot(gs[0, 0])
    ax_hyp.step(epochs, labels_trimmed, where='mid', color='navy', linewidth=1.2)
    ax_hyp.fill_between(epochs, labels_trimmed, step='mid', alpha=0.3, color='navy')

    ax_hyp.set_ylim(-0.5, 4.5)
    ax_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_hyp.set_yticklabels(STAGE_NAMES)
    ax_hyp.invert_yaxis()
    ax_hyp.set_xlim(0, n_epochs)

    ax_hyp.set_ylabel('Sleep Stage', fontweight='bold')
    ax_hyp.set_title(f'{title}\n{subtitle}', fontweight='bold', pad=10)
    ax_hyp.grid(True, alpha=0.3, linestyle='--')
    ax_hyp.tick_params(labelbottom=False)  # Hide x labels for top panel

    # Subtle background shading
    for stage in range(5):
        ax_hyp.axhspan(stage - 0.5, stage + 0.5, alpha=0.03, color='gray')

    # Bottom panel: Transition probability heatmap
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_hyp)

    # Create 1D heatmap
    prob_matrix = probs_trimmed.reshape(1, -1)
    im = ax_heat.imshow(prob_matrix, aspect='auto', cmap='Reds',
                        vmin=0.0, vmax=1.0, extent=[0, n_epochs, 0, 1])

    ax_heat.set_yticks([0.5])
    ax_heat.set_yticklabels(['Trans.\nProb.'], fontsize=9)
    ax_heat.set_xlabel('Epochs (30s each)', fontweight='bold')
    ax_heat.set_xlim(0, n_epochs)

    # Colorbar in separate subplot (right side)
    ax_cbar = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Probability', fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved transition analysis: {output_path}")
    print(f"  Original: {len(labels)} epochs, Trimmed: {n_epochs} epochs")
    print(f"  Prob range: {probs_trimmed.min():.4f} - {probs_trimmed.max():.4f}")


# =============================================================================
# Ablation Comparison Figures
# =============================================================================

def plot_alpha_ablation_comparison(
    results: dict,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Plot alpha ablation results as grouped bar chart.
    """
    alphas = []
    accuracies = []
    kappas = []
    n1_recalls = []

    for r in results:
        alpha = r["config"]["alpha"]
        alphas.append(f'α={alpha}')
        accuracies.append(r["test_accuracy"] * 100)
        kappas.append(r["test_kappa"])
        n1_recalls.append(r["test_n1_recall"])

    x = np.arange(len(alphas))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='steelblue')
    bars2 = ax.bar(x, [k * 100 for k in kappas], width, label='Kappa × 100', color='forestgreen')
    bars3 = ax.bar(x + width, [r * 100 for r in n1_recalls], width, label='N1 Recall (%)', color='coral')

    ax.set_xlabel('Alpha Value', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Alpha Ablation Study Results', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(alphas)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved alpha ablation plot: {output_path}")


def plot_window_ablation_comparison(
    results: dict,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Plot window size ablation results.
    """
    windows = []
    accuracies = []
    kappas = []
    n1_recalls = []

    for r in results:
        window = r["config"]["window_size"]
        windows.append(f'W={window}')
        accuracies.append(r["test_accuracy"] * 100)
        kappas.append(r["test_kappa"])
        n1_recalls.append(r["test_n1_recall"])

    x = np.arange(len(windows))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='steelblue')
    bars2 = ax.bar(x, [k * 100 for k in kappas], width, label='Kappa × 100', color='forestgreen')
    bars3 = ax.bar(x + width, [r * 100 for r in n1_recalls], width, label='N1 Recall (%)', color='coral')

    ax.set_xlabel('Window Size', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Window Size Ablation Study Results', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved window ablation plot: {output_path}")


def plot_channel_ablation_comparison(
    results: dict,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Plot channel configuration ablation results.
    """
    configs = []
    accuracies = []
    kappas = []
    n1_recalls = []

    channel_labels = {
        1: "1ch\n(Fpz-Cz)",
        2: "2ch\n(Fpz-Cz+EOG)",
        3: "3ch\n(Full)"
    }

    for r in results:
        channels = r["config"]["input_channels"]
        configs.append(channel_labels[channels])
        accuracies.append(r["test_accuracy"] * 100)
        kappas.append(r["test_kappa"])
        n1_recalls.append(r["test_n1_recall"])

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='steelblue')
    bars2 = ax.bar(x, [k * 100 for k in kappas], width, label='Kappa × 100', color='forestgreen')
    bars3 = ax.bar(x + width, [r * 100 for r in n1_recalls], width, label='N1 Recall (%)', color='coral')

    ax.set_xlabel('Channel Configuration', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Channel Configuration Ablation Study Results', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved channel ablation plot: {output_path}")


# =============================================================================
# Generate Figures from Existing Results
# =============================================================================

def generate_figures_from_predictions(
    predictions_file: Path,
    results_file: Optional[Path],
    output_prefix: str,
    output_dir: Path
):
    """
    Generate all figures from a predictions file.
    """
    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found: {predictions_file}")
        return

    # Load data
    data = np.load(predictions_file)
    predictions = data["predictions"]
    labels = data["labels"]

    # Get metrics from results file if available
    title = output_prefix
    subtitle = ""
    if results_file and results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        accuracy = results.get("final_accuracy", results.get("test_accuracy", 0))
        kappa = results.get("final_kappa", results.get("test_kappa", 0))
        if isinstance(accuracy, float):
            subtitle = f"Accuracy: {accuracy*100:.2f}%, Kappa: {kappa:.4f}"

    # Generate confusion matrix
    plot_confusion_matrix_publication(
        labels=labels,
        predictions=predictions,
        title=f"Confusion Matrix - {output_prefix}",
        output_path=output_dir / f"confusion_matrix_{output_prefix}.png"
    )

    # Generate transition analysis if available
    if "transition_probs" in data:
        transition_probs = data["transition_probs"]
        plot_transition_analysis_aligned(
            labels=labels,
            transition_probs=transition_probs,
            title=output_prefix,
            subtitle=subtitle,
            output_path=output_dir / f"transition_analysis_{output_prefix}.png"
        )


def generate_ablation_plots(ablation_dir: Path, output_dir: Path):
    """
    Generate comparison plots from ablation results.
    """
    summary_file = ablation_dir / "summary.json"
    if not summary_file.exists():
        print(f"ERROR: Summary file not found: {summary_file}")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    if "alpha" in summary:
        plot_alpha_ablation_comparison(
            summary["alpha"],
            output_dir / "ablation_alpha_comparison.png"
        )

    if "window" in summary:
        plot_window_ablation_comparison(
            summary["window"],
            output_dir / "ablation_window_comparison.png"
        )

    if "channel" in summary:
        plot_channel_ablation_comparison(
            summary["channel"],
            output_dir / "ablation_channel_comparison.png"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--predictions", type=Path, help="Path to predictions .npz file")
    parser.add_argument("--results", type=Path, help="Path to results .json file")
    parser.add_argument("--name", type=str, default="model", help="Output name prefix")
    parser.add_argument("--ablation-dir", type=Path, help="Path to ablation results directory")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        generate_figures_from_predictions(
            args.predictions,
            args.results,
            args.name,
            args.output_dir
        )

    if args.ablation_dir:
        generate_ablation_plots(args.ablation_dir, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
