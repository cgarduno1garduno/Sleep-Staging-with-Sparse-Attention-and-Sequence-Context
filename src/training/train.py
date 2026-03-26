"""
Main Training Script.

Responsibilities:
- Parse command line arguments.
- Instantiate Config, Dataset, Model, Loss, Optimizer.
- Run Training Loop (loops.py).
- Save checkpoints and logs to results/.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR, RESULTS_DIR, CHANNELS
from src.dataloading.dataset import SleepDataset, get_subject_splits
from src.dataloading.samplers import create_weighted_sampler
from src.models.mtl_model import MTLSleepModel
from src.training.loss import UncertaintyLossWrapper
from src.training.loops import train_one_epoch, validate, get_device


def setup_logging(run_name: str = None) -> logging.Logger:
    """
    Set up logging to save a .txt log file for each training run.

    Args:
        run_name: Optional name for the run. If None, uses timestamp.

    Returns:
        Configured logger instance.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = RESULTS_DIR / f"train_{run_name}.txt"

    logger = logging.getLogger("sleep_research")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger, run_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MTL Sleep Staging Model")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing processed .npy files and metadata.csv"
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="Limit training to N subjects (for smoke tests)"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Manual loss weighting factor for transition task (default: 1.0)"
    )

    # Split ratios
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions for confusion matrix"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint .pt file to load for inference/resuming"
    )

    return parser.parse_args()


class ConfigNamespace:
    """Simple namespace to hold config for model initialization."""
    def __init__(self, channels):
        self.CHANNELS = channels


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup logging
    logger, run_name = setup_logging(args.run_name)
    logger.info(f"Arguments: {vars(args)}")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Log alpha (transition loss weighting)
    logger.info(f"Transition loss alpha (manual weighting): {args.alpha}")

    # Load metadata and create subject splits
    metadata_path = args.data_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    train_subjects, val_subjects, test_subjects = get_subject_splits(
        metadata_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Limit subjects for smoke test
    if args.num_subjects is not None:
        all_subjects = train_subjects + val_subjects + test_subjects
        all_subjects = all_subjects[:args.num_subjects]

        n_train = max(1, int(len(all_subjects) * args.train_ratio))
        n_val = max(1, int(len(all_subjects) * args.val_ratio))

        train_subjects = all_subjects[:n_train]
        val_subjects = all_subjects[n_train:n_train + n_val]
        test_subjects = all_subjects[n_train + n_val:]

        # Ensure at least one subject in each split for smoke test
        if len(val_subjects) == 0 and len(train_subjects) > 1:
            val_subjects = [train_subjects.pop()]
        if len(test_subjects) == 0 and len(train_subjects) > 1:
            test_subjects = [train_subjects.pop()]

    logger.info(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    logger.info(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    logger.info(f"Test subjects ({len(test_subjects)}): {test_subjects}")

    # Create datasets
    train_dataset = SleepDataset(
        metadata_path=metadata_path,
        data_dir=args.data_dir,
        subject_ids=train_subjects
    )
    val_dataset = SleepDataset(
        metadata_path=metadata_path,
        data_dir=args.data_dir,
        subject_ids=val_subjects
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create weighted sampler for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = create_weighted_sampler(sample_weights, num_samples=len(train_dataset))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model
    config = ConfigNamespace(CHANNELS)
    model = MTLSleepModel(config)
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize loss functions
    loss_wrapper = UncertaintyLossWrapper(num_tasks=2)
    loss_wrapper.to(device)

    stage_criterion = nn.CrossEntropyLoss()
    transition_criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_wrapper.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Load checkpoint if requested
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loss_wrapper.load_state_dict(checkpoint["loss_wrapper_state_dict"])

    # Training loop
    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    # Initialize val_metrics for the case where epochs=0
    val_metrics = None

    if args.epochs > 0:
        logger.info("Starting training...")

        for epoch in range(args.epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            logger.info(f"{'='*50}")

            # Train
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_wrapper=loss_wrapper,
                stage_criterion=stage_criterion,
                transition_criterion=transition_criterion,
                device=device,
                logger=logger,
                alpha=args.alpha
            )
            history["train"].append(train_metrics)

            # Log uncertainty weights
            log_vars = loss_wrapper.log_vars.detach().cpu().numpy()
            logger.info(f"Uncertainty log_vars: stage={log_vars[0]:.4f}, transition={log_vars[1]:.4f}")

            # Validate
            val_metrics, val_preds, val_labels, val_transition_probs = validate(
                model=model,
                loader=val_loader,
                loss_wrapper=loss_wrapper,
                stage_criterion=stage_criterion,
                transition_criterion=transition_criterion,
                device=device,
                alpha=args.alpha
            )
            history["val"].append(val_metrics)

            # Log metrics
            logger.info(
                f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                f"Acc: {train_metrics['train_accuracy']:.4f}"
            )
            logger.info(
                f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                f"Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Kappa: {val_metrics['val_kappa']:.4f}"
            )

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                checkpoint_path = RESULTS_DIR / f"best_model_{run_name}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                    "val_loss": best_val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

    else:
        # Inference Mode (epochs=0)
        logger.info("Epochs set to 0. Running in Inference Mode (Validation only).")
        val_metrics, val_preds, val_labels, val_transition_probs = validate(
            model=model,
            loader=val_loader,
            loss_wrapper=loss_wrapper,
            stage_criterion=stage_criterion,
            transition_criterion=transition_criterion,
            device=device,
            alpha=args.alpha
        )
        history["val"].append(val_metrics)
        logger.info(
            f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
            f"Acc: {val_metrics['val_accuracy']:.4f}, "
            f"Kappa: {val_metrics['val_kappa']:.4f}"
        )
        # Set best_val_loss for saving results
        best_val_loss = val_metrics["val_loss"]

    # Save final results
    results = {
        "args": vars(args),
        "history": history,
        "best_val_loss": best_val_loss,
        "final_kappa": val_metrics["val_kappa"],
        "final_accuracy": val_metrics["val_accuracy"],
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }

    results_path = RESULTS_DIR / f"results_{run_name}.json"

    # Convert Path objects to strings for JSON serialization
    results["args"]["data_dir"] = str(results["args"]["data_dir"])
    if results["args"]["load_checkpoint"]:
        results["args"]["load_checkpoint"] = str(results["args"]["load_checkpoint"])

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_path = RESULTS_DIR / f"predictions_{run_name}.npz"
        np.savez(
            predictions_path,
            predictions=val_preds,
            labels=val_labels,
            transition_probs=val_transition_probs
        )
        logger.info(f"Saved predictions to {predictions_path}")

    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {val_metrics['val_accuracy']:.4f}")

    # Check if loss decreased
    if len(history["train"]) >= 2:
        initial_loss = history["train"][0]["train_loss"]
        final_loss = history["train"][-1]["train_loss"]
        if final_loss < initial_loss:
            logger.info(f"Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
        else:
            logger.warning(f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}")

    return history


if __name__ == "__main__":
    main()
