"""
Training and Validation Loops.

Functions:
- train_one_epoch(model, loader, optimizer, loss_fn): Run one training epoch.
- validate(model, loader, loss_fn): Evaluate on validation set.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_wrapper: nn.Module,
    stage_criterion: nn.Module,
    transition_criterion: nn.Module,
    device: torch.device = None,
    logger=None,
    alpha: float = 1.0,
    max_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Args:
        model: The MTL model.
        loader: Training data loader.
        optimizer: Optimizer.
        loss_wrapper: UncertaintyLossWrapper for combining losses.
        stage_criterion: Loss function for staging (CrossEntropy).
        transition_criterion: Loss function for transition (BCE).
        device: Device to use.
        logger: Optional logger.
        alpha: Manual weighting factor for transition loss (default: 1.0).

    Returns:
        Dictionary with training metrics.
    """
    if device is None:
        device = get_device()

    model.train()
    model.to(device)
    loss_wrapper.to(device)

    total_loss = 0.0
    stage_loss_sum = 0.0
    transition_loss_sum = 0.0
    correct = 0
    total = 0

    n_batches = len(loader)
    log_interval = max(1, n_batches // 5)  # Log ~5 times per epoch

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        signals = batch["signal"].to(device)
        stage_labels = batch["stage_label"].to(device)
        transition_labels = batch["transition_label"].float().to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(signals)

        # Compute individual losses
        stage_loss = stage_criterion(outputs["stage_logits"], stage_labels)
        transition_loss_raw = transition_criterion(
            outputs["transition_logits"].squeeze(),
            transition_labels
        )

        # Apply alpha weighting to transition loss for gradient computation
        transition_loss_weighted = alpha * transition_loss_raw

        # Combine losses with uncertainty weighting (using weighted transition loss)
        combined_loss = loss_wrapper([stage_loss, transition_loss_weighted])

        # Backward pass with gradient clipping
        combined_loss.backward()
        all_params = list(model.parameters()) + list(loss_wrapper.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_norm)
        optimizer.step()

        # Track metrics (use raw transition loss for comparable logging)
        total_loss += combined_loss.item()
        stage_loss_sum += stage_loss.item()
        transition_loss_sum += transition_loss_raw.item()

        # Accuracy for staging
        preds = outputs["stage_logits"].argmax(dim=1)
        correct += (preds == stage_labels).sum().item()
        total += stage_labels.size(0)

        # Periodic logging for debugging (use raw transition loss for comparability)
        if logger and (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"  Batch {batch_idx + 1}/{n_batches} | "
                f"Combined: {combined_loss.item():.4f} | "
                f"Stage: {stage_loss.item():.4f} | "
                f"Transition: {transition_loss_raw.item():.4f}"
            )

    metrics = {
        "train_loss": total_loss / n_batches,
        "train_stage_loss": stage_loss_sum / n_batches,
        "train_transition_loss": transition_loss_sum / n_batches,
        "train_accuracy": correct / total if total > 0 else 0.0
    }

    return metrics


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_wrapper: nn.Module,
    stage_criterion: nn.Module,
    transition_criterion: nn.Module,
    device: torch.device = None,
    alpha: float = 1.0
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation/test set.

    Args:
        model: The MTL model.
        loader: Validation data loader.
        loss_wrapper: UncertaintyLossWrapper.
        stage_criterion: Loss function for staging.
        transition_criterion: Loss function for transition.
        device: Device to use.
        alpha: Manual weighting factor for transition loss (default: 1.0).

    Returns:
        Tuple of (metrics dict, all_predictions, all_labels, transition_probs).
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)
    loss_wrapper.to(device)

    total_loss = 0.0
    stage_loss_sum = 0.0
    transition_loss_sum = 0.0

    all_preds = []
    all_labels = []
    all_transition_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            signals = batch["signal"].to(device)
            stage_labels = batch["stage_label"].to(device)
            transition_labels = batch["transition_label"].float().to(device)

            # Forward pass
            outputs = model(signals)

            # Compute losses
            stage_loss = stage_criterion(outputs["stage_logits"], stage_labels)
            transition_loss_raw = transition_criterion(
                outputs["transition_logits"].squeeze(),
                transition_labels
            )

            # Apply alpha weighting for combined loss calculation
            transition_loss_weighted = alpha * transition_loss_raw
            combined_loss = loss_wrapper([stage_loss, transition_loss_weighted])

            # Track metrics (use raw transition loss for comparable logging)
            total_loss += combined_loss.item()
            stage_loss_sum += stage_loss.item()
            transition_loss_sum += transition_loss_raw.item()

            # Collect predictions
            preds = outputs["stage_logits"].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(stage_labels.cpu().numpy())

            # Collect transition probabilities (sigmoid of logits)
            transition_probs = torch.sigmoid(outputs["transition_logits"].squeeze())
            all_transition_probs.extend(transition_probs.cpu().numpy())

    n_batches = len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_transition_probs = np.array(all_transition_probs)

    accuracy = (all_preds == all_labels).mean()

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    metrics = {
        "val_loss": total_loss / n_batches,
        "val_stage_loss": stage_loss_sum / n_batches,
        "val_transition_loss": transition_loss_sum / n_batches,
        "val_accuracy": accuracy,
        "val_kappa": kappa
    }

    return metrics, all_preds, all_labels, all_transition_probs
