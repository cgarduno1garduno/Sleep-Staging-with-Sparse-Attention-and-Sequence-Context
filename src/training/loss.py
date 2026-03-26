import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss: FL(pt) = -(1 - pt)^gamma * log(pt)

    Down-weights easy examples so the model focuses harder on difficult ones.
    For N1 specifically: N1/N2 boundary examples are hard (low pt) and get
    higher gradient weight, which should improve precision without sacrificing recall.

    Args:
        gamma: Focusing parameter. 0 = standard cross-entropy. 2 = standard focal.
        weight: Per-class weights tensor (same as CrossEntropyLoss weight).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (B, C) logits, target: (B,) class indices
        log_probs = F.log_softmax(input, dim=-1)
        probs = torch.exp(log_probs)

        # Log-prob and prob of the true class for each sample
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        # Apply per-class weights if provided
        if self.weight is not None:
            class_weight = self.weight[target]
            focal_weight = focal_weight * class_weight

        return (-focal_weight * log_pt).mean()


class UncertaintyLossWrapper(nn.Module):
    """
    Multi-Task Loss with Homoscedastic Uncertainty Weighting.
    Contribution #3: Dynamic Loss Balancing.

    Formula: Loss = 1/(2*sigma^2) * L_task + log(sigma)

    log_vars are clamped to [log_var_min, log_var_max] before use to prevent
    numerical overflow/underflow (e.g. the alpha=0 gradient explosion where the
    unconstrained transition log_var drifts to -inf, making precision -> +inf).
    """
    def __init__(
        self,
        num_tasks: int = 2,
        log_var_min: float = -6.0,
        log_var_max: float = 6.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        # Learnable log_vars (log(sigma^2)) for numerical stability
        # Initialize to 0.0 (sigma=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List of scalar tensors [L_stage, L_transition]
        Returns:
            Weighted scalar loss.
        """
        assert len(losses) == self.num_tasks, f"Expected {self.num_tasks} losses, got {len(losses)}"

        # Clamp prevents exp(-log_var) from overflowing when log_var -> -inf
        clamped = torch.clamp(self.log_vars, self.log_var_min, self.log_var_max)

        total_loss = torch.tensor(0.0, device=clamped.device)
        for i, loss in enumerate(losses):
            precision = 0.5 * torch.exp(-clamped[i])
            task_loss = precision * loss + clamped[i] * 0.5
            total_loss = total_loss + task_loss

        return total_loss
