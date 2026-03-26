import torch
import torch.nn as nn

class SleepStagingHead(nn.Module):
    """
    Standard classification head for Sleep Stage scoring.
    """
    def __init__(self, d_model: int, num_classes: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone output (Batch, Seq_Len, D_model)
        Returns:
            Logits (Batch, Num_Classes)
        """
        device = self._get_device()
        if x.device != device:
            x = x.to(device)

        # Global Average Pooling over time dimension
        # x: (B, L, D) -> Permute to (B, D, L) for pooling
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x) # (B, D, 1)
        x = x.squeeze(2)     # (B, D)

        return self.fc(x)


class TransitionDetectionHead(nn.Module):
    """
    Binary classification head for detecting state transitions.
    Contribution #1: Transition-Aware Regularization.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # We might want to keep temporal resolution if predicting transition *points*,
        # but if the task is "Is this epoch a transition?", we use pooling.

        # NOTE: For "Transition-Aware", we ideally want to know if the CURRENT epoch
        # represents a transition state or contains a transition.
        # Assuming epoch-level binary classification.

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Binary logit
        )

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone output (Batch, Seq_Len, D_model)
        Returns:
            Logit (Batch, 1)
        """
        device = self._get_device()
        if x.device != device:
            x = x.to(device)

        x = x.permute(0, 2, 1) # (B, D, L)
        x = self.avg_pool(x).squeeze(2) # (B, D)

        return self.fc(x)
