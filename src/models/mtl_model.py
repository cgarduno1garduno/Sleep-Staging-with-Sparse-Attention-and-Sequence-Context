import torch
import torch.nn as nn
from src.models.backbones import SparseTransformerBackbone
from src.models.heads import SleepStagingHead, TransitionDetectionHead

class MTLSleepModel(nn.Module):
    """
    Main Multi-Task Learning Architecture.
    Combines:
    1. SparseTransformerBackbone (Contrib #2)
    2. SleepStagingHead
    3. TransitionDetectionHead (Contrib #1)
    """
    def __init__(self, config):
        super().__init__()
        # Extract Config
        # Assuming config is a module or object with attributes
        input_channels = len(config.CHANNELS)
        # Hyperparameters for backbone can be hardcoded or in config.
        # Using defaults from backbone def for now, but ideally in config.
        d_model = 64

        self.backbone = SparseTransformerBackbone(
            input_channels=input_channels,
            d_model=d_model,
            n_layers=4,
            n_heads=4,
            window_size=64
        )

        self.staging_head = SleepStagingHead(d_model=d_model, num_classes=5)
        self.transition_head = TransitionDetectionHead(d_model=d_model) # Binary

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input signal (B, C, T)
        Returns:
            Dict: {'stage_logits': ..., 'transition_logits': ...}
        """
        device = self._get_device()
        if x.device != device:
            x = x.to(device)

        # Backbone Feature Extraction
        features = self.backbone(x) # (B, Seq_Len, D)

        # Heads
        # Note: SleepStagingHead performs pooling internally
        stage_logits = self.staging_head(features)

        # TransitionHead performs pooling internally
        transition_logits = self.transition_head(features)

        return {
            "stage_logits": stage_logits,
            "transition_logits": transition_logits
        }
