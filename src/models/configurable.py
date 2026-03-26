"""
Configurable model for ablation studies.

Supports configurable attention window size (W ∈ {32, 64, 128, None for full attention})
and configurable number of input channels (1, 2, or 3).

This is the model used for all CV experiments. The window_size=None path gives
full O(L²) attention as an ablation baseline.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class ConfigurableAttention(nn.Module):
    """
    Attention that supports both sparse (local) and full attention.
    window_size=None → full O(L²) attention (ablation baseline).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: Optional[int] = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size, seq_len, _ = x.shape

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.window_size is not None:
            indices = torch.arange(seq_len, device=device).unsqueeze(0)
            distance_matrix = torch.abs(indices - indices.transpose(0, 1))
            sparse_mask = distance_matrix > self.window_size
            scores = scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class ConfigurableBackbone(nn.Module):
    """Transformer backbone with configurable attention window and input channels."""

    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: Optional[int] = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # Convolutional embedding: (B, C, 3000) → (B, d_model, 750)
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model // 2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            ConfigurableAttention(d_model, n_heads, window_size=window_size)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)   # (B, D, L)
        x = x.permute(0, 2, 1)  # (B, L, D)

        for attn, norm, ffn in zip(self.layers, self.layer_norms, self.ffns):
            residual = x
            x = attn(x)
            x = residual + x
            x = norm(x)

            residual = x
            x = ffn(x)
            x = residual + x
            x = norm(x)

        return x


class _SleepStagingHead(nn.Module):
    """5-class sleep staging head (global average pool → MLP)."""

    def __init__(self, d_model: int, num_classes: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)       # (B, D, L)
        x = self.avg_pool(x).squeeze(2)  # (B, D)
        return self.fc(x)


class _TransitionDetectionHead(nn.Module):
    """Binary transition detection head (global average pool → MLP → scalar logit)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)       # (B, D, L)
        x = self.avg_pool(x).squeeze(2)  # (B, D)
        return self.fc(x)


class ConfigurableTASA(nn.Module):
    """
    Configurable model used for all CV experiments and ablations.

    Args:
        input_channels: 1, 2, or 3 EEG/EOG channels
        d_model: Transformer hidden dimension (default: 64)
        n_layers: Number of transformer layers (default: 4)
        n_heads: Number of attention heads (default: 4)
        window_size: Sparse attention window. None = full O(L²) attention.
        num_classes: Sleep stage classes (default: 5)
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: Optional[int] = 64,
        num_classes: int = 5,
    ):
        super().__init__()
        self.backbone = ConfigurableBackbone(
            input_channels=input_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
        )
        self.staging_head = _SleepStagingHead(d_model=d_model, num_classes=num_classes)
        self.transition_head = _TransitionDetectionHead(d_model=d_model)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return {
            "stage_logits": self.staging_head(features),
            "transition_logits": self.transition_head(features),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ContextTASA(nn.Module):
    """
    ConfigurableTASA extended with neighboring-epoch sequence context.

    For each epoch, also processes K neighbor epochs on each side through the
    same shared backbone. The resulting epoch embeddings are concatenated and
    fed to a wider staging head; the transition head uses only the center
    epoch's backbone features (unchanged from ConfigurableTASA).

    Args:
        context_window: number of neighbor epochs on each side.
            1 → window of 3 (prev, center, next); 2 → window of 5, etc.
        All other args identical to ConfigurableTASA.

    Input:
        x: (B, 2*context_window+1, C, T)  — center epoch at index context_window
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: Optional[int] = 64,
        num_classes: int = 5,
        context_window: int = 1,
    ):
        super().__init__()
        self.context_window = context_window
        n_ctx = 2 * context_window + 1

        self.backbone = ConfigurableBackbone(
            input_channels=input_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
        )
        # Staging head receives concatenated embeddings from all context epochs
        self.staging_head = nn.Sequential(
            nn.Linear(d_model * n_ctx, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        # Transition head uses only the center epoch's backbone features
        self.transition_head = _TransitionDetectionHead(d_model=d_model)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, n_ctx, C, T)
        B, n_ctx, C, T = x.shape

        # Process all epochs through the shared backbone
        feats = self.backbone(x.view(B * n_ctx, C, T))  # (B*n_ctx, L, D)
        L, D = feats.shape[1], feats.shape[2]

        # Global avg pool → per-epoch embeddings
        epoch_embeds = feats.mean(dim=1).view(B, n_ctx, D)  # (B, n_ctx, D)

        # Staging: concatenate all context embeddings
        stage_logits = self.staging_head(epoch_embeds.reshape(B, n_ctx * D))

        # Transition: center epoch's full sequence features
        center_feats = feats.view(B, n_ctx, L, D)[:, self.context_window].contiguous()
        trans_logits = self.transition_head(center_feats)

        return {"stage_logits": stage_logits, "transition_logits": trans_logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
