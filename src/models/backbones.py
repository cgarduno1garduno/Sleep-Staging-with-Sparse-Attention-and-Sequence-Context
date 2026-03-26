import torch
import torch.nn as nn
import math

class SparseAttention(nn.Module):
    """
    Local/Sparse Attention Mechanism.
    Restricts attention to a local window around each token.
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_device(self):
        """Explicitly check for MPS availability."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        device = self._get_device()
        # Ensure input is on the correct device
        if x.device != device:
            x = x.to(device)

        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create Local Window Mask (Sparse Attention)
        # 1s inside window, 0s outside
        # |i - j| <= window_size
        indices = torch.arange(seq_len, device=device).unsqueeze(0)
        # (L, L) matrix where value is |i - j|
        distance_matrix = torch.abs(indices - indices.transpose(0, 1))

        # Mask: 1 (keep) where distance <= window, 0 (mask) otherwise
        # In PyTorch attention, we usually ADD -inf to masked positions, or use boolean True to mask.
        # Let's say mask=True means "ignore".
        sparse_mask = distance_matrix > self.window_size

        # Apply mask
        scores = scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if mask is not None:
             scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        out = torch.matmul(attn_weights, v) # (B, H, L, D_head)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)

class SparseTransformerBackbone(nn.Module):
    """
    Transformer Backbone using Sparse (Local) Attention.
    Contribution #2: Efficiency-Driven MTL.
    """
    def __init__(self, input_channels: int = 3, d_model: int = 64, n_layers: int = 4, n_heads: int = 4, window_size: int = 64):
        super().__init__()
        self.d_model = d_model

        # Patch Embedding / Feature Projection
        # Reducing 3000 samples -> Sequence Length L
        # E.g. Conv1d with stride can downsample.
        # Let's assume a simple projection that preserves time resolution or downsamples slightly.
        # For Sleep-EDF (30s @ 100Hz = 3000pts), standard models often downsample.
        # Let's use a Conv block to get features and downsample by 4 -> 750 tokens, or similar.
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model // 2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
            # 3000 -> 1500 -> 750 length
        )

        self.layers = nn.ModuleList([
            SparseAttention(d_model, n_heads, window_size=window_size)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(n_layers)
        ])

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Channels, Time) e.g. (B, 3, 3000)
        Returns:
            (B, Seq_Len, D_model)
        """
        device = self._get_device()
        if x.device != device:
            x = x.to(device)

        # Embedding: (B, C, T) -> (B, D, L)
        x = self.embedding(x)

        # Permute for Transformer: (B, L, D)
        x = x.permute(0, 2, 1)

        for attn, norm, ffn in zip(self.layers, self.layer_norms, self.ffns):
            # Attention Block
            residual = x
            x = attn(x)
            x = residual + x
            x = norm(x)

            # FFN Block
            residual = x
            x = ffn(x)
            x = residual + x
            x = norm(x) # Usually LN after FFN too

        return x
