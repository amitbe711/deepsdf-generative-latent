"""Per-shape latent codes for the auto-decoder.

Each training shape owns one trainable code, initialized from a tight Gaussian
(DeepSDF uses std ~= 0.01). Codes are optimized *jointly* with the decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentCodes(nn.Module):
    def __init__(self, num_shapes: int, latent_dim: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.num_shapes = num_shapes
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, shape_idx: Tensor) -> Tensor:
        return self.embedding(shape_idx)

    @property
    def codes(self) -> Tensor:
        """All latent codes as a detached tensor of shape (num_shapes, latent_dim)."""
        return self.embedding.weight.detach()
