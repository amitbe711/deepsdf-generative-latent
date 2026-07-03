"""DeepSDF auto-decoder (Park et al., 2019).

An 8-layer MLP of width 512 with a skip connection that re-injects the input
``[z ; xyz]`` at the middle layer, weight normalization, ReLU + dropout, and an
optional tanh on the scalar SDF output. This is an original implementation;
architectural hyper-parameters follow the paper.

Geometric initialization (Atzmon & Lipman, 2020, "SAL"; Gropp et al., 2020,
"IGR") starts the network as an approximate sphere SDF. Without it, SDF-MLPs can
fall into a degenerate basin where the output is constant in space (the field
never crosses zero, so no surface can be extracted); geometric init makes
training robust across latent dimensions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DeepSDFDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        skip_in: tuple[int, ...] = (4,),
        dropout_prob: float = 0.2,
        use_weight_norm: bool = True,
        use_tanh: bool = True,
        geometric_init: bool = True,
        init_radius: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = latent_dim + 3
        self.skip_in = set(skip_in)
        self.use_tanh = use_tanh
        self.dropout_prob = dropout_prob

        # num_layers hidden layers + 1 output layer.
        dims = [self.input_dim] + [hidden_dim] * num_layers + [1]
        self.num_layers = len(dims) - 1

        linears: list[nn.Linear] = []
        for layer in range(self.num_layers):
            in_dim = dims[layer]
            if layer in self.skip_in:
                in_dim += self.input_dim  # concat original input at the skip layer
            linears.append(nn.Linear(in_dim, dims[layer + 1]))

        if geometric_init:
            self._geometric_init(linears, init_radius)

        # Weight-normalize hidden layers *after* initialization (WN preserves the
        # effective weight at construction time, so the geometric init survives).
        self.layers = nn.ModuleList()
        for layer, linear in enumerate(linears):
            if use_weight_norm and layer < self.num_layers - 1:
                linear = nn.utils.parametrizations.weight_norm(linear)
            self.layers.append(linear)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def _geometric_init(self, linears: list[nn.Linear], radius: float) -> None:
        last = self.num_layers - 1
        for layer, lin in enumerate(linears):
            out_dim, in_dim = lin.weight.shape
            if layer == last:
                # Output layer: approximate ||x|| - radius (negative inside).
                nn.init.normal_(
                    lin.weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-5
                )
                nn.init.constant_(lin.bias, -radius)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0) / math.sqrt(out_dim))
                if layer == 0:
                    # Zero the latent columns so the initial field is a sphere,
                    # identical for every shape code.
                    lin.weight.data[:, : self.latent_dim] = 0.0
                if layer in self.skip_in:
                    # Zero the appended-input columns so the skip does not perturb
                    # the sphere at initialization.
                    lin.weight.data[:, -self.input_dim :] = 0.0

    def forward(self, latent: Tensor, xyz: Tensor) -> Tensor:
        """Predict SDF for ``latent`` [B, D] paired with ``xyz`` [B, 3] -> [B, 1]."""
        inputs = torch.cat([latent, xyz], dim=-1)
        x = inputs
        for layer, linear in enumerate(self.layers):
            if layer in self.skip_in:
                x = torch.cat([x, inputs], dim=-1)
            x = linear(x)
            if layer < self.num_layers - 1:
                x = self.activation(x)
                if self.dropout_prob > 0:
                    x = self.dropout(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
