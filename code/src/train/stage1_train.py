"""Stage 1: train the DeepSDF auto-decoder jointly with per-shape latent codes.

Loss = clamped-L1 SDF reconstruction + latent-code regularization, exactly as in
DeepSDF (Park et al., 2019, Eq. 5 / Sec. 4). The decoder weights ``theta`` and
the codes ``{z_i}`` are optimized together.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..models.decoder import DeepSDFDecoder
from ..models.latent_codes import LatentCodes


def clamped_l1_loss(pred: Tensor, target: Tensor, delta: float) -> Tensor:
    pred_c = torch.clamp(pred, -delta, delta)
    target_c = torch.clamp(target, -delta, delta)
    return torch.abs(pred_c - target_c).mean()


def train_stage1(
    cfg: Any,
    dataset: Any,
    latent_dim: int,
    device: str = "cpu",
    log_every: int = 200,
    progress: bool = False,
) -> dict[str, Any]:
    """Train the auto-decoder. Returns decoder, codes and a loss history."""
    s1 = cfg.stage1
    generator = torch.Generator().manual_seed(int(cfg.seed))

    decoder = DeepSDFDecoder(
        latent_dim=latent_dim,
        hidden_dim=int(cfg.decoder.hidden_dim),
        num_layers=int(cfg.decoder.num_layers),
        skip_in=tuple(cfg.decoder.skip_in),
        dropout_prob=float(cfg.decoder.dropout_prob),
        use_weight_norm=bool(cfg.decoder.use_weight_norm),
        use_tanh=bool(cfg.decoder.use_tanh),
        geometric_init=bool(cfg.decoder.get("geometric_init", True)),
        init_radius=float(cfg.decoder.get("init_radius", 0.5)),
    ).to(device)
    codes = LatentCodes(
        dataset.num_shapes, latent_dim, init_std=float(s1.code_init_std)
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": float(s1.lr_decoder)},
            {"params": codes.parameters(), "lr": float(s1.lr_codes)},
        ]
    )

    delta = float(s1.clamp_delta)
    code_reg = float(s1.code_reg_lambda)
    num_iters = int(s1.num_iters)
    shapes_per_batch = int(s1.shapes_per_batch)
    points_per_shape = int(s1.points_per_shape)

    history: list[dict[str, float]] = []
    iterator = range(num_iters)
    if progress:
        from tqdm import trange

        iterator = trange(num_iters, desc="stage1")

    decoder.train()
    codes.train()
    for step in iterator:
        idx, pts, sdf = dataset.random_batch(
            shapes_per_batch, points_per_shape, generator=generator
        )
        idx, pts, sdf = idx.to(device), pts.to(device), sdf.to(device)

        z = codes(idx)
        pred = decoder(z, pts).squeeze(-1)
        recon = clamped_l1_loss(pred, sdf, delta)
        # Regularize codes toward the origin (MAP under a zero-mean Gaussian prior).
        reg = code_reg * torch.mean(torch.sum(z**2, dim=-1))
        loss = recon + reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_iters - 1:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.item()),
                    "recon": float(recon.item()),
                    "reg": float(reg.item()),
                }
            )

    return {"decoder": decoder, "codes": codes, "history": history}
