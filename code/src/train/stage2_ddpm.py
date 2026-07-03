"""Stage 2 (generator #2): train a DDPM over the frozen latent codes."""

from __future__ import annotations

from typing import Any

import torch

from ..models.ddpm import LatentDDPM


def train_ddpm(
    cfg: Any,
    codes: torch.Tensor,
    device: str = "cpu",
    progress: bool = False,
) -> dict[str, Any]:
    """Train the latent DDPM on ``codes`` [N, D]. Returns model + loss history."""
    d = cfg.ddpm
    codes = codes.detach().float().to(device)
    latent_dim = codes.shape[1]

    model = LatentDDPM(
        latent_dim=latent_dim,
        num_timesteps=int(d.num_timesteps),
        beta_start=float(d.beta_start),
        beta_end=float(d.beta_end),
        schedule=str(d.schedule),
        hidden_dim=int(d.hidden_dim),
        num_layers=int(d.num_layers),
        time_embed_dim=int(d.time_embed_dim),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(d.lr))
    num_iters = int(d.num_iters)
    batch_size = int(d.batch_size)

    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 1)
    history: list[dict[str, float]] = []

    iterator = range(num_iters)
    if progress:
        from tqdm import trange

        iterator = trange(num_iters, desc="ddpm")

    model.train()
    num_codes = codes.shape[0]
    for step in iterator:
        batch_idx = torch.randint(
            0, num_codes, (min(batch_size, num_codes),), generator=generator
        )
        z_0 = codes[batch_idx]
        loss = model.loss(z_0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == num_iters - 1:
            history.append({"step": float(step), "loss": float(loss.item())})

    return {"model": model, "history": history}
