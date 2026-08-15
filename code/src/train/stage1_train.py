"""Stage 1: train the DeepSDF auto-decoder jointly with per-shape latent codes.

Loss = clamped-L1 SDF reconstruction + latent-code regularization, exactly as in
DeepSDF (Park et al., 2019, Eq. 5 / Sec. 4). The decoder weights ``theta`` and
the codes ``{z_i}`` are optimized together.

Optional Eikonal regularization (Gropp et al., 2020, "Implicit Geometric
Regularization for Learning Shapes") penalizes ``|| grad_x f(z, x) || != 1``,
pushing the network toward a genuine *metric* SDF instead of an arbitrary
scalar field that merely fits sampled point values. This directly targets the
"low pointwise loss but incoherent/blobby extracted mesh" failure mode: plain
clamped-L1 only constrains the field AT the sampled points, so between them it
can behave arbitrarily, producing spurious zero-crossings or blob-like
surfaces once resolved on a dense marching-cubes grid. Opt-in via
``stage1.eikonal_lambda`` (default 0.0 — off, matching all existing validated
configs) since it costs a second backward pass (double the memory/time).

CAUTION on ``eikonal_lambda`` scale: with clamped-L1, ``recon`` is bounded by
``clamp_delta`` (tiny, e.g. O(1e-2) at delta=0.1), while ``eikonal_loss`` is
unclamped and naturally O(1). The literature default of 0.1 (Gropp et al.)
assumes an unclamped data loss of comparable O(1) scale; applied here it can
make the eikonal term dominate the total loss by ~10x, collapsing the field
to a degenerate near-z-independent solution with no zero level set inside the
eval cube (eikonal's trivial minimizers are not unique/anchored to the actual
shape). Scale ``eikonal_lambda`` down roughly by ``clamp_delta`` (e.g. 0.01)
so it regularizes rather than dominates, and watch that ``eik`` in the logs
actually trends down over training rather than plateauing high immediately.

Optional ``code_reg_warmup_iters`` ramps ``code_reg_lambda`` linearly from 0
up to its configured value over the first N iterations (default 0 -- no
warmup, full strength from step 1, matching all existing validated configs).
Rationale: Adam normalizes each parameter's step size by that parameter's own
gradient RMS, so early on -- when the codes' recon-driven gradient is still
tiny/noisy (decoder hasn't learned meaningful per-shape structure yet) --
Adam takes near-``lr``-sized steps toward whichever direction is most
*consistent* over time, largely independent of raw gradient magnitude.
``code_reg`` gives a tiny but perfectly consistent pull toward ``z=0`` every
single step, and can win this race outright even when the codes' initial
coupling to the decoder is nonzero (verified: a 100x stronger latent-column
init at ``hidden_dim=512`` still collapsed |z| at the *same rate* as no fix
at all). Silencing ``code_reg`` for a short warmup removes this competing
force during the critical early window, letting whatever (however weak)
recon signal exists establish real per-shape structure uncontested.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..models.decoder import DeepSDFDecoder
from ..models.latent_codes import LatentCodes
from ..utils.log import status


def clamped_l1_loss(pred: Tensor, target: Tensor, delta: float) -> Tensor:
    pred_c = torch.clamp(pred, -delta, delta)
    target_c = torch.clamp(target, -delta, delta)
    return torch.abs(pred_c - target_c).mean()


def eikonal_loss(pred: Tensor, points: Tensor) -> Tensor:
    """Penalize deviation of ``|| grad_x pred ||`` from 1 (unit-norm SDF gradient).

    ``points`` must be a leaf tensor with ``requires_grad_(True)`` and ``pred``
    must have been produced by a graph rooted at it. ``create_graph=True`` lets
    the eikonal term itself be backpropagated into the decoder weights.
    """
    grad = torch.autograd.grad(
        outputs=pred,
        inputs=points,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def train_stage1(
    cfg: Any,
    dataset: Any,
    latent_dim: int,
    device: str = "cpu",
    log_every: int = 200,
    progress: bool = False,
    verbose: bool = False,
    prefix: str | None = None,
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
    eik_lambda = float(s1.get("eikonal_lambda", 0.0))
    code_reg_warmup_iters = int(s1.get("code_reg_warmup_iters", 0))

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
        if eik_lambda > 0:
            pts.requires_grad_(True)

        z = codes(idx)
        pred = decoder(z, pts).squeeze(-1)
        recon = clamped_l1_loss(pred, sdf, delta)
        # Regularize codes toward the origin (MAP under a zero-mean Gaussian prior).
        # Ramp up from 0 over the warmup window so it isn't competing with the
        # still-tiny/noisy recon signal for z during the critical early steps.
        if code_reg_warmup_iters > 0:
            reg_lambda = code_reg * min(1.0, step / code_reg_warmup_iters)
        else:
            reg_lambda = code_reg
        reg = reg_lambda * torch.mean(torch.sum(z**2, dim=-1))
        loss = recon + reg

        eik = None
        if eik_lambda > 0:
            eik = eikonal_loss(pred, pts)
            loss = loss + eik_lambda * eik

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_iters - 1:
            row = {
                "step": float(step),
                "loss": float(loss.item()),
                "recon": float(recon.item()),
                "reg": float(reg.item()),
            }
            if eik is not None:
                row["eikonal"] = float(eik.item())
            history.append(row)
            if verbose:
                z_norm = float(torch.mean(torch.norm(z, dim=-1)).item())
                eik_str = f" eik={row['eikonal']:.4f}" if eik is not None else ""
                status(
                    f"stage1 {int(step) + 1}/{num_iters} "
                    f"loss={row['loss']:.4f} recon={row['recon']:.4f} reg={row['reg']:.6f}"
                    f"{eik_str} |z|={z_norm:.4f}",
                    prefix=prefix,
                )
                if step >= 400 and z_norm < 0.05:
                    status(
                        "WARN: |z| collapsing — codes near zero; "
                        "try use_tanh: false and lr_codes <= lr_decoder",
                        prefix=prefix,
                    )

    return {"decoder": decoder, "codes": codes, "history": history}
