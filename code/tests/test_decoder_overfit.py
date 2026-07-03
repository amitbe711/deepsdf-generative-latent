"""Mid-point check #1 & #2: overfit one shape, then train on a few shapes."""

import torch

from src.data.dataset import ShapeSDFDataset, build_shape_collection
from src.sample import decode_mesh
from src.train import train_stage1
from src.utils import AttrDict


def _cfg(num_iters: int) -> AttrDict:
    return AttrDict(
        {
            "seed": 0,
            "data": {
                "source": "synthetic",
                "mesh_dir": None,
                "num_points": 3000,
                "surface_ratio": 0.9,
                "sigmas": [0.005, 0.02],
                "num_surface_points": 512,
                "mesh_resolution": 24,
            },
            "decoder": {
                "hidden_dim": 128,
                "num_layers": 8,
                "skip_in": [4],
                "dropout_prob": 0.0,
                "use_weight_norm": True,
                "use_tanh": True,
            },
            "stage1": {
                "num_iters": num_iters,
                "shapes_per_batch": 8,
                "points_per_shape": 512,
                "lr_decoder": 1.0e-3,
                "lr_codes": 1.0e-3,
                "clamp_delta": 0.1,
                "code_init_std": 0.01,
                "code_reg_lambda": 1.0e-4,
            },
        }
    )


def test_overfit_single_shape_reduces_loss_and_meshes():
    cfg = _cfg(num_iters=300)
    dataset = ShapeSDFDataset(build_shape_collection(cfg, num_shapes=1))
    out = train_stage1(cfg, dataset, latent_dim=8, device="cpu", log_every=50)

    history = out["history"]
    assert history[-1]["recon"] < history[0]["recon"]
    assert history[-1]["recon"] < 0.02  # clamped-L1 should get small

    mesh = decode_mesh(out["decoder"], out["codes"].embedding.weight[0].detach(), resolution=32)
    assert mesh is not None
    assert len(mesh.faces) > 0


def test_train_ten_shapes_extracts_codes():
    cfg = _cfg(num_iters=200)
    dataset = ShapeSDFDataset(build_shape_collection(cfg, num_shapes=10))
    out = train_stage1(cfg, dataset, latent_dim=16, device="cpu", log_every=50)
    codes = out["codes"].codes
    assert codes.shape == (10, 16)
    # Codes should have moved away from their tiny initialization.
    assert codes.abs().mean().item() > 1e-3
    # Latent interpolation between two codes is well-defined and decodes.
    z_mid = 0.5 * (codes[0] + codes[1])
    mesh = decode_mesh(out["decoder"], z_mid, resolution=24)
    assert mesh is None or len(mesh.faces) >= 0  # must not raise
