"""Debug decode: print SDF grid stats and whether marching cubes finds a surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sample import decode_mesh, decode_sdf_grid  # noqa: E402
from src.utils import load_checkpoint  # noqa: E402
from src.models import DeepSDFDecoder, LatentCodes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shape-idx", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable, using cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    cfg = ckpt.get("config", {})

    dec_cfg = cfg.get("decoder", {})
    decoder = DeepSDFDecoder(
        latent_dim=latent_dim,
        hidden_dim=int(dec_cfg.get("hidden_dim", 512)),
        num_layers=int(dec_cfg.get("num_layers", 8)),
        skip_in=tuple(dec_cfg.get("skip_in", [4])),
        dropout_prob=float(dec_cfg.get("dropout_prob", 0.0)),
        use_weight_norm=bool(dec_cfg.get("use_weight_norm", True)),
        use_tanh=bool(dec_cfg.get("use_tanh", True)),
        geometric_init=bool(dec_cfg.get("geometric_init", True)),
        init_radius=float(dec_cfg.get("init_radius", 0.5)),
    ).to(device)
    codes = LatentCodes(int(ckpt["num_shapes"]), latent_dim).to(device)
    decoder.load_state_dict(ckpt["decoder_state"])
    codes.load_state_dict(ckpt["codes_state"])
    decoder.eval()

    z = codes.embedding.weight[args.shape_idx].detach()
    print(f"shape_idx={args.shape_idx} ||z||={z.norm().item():.4f}")

    vol = decode_sdf_grid(decoder, z, resolution=args.resolution, device=device)
    print(
        f"SDF grid {args.resolution}^3: min={vol.min():.4f} max={vol.max():.4f} "
        f"mean={vol.mean():.4f} crosses_zero={vol.min() < 0 < vol.max()}"
    )

    mesh = decode_mesh(decoder, z, resolution=args.resolution, device=device)
    if mesh is None or len(mesh.faces) == 0:
        print("decode_mesh: FAIL (no surface)")
        sys.exit(1)
    print(f"decode_mesh: OK  verts={len(mesh.vertices)} faces={len(mesh.faces)}")


if __name__ == "__main__":
    main()
