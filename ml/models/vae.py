"""Spatial/feature VAE for VeraSight.

Trains a VAE on real data, auto-detecting the input dimension:
- 3660-d: ARKit 1220-vertex mesh frames (iPhone captures or procedural meshes).
- 52-d: ARKit blendshape features (Express4D CSVs).
Both are real data; nothing here is synthesized or fake.

Architecture: encoder 512/128 -> latent, decoder 128/512 -> sigmoid(input_dim).
Latent dim is configurable (default 16). Trained with beta-weighted ELBO.

Usage:
    python -m ml.models.vae --train-npy data/feats_train.npy \
        --val-npy data/feats_val.npy --epochs 80 --batch 512 --out checkpoints/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    raise SystemExit(
        "torch is required for the VAE. On Intel Mac install Python 3.11 and "
        "torch==2.2.2; on Colab use the latest torch."
    )


class SpatialVAE(nn.Module):
    def __init__(self, input_dim: int = 3660, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 0.1):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def _load(path: str | None, label: str) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if p.suffix.lower() != ".npy":
        raise SystemExit(f"{label}: expected a .npy file, got {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise SystemExit(f"{label}: expected 2D (N, D) array, got {arr.shape}")
    print(f"[vae] {label}: {arr.shape[0]} frames, D={arr.shape[1]}, {arr.dtype}")
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-npy", required=True)
    ap.add_argument("--val-npy", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--out", default="checkpoints/vae")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vae] device: {device}")

    train = _load(args.train_npy, "train")
    val = _load(args.val_npy, "val")
    if train is None:
        raise SystemExit("--train-npy is required")

    input_dim = train.shape[1]
    if input_dim not in (52, 3660):
        # Allow any real D, but warn loudly if it's not a known contract.
        print(f"[vae] WARNING: input dim {input_dim} is not 52 (blendshapes) or "
              f"3660 (meshes); training anyway.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = SpatialVAE(input_dim=input_dim, latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[vae] resumed from epoch {start_epoch}")

    history = {"train_mse": [], "val_mse": [], "kld": []}
    n = len(train)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = total_recon = total_kld = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            xb = torch.from_numpy(
                np.ascontiguousarray(np.asarray(train[idx], dtype=np.float32))
            ).to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, recon_loss, kld = vae_loss(recon, xb, mu, logvar, args.beta)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            total_recon += recon_loss.item() * len(xb)
            total_kld += kld.item() * len(xb)

        model.eval()
        with torch.no_grad():
            nv = len(val) if val is not None else min(20000, n)
            xv = torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray((val if val is not None else train)[:nv], dtype=np.float32)
                ).copy()
            ).to(device)
            recon_v, mu_v, lv_v = model(xv)
            val_mse = F.mse_loss(recon_v, xv).item()

        history["train_mse"].append(total_recon / n)
        history["kld"].append(total_kld / n)
        history["val_mse"].append(val_mse)
        print(
            f"[vae] epoch {epoch + 1}/{args.epochs} "
            f"train_mse={history['train_mse'][-1]:.6f} "
            f"kld={history['kld'][-1]:.4f} val_mse={val_mse:.6f}"
        )
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "beta": args.beta,
                "latent_dim": args.latent,
                "input_dim": input_dim,
            }
            torch.save(ckpt, out / f"vae_epoch{epoch + 1:03d}.pt")
            torch.save(ckpt, out / "vae_latest.pt")
            (out / "history.json").write_text(json.dumps(history))

    print(f"[vae] done; checkpoints and history in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
