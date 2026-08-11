"""Predictive GRU for anomaly filtering on abstract facial states.

Implements the corrected version of the essay's Siamese GRU: a single GRU
encoder reads a 30-frame window of the abstract state a_t = [z_16 | b_52 |
conf] and predicts the mean state over the next k=15 frames. The essay's
contrastive margin term is intentionally dropped (see pipeline-audit.md P1-2:
it was vacuous/self-contradictory). The optional contrastive term can be
re-enabled as temporal positives/negatives if a use case shows it helps.

Usage:
    python -m ml.models.gru --data data/user01_features.npy \
        --window 30 --horizon 15 --epochs 50 --out checkpoints/gru/user01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise SystemExit(
        "torch is required. On Intel Mac install Python 3.11 and torch==2.2.2; "
        "on Colab use the latest torch."
    )


class PredictiveGRU(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            d_in, hidden, layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, d_in),
        )

    def forward(self, window):
        # window: (B, W, D)
        out, _ = self.gru(window)
        return self.head(out[:, -1, :])


def _build_windows(features: np.ndarray, window: int, horizon: int):
    xs, ys = [], []
    for t in range(window, len(features) - horizon):
        xs.append(features[t - window : t])
        ys.append(features[t : t + horizon].mean(axis=0))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="(N, D) feature npy")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="checkpoints/gru")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.asarray(np.load(args.data, mmap_mode="r"), dtype=np.float32)
    print(f"[gru] features: {features.shape}, device: {device}")

    xs, ys = _build_windows(features, args.window, args.horizon)
    n = len(xs)
    n_val = int(n * args.val_frac)
    perm = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[gru] windows={n}, train={len(train_idx)}, val={len(val_idx)}")

    d_in = features.shape[1]
    model = PredictiveGRU(d_in, args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    xt = torch.tensor(xs[train_idx], dtype=torch.float32)
    yt = torch.tensor(ys[train_idx], dtype=torch.float32)
    xv = torch.tensor(xs[val_idx], dtype=torch.float32)
    yv = torch.tensor(ys[val_idx], dtype=torch.float32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm_t = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), args.batch):
            idx = perm_t[i : i + args.batch]
            opt.zero_grad()
            pred = model(xt[idx].to(device))
            loss = criterion(pred, yt[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xv.to(device)), yv.to(device)).item()
        print(
            f"[gru] epoch {epoch + 1}/{args.epochs} "
            f"train_mse={total / len(xt):.6f} val_mse={val_loss:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "d_in": d_in,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "window": args.window,
                    "horizon": args.horizon,
                    "val_mse": val_loss,
                },
                out / "gru_best.pt",
            )
    torch.save(
        {
            "model": model.state_dict(),
            "d_in": d_in,
            "hidden": args.hidden,
            "layers": args.layers,
            "window": args.window,
            "horizon": args.horizon,
            "val_mse": best_val,
        },
        out / "gru_latest.pt",
    )
    print(f"[gru] done; best val_mse={best_val:.6f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
