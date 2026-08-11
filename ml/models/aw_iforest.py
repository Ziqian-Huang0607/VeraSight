"""Anatomically-Weighted Isolation Forest (custom, numpy-only) + sklearn baseline.

sklearn's IsolationForest does not support per-feature split probabilities, so
the weighted version is a small custom implementation: each tree samples the
split feature from a categorical distribution proportional to W_ii, then
partitions uniformly between the feature's min and max in the node. Scoring uses
the standard isolation score 2^(-E(h)/c(n)).

Usage (fit on a real calibration feature file, (N, 260)):
    python -m ml.models.aw_iforest --fit data/user01_calib.npy \
        --val data/user01_calib_val.npy --out checkpoints/iForest/user01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _harmonic(n: float) -> float:
    # Exact small-n harmonic numbers; c(n) is the expected path length.
    if n <= 1:
        return 0.0
    return float(np.sum(1.0 / np.arange(1.0, n)))


def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


class _Node:
    __slots__ = ("feature", "split", "left", "right", "size")

    def __init__(self, feature=None, split=None, left=None, right=None, size=0):
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right
        self.size = size


class WeightedIsolationForest:
    """Isolation Forest with per-feature split probabilities."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int = 256,
        weights: np.ndarray | None = None,
        max_depth: int = 9,
        random_state: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.random_state = random_state
        self.weights = weights  # (D,) non-negative; normalized internally
        self._trees: list[_Node] = []
        self._prob: np.ndarray | None = None
        self._max_len = 0.0

    def fit(self, X: np.ndarray) -> "WeightedIsolationForest":
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._max_len = max(1.0, _c(self.max_samples))
        if self.weights is None:
            self.weights = np.ones(d, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64).ravel()
        if w.shape[0] != d:
            raise ValueError(f"weights length {w.shape[0]} != features {d}")
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones(d)
        self._prob = w / w.sum()

        rng = np.random.default_rng(self.random_state)
        self._trees = []
        for t in range(self.n_estimators):
            idx = rng.choice(n, size=min(self.max_samples, n), replace=False)
            root = self._build(X[idx], rng, depth=0)
            self._trees.append(root)
        return self

    def _build(self, X: np.ndarray, rng, depth: int) -> _Node:
        n = X.shape[0]
        if n <= 1 or depth >= self.max_depth or _c(n) <= 0:
            return _Node(size=n)
        # All columns identical in this node -> leaf.
        if np.all(X.max(axis=0) == X.min(axis=0)):
            return _Node(size=n)
        active = X.max(axis=0) > X.min(axis=0)
        prob = self._prob * active.astype(np.float64)
        prob = prob / prob.sum()
        feat = int(rng.choice(len(prob), p=prob))
        lo, hi = float(X[:, feat].min()), float(X[:, feat].max())
        if hi - lo < 1e-9:
            return _Node(size=n)
        split = rng.uniform(lo, hi)
        mask = X[:, feat] < split
        if mask.all() or not mask.any():
            return _Node(size=n)
        return _Node(
            feature=feat,
            split=split,
            left=self._build(X[mask], rng, depth + 1),
            right=self._build(X[~mask], rng, depth + 1),
            size=n,
        )

    def _path_length(self, x: np.ndarray, node: _Node, depth: int) -> float:
        if node.feature is None:
            return depth + _c(node.size)
        if x[node.feature] < node.split:
            return self._path_length(x, node.left, depth + 1)
        return self._path_length(x, node.right, depth + 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        for i, x in enumerate(X):
            lengths = [self._path_length(x, t, 0) for t in self._trees]
            avg = float(np.mean(lengths)) / self._max_len
            out[i] = 2.0 ** (-avg)
        return out

    def threshold_for_quantile(self, scores: np.ndarray, q: float = 0.95) -> float:
        return float(np.quantile(scores, q))


def fit_sklearn_baseline(X: np.ndarray, n_estimators: int = 200):
    """Uniform-feature IsolationForest for the ablation baseline."""
    from sklearn.ensemble import IsolationForest

    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(256, len(X)),
        random_state=0,
        n_jobs=-1,
    ).fit(X)


def default_anatomical_weights(n_features: int) -> np.ndarray:
    """2.5x on brow/eye channels, 2.0x on mouth-asymmetry, else 1.0.

    Maps the 260-feature layout [b_52 | d1_52 | d10_52 | d100_52 | d1000_52].
    The exact channel sets are documented in pipeline-audit.md; this is the
    initial mapping, to be tuned on real data.
    """
    upper = {
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
        "browOuterUpRight", "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight", "eyeBlinkLeft", "eyeBlinkRight",
    }
    asym = {
        "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    }
    from ml.data.contract import ARKIT_52

    w = np.ones(n_features, dtype=np.float64)
    for block in range(5):  # b, d1, d10, d100, d1000
        for j, name in enumerate(ARKIT_52):
            i = block * 52 + j
            if name in upper:
                w[i] = 2.5
            elif name in asym:
                w[i] = 2.0
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", required=True, help="(N, D) calibration features npy")
    ap.add_argument("--val", default=None, help="held-out features npy for threshold")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--baseline", action="store_true", help="also fit sklearn baseline")
    ap.add_argument("--out", default="checkpoints/iforest")
    args = ap.parse_args()

    X = np.load(args.fit, mmap_mode="r")
    if X.ndim != 2:
        raise SystemExit(f"--fit must be (N, D), got {X.shape}")
    print(f"[iforest] fit data: {X.shape}")

    weights = default_anatomical_weights(X.shape[1])
    model = WeightedIsolationForest(
        n_estimators=args.trees,
        max_samples=min(args.max_samples, len(X)),
        weights=weights,
        random_state=0,
    )
    model.fit(np.asarray(X))
    scores = model.score_samples(np.asarray(X))
    thr = model.threshold_for_quantile(scores, args.quantile)
    print(
        f"[iforest] fit ok; anomaly score mean={scores.mean():.4f} "
        f"p95={thr:.4f} (quantile on fit data; use --val for honest threshold)"
    )

    if args.val:
        Xv = np.asarray(np.load(args.val, mmap_mode="r"))
        sv = model.score_samples(Xv)
        thr_v = model.threshold_for_quantile(sv, args.quantile)
        flagged = float((sv > thr).mean())
        print(
            f"[iforest] val: p95={thr_v:.4f}, "
            f"fraction>fit_p95={flagged:.3f}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": model.weights.tolist(),
        "max_depth": model.max_depth,
        "quantile": args.quantile,
        "threshold": thr,
        "n_features": X.shape[1],
    }
    (out / "iforest.json").write_text(json.dumps(payload, indent=2))
    print(f"[iforest] saved {out / 'iforest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
