"""Sanity-check real dataset files before training.

Usage:
    python -m ml.data.stats data/express4d/W000_000.csv
    python -m ml.data.stats data/express4d/*.csv data/captures/*.csv

Prints per-file frame counts, fps, channel ranges, and flags problems (NaN,
all-zero clips, constant channels, out-of-range values). Exits nonzero if any
file fails basic checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .loaders import (
    estimate_fps,
    express4d_nominal_fps,
    load_blendshape_csv,
    load_mesh_memmap,
)


def _channel_stats(bs: np.ndarray) -> dict:
    finite = np.isfinite(bs)
    flags = []
    if not finite.all():
        flags.append("contains_non_finite")
    clean = np.where(finite, bs, 0.0)
    mn, mx = float(clean.min()), float(clean.max())
    mean = float(clean.mean())
    std = float(clean.std())
    frac_zero = float((clean == 0.0).mean())
    if mx > 1.5:
        flags.append("out_of_range(>1.5)")
    if mn < -0.01:
        flags.append("negative")
    if std < 1e-6:
        flags.append("constant_channels")
    if frac_zero > 0.95:
        flags.append("mostly_zero")
    return {
        "min": round(mn, 4),
        "max": round(mx, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "frac_zero": round(frac_zero, 4),
        "flags": flags,
    }


def check_csv(path: Path, kind: str = "blendshape_csv") -> dict:
    data = load_blendshape_csv(path)
    bs = data["blendshapes"]
    fps = express4d_nominal_fps() if "iPhone" in str(path) else None
    if fps is None and data["time"] is not None:
        fps = estimate_fps(data["time"])
    return {
        "file": str(path),
        "kind": kind,
        "frames": int(bs.shape[0]),
        "fps": round(fps, 2) if fps else None,
        "channels": _channel_stats(bs),
    }


def check_express4d_npy(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] not in (52, 61):
        return {
            "file": str(path),
            "kind": "express4d_npy",
            "error": f"expected (N, 52|61), got {arr.shape}",
            "flags": ["unexpected_shape"],
        }
    bs = np.asarray(arr[:, :52], dtype=np.float32)
    return {
        "file": str(path),
        "kind": "express4d_npy",
        "frames": int(bs.shape[0]),
        "cols": int(arr.shape[1]),
        "channels": _channel_stats(bs),
    }


def check_mesh(path: Path) -> dict:
    arr = load_mesh_memmap(path)
    finite = np.isfinite(arr)
    flags = []
    if not finite.all():
        flags.append("contains_non_finite")
    clean = np.where(finite, arr, 0.0)
    mn, mx = float(clean.min()), float(clean.max())
    if mx > 1.5 or mn < -0.01:
        flags.append("not_normalized_0_1")
    return {
        "file": str(path),
        "kind": "mesh_npy",
        "frames": int(arr.shape[0]),
        "dims": int(arr.shape[1]),
        "min": round(mn, 4),
        "max": round(mx, 4),
        "flags": flags,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="CSV/parquet/npy files to check")
    ap.add_argument("--json", action="store_true", help="print machine-readable summary")
    ap.add_argument(
        "--kind",
        choices=["blendshape_csv", "express4d_npy", "mesh_npy"],
        default=None,
        help="force interpretation for non-standard files",
    )
    args = ap.parse_args()

    results = []
    failed = False
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            results.append({"file": raw, "error": "missing"})
            failed = True
            continue
        suffix = p.suffix.lower()
        kind = args.kind
        try:
            if kind == "mesh_npy" or (suffix == ".npy" and kind == "mesh_npy"):
                r = check_mesh(p)
            elif kind == "express4d_npy" or (
                suffix == ".npy" and args.kind is None and not _looks_like_mesh(p)
            ):
                r = check_express4d_npy(p)
            elif suffix == ".npy":
                r = check_mesh(p)
            else:
                r = check_csv(p)
        except Exception as exc:  # noqa: BLE001
            r = {"file": str(p), "error": f"{type(exc).__name__}: {exc}"}
            failed = True
        flags = r.get("channels", {}).get("flags", []) or r.get("flags", [])
        if flags:
            failed = True
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(json.dumps(r, indent=2))
    return 1 if failed else 0


def _looks_like_mesh(path: Path) -> bool:
    """Heuristic: mesh npy is (N, 3660); Express4D npy is (N, 52|61)."""
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.ndim == 2 and arr.shape[1] == 3660
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
