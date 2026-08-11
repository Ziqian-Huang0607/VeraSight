"""Build a training-ready memmap dataset from the raw Express4D files.

The Express4D repo's official loader needs extra artifacts (split JSON,
mean/std, FLAME/SPACY). This module bypasses it and reads the raw CSV/NPY
files directly, which is what the VeraSight VAE/GRU need.

Outputs under --out:
  meshes_train.npy   (N, 3660) float16  - if --mesh (own captures)
  feats_train.npy    (N, D)    float16  - ARKit-52 blendshape + optional rotation
  feats_val.npy      (N, D)    float16
  manifest.json      per-clip metadata (subject, frames, source file)
  stats.json         per-feature min/mean/std for later normalization

Usage (CPU, local or Colab):
    python -m ml.data.build_express4d_dataset --data-dir data/express4d/data \
        --out data/express4d_built --val-frac 0.15 --max-clips 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .loaders import load_blendshape_csv


def _subject_of(name: str) -> str:
    # e.g. MySlate_100_iPhone_cal -> "MySlate_100"
    return name.rsplit("_iPhone", 1)[0]


def build(args) -> int:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSV files under {data_dir}")
    print(f"[build] found {len(csvs)} CSV files in {data_dir}")

    rng = np.random.default_rng(args.seed)
    # Subject-disjoint split: hold out whole subjects for validation.
    subjects = sorted({_subject_of(c.stem) for c in csvs})
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * args.val_frac))
    val_subjects = set(subjects[:n_val])
    print(
        f"[build] subjects={len(subjects)}, val_subjects={len(val_subjects)} "
        f"(subject-disjoint)"
    )

    manifest = []
    feats_train, feats_val = [], []
    n_train = n_val = 0
    for c in csvs:
        subj = _subject_of(c.stem)
        try:
            d = load_blendshape_csv(c)
            bs = d["blendshapes"]  # (T, 52) canonical
            if bs.shape[0] < args.min_frames:
                continue
            if not np.isfinite(bs).all():
                continue
            cols = [bs]
            if args.rotations and d["rotations"] is not None:
                cols.append(d["rotations"][: bs.shape[0]])
            feats = np.concatenate(cols, axis=1).astype(np.float16)
        except Exception as exc:  # noqa: BLE001
            print(f"[build] skipping {c.name}: {exc}")
            continue
        row = {
            "file": c.name,
            "subject": subj,
            "frames": int(feats.shape[0]),
            "split": "val" if subj in val_subjects else "train",
        }
        manifest.append(row)
        if subj in val_subjects:
            feats_val.append(feats)
            n_val += feats.shape[0]
        else:
            feats_train.append(feats)
            n_train += feats.shape[0]

    if not feats_train:
        raise SystemExit("no usable training clips after filtering")

    # Limit clips for a quick smoke run if requested.
    if args.max_clips and args.max_clips > 0:
        feats_train = feats_train[: args.max_clips]
        n_train = sum(f.shape[0] for f in feats_train)

    def _save(feats, name):
        if not feats:
            return
        arr = np.concatenate(feats, axis=0).astype(np.float16)
        np.save(out_dir / name, arr)
        return arr

    train_arr = _save(feats_train, "feats_train.npy")
    val_arr = _save(feats_val, "feats_val.npy")

    stats = None
    if train_arr is not None:
        stats = {
            "min": float(train_arr.min()),
            "mean": float(train_arr.mean()),
            "std": float(train_arr.std()),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats or {}, indent=2))
    print(
        f"[build] done: train={n_train} frames, val={n_val} frames, "
        f"D={train_arr.shape[1] if train_arr is not None else 0} -> {out_dir}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--rotations", action="store_true", help="append 9 rotation cols")
    ap.add_argument("--max-clips", type=int, default=0, help="limit train clips (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
