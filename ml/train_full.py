"""Full VeraSight training run on the complete Express4D corpus.

Runs on a Colab VM (T4). Steps:
  1. Build a subject-disjoint train/val split from every CSV in DATA.
  2. Train the feature-space VAE for 80 epochs (checkpoints + history).
  3. Per-user AW-iForest + predictive GRU on the first K calibration clips.
  4. Write a summary metrics JSON and print progress to stdout.

Intended to run in the background:
    nohup python train_full.py > /content/train.log 2>&1 &
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

DATA = os.environ.get("EXPRESS4D_DATA", "/content/express4d/data")
BUILT = "/content/built"
OUT = "/content/out"
VAE_EPOCHS = int(os.environ.get("VAE_EPOCHS", "80"))
K_USERS = int(os.environ.get("K_USERS", "6"))


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def run(cmd):
    log("run: " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    os.makedirs(BUILT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    csvs = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    log(f"found {len(csvs)} CSVs")
    if len(csvs) < 1000:
        log("WARNING: fewer than 1000 CSVs; dataset download likely incomplete")

    # 1. Build split (subject-disjoint by clip filename).
    rc = run(
        [
            sys.executable, "-m", "ml.data.build_express4d_dataset",
            "--data-dir", DATA,
            "--out", BUILT,
            "--val-frac", "0.15",
            "--min-frames", "60",
            "--rotations",
            "--seed", "0",
        ]
    )
    if rc != 0:
        log("dataset build failed")
        return 1

    # 2. Train VAE.
    rc = run(
        [
            sys.executable, "-m", "ml.models.vae",
            "--train-npy", f"{BUILT}/feats_train.npy",
            "--val-npy", f"{BUILT}/feats_val.npy",
            "--epochs", str(VAE_EPOCHS),
            "--batch", "512",
            "--latent", "16",
            "--out", f"{OUT}/vae_final",
        ]
    )
    if rc != 0:
        log("VAE training failed")
        return 1

    # 3. Per-user iForest + GRU on the first K clips.
    from ml.data.loaders import load_blendshape_csv
    from ml.models.aw_iforest import WeightedIsolationForest, default_anatomical_weights

    def make_features(bs, taus=(1, 10, 100, 1000)):
        parts = [bs]
        for tau in taus:
            s = np.zeros_like(bs)
            s[tau:] = bs[:-tau]
            parts.append(bs - s)
        return np.concatenate(parts, axis=1).astype(np.float32)

    users = {}
    for c in csvs[:K_USERS]:
        name = os.path.basename(c).replace(".csv", "")
        d = load_blendshape_csv(c)
        b = d["blendshapes"].astype(np.float32)
        if len(b) < 100:
            continue
        X = make_features(b)
        m = WeightedIsolationForest(
            n_estimators=200,
            max_samples=min(256, len(X)),
            weights=default_anatomical_weights(260),
            random_state=0,
        )
        m.fit(X)
        scores = m.score_samples(X)
        thr = float(m.threshold_for_quantile(scores, 0.95))

        np.save(f"{BUILT}/{name}_feats.npy", b)
        rc = run(
            [
                sys.executable, "-m", "ml.models.gru",
                "--data", f"{BUILT}/{name}_feats.npy",
                "--window", "30",
                "--horizon", "15",
                "--epochs", "50",
                "--out", f"{OUT}/gru_{name}",
            ]
        )
        users[name] = {
            "frames": int(len(b)),
            "iforest_score_p95": round(thr, 4),
            "gru_rc": rc,
        }
        log(f"user {name}: frames={len(b)}, iforest_p95={thr:.4f}, gru_rc={rc}")

    summary = {
        "csv_count": len(csvs),
        "vae_epochs": VAE_EPOCHS,
        "users": users,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done in {summary['elapsed_s']}s; summary -> {OUT}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
