"""Build JSON frames for MeshGenerator.swift from Express4D CSV/feature files.

Reads the 52 ARKit blendshape columns + the 9 rotation columns (HeadYaw/
Pitch/Roll, LeftEyeYaw/Pitch/Roll, RightEyeYaw/Pitch/Roll) and writes the
frame JSON that ml/tools/MeshGenerator.swift consumes.

Usage:
    python -m ml.data.make_meshes --csv data/MySlate_1000_iPhone_cal.csv \
        --out frames_1000.json

Then, on a Mac/iPhone with ARKit:
    swift ml/tools/MeshGenerator.swift --input frames_1000.json \
        --output meshes_1000.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .contract import ARKIT_52, ARKIT_SET, TIME_COLUMN_NAMES

ROTATION_COLUMNS = [
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # Map ARKit-52 by name (case-insensitive).
    present = []
    for name in ARKIT_52:
        src = lower.get(name.lower())
        if src is not None:
            present.append((name, src))

    # Rotation columns by their exact header names (case-insensitive lookup).
    rot_present = []
    for c in ROTATION_COLUMNS:
        src = lower.get(c.lower())
        if src is not None:
            rot_present.append(src)
    if len(rot_present) < 9:
        print(
            f"WARNING: found {len(rot_present)}/9 rotation columns; "
            "head rotation only if all three present."
        )

    n = len(df)
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    frames = []
    for i in range(n):
        bs = {}
        for name, src in present:
            bs[name] = float(df[src].iloc[i])
        head = [float(df[c].iloc[i]) for c in ROTATION_COLUMNS[0:3]]
        leye = [float(df[c].iloc[i]) for c in ROTATION_COLUMNS[3:6]]
        reye = [float(df[c].iloc[i]) for c in ROTATION_COLUMNS[6:9]]
        frames.append(
            {
                "blendshapes": bs,
                "head_rotation": head,
                "left_eye_rotation": leye,
                "right_eye_rotation": reye,
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(frames))
    print(f"wrote {len(frames)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
