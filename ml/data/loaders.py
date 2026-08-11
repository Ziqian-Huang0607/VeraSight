"""Loaders for the datasets the VeraSight ensemble trains on.

No data is synthesized here. Every loader reads real files and returns arrays
in the canonical ARKit-52 column order defined in contract.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contract import (
    ARKIT_52,
    EXPRESS4D_COUNT_COLUMNS,
    EXPRESS4D_TIME_COLUMNS,
    TIME_COLUMN_NAMES,
    remap_to_canonical,
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_express4d_time(values: pd.Series) -> np.ndarray:
    """Parse Express4D Timecode column into elapsed seconds from the first frame.

    Express4D CSV time is e.g. '19:31:54:11.129'. The format is
    hours:minutes:seconds:milliseconds (the last field before the dot is
    milliseconds, e.g. 54 seconds + 11.129 ms). Because each frame increments
    that field by ~1 ms (not ~16.7 ms), the ms field under-counts time between
    frames; this is a known Express4D timestamp quirk. We parse it as
    h*3600 + m*60 + s + ms/1000, then unwrap the minute/second rollovers.
    Non-numeric fallback returns sequential frame indices at 60 fps.
    """
    raw = values.astype(str)
    def to_seconds(parts):
        try:
            parts = [float(p) for p in parts]
        except (ValueError, TypeError):
            return None
        if len(parts) == 4:
            h, m, s, ms = parts
            return h * 3600 + m * 60 + s + ms / 1000.0
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        return sum(float(p) * (60 ** (len(parts) - 1 - i)) for i, p in enumerate(parts))

    nums = raw.str.split(":").apply(to_seconds)
    if nums.isna().any():
        # Fallback: sequential frames.
        return np.arange(len(values), dtype="float64") / 60.0
    t = nums.to_numpy(dtype="float64").copy()
    # The day-clock wraps every minute (the :ss.fff field resets to ~0 after
    # 59.xxx). Unwrap decreases by adding 60 seconds per wrap.
    for i in range(1, len(t)):
        if t[i] < t[i - 1] - 30:
            t[i:] += 60.0
    if len(t):
        t = t - t[0]
    return t


def load_blendshape_csv(path: str | Path) -> dict:
    """Load an Express4D-style or VeraSight-style CSV of ARKit blendshapes.

    Expected layout: one row per frame; a time column (time/timecode/timestamp)
    and any subset of the 52 ARKit blendshape names as columns. Extra numeric
    columns (e.g. head/eye rotations) are returned as-is.

    Returns dict with: time (T,), blendshapes (T, 52) canonical order,
    rotations (T, R) or None, column_names.
    """
    path = Path(path)
    df = _clean_columns(pd.read_csv(path))
    cols = list(df.columns)

    # Express4D ships a BlendshapeCount column; the 52 named blendshape columns
    # follow it. Use the count when present to select the blendshape block.
    count_col = next(
        (c for c in cols if c.lower() in EXPRESS4D_COUNT_COLUMNS), None
    )
    count = None
    if count_col is not None:
        count = int(df[count_col].iloc[0])
    time_col = next(
        (c for c in cols if c.lower() in TIME_COLUMN_NAMES
         or c.lower() in EXPRESS4D_TIME_COLUMNS),
        None,
    )
    time = None
    if time_col is not None:
        numeric = pd.to_numeric(df[time_col], errors="coerce")
        if numeric.notna().all():
            time = numeric.to_numpy(dtype="float64")
        else:
            time = _parse_express4d_time(df[time_col])

    # BlendshapeCount is the full feature-vector width (52 blendshapes + 9
    # rotations = 61 in Express4D), not the number of named blendshape
    # columns. Always select the named ARKit-52 columns explicitly.
    bs, present = remap_to_canonical(df, cols)

    # Rotations are the columns NOT in the ARKit-52 vocabulary (case-insensitive).
    ark_lower = {a.lower() for a in ARKIT_52}
    rot_cols = [
        c for c in cols
        if c.lower() not in ark_lower
        and (time_col is None or c != time_col)
        and (count_col is None or c != count_col)
    ]
    rotations = None
    if rot_cols:
        rotations = df[rot_cols].to_numpy(dtype="float32")

    return {
        "path": str(path),
        "time": time,
        "blendshapes": bs,
        "rotations": rotations,
        "column_names": present,
    }


def load_verasight_csv(path: str | Path) -> dict:
    """Alias for load_blendshape_csv; VeraSight export uses the same contract."""
    return load_blendshape_csv(path)


def load_mesh_memmap(path: str | Path) -> np.ndarray:
    """Load a (N, 3660) mesh array (own captures, fp16 or fp32, memmapped)."""
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 3660:
        raise ValueError(
            f"{path}: expected shape (N, 3660), got {arr.shape}. "
            "Not an ARKit 1220-vertex mesh file."
        )
    return arr


def estimate_fps(time: np.ndarray | None) -> float | None:
    """Median frame rate from a monotonic time column."""
    if time is None or len(time) < 2:
        return None
    dt = np.diff(time)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return float(1.0 / np.median(dt))


def express4d_nominal_fps() -> float:
    """Express4D was captured at 60 fps; its CSV timestamps under-count time
    (1 ms tick per frame instead of ~16.7 ms), so use the declared rate rather
    than timestamps for FPS. See .agents/research/dataset-plan.md."""
    return 60.0
