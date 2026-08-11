"""Canonical ARKit-52 ordering and channel policy.

The order matches Apple's ARFaceAnchor.BlendShapeLocation list used by the iOS
client. Express4D CSVs and MediaPipe output are all
remapped to this order by name before training.
"""

import numpy as np

ARKIT_52 = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

ARKIT_SET = set(ARKIT_52)

TIME_COLUMN_NAMES = {"time", "timecode", "timestamp", "time_s"}
EXPRESS4D_TIME_COLUMNS = {"timecode"}
EXPRESS4D_COUNT_COLUMNS = {"blendshapecount"}

# Channels that 2D/MediaPipe-style estimators cannot recover reliably. Zero them
# when ingesting MediaPipe-derived streams (see .agents/research/dataset-plan.md).
UNRELIABLE_2D = {
    "jawForward", "jawLeft", "jawRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "cheekPuff", "tongueOut",
}


def remap_to_canonical(df, value_columns):
    """Reorder a frame table's blendshape columns into ARKIT_52 order.

    Returns (canonical_array (T, 52) float32, column_names_used). Missing
    channels become zeros; extra non-blendshape columns are ignored. Matching is
    case-insensitive (Express4D ships 'BrowDownLeft', ARKit is 'browDownLeft').
    """
    value_lower = {c.lower(): c for c in value_columns}
    present = []
    idx = []
    for j, name in enumerate(ARKIT_52):
        src = value_lower.get(name.lower())
        if src is not None:
            present.append(src)
            idx.append(j)
    out = np.zeros((len(df), len(ARKIT_52)), dtype="float32")
    if present:
        out[:, idx] = df[present].to_numpy(dtype="float32")
    return out, present
