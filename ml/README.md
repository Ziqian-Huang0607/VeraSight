# VeraSight ML Ensemble: Getting Started

## Decisions (made, not open questions)

1. **Primary data: your own iPhone captures.** The record mode is the only source
   of (a) ARKit-topology mesh frames for the VAE, (b) per-user calibration
   blocks, and (c) voluntary/involuntary anomaly ground truth. Nothing external
   replaces it.
2. **First external corpus: Express4D.** 1,205 sequences, 18 participants,
   60 Hz, true iPhone TrueDepth ARKit-52 blendshapes + rotations, CSV. Same
   sensor modality as VeraSight. Get it now via the Google Form:
   https://forms.gle/uSgMH7J6cpPC4oMY9 (repo: https://github.com/jaron1990/Express4D)
   Hugging Face (14,703 clips, 30 fps, MediaPipe-extracted ARKit-52 from real
   video: MEAD/HDTF/CREMA-D). No license gate; `load_dataset` or parquet. Use
   ground truth; zero unreliable 2D channels.
4. **Existing DB: do not use.** `core/data/verasight_training.db` and
   `core/src/scripts/unsupervised_pipeline.py` are the retired mock-noise path
   (random arrays, KMeans). Nothing useful trains from them.
5. **No fake data.** Never generate random/noise arrays as training data. The
   only synthetic input allowed is `ARFaceGeometry(blendShapes:)` procedural
   meshes produced by the iOS code path, used as VAE augmentation alongside real
   captures.

## Order of operations

1. Download Express4D (Google Form) and dump a few CSVs into `data/express4d/`.
3. Run the data checker on each real file:
   Verify the printed statistics look sane before any training.
4. Record your own capture sessions (existing pipeline) and export them to the
   same CSV contract (time + 52 blendshape columns, canonical order) plus mesh
   `.npy` files for the VAE.
5. Train the VAE (Colab preferred): `python -m ml.models.vae --train-npy ...`.
6. Train per-user AW-iForest + GRU on calibration data:
   `python -m ml.models.aw_iforest --fit ...`
   `python -m ml.models.gru --data ...`

## Local setup (Intel Mac)

torch 2.2.2 is the last version with macOS x86_64 wheels and it is CPU-only
(no MPS). It requires Python <= 3.12, so use a dedicated interpreter:

```bash
# install Python 3.11 (e.g. brew install python@3.11, pyenv, or miniforge)
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r ml/requirements.txt        # torch==2.2.2 on this machine
```

Everything in `ml/` is written to run on torch 2.2.2 CPU (Colab runs the same
code with latest torch on T4). The models are small enough that CPU-only local
inference is not a bottleneck: VAE encoder ~4M params, GRU hidden-64, iForest
trees. No MPS branch is required.

## Colab workflow

The training CLIs are the same there:

```python
!pip install numpy pandas scikit-learn torch --quiet
!git clone <repo>   # or mount Drive with data
!python -m ml.models.vae --train-npy data/captures_train.npy --val-npy data/captures_val.npy \
  --epochs 80 --batch 512 --out checkpoints/
```

Checkpoint/resume is built into the VAE and GRU CLIs. Keep converted feature
data under ~8 GB on Drive. Full session budgets are in
`.agents/research/training-plan-colab.md` and `.agents/research/model-feasibility.md`.

## What to verify before training

- Blendshape values in `[0, 1]`, no NaN, no all-zero clips, no constant channels.
- Subject-disjoint train/val/test splits (never the same person in two splits).
- MediaPipe-derived channels that 2D cannot recover are zeroed
  (`jawForward`, `jawLeft/Right`, `mouthDimple*`, `cheekPuff`, `tongueOut`).

## File map

| File | Purpose |
| --- | --- |
| `ml/data/contract.py` | Canonical ARKit-52 order + unreliable-channel set |
| `ml/data/stats.py` | CLI data sanity checker (`python -m ml.data.stats <paths>`) |
| `ml/models/vae.py` | Spatial VAE + training CLI |
| `ml/models/aw_iforest.py` | Weighted Isolation Forest (custom, numpy) + sklearn baseline |
| `ml/models/gru.py` | Predictive GRU + training CLI |
| `tests/fixture_express4d_format.csv` | Format smoke-test fixture only (not training data) |
