---
title: QCal Copilot
emoji: ⚛️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: AI-assisted quantum calibration + CUDA-Q + Ising decoder
---

# QCal Copilot — MVP

AI-assisted quantum calibration. Upload a calibration plot or CSV, get an
analysis from NVIDIA's Ising Calibration vision-language model, receive a
ready-to-run CUDA-Q Python script with the suggested tuning, and execute it
on the local `cudaq` simulator.

```
┌─────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Upload │──▶│   Analyzer   │──▶│  Code gen    │──▶│  Simulator   │
│ (img/csv│   │ (Ising VLM)  │   │  (CUDA-Q)    │   │ (cudaq.sample│
└─────────┘   └──────┬───────┘   └──────────────┘   └──────────────┘
                     │
                     ▼
             ┌───────────────────────────────┐
             │   Decoder  (optional)         │
             │   Ising 3D CNN → PyMatching   │
             └───────────────────────────────┘
```

## Layout

```
app.py                 # Gradio UI + pipeline wiring
qcal/
  data.py              # image/CSV preprocessing
  analyzer.py          # Ising VLM (local HF or NIM)
  codegen.py           # CUDA-Q script generator
  simulator.py         # executes the generated script
  decoder.py           # Ising 3D CNN pre-decoder + MWPM
requirements.txt
```

The analyzer and simulator are decoupled, so adding a later-stage 3D CNN
decoder or swapping in a different model is a one-file change.

## Quick start

### 1. System requirements

- NVIDIA GPU (≥ 24 GB VRAM recommended for local 35B inference; otherwise use
  the NIM endpoint, which runs the model remotely).
- CUDA 12.x drivers.
- Python 3.10 – 3.12.

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install CUDA-Q

CUDA-Q ships from NVIDIA as a standalone package. Pick one:

```bash
# Option A — pip wheels (Linux x86_64, CUDA 12):
pip install cudaq

# Option B — NVIDIA container (recommended for reproducibility):
docker pull nvcr.io/nvidia/nightly/cuda-quantum:latest
```

Confirm the install:

```bash
python -c "import cudaq; print(cudaq.__version__)"
```

### 4. Get access to the Ising Calibration model

You have two options.

**A. Hosted inference via NVIDIA NIM (easiest, no local download)**

1. Sign in at <https://build.nvidia.com/> and request access to
   `nvidia/ising-calibration-1-35b-a3b`.
2. Create an API key.
3. Export it before starting the app:

   ```bash
   export NVIDIA_API_KEY="nvapi-…"
   ```

**B. Local Hugging Face weights (needs substantial VRAM)**

1. Request access to `nvidia/Ising-Calibration-1-35B-A3B` on Hugging Face.
2. Authenticate:

   ```bash
   huggingface-cli login
   ```
3. The analyzer will download weights on first use. Override the model id with
   `QCAL_MODEL_ID` if you want to try a different checkpoint.

### 5. Run the app

```bash
python app.py
```

Open <http://localhost:7860>. Upload a calibration plot, click
**Analyze calibration**, inspect the generated CUDA-Q script, then click
**Run simulation** to execute it on the `cudaq` simulator.

## Deploy to Hugging Face Spaces

This repo is ready to deploy as a Gradio Space (e.g. `athurlow/qcal`). The
YAML frontmatter at the top of this README tells Spaces which SDK to use and
which file to run.

1. Push the repo to the Space:

   ```bash
   git remote add space https://huggingface.co/spaces/athurlow/qcal
   git push space claude/qcal-copilot-mvp-OZ9wj:main
   ```
2. In the Space **Settings → Variables and secrets**, add:
   - `NVIDIA_API_KEY` — required; the hosted Space can't download the 35B
     VLM locally, so the app should call the NIM endpoint.
3. (Optional) Override model ids via Space secrets if you have custom
   deployments: `QCAL_NIM_MODEL`, `QCAL_DECODER_FAST_ID`,
   `QCAL_DECODER_ACCURATE_ID`.
4. **Hardware:** a free CPU Space runs the decoder's small CNN (~1.8M params)
   and the NIM-backed analyzer fine. A GPU Space (T4 or better) is only
   needed if you want to host the calibration VLM locally; `cudaq` requires
   an NVIDIA GPU Space to run the simulation stage.

The app falls back gracefully when dependencies are missing: no
`NVIDIA_API_KEY` → analyzer reports the missing key; no `cudaq` → simulator
button surfaces the install hint; no `pymatching` → decoder shows density
metrics without MWPM timing.

## Error-correction decoder (optional stage)

After a successful calibration analysis, expand the
**"Error-correction decoder (Ising 3D CNN)"** panel to run an NVIDIA Ising
surface-code pre-decoder on a synthetic syndrome volume. The panel lets you:

- Pick the `fast` (~912k params) or `accurate` (~1.79M params) variant.
- Set code distance (d), syndrome rounds (T), physical error rate (p),
  and shot count.
- See before/after metrics: syndrome density, CNN inference time, MWPM
  decode time on raw vs denoised syndromes (via PyMatching), and a
  logical-error-rate proxy.
- View a side-by-side plot of the raw and denoised syndrome slices plus a
  before/after bar chart.

The `p` slider auto-populates from the calibration analysis (larger drive
amplitude mismatch → larger `p`). Running the decoder regenerates the
CUDA-Q script with a header block documenting the decoder variant and
improvement metrics.

**What's synthetic and what's real:** syndromes are sampled from Bernoulli(p)
with a few injected correlated chains, the matching graph is a toy 6-nearest-
neighbor graph (not a stim-generated DEM), and "LER" is a syndrome-weight
proxy. If the Ising decoder weights aren't reachable, the module falls back
to a 3D neighbor-support sparsifier so the pipeline still demos end-to-end.

**Swapping in real data:** call
`qcal.decoder.run_decoder(...)` directly with your own `numpy` volume in place
of the generated one, or replace `generate_syndromes()` with a stim-backed
sampler.

## Environment variables

| Variable                    | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| `NVIDIA_API_KEY`            | API key for the NIM endpoint (backend = `nim`)   |
| `QCAL_MODEL_ID`             | Override local HF calibration VLM id             |
| `QCAL_NIM_MODEL`            | Override NIM model name                          |
| `QCAL_NIM_ENDPOINT`         | Override NIM base URL                            |
| `QCAL_DECODER_FAST_ID`      | Override HF id for the fast decoder variant      |
| `QCAL_DECODER_ACCURATE_ID`  | Override HF id for the accurate decoder variant  |
| `QCAL_HOST`                 | Gradio bind host (default `0.0.0.0`)             |
| `QCAL_PORT`                 | Gradio port (default `7860`)                     |
| `QCAL_SHARE`                | Set to `1` to enable Gradio public share link    |

## Input formats

- **Images** — `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`. Any
  calibration artifact the VLM understands: Rabi chevrons, T1/T2 decays,
  Ramsey fringes, readout histograms, resonator spectroscopy, oscilloscope
  traces.
- **CSV / TSV** — We render a preview image of the first 25 rows so the VLM
  can still reason about tabular sweeps. Large tables are truncated.

## Roadmap (post-MVP)

- 3D CNN surface-code decoder stage wired in after analysis.
- Persist analyses + scripts per session.
- Real Rabi fit instead of the linear `suggested_theta` mapping.
- Auth + multi-user deployment.
- Regression-test golden plots against expected analyses.

## Development

Module boundaries to keep the MVP clean:

- `qcal.data` — file I/O and normalization only.
- `qcal.analyzer` — model calls; returns a strict JSON dict.
- `qcal.codegen` — pure function: analysis dict (+ optional decoder info) → script text.
- `qcal.simulator` — executes script text; never imports `cudaq` itself.
- `qcal.decoder` — Ising 3D CNN sparsifier + optional PyMatching MWPM.
- `app.py` — UI glue only; no ML logic.

### Testing the decoder stage

Without launching the UI:

```python
from qcal.decoder import run_decoder

r = run_decoder(variant="fast", distance=5, rounds=5, error_rate=0.01, n_shots=64)
print(r.markdown())
print("density reduction:", round(r.density_reduction * 100, 1), "%")
```

Through the UI:

1. Upload any calibration plot and click **Analyze calibration**.
2. Expand **Error-correction decoder (Ising 3D CNN)**.
3. Pick `fast` or `accurate`, adjust `d` / `T` / `p`, click **Run decoder**.
4. Inspect the metrics panel (density reduction, MWPM timing, LER proxy
   improvement) and the side-by-side plot.
5. The CUDA-Q script auto-refreshes with a decoder header block.
