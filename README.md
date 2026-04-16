# QCal Copilot — MVP

AI-assisted quantum calibration. Upload a calibration plot or CSV, get an
analysis from NVIDIA's Ising Calibration vision-language model, receive a
ready-to-run CUDA-Q Python script with the suggested tuning, and execute it
on the local `cudaq` simulator.

```
┌─────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Upload │──▶│   Analyzer   │──▶│  Code gen    │──▶│  Simulator   │
│ (img/csv│   │ (Ising VLM)  │   │  (CUDA-Q)    │   │ (cudaq.sample│
└─────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

## Layout

```
app.py                 # Gradio UI + pipeline wiring
qcal/
  data.py              # image/CSV preprocessing
  analyzer.py          # Ising VLM (local HF or NIM)
  codegen.py           # CUDA-Q script generator
  simulator.py         # executes the generated script
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

## Environment variables

| Variable            | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| `NVIDIA_API_KEY`    | API key for the NIM endpoint (backend = `nim`)        |
| `QCAL_MODEL_ID`     | Override local HF model id                            |
| `QCAL_NIM_MODEL`    | Override NIM model name                               |
| `QCAL_NIM_ENDPOINT` | Override NIM base URL                                 |
| `QCAL_HOST`         | Gradio bind host (default `0.0.0.0`)                  |
| `QCAL_PORT`         | Gradio port (default `7860`)                          |
| `QCAL_SHARE`        | Set to `1` to enable Gradio public share link         |

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
- `qcal.codegen` — pure function: analysis dict → script text.
- `qcal.simulator` — executes script text; never imports `cudaq` itself.
- `app.py` — UI glue only; no ML logic.
