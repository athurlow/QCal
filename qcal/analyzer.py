"""Vision-language analysis for quantum calibration plots.

Wraps two inference backends:
  * local Hugging Face load of `nvidia/Ising-Calibration-1-35B-A3B`, and
  * the NVIDIA NIM endpoint on build.nvidia.com (fallback / low-VRAM path).

The analyzer returns a JSON-shaped dict the rest of the app can render and
feed into CUDA-Q code generation.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image


DEFAULT_MODEL_ID = os.getenv("QCAL_MODEL_ID", "nvidia/Ising-Calibration-1-35B-A3B")
NIM_ENDPOINT = os.getenv(
    "QCAL_NIM_ENDPOINT",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)
NIM_MODEL = os.getenv("QCAL_NIM_MODEL", "nvidia/ising-calibration-1-35b-a3b")

SYSTEM_PROMPT = """You are QCal Copilot, an expert quantum-hardware calibration assistant.
You receive an image of a calibration artifact (e.g. Rabi chevron, T1/T2 decay,
readout histogram, resonator spectroscopy, oscilloscope trace, Ramsey fringes)
and must return a strict JSON object with this schema:

{
  "experiment": "<string — detected experiment type>",
  "qubit_id": "<string or null>",
  "issues": ["<short string>", ...],
  "metrics": {"<metric name>": "<value with units>", ...},
  "recommended_parameters": {
      "drive_amplitude": <float>,
      "drive_frequency_ghz": <float>,
      "pulse_duration_ns": <float>,
      "readout_threshold": <float>,
      "<any other relevant knob>": <value>
  },
  "drift_prediction": "<short string>",
  "confidence": <float between 0 and 1>,
  "notes": "<1-3 sentence plain-English summary>"
}

Only output the JSON. Do not wrap it in markdown fences.
"""

USER_PROMPT_TEMPLATE = (
    "Analyze this quantum calibration artifact ({source}) and return the JSON "
    "described in the system prompt.{extra}"
)


@dataclass
class AnalysisResult:
    raw_text: str
    parsed: dict = field(default_factory=dict)
    backend: str = "unknown"
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.parsed)

    def markdown(self) -> str:
        if self.error:
            return f"**Analyzer error ({self.backend}):** {self.error}\n\n```\n{self.raw_text}\n```"
        p = self.parsed
        lines = [
            f"**Experiment:** {p.get('experiment', 'n/a')}",
            f"**Qubit:** {p.get('qubit_id', 'n/a')}",
            f"**Confidence:** {p.get('confidence', 'n/a')}",
            "",
            "**Detected issues:**",
        ]
        for issue in p.get("issues", []) or ["(none reported)"]:
            lines.append(f"- {issue}")
        lines.append("")
        lines.append("**Metrics:**")
        for k, v in (p.get("metrics") or {}).items():
            lines.append(f"- `{k}` = {v}")
        lines.append("")
        lines.append("**Recommended parameters:**")
        for k, v in (p.get("recommended_parameters") or {}).items():
            lines.append(f"- `{k}` = {v}")
        lines.append("")
        lines.append(f"**Drift prediction:** {p.get('drift_prediction', 'n/a')}")
        lines.append("")
        lines.append(f"**Notes:** {p.get('notes', '')}")
        lines.append("")
        lines.append(f"_Backend: {self.backend}_")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend: NVIDIA NIM (HTTP)
# ---------------------------------------------------------------------------

def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _analyze_via_nim(image: Image.Image, extra: str, source: str) -> AnalysisResult:
    import requests

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
    if not api_key:
        return AnalysisResult(
            raw_text="",
            backend="nim",
            error="NVIDIA_API_KEY is not set; cannot call NIM endpoint.",
        )

    payload = {
        "model": NIM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPT_TEMPLATE.format(source=source, extra=extra),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(image)},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(
            NIM_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        return AnalysisResult(raw_text=text, parsed=_safe_json(text), backend="nim")
    except Exception as exc:  # noqa: BLE001 — surface to UI
        return AnalysisResult(raw_text="", backend="nim", error=str(exc))


# ---------------------------------------------------------------------------
# Backend: local Hugging Face transformers
# ---------------------------------------------------------------------------

_LOCAL_PIPE = None  # cached across calls


def _load_local_pipeline():
    global _LOCAL_PIPE
    if _LOCAL_PIPE is not None:
        return _LOCAL_PIPE

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        DEFAULT_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _LOCAL_PIPE = (processor, model)
    return _LOCAL_PIPE


def _analyze_via_local(image: Image.Image, extra: str, source: str) -> AnalysisResult:
    try:
        processor, model = _load_local_pipeline()
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(raw_text="", backend="local", error=f"Model load failed: {exc}")

    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": USER_PROMPT_TEMPLATE.format(source=source, extra=extra),
                    },
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # strip echoed prompt if present
        text = text.split("assistant", 1)[-1].strip(" :\n")
        return AnalysisResult(raw_text=text, parsed=_safe_json(text), backend="local")
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(raw_text="", backend="local", error=str(exc))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def _safe_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from the model output."""
    if not text:
        return {}
    # trim ```json fences if the model added them
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def analyze(
    image: Image.Image,
    source: str = "uploaded file",
    table_preview: Optional[str] = None,
    backend: str = "auto",
) -> AnalysisResult:
    """Run the Ising Calibration VLM on a calibration image.

    backend:
      "auto"   — NIM if NVIDIA_API_KEY is set, else local HF
      "nim"    — force NIM
      "local"  — force local HF
    """
    if image is None:
        return AnalysisResult(raw_text="", backend=backend, error="No image provided.")

    extra = f"\n\nAccompanying table preview (markdown):\n{table_preview}" if table_preview else ""

    choice = backend
    if choice == "auto":
        choice = "nim" if os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") else "local"

    if choice == "nim":
        return _analyze_via_nim(image, extra, source)
    return _analyze_via_local(image, extra, source)
