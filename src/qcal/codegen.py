"""Turn analyzer output into a runnable CUDA-Q calibration script.

The generated script is deliberately small and self-contained so a user can
copy-paste it into their own CUDA-Q environment, tweak knobs, and run it.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any


_SCRIPT_TEMPLATE = '''"""
Auto-generated CUDA-Q calibration script from QCal Copilot.

Experiment: {experiment}
Qubit: {qubit_id}
Notes: {notes}
{decoder_header}
"""

import cudaq

# --- Calibration parameters suggested by QCal Copilot ------------------------
PARAMS = {params_repr}

# --- Kernel ------------------------------------------------------------------

@cudaq.kernel
def calibrated_circuit(theta: float):
    """Apply a calibrated single-qubit rotation + measurement.

    `theta` is derived from the recommended pulse amplitude/duration so that a
    nominal "pi-pulse" maps to a rotation of pi radians.
    """
    q = cudaq.qubit()
    rx(theta, q)
    mz(q)


def suggested_theta() -> float:
    """Derive a rotation angle from the recommended drive amplitude.

    This is a deliberately simple mapping for the MVP: a real calibration
    pipeline would fit a full Rabi curve. Values are clamped to [0, 2*pi].
    """
    import math

    amp = float(PARAMS.get("drive_amplitude", 0.5))
    duration_ns = float(PARAMS.get("pulse_duration_ns", 20.0))
    # Linear scaling: amp * duration / 20ns ≈ 0.5 -> pi/2, 1.0 -> pi.
    theta = math.pi * amp * (duration_ns / 20.0)
    return max(0.0, min(2.0 * math.pi, theta))


def run(shots: int = 2000) -> dict:
    """Run the calibrated circuit and return measurement statistics."""
    theta = suggested_theta()
    counts = cudaq.sample(calibrated_circuit, theta, shots_count=shots)
    stats = {{k: counts.count(k) for k in ("0", "1")}}
    total = sum(stats.values()) or 1
    fidelity_proxy = stats.get("1", 0) / total  # P(|1>) after nominal pi-pulse
    return {{
        "theta": theta,
        "counts": stats,
        "shots": shots,
        "p1": fidelity_proxy,
    }}


if __name__ == "__main__":
    result = run()
    print("theta:", result["theta"])
    print("counts:", result["counts"])
    print("P(|1>):", round(result["p1"], 4))
'''


def _params_repr(params: dict[str, Any]) -> str:
    """Format params dict as a readable Python literal."""
    if not params:
        params = {
            "drive_amplitude": 0.5,
            "drive_frequency_ghz": 5.0,
            "pulse_duration_ns": 20.0,
            "readout_threshold": 0.0,
        }
    dumped = json.dumps(params, indent=4, default=str)
    return dumped


def _decoder_header(decoder_info: dict[str, Any] | None) -> str:
    """Format a header block summarizing the decoder stage, if present."""
    if not decoder_info:
        return ""
    variant = decoder_info.get("variant", "n/a")
    model_id = decoder_info.get("model_id", "n/a")
    distance = decoder_info.get("distance", "n/a")
    rounds = decoder_info.get("rounds", "n/a")
    density_reduction = decoder_info.get("density_reduction", 0.0)
    ler_improvement = decoder_info.get("ler_improvement", 1.0)
    return (
        "\nError-correction decoder (applied upstream):\n"
        f"  variant:           {variant}\n"
        f"  model_id:          {model_id}\n"
        f"  surface code:      d={distance}, rounds={rounds}\n"
        f"  syndrome density:  {density_reduction*100:.1f}% reduction\n"
        f"  LER proxy:         {ler_improvement:.2f}x improvement"
    )


def generate_script(
    analysis: dict[str, Any],
    decoder_info: dict[str, Any] | None = None,
) -> str:
    """Build the CUDA-Q script text from analyzer (+ optional decoder) output."""
    experiment = analysis.get("experiment") or "unspecified"
    qubit_id = analysis.get("qubit_id") or "q0"
    notes = (analysis.get("notes") or "").replace('"""', "'''")
    params = analysis.get("recommended_parameters") or {}

    script = _SCRIPT_TEMPLATE.format(
        experiment=experiment,
        qubit_id=qubit_id,
        notes=textwrap.shorten(notes, width=240, placeholder="…"),
        params_repr=_params_repr(params),
        decoder_header=_decoder_header(decoder_info),
    )
    return script
