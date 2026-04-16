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


def generate_script(analysis: dict[str, Any]) -> str:
    """Build the CUDA-Q script text from analyzer output."""
    experiment = analysis.get("experiment") or "unspecified"
    qubit_id = analysis.get("qubit_id") or "q0"
    notes = (analysis.get("notes") or "").replace('"""', "'''")
    params = analysis.get("recommended_parameters") or {}

    script = _SCRIPT_TEMPLATE.format(
        experiment=experiment,
        qubit_id=qubit_id,
        notes=textwrap.shorten(notes, width=240, placeholder="…"),
        params_repr=_params_repr(params),
    )
    return script
