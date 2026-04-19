"""Auto-fitting for common calibration experiments.

Each fit returns a :class:`FitResult` with human-readable, unit-bearing
parameters plus a normalized quality metric in ``[0, 1]``. These get woven
into the VLM prompt so Ising can cross-check its vision analysis against
hard numerical fits — the single biggest quality lever for the whole
pipeline.

Every fit is defensive: bad data returns ``FitResult.failed(reason=...)``
instead of raising. Callers should always check ``result.ok``.

Public entrypoints
------------------
* :func:`fit_rabi`     — damped sine (amplitude/duration sweep)
* :func:`fit_ramsey`   — damped cosine with phase
* :func:`fit_t1`       — exponential decay
* :func:`fit_t2_echo`  — exponential decay (alias for T1 shape)
* :func:`autofit`      — dispatch by experiment type
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class FitResult:
    """Structured fit output.

    ``params`` uses named, unit-bearing keys (e.g. ``"tau_us"``,
    ``"freq_mhz"``) so downstream code and the VLM prompt never have to
    guess what each number means.
    """

    experiment: str
    model: str
    params: dict[str, float] = field(default_factory=dict)
    fit_quality: float = 0.0  # R^2 in [0, 1]; 1 is perfect
    residual_rms: float = 0.0
    n_points: int = 0
    ok: bool = True
    reason: Optional[str] = None

    @classmethod
    def failed(cls, experiment: str, model: str, reason: str) -> "FitResult":
        return cls(
            experiment=experiment, model=model, ok=False, reason=reason
        )

    def summary_text(self) -> str:
        """Compact one-line summary for injecting into the VLM prompt."""
        if not self.ok:
            return f"Fit ({self.experiment}, {self.model}) failed: {self.reason}"
        bits = ", ".join(f"{k}={self._fmt(v)}" for k, v in self.params.items())
        return (
            f"Fit ({self.experiment} → {self.model}): {bits} "
            f"| R²={self.fit_quality:.3f}, n={self.n_points}"
        )

    def markdown(self) -> str:
        if not self.ok:
            return f"**Fit failed** ({self.experiment}): {self.reason}"
        lines = [
            f"**Fit:** `{self.experiment}` → `{self.model}`",
            f"- R² = {self.fit_quality:.4f} (RMS residual {self.residual_rms:.4g})",
            f"- n = {self.n_points} points",
            "",
            "| param | value |",
            "|---|---|",
        ]
        for k, v in self.params.items():
            lines.append(f"| `{k}` | {self._fmt(v)} |")
        return "\n".join(lines)

    @staticmethod
    def _fmt(v: float) -> str:
        if not np.isfinite(v):
            return "n/a"
        av = abs(v)
        if av >= 1e4 or (0 < av < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4g}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _curve_fit(
    model: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    p0: list[float],
    bounds: Optional[tuple[list[float], list[float]]] = None,
    maxfev: int = 5000,
):
    """Thin wrapper around ``scipy.optimize.curve_fit`` with kwargs filled."""
    from scipy.optimize import curve_fit  # local import — scipy is optional-heavy

    kwargs: dict[str, Any] = {"p0": p0, "maxfev": maxfev}
    if bounds is not None:
        kwargs["bounds"] = bounds
    return curve_fit(model, x, y, **kwargs)


def _prep(x: Optional[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs to 1-D float arrays."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4:
        raise ValueError("need at least 4 points to fit")
    if x is None:
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y shape mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _dominant_frequency(x: np.ndarray, y: np.ndarray) -> float:
    """FFT-based seed for sinusoidal fits. Returns cycles per x-unit."""
    if x.size < 4:
        return 1.0
    dx = float(np.median(np.diff(x)))
    if dx <= 0:
        return 1.0
    y_centered = y - y.mean()
    fft = np.fft.rfft(y_centered)
    freqs = np.fft.rfftfreq(y.size, d=dx)
    if freqs.size <= 1:
        return 1.0
    idx = int(np.argmax(np.abs(fft[1:])) + 1)  # skip DC
    return float(freqs[idx])


# ---------------------------------------------------------------------------
# Rabi — damped sine
# ---------------------------------------------------------------------------

def _rabi_model(t, A, freq, tau, offset, phase):
    return A * np.exp(-t / tau) * np.sin(2 * np.pi * freq * t + phase) + offset


def fit_rabi(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    x_unit: str = "a.u.",
) -> FitResult:
    """Fit a Rabi oscillation: A·exp(-t/τ)·sin(2π·f·t + φ) + c.

    Returned params keyed for readability:
      ``amplitude``, ``freq`` (cycles per ``x_unit``), ``tau``
      (in ``x_unit``), ``offset``, ``phase_rad``.
    """
    try:
        x_, y_ = _prep(x, y)
    except ValueError as exc:
        return FitResult.failed("rabi", "damped_sine", str(exc))

    try:
        amp_seed = 0.5 * (y_.max() - y_.min())
        off_seed = 0.5 * (y_.max() + y_.min())
        freq_seed = _dominant_frequency(x_, y_)
        tau_seed = max(float(x_.max() - x_.min()), 1e-9)
        p0 = [amp_seed, freq_seed, tau_seed, off_seed, 0.0]
        bounds = (
            [0.0, 0.0, 1e-12, -np.inf, -2 * np.pi],
            [np.inf, np.inf, np.inf, np.inf, 2 * np.pi],
        )
        popt, _ = _curve_fit(_rabi_model, x_, y_, p0=p0, bounds=bounds)
        y_hat = _rabi_model(x_, *popt)
        return FitResult(
            experiment="rabi",
            model="damped_sine",
            params={
                "amplitude": float(popt[0]),
                f"freq_per_{x_unit}": float(popt[1]),
                f"tau_{x_unit}": float(popt[2]),
                "offset": float(popt[3]),
                "phase_rad": float(popt[4]),
            },
            fit_quality=_r_squared(y_, y_hat),
            residual_rms=float(np.sqrt(np.mean((y_ - y_hat) ** 2))),
            n_points=int(y_.size),
        )
    except Exception as exc:  # noqa: BLE001
        return FitResult.failed("rabi", "damped_sine", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Ramsey — damped cosine with phase
# ---------------------------------------------------------------------------

def _ramsey_model(t, A, freq, tau, offset, phase):
    return A * np.exp(-t / tau) * np.cos(2 * np.pi * freq * t + phase) + offset


def fit_ramsey(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    x_unit: str = "us",
) -> FitResult:
    """Fit a Ramsey fringe: A·exp(-t/τ)·cos(2π·δf·t + φ) + c.

    ``freq`` here is the detuning between drive and qubit frequency in
    cycles per ``x_unit``.
    """
    try:
        x_, y_ = _prep(x, y)
    except ValueError as exc:
        return FitResult.failed("ramsey", "damped_cosine", str(exc))

    try:
        amp_seed = 0.5 * (y_.max() - y_.min())
        off_seed = 0.5 * (y_.max() + y_.min())
        freq_seed = max(_dominant_frequency(x_, y_), 1e-6)
        tau_seed = max(float(x_.max() - x_.min()), 1e-9)
        p0 = [amp_seed, freq_seed, tau_seed, off_seed, 0.0]
        bounds = (
            [0.0, 0.0, 1e-12, -np.inf, -2 * np.pi],
            [np.inf, np.inf, np.inf, np.inf, 2 * np.pi],
        )
        popt, _ = _curve_fit(_ramsey_model, x_, y_, p0=p0, bounds=bounds)
        y_hat = _ramsey_model(x_, *popt)
        return FitResult(
            experiment="ramsey",
            model="damped_cosine",
            params={
                "amplitude": float(popt[0]),
                f"detuning_per_{x_unit}": float(popt[1]),
                f"t2star_{x_unit}": float(popt[2]),
                "offset": float(popt[3]),
                "phase_rad": float(popt[4]),
            },
            fit_quality=_r_squared(y_, y_hat),
            residual_rms=float(np.sqrt(np.mean((y_ - y_hat) ** 2))),
            n_points=int(y_.size),
        )
    except Exception as exc:  # noqa: BLE001
        return FitResult.failed("ramsey", "damped_cosine", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# T1 / T2 — exponential decay
# ---------------------------------------------------------------------------

def _exp_decay(t, A, tau, offset):
    return A * np.exp(-t / tau) + offset


def fit_t1(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    x_unit: str = "us",
) -> FitResult:
    """Fit T1 relaxation: A·exp(-t/τ) + c."""
    try:
        x_, y_ = _prep(x, y)
    except ValueError as exc:
        return FitResult.failed("t1", "exp_decay", str(exc))
    return _fit_exp_decay(x_, y_, experiment="t1", x_unit=x_unit)


def fit_t2_echo(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    x_unit: str = "us",
) -> FitResult:
    """Fit T2 (echo / Hahn) decay: A·exp(-t/τ) + c."""
    try:
        x_, y_ = _prep(x, y)
    except ValueError as exc:
        return FitResult.failed("t2_echo", "exp_decay", str(exc))
    return _fit_exp_decay(x_, y_, experiment="t2_echo", x_unit=x_unit)


def _fit_exp_decay(
    x: np.ndarray, y: np.ndarray, experiment: str, x_unit: str
) -> FitResult:
    try:
        sign = 1.0 if y[0] > y[-1] else -1.0
        amp_seed = sign * (y[0] - y[-1])
        off_seed = float(y[-1])
        tau_seed = max(float(x.max() - x.min()) / 3.0, 1e-9)
        p0 = [amp_seed, tau_seed, off_seed]
        # tau must be positive; offset unconstrained; amplitude signed
        bounds = (
            [-np.inf, 1e-12, -np.inf],
            [np.inf, np.inf, np.inf],
        )
        popt, _ = _curve_fit(_exp_decay, x, y, p0=p0, bounds=bounds)
        y_hat = _exp_decay(x, *popt)
        return FitResult(
            experiment=experiment,
            model="exp_decay",
            params={
                "amplitude": float(popt[0]),
                f"tau_{x_unit}": float(popt[1]),
                "offset": float(popt[2]),
            },
            fit_quality=_r_squared(y, y_hat),
            residual_rms=float(np.sqrt(np.mean((y - y_hat) ** 2))),
            n_points=int(y.size),
        )
    except Exception as exc:  # noqa: BLE001
        return FitResult.failed(experiment, "exp_decay", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FIT_DISPATCH: dict[str, Callable[..., FitResult]] = {
    "rabi": fit_rabi,
    "ramsey": fit_ramsey,
    "t1": fit_t1,
    "t2": fit_t2_echo,
    "t2_echo": fit_t2_echo,
}


def autofit(
    experiment_type: str,
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Optional[FitResult]:
    """Run the fit matching ``experiment_type``. Returns ``None`` if there
    is no fitter registered for that experiment (e.g. 2D chevron data)."""
    fn = _FIT_DISPATCH.get(experiment_type.lower())
    if fn is None:
        return None
    return fn(y=y, x=x, **kwargs)


def supported_experiments() -> list[str]:
    """List experiment types that ``autofit`` can handle."""
    return sorted(_FIT_DISPATCH.keys())
