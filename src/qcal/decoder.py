"""Ising 3D CNN surface-code pre-decoder stage.

Runs one of the NVIDIA Ising Decoder 3D CNN pre-decoders on a noisy syndrome
volume to *sparsify* it, then hands the cleaned volume to PyMatching for final
MWPM correction. This is a drop-in stage that sits after the calibration
analyzer — it never mutates calibration state.

Two model variants are supported:

  * "fast"     — nvidia/Ising-Decoder-SurfaceCode-1-Fast     (~912k params)
  * "accurate" — nvidia/Ising-Decoder-SurfaceCode-1-Accurate (~1.79M params)

When the Hugging Face weights can't be loaded (no access, no internet, no
GPU), we fall back to a deterministic "neighbor-support" denoiser so the demo
still runs and the architecture is exercised end-to-end.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VARIANT_FAST = "fast"
VARIANT_ACCURATE = "accurate"
VARIANTS = (VARIANT_FAST, VARIANT_ACCURATE)

MODEL_IDS: dict[str, str] = {
    VARIANT_FAST: os.getenv(
        "QCAL_DECODER_FAST_ID", "nvidia/Ising-Decoder-SurfaceCode-1-Fast"
    ),
    VARIANT_ACCURATE: os.getenv(
        "QCAL_DECODER_ACCURATE_ID", "nvidia/Ising-Decoder-SurfaceCode-1-Accurate"
    ),
}
APPROX_PARAMS: dict[str, int] = {
    VARIANT_FAST: 912_000,
    VARIANT_ACCURATE: 1_790_000,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DecoderResult:
    """Structured output of a full decoder run (CNN + optional MWPM)."""

    variant: str
    distance: int
    rounds: int
    n_shots: int
    error_rate: float

    density_before: float = 0.0
    density_after: float = 0.0

    inference_ms: float = 0.0
    mwpm_ms_before: Optional[float] = None
    mwpm_ms_after: Optional[float] = None

    ler_proxy_before: float = 0.0
    ler_proxy_after: float = 0.0

    raw_example: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    denoised_example: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    backend_note: str = ""
    model_id: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def density_reduction(self) -> float:
        if self.density_before <= 0:
            return 0.0
        return 1.0 - (self.density_after / self.density_before)

    @property
    def ler_improvement(self) -> float:
        if self.ler_proxy_after <= 0:
            return float("inf") if self.ler_proxy_before > 0 else 1.0
        return self.ler_proxy_before / self.ler_proxy_after

    @property
    def mwpm_speedup(self) -> Optional[float]:
        if not self.mwpm_ms_before or not self.mwpm_ms_after:
            return None
        return self.mwpm_ms_before / max(self.mwpm_ms_after, 1e-6)

    def markdown(self) -> str:
        if self.error:
            return f"**Decoder error:** {self.error}"
        approx = APPROX_PARAMS.get(self.variant, 0)
        lines = [
            f"**Variant:** `{self.variant}` (~{approx/1e6:.2f}M params)",
            f"**Model id:** `{self.model_id}`",
            f"**Surface code:** distance = {self.distance}, rounds = {self.rounds}",
            f"**Shots:** {self.n_shots}",
            f"**Physical error rate (p):** {self.error_rate:.4f}",
            "",
            "**Syndrome density**",
            f"- before CNN: {self.density_before:.4f}",
            f"- after  CNN: {self.density_after:.4f}",
            f"- reduction: **{self.density_reduction*100:.1f}%**",
            "",
            "**Timing**",
            f"- CNN inference: {self.inference_ms:.2f} ms total"
            f" ({self.inference_ms/max(self.n_shots,1):.3f} ms/shot)",
        ]
        if self.mwpm_ms_before is not None:
            lines += [
                f"- MWPM on raw syndromes: {self.mwpm_ms_before:.2f} ms",
                f"- MWPM on denoised syndromes: {self.mwpm_ms_after:.2f} ms",
            ]
            speedup = self.mwpm_speedup
            if speedup:
                lines.append(f"- MWPM speedup: **{speedup:.2f}×**")
        else:
            lines.append("- MWPM stage: _skipped (pymatching not installed)_")
        lines += [
            "",
            "**Logical-error-rate proxy** _(syndrome-weight threshold — demo only)_",
            f"- before CNN: {self.ler_proxy_before:.4f}",
            f"- after  CNN: {self.ler_proxy_after:.4f}",
            f"- improvement: **{self.ler_improvement:.2f}×**",
            "",
            f"_{self.backend_note}_",
        ]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:  # Jupyter renders this directly
        return self.markdown()


# ---------------------------------------------------------------------------
# Synthetic syndrome generation
# ---------------------------------------------------------------------------

def generate_syndromes(
    distance: int,
    rounds: int,
    error_rate: float,
    n_shots: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Generate synthetic syndrome volumes.

    Shape: (n_shots, distance, distance, rounds) of uint8.

    We sample each space-time cell independently from Bernoulli(p), then add
    a small amount of spatial correlation (a 2-cell "chain" event per shot)
    so the denoiser has structure to exploit. This mimics what a detector
    error model produces for a depolarizing noise channel at low p.
    """
    if distance < 3:
        raise ValueError("distance must be >= 3 for a meaningful demo")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not 0.0 <= error_rate <= 0.5:
        raise ValueError("error_rate must be in [0, 0.5]")

    rng = np.random.default_rng(seed)
    shape = (n_shots, distance, distance, rounds)
    volumes = (rng.random(shape) < error_rate).astype(np.uint8)

    # Inject correlated 2-cell chains — realistic time-like errors.
    n_chains = max(1, int(n_shots * rounds * error_rate * 0.5))
    for _ in range(n_chains):
        s = rng.integers(0, n_shots)
        x = rng.integers(0, distance)
        y = rng.integers(0, distance)
        t = rng.integers(0, max(rounds - 1, 1))
        volumes[s, x, y, t] = 1
        if rounds > 1:
            volumes[s, x, y, t + 1] = 1
    return volumes


# ---------------------------------------------------------------------------
# Model loading (HF) and fallback
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}


def _try_load_hf_decoder(variant: str) -> tuple[Any, str]:
    """Attempt to load an Ising decoder via Hugging Face.

    Returns (callable_or_None, backend_note).
    """
    model_id = MODEL_IDS[variant]
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id], f"Loaded `{model_id}` from cache."

    try:
        import torch
        from transformers import AutoModel
    except Exception as exc:  # noqa: BLE001
        return None, f"PyTorch/transformers unavailable ({exc}); using fallback denoiser."

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        model.eval()
        _MODEL_CACHE[model_id] = (model, device)
        return (model, device), f"Loaded `{model_id}` on {device}."
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            f"Could not load `{model_id}` ({type(exc).__name__}: {exc}). "
            "Falling back to built-in neighbor-support denoiser.",
        )


def _fallback_denoise(volumes: np.ndarray) -> np.ndarray:
    """Drop any detection event with no 3D neighbor — a cheap sparsifier.

    Isolated single-cell detections are almost always measurement noise at the
    p-regimes the Ising CNNs target; neighborhood support indicates a real
    error chain. This is a useful first-order stand-in when the HF weights
    aren't available.
    """
    v = volumes.astype(np.int16)
    neigh = np.zeros_like(v)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dt in (-1, 0, 1):
                if dx == dy == dt == 0:
                    continue
                shifted = np.roll(v, shift=(dx, dy, dt), axis=(1, 2, 3))
                # zero out the wrap-around planes so boundaries don't leak
                if dx == 1:
                    shifted[:, 0, :, :] = 0
                elif dx == -1:
                    shifted[:, -1, :, :] = 0
                if dy == 1:
                    shifted[:, :, 0, :] = 0
                elif dy == -1:
                    shifted[:, :, -1, :] = 0
                if dt == 1:
                    shifted[:, :, :, 0] = 0
                elif dt == -1:
                    shifted[:, :, :, -1] = 0
                neigh += shifted
    return (v * (neigh >= 1)).astype(np.uint8)


def _run_hf_inference(loaded: tuple[Any, str], volumes: np.ndarray) -> np.ndarray:
    """Run the Ising decoder CNN on a stack of syndrome volumes.

    We try the common Vision-style entrypoints; model authors expose different
    call conventions, so we guard each attempt. On failure, raise so the
    caller can fall back.
    """
    import torch

    model, device = loaded
    x = torch.from_numpy(volumes).float().unsqueeze(1).to(device)  # (N, 1, D, D, T)
    with torch.no_grad():
        try:
            out = model(x)
        except TypeError:
            out = model(pixel_values=x)

        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, (tuple, list)):
            out = out[0]

        # The pre-decoder produces a per-cell "keep" probability; threshold it.
        out = torch.sigmoid(out) if out.dtype.is_floating_point else out.float()
        out = out.squeeze(1)  # (N, D, D, T)
        mask = (out >= 0.5).to(torch.uint8)
        # The CNN is a sparsifier, not a generator — never turn on new bits.
        mask = mask * torch.from_numpy(volumes).to(mask.device)
    return mask.cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Optional MWPM stage
# ---------------------------------------------------------------------------

def _build_demo_matching(distance: int, rounds: int):
    """Build a small PyMatching graph over the space-time volume.

    NOTE: this is **not** a real surface-code detector error model. We connect
    each cell to its 6-neighborhood so PyMatching has something MWPM-able;
    real deployments should feed a `stim`-generated DEM.
    """
    import pymatching

    m = pymatching.Matching()

    def node(x: int, y: int, t: int) -> int:
        return t * distance * distance + y * distance + x

    bit_id = 0
    for t in range(rounds):
        for y in range(distance):
            for x in range(distance):
                if x + 1 < distance:
                    m.add_edge(node(x, y, t), node(x + 1, y, t), fault_ids={bit_id}, weight=1.0)
                    bit_id += 1
                if y + 1 < distance:
                    m.add_edge(node(x, y, t), node(x, y + 1, t), fault_ids={bit_id}, weight=1.0)
                    bit_id += 1
                if t + 1 < rounds:
                    m.add_edge(node(x, y, t), node(x, y, t + 1), fault_ids={bit_id}, weight=1.5)
                    bit_id += 1
        # Boundary edges along the x=0 / x=d-1 columns let isolated detectors terminate.
        for y in range(distance):
            m.add_boundary_edge(node(0, y, t), fault_ids=set(), weight=2.0)
            m.add_boundary_edge(node(distance - 1, y, t), fault_ids=set(), weight=2.0)
    return m


def _time_mwpm(volumes: np.ndarray, matching) -> float:
    """Decode every shot and return total wall time in ms."""
    flat = volumes.reshape(volumes.shape[0], -1)
    t0 = time.perf_counter()
    for row in flat:
        matching.decode(row)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# LER proxy
# ---------------------------------------------------------------------------

def _ler_proxy(volumes: np.ndarray, distance: int) -> float:
    """Fraction of shots with syndrome weight above a distance-scaled threshold.

    This isn't a real logical error rate — it's a first-order proxy used to
    show the relative improvement from denoising in the UI.
    """
    if volumes.size == 0:
        return 0.0
    per_shot = volumes.reshape(volumes.shape[0], -1).sum(axis=1)
    threshold = max(1, distance)
    return float((per_shot > threshold).mean())


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_decoder(
    variant: str = VARIANT_FAST,
    distance: int = 5,
    rounds: int = 5,
    error_rate: float = 0.005,
    n_shots: int = 128,
    seed: Optional[int] = 42,
) -> DecoderResult:
    """Full pipeline: synthesize syndromes → CNN sparsify → (MWPM) → metrics."""
    if variant not in VARIANTS:
        return DecoderResult(
            variant=variant,
            distance=distance,
            rounds=rounds,
            n_shots=n_shots,
            error_rate=error_rate,
            error=f"Unknown variant '{variant}'. Choose one of {VARIANTS}.",
        )

    try:
        raw = generate_syndromes(distance, rounds, error_rate, n_shots, seed=seed)
    except Exception as exc:  # noqa: BLE001
        return DecoderResult(
            variant=variant, distance=distance, rounds=rounds,
            n_shots=n_shots, error_rate=error_rate,
            error=f"Syndrome generation failed: {exc}",
        )

    loaded, backend_note = _try_load_hf_decoder(variant)

    t0 = time.perf_counter()
    if loaded is not None:
        try:
            denoised = _run_hf_inference(loaded, raw)
        except Exception as exc:  # noqa: BLE001
            backend_note = (
                f"HF model load succeeded but inference failed ({exc}); "
                "falling back to neighbor-support denoiser."
            )
            denoised = _fallback_denoise(raw)
    else:
        denoised = _fallback_denoise(raw)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    density_before = float(raw.mean())
    density_after = float(denoised.mean())

    # Optional MWPM stage
    mwpm_before = mwpm_after = None
    try:
        import pymatching  # noqa: F401

        matching = _build_demo_matching(distance, rounds)
        mwpm_before = _time_mwpm(raw, matching)
        mwpm_after = _time_mwpm(denoised, matching)
    except ImportError:
        backend_note += " (pymatching not installed — MWPM skipped)"
    except Exception as exc:  # noqa: BLE001
        backend_note += f" (MWPM stage failed: {exc})"

    mid_t = rounds // 2
    result = DecoderResult(
        variant=variant,
        distance=distance,
        rounds=rounds,
        n_shots=n_shots,
        error_rate=error_rate,
        density_before=density_before,
        density_after=density_after,
        inference_ms=inference_ms,
        mwpm_ms_before=mwpm_before,
        mwpm_ms_after=mwpm_after,
        ler_proxy_before=_ler_proxy(raw, distance),
        ler_proxy_after=_ler_proxy(denoised, distance),
        raw_example=raw[0, :, :, mid_t].copy(),
        denoised_example=denoised[0, :, :, mid_t].copy(),
        backend_note=backend_note,
        model_id=MODEL_IDS[variant],
    )
    return result


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def plot_comparison(result: DecoderResult):
    """Return a matplotlib Figure comparing raw vs denoised syndrome slices."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(result.raw_example, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Raw syndrome\n(shot 0, t={result.rounds // 2})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(
        result.denoised_example, cmap="Greens", vmin=0, vmax=1, interpolation="nearest"
    )
    axes[1].set_title(f"After {result.variant} CNN\n(sparsified)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    categories = ["density", "LER proxy"]
    before = [result.density_before, result.ler_proxy_before]
    after = [result.density_after, result.ler_proxy_after]
    x = np.arange(len(categories))
    width = 0.38
    axes[2].bar(x - width / 2, before, width, label="raw", color="#d62728")
    axes[2].bar(x + width / 2, after, width, label="denoised", color="#2ca02c")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].set_title("Before vs After")
    axes[2].legend()
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: derive a sensible error rate from calibration analysis
# ---------------------------------------------------------------------------

def suggest_error_rate(analysis: Optional[dict]) -> float:
    """Map a calibration analysis dict to a plausible physical error rate.

    Rough heuristic: start at 1e-3 (good transmon) and inflate based on how
    far the recommended drive amplitude deviates from a nominal 1.0.
    """
    if not analysis:
        return 0.005
    params = analysis.get("recommended_parameters") or {}
    try:
        amp = float(params.get("drive_amplitude", 1.0))
    except (TypeError, ValueError):
        amp = 1.0
    deviation = abs(1.0 - amp)
    return float(min(0.03, 0.001 + 0.01 * deviation))
