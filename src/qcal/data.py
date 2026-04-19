"""Input preprocessing for QCal Copilot.

Normalizes user inputs — image, CSV/TSV, or raw numpy arrays from control
hardware — into a :class:`CalibrationPayload` the analyzer can send to the
vision-language model. For numpy input we also auto-fit standard calibration
models (Rabi, Ramsey, T1, T2-echo) and attach the fit parameters as text
context; Ising cross-references those numbers against the rendered plot,
which is the single biggest quality lever in the pipeline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from .fit import FitResult, autofit


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SUPPORTED_TABLE_EXTS = {".csv", ".tsv"}
SUPPORTED_ARRAY_EXTS = {".npy", ".npz"}

# Experiment types this module knows how to render.
EXPERIMENT_RENDERERS: dict[str, str] = {
    "rabi": "line",
    "ramsey": "line",
    "t1": "line",
    "t2": "line",
    "t2_echo": "line",
    "resonator_spec": "line",
    "iq_trace": "line",
    "rabi_chevron": "heatmap",
    "readout_iq": "scatter",
    "unknown": "line",
}


ArrayLike = Union[np.ndarray, "list[float]", "tuple[float, ...]"]


@dataclass
class CalibrationPayload:
    """Container holding normalized calibration data for downstream analysis."""

    image: Optional[Image.Image] = None
    table: Optional[pd.DataFrame] = None
    source_name: str = ""
    kind: str = "unknown"  # "image" | "csv" | "array" | "unknown"
    experiment_type: Optional[str] = None
    fit: Optional[FitResult] = None
    metadata: dict = field(default_factory=dict)
    numeric_summary: str = ""

    def summary(self) -> str:
        if self.kind == "image" and self.image is not None:
            w, h = self.image.size
            return f"Image `{self.source_name}` ({w}x{h}, mode={self.image.mode})"
        if self.kind == "csv" and self.table is not None:
            rows, cols = self.table.shape
            col_list = ", ".join(map(str, self.table.columns[:8]))
            more = " …" if self.table.shape[1] > 8 else ""
            return (
                f"Table `{self.source_name}` ({rows} rows × {cols} cols). "
                f"Columns: {col_list}{more}"
            )
        if self.kind == "array":
            exp = self.experiment_type or "unknown"
            return f"Array `{self.source_name}` (experiment={exp})"
        return "No data provided."

    def table_preview_markdown(self, max_rows: int = 10) -> str:
        if self.table is None:
            return ""
        return self.table.head(max_rows).to_markdown(index=False)

    def prompt_context(self) -> Optional[str]:
        """Text appended to the VLM user prompt, if any."""
        chunks: list[str] = []
        if self.numeric_summary:
            chunks.append(self.numeric_summary)
        if self.fit is not None and self.fit.ok:
            chunks.append(self.fit.summary_text())
        if self.metadata:
            chunks.append(
                "Metadata: "
                + ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            )
        if self.table is not None:
            chunks.append(
                "Table preview (markdown):\n" + self.table_preview_markdown()
            )
        return "\n".join(chunks) if chunks else None


# ---------------------------------------------------------------------------
# Plot rendering helpers
# ---------------------------------------------------------------------------

def _render_table_as_image(df: pd.DataFrame) -> Image.Image:
    """Render a small preview image of a table so the VLM can still see it."""
    import matplotlib.pyplot as plt

    preview = df.head(25)
    fig, ax = plt.subplots(
        figsize=(min(2 + 1.1 * len(preview.columns), 14), min(1 + 0.3 * len(preview), 10))
    )
    ax.axis("off")
    tbl = ax.table(
        cellText=preview.round(4).astype(str).values,
        colLabels=[str(c) for c in preview.columns],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _render_csv_for_vlm(
    df: pd.DataFrame,
    *,
    experiment_type: str = "unknown",
    title: Optional[str] = None,
) -> Image.Image:
    """Render a user-uploaded CSV as whatever image the VLM can actually analyze.

    The Ising Calibration VLM is trained on *plots* (Rabi traces, T1 decays,
    IQ scatter, etc.), not on screenshots of numeric tables — feeding it a
    table grid drops recognition confidence to ~0.2 and produces the "no clear
    oscillations" failure mode. So for common CSV shapes we render a proper
    line or scatter plot; only truly arbitrary tables fall back to the grid.
    """
    import matplotlib.pyplot as plt  # noqa: F401 — keeps mpl import local

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Two numeric columns: classic sweep (x, y). Covers Rabi / Ramsey / T1 / T2
    # / resonator sweeps out of the box.
    if len(numeric_cols) == 2:
        x_col, y_col = numeric_cols
        return _render_line(
            df[y_col].to_numpy(),
            df[x_col].to_numpy(),
            experiment=experiment_type,
            x_label=str(x_col),
            y_label=str(y_col),
            title=title,
            fit=None,
        )

    # Readout IQ: two columns named like I/Q (any case, any order).
    lower = {c.lower(): c for c in df.columns}
    if "i" in lower and "q" in lower:
        iq = df[[lower["i"], lower["q"]]].to_numpy()
        return _render_scatter(iq, title=title)

    # Single numeric column: plot vs row index.
    if len(numeric_cols) == 1:
        y_col = numeric_cols[0]
        return _render_line(
            df[y_col].to_numpy(),
            None,
            experiment=experiment_type,
            x_label="sample index",
            y_label=str(y_col),
            title=title,
            fit=None,
        )

    # Fall back to the table screenshot for wide/categorical tables the VLM
    # probably can't interpret anyway.
    return _render_table_as_image(df)


def _fig_to_pil(fig) -> Image.Image:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _render_line(
    y: np.ndarray,
    x: Optional[np.ndarray],
    *,
    experiment: str,
    x_label: str,
    y_label: str,
    title: Optional[str],
    fit: Optional[FitResult],
) -> Image.Image:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = x if x is not None else np.arange(y.size)
    ax.plot(xs, y, marker="o", markersize=3, linewidth=1.0, color="#1f77b4", label="data")

    # Overlay the fit curve when available.
    if fit is not None and fit.ok:
        try:
            x_dense = np.linspace(float(xs.min()), float(xs.max()), 400)
            y_fit = _evaluate_fit(fit, x_dense)
            if y_fit is not None:
                ax.plot(x_dense, y_fit, color="#d62728", linewidth=1.5, label="fit")
        except Exception:  # noqa: BLE001 — fit overlay is best-effort
            pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"{experiment} sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    return _fig_to_pil(fig)


def _render_heatmap(
    z: np.ndarray,
    *,
    experiment: str,
    x_label: str,
    y_label: str,
    title: Optional[str],
    extent: Optional[tuple[float, float, float, float]] = None,
) -> Image.Image:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="signal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"{experiment}")
    return _fig_to_pil(fig)


def _render_scatter(
    iq: np.ndarray,
    *,
    title: Optional[str],
) -> Image.Image:
    """Scatter + marginal histograms for readout IQ shots (shape (N, 2))."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(iq[:, 0], iq[:, 1], s=4, alpha=0.5, color="#1f77b4")
    ax.set_xlabel("I (a.u.)")
    ax.set_ylabel("Q (a.u.)")
    ax.set_title(title or "Readout IQ histogram")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    return _fig_to_pil(fig)


def _evaluate_fit(fit: FitResult, x: np.ndarray) -> Optional[np.ndarray]:
    """Recreate the fitted curve from stored parameters for overlay plotting."""
    p = fit.params
    if fit.model == "damped_sine":
        A = p["amplitude"]
        freq = next((v for k, v in p.items() if k.startswith("freq_per_")), 0.0)
        tau = next((v for k, v in p.items() if k.startswith("tau_")), 1.0)
        return (
            A * np.exp(-x / tau) * np.sin(2 * np.pi * freq * x + p["phase_rad"])
            + p["offset"]
        )
    if fit.model == "damped_cosine":
        A = p["amplitude"]
        freq = next((v for k, v in p.items() if k.startswith("detuning_per_")), 0.0)
        tau = next((v for k, v in p.items() if k.startswith("t2star_")), 1.0)
        return (
            A * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + p["phase_rad"])
            + p["offset"]
        )
    if fit.model == "exp_decay":
        A = p["amplitude"]
        tau = next((v for k, v in p.items() if k.startswith("tau_")), 1.0)
        return A * np.exp(-x / tau) + p["offset"]
    return None


# ---------------------------------------------------------------------------
# Numeric summary
# ---------------------------------------------------------------------------

def _numeric_summary(arr: np.ndarray, experiment: str) -> str:
    arr = np.asarray(arr)
    shape = arr.shape
    finite = arr[np.isfinite(arr)] if arr.dtype.kind in "fc" else arr.ravel()
    if finite.size == 0:
        return f"Numeric summary: shape={shape}, no finite values"
    parts = [
        f"Numeric summary for experiment `{experiment}`:",
        f"- shape: {shape}, dtype: {arr.dtype}",
        f"- range: [{float(finite.min()):.4g}, {float(finite.max()):.4g}]",
        f"- mean: {float(finite.mean()):.4g}, std: {float(finite.std()):.4g}",
    ]
    if arr.ndim == 1:
        parts.append(
            f"- argmax index: {int(np.nanargmax(arr))}, argmin index: {int(np.nanargmin(arr))}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public: from_array
# ---------------------------------------------------------------------------

def from_array(
    array: ArrayLike,
    experiment_type: str = "unknown",
    *,
    x: Optional[ArrayLike] = None,
    x_unit: str = "us",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    fit: bool = True,
    source_name: str = "array",
) -> CalibrationPayload:
    """Build a :class:`CalibrationPayload` from raw numpy data.

    Parameters
    ----------
    array
        The measurement. Shape depends on ``experiment_type``:
          * 1-D sweep (``rabi``, ``ramsey``, ``t1``, ``t2``, ``t2_echo``,
            ``resonator_spec``, ``iq_trace``)
          * 2-D heatmap (``rabi_chevron``) — shape ``(len(y_axis), len(x_axis))``
          * 2-D scatter (``readout_iq``) — shape ``(N, 2)`` for I/Q shots
    experiment_type
        One of :data:`EXPERIMENT_RENDERERS`; drives plot shape and fit model.
    x
        Independent variable for 1-D sweeps (times, amplitudes, frequencies).
    x_unit
        Unit label used in fit parameter keys and axis labels. Ignored for 2-D.
    x_label, y_label, title
        Optional axis overrides; sensible defaults per ``experiment_type``.
    metadata
        Free-form key/value pairs (qubit id, temperature, run id…) — appended
        verbatim to the VLM prompt.
    fit
        If ``True``, auto-fit the curve for supported 1-D experiments and
        attach :class:`~qcal.fit.FitResult` to the payload. Set ``False`` for
        air-gapped installs without ``scipy`` or when the data isn't fittable.
    source_name
        Display name threaded into the summary and the VLM prompt.

    Returns
    -------
    CalibrationPayload
        Ready to pass to :func:`qcal.analyzer.analyze`.
    """
    arr = np.asarray(array)
    md = dict(metadata or {})
    kind_hint = EXPERIMENT_RENDERERS.get(experiment_type, "line")

    # ------ render + optional fit ------
    fit_result: Optional[FitResult] = None
    if kind_hint == "line":
        if arr.ndim != 1:
            raise ValueError(
                f"experiment_type '{experiment_type}' expects a 1-D array, "
                f"got shape {arr.shape}"
            )
        x_arr = np.asarray(x, dtype=float) if x is not None else None
        if fit:
            fit_result = autofit(experiment_type, y=arr, x=x_arr, x_unit=x_unit)
        img = _render_line(
            arr,
            x_arr,
            experiment=experiment_type,
            x_label=x_label or f"sweep ({x_unit})",
            y_label=y_label or "signal (a.u.)",
            title=title,
            fit=fit_result,
        )
    elif kind_hint == "heatmap":
        if arr.ndim != 2:
            raise ValueError(
                f"experiment_type '{experiment_type}' expects a 2-D array, "
                f"got shape {arr.shape}"
            )
        img = _render_heatmap(
            arr,
            experiment=experiment_type,
            x_label=x_label or "drive amp (a.u.)",
            y_label=y_label or "drive freq (MHz)",
            title=title,
        )
    elif kind_hint == "scatter":
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"experiment_type '{experiment_type}' expects shape (N, 2), "
                f"got {arr.shape}"
            )
        img = _render_scatter(arr, title=title)
    else:
        raise ValueError(f"Unknown experiment_type '{experiment_type}'")

    return CalibrationPayload(
        image=img,
        source_name=source_name,
        kind="array",
        experiment_type=experiment_type,
        fit=fit_result,
        metadata=md,
        numeric_summary=_numeric_summary(arr, experiment_type),
    )


def from_npy(
    path: str | Path,
    experiment_type: str = "unknown",
    *,
    x_path: Optional[str | Path] = None,
    **kwargs,
) -> CalibrationPayload:
    """Load a ``.npy`` file and wrap it via :func:`from_array`.

    For ``.npz``, use :func:`from_npz` — it understands multi-array archives.
    """
    p = Path(path)
    arr = np.load(p, allow_pickle=False)
    x_arr = np.load(x_path, allow_pickle=False) if x_path else None
    return from_array(
        arr,
        experiment_type=experiment_type,
        x=x_arr,
        source_name=p.name,
        **kwargs,
    )


def from_npz(
    path: str | Path,
    experiment_type: str = "unknown",
    *,
    y_key: str = "y",
    x_key: Optional[str] = "x",
    **kwargs,
) -> CalibrationPayload:
    """Load a ``.npz`` archive. Expects ``y`` (required) and ``x`` (optional)."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        if y_key not in data:
            raise KeyError(
                f"{p.name}: missing required key '{y_key}'. Keys present: {list(data.keys())}"
            )
        y = np.asarray(data[y_key])
        x = np.asarray(data[x_key]) if x_key and x_key in data else None
    return from_array(
        y,
        experiment_type=experiment_type,
        x=x,
        source_name=p.name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# File loader (existing callers)
# ---------------------------------------------------------------------------

def load_payload(
    file_path: str | Path,
    experiment_type: str = "unknown",
    fit: bool = True,
) -> CalibrationPayload:
    """Load an uploaded file into a :class:`CalibrationPayload`.

    Auto-dispatches by extension across image, CSV/TSV, and ``.npy``/``.npz``.
    """
    if file_path is None:
        return CalibrationPayload()

    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    if ext in SUPPORTED_IMAGE_EXTS:
        img = Image.open(path).convert("RGB")
        return CalibrationPayload(image=img, source_name=name, kind="image")

    if ext in SUPPORTED_TABLE_EXTS:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        img = _render_csv_for_vlm(df, experiment_type=experiment_type, title=name)
        return CalibrationPayload(image=img, table=df, source_name=name, kind="csv")

    if ext == ".npy":
        return from_npy(path, experiment_type=experiment_type, fit=fit)
    if ext == ".npz":
        return from_npz(path, experiment_type=experiment_type, fit=fit)

    raise ValueError(
        f"Unsupported file type '{ext}'. Accepted: "
        f"{sorted(SUPPORTED_IMAGE_EXTS | SUPPORTED_TABLE_EXTS | SUPPORTED_ARRAY_EXTS)}"
    )
