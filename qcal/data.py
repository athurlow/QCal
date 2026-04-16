"""Input preprocessing for QCal Copilot.

Normalizes user uploads (image file or CSV) into a structured payload the
analyzer module can send to the vision-language model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SUPPORTED_TABLE_EXTS = {".csv", ".tsv"}


@dataclass
class CalibrationPayload:
    """Container holding normalized calibration data for downstream analysis."""

    image: Optional[Image.Image] = None
    table: Optional[pd.DataFrame] = None
    source_name: str = ""
    kind: str = "unknown"  # "image" | "csv" | "unknown"

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
        return "No data provided."

    def table_preview_markdown(self, max_rows: int = 10) -> str:
        if self.table is None:
            return ""
        return self.table.head(max_rows).to_markdown(index=False)


def _render_table_as_image(df: pd.DataFrame) -> Image.Image:
    """Render a small preview image of a table so the VLM can still see it.

    We only render a capped view — enough for the model to reason about shape
    and rough values without blowing up token budgets.
    """
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


def load_payload(file_path: str | Path) -> CalibrationPayload:
    """Load an uploaded file into a CalibrationPayload."""
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
        img = _render_table_as_image(df)
        return CalibrationPayload(image=img, table=df, source_name=name, kind="csv")

    raise ValueError(
        f"Unsupported file type '{ext}'. Accepted: "
        f"{sorted(SUPPORTED_IMAGE_EXTS | SUPPORTED_TABLE_EXTS)}"
    )
