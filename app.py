"""QCal Copilot — Gradio MVP.

Upload a calibration plot (image) or CSV, get an AI analysis from the
Ising Calibration VLM, generate a runnable CUDA-Q script, and optionally
execute it on the local cudaq simulator.

Run:
    python app.py

Environment (optional):
    NVIDIA_API_KEY        API key for build.nvidia.com NIM endpoint
    QCAL_MODEL_ID         HF model id (default: nvidia/Ising-Calibration-1-35B-A3B)
    QCAL_NIM_MODEL        NIM model name
    QCAL_NIM_ENDPOINT     NIM base URL
"""

from __future__ import annotations

import os

import gradio as gr

from qcal import analyzer, codegen, data, simulator


# ---------------------------------------------------------------------------
# Pipeline steps — each step is a pure-ish function Gradio wires together.
# ---------------------------------------------------------------------------

def step_analyze(file_obj, backend_choice: str):
    """Load the file, call the VLM, and prepare the follow-on script."""
    if file_obj is None:
        return (
            gr.update(value="Please upload an image or CSV first."),
            gr.update(value=""),
            gr.update(value=""),
            None,
        )

    try:
        payload = data.load_payload(file_obj.name if hasattr(file_obj, "name") else file_obj)
    except Exception as exc:  # noqa: BLE001
        return gr.update(value=f"**Input error:** {exc}"), "", "", None

    summary = payload.summary()
    table_md = payload.table_preview_markdown() if payload.kind == "csv" else None

    result = analyzer.analyze(
        image=payload.image,
        source=payload.source_name or "upload",
        table_preview=table_md,
        backend=backend_choice,
    )

    header = f"### Input\n{summary}\n\n### Analysis\n"
    analysis_md = header + result.markdown()

    script = codegen.generate_script(result.parsed) if result.ok else ""
    script_md_hint = "" if result.ok else "_(no script generated — fix the analysis error first)_"

    return analysis_md, script, script_md_hint, result.parsed


def step_run_simulation(script_text: str):
    if not script_text.strip():
        return "_No script to run yet. Analyze a file first._"
    result = simulator.run_script(script_text)
    return simulator.format_result_markdown(result)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
#qcal-header { text-align: center; }
#qcal-header h1 { margin-bottom: 0; }
#qcal-header p  { margin-top: 4px; color: var(--body-text-color-subdued); }
footer { visibility: hidden; }
"""


def build_ui() -> gr.Blocks:
    default_backend = "nim" if os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") else "local"

    with gr.Blocks(title="QCal Copilot", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            <div id="qcal-header">
              <h1>QCal Copilot</h1>
              <p>AI-assisted quantum calibration · Ising Calibration VLM + CUDA-Q</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Upload calibration plot or CSV",
                    file_types=[
                        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp",
                        ".csv", ".tsv",
                    ],
                )
                backend_choice = gr.Radio(
                    label="Inference backend",
                    choices=["auto", "nim", "local"],
                    value="auto",
                    info=(
                        f"Auto-detected default: **{default_backend}**. "
                        "Use `nim` for build.nvidia.com, `local` for Hugging Face."
                    ),
                )
                analyze_btn = gr.Button("Analyze calibration", variant="primary")
                run_btn = gr.Button("Run simulation")
                gr.Markdown(
                    "Tip: set `NVIDIA_API_KEY` to use the NIM endpoint without "
                    "downloading the 35B model locally."
                )

            with gr.Column(scale=2):
                analysis_out = gr.Markdown(label="Analysis")
                with gr.Accordion("Generated CUDA-Q script", open=True):
                    script_out = gr.Code(
                        language="python",
                        label="cudaq script (editable before running)",
                        value="",
                    )
                    script_hint = gr.Markdown()
                sim_out = gr.Markdown(label="Simulation result")

        # State holds the parsed analysis dict so we could extend the flow
        # later (e.g. decoder stage) without re-calling the VLM.
        analysis_state = gr.State(value=None)

        analyze_btn.click(
            fn=step_analyze,
            inputs=[file_in, backend_choice],
            outputs=[analysis_out, script_out, script_hint, analysis_state],
        )
        run_btn.click(
            fn=step_run_simulation,
            inputs=[script_out],
            outputs=[sim_out],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name=os.getenv("QCAL_HOST", "0.0.0.0"),
        server_port=int(os.getenv("QCAL_PORT", "7860")),
        share=os.getenv("QCAL_SHARE", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
