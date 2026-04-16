"""QCal Copilot — Gradio MVP.

Upload a calibration plot (image) or CSV, get an AI analysis from the
Ising Calibration VLM, generate a runnable CUDA-Q script, execute it on
the local cudaq simulator, and optionally run the Ising 3D CNN decoder
stage on a synthetic surface-code syndrome volume.

Run:
    python app.py

Environment (optional):
    NVIDIA_API_KEY              API key for build.nvidia.com NIM endpoint
    QCAL_MODEL_ID               HF model id for the calibration VLM
    QCAL_NIM_MODEL              NIM model name
    QCAL_NIM_ENDPOINT           NIM base URL
    QCAL_DECODER_FAST_ID        Override fast decoder HF id
    QCAL_DECODER_ACCURATE_ID    Override accurate decoder HF id
"""

from __future__ import annotations

import os

import gradio as gr

from qcal import analyzer, codegen, data, decoder, simulator


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
            gr.update(value=decoder.suggest_error_rate(None)),
        )

    try:
        payload = data.load_payload(file_obj.name if hasattr(file_obj, "name") else file_obj)
    except Exception as exc:  # noqa: BLE001
        return (
            gr.update(value=f"**Input error:** {exc}"),
            "", "", None,
            gr.update(value=decoder.suggest_error_rate(None)),
        )

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
    script_hint = "" if result.ok else "_(no script generated — fix the analysis error first)_"
    suggested_p = decoder.suggest_error_rate(result.parsed)

    return analysis_md, script, script_hint, result.parsed, gr.update(value=suggested_p)


def step_run_simulation(script_text: str):
    if not script_text.strip():
        return "_No script to run yet. Analyze a file first._"
    result = simulator.run_script(script_text)
    return simulator.format_result_markdown(result)


def step_run_decoder(
    variant: str,
    distance: int,
    rounds: int,
    error_rate: float,
    n_shots: int,
    analysis: dict | None,
    script_text: str,
):
    """Run the Ising 3D CNN decoder and refresh metrics/plots/script."""
    result = decoder.run_decoder(
        variant=variant,
        distance=int(distance),
        rounds=int(rounds),
        error_rate=float(error_rate),
        n_shots=int(n_shots),
    )
    metrics_md = result.markdown()

    try:
        fig = decoder.plot_comparison(result) if result.ok else None
    except Exception as exc:  # noqa: BLE001
        fig = None
        metrics_md += f"\n\n_(plot unavailable: {exc})_"

    # Re-generate the CUDA-Q script with a decoder header block, so the user
    # can copy a script that documents which decoder ran upstream.
    new_script = script_text
    if result.ok and analysis:
        decoder_info = {
            "variant": result.variant,
            "model_id": result.model_id,
            "distance": result.distance,
            "rounds": result.rounds,
            "density_reduction": result.density_reduction,
            "ler_improvement": result.ler_improvement,
        }
        new_script = codegen.generate_script(analysis, decoder_info=decoder_info)

    return metrics_md, fig, new_script


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
              <p>AI-assisted quantum calibration · Ising VLM + 3D CNN decoder + CUDA-Q</p>
            </div>
            """
        )

        # State holds the parsed analysis dict so downstream stages (decoder,
        # future 3D CNN tile, etc.) can read it without re-calling the VLM.
        analysis_state = gr.State(value=None)

        # ---- Stage 1: calibration analysis ---------------------------------
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

        # ---- Stage 2: error-correction decoder -----------------------------
        with gr.Accordion("Error-correction decoder (Ising 3D CNN)", open=False):
            gr.Markdown(
                "Sparsify a synthetic surface-code syndrome volume with one of "
                "the Ising pre-decoders, then hand off to MWPM (PyMatching). "
                "Error rate defaults are suggested from your calibration analysis."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    variant_choice = gr.Radio(
                        label="Decoder variant",
                        choices=[decoder.VARIANT_FAST, decoder.VARIANT_ACCURATE],
                        value=decoder.VARIANT_FAST,
                        info=(
                            "`fast` ≈ 912k params — lower latency. "
                            "`accurate` ≈ 1.79M params — better LER."
                        ),
                    )
                    distance_slider = gr.Slider(
                        3, 11, value=5, step=2,
                        label="Code distance (d)",
                    )
                    rounds_slider = gr.Slider(
                        1, 17, value=5, step=1,
                        label="Syndrome rounds (T)",
                    )
                    error_rate_slider = gr.Slider(
                        0.0, 0.05, value=0.005, step=0.001,
                        label="Physical error rate (p)",
                    )
                    shots_slider = gr.Slider(
                        16, 1024, value=128, step=16,
                        label="Shots",
                    )
                    run_decoder_btn = gr.Button(
                        "Run decoder", variant="primary"
                    )

                with gr.Column(scale=2):
                    decoder_metrics = gr.Markdown(
                        "_Run the decoder to see density reduction, MWPM timing, "
                        "and LER-proxy improvement here._"
                    )
                    decoder_plot = gr.Plot(label="Raw vs denoised syndromes")

        analyze_btn.click(
            fn=step_analyze,
            inputs=[file_in, backend_choice],
            outputs=[
                analysis_out, script_out, script_hint, analysis_state,
                error_rate_slider,
            ],
        )
        run_btn.click(
            fn=step_run_simulation,
            inputs=[script_out],
            outputs=[sim_out],
        )
        run_decoder_btn.click(
            fn=step_run_decoder,
            inputs=[
                variant_choice, distance_slider, rounds_slider,
                error_rate_slider, shots_slider, analysis_state, script_out,
            ],
            outputs=[decoder_metrics, decoder_plot, script_out],
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
