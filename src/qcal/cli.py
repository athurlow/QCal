"""Command-line interface for QCal Copilot.

Installed as the ``qcal`` console script via :mod:`pyproject.toml`.

Commands
--------
* ``qcal analyze FILE``      — run the Ising VLM on a calibration artifact
* ``qcal decode``            — run the Ising 3D CNN decoder on synthetic syndromes
* ``qcal generate FILE``     — write a CUDA-Q script from a saved analysis JSON
* ``qcal serve``             — launch the Gradio UI (needs ``qcal[gui]``)
* ``qcal login``             — store an NVIDIA NIM API key in the user config
* ``qcal logout``            — clear the stored API key
* ``qcal config``            — print resolved config
* ``qcal version``           — print installed version

Design notes
------------
All commands honor ``--json`` for machine-readable output so labs can wire
``qcal`` into CI or drift-monitoring cron jobs. Exit codes: 0 on success,
1 on analyzer/decoder error, 2 on user error, 3 on missing optional dep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__, config
from .data import load_payload


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AI-assisted quantum calibration (Ising VLM + 3D CNN decoder + CUDA-Q).",
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Input file (.npy, .npz, .csv, .png, .jpg, …)."),
    experiment: str = typer.Option(
        "unknown",
        "--experiment", "-e",
        help="Experiment type for .npy inputs: rabi|ramsey|t1|t2_echo|readout_iq|rabi_chevron|iq_trace|resonator_spec|unknown.",
    ),
    backend: str = typer.Option("auto", "--backend", "-b", help="auto | nim | local"),
    no_fit: bool = typer.Option(False, "--no-fit", help="Skip scipy curve-fitting for .npy inputs."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the markdown report to this path."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write the raw analyzer JSON to this path."),
    script_out: Optional[Path] = typer.Option(None, "--script", help="Write the generated CUDA-Q script to this path."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress stdout; only write files + exit code."),
) -> None:
    """Analyze a calibration artifact with the Ising VLM."""
    from . import analyzer, codegen  # local imports so `qcal --help` stays fast

    try:
        payload = load_payload(file, experiment_type=experiment, fit=not no_fit)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Failed to load {file}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    result = analyzer.analyze_payload(payload, backend=backend)

    if not result.ok:
        typer.secho(
            f"Analyzer error ({result.backend}): {result.error or 'empty response'}",
            err=True, fg=typer.colors.RED,
        )
        if result.raw_text and not quiet:
            typer.echo(result.raw_text)
        raise typer.Exit(code=1)

    md = result.markdown()
    if out:
        out.write_text(md, encoding="utf-8")
    if json_out:
        json_out.write_text(
            json.dumps(
                {
                    "analysis": result.parsed,
                    "fit": result.fit_params,
                    "backend": result.backend,
                    "source": result.source,
                    "qcal_version": __version__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if script_out:
        script_text = codegen.generate_script(result.parsed)
        script_out.write_text(script_text, encoding="utf-8")

    if not quiet:
        typer.echo(md)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

@app.command()
def decode(
    variant: str = typer.Option("fast", "--variant", "-v", help="fast | accurate"),
    distance: int = typer.Option(5, "--distance", "-d", min=3, max=15),
    rounds: int = typer.Option(5, "--rounds", "-r", min=1, max=25),
    error_rate: float = typer.Option(0.005, "--p", help="Physical error rate."),
    shots: int = typer.Option(128, "--shots", "-n", min=1),
    seed: int = typer.Option(42, "--seed"),
    json_out: Optional[Path] = typer.Option(None, "--json"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run the Ising 3D CNN pre-decoder on a synthetic syndrome volume."""
    from . import decoder

    result = decoder.run_decoder(
        variant=variant,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
        n_shots=shots,
        seed=seed,
    )
    if not result.ok:
        typer.secho(f"Decoder error: {result.error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if json_out:
        payload = {
            "variant": result.variant,
            "model_id": result.model_id,
            "distance": result.distance,
            "rounds": result.rounds,
            "error_rate": result.error_rate,
            "n_shots": result.n_shots,
            "density_before": result.density_before,
            "density_after": result.density_after,
            "density_reduction": result.density_reduction,
            "inference_ms": result.inference_ms,
            "mwpm_ms_before": result.mwpm_ms_before,
            "mwpm_ms_after": result.mwpm_ms_after,
            "ler_proxy_before": result.ler_proxy_before,
            "ler_proxy_after": result.ler_proxy_after,
            "ler_improvement": result.ler_improvement,
            "backend_note": result.backend_note,
        }
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not quiet:
        typer.echo(result.markdown())


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@app.command()
def generate(
    analysis_json: Path = typer.Argument(..., exists=True, readable=True, help="Analyzer JSON produced by `qcal analyze --json`."),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
) -> None:
    """Generate a CUDA-Q script from a saved analyzer JSON."""
    from . import codegen

    try:
        blob = json.loads(analysis_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.secho(f"Invalid JSON in {analysis_json}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    analysis = blob.get("analysis") if isinstance(blob, dict) and "analysis" in blob else blob
    script = codegen.generate_script(analysis)
    if out:
        out.write_text(script, encoding="utf-8")
    else:
        typer.echo(script)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", envvar="QCAL_HOST"),
    port: int = typer.Option(7860, envvar="QCAL_PORT"),
    share: bool = typer.Option(False, "--share", envvar="QCAL_SHARE"),
) -> None:
    """Launch the Gradio UI locally (requires ``qcal[gui]``)."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        typer.secho(
            "Gradio isn't installed. Run: pip install 'qcal-copilot[gui]'",
            err=True, fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    # Import the app defined at repo root; when running from an installed
    # wheel we fall back to a minimal in-package launcher.
    try:
        sys.path.insert(0, str(Path.cwd()))
        import app as gradio_app  # type: ignore[import-not-found]

        demo = gradio_app.build_ui()
    except Exception:  # noqa: BLE001
        from . import app_inproc

        demo = app_inproc.build_ui()

    demo.queue(max_size=8).launch(server_name=host, server_port=port, share=share)


# ---------------------------------------------------------------------------
# login / logout / config
# ---------------------------------------------------------------------------

@app.command()
def login(
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="NIM API key. If omitted, you'll be prompted (hidden input).",
    ),
) -> None:
    """Store an NVIDIA NIM API key under ``~/.config/qcal/config.toml``."""
    key = api_key or typer.prompt("NVIDIA NIM API key", hide_input=True)
    if not key.strip():
        typer.secho("Empty key — nothing saved.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)
    path = config.set_api_key(key.strip())
    typer.secho(f"Saved API key to {path} (mode 0600).", fg=typer.colors.GREEN)


@app.command()
def logout() -> None:
    """Remove the stored NIM API key."""
    path = config.clear_api_key()
    if path:
        typer.echo(f"Cleared API key from {path}.")
    else:
        typer.echo("No stored API key to clear.")


@app.command(name="config")
def config_cmd(show_key: bool = typer.Option(False, "--show-key", help="Print the stored API key in full.")) -> None:
    """Print the resolved config (key masked unless ``--show-key``)."""
    data = config.load()
    key = config.get_api_key()
    if key and not show_key:
        masked = key[:6] + "…" + key[-4:] if len(key) > 12 else "***"
        data.setdefault("nvidia", {})["api_key"] = masked
    elif key:
        data.setdefault("nvidia", {})["api_key"] = key
    typer.echo(json.dumps(data, indent=2))
    typer.echo(f"\nconfig path: {config.config_path()}")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the installed qcal-copilot version."""
    typer.echo(__version__)


def main() -> None:  # entry point target for scripts
    app()


if __name__ == "__main__":
    main()
