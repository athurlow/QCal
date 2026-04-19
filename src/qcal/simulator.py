"""Execute the generated CUDA-Q script in an isolated namespace.

We write the script to a temp file and exec it so users see the *exact* code
that runs — identical to what they copy out of the UI.
"""

from __future__ import annotations

import io
import sys
import tempfile
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any


def run_script(script_text: str) -> dict[str, Any]:
    """Run a generated CUDA-Q script and capture its output.

    Returns a dict with keys:
      - ok: bool
      - stdout, stderr: captured streams
      - result: dict returned by the script's run() function (if present)
      - error: traceback string on failure
    """
    tmp = Path(tempfile.mkstemp(prefix="qcal_gen_", suffix=".py")[1])
    tmp.write_text(script_text)

    namespace: dict[str, Any] = {"__name__": "__qcal_generated__", "__file__": str(tmp)}
    stdout, stderr = io.StringIO(), io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = compile(script_text, str(tmp), "exec")
            exec(code, namespace)  # noqa: S102 — script author is the same user
            result = namespace["run"]() if callable(namespace.get("run")) else None
        return {
            "ok": True,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": result,
            "error": None,
        }
    except ModuleNotFoundError as exc:
        return {
            "ok": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": None,
            "error": (
                f"Missing dependency: {exc.name}. "
                "If it's `cudaq`, install CUDA-Q (see README)."
            ),
        }
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": None,
            "error": traceback.format_exc(),
        }
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def format_result_markdown(result: dict[str, Any]) -> str:
    if not result:
        return "_(no simulation run yet)_"
    if not result["ok"]:
        body = [f"**Simulation failed.**\n\n```\n{result['error']}\n```"]
        if result["stdout"]:
            body.append(f"**stdout:**\n```\n{result['stdout']}\n```")
        if result["stderr"]:
            body.append(f"**stderr:**\n```\n{result['stderr']}\n```")
        return "\n\n".join(body)

    lines = ["**Simulation complete.**", ""]
    r = result.get("result") or {}
    if r:
        lines.append("**Measurement stats:**")
        lines.append(f"- `theta` (rad) = {r.get('theta')}")
        lines.append(f"- `shots` = {r.get('shots')}")
        lines.append(f"- `counts` = {r.get('counts')}")
        lines.append(f"- `P(|1>)` = {r.get('p1')}")
    if result["stdout"]:
        lines.append("")
        lines.append("**stdout:**")
        lines.append("```")
        lines.append(result["stdout"].rstrip())
        lines.append("```")
    return "\n".join(lines)
