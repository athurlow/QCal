"""Persistent config for QCal Copilot.

Stores user-level settings (API keys, endpoint overrides) in a TOML file
under ``~/.config/qcal/config.toml`` (XDG-compliant via ``platformdirs``).
Callers should always prefer environment variables when present; the config
file is a convenience for long-lived interactive use, never a security
boundary.

Schema (TOML)::

    [nvidia]
    api_key       = "nvapi-..."
    nim_endpoint  = "https://integrate.api.nvidia.com/v1/chat/completions"
    nim_model     = "nvidia/ising-calibration-1-35b-a3b"
    vlm_model_id  = "nvidia/Ising-Calibration-1-35B-A3B"
    decoder_fast  = "nvidia/Ising-Decoder-SurfaceCode-1-Fast"
    decoder_accurate = "nvidia/Ising-Decoder-SurfaceCode-1-Accurate"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from platformdirs import user_config_dir

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[attr-defined]
else:  # pragma: no cover — 3.10 fallback declared in pyproject
    import tomli as tomllib  # type: ignore[no-redef]


APP_NAME = "qcal"
CONFIG_FILENAME = "config.toml"


def config_path() -> Path:
    """Absolute path to the config file. Directory is created on first write."""
    override = os.getenv("QCAL_CONFIG_PATH")
    if override:
        return Path(override)
    return Path(user_config_dir(APP_NAME)) / CONFIG_FILENAME


def load() -> dict[str, Any]:
    """Return the parsed config, or an empty dict if none exists or is unreadable."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:  # noqa: BLE001 — bad config should never crash analysis
        return {}


def save(data: dict[str, Any]) -> Path:
    """Write ``data`` to the config file, creating parent dirs if needed.

    Uses a hand-rolled TOML writer to avoid pulling in another dep. Only
    supports the flat [section]-of-scalars shape this app needs.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for section, values in sorted(data.items()):
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in sorted(values.items()):
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    # Tighten perms on the file since it may hold a secret.
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    # default: quoted string, TOML basic-string escaping
    s = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get(section: str, key: str, default: Optional[Any] = None) -> Any:
    return (load().get(section) or {}).get(key, default)


def set_value(section: str, key: str, value: Any) -> Path:
    data = load()
    data.setdefault(section, {})[key] = value
    return save(data)


def get_api_key() -> Optional[str]:
    """NIM API key: env wins, then config file."""
    env = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
    if env:
        return env
    return get("nvidia", "api_key")


def set_api_key(value: str) -> Path:
    return set_value("nvidia", "api_key", value)


def clear_api_key() -> Optional[Path]:
    data = load()
    nvidia = data.get("nvidia") or {}
    if "api_key" in nvidia:
        nvidia.pop("api_key")
        data["nvidia"] = nvidia
        return save(data)
    return None
