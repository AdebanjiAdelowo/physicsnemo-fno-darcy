"""Run metadata: code version, environment and device."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import physicsnemo
import torch
import warp

from .config import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def physicsnemo_source() -> dict:
    """Version of the installed PhysicsNeMo and, for a git install, its commit."""
    info = {"version": physicsnemo.__version__, "commit": None, "url": None}
    try:
        text = metadata.distribution("nvidia-physicsnemo").read_text("direct_url.json")
        if text:
            direct = json.loads(text)
            info["url"] = direct.get("url")
            info["commit"] = direct.get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        pass
    return info


def collect_metadata(device: torch.device) -> dict:
    """Environment record stored with every study."""
    status = _git("status", "--porcelain")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "device": str(device),
        "gpu_name": gpu,
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "physicsnemo": physicsnemo_source(),
        "warp_version": warp.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
