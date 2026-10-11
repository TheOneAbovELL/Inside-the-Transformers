"""Seeding, environment capture, artifact locations and resource diagnostics."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    """Root directory for all generated artifacts (runs, experiments, reports).

    Defaults to ``<project>/artifacts``; override with the ``ITL_ARTIFACT_ROOT`` environment
    variable (e.g. to keep large files outside a cloud-synced folder).
    """
    root = Path(os.environ.get("ITL_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch global RNGs.

    The library itself mostly uses *explicit* generators (``np.random.Generator`` /
    ``torch.Generator``) so results do not depend on global state; this is a belt-and-braces call
    for any third-party code.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def derive_seeds(seed: int, n: int) -> list[int]:
    """Derive ``n`` statistically independent 63-bit seeds from one experiment seed."""
    ss = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1)) for child in ss.spawn(n)]


def resolve_device(requested: str = "cpu") -> torch.device:
    """Return the requested device, degrading gracefully to CPU if it is unavailable."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[transformer_lab] {requested!r} requested but CUDA is unavailable; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def configure_threads(num_threads: Optional[int]) -> int:
    if num_threads:
        torch.set_num_threads(int(num_threads))
    return torch.get_num_threads()


def memory_info() -> dict:
    """Process RSS and system memory in MB (psutil optional; returns what is available)."""
    info: dict = {}
    try:
        import psutil  # optional dependency

        proc = psutil.Process()
        vm = psutil.virtual_memory()
        info.update(
            process_rss_mb=round(proc.memory_info().rss / 2**20, 1),
            system_total_mb=round(vm.total / 2**20, 1),
            system_available_mb=round(vm.available / 2**20, 1),
        )
    except Exception:  # pragma: no cover - depends on optional package
        info["note"] = "psutil not installed; memory diagnostics unavailable"
    return info


def code_fingerprint() -> str:
    """SHA-256 (truncated) over the library's source files, so every artifact records which code
    produced it even though the repository has no commits."""
    import hashlib

    h = hashlib.sha256()
    src = Path(__file__).resolve().parent
    for p in sorted(src.rglob("*.py")):
        h.update(p.relative_to(src).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def environment_info() -> dict:
    """Software/hardware snapshot stored with every run for reproducibility."""
    info = {
        "code_fingerprint": code_fingerprint(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    info.update(memory_info())
    return info


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)  # atomic on the same filesystem


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@contextmanager
def timer() -> Iterator[dict]:
    """``with timer() as t: ...`` then ``t["seconds"]``."""
    out: dict = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start
