"""The one-click launcher (start.py / start_lab.bat) dispatches actions and rejects unknown ones."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
START = ROOT / "start.py"


def _run(*args):
    return subprocess.run([sys.executable, str(START), *args], capture_output=True, text=True,
                          cwd=ROOT, stdin=subprocess.DEVNULL, timeout=120)


def test_launcher_files_exist():
    assert START.exists()
    bat = (ROOT / "start_lab.bat").read_bytes()
    assert b"\r\n" in bat and b"start.py" in bat


def test_unknown_action_is_rejected():
    out = _run("no-such-action")
    assert out.returncode == 2
    assert "Unknown action" in out.stdout


def test_help_lists_actions():
    out = _run("--help")
    assert out.returncode == 0
    assert "ui | check | test | audit | runs | shell | env" in out.stdout


@pytest.mark.skipif(not (ROOT / ".venv" / "Scripts" / "python.exe").exists(), reason="needs the project .venv")
def test_env_action_runs():
    out = _run("env")
    assert out.returncode == 0
    assert '"code_fingerprint"' in out.stdout
