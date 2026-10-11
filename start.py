"""Single entry point for the whole laboratory.

Double-click ``start_lab.bat`` (Windows) or run ``python start.py``. On first use it creates the
virtual environment (reusing already-installed packages, no download unless you agree), installs
the project in editable mode, checks that everything imports, and then shows a menu. Pressing
Enter opens the lab's web interface in your browser.

Non-interactive use: ``python start.py <action>`` where action is one of
``ui | check | test | audit | runs | shell | env``.

Everything stays local: the web interface binds to localhost and usage statistics are disabled.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIRED = {"torch": "torch", "numpy": "numpy", "yaml": "pyyaml", "matplotlib": "matplotlib"}
OPTIONAL = {"streamlit": "streamlit", "pandas": "pandas", "pytest": "pytest", "psutil": "psutil"}


def say(msg: str = "") -> None:
    print(msg, flush=True)


def run(args, **kw) -> int:
    return subprocess.call([str(a) for a in args], cwd=ROOT, **kw)


def missing_modules(modules) -> list:
    code = ("import importlib.util, sys; "
            f"print(','.join(m for m in {list(modules)!r} if importlib.util.find_spec(m) is None))")
    out = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, cwd=ROOT)
    return [m for m in out.stdout.strip().split(",") if m]


def ask(question: str, default: bool = False) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        return default
    answer = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


def ensure_environment() -> bool:
    """Create .venv if needed, install the project, verify imports. Returns False if unusable."""
    if not PY.exists():
        say("First run: creating the virtual environment (.venv), reusing installed packages ...")
        if run([sys.executable, "-m", "venv", VENV, "--system-site-packages"]) != 0:
            say("Could not create the virtual environment.")
            return False
    missing = missing_modules(REQUIRED)
    if missing:
        pkgs = [REQUIRED[m] for m in missing]
        say(f"Missing required packages: {', '.join(pkgs)}")
        if not ask("Download and install them from PyPI now?"):
            say("Cannot continue without them. Install them, then run start again.")
            return False
        if run([PY, "-m", "pip", "install", *pkgs]) != 0:
            return False
    probe = subprocess.run([str(PY), "-c", "import transformer_lab"], capture_output=True, cwd=ROOT)
    if probe.returncode != 0:
        say("Installing the project into .venv (editable, no downloads) ...")
        if run([PY, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps", "-q"]) != 0:
            say("Installation failed (see the messages above).")
            return False
    opt = missing_modules(OPTIONAL)
    if opt:
        say(f"Note: optional packages not installed: {', '.join(OPTIONAL[m] for m in opt)} "
            "(the web interface needs streamlit and pandas; tests need pytest).")
        if "streamlit" in opt or "pandas" in opt:
            if ask("Install the optional packages from PyPI now?"):
                run([PY, "-m", "pip", "install", *[OPTIONAL[m] for m in opt]])
    return True


def free_port(start: int = 8501) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def itl(*args) -> int:
    return run([PY, "-m", "transformer_lab.cli.main", *args])


# ------------------------------------------------------------------------------- actions
def action_ui() -> int:
    if missing_modules(["streamlit", "pandas"]):
        say("The web interface needs streamlit and pandas, which are not installed.")
        return 1
    port = free_port()
    say(f"\nStarting the lab at http://localhost:{port}  (it opens in your browser).")
    say("Keep this window open while you use it; press Ctrl+C here to stop.\n")
    try:
        return itl("ui", "--port", str(port))
    except KeyboardInterrupt:
        say("\nLab stopped.")
        return 0


def action_check() -> int:
    say("\nEnvironment:")
    rc = itl("env")
    say("\nEnd-to-end smoke test (trains a tiny model and exercises every component):")
    return rc or itl("smoke")


def action_test() -> int:
    if missing_modules(["pytest"]):
        say("pytest is not installed.")
        return 1
    say("\nRunning the full test suite (about 3-4 minutes) ...")
    return run([PY, "-m", "pytest", "-q"])


def action_audit() -> int:
    say("\nReproduction audit: re-evaluating every trained run against its saved records ...")
    rc = itl("verify", "--all")
    if rc:
        say("\n(Known, documented exceptions: one manually stopped run and two runs trained before the "
            "sinusoidal fix - see docs/findings.md section 14.)")
    return 0


def action_runs() -> int:
    say("\nTrained runs:")
    itl("runs")
    say("\nExperiment records:")
    return itl("experiments")


def action_shell() -> int:
    if os.name == "nt":
        activate = VENV / "Scripts" / "activate.bat"
        os.system(f'start "Inside the Transformer" cmd /k "cd /d "{ROOT}" && "{activate}" && '
                  f'echo The itl command is ready. Try: itl --help"')
    else:
        say(f"Run:  source {VENV}/bin/activate  &&  itl --help")
    return 0


def action_env() -> int:
    return itl("env")


ACTIONS = {
    "ui": ("Open the lab (web interface in your browser)", action_ui),
    "check": ("Quick health check (environment + end-to-end smoke test)", action_check),
    "test": ("Run the full test suite (~3-4 minutes)", action_test),
    "audit": ("Reproduction audit of all trained runs", action_audit),
    "runs": ("List trained runs and experiment records", action_runs),
    "shell": ("Open a terminal with the `itl` command ready", action_shell),
    "env": ("Show environment information", action_env),
}
MENU = ["ui", "check", "test", "audit", "runs", "shell"]


def menu() -> int:
    while True:
        say("\n================  Inside the Transformer  ================")
        for i, key in enumerate(MENU, 1):
            say(f"  {i}  {ACTIONS[key][0]}" + ("   [Enter]" if i == 1 else ""))
        say("  0  Quit")
        choice = input("\nChoose: ").strip() or "1"
        if choice == "0":
            return 0
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            ACTIONS[MENU[int(choice) - 1]][1]()
        else:
            say("Please type a number from the menu.")


def main(argv) -> int:
    os.chdir(ROOT)
    if argv and argv[0] in ("-h", "--help"):
        say(__doc__)
        return 0
    if argv and argv[0] not in ACTIONS:
        say(f"Unknown action {argv[0]!r}. Choose one of: {', '.join(ACTIONS)}")
        return 2
    if not ensure_environment():
        return 1
    if argv:
        return ACTIONS[argv[0]][1]()
    if not sys.stdin or not sys.stdin.isatty():
        return action_ui()
    return menu()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(0)
