"""Train a list of (config, seed) pairs, N at a time, each in its own process.

Usage: python scripts/train_queue.py --parallel 2 configs/experiments/a.yaml:0,1,2 configs/experiments/b.yaml:0
Each run writes its own directory under artifacts/runs and a line to artifacts/train_queue.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="config.yaml:seed[,seed...]")
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()
    queue = []
    for job in args.jobs:
        cfg, seeds = job.rsplit(":", 1)
        queue += [(cfg, int(s)) for s in seeds.split(",")]
    log = ROOT / "artifacts" / "train_queue.log"
    log.parent.mkdir(exist_ok=True)
    running = []
    while queue or running:
        while queue and len(running) < args.parallel:
            cfg, seed = queue.pop(0)
            out = open(ROOT / "artifacts" / f"train_{Path(cfg).stem}_s{seed}.log", "w")
            p = subprocess.Popen([sys.executable, "-m", "transformer_lab.cli.main", "train", "--config", cfg,
                                  "--set", f"seed={seed}", "--quiet"], stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((p, cfg, seed, time.time()))
        for item in list(running):
            p, cfg, seed, t0 = item
            if p.poll() is not None:
                with open(log, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} exit={p.returncode} {cfg} seed={seed} {time.time() - t0:.0f}s\n")
                running.remove(item)
        time.sleep(2)


if __name__ == "__main__":
    main()
