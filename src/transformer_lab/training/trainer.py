"""Configuration-driven, resumable training.

Run directory layout (``artifacts/runs/<name>_s<seed>_<timestamp>/``)::

    config.yaml            resolved ExperimentConfig (re-run with `itl train --config config.yaml`)
    resolved_model.json    the exact ModelConfig built (vocab/n_ctx/seed filled in)
    env.json               software + hardware snapshot
    metrics.jsonl          one JSON object per log/eval event
    summary.json           final machine-readable summary (status, metrics, baselines, leakage audit)
    checkpoints/latest.pt  rolling full state for resuming (model, optimiser, scheduler, RNG)
    checkpoints/step_XXXXXX.pt  permanent snapshots at ``train.checkpoint_steps``
    checkpoints/final.pt   final model

Seeds: the experiment seed is expanded with ``numpy.random.SeedSequence`` into independent seeds
for (model init, training stream, validation set, test set). Validation (same distribution as
training, different seed) drives early stopping; the test set (``split="eval"``, held-out pairs where
the task supports it) is only reported, never used for decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import ExperimentConfig, ModelConfig
from ..data.tasks import Task, TaskBatch, make_task
from ..evaluation.evaluate import baselines, evaluate_model
from ..models import Transformer, lm_loss
from ..utils import (artifact_root, configure_threads, derive_seeds, environment_info, memory_info, read_json,
                     resolve_device, seed_everything, timestamp_id, write_json)
from .checkpoint import load_checkpoint, save_checkpoint


class TrainingDiverged(RuntimeError):
    pass


def build_task_and_model_config(cfg: ExperimentConfig) -> tuple[Task, ModelConfig, dict]:
    """Instantiate the task and derive the complete model config + all seeds."""
    task = make_task(cfg.task.name, cfg.task.params)
    s_model, s_train, s_val, s_test = derive_seeds(cfg.seed, 4)
    n_ctx = max(cfg.model.n_ctx or 0, task.seq_len)
    model_cfg = replace(cfg.model, vocab_size=task.vocab_size, n_ctx=n_ctx, seed=s_model % (2**63))
    seeds = {"model": s_model, "train_data": s_train, "val_data": s_val, "test_data": s_test}
    return task, model_cfg, seeds


def eval_sets(task: Task, seeds: dict, size: int) -> tuple[TaskBatch, TaskBatch]:
    val = task.sample(size, np.random.default_rng(seeds["val_data"]), split="train")
    test = task.sample(size, np.random.default_rng(seeds["test_data"]), split="eval")
    return val, test


def lr_factor(step: int, cfg) -> float:
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return 1.0
    span = max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    if cfg.schedule == "cosine":
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return 1.0 - (1 - cfg.min_lr_ratio) * progress  # linear


def make_optimizer(model: Transformer, cfg) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only (not biases / LayerNorm gains)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=cfg.betas,
    )


def new_run_dir(cfg: ExperimentConfig, root: Optional[Path] = None) -> Path:
    root = Path(root) if root else artifact_root() / "runs"
    base = root / f"{cfg.name}_s{cfg.seed}_{timestamp_id()}"
    path, k = base, 1
    while path.exists():
        path = base.with_name(f"{base.name}_{k}")
        k += 1
    path.mkdir(parents=True)
    return path


def _row_hashes(tokens: torch.Tensor) -> set:
    arr = tokens.numpy()
    return {hashlib.blake2b(row.tobytes(), digest_size=12).digest() for row in arr}


def training_stream_hashes(task: Task, train_seed: int, steps: int, batch_size: int) -> set:
    """Hashes of every training sequence of a run, regenerated deterministically from its seed."""
    rng = np.random.default_rng(train_seed)
    seen: set = set()
    for _ in range(steps):
        seen |= _row_hashes(task.sample(batch_size, rng, split="train").tokens)
    return seen


def unseen_mask(batch: TaskBatch, seen: set) -> np.ndarray:
    """Boolean mask of examples whose exact token sequence never occurred in training."""
    return np.array([hashlib.blake2b(r.tobytes(), digest_size=12).digest() not in seen for r in batch.tokens.numpy()])


def leakage_audit(task: Task, train_seed: int, steps: int, batch_size: int, *batches: TaskBatch) -> dict:
    """Regenerate the exact training stream from its seed and count eval sequences that occurred in it."""
    rng = np.random.default_rng(train_seed)
    seen: set = set()
    for _ in range(steps):
        seen |= _row_hashes(task.sample(batch_size, rng, split="train").tokens)
    out = {"train_sequences_checked": steps * batch_size, "unique_train_sequences": len(seen)}
    for i, b in enumerate(batches):
        hashes = [hashlib.blake2b(r.tobytes(), digest_size=12).digest() for r in b.tokens.numpy()]
        out[f"set{i}_overlap"] = int(sum(h in seen for h in hashes))
        out[f"set{i}_size"] = len(hashes)
    return out


class Trainer:
    def __init__(self, cfg: ExperimentConfig, run_dir: Optional[Path] = None, resume: bool = False,
                 verbose: bool = True, audit_leakage: bool = True) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self.audit_leakage = audit_leakage
        self.run_dir = Path(run_dir) if run_dir else new_run_dir(cfg)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.resume = resume

        tc = cfg.train
        configure_threads(tc.num_threads)
        self.device = resolve_device(tc.device)
        seed_everything(cfg.seed)
        self.task, self.model_cfg, self.seeds = build_task_and_model_config(cfg)
        self.model = Transformer(self.model_cfg).to(self.device)
        self.opt = make_optimizer(self.model, tc)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(s, tc))
        self.rng = np.random.default_rng(self.seeds["train_data"])
        self.val_batch, self.test_batch = eval_sets(self.task, self.seeds, tc.eval_size)
        self.step = 0
        self.elapsed = 0.0
        self.best = {"metric": None, "step": None}
        self.bad_evals = 0
        self.history: list[dict] = []
        self.initial_eval: Optional[dict] = None

        if resume:
            self._load_state()
        else:
            cfg.save(self.run_dir / "config.yaml")
            write_json(self.run_dir / "resolved_model.json", self.model_cfg.to_dict())
            write_json(self.run_dir / "env.json", environment_info())
            write_json(self.run_dir / "task_spec.json", self.task.spec())
            write_json(self.run_dir / "seeds.json", self.seeds)

    # ------------------------------------------------------------------ state
    def _state(self) -> dict:
        return dict(
            step=self.step, elapsed=self.elapsed, optimizer=self.opt.state_dict(), scheduler=self.sched.state_dict(),
            rng_train=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(), best=self.best,
            bad_evals=self.bad_evals, experiment_config=self.cfg.to_dict(), initial_eval=self.initial_eval,
        )

    def _load_state(self) -> None:
        path = self.ckpt_dir / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint to resume from at {path}")
        ckpt = load_checkpoint(path)
        if ckpt["experiment_config"] != self.cfg.to_dict():
            raise ValueError("Config differs from the checkpoint's; refusing to resume a different experiment")
        self.model.load_state_dict(ckpt["state_dict"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.sched.load_state_dict(ckpt["scheduler"])
        self.rng.bit_generator.state = ckpt["rng_train"]
        torch.set_rng_state(ckpt["torch_rng"])
        self.step, self.elapsed = ckpt["step"], ckpt["elapsed"]
        self.best, self.bad_evals = ckpt["best"], ckpt["bad_evals"]
        self.initial_eval = ckpt.get("initial_eval")
        self._log({"kind": "resume", "step": self.step})

    def save_latest(self) -> None:
        save_checkpoint(self.ckpt_dir / "latest.pt", self.model, **self._state())

    # ---------------------------------------------------------------- logging
    def _log(self, rec: dict) -> None:
        rec = {"time": round(time.time(), 3), **rec}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.history.append(rec)

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------- eval
    def evaluate(self) -> dict:
        val = evaluate_model(self.model, self.val_batch, n_boot=200)
        test = evaluate_model(self.model, self.test_batch, n_boot=200)
        self.model.train()
        rec = {"kind": "eval", "step": self.step}
        for split, s in (("val", val), ("test", test)):
            for k in ("loss", "accuracy", "seq_correct"):
                rec[f"{split}_{k}"] = s[k]["mean"]
                rec[f"{split}_{k}_lo"], rec[f"{split}_{k}_hi"] = s[k]["lo"], s[k]["hi"]
        self._log(rec)
        return rec

    def _early_stop(self, rec: dict) -> bool:
        tc = self.cfg.train
        if tc.target_accuracy is not None and rec["val_accuracy"] >= tc.target_accuracy:
            self._print(f"  target accuracy {tc.target_accuracy} reached at step {self.step}")
            return True
        if tc.early_stopping_patience is None:
            return False
        key = tc.early_stopping_metric
        val = rec[key] if key == "val_accuracy" else -rec[key]  # higher is better
        if self.best["metric"] is None or val > self.best["metric"] + tc.early_stopping_min_delta:
            self.best = {"metric": val, "step": self.step}
            self.bad_evals = 0
            return False
        self.bad_evals += 1
        return self.bad_evals >= tc.early_stopping_patience

    # ------------------------------------------------------------------- main
    def train(self) -> dict:
        tc = self.cfg.train
        status = "completed"
        self.model.train()
        if self.step == 0:
            self.initial_eval = self.evaluate()  # baseline behaviour of the untrained model
            if 0 in tc.checkpoint_steps:
                save_checkpoint(self.ckpt_dir / "step_000000.pt", self.model, step=0, eval=self.initial_eval)
        self._print(f"[train] {self.cfg.name} seed={self.cfg.seed} params={self.model.n_params():,} "
                    f"task={self.task.name} device={self.device} threads={torch.get_num_threads()} -> {self.run_dir}")
        t_last = time.perf_counter()
        rng_at_step_start = self.rng.bit_generator.state
        try:
            while self.step < tc.steps:
                # Snapshot so an interrupt mid-step rolls back to a consistent state (otherwise a
                # resumed run would skip the batch drawn for the unfinished step).
                rng_at_step_start = self.rng.bit_generator.state
                batch = self.task.sample(tc.batch_size, self.rng, split="train").to(self.device)
                logits = self.model(batch.tokens)
                loss = lm_loss(logits, batch.tokens, batch.training_mask)
                if not torch.isfinite(loss):
                    raise TrainingDiverged(f"non-finite loss {loss.item()} at step {self.step}")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                clip = tc.grad_clip if tc.grad_clip is not None else float("inf")
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if not torch.isfinite(gnorm):
                    raise TrainingDiverged(f"non-finite gradient norm at step {self.step}")
                self.opt.step()
                self.sched.step()
                self.step += 1
                rng_at_step_start = self.rng.bit_generator.state  # step is now complete
                now = time.perf_counter()
                self.elapsed += now - t_last
                t_last = now

                if self.step % tc.log_every == 0 or self.step == tc.steps:
                    self._log({"kind": "train", "step": self.step, "loss": loss.item(), "grad_norm": gnorm.item(),
                               "lr": self.sched.get_last_lr()[0], "elapsed_s": round(self.elapsed, 2)})
                stop = False
                if self.step % tc.eval_every == 0 or self.step == tc.steps:
                    rec = self.evaluate()
                    self._print(f"  step {self.step:6d} | train {loss.item():.4f} | val loss {rec['val_loss']:.4f} "
                                f"acc {rec['val_accuracy']:.3f} | test acc {rec['test_accuracy']:.3f} | "
                                f"{self.elapsed:.0f}s")
                    stop = self._early_stop(rec)
                if self.step in tc.checkpoint_steps:
                    save_checkpoint(self.ckpt_dir / f"step_{self.step:06d}.pt", self.model, step=self.step)
                if self.step % tc.checkpoint_every == 0:
                    self.save_latest()
                if stop:
                    status = "early_stopped"
                    break
        except KeyboardInterrupt:
            # Parameters only change inside opt.step(), immediately followed by step += 1, so
            # (model, step) are consistent; restore the data RNG to the last completed step.
            self.rng.bit_generator.state = rng_at_step_start
            self.save_latest()
            self._log({"kind": "interrupted", "step": self.step})
            self._write_summary("interrupted")
            self._print(f"[train] interrupted at step {self.step}; resume with `itl train --resume {self.run_dir}`")
            raise
        except TrainingDiverged as e:
            self._log({"kind": "diverged", "step": self.step, "error": str(e)})
            self._write_summary("diverged", error=str(e))
            raise

        self.save_latest()
        done = [h for h in self.history if h["kind"] == "eval" and h["step"] == self.step]
        final = done[-1] if done else self.evaluate()
        save_checkpoint(self.ckpt_dir / "final.pt", self.model, step=self.step, eval=final)
        return self._write_summary(status)

    def _write_summary(self, status: str, error: Optional[str] = None) -> dict:
        evals = [h for h in self.history if h["kind"] == "eval"]
        final_eval = evals[-1] if evals else None
        summary = {
            "status": status,
            "error": error,
            "run_dir": str(self.run_dir),
            "experiment": self.cfg.to_dict(),
            "model_config": self.model_cfg.to_dict(),
            "n_params": self.model.n_params(),
            "task": self.task.spec(),
            "seeds": self.seeds,
            "steps_completed": self.step,
            "elapsed_s": round(self.elapsed, 2),
            "steps_per_s": round(self.step / self.elapsed, 2) if self.elapsed else None,
            "initial_eval": self.initial_eval,
            "final_eval": final_eval,
            "baselines_test": baselines(self.task, self.test_batch),
            "baselines_val": baselines(self.task, self.val_batch),
            "memory": memory_info(),
        }
        if self.audit_leakage and status in ("completed", "early_stopped"):
            summary["leakage_audit"] = leakage_audit(self.task, self.seeds["train_data"], self.step,
                                                     self.cfg.train.batch_size, self.val_batch, self.test_batch)
            summary["leakage_audit"]["sets"] = ["val", "test"]
        write_json(self.run_dir / "summary.json", summary)
        return summary


def train_from_config(cfg: ExperimentConfig, run_dir=None, verbose=True, audit_leakage=True) -> dict:
    return Trainer(cfg, run_dir=run_dir, verbose=verbose, audit_leakage=audit_leakage).train()


def resume_run(run_dir, verbose=True) -> dict:
    run_dir = Path(run_dir)
    cfg = ExperimentConfig.load(run_dir / "config.yaml")
    return Trainer(cfg, run_dir=run_dir, resume=True, verbose=verbose).train()


def load_run(run_dir, checkpoint: str = "final"):
    """Load ``(model, task, summary_or_None, cfg)`` from a run directory."""
    from .checkpoint import load_model

    run_dir = Path(run_dir)
    cfg = ExperimentConfig.load(run_dir / "config.yaml")
    task = make_task(cfg.task.name, cfg.task.params)
    path = run_dir / "checkpoints" / (checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    model, _ = load_model(path)
    summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else None
    return model, task, summary, cfg
