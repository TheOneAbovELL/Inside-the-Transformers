# Environment audit (Phase 0, 2026-10-11)

| item | stated in the brief | verified |
|---|---|---|
| OS | Windows | Windows 11 Home Single Language 10.0.26200 |
| CPU | AMD Ryzen 5 5300U | **AMD Ryzen 5 7530U** with Radeon Graphics, 6 cores / 12 threads |
| RAM | ~16 GB | 15.3 GB total; **~1.7–2.1 GB free** during the session |
| GPU | integrated, no NVIDIA | AMD Radeon (integrated); `torch.cuda.is_available() == False` |
| Storage | ~500 GB SSD | C: **~30 GB free** |
| Python | 3.11 or compatible | 3.13.3 (`C:\Program Files\Python313`) |
| PyTorch | — | 2.10.0+cpu (6 intra-op threads by default) |
| Other | — | numpy 2.5.2, matplotlib 3.10.8, pandas 3.0.0, streamlit 1.58.0, pytest 8.3.4, PyYAML 6.0.3, psutil 7.2.2, setuptools 82; no `uv` |
| Project directory | — | empty at start; not a git repository (initialised locally, never pushed) |

**Background load.** During all measurements CPU load was ~90% from other processes
(`OneDrive.Sync.Service`, a media player, browsers, others). Step times varied 15–130 ms for the
same model; all runtimes reported are wall-clock under that load, not benchmarks. Training used
2–4 threads per process; 4 threads was fastest in a quick benchmark under contention.

**OneDrive.** The Desktop (and therefore this project) is redirected into a OneDrive folder; the
OneDrive client was not running but its sync service was. At the user's choice, artifacts are kept
in `./artifacts`; `ITL_ARTIFACT_ROOT` relocates them.

**Implications adopted.** Tiny models (d_model 64, ≤ 2 layers); CPU-only code paths with graceful
fallback (`resolve_device`); a virtual environment with `--system-site-packages` so nothing had
to be downloaded; experiment sizes chosen to finish in minutes.
