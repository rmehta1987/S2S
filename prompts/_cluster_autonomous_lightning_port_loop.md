# Autonomous cluster driver: port S2S to PyTorch Lightning (SNFO-mirrored)

You are running **autonomously on Midway3 (RCC), no operator in the loop.** Your job: incrementally port the
S2S model codebase (`v2.0/`, the canonical bench-instrumented `train.py`/`inference.py`) onto **PyTorch
Lightning**, restructured to mirror the sibling project at `/home/rmeht/Projects/SNFO`, so that S2S and SNFO
ultimately share **one** codebase whose **only material difference is the model definition**
(`PanguModel_Plasim` vs SNFO's models).

You do the work by **delegating each phase to the `lightning-porter` subagent** (`.claude/agents/lightning-porter.md`)
via the `Agent` tool, then verifying its green smoke + commit and emitting the loop sentinel. The porter
**reuses (never rewrites)** the S2S model/losses/HDF5 loaders, preserves the S2S_BENCH/NVTX instrumentation, and
gates every commit on a real smoke test. This driver is modeled on the L2LGWAS dir-57 autonomous loop and
inherits its implement→test→commit-after-green discipline.

> **CENTRAL DISCIPLINE: verify, never assume.** Never claim a smoke "passed" you did not run and read. Never
> claim a phase is committed without confirming a new commit on the `lightning-port` branch. Forward-looking
> optimism ("this should run") is a load-bearing claim — gate it on the smoke, do not assert it.

---

## 0. Authority, invariants, fences (read before anything destructive)

**Read `CLAUDE.md` FIRST.** Its hard constraints are non-negotiable; carry them verbatim into the ported code:
- **`train.py`/`inference.py` are the canonical, bench-instrumented variants;** `train_optimized.py`/
  `inference_optimized.py` are **OLDER** despite the name. **Base the port on `train.py`/`inference.py`. NEVER
  invert the attribution; NEVER edit the `_optimized` files** (a `PreToolUse` hook hard-blocks edits to them).
- **`PYTHONPATH=v2.0/` is required** for `from utils...`/`from networks...`. Keep the ported tree importable;
  every smoke you run sets it.
- **HDF5 data lives at `/project/pedramh/h5data/h5data`** — readable from the test + `pedramh-gpu` partitions;
  **no staging step.** Do not document otherwise.
- **"handoff", not "dispatch"** in new prose/docstrings (except named tests/filenames).
- **The NGC key (`APPTAINER_DOCKER_PASSWORD`) in `v2.0/HPC_scripts/nvidia_*.sh` is a leak** — never
  propagate/echo it (hook-blocked); never edit those files (hook-blocked).
- **Configs are cluster-specific and fail deep, not early.**

**Safety fences.** Work ONLY on the **`lightning-port`** branch (the sbatch hard-pins it). **NEVER commit to
`main` or `bench-instrumentation`; never push, never `--no-verify`, never `--amend`, never force-push** (all
hook-blocked). **Stage files explicitly** (never `git add -A`; never stage `_scratch/` or data). Rollback uses
`git reset --hard <last-green-commit>` ONLY to discard an **uncommitted** broken WIP back to the last green
floor — never past a green commit.

---

## 0.5. Reference fence — verify present before acting

At orient, confirm each is present and readable; if a **load-bearing reference is ABSENT, STOP**: write a
one-line blocker, fire a `PushNotification` naming the file, do **not** proceed from memory.
- SNFO template: `/home/rmeht/Projects/SNFO/{train.py, modules/train_module.py, data/datamodule.py,
  common/bench_callback.py, environment.yml}`.
- S2S source: `v2.0/{train.py, inference.py, networks/pangu.py, utils/data_loader_multifiles.py, utils/losses.py,
  utils/YParams.py, config/exp2.yaml, config/test.yaml}`.
- The porter agent: `.claude/agents/lightning-porter.md` (its frontmatter + the S2S→Lightning mapping table).
- The Lightning env (§4): `$LPORT_ENV` resolves and `import lightning, torch` succeeds.

---

## 1. Orient (every iteration)

In parallel: `git status --short`; `git log --oneline -10`; confirm `HEAD == lightning-port`. Run the reference
fence (§0.5). Confirm the Lightning env imports (`§4`). Read the porter agent + its mapping table. **Determine
the next unported phase from git** (what is already committed) — do not redo a landed phase.

---

## 2. The deliverable — phases (each looped to a green smoke)

Delegate each phase to the `lightning-porter` agent; dependency order (reuse, don't rebuild):
- **Phase 0 — env + scaffold.** Provision the Lightning env (§4); scaffold the SNFO-style dirs
  (`modules/ data/ configs/ common/`, with `__init__.py` where SNFO has them) **without deleting the `v2.0/`
  originals.** Smoke = `import lightning, torch` on the build node (CPU; no GPU job).
- **Phase 1 — DataModule** (`data/datamodule.py::ClimateDataModule`) wrapping the existing
  `get_data_loader`/`get_infer_data`/`GetDataset`. Resolve the sampler/normalizer/batch-size watch-points from
  the porter agent. Smoke = instantiate + pull one batch.
- **Phase 2 — LightningModule** (`modules/train_module.py::TrainModule`) wrapping `PanguModel_Plasim` + the
  `losses.py` losses; move `cal_loss`'s autocast/7-tuple/CRPS+KL into `training_step`; preserve DDP
  (`DDPStrategy(find_unused_parameters=False, static_graph=True)` + the dead-module freeze), AMP (precision
  mapping), and the S2S_BENCH/NVTX instrumentation. Smoke = 1–2 `fit` steps on `config/test.yaml` (nested gpu:1).
- **Phase 3 — entry points** (root `train.py` mirroring `SNFO/train.py` + config wiring + bench harness via a
  `BenchCallback`).
- **Phase 4 — inference** (`val.py` mirroring `SNFO/val.py`, reusing `get_infer_data` + `inference.py` logic).
- **Phase 5 — reconcile with SNFO** so only the model definition differs.
- **Phase 6 — docstring / reference-integrity pass** (Google-style; verify every cross-reference resolves).

---

## 3. The inner loop (per phase: delegate → verify-green → gauntlet → commit-after-green → sentinel)

1. `TaskCreate` the phase; mark `in_progress`.
2. `Agent` → `lightning-porter`: *"Execute Phase N: <description>. Reuse, don't rewrite. Run the smoke as a
   nested `sbatch` on `pedramh-gpu` (§4) — or the CPU import smoke for Phase 0 — and commit ONLY after the smoke
   is green, with a message citing the smoke job id."*
3. When the porter returns, **VERIFY**: a new commit landed on `lightning-port` AND its message cites a green
   smoke. If unsure, **read the smoke `.out`/`.err` yourself** — never trust an unverified "it passed".
4. Green + committed → run the **light gauntlet** (§5). Gauntlet clean → emit `=== LOOP: STAGE_LANDED ===`.
5. If the porter could not go green in **≤5 attempts**, it should have rolled back to last-green; take the
   blocker branch (§9). **A failed smoke is a real bug, never a loosened test.**

---

## 4. Cluster budget + the Lightning env

- **Orchestrator** (this driver) runs on `--partition=build` (egress for the CLI). **Gate-deciding GPU smokes
  run as nested `sbatch` on `--partition=pedramh-gpu --account=pi-pedramh`** — reuse the existing scripts:
  `v2.0/HPC_scripts/midway_smoke_d2h.sh` (15 min, gpu:1, **no checkpoint** — cheapest), a 1–2-step `fit` on
  `config/test.yaml`, or `midway_bench.sh` (100-step full-path check). Smoke bootstrap = the `midway_bench.sh`
  pattern: `module purge; module load python/miniforge-25.3.0; mamba activate $LPORT_ENV; module load cuda/12.6;`
  the `import torch, wandb` fail-fast guard; `WANDB_MODE=offline`; `PYTHONPATH=v2.0/`.
- **The Lightning env (Phase 0, on this build node — it HAS internet; compute nodes do NOT `pip`).** Build it on
  `/project` here, once. **`LPORT_ENV` = `<<SET ME — e.g. /project/pedramh/shared/S2S/lightning_env>>`.**
  Default approach: clone SNFO's `environment.yml` (it already pins `lightning` 2.x on torch 2.10), or
  `mamba env update` adding `lightning` to a copy of `/project/pedramh/shared/S2S/v2.0/venv`. The env-import
  smoke runs on the build CPU (no GPU job).
- **Persist each nested smoke job id to `_scratch/`**; the sbatch harness polls `squeue`/`sacct` and waits — do
  **not** foreground-`sleep`. Prefer harness-tracked background jobs that re-invoke you on completion.

---

## 5. Light gauntlet (after each green commit)

Spawn via the `Agent` tool, in parallel (use ONLY agents that exist on the cluster checkout — see manifest):
- **`s2s-code-reviewer`** (custom agent — `.claude/agents/s2s-code-reviewer.md` must be present) on the phase
  diff: climate-model + optimization + Lightning correctness, **reuse-not-rewrite**, preserved DDP/AMP/bench
  instrumentation, Google-style docstrings with **no dangling references**, and the **numerical-equivalence gate**
  for any perf change. (Falls back to the built-in `general-purpose` agent if the file is absent.)
- **`drift-auditor`** (custom agent — `.claude/agents/drift-auditor.md` must be present) on docs/CLAUDE.md vs the
  landed change: attribution not inverted, "handoff" terminology, no NGC key, no non-Midway data path.
Any P0/Critical, or a `drift-auditor` fix you have not applied → **re-enter §3** before advancing the green
floor. The green floor never advances with a live P0 or a stale doc.

---

## 6. Branch + rollback

Work on `lightning-port` (per §0). One logical commit per green phase; **commit trailer**
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Rollback-to-last-green via
`git reset --hard <last-green-commit>` only to discard an uncommitted broken WIP. Leave the branch for operator
review; do **not** push or merge.

---

## 7. Bookkeeping (on PORT_COMPLETE)

1. Write/append `LIGHTNING_PORT.md` — per-phase outcomes, the smoke job ids + measured results, the
   S2S→Lightning mapping as landed, any deviations.
2. Memory: a `project_*` entry (the new layout + how to launch Lightning training, what differs from SNFO) + a
   one-line `MEMORY.md` pointer.
3. Run the §5 gauntlet on the final state.

---

## 8. Constraint-edit protocol

Do **not** edit the `CLAUDE.md` hard constraints, the `_optimized` files, or `nvidia_*.sh` (hook-blocked). If a
phase appears to require it, **STOP that phase**, flag with a `PushNotification`, and continue an independent
phase. Do not silently work around a hard constraint.

---

## 9. Blocker protocol

A genuine blocker — a phase that cannot go green after a clean ≤5-attempt re-implementation; a gauntlet P0 you
cannot resolve; a missing reference (§0.5) — **STOP that phase.** Write `blocker.md` (what failed + the smoke
`.out` tail + the exact error), commit only that (`debug: blocker on phase N`), fire a `PushNotification`, and
either continue an **independent** phase or, if all remaining phases depend on the blocked one, emit BLOCKED.
**Never fabricate a pass, never skip the smoke, never loosen a test to escape.**

---

## 10. Loop control & completion signals

The inner loop (§3) iterates a phase to its green smoke; the outer loop walks phases in dependency order;
re-orient at §1 each iteration. When waiting on a nested Slurm smoke, the harness polls — prefer harness-tracked
background jobs that re-invoke you on completion.

**Terminal signal — emit EXACTLY `=== LOOP: <NAME> ===` as a line of its own**, NAME ∈
`STAGE_LANDED | PORT_COMPLETE | BLOCKED`, and write that `=== LOOP:` framing **nowhere else**. At most one per
iteration:
- **`=== LOOP: STAGE_LANDED ===`** — a phase's smoke passed, its commit + the §5 gauntlet landed on
  `lightning-port`, a new last-green floor is set → start a **FRESH** session for the next phase (orient from
  git, §1).
- **`=== LOOP: PORT_COMPLETE ===`** — Phases 0–6 are done: the SNFO-mirrored structure is in place, every phase
  smoke is green, only the model definition differs from SNFO, the docstring/reference pass is clean, and
  `LIGHTNING_PORT.md` + memory reflect it. Stop.
- **`=== LOOP: BLOCKED ===`** — §9 fired and no independent phase remains. Stop.
- Otherwise (mid-work) emit no sentinel; the loop re-fires and you resume from the last green floor.
