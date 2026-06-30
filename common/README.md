# `common/` — shared Lightning-port utilities (Phase 0 scaffold)

This directory mirrors `$SNFO_DIR/common/` (`/project/pedramh/shared/anthonyz/common/`)
as part of the S2S -> PyTorch Lightning port (restructure-in-place; the `v2.0/`
originals are left untouched).

## Target contents (populated in later phases — see the S2S -> Lightning mapping)

- `utils.py` — config + checkpoint helpers, mirrors `common/utils.py`
  (`get_yaml`, `save_yaml`, `dict2namespace`, assemble/disassemble helpers).
- `loss.py` — shared metrics, mirrors `common/loss.py`
  (`latitude_weighted_rmse`). The S2S training losses themselves are
  **reused** from `v2.0/utils/losses.py`, not reimplemented here.
- `plotting.py` — mirrors `common/plotting.py`.
- `bench_callback.py` — `BenchCallback(L.Callback)`, mirrors
  `common/bench_callback.py`; this is where S2S's in-loop `S2S_BENCH` / NVTX
  instrumentation moves onto Lightning hooks (Phase 3).

## Packaging note

SNFO's source packages contain **no `__init__.py`** — they are PEP 420 implicit
namespace packages imported absolutely (`from common.utils import ...`) with the
repo root on the path. This scaffold mirrors that exactly: **do not add
`__init__.py` here.** Run Lightning entry points from the repo root.

## Environment (LPORT_ENV — recorded for later phases)

The known-good interpreter for this port is:

    LPORT_ENV=/project/pedramh/shared/S2S/v2.0/venv

It is S2S's own venv (where `v2.0/networks/pangu.py`, `v2.0/utils/losses.py`,
and the HDF5 loaders run) and already provides `lightning`:
`torch 2.6.0+cu124`, `lightning 2.5.0.post0`, Python 3.11.11. Do **not** clone it
or run a conda/mamba solve. Unifying with SNFO's torch-2.11 / py-3.13 env is a
Phase-5 concern.

Note: `PYTHONPATH=v2.0/` is still required so `from utils...` / `from networks...`
resolve against S2S's existing code (see the S2S hard constraints).

## `data/` coexistence (Phase 0 collision, resolved without clobbering)

The scaffold mirrors SNFO's `modules/`, `data/`, `configs/`, `common/`. Of these,
**`data/` already existed in S2S** as a data-asset folder holding
`data/constant_mask/` (`land_mask.npy`, `soil_type.npy`, `topography.npy`). It is
**left exactly as-is** — not clobbered. SNFO's `data/` is a PEP 420 namespace
package with no `__init__.py`, so in Phase 1 the ported `data/datamodule.py`
(wrapping S2S's existing `get_data_loader` / `get_infer_data` / `GetDataset` from
`v2.0/utils/data_loader_multifiles.py`) can be added **alongside**
`data/constant_mask/` without conflict.

Caveat for later phases: S2S's `.gitignore` has a blanket `data/` rule (the
`constant_mask/*.npy` assets are tracked only because they predate that rule).
New tracked Python sources placed under `data/` in Phase 1 will need an explicit
`git add -f` (or a `.gitignore` carve-out) — a deliberate decision to make then,
not silently in Phase 0. That is why this scaffold ships **no** tracked
`data/README.md`.
