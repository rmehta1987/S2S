# `configs/` — Lightning-port configs (Phase 0 scaffold)

Mirrors `$SNFO_DIR/configs/` (`/project/pedramh/shared/anthonyz/configs/`), which
holds plain YAML files loaded with `yaml.safe_load` (via `common/utils.py::get_yaml`)
plus an archived `configs/old/`. SNFO YAML has three top-level keys — `model:`,
`data:`, `training:` — with cluster suffixes (`*_midway.yaml`, `*_NCAR.yaml`).

## Deferred decision (Phase 3 — do NOT resolve in Phase 0)

The config-system convergence is **open**: keep S2S's `v2.0/utils/YParams.py`
+ sectioned YAML (`v2.0/config/*.yaml`, e.g. `exp2.yaml`), or move to SNFO's flat
`get_yaml` + `model`/`data`/`training` split. This directory is scaffolded now;
the actual port configs land in Phase 3 once that decision is made.

The S2S hard constraint stands: configs are cluster-specific and fail deep (not
early) — `data_dir`, `checkpoint_path`, and mean/std `.nc` filenames assume a
specific filesystem. On Midway the HDF5 dataset is `/project/pedramh/h5data/h5data`
(no staging step).

No `__init__.py` (configs are data, not a Python package).
