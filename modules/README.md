# `modules/` — Lightning modules + ported model (Phase 0 scaffold)

Mirrors `$SNFO_DIR/modules/` (`/project/pedramh/shared/anthonyz/modules/`) for the
S2S -> PyTorch Lightning port (restructure-in-place; `v2.0/` is left untouched).

## Target contents (populated in later phases)

- `train_module.py` — `TrainModule(L.LightningModule)`, mirrors
  `modules/train_module.py`. Holds S2S's `PanguModel_Plasim`
  (from `v2.0/networks/pangu.py`) as `self.model`; instantiates the losses from
  `v2.0/utils/losses.py`; moves `v2.0/train.py::Trainer.cal_loss`'s
  autocast / 7-tuple / CRPS+KL logic into `training_step` (Phase 2).
- `ae_module.py` / `combined_module.py` — only if an S2S analogue is needed;
  these are SNFO-specific (autoencoder / evaluation modules).
- `models/` — where the **ported** S2S architecture lives. SNFO keeps its model
  definitions under `modules/models/` (`DiT.py`, `Unet.py`, ...); S2S's
  `PanguModel_Plasim` and its blocks (`EarthSpecificLayer`, `EarthAttention3D`,
  patch embed/recover) belong here. **This model is the one intended
  S2S<->SNFO difference.**
- `layers/` — supporting building blocks if the model is split out, mirroring
  `modules/layers/`.

## Packaging note

SNFO source packages have **no `__init__.py`** (PEP 420 namespace packages,
absolute imports, run from repo root). Mirror that: **do not add `__init__.py`.**

See `common/README.md` for the LPORT_ENV designation.
