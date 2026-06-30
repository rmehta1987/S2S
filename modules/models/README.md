# `modules/models/` — ported model definitions (Phase 0 scaffold)

Mirrors `$SNFO_DIR/modules/models/` (`/project/pedramh/shared/anthonyz/modules/models/`,
which holds `AE.py`, `Decoder.py`, `DiT.py`, `Unet.py`).

This is where S2S's `PanguModel_Plasim` (and its blocks) from
`v2.0/networks/pangu.py` will be placed during the port. The model is the **only**
intended material difference between the S2S and SNFO trees — everything else
(entry points, DataModule, LightningModule scaffolding, bench harness, common
utils) converges on SNFO's shape.

Reuse, do not rewrite: the architecture in `v2.0/networks/pangu.py` is held as
`self.model` inside `modules/train_module.py::TrainModule`, not reimplemented.

No `__init__.py` (mirrors SNFO's namespace-package style).
