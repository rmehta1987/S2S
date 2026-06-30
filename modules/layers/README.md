# `modules/layers/` — supporting model building blocks (Phase 0 scaffold)

Mirrors `$SNFO_DIR/modules/layers/` (`/project/pedramh/shared/anthonyz/modules/layers/`,
which holds `basics.py`, `conv.py`, `embedding.py`, `patchify.py`, ...).

If S2S's `PanguModel_Plasim` is split into reusable blocks during the port, the
sub-modules from `v2.0/networks/pangu.py` (`EarthSpecificLayer`,
`EarthAttention3D`, patch embed/recover, up/down-sample blocks) land here.
If the model is kept whole inside `modules/models/`, this directory may stay a
thin placeholder. Reuse, do not rewrite.

No `__init__.py` (mirrors SNFO's namespace-package style).
