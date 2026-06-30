"""Lightning ``DataModule`` wrapping the S2S HDF5 data loaders.

This module ports the data-loading half of ``v2.0/train.py::Trainer.get_dataset``
onto a :class:`lightning.LightningDataModule`, mirroring the SNFO template at
``$SNFO_DIR/data/datamodule.py``. It **reuses** the existing S2S loaders rather
than reimplementing them: the train and validation loaders are built eagerly in
:meth:`ClimateDataModule.__init__` by calling
:func:`utils.data_loader_multifiles.get_data_loader`, and the underlying
:class:`utils.data_loader_multifiles.GetDataset` (HDF5-backed) is preserved
intact. The inference path (Phase 4) will route through
:func:`utils.data_loader_multifiles.get_infer_data`.

The ``utils.*`` imports resolve only when ``v2.0/`` is on ``PYTHONPATH``
(``PYTHONPATH=v2.0/``), matching the rest of the ported tree.
"""

import lightning as L
import torch

from utils.data_loader_multifiles import (
    get_data_loader,
    get_infer_data,
    GetDataset,
)


class ClimateDataModule(L.LightningDataModule):
    """Wrap S2S's train/val HDF5 loaders as a Lightning ``DataModule``.

    The train and validation loaders are constructed eagerly in
    :meth:`__init__` (matching the SNFO template), so the heavy
    :class:`~utils.data_loader_multifiles.GetDataset` setup — reading dates,
    constant-boundary data, the land mask, and mean/std statistics off the HDF5
    tree at ``params.data_dir`` — happens once at construction time rather than
    in :meth:`setup`.

    Three integration watch-points carried over from the manual
    ``Trainer.get_dataset`` loop are resolved here:

    (a) **Distributed sampler.** S2S's
        :func:`~utils.data_loader_multifiles.get_data_loader` builds its own
        :class:`torch.utils.data.distributed.DistributedSampler` (and returns it
        as the third element of the training tuple), so the prebuilt loader is
        already DDP-correct. Lightning would otherwise inject a *second*
        distributed sampler. The contract is therefore that the entry point
        constructs the ``Trainer`` with ``use_distributed_sampler=False`` (wired
        in Phase 3); this module keeps the loader as-is and retains the sampler
        as :attr:`_train_sampler` so the training loop can call
        ``set_epoch`` on it for correct cross-epoch shuffling.

    (b) **Normalizer.** The training dataset doubles as the normalization /
        statistics source — it holds ``constant_boundary_data``, ``land_mask``,
        and the surface/upper-air/diagnostic/boundary means and stds. It is
        exposed as :attr:`train_dataset` exactly as the SNFO template does, so
        the entry point can pass ``normalizer=datamodule.train_dataset`` into the
        ``LightningModule`` (Phase 2/3).

    (c) **Batch size.** The per-GPU batch size is read directly from
        ``params.batch_size`` by the loader. Unlike the old ``__main__`` in
        ``v2.0/train.py`` (which divided ``batch_size`` by world size before
        constructing loaders), this module does **not** divide: under Lightning
        the config carries the per-GPU batch size and the DDP strategy handles
        scaling across ranks. ``params`` is passed through untouched.

    The ``distributed`` flag handed to
    :func:`~utils.data_loader_multifiles.get_data_loader` reflects the
    process-group state at construction time
    (``torch.distributed.is_available() and torch.distributed.is_initialized()``).
    It is ``False`` in a single-process smoke and ``True`` once Lightning's DDP
    strategy has initialized the process group before the module is built.

    Attributes:
        params: The S2S parameter object (e.g. a
            :class:`utils.YParams.YParams` instance, or any mapping exposing the
            same attribute/item access). Passed through to the loaders.
        train_dataset (GetDataset): The training dataset; also the normalizer /
            statistics source (see watch-point (b)).
        val_dataset (GetDataset): The validation dataset.

    See Also:
        utils.data_loader_multifiles.get_data_loader: Builds the train/val
            loaders reused here.
        utils.data_loader_multifiles.get_infer_data: The inference loader used by
            the Phase 4 inference path.
        utils.data_loader_multifiles.GetDataset: The HDF5-backed dataset wrapped
            by both loaders.
    """

    def __init__(self, params) -> None:
        """Build the train and validation loaders eagerly.

        Args:
            params: S2S parameter object (attribute- and item-accessible, e.g.
                :class:`utils.YParams.YParams`) carrying ``data_dir``,
                ``batch_size``, ``num_data_workers``, the year ranges
                (``train_year_start`` / ``train_year_end`` / ``val_year_start`` /
                ``val_year_end``), ``num_inferences``, and the variable / stats
                configuration consumed by
                :class:`~utils.data_loader_multifiles.GetDataset`.
        """
        super().__init__()
        self.params = params

        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        # Training loader: get_data_loader returns a 3-tuple (loader, dataset,
        # sampler) when train=True. files_pattern == params.data_dir, matching
        # v2.0/train.py::Trainer.get_dataset.
        (
            self._train_loader,
            self.train_dataset,
            self._train_sampler,
        ) = get_data_loader(
            params,
            params.data_dir,
            distributed,
            year_start=params.train_year_start,
            year_end=params.train_year_end,
            train=True,
            num_inferences=0,
            validate=False,
        )

        # Validation loader: get_data_loader returns a 2-tuple (loader, dataset)
        # when train=False. validate=True selects the multi-lead-time validation
        # __getitem__ branch.
        self._val_loader, self.val_dataset = get_data_loader(
            params,
            params.data_dir,
            distributed,
            year_start=params.val_year_start,
            year_end=params.val_year_end,
            train=False,
            num_inferences=params.num_inferences,
            validate=True,
        )

    def prepare_data(self) -> None:
        """Lightning hook for one-time, single-process data preparation.

        No-op: the S2S HDF5 dataset at ``params.data_dir`` is read in place on a
        shared filesystem (no download or staging step), so there is nothing to
        do here. Mirrors the SNFO template.
        """
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook for per-process dataset assignment.

        No-op for every stage: the train and validation datasets are already
        constructed in :meth:`__init__` (matching the SNFO template, which builds
        its loaders eagerly). Kept for interface completeness.

        Args:
            stage: The Lightning stage (``"fit"``, ``"validate"``, ``"test"``, or
                ``"predict"``).
        """
        pass

    def train_dataloader(self):
        """Return the prebuilt training dataloader.

        Returns:
            torch.utils.data.DataLoader: The training loader built in
            :meth:`__init__`, including its manual
            :class:`~torch.utils.data.distributed.DistributedSampler` (see
            watch-point (a) in the class docstring).
        """
        return self._train_loader

    def val_dataloader(self):
        """Return the prebuilt validation dataloader.

        Returns:
            torch.utils.data.DataLoader: The validation loader built in
            :meth:`__init__`.
        """
        return self._val_loader

    def test_dataloader(self):
        """Return the test dataloader.

        Returns:
            None: S2S has no separate test split; mirrors the SNFO template.
        """
        return None

    def predict_dataloader(self):
        """Return the prediction dataloader.

        Returns:
            None: The prediction / inference path is wired in Phase 4 via
            :func:`utils.data_loader_multifiles.get_infer_data`; mirrors the SNFO
            template.
        """
        return None
