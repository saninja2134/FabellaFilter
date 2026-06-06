"""Custom nnU-Net v2 trainer for small-object fabella detection on knee radiographs.

nnU-Net's CLI only discovers trainers that live inside the installed package at
``nnunetv2/training/nnUNetTrainer/``.  The training pipeline in ``trainer.py``
automatically copies this file there (renaming it ``nnUNetTrainerFabella.py``)
before invoking ``nnUNetv2_train``.  You can also install it manually:

    cp nnunet_trainer_fabella.py \\
       $(python -c "import nnunetv2.training.nnUNetTrainer as p; import os; print(os.path.dirname(p.__file__))")/nnUNetTrainerFabella.py

Differences from nnUNetTrainer
--------------------------------
* ``oversample_foreground_percent`` raised to **0.50** — half of every mini-batch
  will contain a patch that includes the fabella (default 0.33).
* Loss replaced with the built-in **DC_and_Focal_loss** (Dice + Focal, α=0.25, γ=2).

Recommended training command
------------------------------
    nnUNetv2_train 1 2d 0 -tr nnUNetTrainerFabella -p nnUNetResEncUNetMPlans --npz
"""

from __future__ import annotations

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.loss.compound_losses import DC_and_Focal_loss
    from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

    class nnUNetTrainerFabella(nnUNetTrainer):
        """nnU-Net trainer tuned for tiny-object fabella segmentation.

        Key changes
        -----------
        ``oversample_foreground_percent = 0.50``
            50 % of every training batch will contain the fabella region.
            The default of 0.33 is too low for such a rare, tiny target.

        ``_build_loss`` → ``DC_and_Focal_loss``
            Uses nnU-Net's built-in compound Dice + Focal loss. Focal loss
            (α=0.25, γ=2) concentrates gradient on hard boundary voxels and
            suppresses the overwhelming easy-background signal.
        """

        # Raise foreground oversampling from default 0.33 → 0.50
        oversample_foreground_percent: float = 0.50

        def _build_loss(self):
            loss = DC_and_Focal_loss(
                # Soft Dice component
                soft_dice_kwargs={
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                # Focal component: alpha=0.25 down-weights easy background,
                # gamma=2 focuses gradient on hard foreground boundaries
                focal_kwargs={
                    "alpha": 0.25,
                    "gamma": 2.0,
                    "smooth": 1e-5,
                    "do_bg": False,
                },
                weight_ce=1.0,
                weight_dice=1.0,
                ignore_label=self.label_manager.ignore_label,
            )
            return loss

except ImportError:
    # nnunetv2 not installed — silently define a stub so import does not crash
    class nnUNetTrainerFabella:  # type: ignore[no-redef]
        """Stub — install nnunetv2 to use this trainer: pip install nnunetv2"""
        pass
