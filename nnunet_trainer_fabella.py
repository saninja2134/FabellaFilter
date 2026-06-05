"""Custom nnU-Net v2 trainer for small-object fabella detection on knee radiographs.

Differences from the default nnUNetTrainer
-------------------------------------------
* Foreground oversampling raised to 0.50 (50 % of patches contain the fabella).
  Default is 0.33; fabella is extremely small relative to the full radiograph, so
  the model needs to see it more often to learn its appearance.
* Loss function replaced with Compound Focal + Dice loss.
  Focal loss down-weights easy background patches and focuses gradient on the hard
  small-object boundary; Dice loss handles class imbalance at the voxel level.
* 1 000 epoch default is preserved (same as base trainer).

Usage
-----
Place this file in your Python path (or the same directory as trainer.py), then
train with:

    nnUNetv2_train DATASET_ID CONFIG fold \
        --trainer nnUNetTrainerFabella \
        -p nnUNetPlans

Example (Dataset001, 2d config, fold 0):

    nnUNetv2_train 1 2d 0 --trainer nnUNetTrainerFabella -p nnUNetPlans
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Focal Loss component
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """Per-voxel sigmoid focal loss for binary segmentation.

    Parameters
    ----------
    alpha : float
        Balancing factor for the foreground class (default 0.25).
    gamma : float
        Focusing exponent. Higher values focus more on hard examples (default 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, *spatial)  targets: (B, 1, *spatial) or same shape
        probs = torch.sigmoid(logits)
        # Binary focal per-class then mean over classes
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Compound Focal + Dice loss
# ---------------------------------------------------------------------------

class CompoundFocalDiceLoss(nn.Module):
    """Weighted sum of BinaryFocalLoss and soft Dice loss.

    Parameters
    ----------
    focal_weight : float
        Contribution of focal loss (default 0.5).
    dice_weight : float
        Contribution of dice loss (default 0.5).
    smooth : float
        Laplace smoothing for Dice denominator (default 1.0).
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        # Flatten spatial dims
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        tgt_flat = targets.view(targets.shape[0], targets.shape[1], -1)
        intersection = (probs_flat * tgt_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + tgt_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(logits, targets)
            + self.dice_weight * self._dice_loss(logits, targets)
        )


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class nnUNetTrainerFabella(nnUNetTrainer):
        """nnU-Net trainer tuned for the small-object fabella detection task.

        Key changes vs the base nnUNetTrainer
        --------------------------------------
        * ``oversample_foreground_percent`` raised to **0.50** — half of every
          training batch will contain a patch that includes the fabella.
        * Loss replaced with a **Compound Focal + Dice loss** to focus learning
          on the hard, tiny foreground boundary and handle extreme class imbalance.
        """

        # Raise foreground oversampling from default 0.33 → 0.50
        oversample_foreground_percent: float = 0.50

        def _build_loss(self):
            return CompoundFocalDiceLoss(
                focal_weight=0.5,
                dice_weight=0.5,
                alpha=0.25,
                gamma=2.0,
                smooth=1.0,
            )

except ImportError:
    # nnunetv2 not installed — silently define a stub so import does not crash
    class nnUNetTrainerFabella:  # type: ignore[no-redef]
        """Stub — install nnunetv2 to use this trainer."""
        pass
