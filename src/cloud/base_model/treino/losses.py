"""
losses.py — Custom Loss Functions for QuantGod Base Model
==========================================================
Provides:
  - FocalLossWithSmoothing: Focal Loss + Label Smoothing for imbalanced multiclass.
  - compute_alpha_from_labels: Computes inverse-frequency class weights from training targets.

Usage:
    alpha = compute_alpha_from_labels(y_train, num_classes=3, device=DEVICE)
    criterion = FocalLossWithSmoothing(alpha=alpha, gamma=2.0, smoothing=0.1)
    loss = criterion(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss combined with Label Smoothing.

    Focal Loss reduces the contribution of easy (well-classified) examples and
    focuses training on hard (misclassified) examples. Label Smoothing prevents
    overconfident predictions that cause Val Loss to explode despite good F1.

    Args:
        alpha  : Optional tensor of shape (num_classes,) with per-class weights
                 (inverse frequency). If None, all classes are weighted equally.
        gamma  : Focusing parameter. Higher values reduce easy-example contribution
                 more aggressively. Recommended: 2.0.
        smoothing: Label smoothing factor in [0.0, 1.0). Recommended: 0.1.
        reduction: 'mean' (default) | 'sum' | 'none'.

    Mathematical formulation:
        CE_loss = cross_entropy(logits, targets, weight=alpha, label_smoothing=smoothing)
        pt = exp(-CE_loss)                      # probability of correct class
        focal_loss = (1 - pt)^gamma * CE_loss   # suppress easy examples

    The label_smoothing and alpha (weight) are handled inside PyTorch's
    F.cross_entropy for numerical stability. We only apply the focal modulation.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, num_classes) raw logits from model.
            targets : (B,) integer class indices.

        Returns:
            Scalar focal loss (if reduction='mean'|'sum') or (B,) tensor.
        """
        # PyTorch applies label_smoothing and alpha (weight) internally with
        # numerical stability (log-sum-exp trick). reduction='none' gives per-sample loss.
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            label_smoothing=self.smoothing,
            reduction='none',
        )

        # pt: estimated probability of the correct class under the CE loss.
        # exp(-CE) ≈ p_correct when no smoothing; stable approximation with smoothing.
        pt = torch.exp(-ce_loss)

        # Apply focal modulation: down-weight easy examples (pt → 1), keep hard ones.
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_alpha_from_labels(
    y: np.ndarray,
    num_classes: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute per-class weights using the balanced inverse-frequency formula:

        alpha_i = total_samples / (num_classes * class_count_i)

    This is the standard sklearn 'balanced' strategy, which:
    - Makes weight(Neutral) drop below 1.0  (e.g. ~0.43 for 78% neutral)
    - Makes weight(SELL/BUY) rise above 1.0  (e.g. ~3.03 for 11% each)

    Args:
        y          : 1-D numpy array of integer class labels from TRAINING SET only.
        num_classes: Total number of classes (default 3: SELL=0, NEU=1, BUY=2).
        device     : torch.device to place the tensor on.

    Returns:
        Tensor of shape (num_classes,) with dtype float32, on `device`.
    """
    total = len(y)
    alpha_list = []
    for c in range(num_classes):
        count = int(np.sum(y == c))
        if count == 0:
            # Avoid division by zero for unseen classes; assign weight=1.0
            alpha_list.append(1.0)
        else:
            alpha_list.append(total / (num_classes * count))

    alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32)
    if device is not None:
        alpha_tensor = alpha_tensor.to(device)
    return alpha_tensor
