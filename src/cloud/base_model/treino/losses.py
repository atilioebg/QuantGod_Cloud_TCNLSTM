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
        use_sniper: bool = False,
        sniper_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.use_sniper = use_sniper
        self.sniper_weight = sniper_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, num_classes) raw logits from model.
            targets : (B,) integer class indices.

        Returns:
            Scalar focal loss (if reduction='mean'|'sum') or (B,) tensor.
        """
        # PyTorch applies label_smoothing and alpha (weight) internally
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            label_smoothing=self.smoothing,
            reduction='none',
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        # ── Sniper Loss Extension: Penalize directional inversions ────────────
        if self.use_sniper:
            # We assume classes are ordered: 0=SELL, 1=NEUTRAL, 2=BUY
            # Erring from 2 to 0 (SELL instead of BUY) is much worse than 2 to 1.
            probs = F.softmax(logits, dim=1)
            num_classes = logits.size(1)
            
            # Create a distance matrix: (indices - target)^2
            indices = torch.arange(num_classes, device=logits.device).float() # [0, 1, 2]
            t = targets.view(-1, 1).float()                                   # (B, 1)
            
            # Squared distance penalizes opposite ends much harder (2^2=4 vs 1^2=1)
            dist_sq = (indices - t) ** 2                                     # (B, num_classes)
            
            # Directional penalty is the weighted sum of probs based on distance
            directional_penalty = torch.sum(probs * dist_sq, dim=1)           # (B,)
            
            focal_loss = focal_loss + (self.sniper_weight * directional_penalty)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss: treats each class (SELL, NEUTRAL, BUY) with independent focal focus.
    
    This 3-way split allows the optimizer to discover different 'difficulty' levels 
    for each market direction.
    
    Args:
        alpha: (3,) Tensor of class weights (inverse frequency).
        gammas: (3,) Tensor of focal focusing parameters [gamma_sell, gamma_neu, gamma_buy].
        smoothing: Label smoothing factor.
    """
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gammas: torch.Tensor | None = None,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gammas = gammas if gammas is not None else torch.tensor([2.0, 2.0, 2.0])
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross Entropy with Label Smoothing and Class Weights
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, 
            label_smoothing=self.smoothing, reduction='none'
        )
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Map per-sample gamma based on target class indices [0, 1, 2]
        # self.gammas must be on the same device as targets
        batch_gammas = self.gammas.to(targets.device)[targets]
        
        # Apply asymmetric focal modulation
        asym_focal_loss = ((1.0 - pt) ** batch_gammas) * ce_loss
        
        if self.reduction == 'mean':
            return asym_focal_loss.mean()
        elif self.reduction == 'sum':
            return asym_focal_loss.sum()
        return asym_focal_loss


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
