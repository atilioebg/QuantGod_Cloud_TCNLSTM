
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution: ensures that output at time t only depends on
    inputs at times <= t (zero future leakage).
    Achieved by left-padding exactly: padding = dilation * (kernel_size - 1)
    then removing the right side of the output.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        # Remove the right-side padding (causal enforcement)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """
    One TCN block: CausalConv1D → BatchNorm1d → GELU → SpatialDropout.
    Includes optional residual 1x1 conv when channels change.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        # Spatial dropout: drops entire channels (time-step agnostic)
        self.dropout = nn.Dropout(dropout)

        # 1x1 residual projection if channels change
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (B, C_in, T)
        residual = self.residual_proj(x)
        out = self.causal_conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class Hybrid_TCN_LSTM(nn.Module):
    """
    Hybrid TCN + LSTM model for financial time-series classification.

    Architecture:
        Input: (Batch, Seq_Len, Num_Features)     e.g. (B, 720, 9)
        TCN:   Dilated causal convolutions         [1, 2, 4, 8]
        LSTM:  Sequential memory over TCN output
        Head:  MLP → softmax → {logits, probs}

    Output: dict with keys:
        - "logits": raw pre-softmax scores  (B, num_classes)
        - "probs":  calibrated probabilities (B, num_classes)

    Training:
        Use output["logits"] with CrossEntropyLoss(weight=class_weights).
        class_weights are loaded from base_model_config.yaml — not hardcoded here.

    Inference / Meta-feature extraction:
        Use output["probs"] as input to the XGBoost auditor.
    """

    def __init__(
        self,
        num_features: int = 9,
        seq_len: int = 720,
        tcn_channels: int = 64,
        lstm_hidden: int = 256,
        num_lstm_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len

        # ── TCN Stack ─────────────────────────────────────────────────────────
        # Dilations: [1, 2, 4, 8] — receptive field = 1 + (3-1)*(1+2+4+8) = 31 timesteps per stack
        # With 4 dilated blocks, the effective receptive field covers the local structure well.
        dilations = [1, 2, 4, 8]
        tcn_layers = []
        in_ch = num_features
        for i, d in enumerate(dilations):
            out_ch = tcn_channels
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size=3, dilation=d, dropout=dropout * 0.5))
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        # ── LSTM ──────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── Classifier Head ───────────────────────────────────────────────────
        # Strong dropout (0.4) to prevent memorization of temporal patterns
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(lstm_hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, T, F) — e.g. (batch, 720, 9)

        Returns:
            dict:
                "logits": (B, num_classes) — use with CrossEntropyLoss for training
                "probs":  (B, num_classes) — calibrated probabilities for XGBoost input
        """
        # Input: (B, T, num_feats)
        B, T, num_feats = x.shape

        # TCN expects (B, C, T) → channel-first
        x = x.permute(0, 2, 1)           # (B, F, T)
        x = self.tcn(x)                   # (B, tcn_channels, T)
        x = x.permute(0, 2, 1)           # (B, T, tcn_channels)

        # LSTM: full sequence → take last hidden state
        _, (h_n, _) = self.lstm(x)        # h_n: (num_layers, B, hidden)
        last_hidden = h_n[-1]             # (B, hidden) — top-layer final state

        # Classifier
        logits = self.classifier(last_hidden)   # (B, num_classes)
        probs = F.softmax(logits, dim=-1)        # (B, num_classes)

        return {"logits": logits, "probs": probs}
