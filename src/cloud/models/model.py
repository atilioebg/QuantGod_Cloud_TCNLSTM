import torch
import torch.nn as nn
import math

class QuantGodModel(nn.Module):
    """
    Time-Series Transformer for Tabular Data (L2 Features).
    Input: (Batch, Seq_Len, Num_Features)
    Output: (Batch, Num_Classes)
    """
    def __init__(self, num_features, seq_len, d_model=128, nhead=4, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Feature Projection (Input -> d_model)
        self.input_proj = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Num_Features)
        b, t, f = x.shape
        
        # Projection
        x = self.input_proj(x) # (B, T, d_model)
        
        # Add Positional Encoding
        # Allow handling sequences shorter than max seq_len if needed (though usually fixed)
        if t <= self.pos_embedding.shape[1]:
             x = x + self.pos_embedding[:, :t, :]
        else:
             # Basic handling, though robust training ensures t <= max
             x = x + self.pos_embedding[:, :self.pos_embedding.shape[1], :]
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Take the last token state
        last_state = x[:, -1, :]
        
        # Classify
        logits = self.classifier(last_state)
        
        return logits
