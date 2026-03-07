import torch
import numpy as np
import xgboost as xgb
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Internal imports
from src.cloud.base_model.models.model import Hybrid_TCN_LSTM

logger = logging.getLogger(__name__)

class InferenceService:
    """
    3-Layer Inference Stack:
    1. Foundation Layer (TCN-LSTM)
    2. Specialist Ensemble (5-Fold TCN-LSTM)
    3. Auditor Layer (XGBoost Meta-Model)
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ── Load Architecture ────────────────────────────────────────────────
        self.arch_params = self._load_best_params()
        
        # Load Model Parameters
        self.num_features = len(config['model']['feature_names'])
        self.seq_len = self.arch_params.get('seq_len', config['optimization'].get('seq_len', 60))
        self.num_classes = 3 # Sell, Neutral, Buy
        
        # ── Layer 1: Foundation ──────────────────────────────────────────────
        self.foundation_model = self._load_tcn_lstm(
            self.config['pipeline_paths']['best_tcn_lstm_model']
        )
        
        # ── Layer 2: Specialist Ensemble ─────────────────────────────────────
        self.specialist_models = self._load_kfold_specialists()
        
        # ── Layer 3: Auditor ──────────────────────────────────────────────────
        self.auditor_model = self._load_auditor()
        
        logger.info(f"✅ Full Inference Stack loaded on {self.device}")

    def _load_best_params(self) -> Dict[str, Any]:
        """Loads the best hyperparameters found by Optuna."""
        # Check both Macro and Directional variants, prefer Macro as it's the primary study objective
        paths = [
            Path("src/cloud/base_model/otimizacao/best_params.json"),
            Path("src/cloud/base_model/otimizacao/best_dir_params.json")
        ]
        
        for p in paths:
            if p.exists():
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                    logger.info(f"🎯 Loaded model architecture from {p}")
                    return params
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse {p}: {e}")
        
        logger.warning("⚠️ No best_params.json found. Using defaults from master_config/class defaults.")
        # Return common defaults if missing (fallback based on user's manual fix or training context)
        return {
            "tcn_channels": 256, # Derived from user error
            "lstm_hidden": 64,   # Derived from user error
            "num_lstm_layers": 2,
            "dropout": 0.3
        }

    def _load_tcn_lstm(self, model_path: str) -> Hybrid_TCN_LSTM:
        # Resolve hyperparams from JSON or fallback to class defaults
        model = Hybrid_TCN_LSTM(
            num_features=self.num_features,
            seq_len=self.seq_len,
            tcn_channels=self.arch_params.get('tcn_channels', 64),
            lstm_hidden=self.arch_params.get('lstm_hidden', 256),
            num_lstm_layers=self.arch_params.get('num_lstm_layers', 2),
            num_classes=self.num_classes,
            dropout=self.arch_params.get('dropout', 0.3)
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_kfold_specialists(self) -> List[Hybrid_TCN_LSTM]:
        # We look for model_fold_k.pt in the OOF directory
        oof_dir = Path(self.config.get('pre_processing', {}).get('kfold', {}).get('oof_output_dir', 'data/auditor/oof_predictions'))
        models = []
        
        for i in range(5): # Assuming 5 folds
            path = oof_dir / f"model_fold_{i}.pt"
            if path.exists():
                models.append(self._load_tcn_lstm(str(path)))
            else:
                logger.warning(f"⚠️ Specialist Fold {i} model not found at {path}")
        
        if not models:
            logger.error("❌ No Specialist models found! Paper Trading will likely fail Auditor stage.")
        return models

    def _load_auditor(self) -> xgb.Booster:
        model_path = self.config['pipeline_paths'].get('auditor_model', 'data/auditor/auditor_xgboost.json')
        bst = xgb.Booster()
        bst.load_model(model_path)
        return bst

    @torch.no_grad()
    def predict(self, foundation_input: np.ndarray, auditor_context: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full 3-layer pipeline.
        foundation_input: (seq_len, 32)
        auditor_context: (1, 14) 
        """
        # (1, seq_len, 32)
        x = torch.from_numpy(foundation_input).unsqueeze(0).to(self.device).float()
        
        # 1. Foundation Probabilities
        f_out = self.foundation_model(x)
        f_probs = f_out['probs'].cpu().numpy()[0] # [p_sell, p_neu, p_buy]
        
        # 2. Specialist Ensemble Probabilities
        if self.specialist_models:
            s_probs_list = []
            for m in self.specialist_models:
                s_out = m(x)
                s_probs_list.append(s_out['probs'].cpu().numpy()[0])
            s_probs = np.mean(s_probs_list, axis=0) # Average across folds
        else:
            s_probs = f_probs # Fallback if no specialists
            
        # 3. Construct Auditor Input
        # Format: [base_prob_sell, base_prob_neu, base_prob_buy, spec_prob_sell, spec_prob_neu, spec_prob_buy, context...]
        # Context contains the 14 Alpha Sensors
        auditor_input = np.concatenate([f_probs, s_probs, auditor_context.flatten()]).reshape(1, -1)
        
        # 4. Auditor Decision (XGBoost)
        dmatrix = xgb.DMatrix(auditor_input)
        auditor_score = self.auditor_model.predict(dmatrix)[0] # Confidence [0-1]
        
        # Result Determination
        # The primary direction is determined by the specialist ensemble (highest prob)
        direction_idx = np.argmax(s_probs)
        directions = ["SELL", "NEUTRAL", "BUY"]
        raw_direction = directions[direction_idx]
        
        # Final Signal (Audit Check)
        threshold = 0.92 # Recommended in training analysis
        signal = raw_direction if (auditor_score > threshold and raw_direction != "NEUTRAL") else "NEUTRAL"
        
        return {
            "signal": signal,
            "direction": raw_direction,
            "auditor_score": float(auditor_score),
            "probs_foundation": f_probs.tolist(),
            "probs_specialist": s_probs.tolist()
        }

if __name__ == "__main__":
    # Test loading
    from src.cloud.base_model.utils.config_utils import load_config
    cfg = load_config()
    try:
        service = InferenceService(cfg)
        print("InferenceService Loaded Successfully.")
    except Exception as e:
        print(f"Error loading InferenceService: {e}")
