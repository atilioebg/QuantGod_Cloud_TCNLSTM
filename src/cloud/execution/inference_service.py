import torch
import numpy as np
import xgboost as xgb
import logging
import json
import pickle
import joblib
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

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
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._project_root = Path(__file__).parents[3]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- 1. Path Audit Report (V3.4 Debug) ---
        from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
        virtual_path = get_drive_session_path("MODELOS", self.config)
        resolved_path = resolve_local_project(virtual_path, self._project_root)
        
        logger.info(f"🔍 [V3.4 Audit] Project Root: {self._project_root}")
        logger.info(f"🔍 [V3.4 Audit] Virtual Path: {virtual_path}")
        logger.info(f"🔍 [V3.4 Audit] Resolved Local: {resolved_path}")
        logger.info(f"🔍 [V3.4 Audit] Directory Exists? {resolved_path.exists()}")
        
        # --- 2. Resolve & Lock Paths (V3.5 Authoritative Fix) ---
        if resolved_path.exists():
            self.base_path = resolved_path
            logger.info(f"🎯 [V3.5 Final] Authority Anchor Locked: {self.base_path}")
        else:
            logger.warning(f"⚠️ [V3.5] Audit failed. Falling back to project root.")
            self.base_path = self._project_root
            
        self.arch_params = self._load_best_params()
        
        # Extract metadata
        self.num_features = int(self.arch_params.get('num_features', 97))
        self.seq_len = self.arch_params.get('seq_len', config['optimization'].get('seq_len', 60))
        self.num_classes = 3
        
        # ── Layer 1: Foundation ──────────────────────────────────────────────
        foundation_path = self.config.get('pipeline_paths', {}).get('best_tcn_lstm_model', 'BASE_MODEL/best_tcn_lstm.pt')
        self.foundation_model = self._load_tcn_lstm(str(foundation_path))
        
        # ── Layer 2: Auditor (XGB) ───────────────────────────────────────────
        auditor_path = self.config.get('pipeline_paths', {}).get('auditor_model', 'AUDITOR/auditor_xgboost.json')
        self.auditor = xgb.Booster()
        self.auditor.load_model(str(self.base_path / auditor_path if not Path(auditor_path).is_absolute() else auditor_path))
        
        # ── Layer 3: Scalers ──────────────────────────────────────────────────
        scaler_foundation_path = self.config.get('pipeline_paths', {}).get('scaler_foundation', 'BASE_MODEL/scaler_foundation.pkl')
        scaler_auditor_path = self.config.get('pipeline_paths', {}).get('scaler_auditor', 'AUDITOR/scaler_auditor.pkl')
        
        def _load_p(rel_p):
            p = Path(rel_p)
            abs_p = self.base_path / p if not p.is_absolute() else p
            if not abs_p.exists():
                logger.warning(f"⚠️ Scaler not found: {abs_p}")
                return None
            try:
                # Scalers in this project are dumped via joblib
                return joblib.load(abs_p)
            except Exception as e:
                logger.warning(f"⚠️ Joblib load failed for {abs_p}, trying pickle: {e}")
                with open(abs_p, 'rb') as f:
                    return pickle.load(f)

        self.scaler_foundation = _load_p(scaler_foundation_path)
        self.scaler_auditor = _load_p(scaler_auditor_path)
             
        # ── Layer 4: Specialists (K-Fold) ─────────────────────────────────────
        self.specialist_pairs = self._load_kfold_specialists()
        
        self.threshold = self._load_auditor_threshold()
        
        logger.info(f"✅ Full Inference Stack loaded on {self.device}")
        logger.info(f"🛡️ Active Auditor Threshold: {self.threshold:.4f}")

    def _get_models_dir(self) -> Path:
        """Dynamically resolve models directory using project's native path_utils logic."""
        from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
        
        explicit = self.config.get('execution', {}).get('models_local_dir')
        
        if explicit and "drive:" in str(explicit):
            # Usar a lógica nativa para converter caminho virtual em local da nuvem
            base_dir = get_drive_session_path("MODELOS", self.config)
            p = resolve_local_project(base_dir, self._project_root)
            if p.exists():
                logger.info(f"📁 [V3.0 Native] Resolved models directory: {p}")
                return p.resolve()
            else:
                logger.warning(f"⚠️ Native path resolution failed to find directory: {p}")
        
        # Fallback para caminho literal ou raiz do projeto
        if explicit:
            p = Path(explicit)
            if not p.is_absolute():
                p = self._project_root / p
            if p.exists():
                return p.resolve()
        
        return self._project_root
        # Fallback: dynamic resolution
        from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
        base_dir = get_drive_session_path("MODELOS", self.config)
        return resolve_local_project(base_dir, self._project_root)

    def _load_auditor_threshold(self) -> float:
        """Determines the threshold based on config (dynamic vs manual)."""
        exec_cfg = self.config.get('execution', {})
        # Pull from manual_security_threshold (backtest) or model.auditor.manual_threshold (0.68)
        manual_val = exec_cfg.get('manual_security_threshold')
        if manual_val is None:
            manual_val = self.config.get('model', {}).get('auditor', {}).get('manual_threshold', 0.68)
        
        if exec_cfg.get('dynamic_security_threshold', True):
            # Try to find auditor_config.json in the same folder as the model
            model_path_str = self.config['pipeline_paths'].get('auditor_model', '')
            if model_path_str:
                model_path = self._get_models_dir() / model_path_str
                config_path = model_path.parent / "auditor_config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        auto_val = meta.get('best_threshold')
                        if auto_val:
                            logger.info(f"🤖 Dynamic Threshold detected in file: {auto_val}")
                            return float(auto_val)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to read auditor_config.json: {e}")
            
            logger.info(f"ℹ️ No dynamic threshold file found. Falling back to manual: {manual_val}")
            
        return float(manual_val)

    def _load_best_params(self) -> Dict[str, Any]:
        """Loads the best hyperparameters using native project logic or fallbacks."""
        from src.cloud.base_model.utils.path_utils import get_drive_session_path, resolve_local_project
        
        paths = []
        # 1. Native Dynamic Resolution (Priority V3.0)
        try:
            base_dir = get_drive_session_path("MODELOS", self.config)
            p_base = resolve_local_project(base_dir, self._project_root)
            p_json = p_base / "CONFIG" / "best_params.json"
            paths.append(p_json)
        except Exception:
            pass
            
        # 2. Explicit models_local_dir from config
        explicit = self.config.get('execution', {}).get('models_local_dir')
        if explicit:
            p_base = Path(explicit)
            if not p_base.is_absolute():
                p_base = self._project_root / p_base
            p_json = p_base / "CONFIG" / "best_params.json"
            paths.append(p_json.resolve())
            
        # 3. Final Fallback: production default path
        paths.append(self._project_root / "src/cloud/base_model/otimizacao/best_params.json")
        
        for p in paths:
            if p.exists():
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                    logger.info(f"🎯 Loaded model architecture from {p}")
                    return params
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse {p}: {e}")
        
        logger.error("❌ No best_params.json found! Cannot initialize model architecture.")
        raise FileNotFoundError("Critical model architecture configuration (best_params.json) is missing.")

    def _load_tcn_lstm(self, model_path: str) -> Hybrid_TCN_LSTM:
        """Loads weights and dynamically adjusts architecture if there's a mismatch (V3.7)."""
        p = Path(model_path)
        if not p.is_absolute():
            p = self.base_path / p
            
        if not p.exists():
            logger.error(f"❌ Model file not found: {p}")
            raise FileNotFoundError(f"Model file not found: {p}")
            
        # 1. Inspect checkpoint for architecture (Self-Healing V3.7)
        state_dict = torch.load(p, map_location=self.device)
        
        # We need a copy of arch_params for this specific model instance
        instance_params = self.arch_params.copy()
        
        # Detect TCN Channels
        if 'tcn.0.causal_conv.conv.weight' in state_dict:
            detected_tcn = state_dict['tcn.0.causal_conv.conv.weight'].shape[0]
            if instance_params.get('tcn_channels') != detected_tcn:
                logger.info(f"🔄 Auto-Detect: Adjusted tcn_channels from {instance_params.get('tcn_channels')} to {detected_tcn}")
                instance_params['tcn_channels'] = detected_tcn
        
        # Detect LSTM Hidden
        if 'lstm.weight_ih_l0' in state_dict:
            detected_hidden = state_dict['lstm.weight_ih_l0'].shape[0] // 4
            if instance_params.get('lstm_hidden') != detected_hidden:
                logger.info(f"🔄 Auto-Detect: Adjusted lstm_hidden from {instance_params.get('lstm_hidden')} to {detected_hidden}")
                instance_params['lstm_hidden'] = detected_hidden

        # 2. Build model with corrected blueprint
        # We must filter ONLY architecture-related params for the constructor
        valid_arch_keys = [
            'tcn_channels', 'lstm_hidden', 'num_lstm_layers', 
            'seq_len', 'dropout', 'kernel_size'
        ]
        filtered_params = {k: v for k, v in instance_params.items() if k in valid_arch_keys}
        
        # Ensure correct types for architecture
        for k in ['tcn_channels', 'lstm_hidden', 'num_lstm_layers', 'seq_len']:
            if k in filtered_params:
                filtered_params[k] = int(float(filtered_params[k]))
        
        if 'dropout' in filtered_params:
            filtered_params['dropout'] = float(filtered_params['dropout'])

        model = Hybrid_TCN_LSTM(
            num_features=self.num_features,
            num_classes=self.num_classes,
            **filtered_params
        ).to(self.device)
        
        # 3. Load adjusted state dict
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_foundation_scaler(self) -> Optional[Any]:
        import joblib
        scaler_rel = self.config['pipeline_paths'].get('scaler_foundation', 'BASE_MODEL/scaler_foundation.pkl')
        scaler_path = self._get_models_dir() / scaler_rel
        
        if scaler_path.exists():
            try:
                return joblib.load(scaler_path)
            except Exception as e:
                logger.error(f"⚠️ Failed to load Foundation Scaler: {e}")
        return None

    def _load_kfold_specialists(self) -> List[tuple]:
        """Loads all specialist models and their scalers using relative anchors."""
        import joblib
        
        oof_dir_rel = self.config['pipeline_paths'].get('auditor_oof_dir', 'SPECIALIST')
        # Use our resolved base_path anchor (V3.1 Fix)
        oof_dir = self.base_path / oof_dir_rel
        
        if not oof_dir.exists():
            logger.warning(f"⚠️ Specialist directory not found: {oof_dir}. Check pipeline_paths[auditor_oof_dir].")
            return []
            
        pairs = []
        for i in range(5): # Assuming 5 folds
            m_path = oof_dir / f"model_fold_{i}.pt"
            s_path = oof_dir / f"scaler_fold_{i}.pkl"
            
            if m_path.exists() and s_path.exists():
                model = self._load_tcn_lstm(str(m_path))
                try:
                    scaler = joblib.load(s_path)
                    pairs.append((model, scaler))
                    logger.info(f"🎭 Specialist Pair {i} loaded: {m_path.name} + {s_path.name}")
                except Exception as e:
                    logger.error(f"⚠️ Failed to load Specialist Scaler {i}: {e}")
            else:
                logger.warning(f"⚠️ Specialist Fold {i} model or scaler missing in {oof_dir}")
        
        if not pairs:
            logger.error(f"❌ No Specialist pairs found in {oof_dir}!")
        return pairs

    def _load_auditor_scaler(self) -> Optional[Any]:
        import joblib
        scaler_rel = self.config['pipeline_paths'].get('scaler_auditor', 'AUDITOR/scaler_auditor.pkl')
        scaler_path = self._get_models_dir() / scaler_rel
        
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Auditor Scaler loaded: {scaler_path}")
                return scaler
            except Exception as e:
                logger.error(f"⚠️ Failed to load Auditor Scaler: {e}")
        else:
            logger.warning(f"⚠️ Auditor Scaler NOT found at {scaler_path}. Accuracy will be severely degraded.")
        return None

    @torch.no_grad()
    def predict(self, raw_foundation_input: np.ndarray, auditor_context: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full 3-layer pipeline.
        raw_foundation_input: (seq_len, 30) -- DATA WITHOUT SCALING
        auditor_context: (1, 14) 
        """
        # 1. Foundation Probabilities (Scaled with Foundation Scaler)
        if self.scaler_foundation:
            f_norm = self.scaler_foundation.transform(raw_foundation_input).astype(np.float32)
        else:
            f_norm = raw_foundation_input
            
        x_f = torch.from_numpy(f_norm).unsqueeze(0).to(self.device).float()
        f_out = self.foundation_model(x_f)
        f_probs = f_out['probs'].cpu().numpy()[0] # [p_sell, p_neu, p_buy]
        
        # 2. Specialist Ensemble Probabilities (Each with its OWN Scaler)
        if self.specialist_pairs:
            s_probs_list = []
            for model, scaler in self.specialist_pairs:
                # Normalização Dedicada do Fold (Mathematical Parity)
                s_norm = scaler.transform(raw_foundation_input).astype(np.float32)
                x_s = torch.from_numpy(s_norm).unsqueeze(0).to(self.device).float()
                
                s_out = model(x_s)
                s_probs_list.append(s_out['probs'].cpu().numpy()[0])
            s_probs = np.mean(s_probs_list, axis=0) # Average across folds
        else:
            s_probs = f_probs # Fallback if no specialists
            
        # 3. Construct Auditor Input
        # Format: [base_prob_sell, base_prob_neu, base_prob_buy, spec_prob_sell, spec_prob_neu, spec_prob_buy, context...]
        # Context contains the 14 Alpha Sensors
        auditor_input = np.concatenate([f_probs, s_probs, auditor_context.flatten()]).reshape(1, -1)
        
        # Apply Normalization to Auditor Input (CRITICAL BUG FIX)
        if self.scaler_auditor:
            auditor_input = self.scaler_auditor.transform(auditor_input.astype(np.float32))

        # 4. Auditor Decision (XGBoost)
        dmatrix = xgb.DMatrix(auditor_input)
        auditor_score = self.auditor_model.predict(dmatrix)[0] # Confidence [0-1]
            # Result Determination
        # The primary direction is determined by the specialist ensemble (highest prob)
        direction_idx = np.argmax(s_probs)
        directions = ["SELL", "NEUTRAL", "BUY"]
        raw_direction = directions[direction_idx]
        
        # Final Signal (Audit Check)
        signal = raw_direction if (auditor_score > self.threshold and raw_direction != "NEUTRAL") else "NEUTRAL"
        
        return {
            "signal": signal,
            "direction": raw_direction,
            "auditor_score": float(auditor_score),
            "probs_foundation": f_probs.tolist(),
            "probs_specialist": s_probs.tolist()
        }

    @torch.no_grad()
    def predict_batch(self, x_base_raw: np.ndarray, x_context_raw: np.ndarray, batch_size: int = 4096) -> Dict[str, np.ndarray]:
        """
        TURBO 3.0 INFERENCE - Balanced Stability:
        1. Pre-scaling once (CPU side).
        2. Mixed Precision (AMP FP16) for Tensor Core acceleration.
        3. Contiguous memory copies to minimize PCIe latency.
        """
        import gc
        from numpy.lib.stride_tricks import sliding_window_view
        
        N_total = x_base_raw.shape[0]
        S = self.seq_len
        N_windows = N_total - S + 1
        
        if N_windows <= 0:
            return {"signals": np.array([]), "auditor_scores": np.array([]), "probs_specialist": np.array([])}

        # ── 1. Pre-scaling (Contiguous Float32) ──
        logger.info(f"⚖️ Turbo Scaling Found. ({N_total} bars)...")
        f_scaled = self.scaler_foundation.transform(x_base_raw).astype(np.float32) if self.scaler_foundation else x_base_raw.astype(np.float32)
        s_scaled_list = [pair[1].transform(x_base_raw).astype(np.float32) for pair in self.specialist_pairs]
        
        # ── 2. Optimized Window views ──
        f_win_view = sliding_window_view(f_scaled, window_shape=S, axis=0).transpose(0, 2, 1)
        s_win_views = [sliding_window_view(ss, window_shape=S, axis=0).transpose(0, 2, 1) for ss in s_scaled_list]
        
        all_f_probs = []
        all_s_probs = []
        
        # ── 3. GPU Multi-Model AMP Batch Loop ──
        pbar = tqdm(range(0, N_windows, batch_size), desc="🚀 TURBO GPU", leave=True)
        for start in pbar:
            end = min(start + batch_size, N_windows)
            
            # Use AMP for massive speedup on Ampere+ GPUs (RTX A4500)
            with torch.amp.autocast('cuda'):
                # 1. Foundation (Contiguous slice copy)
                x_f_batch = torch.from_numpy(f_win_view[start:end].copy()).to(self.device, non_blocking=True)
                f_out = self.foundation_model(x_f_batch)
                all_f_probs.append(f_out['probs'].cpu().float().numpy())
                
                # 2. Specialists mean
                fold_probs = []
                for i in range(len(self.specialist_pairs)):
                    x_s_batch = torch.from_numpy(s_win_views[i][start:end].copy()).to(self.device, non_blocking=True)
                    s_out = self.specialist_pairs[i][0](x_s_batch)
                    fold_probs.append(s_out['probs'].cpu().float().numpy())
                
                all_s_probs.append(np.mean(fold_probs, axis=0))
            
        f_probs_all = np.concatenate(all_f_probs, axis=0)
        s_probs_all = np.concatenate(all_s_probs, axis=0)
        
        # Cleanup
        del all_f_probs, all_s_probs, f_win_view, s_win_views
        gc.collect()
        torch.cuda.empty_cache()
        
        # ── 4. Auditor Batch (XGBoost) ──
        x_context_aligned = x_context_raw[S-1:]
        auditor_input = np.concatenate([f_probs_all, s_probs_all, x_context_aligned], axis=1).astype(np.float32)
        
        if self.scaler_auditor:
            auditor_input = self.scaler_auditor.transform(auditor_input)
            
        # Optional: Set XGBoost to use GPU if available (requires gpu_hist support)
        # Note: For small feature sets like 20 features, CPU might be faster due to transfer overhead
        try:
            dmatrix = xgb.DMatrix(auditor_input)
            auditor_scores = self.auditor_model.predict(dmatrix)
        except:
            dmatrix = xgb.DMatrix(auditor_input)
            auditor_scores = self.auditor_model.predict(dmatrix)
        
        # ── 5. Decisions ──
        directions_idx = np.argmax(s_probs_all, axis=1)
        final_signals = np.full(N_windows, 1) # Neutral
        valid_mask = (auditor_scores > self.threshold) & (directions_idx != 1)
        final_signals[valid_mask] = directions_idx[valid_mask]
        
        return {
            "signals": final_signals,
            "auditor_scores": auditor_scores,
            "probs_specialist": s_probs_all
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
