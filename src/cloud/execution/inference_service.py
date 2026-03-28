import torch
import numpy as np
import xgboost as xgb
import logging
import json
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
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._project_root = Path(__file__).parents[3]
        
        # ── Load Architecture ────────────────────────────────────────────────
        self.arch_params = self._load_best_params()
        
        # Load Model Parameters
        self.num_features = len(config['model']['feature_names'])
        self.seq_len = self.arch_params.get('seq_len', config['optimization'].get('seq_len', 60))
        self.num_classes = 3 # Sell, Neutral, Buy
        
        # ── Layer 1: Foundation ──────────────────────────────────────────────
        models_dir = self._get_models_dir()
        foundation_path = models_dir / self.config['pipeline_paths']['best_tcn_lstm_model']
        self.foundation_model = self._load_tcn_lstm(str(foundation_path))
        self.scaler_foundation = self._load_foundation_scaler()
        
        # ── Layer 2: Specialist Ensemble ─────────────────────────────────────
        # Now returns a list of (model, scaler) tuples
        self.specialist_pairs = self._load_kfold_specialists()
        
        # ── Layer 3: Auditor ──────────────────────────────────────────────────
        self.auditor_model = self._load_auditor()
        self.threshold = self._load_auditor_threshold()
        self.scaler_auditor = self._load_auditor_scaler()
        
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
        # Load hyperparams strictly from the global arch_params (Shared Identity)
        p = Path(model_path)
        if not p.is_absolute():
            # Use the dynamically resolved base_path (V3.1 Fix)
            p = self.base_path / p
            
        if not p.exists():
            logger.error(f"❌ Model file not found: {p}")
            raise FileNotFoundError(f"Model file not found: {p}")

        try:
            model = Hybrid_TCN_LSTM(
                num_features=self.num_features,
                seq_len=self.arch_params.get('seq_len', self.seq_len),
                tcn_channels=int(self.arch_params['tcn_channels']),
                lstm_hidden=int(self.arch_params['lstm_hidden']),
                num_lstm_layers=int(self.arch_params['num_lstm_layers']),
                num_classes=self.num_classes,
                dropout=float(self.arch_params.get('dropout', 0.3))
            ).to(self.device)
        except KeyError as e:
            logger.error(f"❌ Missing required architecture parameter: {e}")
            raise KeyError(f"Missing required architecture parameter in best_params.json: {e}")
        
        state_dict = torch.load(p, map_location=self.device)
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
