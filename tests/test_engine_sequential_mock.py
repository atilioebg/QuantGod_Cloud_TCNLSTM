
import pandas as pd
import numpy as np
import yaml
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

# We need to ensure we can import from src.backtest.engine_sequential
# Even if InferenceService is mocked.

from src.backtest.engine_sequential import SequentialBacktestEngine

class TestSequentialEngineMock(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("c:/Users/Atilio/Desktop/PROJETOS/PESSOAL/QuantGod_Cloud_TCNLSTM/tmp/test_sequential_mock")
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.labelled_dir = self.test_dir / "labelled"
        self.labelled_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy master_config
        self.config_dir = self.test_dir / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.master_config = {
            'project_root': str(project_root),
            'model': {
                'feature_names': [f"base_feat_{i}" for i in range(30)],
                'seq_len': 10
            },
            'pipeline_paths': {
                'local_data_root': str(self.test_dir),
                'labelled_dir': str(self.labelled_dir)
            },
            'simulation': {
                'initial_capital': 10000,
                'trading_fee': 0.0006
            }
        }
        
        self.master_path = self.config_dir / "master_config.yaml"
        with open(self.master_path, 'w') as f:
            yaml.dump(self.master_config, f)
            
        self.bt_config_path = self.config_dir / "backtest_config.yaml"
        with open(self.bt_config_path, 'w') as f:
            yaml.dump({'simulation': {'initial_capital': 1000}}, f)

        # Create dummy data
        n_rows = 50
        data = {f"base_feat_{i}": np.random.randn(n_rows).astype(np.float32) for i in range(30)}
        # 14 context features (some with alpha_, some without to test robustness)
        context_features = [
            'ema_trend', 'ema_cross_dist', 'bb_pct', 'rsi_14', 'stoch_14', 
            'atr_norm', 'vol_1h', 'vol_zscore_1h', 'delta_vol_24h',
            'adx_14', 'vwap_zscore', 'mfi_14', 'book_skew_bid', 'book_skew_ask'
        ]
        for i, f in enumerate(context_features):
            if i % 2 == 0:
                data[f] = np.random.randn(n_rows).astype(np.float32)
            else:
                data[f"alpha_{f}"] = np.random.randn(n_rows).astype(np.float32)
        
        data['ts'] = np.arange(1000, 1000 + n_rows * 1000, 1000) # 1s intervals
        data['close'] = np.linspace(100, 110, n_rows)
        data['target'] = np.random.choice([0, 1, 2], n_rows).astype(np.int64)
        
        self.df = pd.DataFrame(data)
        self.df.to_parquet(self.labelled_dir / "test_data.parquet")

    def tearDown(self):
        import shutil
        # if self.test_dir.exists():
        #    shutil.rmtree(self.test_dir)
        pass

    @patch('src.backtest.engine_sequential.InferenceService')
    @patch('src.backtest.engine_sequential.Path')
    def test_engine_logic(self, mock_path_class, mock_inference_class):
        # Setup Mocks
        mock_inference = MagicMock()
        mock_inference.seq_len = 10
        
        n_preds = 50 - 10 + 1
        signals = np.ones(n_preds) * 1 # default neutral
        signals[0] = 2 # BUY at first possible index (idx 9)
        signals[20] = 0 # SELL at index 29 (30 rows later?? no, index in signals is offset)
        
        scores = np.random.rand(n_preds)
        
        mock_inference.predict_batch.return_value = {
            'signals': signals,
            'auditor_scores': scores
        }
        mock_inference_class.return_value = mock_inference
        
        # Patch Path to return our temp master_config when requested
        def side_effect(*args, **kwargs):
            p = Path(*args, **kwargs)
            if str(p).endswith("master_config.yaml"):
                return self.master_path
            return p
        mock_path_class.side_effect = side_effect
        
        # Instantiate Engine
        engine = SequentialBacktestEngine(str(self.bt_config_path))
        
        # Run Backtest
        engine.run_sequential_backtest()
        
        # Verify
        results_path = self.test_dir / "backtest_results_SEQUENTIAL_COMPLETE.parquet"
        self.assertTrue(results_path.exists(), f"Results file should be created at {results_path}")
        
        df_results = pd.read_parquet(results_path)
        print(f"\nExecuted {len(df_results)} trades in mock test.")
        self.assertGreaterEqual(len(df_results), 1, "Should have executed at least one trade")
        
        # Verify first trade was a BUY
        self.assertEqual(df_results.iloc[0]['type'], 'BUY')
        print("✅ Mock test passed!")

if __name__ == "__main__":
    unittest.main()
