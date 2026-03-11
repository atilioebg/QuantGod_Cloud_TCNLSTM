import sys
from pathlib import Path
import json
import logging
import numpy as np

project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from src.cloud.execution.inference_service import InferenceService
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MockPaperTrading")

def run_mock():
    logger.info("🔧 Starting Mock Paper Trading...")
    config_path = project_root / "src/cloud/base_model/configs/master_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger.info("🛡️ Initializing Inference Stack...")
    inference = InferenceService(config)
    
    logger.info("🧪 Generating Mock Data...")
    seq_len = 120
    # Foundation features: 30
    mock_foundation = np.random.randn(seq_len, 30).astype(np.float32)
    
    # Auditor Context features: 14 sensors
    mock_auditor = np.random.randn(1, 14).astype(np.float32)
    
    logger.info("🚀 Running Predict...")
    try:
        result = inference.predict(
            foundation_input=mock_foundation,
            auditor_context=mock_auditor
        )
        logger.info(f"✅ Prediction Success!")
        logger.info(f"📊 Signal: {result['signal']}")
        logger.info(f"🧠 Auditor Score: {result['auditor_score']:.4f}")
        logger.info(f"🎲 Base Probs: {result['probs_foundation']}")
        logger.info(f"🎯 Spec Probs: {result['probs_specialist']}")
    except Exception as e:
        logger.error(f"💥 Prediction Failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_mock()
