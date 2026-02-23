#!/bin/bash
# run_pipeline.sh
# Fail-fast execution wrapper for the QuantGod Data Pipeline
# If any script or pytest validation fails, the entire pipeline halts immediately.

set -e # Exit immediately if a command exits with a non-zero status
set -o pipefail # Return value of a pipeline is the value of the last command to exit with a non-zero status

echo "=========================================================="
echo "🚀 STARTING QUANTGOD AUTOMATED PIPELINE (FAIL-FAST MODE) "
echo "=========================================================="

echo "[1/4] Running L2 Pre-Processing (Time-Aware ETL)..."
python src/cloud/base_model/pre_processamento/run_pipeline.py

echo "✅ ETL Complete. Validating outputs..."
echo "[2/4] Running strict ETL Audit via Pytest (12 workers)..."
pytest tests/test_cloud_etl_output.py -v -n 12

echo "✅ ETL Data Validated. Proceeding to Labelling..."
echo "[3/4] Running Labelling (Alpha Targets)..."
python src/cloud/base_model/labelling/run_labelling.py

echo "✅ Labelling Complete. Validating outputs..."
echo "[4/4] Running strict Labelling Audit via Pytest (12 workers)..."
pytest tests/test_labelling_output.py -v -n 12

echo "=========================================================="
echo "🎯 PIPELINE COMPLETE: DATA IS PERFECT AND READY FOR OPTUNA"
echo "=========================================================="
