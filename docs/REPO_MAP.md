# 🗺️ Repo Map (Atlas)

> **Target Audience**: Developers.

A comprehensive map of the QuantGod codebase.

## Root Directory
- `README.md`: The Master Index.
- `requirements.txt`: Python dependencies.
- `pytest.ini`: Configuration for tests.
- `.env`: Environment variables (Secrets).

## Source Code (`src/`)
### Config & Utils
- `src/config.py`: Global settings (Pydantic).
- `src/utils/logger.py`: Centralized logging configuration.

### Data Ingestion (`src/collectors/`)
- `src/collectors/historical.py`: Monthly downloader (Async).
- `src/collectors/stream.py`: Real-time WebSocket manager.

### Data Processing (`src/processing/`)
- `src/processing/tensor_builder.py`: **Core Engine**. Builds 6D Tensors + Wavelet Denoising.
- `src/processing/processor.py`: Meta-Feature Engineering (Pandas TA).
- `src/processing/labeling.py`: Hierarchical Labeling Logic.
- `src/processing/simulation.py`: Reconstructs Order Book from trades.
- `src/processing/normalization.py`: Z-Score and Log transforms.

### Models (`src/models/`)
- `src/models/vivit.py`: **QuantGodViViT** Architecture (CNN + Transformer).

### Training (`src/training/`)
- `src/training/train.py`: Main training loop.
- `src/training/streaming_dataset.py`: Lazy loading of Parquet files.

### Dashboard (`src/dashboard/`)
- `src/dashboard/app.py`: Streamlit Cockpit.

### Live Execution (`src/live/`)
- `src/live/connector.py`: Interface to Exchange API.
- `src/live/predictor.py`: Inference Engine.

### Audit & Debug (`src/audit/`, `src/debug/`)
- `src/audit/check_stream.py`: Health check for WebSocket.
- `src/debug/inspect_batch.py`: Visual debugging of tensors.

## Tests (`tests/`)
- `tests/integration/test_pipeline_6d.py`: End-to-End integration test.
- `tests/test_models/test_vivit.py`: Unit test for Model.
- `tests/test_processing/test_tensor_builder.py`: Unit test for Tensor Builder.
- `tests/test_processing/test_labeling.py`: Unit test for Labeling.
- `tests/test_meta_features.py`: Unit test for Meta-Features.

---
> *Code is Law.* ⚖️
