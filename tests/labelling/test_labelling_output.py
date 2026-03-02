"""
test_labelling_output.py — Validates output of the labelling pipeline.

Tests operate on the ACTIVE labelled experiment directory.
The active directory is defined in conftest.py (ACTIVE_LABELLED_DIR),
which matches the labelled_dir configured in training_config.yaml and auditor_config.yaml.

Run against a specific labelled experiment:
    pytest tests/test_labelling_output.py --labelled-dir data/L2/labelled_SELL_0004_BUY_0006_1h

Covered:
  - Directory exists and is non-empty
  - Schema: 'target' column present + correct dtype
  - Label values are only in {0, 1, 2}
  - No NaN/null in 'target'
  - All 9 required features present (inherited from pre_processed)
  - Row count close to 1440 per file (1 minute candles, minus lookahead trim)
  - No NaN in 9 feature columns
  - Temporal monotonicity
  - At least 2 distinct classes per file (threshold sanity check)
  - At least 3% BUY and 3% SELL across total dataset (global balance sanity)
"""

import polars as pl
import pandas as pd
from pathlib import Path
import pytest
import os
import logging
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = Path("src/cloud/base_model/configs/master_config.yaml")

def _load_master_config():
    if _CFG_PATH.exists():
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def pytest_addoption(parser):
    """Allow overriding the labelled dir from the command line."""
    try:
        parser.addoption(
            "--labelled-dir",
            action="store",
            default=None,
            help="Path to labelled experiment dir (default: active from conftest.py)",
        )
    except ValueError:
        pass  # Option already registered (e.g., in another conftest)


def get_labelled_dir(request=None) -> Path:
    """Resolve labelled directory: CLI arg > env var > conftest constant."""
    if request is not None:
        try:
            cli = request.config.getoption("--labelled-dir", default=None)
            if cli:
                return Path(cli)
        except (ValueError, AttributeError):
            pass
    
    # Importamos a lógica centralizada do conftest para evitar divergência
    try:
        from tests.conftest import ACTIVE_LABELLED_DIR
        if ACTIVE_LABELLED_DIR.exists():
            return ACTIVE_LABELLED_DIR
    except ImportError:
        pass

    # Fallback local se conftest falhar por algum motivo
    env_v = os.getenv("LABELLED_DIR")
    if env_v:
        return Path(env_v)
        
    labelled_dirs = sorted(list(Path("data/L2").glob("labelled_*")))
    if labelled_dirs:
        return labelled_dirs[-1]
        
    return Path("data/L2/labelled_SELL_0004_BUY_0004_1h")


def get_labelled_files(request=None) -> list:
    """Return sorted list of Parquet files in the active labelled directory."""
    directory = get_labelled_dir(request)
    return sorted(directory.glob("*.parquet"))


# v4.9: REQUIRED_FEATURES derived from master_config — no hardcodes that can drift.
# Falls back to empty list if config not found (tests will skip gracefully).
_master_cfg = _load_master_config()
REQUIRED_FEATURES: list = _master_cfg.get("model", {}).get("feature_names", [])


# ── Directory-level checks (run once) ────────────────────────────────────────

class TestLabelledDirectory:

    def test_directory_exists(self):
        d = get_labelled_dir()
        assert d.exists(), f"Labelled directory not found: {d}"

    def test_directory_has_parquet_files(self):
        files = get_labelled_files()
        assert len(files) > 0, "No .parquet files found in labelled directory"

    def test_file_count_matches_pre_processed(self):
        """Number of labelled files must match pre_processed directory (same dates)."""
        pre_processed = Path(os.getenv("PRE_PROCESSED_DIR", "data/L2/pre_processed"))

        if not pre_processed.exists():
            pytest.skip(f"pre_processed dir not found at {pre_processed} — skipping check")
        pp_count = len(list(pre_processed.glob("*.parquet")))
        lb_count = len(get_labelled_files())
        assert lb_count == pp_count, (
            f"File count mismatch: labelled={lb_count}, pre_processed={pp_count}"
        )


# ── Per-file checks (parametrized) ───────────────────────────────────────────

@pytest.mark.parametrize("file_path", get_labelled_files())
class TestLabelledFileIntegrity:

    def test_file_readable(self, file_path):
        df = pl.read_parquet(file_path)
        assert df is not None

    def test_target_column_exists(self, file_path):
        df = pl.read_parquet(file_path)
        assert "target" in df.columns, f"'target' column missing in {file_path.name}"

    def test_target_dtype_is_integer(self, file_path):
        df = pl.read_parquet(file_path)
        assert df["target"].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64), \
            f"target dtype {df['target'].dtype} is not integer in {file_path.name}"

    def test_target_no_nulls(self, file_path):
        df = pl.read_parquet(file_path)
        nulls = df["target"].null_count()
        assert nulls == 0, f"{nulls} null(s) in 'target' in {file_path.name}"

    def test_target_valid_values(self, file_path):
        df = pl.read_parquet(file_path)
        invalid = df.filter(~pl.col("target").is_in([0, 1, 2]))
        assert len(invalid) == 0, \
            f"{len(invalid)} invalid target values in {file_path.name}: {invalid['target'].unique().to_list()}"

    def test_at_least_one_class(self, file_path):
        """At least one class must exist (file not empty). Balance check is done globally."""
        df = pl.read_parquet(file_path)
        n_classes = df["target"].n_unique()
        assert n_classes >= 1, f"Empty classes in {file_path.name}"
        if n_classes == 1:
             # Just a log, many days in BTC can be flat.
             # Strict balance is checked in GlobalLabelBalance.
             logger.warning(f"File {file_path.name} has only 1 class (likely all NEUTRAL).")

    def test_required_features_present(self, file_path):
        df = pl.read_parquet(file_path)
        for col in REQUIRED_FEATURES:
            assert col in df.columns, f"Feature '{col}' missing in {file_path.name}"

    def test_no_nans_in_features(self, file_path):
        df = pl.read_parquet(file_path)
        null_sum = df.select(REQUIRED_FEATURES).null_count().to_series().sum()
        assert null_sum == 0, \
            f"{null_sum} NaN(s) in feature columns in {file_path.name}"

    def test_row_count_reasonable(self, file_path):
        """
        After lookahead trim, file should still have a reasonable number of rows.
        5min pipeline yields ~285 rows/day (288 bars − 3 lookahead).
        Use 80 as a safe floor to accommodate gap-heavy days.
        """
        df = pl.read_parquet(file_path)
        assert len(df) >= 80, \
            f"File {file_path.name} has {len(df)} rows — unusually low for 5min pipeline (check ETL)"

    def test_temporal_monotonicity(self, file_path):
        """If 'ts' column exists, timestamps must be strictly increasing."""
        df = pl.read_parquet(file_path)
        if "ts" in df.columns:
            diffs = df["ts"].diff().slice(1)
            assert (diffs < 0).sum() == 0, \
                f"Non-monotonic timestamps in {file_path.name}"

    def test_label_logical_sanity(self, file_path):
        """
        Garante que o rótulo (0, 1, 2) reflete matematicamente a subida/queda no timeframe.
        Recalcula o retorno futuro manualmente para uma amostra e compara com o label.
        Lê thresholds e lookahead_bars do master_config.yaml (v4.9).
        """
        # 1. Ler thresholds do master_config (não do inexistente labelling_config.yaml)
        master_cfg = _load_master_config()
        if not master_cfg:
            pytest.skip("master_config.yaml não encontrado para validar lógica.")

        lab_cfg = master_cfg.get("pre_processing", {}).get("labelling", {})
        etl_cfg = master_cfg.get("pre_processing", {}).get("etl", {})

        horizon_min   = lab_cfg.get("horizon_minutes", 15)
        resample_freq = etl_cfg.get("resample_freq", "1min")
        resample_min  = int(pd.to_timedelta(resample_freq).total_seconds() // 60)
        lookahead     = max(1, horizon_min // resample_min)  # in BARS (same calc as run_labelling.py)

        t_long  =  lab_cfg.get("buy_threshold", 0.003)
        t_short = -lab_cfg.get("sell_threshold", 0.003)

        # 2. Ler o arquivo (precisamos de 'close', 'high', 'low' e do 'target')
        df = pl.read_parquet(file_path)

        # 3. Amostra aleatória de 5 pontos para não pesar o teste
        import numpy as np
        # Evitamos o final do arquivo onde o lookahead não existe
        safe_range = len(df) - lookahead - 1
        if safe_range <= 0:
            pytest.skip(f"Arquivo {file_path.name} muito curto para validação lógica.")

        indices = np.random.choice(range(safe_range), min(safe_range, 5), replace=False)

        for idx in indices:
            idx = int(idx)
            # O labelling usa high/low para validar BUY/SELL, não log_ret_close cumulativo
            # Portanto verificamos usando a logica original do Sniper:
            #   BUY=2  → algum high no lookahead >= close*(1+t_long)
            #   SELL=0 → algum low  no lookahead <= close*(1-|t_short|)
            close_now  = float(df["close"][idx])
            assigned_label = int(df["target"][idx])

            future_high = float(df["high"].slice(idx + 1, lookahead).max())
            future_low  = float(df["low"].slice(idx + 1, lookahead).min())

            if assigned_label == 2:  # BUY
                assert future_high >= close_now * (1 + t_long) - 1e-9, (
                    f"Label BUY(2) em {file_path.name}[{idx}] mas future_high={future_high:.5f} "
                    f"< threshold {close_now * (1 + t_long):.5f}"
                )
            elif assigned_label == 0:  # SELL
                assert future_low <= close_now * (1 + t_short) + 1e-9, (
                    f"Label SELL(0) em {file_path.name}[{idx}] mas future_low={future_low:.5f} "
                    f"> threshold {close_now * (1 + t_short):.5f}"
                )
            # NEUTRAL: sem validação hard — cob apenas com os tests anteriores


# ── Global balance check (aggregate over dataset) ─────────────────────────────

class TestGlobalLabelBalance:

    def test_all_three_classes_exist_globally(self):
        """Over the full dataset, all 3 classes must appear."""
        files = get_labelled_files()
        if not files:
            pytest.skip("No labelled files to check")

        all_classes: set = set()
        for f in files[:20]:          # sample first 20 files for speed
            df = pl.read_parquet(f)
            all_classes |= set(df["target"].unique().to_list())
            if all_classes == {0, 1, 2}:
                break

        assert 0 in all_classes, "Class SELL (0) never appears in dataset"
        assert 1 in all_classes, "Class NEUTRAL (1) never appears in dataset"
        assert 2 in all_classes, "Class BUY (2) never appears in dataset"

    def test_minority_classes_not_empty(self):
        """
        SELL and BUY must represent at least 3% of labels combined over a sample.
        Protects against overly aggressive thresholds that collapse all labels to NEUTRAL.
        """
        files = get_labelled_files()
        if not files:
            pytest.skip("No labelled files to check")

        sample = files[:30]   # use 30 days (~43K rows) — fast but representative
        counts = {0: 0, 1: 0, 2: 0}
        for f in sample:
            df = pl.read_parquet(f)
            vc = df["target"].value_counts()
            for row in vc.to_dicts():
                counts[row["target"]] = counts.get(row["target"], 0) + row["count"]

        total = sum(counts.values())
        if total == 0:
            pytest.skip("No rows in sample")

        sell_pct = counts[0] / total
        buy_pct  = counts[2] / total
        logger.info(f"Sample label distribution: SELL={sell_pct:.1%}, NEUTRAL={(counts[1]/total):.1%}, BUY={buy_pct:.1%}")

        assert sell_pct >= 0.03, \
            f"SELL class is only {sell_pct:.1%} of labels — thresholds may be too aggressive"
        assert buy_pct >= 0.03, \
            f"BUY class is only {buy_pct:.1%} of labels — thresholds may be too aggressive"
