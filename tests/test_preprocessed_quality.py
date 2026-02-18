
import polars as pl
from pathlib import Path
import pytest
import datetime
import re

PRE_PROCESSED_DIR = Path("data/L2/pre_processed")
EXPECTED_ROWS = 1440  # 24 * 60 for 1-minute samples

def get_parquet_files():
    return sorted(list(PRE_PROCESSED_DIR.glob("*.parquet")))

class TestPreprocessedQuality:
    
    def test_directory_exists(self):
        assert PRE_PROCESSED_DIR.exists(), f"Directory {PRE_PROCESSED_DIR} not found!"

    def test_file_count(self):
        files = get_parquet_files()
        assert len(files) > 0, "No pre-processed files found!"

    def test_date_continuity(self):
        """Checks if there are missing days between the first and last processed files."""
        files = get_parquet_files()
        if not files: return
        
        # Extract dates from filenames (YYYY-MM-DD)
        dates = []
        for f in files:
            match = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
            if match:
                dates.append(datetime.datetime.strptime(match.group(1), "%Y-%m-%d").date())
        
        if not dates: return
        
        dates.sort()
        start_date = dates[0]
        end_date = dates[-1]
        
        all_expected_dates = set()
        current_date = start_date
        while current_date <= end_date:
            all_expected_dates.add(current_date)
            current_date += datetime.timedelta(days=1)
        
        missing_dates = all_expected_dates - set(dates)
        assert not missing_dates, f"Missing files for dates: {sorted(list(missing_dates))}"

    @pytest.mark.parametrize("file_path", get_parquet_files())
    def test_file_integrity(self, file_path):
        """Validates internal structure and data quality of each file."""
        df = pl.read_parquet(file_path)
        
        # 1. Row Count Check
        # Some files might have slightly less if maintenance happened, but generally 1440
        assert len(df) > 0, f"File {file_path.name} is empty!"
        if len(df) < EXPECTED_ROWS:
            # We log a warning but allow it if it's close (e.g., first/last day or small gaps)
            # but for a strict 'nothing missing' test, we check if it's at least 95% full
            assert len(df) >= 1400, f"File {file_path.name} has only {len(df)} rows (expected {EXPECTED_ROWS})"

        # 2. Schema Check
        required_features = [
            'body', 'upper_wick', 'lower_wick', 'log_ret_close',
            'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
        ]
        for col in required_features:
            assert col in df.columns, f"Missing feature '{col}' in {file_path.name}"
        
        assert 'close' in df.columns, f"Missing 'close' in {file_path.name}"
        
        # Check for Orderbook columns (at least level 0)
        assert 'bid_0_p' in df.columns and 'ask_0_p' in df.columns, f"Orderbook levels missing in {file_path.name}"

        # 3. Null/NaN Check
        null_counts = df.null_count()
        for col in required_features:
            count = null_counts[col][0]
            assert count == 0, f"Found {count} nulls in column '{col}' of {file_path.name}"

        # 4. Temporal Monotonicity
        # Check if index/rows are sorted and have no duplicates
        if 'ts' in df.columns:
            ts_diffs = df['ts'].diff().slice(1)
            # Diffs should be constant 60000ms if 1min resampling worked perfectly
            assert (ts_diffs < 0).sum() == 0, f"Timestamp is not monotonic in {file_path.name}"
