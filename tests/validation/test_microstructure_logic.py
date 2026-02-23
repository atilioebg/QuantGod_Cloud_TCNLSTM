import pandas as pd
import numpy as np

def mock_apply_feature_engineering():
    # 1. Create a dummy dataframe with exactly 5 minutes of data, but with a GAP.
    # T=0, T=1, T=2, (GAP T=3), T=4
    dates = pd.to_datetime([
        "2023-01-01 10:00:00",
        "2023-01-01 10:01:00",
        "2023-01-01 10:02:00",
        "2023-01-01 10:04:00" # Gap at 10:03
    ])
    
    # We will simulate the aggregated columns after resample('1min')
    df = pd.DataFrame({
        'close': [100.0, 101.0, 101.5, 102.0],              # Prices should FFILL during gap
        'bid_0_p': [99.5, 100.5, 101.0, 101.5],             # Prices FFILL
        'ofi': [1.5, -0.5, 2.0, 1.0],                       # OFI should be 0 during gap
        'tick_count': [50, 45, 60, 40],                     # Volume should be 0 during gap
        'micro_price_momentum': [0.001, -0.002, 0.005, 0.002] # Flow should be 0 during gap
    }, index=dates)

    print("--- RAW AGGREGATED DATA (WITH GAP AT 10:03) ---")
    print(df)
    
    # 2. Reindexing to the Full Day (Mocked to a 5-min window for visibility)
    # Expected output: 10:00 to 10:04 (5 rows)
    full_idx = pd.date_range("2023-01-01 10:00:00", "2023-01-01 10:04:00", freq="1min")
    df_reindexed = df.reindex(full_idx)
    
    print("\n--- AFTER REINDEX (RAW GAPS) ---")
    print(df_reindexed)
    
    # 3. Apply Imputation logic
    # Static / Price features -> ffill
    price_cols = [c for c in df_reindexed.columns if c not in ['ofi', 'tick_count', 'micro_price_momentum', 'log_volume']]
    df_reindexed[price_cols] = df_reindexed[price_cols].ffill()
    
    # Dynamic / Flow features -> fillna(0)
    flow_cols = ['ofi', 'tick_count', 'micro_price_momentum']
    df_reindexed[flow_cols] = df_reindexed[flow_cols].fillna(0)
    
    print("\n--- AFTER TIME-AWARE IMPUTATION (FFILL PRICING, ZERO FLOW) ---")
    print(df_reindexed)

if __name__ == "__main__":
    mock_apply_feature_engineering()
