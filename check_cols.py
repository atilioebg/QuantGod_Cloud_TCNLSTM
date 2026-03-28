import pandas as pd
import glob
import os
import yaml

def check():
    files = glob.glob('data/backtest/L2/labelled/*.parquet')
    if not files:
        files = glob.glob('data/L2/labelled/*.parquet')

    if not files:
        print("No parquet files found.")
        return

    df = pd.read_parquet(files[0])
    cols = df.columns.tolist()
    print(f"Total columns: {len(cols)}")
    print("First 20 columns:", cols[:20])
    
    # Check for specific auditor features from master_config
    try:
        with open('src/cloud/base_model/configs/master_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        auditor_feats = config['model']['auditor_features']
        print("\nChecking for Auditor Features (indices 6-19):")
        context_needed = auditor_feats[6:]
        found = []
        missing = []
        for feat in context_needed:
            if feat in cols:
                found.append(feat)
            elif f"alpha_{feat}" in cols:
                found.append(f"alpha_{feat}")
            else:
                missing.append(feat)
        
        print(f"Found: {len(found)}/{len(context_needed)}")
        if found: print(f"Sample found: {found[:5]}")
        if missing: print(f"Missing: {missing}")
        
    except Exception as e:
        print(f"Error reading config: {e}")

if __name__ == "__main__":
    check()
