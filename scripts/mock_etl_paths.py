import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.cloud.base_model.utils.path_utils import get_drive_session_path, get_drive_session_root
import yaml

def mock_test():
    # Load config
    with open("src/cloud/base_model/configs/master_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    years = [2023, 2024, 2025, 2026]
    
    print("=== MONITORAMENTO DE ESTRUTURA DE PASTAS (DRIVE) ===\n")
    
    # Simulating shared session (if fixed in config) vs new sessions
    session_ts = config.get('pipeline_paths', {}).get('session_timestamp')
    
    root = get_drive_session_root(config)
    print(f"RAIZ DO BUCKET (Thresholds Base): {root}\n")

    for year in years:
        # Mocking config update for the year
        config['pre_processing']['etl']['start_year'] = year
        config['pre_processing']['etl']['end_year'] = year
        
        # In a real scenario, if session_timestamp is null in config, 
        # path_utils generates one per process.
        # We'll simulate that by clearing the cache if it were real, 
        # but here we just want to see the path logic.
        
        remote_path = get_drive_session_path("PRE_PROCESSED", config)
        print(f"ANO {year}:")
        print(f"  Local Source: data/L2/pre_processed/")
        print(f"  Drive Destination: {remote_path}")
        print("-" * 50)

    print("\nOBSERVAÇÕES:")
    if not session_ts:
        print("⚠️ AVISO: session_timestamp NÃO está fixo no master_config.yaml.")
        print("Cada execução anual criará uma nova pasta com timestamp (Ex: 260319_233400).")
        print("Isso isola os arquivos, mas pode fragmentar o dataset no Drive.")
    else:
        print(f"✅ session_timestamp FIXO: {session_ts}")
        print("Todas as execuções (2023-2026) cairão na mesma pasta, consolidando o dataset.")

if __name__ == "__main__":
    mock_test()
