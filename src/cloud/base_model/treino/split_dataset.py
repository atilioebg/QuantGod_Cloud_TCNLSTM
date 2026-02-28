import os
import shutil
import logging
from pathlib import Path
import sys
import yaml
import json
import hashlib

# Add project root to path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

def sha256_file(filepath):
    """Calculates SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)
    return h.hexdigest()

def execute_split(stage_name, source_files, target_base_dir, train_ratio):
    """Executes a chronological split (shuffle=False) to prevent data leakage."""
    if not source_files:
        logger.warning(f"No files to split for {stage_name}")
        return [], []
        
    total_files = len(source_files)
    split_idx = int(train_ratio * total_files)
    
    # SHUFFLE=FALSE -> Garantia cronologica absoluta de OOF
    train_files = source_files[:split_idx]
    val_files = source_files[split_idx:]
    
    train_dir = target_base_dir / "train"
    val_dir = target_base_dir / "val"
    
    # Limpa e recria diretórios de destino
    for d in [train_dir, val_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        
    logger.info(f"[{stage_name}] Splitting {total_files} files -> {len(train_files)} Train ({train_ratio*100:.0f}%), {len(val_files)} Val")
    
    # Copia arquivos (Tenta Hardlink primeiro para maxima performance/0 disk space)
    for f in train_files:
        dest = train_dir / f.name
        try: os.link(src=f, dst=dest)
        except OSError: shutil.copy2(f, dest)
            
    for f in val_files:
        dest = val_dir / f.name
        try: os.link(src=f, dst=dest)
        except OSError: shutil.copy2(f, dest)
        
    return train_files, val_files

def enforce_purge_gap(train_dir, config, stage_name):
    """
    Aplica o Purge Gap: Remove `horizon_minutes` barras do FINAL do último arquivo 
    do bloco de treino para garantir que o lookahead de rótulos não vaze para a validação.
    """
    import polars as pl
    from src.cloud.base_model.pre_processamento.etl.transform import _parse_resample_minutes

    train_files = sorted(list(Path(train_dir).glob("*.parquet")))
    if not train_files:
        return 0

    last_file = train_files[-1]
    df = pl.read_parquet(last_file)
    
    horizon_min = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    resample_freq = config['pre_processing']['etl'].get('resample_freq', '1min')
    resample_min = _parse_resample_minutes(resample_freq)
    
    purge_bars = max(1, horizon_min // resample_min)
    
    if len(df) > purge_bars:
        df_purged = df.slice(0, len(df) - purge_bars)
        df_purged.write_parquet(last_file)
        logger.info(f"🛡️ [{stage_name}] PURGE GAP APLICADO: {purge_bars} barras ({horizon_min} min) removidas do final de {last_file.name}")
        return purge_bars
    else:
        logger.warning(f"⚠️ [{stage_name}] O último arquivo {last_file.name} tem menos barras que o purge_gap ({len(df)} < {purge_bars}). Ele será deletado.")
        last_file.unlink()
        return len(df)

def split_and_segregate():
    # Setup logger
    setup_logger("split_dataset", "")
    
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Reconstruindo a resolucao das pastas baseadas na configuracao
    sell_th = config['pre_processing']['labelling'].get('sell_threshold', 0.003)
    buy_th  = config['pre_processing']['labelling'].get('buy_threshold', 0.003)
    mins    = config['pre_processing']['labelling'].get('horizon_minutes', 15)
    
    base_labelled_name = f"labelled_SELL_{sell_th:.4f}_BUY_{buy_th:.4f}_{mins}min".replace(".", "")
    # O script run_labelling cria a pasta com prefixo 'splits_'
    source_dir = Path(f"data/L2/splits_{base_labelled_name}")

    if not source_dir.exists():
        # Fallback para o nome sem prefixo caso ja exista de outra forma
        source_dir = Path(f"data/L2/{base_labelled_name}")
        
    if not source_dir.exists():
        logger.error(f"❌ Source labelled (Master Dataset) directory not found: data/L2/splits_{base_labelled_name}")
        return

    # Garante a ordem do timestamp puramente baseando no texto da filename (arquivos parquet nomeados temporalmente)
    all_files = sorted(list(source_dir.glob("*.parquet")), key=lambda x: x.name)
    if not all_files:
        logger.error("❌ No parquet files found.")
        return

    logger.info(f"🚀 Starting Strict Out-of-Fold (OOF) Splitting over {len(all_files)} total files.")
    
    # Proporções
    base_train_pct = config['pre_processing']['split']['base'].get('train_ratio', 0.70)
    spec_train_pct = config['pre_processing']['split']['specialized'].get('train_ratio', 0.80)
    
    # Destinos
    base_split_dir = Path(f"data/L2/splits_{base_labelled_name}")
    spec_split_dir = Path(f"data/L2/splits_specialized_{base_labelled_name}")
    
    # ── 1. LAYER 1: Foundation (splits_base) ───────────────────────────
    # Pega TODO O DATASET e corta o futuro em 30% pra validacao (invisivel aos pesos da base)
    base_train_files, base_val_files = execute_split(
        stage_name="Foundation",
        source_files=all_files,
        target_base_dir=base_split_dir,
        train_ratio=base_train_pct
    )
    
    # ── 2. LAYER 2: Specialist (splits_specialized) ────────────────────
    # Regra de Ouro Strict OOF: Pega apenas o VAL da Fundacao e divide. 
    # O Especialista nunca verá a mesma janela temporal (Train Base)
    spec_train_files, spec_val_files = execute_split(
        stage_name="Specialist",
        source_files=base_val_files,
        target_base_dir=spec_split_dir,
        train_ratio=spec_train_pct
    )
    
    # ── 2.5 APLICAÇÃO DO PURGE GAP TEMPORAL ──────────────────────────────
    base_purged_bars = enforce_purge_gap(base_split_dir / "train", config, "Foundation")
    spec_purged_bars = enforce_purge_gap(spec_split_dir / "train", config, "Specialist")
    
    # ── 3. Checklists & Hashing (split_summary.json) ────────────────────
    logger.info("⏳ Generating split_summary.json with hashes and timestamp boundaries...")
    
    summary = {
        "foundation": {
            "train": {
                "count": len(base_train_files),
                "first_file": base_train_files[0].name if base_train_files else None,
                "last_file": base_train_files[-1].name if base_train_files else None,
                "files": {f.name: sha256_file(base_split_dir/"train"/f.name) for f in base_train_files}
            },
            "val": {
                "count": len(base_val_files),
                "first_file": base_val_files[0].name if base_val_files else None,
                "last_file": base_val_files[-1].name if base_val_files else None,
                "files": {f.name: sha256_file(base_split_dir/"val"/f.name) for f in base_val_files}
            }
        },
        "specialist": {
            "train": {
                "count": len(spec_train_files),
                "first_file": spec_train_files[0].name if spec_train_files else None,
                "last_file": spec_train_files[-1].name if spec_train_files else None,
                "files": {f.name: sha256_file(spec_split_dir/"train"/f.name) for f in spec_train_files}
            },
            "val": {
                "count": len(spec_val_files),
                "first_file": spec_val_files[0].name if spec_val_files else None,
                "last_file": spec_val_files[-1].name if spec_val_files else None,
                "files": {f.name: sha256_file(spec_split_dir/"val"/f.name) for f in spec_val_files}
            }
        },
        "purge_gap_proof": {
            "Foundation": f"{base_purged_bars} barras descartadas da borda do Treino para barrar Leakage",
            "Specialist": f"{spec_purged_bars} barras descartadas da borda do Treino para barrar Leakage",
            "gap_minutes_applied": config['pre_processing']['labelling'].get('horizon_minutes', 15)
        }
    }
    
    # Salva na raiz logistica do dataset
    summary_path = Path("data/L2") / f"split_summary_{base_labelled_name}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"✅ Strict OOF Splitting complete! Summary saved to {summary_path}")
    logger.info("   ↳ Rclone will upload this summary artifact in the Final Transfer.")

if __name__ == "__main__":
    split_and_segregate()
