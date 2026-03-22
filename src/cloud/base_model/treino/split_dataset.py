import os
import shutil
import logging
from pathlib import Path
import sys
import yaml
import json
import hashlib
import gc
import time

# Add project root to path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger, upload_audit_to_drive
from src.cloud.base_model.utils.path_utils import get_labelled_dir, get_specialized_dir, get_logs_root

logger = logging.getLogger(__name__)

def sha256_file(filepath):
    # Calculations SHA256 hash of a file.
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data: break
            sha.update(data)
    return sha.hexdigest()


def execute_split(stage_name, source_files, target_base_dir, train_ratio, split_by_bars=True):
    """
    Executes a chronological split.
    - If split_by_bars=True [AFML SOTA]: Calculates cut-point by cumulative bar count.
    - If split_by_bars=False [Legacy]: Calculates cut-point by file count.
    """
    if not source_files:
        logger.warning(f"No files to split for {stage_name}")
        return [], []

    import polars as pl
    # Configuração de diretórios
    train_dir = target_base_dir / "train"
    val_dir   = target_base_dir / "val"
    for d in [train_dir, val_dir]:
        if d.exists():
            try: shutil.rmtree(d)
            except Exception: pass
        d.mkdir(parents=True, exist_ok=True)

    if split_by_bars:
        # AFML SOTA: Exact bar-level split (Robust Row Counting)
        row_counts = []
        for f in source_files:
            try:
                count = pl.scan_parquet(f).select(pl.len()).collect().item()
                row_counts.append(count)
            except Exception:
                row_counts.append(0)

        total_rows = sum(row_counts)
        target_train_rows = int(train_ratio * total_rows)

        cumul = 0
        split_idx = -1
        rows_to_take_from_split_file = 0

        for i, count in enumerate(row_counts):
            if cumul + count >= target_train_rows:
                split_idx = i
                rows_to_take_from_split_file = target_train_rows - cumul
                break
            cumul += count

        # Caso ratio=100% ou erro
        if split_idx == -1: split_idx = len(source_files) - 1

        # ── DISTRIBUIÇÃO FÍSICA ──────────────────────────────────────────────
        # 1. Arquivos totalmente para TREINO
        for i in range(split_idx):
            f = source_files[i]
            shutil.copy2(f, train_dir / f.name)

        # 2. Arquivo de FRONTEIRA (Dividido)
        f_mid = source_files[split_idx]
        count_mid = row_counts[split_idx]
        
        if rows_to_take_from_split_file > 0 and rows_to_take_from_split_file < count_mid:
            # FISICAMENTE DIVIDIR O ARQUIVO
            df_mid = pl.read_parquet(f_mid, memory_map=False)
            df_train = df_mid.slice(0, rows_to_take_from_split_file)
            df_val = df_mid.slice(rows_to_take_from_split_file)
            
            df_train.write_parquet(train_dir / f"{f_mid.stem}_partA.parquet")
            df_val.write_parquet(val_dir / f"{f_mid.stem}_partB.parquet")
            logger.info(f"✂️ [{stage_name}] Split boundary file: {f_mid.name} -> {rows_to_take_from_split_file} bars to Train / Rest to Val")
        elif rows_to_take_from_split_file >= count_mid:
            shutil.copy2(f_mid, train_dir / f_mid.name)
        else:
            shutil.copy2(f_mid, val_dir / f_mid.name)

        # 3. Arquivos totalmente para VALIDAÇÃO
        for i in range(split_idx + 1, len(source_files)):
            f = source_files[i]
            shutil.copy2(f, val_dir / f.name)

        actual_train_rows = target_train_rows
        actual_val_rows   = total_rows - target_train_rows
        actual_ratio = actual_train_rows / total_rows if total_rows > 0 else 0.0
        msg = f"[{stage_name}] True Bar-level Split: {total_rows:,} barras → {actual_train_rows:,} Train ({actual_ratio*100:.1f}%)"
    else:
        # Legacy: File-level split
        total_files = len(source_files)
        split_idx = int(train_ratio * total_files)
        train_files = source_files[:split_idx]
        val_files   = source_files[split_idx:]
        
        for f in train_files: shutil.copy2(f, train_dir / f.name)
        for f in val_files:   shutil.copy2(f, val_dir / f.name)
        
        msg = f"[{stage_name}] File-level Split (Legacy): {total_files} arquivos → {split_idx} Train"

    logger.info(f"{msg} | Result: {len(list(train_dir.glob('*.parquet')))} train files / {len(list(val_dir.glob('*.parquet')))} val files")
    
    time.sleep(0.5) 
    gc.collect()
    
    # Retornar listas de caminhos relativos para os diretórios criados (conforme contrato do script)
    return sorted(list(train_dir.glob("*.parquet"))), sorted(list(val_dir.glob("*.parquet")))

def enforce_purge_gap(train_dir, config, stage_name):
    """
    Aplica o Purge Gap: Remove `horizon_minutes` barras do FINAL do último arquivo
    do bloco de treino para garantir que o lookahead de rótulos não vaze para a validação.
    """
    import polars as pl
    from src.cloud.base_model.pre_processamento.etl.transform import _parse_resample_minutes

    from datetime import timedelta

    train_files = sorted(list(Path(train_dir).glob("*.parquet")))
    if not train_files:
        return 0

    last_file = train_files[-1]
    df = pl.read_parquet(last_file, memory_map=False)

    if "datetime" not in df.columns:
        logger.warning(f"⚠️ Column 'datetime' not found in {last_file.name}. Falling back to bar-count purge.")
        resample_freq = config['pre_processing']['etl'].get('resample_freq', '5min')
        from src.cloud.base_model.pre_processamento.etl.transform import _parse_resample_minutes
        resample_min = _parse_resample_minutes(resample_freq)
        horizon_min = config['pre_processing']['labelling'].get('horizon_minutes', 15)
        purge_bars = max(1, horizon_min // resample_min)
        df_purged = df.slice(0, len(df) - purge_bars)
    else:
        # AFML SOTA: Temporal Purge
        horizon_min = config['pre_processing']['labelling'].get('horizon_minutes', 15)
        cutoff_ts = df["datetime"].max() - timedelta(minutes=horizon_min)
        df_purged = df.filter(pl.col("datetime") <= cutoff_ts)
        purge_bars = len(df) - len(df_purged)

    if len(df_purged) > 0:
        for attempt in range(5):
            try:
                df_purged.write_parquet(last_file)
                logger.info(f"🛡️ [{stage_name}] TEMPORAL PURGE: Removed {purge_bars} bars to keep {horizon_min}m gap before validation.")
                return purge_bars
            except Exception as e:
                if "1224" in str(e) and attempt < 4:
                    time.sleep(1.0); gc.collect()
                else: raise e
        return purge_bars
    else:
        logger.warning(f"⚠️ [{stage_name}] {last_file.name} fully purged. Deleting.")
        last_file.unlink()
        return len(df)


def enforce_embargo(val_dir, config, stage_name):
    """
    [v9.0 AFML-Aligned] Aplica o Embargo Period: Remove as primeiras `embargo_bars`
    do PRIMEIRO arquivo da validação, eliminando autocorrelação serial entre treino e val.

    De Prado (AFML, Cap. 7): após o purge gap, existe ainda correlação serial
    entre as últimas barras do treino e as primeiras da validação. O embargo
    remove uma janela adicional do início da validação para garantir independência.
    """
    import polars as pl
    from src.cloud.base_model.pre_processamento.etl.transform import _parse_resample_minutes

    from datetime import timedelta

    val_files = sorted(list(Path(val_dir).glob("*.parquet")))
    if not val_files:
        return 0

    first_file = val_files[0]
    df = pl.read_parquet(first_file, memory_map=False)

    stage_key = stage_name.lower()
    root_split = config['pre_processing']['labelling']['split']
    split_cfg  = root_split.get(stage_key, root_split.get('base', {}))
    embargo_min = split_cfg.get('embargo_minutes', config['pre_processing']['labelling'].get('horizon_minutes', 15))

    if "datetime" not in df.columns:
        resample_freq = config['pre_processing']['etl'].get('resample_freq', '5min')
        from src.cloud.base_model.pre_processamento.etl.transform import _parse_resample_minutes
        resample_min = _parse_resample_minutes(resample_freq)
        embargo_bars = max(1, embargo_min // resample_min)
        df_embargoed = df.slice(embargo_bars)
    else:
        # AFML SOTA: Temporal Embargo
        cutoff_ts = df["datetime"].min() + timedelta(minutes=embargo_min)
        df_embargoed = df.filter(pl.col("datetime") >= cutoff_ts)
        embargo_bars = len(df) - len(df_embargoed)

    if len(df_embargoed) > 0:
        for attempt in range(5):
            try:
                df_embargoed.write_parquet(first_file)
                logger.info(f"🚫 [{stage_name}] TEMPORAL EMBARGO: Removed {embargo_bars} bars to add {embargo_min}m safety after boundary.")
                return embargo_bars
            except Exception as e:
                if "1224" in str(e) and attempt < 4:
                    time.sleep(1.0); gc.collect()
                else: raise e
        return embargo_bars
    else:
        logger.warning(f"⚠️ [{stage_name}] {first_file.name} fully embargoed. Skipping write.")
        return len(df)

def split_and_segregate():
    # Setup logger
    setup_logger("split_dataset", "")
    
    master_cfg_path = Path("src/cloud/base_model/configs/master_config.yaml")
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Resolvendo os diretórios baseados no config centralizado
    source_dir = Path(get_labelled_dir(config))

    # ── [v9.5] Auto-Sync From Drive (Process all 4 years together) ──────────
    try:
        from src.cloud.base_model.utils.path_utils import get_drive_session_path
        remote_src = get_drive_session_path("LABELLED", config)
        rclone_cfg = Path("rclone.conf")
        rclone_bin = "rclone"
        import subprocess
        if os.name == 'nt' and Path("rclone.exe").exists():
            rclone_bin = str(Path("rclone.exe").absolute())
        
        logger.info(f"🔄 Syncing full Labelled history (2023-2026) from Drive: {remote_src}...")
        sync_cmd = [rclone_bin, "sync", remote_src, str(source_dir.absolute()), "-P", "--transfers", "32"]
        if rclone_cfg.exists():
            sync_cmd += ["--config", str(rclone_cfg)]
        
        subprocess.run(sync_cmd, check=True)
        logger.info("✅ Labelled local directory is now up to date with Drive.")
    except Exception as e:
        logger.warning(f"⚠️ Auto-sync failed (using local files only): {e}")

    if not source_dir.exists():
        logger.error(f"❌ Source labelled (Master Dataset) directory not found: {source_dir}")
        return

    # Garante a ordem do timestamp puramente baseando no texto da filename (arquivos parquet nomeados temporalmente)
    all_files = sorted(list(source_dir.glob("*.parquet")), key=lambda x: x.name)
    if not all_files:
        logger.error("❌ No parquet files found.")
        return

    logger.info(f"🚀 Starting Strict Out-of-Fold (OOF) Splitting over {len(all_files)} total files.")

    # Label used for naming the split_summary file
    base_labelled_name = source_dir.name  # e.g. "labelled_SELL_00030_BUY_00030_5min_..."

    # Proporções e Toggles (v9.7 Corrected YAML path)
    split_cfg_root = config['pre_processing']['labelling']['split']
    split_by_bars  = split_cfg_root.get('split_by_bars', True)
    
    base_train_pct = split_cfg_root['base'].get('train_ratio', 0.70)
    spec_train_pct = split_cfg_root['auditor'].get('train_ratio', 0.80)
    
    # Destinos
    base_split_dir = source_dir
    spec_split_dir = Path(get_specialized_dir(config))
    
    # ── 1. LAYER 1: Foundation (splits_base) ───────────────────────────
    # Pega TODO O DATASET e corta o futuro em 30% pra validacao (invisivel aos pesos da base)
    base_train_files, base_val_files = execute_split(
        stage_name="Foundation",
        source_files=all_files,
        target_base_dir=base_split_dir,
        train_ratio=base_train_pct,
        split_by_bars=split_by_bars
    )
    
    # ── 2. LAYER 2: Specialist (splits_specialized) ────────────────────
    # Regra de Ouro Strict OOF: Pega apenas o VAL da Fundacao e divide. 
    # O Especialista nunca verá a mesma janela temporal (Train Base)
    spec_train_files, spec_val_files = execute_split(
        stage_name="Specialist",
        source_files=base_val_files,
        target_base_dir=spec_split_dir,
        train_ratio=spec_train_pct,
        split_by_bars=split_by_bars
    )
    
    # ── 2.5 APLICAÇÃO DO PURGE GAP + EMBARGO TEMPORAL ────────────────────────
    # Purge Gap: remove horizon_minutes do FINAL do treino (AFML Cap. 3)
    # Embargo:   remove embargo_minutes do INÍCIO da val (AFML Cap. 7)
    base_purged_bars = enforce_purge_gap(base_split_dir / "train", config, "Foundation")
    spec_purged_bars = enforce_purge_gap(spec_split_dir / "train", config, "Specialist")

    base_embargo_bars = enforce_embargo(base_split_dir / "val", config, "Foundation")
    spec_embargo_bars = enforce_embargo(spec_split_dir / "val", config, "Specialist")

    # ── 2.6 VALIDAÇÃO CRONOLÓGICA PÓS-SPLIT (v4.9 Gold) ──────────────────
    # Garante que o PRIMEIRO arquivo do val é TEMPORALMENTE POSTERIOR ao ÚLTIMO do treino.
    # Protege contra qualquer inversão de ordem por nomes de arquivo não padronizados.
    logger.info("🕵️ [CHRONO GUARD] Validating chronological integrity of splits...")
    for stage_name, train_f, val_f, split_dir in [
        ("Foundation", base_train_files, base_val_files, base_split_dir),
        ("Specialist", spec_train_files, spec_val_files, spec_split_dir),
    ]:
        try:
            if train_f and val_f:
                import polars as pl
                last_train_path  = split_dir / "train" / train_f[-1].name
                first_val_path   = split_dir / "val"   / val_f[0].name

                last_train_ts  = pl.read_parquet(last_train_path, columns=['datetime']).row(-1)[0]
                first_val_ts   = pl.read_parquet(first_val_path,  columns=['datetime']).row(0)[0]

                if last_train_ts >= first_val_ts:
                    logger.error(
                        f"❌ [{stage_name}] CHRONOLOGICAL VIOLATION: last train ts ({last_train_ts}) "
                        f">= first val ts ({first_val_ts}). DATA LEAKAGE RISK!"
                    )
                else:
                    gap_delta = (first_val_ts - last_train_ts)
                    gap_ms = int(gap_delta.total_seconds() * 1000) if hasattr(gap_delta, 'total_seconds') else gap_delta
                    logger.info(f"✅ [{stage_name}] Chrono Guard PASSED — gap between train/val boundary: {gap_ms} ms")
        except Exception as e:
            logger.warning(f"⚠️ [{stage_name}] Chrono Guard check failed (non-fatal): {e}")

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
        },
        "embargo_proof": {
            "Foundation": f"{base_embargo_bars} barras removidas do início do Val (autocorrelação serial — AFML Cap.7)",
            "Specialist": f"{spec_embargo_bars} barras removidas do início do Val (autocorrelação serial — AFML Cap.7)",
            "embargo_minutes_applied": {
                "foundation": config['pre_processing']['labelling']['split']['base'].get('embargo_minutes', 15),
                "specialist": config['pre_processing']['labelling']['split']['auditor'].get('embargo_minutes', 15),
            }
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
    # Audit Logs → Drive  (PROJETOS/AUDITORIA_.../SPLIT)
    import yaml as _yaml
    with open("src/cloud/base_model/configs/master_config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    upload_audit_to_drive(
        local_dirs=[f"{get_logs_root(_cfg)}/split_dataset"],
        stage_name="SPLIT",
        config=_cfg,
    )
