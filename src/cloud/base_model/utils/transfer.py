import shutil
import os
import yaml
import logging
from pathlib import Path
import subprocess
import sys
import argparse
from datetime import datetime

logger = logging.getLogger(__name__)

def cleanup_workspace():
    """
    Limpa pastas temporárias e dados intermediários para resetar o ambiente.
    Remove dados brutos, processados, rotulados, logs, modelos e cache do pytest.
    """
    project_root = Path(__file__).parents[4]
    
    master_cfg_path = project_root / "src/cloud/base_model/configs/master_config.yaml"
    config = {}
    if master_cfg_path.exists():
        with open(master_cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    targets = [
        project_root / "data/L2/raw",
        project_root / "data/L2/pre_processed",
        project_root / "data/auditor",
        project_root / "logs",
        project_root / "models",
        project_root / "artifacts",
        project_root / ".pytest_cache"
    ]
    
    # Adiciona pastas dinâmicas de labelled e splits
    pipeline_paths = config.get('pipeline_paths', {})
    l2_base = project_root / pipeline_paths.get('local_data_root', 'data/L2')
    if l2_base.exists():
        targets.extend(list(l2_base.glob(f"{pipeline_paths.get('labelled_prefix', 'splits_labelled')}*")))
        targets.extend(list(l2_base.glob("splits*")))  # Retrocompatibilidade
        targets.extend(list(l2_base.glob(f"{pipeline_paths.get('specialized_prefix', 'splits_specialized_labelled')}*")))

    print("\n--- INICIANDO LIMPEZA DO WORKSPACE ---")
    for target in targets:
        if target.exists():
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                    print(f"   Removido diretorio: {target.relative_to(project_root)}")
                else:
                    target.unlink()
                    print(f"   Removido arquivo: {target.relative_to(project_root)}")
            except Exception as e:
                print(f"   Erro ao remover {target}: {e}")
    
    # Recriar estrutura minima necessaria
    (project_root / pipeline_paths.get('local_logs_root', 'logs')).mkdir(exist_ok=True)
    temp_raw = project_root / pipeline_paths.get('local_data_root', 'data/L2') / pipeline_paths.get('temp_raw_dir', 'temp_raw')
    temp_raw.mkdir(parents=True, exist_ok=True)
    print("Workspace limpo e resetado! Estrutura base recriada.\n")

def transfer_results(log_filename: str, run_type: str):
    """
    Coleta os artefatos baseados no tipo de corrida ('foundation' ou 'specialized')
    e envia para a pasta de resultados hierárquica no Google Drive.
    Destino: RESULTADOS_{SELL}_{BUY}_{min}_{timestamp}/MODELOS/{run_type}
    """
    if run_type not in ["foundation", "specialized", "all"]:
        logger.error(f"Erro: Tipo invalido '{run_type}'. Use 'foundation', 'specialized' or 'all'.")
        return

    # 1. Configurações de Caminhos Base
    project_root = Path(__file__).parents[4]
    
    master_cfg_path = project_root / "src/cloud/base_model/configs/master_config.yaml"
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    from src.cloud.base_model.utils.path_utils import get_drive_session_path

    # Destino: pasta centralizada de sessão / MODELOS / run_type
    remote_path = get_drive_session_path(f"MODELOS/{run_type}", config)
    logger.info(f"--- Iniciando transferencia [{run_type.upper()}] para: {remote_path} ---")

    # 1.5 Gerar Landscape CSV (apenas no foundation para refletir a otimização)
    if run_type == "foundation":
        from src.cloud.base_model.utils.optuna_utils import export_landscape_to_csv
        db_path = config.get('pipeline_paths', {}).get('db_path') or \
                  config.get('optimization', {}).get('db_path', 'sqlite:///optuna_tcn_lstm_v0.db')
        study_name = config.get('optimization', {}).get('study_name', 'quantgod_tcn_lstm_v1')
        landscape_csv = project_root / "landscape_optuna_trials.csv"
        
        logger.info(f"Gerando panorama de hiperparametros em {landscape_csv.name}...")
        if export_landscape_to_csv(db_path, study_name, str(landscape_csv)):
            logger.info("✅ Landscape CSV gerado com sucesso.")
        else:
            logger.warning("⚠️ Falha ao gerar Landscape CSV.")

    # Lista Base (Configs comuns que vão para ambos garantindo autonomia)
    files_to_transfer = [
        master_cfg_path,
        project_root / "src/cloud/base_model/otimizacao/best_params.json",
        project_root / "src/cloud/base_model/otimizacao/best_dir_params.json",
    ]

    # Popula lista baseada no tipo
    if run_type == "foundation":
        # ── Coleção de LOGS e REPORTS ─────────────────────────────────────────
        # Log da Otimização
        log_root = config['pipeline_paths'].get('local_logs_root', 'logs')
        log_path = project_root / log_root / "optimization" / log_filename
        if log_path.exists():
            files_to_transfer.append(log_path)
            
        # Logs de ETL, Labelling, Treino e Tests
        log_root = config['pipeline_paths'].get('local_logs_root', 'logs')
        log_patterns = [
            f"{log_root}/etl/*.log",
            f"{log_root}/labelling/*.log",
            f"{log_root}/treino/*.log",           # ← logs do run_training.py (adicionado)
            f"{log_root}/treino_specialization/*.log",  # ← especialização (foundation tb carrega)
            f"{log_root}/auditor_preprocessing/*.log",
            f"{log_root}/auditor_labelling/*.log",
            f"{log_root}/train_xgboost/*.log",
            f"{log_root}/optimization/*.log",     # ← todos os logs de optuna, não só o passado
            f"{log_root}/tests/*.log",            # ← last_run.log + failed_files_report.log
        ]
        for pattern in log_patterns:
            files_to_transfer.extend(list(project_root.glob(pattern)))
            
        # Reports de Auditoria
        report_root = config['pipeline_paths'].get('local_reports_root', 'docs/reports')
        files_to_transfer.extend(list((project_root / report_root).glob("*.md")))
        
        # ── DATABASE & MODELS ─────────────────────────────────────────────────
        # Priority: pipeline_paths -> optimization -> default fallback
        db_path = config.get('pipeline_paths', {}).get('db_path') or \
                  config.get('optimization', {}).get('db_path', 'sqlite:///optuna_tcn_lstm_v0.db')
        
        db_filename = db_path.replace("sqlite:///", "")
        files_to_transfer.append(project_root / db_filename)
        
        # Adding items directly from master_config mappings
        for key in ['best_tcn_lstm_model', 'best_tcn_lstm_dir_model', 'scaler_foundation']:
            val = config['pipeline_paths'].get(key)
            if val:
                files_to_transfer.append(project_root / val)
        
    elif run_type == "specialized":
        log_root = config['pipeline_paths'].get('local_logs_root', 'logs')
        for log_folder in ["treino_specialization", "auditor_preprocessing", "auditor_labelling", "train_xgboost"]:
            spec_logs_dir = project_root / log_root / log_folder
            if spec_logs_dir.exists():
                logs = sorted(list(spec_logs_dir.glob("*.log")))
                if logs:
                     files_to_transfer.append(logs[-1])

        files_to_transfer.extend([
            project_root / config['pipeline_paths']['best_specialized_model'],
            project_root / config['pipeline_paths']['scaler_specialized'],
            project_root / "data/models/auditor_xgboost.json"
        ])
        
    elif run_type == "all":
        # ── COLECAO ABSOLUTA (PIPELINE COMPLETO V4.3) ─────────────────────────
        log_root = config['pipeline_paths'].get('local_logs_root', 'logs')
        data_root = config['pipeline_paths'].get('local_data_root', 'data/L2')
        log_patterns = [
            f"{log_root}/etl/*.log", f"{log_root}/labelling/*.log", f"{log_root}/treino/*.log", 
            f"{log_root}/treino_specialization/*.log", f"{log_root}/auditor_preprocessing/*.log",
            f"{log_root}/auditor_labelling/*.log", f"{log_root}/train_xgboost/*.log",
            f"{log_root}/optimization/*.log", f"{log_root}/tests/*.log",
            f"{log_root}/QA/*.log",                  # Strict OOF QA Reports
            f"{log_root}/run_manager/*.log",
            f"{data_root}/split_summary_*.json"    # Temporal integrity checksums
        ]
        for pattern in log_patterns:
            files_to_transfer.extend(list(project_root.glob(pattern)))
            
        report_root = config['pipeline_paths'].get('local_reports_root', 'docs/reports')
        files_to_transfer.extend(list((project_root / report_root).glob("*.md")))
        
        # All databases
        files_to_transfer.extend(list(project_root.glob("*.db")))
        
        # All models & scalers
        for key in ['best_tcn_lstm_model', 'best_tcn_lstm_dir_model', 'best_specialized_model', 
                    'scaler_foundation', 'scaler_specialized', 'auditor_model', 'scaler_auditor']:
            val = config['pipeline_paths'].get(key)
            if val: files_to_transfer.append(project_root / val)
        
    # Relatorio de Feature Importance (comum a ambos, mas gerado no foundation agora)
    report_root = config['pipeline_paths'].get('local_reports_root', 'docs/reports')
    fi_csv = project_root / report_root / "feature_importance.csv"
    if fi_csv.exists():
        files_to_transfer.append(fi_csv)

    # Landscape Optuna (gerado no passo anterior)
    landscape_csv = project_root / "landscape_optuna_trials.csv"
    if landscape_csv.exists():
        files_to_transfer.append(landscape_csv)

    # ── Validação e Filtro Tolerante a Falhas ──────────────────────────────────
    valid_files = set()
    missing_files = set()
    
    for f in files_to_transfer:
        if f.exists():
            valid_files.add(f)
        else:
            missing_files.add(f)
            
    if missing_files:
        logger.warning("\n⚠️ AVISO: Os seguintes arquivos nao foram encontrados e serao ignorados:")
        for mf in sorted(list(missing_files)):
            try:
                logger.warning(f"   -> {mf.relative_to(project_root)}")
            except ValueError:
                logger.warning(f"   -> {mf.name}")
    
    files_to_transfer = sorted(list(valid_files))
    
    if not files_to_transfer:
        logger.error("❌ Nenhum arquivo válido para transferir. Abortando.")
        return

    # ── Executar Transferência ───────────────────────────────────────────────
    try:

        if os.name != 'nt':
            temp_staging = project_root / "data" / "temp_results" / run_type
            if temp_staging.exists():
                shutil.rmtree(temp_staging)
            temp_staging.mkdir(parents=True)

            logger.info(f"Agrupando {len(files_to_transfer)} arquivos em {temp_staging.relative_to(project_root)}...")
            for src in files_to_transfer:
                shutil.copy2(src, temp_staging / src.name)

            rclone_cfg = project_root / "rclone.conf"
            
            rclone_transfers = str(min(32, (os.cpu_count() or 4) * 2))
            cmd = ["rclone", "copy", str(temp_staging), remote_path, "-P", "--transfers", rclone_transfers, "--checkers", rclone_transfers]
            if rclone_cfg.exists():
                cmd += ["--config", str(rclone_cfg)]
            
            result = subprocess.run(cmd)

            if result.returncode == 0:
                logger.info(f"SUCESSO! Resultados copiados (rclone) para: {remote_path}")
                shutil.rmtree(project_root / "data" / "temp_results")
                return

        # Fallback (Windows sem rclone)
        dest_dir = project_root / "data" / "temp_results" / run_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in files_to_transfer:
            logger.info(f"   📂 Copiando: {src.name}...")
            shutil.copy2(src, dest_dir / src.name)

        logger.info(f"\nSUCESSO! Resultados copiados localmente para: {dest_dir}")

    except Exception as e:
        logger.error(f"Erro na transferencia: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerenciador de Resultados e Workspace QuantGod.")
    parser.add_argument("--clean", action="store_true", help="Limpa o workspace e as pastas de dados/logs.")
    parser.add_argument("log_filename", type=str, nargs='?', help="Nome do log base para exportação.")
    parser.add_argument("run_type", type=str, nargs='?', choices=["foundation", "specialized", "all"], help="Tipo de exportação.")
    
    args = parser.parse_args()

    if args.clean:
        cleanup_workspace()
    
    if args.log_filename and args.run_type:
        from src.cloud.base_model.utils.logging_utils import setup_logger
        setup_logger("transfer", f"_{args.run_type}")
        transfer_results(args.log_filename, args.run_type)
    elif not args.clean:
        parser.print_help()
