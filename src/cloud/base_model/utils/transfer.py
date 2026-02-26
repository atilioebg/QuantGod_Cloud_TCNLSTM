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
    l2_base = project_root / "data/L2"
    if l2_base.exists():
        targets.extend(list(l2_base.glob("labelled*")))
        targets.extend(list(l2_base.glob("splits*")))
        targets.extend(list(l2_base.glob("specialized*")))

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
    (project_root / "logs").mkdir(exist_ok=True)
    (project_root / "data/L2/raw").mkdir(parents=True, exist_ok=True)
    (project_root / "data/L2/pre_processed").mkdir(parents=True, exist_ok=True)
    print("Workspace limpo e resetado! Estrutura base recriada.\n")

def transfer_results(log_filename: str, run_type: str):
    """
    Coleta os artefatos baseados no tipo de corrida ('foundation' ou 'specialized')
    e envia para a pasta de resultados hierárquica no Google Drive.
    """
    if run_type not in ["foundation", "specialized"]:
        logger.error(f"Erro: Tipo invalido '{run_type}'. Use 'foundation' or 'specialized'.")
        return

    # 1. Configurações de Caminhos Base
    project_root = Path(__file__).parents[4]
    
    master_cfg_path = project_root / "src/cloud/base_model/configs/master_config.yaml"
    with open(master_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Nome da pasta de destino raiz
    folder_name = log_filename.replace(".log", "")
    
    # Definir base do Drive (Detecta se Windows ou Linux/Pod)
    if os.environ.get('MOCK_DRIVE') == '1':
        drive_base = Path("mock_drive_results")
    elif os.name == 'nt':
        drive_base = Path("Z:/PROJETOS/RESULTADOS")
    else:
        drive_base = project_root / "drive" / "PROJETOS" / "RESULTADOS"
    
    # Destino Final Especifico do Tipo
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    # Destino Final Especifico do Tipo
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    dest_dir = drive_base / f"{folder_name}_{timestamp}" / run_type
    logger.info(f"--- Iniciando transferencia [{run_type.upper()}] para: {dest_dir} ---")

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
        log_path = project_root / "logs" / "optimization" / log_filename
        if log_path.exists():
            files_to_transfer.append(log_path)
            
        # Logs de ETL, Labelling, Treino e Tests
        log_patterns = [
            "logs/etl/*.log",
            "logs/labelling/*.log",
            "logs/treino/*.log",           # ← logs do run_training.py (adicionado)
            "logs/treino_specialization/*.log",  # ← especialização (foundation tb carrega)
            "logs/auditor_preprocessing/*.log",
            "logs/auditor_labelling/*.log",
            "logs/train_xgboost/*.log",
            "logs/optimization/*.log",     # ← todos os logs de optuna, não só o passado
            "logs/tests/*.log",            # ← last_run.log + failed_files_report.log
        ]
        for pattern in log_patterns:
            files_to_transfer.extend(list(project_root.glob(pattern)))
            
        # Reports de Auditoria
        files_to_transfer.extend(list((project_root / "docs/reports").glob("*.md")))
        
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
        # Pega logs de specialization e auditoria
        for log_folder in ["treino_specialization", "auditor_preprocessing", "auditor_labelling", "train_xgboost"]:
            spec_logs_dir = project_root / "logs" / log_folder
            if spec_logs_dir.exists():
                logs = sorted(list(spec_logs_dir.glob("*.log")))
                if logs:
                     files_to_transfer.append(logs[-1])

        # Modelos e Scalers do Especialista e XGBoost Auditor
        files_to_transfer.extend([
            project_root / config['pipeline_paths']['best_specialized_model'],
            project_root / config['pipeline_paths']['scaler_specialized'],
            project_root / "data/models/auditor_xgboost.json"
        ])
        
    # Relatorio de Feature Importance (comum a ambos, mas gerado no foundation agora)
    fi_csv = project_root / "docs" / "reports" / "feature_importance.csv"
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
            remote_path = f"drive:PROJETOS/RESULTADOS/{folder_name}_{timestamp}/{run_type}"
            
            cmd = ["rclone", "copy", str(temp_staging), remote_path, "-P"]
            if rclone_cfg.exists():
                cmd += ["--config", str(rclone_cfg)]
            
            result = subprocess.run(cmd)

            if result.returncode == 0:
                logger.info(f"SUCESSO! Resultados copiados (rclone) para: {remote_path}")
                shutil.rmtree(project_root / "data" / "temp_results")
                return

        # Fallback ou Windows
        if not dest_dir.exists():
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
    parser.add_argument("run_type", type=str, nargs='?', choices=["foundation", "specialized"], help="Tipo de exportação.")
    
    args = parser.parse_args()

    if args.clean:
        cleanup_workspace()
    
    if args.log_filename and args.run_type:
        from src.cloud.base_model.utils.logging_utils import setup_logger
        setup_logger("transfer", f"_{args.run_type}")
        transfer_results(args.log_filename, args.run_type)
    elif not args.clean:
        parser.print_help()
