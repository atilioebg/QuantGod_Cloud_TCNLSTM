import shutil
import os
import yaml
import logging
from pathlib import Path
import subprocess
import sys
import argparse

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

    print("\n🧹 INICIANDO LIMPEZA DO WORKSPACE...")
    for target in targets:
        if target.exists():
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                    print(f"   🗑️  Removido diretório: {target.relative_to(project_root)}")
                else:
                    target.unlink()
                    print(f"   🗑️  Removido arquivo: {target.relative_to(project_root)}")
            except Exception as e:
                print(f"   ⚠️ Erro ao remover {target}: {e}")
    
    # Recriar estrutura mínima necessária
    (project_root / "logs").mkdir(exist_ok=True)
    (project_root / "data/L2/raw").mkdir(parents=True, exist_ok=True)
    (project_root / "data/L2/pre_processed").mkdir(parents=True, exist_ok=True)
    print("✨ Workspace limpo e resetado! Estrutura base recriada.\n")

def transfer_results(log_filename: str, run_type: str):
    """
    Coleta os artefatos baseados no tipo de corrida ('foundation' ou 'specialized')
    e envia para a pasta de resultados hierárquica no Google Drive.
    """
    if run_type not in ["foundation", "specialized"]:
        print(f"❌ Erro: Tipo inválido '{run_type}'. Use 'foundation' ou 'specialized'.")
        return

    # 1. Configurações de Caminhos Base
    project_root = Path(__file__).parents[4]
    
    # Nome da pasta de destino raiz
    folder_name = log_filename.replace(".log", "")
    
    # Definir base do Drive (Detecta se Windows ou Linux/Pod)
    if os.environ.get('MOCK_DRIVE') == '1':
        drive_base = Path("mock_drive_results")
    elif os.name == 'nt':
        drive_base = Path("Z:/PROJETOS/RESULTADOS")
    else:
        drive_base = project_root / "drive" / "PROJETOS" / "RESULTADOS"
    
    # Destino Final Específico do Tipo
    dest_dir = drive_base / folder_name / run_type
    print(f"🚀 Iniciando transferência [{run_type.upper()}] para: {dest_dir}")

    # Lista Base (Configs comuns que vão para ambos garantindo autonomia)
    opt_config_path = project_root / "src/cloud/base_model/otimizacao/optimization_config.yaml"
    
    files_to_transfer = [
        project_root / "src/cloud/base_model/configs/base_model_config.yaml",
        opt_config_path,
        project_root / "src/cloud/base_model/treino/training_config.yaml",
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
            
        # Logs de ETL, Labelling e Tests
        log_patterns = ["logs/etl/*.log", "logs/labelling/*.log", "logs/tests/*.log"]
        for pattern in log_patterns:
            files_to_transfer.extend(list(project_root.glob(pattern)))
            
        # Reports de Auditoria
        files_to_transfer.extend(list((project_root / "docs/reports").glob("*.md")))
        
        # ── DATABASE & MODELS ─────────────────────────────────────────────────
        if opt_config_path.exists():
            with open(opt_config_path, 'r') as f:
                opt_cfg = yaml.safe_load(f)
            db_uri = opt_cfg['paths'].get('db_path', 'sqlite:///optuna_tcn_lstm_v0.db')
            db_filename = db_uri.replace("sqlite:///", "")
            files_to_transfer.append(project_root / db_filename)
            
        files_to_transfer.extend([
            project_root / "data" / "models" / "best_tcn_lstm.pt",
            project_root / "data" / "models" / "best_tcn_lstm_dir.pt",
            project_root / "data" / "models" / "scaler_finetuning.pkl",
            project_root / "data" / "models" / "scaler_finetuning_dir.pkl"
        ])
        
    elif run_type == "specialized":
        # Pega logs de specialization
        spec_logs_dir = project_root / "logs" / "treino_specialization"
        if spec_logs_dir.exists():
            logs = sorted(list(spec_logs_dir.glob("*.log")))
            if logs:
                 files_to_transfer.append(logs[-1])

        # Modelos e Scalers do Especialista
        files_to_transfer.extend([
            project_root / "data" / "models" / "treino_best_model.pt",
            project_root / "data" / "models" / "treino_scaler_finetuning.pkl"
        ])

    # ── Executar Transferência ───────────────────────────────────────────────
    try:
        # Remover duplicatas e arquivos inexistentes
        files_to_transfer = sorted(list(set([f for f in files_to_transfer if f.exists()])))

        if os.name != 'nt':
            temp_staging = project_root / "data" / "temp_results" / run_type
            if temp_staging.exists():
                shutil.rmtree(temp_staging)
            temp_staging.mkdir(parents=True)

            print(f"📦 Agrupando {len(files_to_transfer)} arquivos em {temp_staging.relative_to(project_root)}...")
            for src in files_to_transfer:
                shutil.copy2(src, temp_staging / src.name)

            rclone_cfg = project_root / "rclone.conf"
            remote_path = f"drive:PROJETOS/RESULTADOS/{folder_name}/{run_type}"
            
            cmd = ["rclone", "copy", str(temp_staging), remote_path, "-P"]
            if rclone_cfg.exists():
                cmd += ["--config", str(rclone_cfg)]
            
            result = subprocess.run(cmd)

            if result.returncode == 0:
                print(f"✅ SUCESSO! Resultados copiados (rclone) para: {remote_path}")
                shutil.rmtree(project_root / "data" / "temp_results")
                return

        # Fallback ou Windows
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            
        for src in files_to_transfer:
            print(f"   📂 Copiando: {src.name}...")
            shutil.copy2(src, dest_dir / src.name)

        print(f"\n✅ SUCESSO! Resultados copiados localmente para: {dest_dir}")

    except Exception as e:
        print(f"❌ Erro na transferência: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerenciador de Resultados e Workspace QuantGod.")
    parser.add_argument("--clean", action="store_true", help="Limpa o workspace e as pastas de dados/logs.")
    parser.add_argument("log_filename", type=str, nargs='?', help="Nome do log base para exportação.")
    parser.add_argument("run_type", type=str, nargs='?', choices=["foundation", "specialized"], help="Tipo de exportação.")
    
    args = parser.parse_args()

    if args.clean:
        cleanup_workspace()
    
    if args.log_filename and args.run_type:
        transfer_results(args.log_filename, args.run_type)
    elif not args.clean:
        parser.print_help()
