import shutil
import os
import yaml
import logging
from pathlib import Path
import subprocess
import sys
import argparse

logger = logging.getLogger(__name__)

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
    if os.name == 'nt':
        drive_base = Path("Z:/PROJETOS/RESULTADOS")
    else:
        drive_base = Path("/workspace/drive/PROJETOS/RESULTADOS")
    
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
        log_path = project_root / "logs" / "optimization" / log_filename
        if not log_path.exists():
             print(f"⚠️ Aviso: Log '{log_filename}' não encontrado em {log_path.parent}")
             
        files_to_transfer.append(log_path)
        
        # Identificar o DB Optuna
        if opt_config_path.exists():
            with open(opt_config_path, 'r') as f:
                opt_cfg = yaml.safe_load(f)
            db_uri = opt_cfg['paths'].get('db_path', 'sqlite:///optuna_tcn_lstm_v8.db')
            db_filename = db_uri.replace("sqlite:///", "")
            files_to_transfer.append(project_root / db_filename)
            
        # Adicionar Modelos e Scalers do Foundation
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
            # Pega o log com nome mais recente
            logs = sorted(list(spec_logs_dir.glob("*.log")))
            if logs:
                 files_to_transfer.append(logs[-1])
            else:
                 print(f"⚠️ Aviso: Nenhum log '.log' encontrado em {spec_logs_dir}")
        else:
             print(f"⚠️ Aviso: Diretório de logs {spec_logs_dir} não existe.")

        # Modelos e Scalers do Especialista
        files_to_transfer.extend([
            project_root / "data" / "models" / "treino_best_model.pt",
            project_root / "data" / "models" / "treino_scaler_finetuning.pkl"
        ])

    # 4. Executar Transferência
    try:
        if os.name != 'nt':
            # Construir localmente uma hierarquia exata refletindo a separação
            temp_staging = project_root / "data" / "temp_results" / run_type
            if temp_staging.exists():
                shutil.rmtree(temp_staging)
            temp_staging.mkdir(parents=True)

            print(f"📦 Agrupando arquivos localmente em {temp_staging}...")
            for src in files_to_transfer:
                if src.exists():
                    shutil.copy2(src, temp_staging / src.name)
                else:
                    print(f"   ⚠️  Arquivo não encontrado (esquipado): {src.name}")

            rclone_cfg = project_root / "rclone.conf"
            print(f"📡 Enviando via rclone direto para o Drive...")
            # O remote_path apontará direto para a pasta hierarquica do tipo
            remote_path = f"drive:PROJETOS/RESULTADOS/{folder_name}/{run_type}"
            
            cmd = ["rclone", "copy", str(temp_staging), remote_path, "-P"]
            if rclone_cfg.exists():
                cmd += ["--config", str(rclone_cfg)]
                print(f"   🔧 Usando config: {rclone_cfg}")
            
            result = subprocess.run(cmd)

            if result.returncode == 0:
                print(f"✅ SUCESSO! Resultados copiados (rclone) para: {remote_path}")
                shutil.rmtree(project_root / "data" / "temp_results") # Limpa tudo
                return
            else:
                print("⚠️ Falha no 'rclone copy' direto. Tentando fallback local (shutil)...")

        # Fallback ou Windows (shutil tradicional)
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            
        for src in files_to_transfer:
            if src.exists():
                print(f"   📂 Copiando: {src.name}...")
                shutil.copy2(src, dest_dir / src.name)
            else:
                print(f"   ⚠️  Arquivo não encontrado (esquipado): {src.name}")

        print(f"\n✅ SUCESSO! Resultados copiados localmente para: {dest_dir}")

    except Exception as e:
        print(f"❌ Erro na transferência: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfere arquivos do pipeline ML para o Google Drive.")
    parser.add_argument("log_filename", type=str, help="Nome do log base (usado para gerar a pasta mãe).")
    parser.add_argument("run_type", type=str, choices=["foundation", "specialized"], help="Tipo de arquitetura sendo armezanada.")
    
    args = parser.parse_args()
    transfer_results(args.log_filename, args.run_type)
