
import shutil
import os
import yaml
import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

def transfer_results(log_filename: str):
    """
    Coleta todos os artefatos de um fine-tuning (logs, db, modelos, configs)
    e envia para a pasta de resultados no Google Drive.
    """
    # 1. Configurações de Caminhos
    # O arquivo está em src/cloud/base_model/utils/transfer.py (depth 4)
    project_root = Path(__file__).parents[4]
    log_path = project_root / "logs" / "optimization" / log_filename
    
    if not log_path.exists():
        print(f"❌ Erro: Log '{log_filename}' não encontrado em {log_path.parent}")
        return

    # Nome da pasta de destino (Removendo .log)
    folder_name = log_filename.replace(".log", "")
    
    # Definir base do Drive (Detecta se Windows ou Linux/Pod)
    if os.name == 'nt':
        drive_base = Path("Z:/PROJETOS/RESULTADOS")
    else:
        # No Pod, geralmente montado em /workspace/drive ou similar
        drive_base = Path("/workspace/drive/PROJETOS/RESULTADOS")
    
    dest_dir = drive_base / folder_name
    print(f"🚀 Iniciando transferência para: {dest_dir}")

    # 2. Ler optimization_config.yaml para identificar o DB e caminhos
    opt_config_path = project_root / "src/cloud/base_model/otimizacao/optimization_config.yaml"
    if not opt_config_path.exists():
        print(f"⚠️ Aviso: Config de otimização não encontrada em {opt_config_path}")
        opt_cfg = {'paths': {'db_path': 'sqlite:///optuna_tcn_lstm_v8.db'}}
    else:
        with open(opt_config_path, 'r') as f:
            opt_cfg = yaml.safe_load(f)
    
    db_uri = opt_cfg['paths'].get('db_path', 'sqlite:///optuna_tcn_lstm_v8.db')
    db_filename = db_uri.replace("sqlite:///", "")
    db_path = project_root / db_filename

    # 3. Lista de arquivos a coletar
    files_to_transfer = [
        log_path,
        db_path,
        project_root / "data/models/best_tcn_lstm.pt",
        project_root / "data/models/best_tcn_lstm_dir.pt",
        project_root / "src/cloud/base_model/otimizacao/best_params.json",
        project_root / "src/cloud/base_model/otimizacao/best_dir_params.json",
        opt_config_path,
        project_root / "src/cloud/base_model/treino/training_config.yaml",
        project_root / "src/cloud/base_model/configs/base_model_config.yaml",
    ]

    models_dir = project_root / "data/models"
    if models_dir.exists():
        files_to_transfer.extend(list(models_dir.glob("*.pkl")))

    # 4. Executar Transferência
    try:
        # No Linux/Pod, tentamos rclone copy direto para o remote para evitar lag de sincronização
        if os.name != 'nt':
            # Criar pasta temporária local para agrupar tudo
            temp_staging = project_root / "data" / "temp_results"
            if temp_staging.exists(): shutil.rmtree(temp_staging)
            temp_staging.mkdir(parents=True)

            print(f"📦 Agrupando arquivos localmente em {temp_staging}...")
            for src in files_to_transfer:
                if src.exists():
                    shutil.copy2(src, temp_staging / src.name)

            # Tenta rclone copy para o remote 'drive:'
            # Busca o rclone.conf na raiz do projeto (detectado via ls)
            rclone_cfg = project_root / "rclone.conf"
            print(f"📡 Enviando via rclone direto para o Drive...")
            remote_path = f"drive:PROJETOS/RESULTADOS/{folder_name}"
            
            cmd = ["rclone", "copy", str(temp_staging), remote_path, "-P"]
            if rclone_cfg.exists():
                cmd += ["--config", str(rclone_cfg)]
                print(f"   🔧 Usando config: {rclone_cfg}")
            
            result = subprocess.run(cmd)

            if result.returncode == 0:
                print(f"✅ SUCESSO! Resultados enviados via rclone para {remote_path}")
                shutil.rmtree(temp_staging)
                return
            else:
                print("⚠️ Falha no 'rclone copy' direto. Tentando cópia via pasta montada (shutil)...")

        # Fallback ou Windows (shutil tradicional)
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            
        for src in files_to_transfer:
            if src.exists():
                print(f"   📂 Copiando: {src.name}...")
                shutil.copy2(src, dest_dir / src.name)
            else:
                print(f"   ⚠️  Pulo: {src.name} não encontrado.")

        print(f"\n✅ SUCESSO! Resultados copiados para {dest_dir}")

    except Exception as e:
        print(f"❌ Erro na transferência: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        transfer_results(sys.argv[1])
    else:
        print("Uso: python transfer_results.py <nome_do_log.log>")
