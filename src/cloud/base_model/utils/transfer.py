
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
    project_root = Path(__file__).parents[3]
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
    with open(opt_config_path, 'r') as f:
        opt_cfg = yaml.safe_load(f)
    
    db_uri = opt_cfg['paths']['db_path']
    db_filename = db_uri.replace("sqlite:///", "")
    db_path = project_root / db_filename

    # 3. Lista de arquivos a coletar
    files_to_transfer = [
        # Logs
        log_path,
        
        # Banco de Dados de Otimização
        db_path,
        
        # Modelos (Macro e Direcional)
        project_root / "data/models/best_tcn_lstm.pt",
        project_root / "data/models/best_tcn_lstm_dir.pt",
        
        # Parâmetros JSON
        project_root / "src/cloud/base_model/otimizacao/best_params.json",
        project_root / "src/cloud/base_model/otimizacao/best_dir_params.json",
        
        # Configurações usadas
        opt_config_path,
        project_root / "src/cloud/base_model/treino/training_config.yaml",
        project_root / "src/cloud/base_model/configs/base_model_config.yaml",
    ]

    # Adicionar todos os .pkl (scalers) e outros .pt encontrados em data/models
    # que possam ter sido gerados
    models_dir = project_root / "data/models"
    if models_dir.exists():
        files_to_transfer.extend(list(models_dir.glob("*.pkl")))

    # 4. Executar a cópia
    # Se o Drive estiver montado, usamos shutil. Se não, poderíamos usar rclone direto.
    # Vou usar uma abordagem híbrida: tenta criar a pasta e copiar.
    try:
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            
        for src in files_to_transfer:
            if src.exists():
                print(f"   📦 Copiando: {src.name}...")
                shutil.copy2(src, dest_dir / src.name)
            else:
                print(f"   ⚠️  Aviso: Arquivo {src.name} não encontrado, pulando...")

        print(f"\n✅ SUCESSO! Todos os resultados foram transferidos para:")
        print(f"📍 {dest_dir}")

    except Exception as e:
        print(f"❌ Erro crítico na transferência: {e}")
        print("Dica: Verifique se o rclone está montado corretamente no caminho esperado.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        transfer_results(sys.argv[1])
    else:
        print("Uso: python transfer_results.py <nome_do_log.log>")
