import os
import shutil
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cloud.base_model.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

def create_specialized_splits(source_dir: str):
    """
    Agrega os dados de `val` e `test` da pasta base,
    ordena cronologicamente pelo nome do arquivo
    e gera uma nova pasta de 'specialized_splits_...' fatiada em 85% (Train) e 15% (Val).
    """
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        logger.error(f"❌ Source directory '{source_path}' does not exist.")
        return

    val_dir  = source_path / "val"
    test_dir = source_path / "test"

    if not val_dir.exists() or not test_dir.exists():
        logger.error(f"❌ Could not find both 'val' and 'test' inside {source_path}")
        return

    # 1. Coletar todos os parquets
    all_files = []
    all_files.extend(list(val_dir.glob("*.parquet")))
    all_files.extend(list(test_dir.glob("*.parquet")))

    if not all_files:
        logger.error(f"❌ No parquet files found in {val_dir} or {test_dir}")
        return

    # 2. Ordenação Cronológica (Assumindo que o nome do file tem o timestamp: book_YYYYMMDD_...)
    all_files.sort(key=lambda x: x.name)
    total_files = len(all_files)
    logger.info(f"📊 Found {total_files} parquet files across 'val' and 'test'.")

    # 3. Calcular quebra de 85/15
    split_idx = int(0.85 * total_files)
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]

    logger.info(f"✂️  Splitting into {len(train_files)} (85%) for train and {len(val_files)} (15%) for val.")

    # 4. Criar estrutura do destino
    target_path = source_path.parent / f"specialized_{source_path.name}"
    
    target_train = target_path / "train"
    target_val   = target_path / "val"

    # Wipe destination if exists to ensure clean slate
    if target_path.exists():
        logger.warning(f"🧹 Clearing existing target directory: {target_path}")
        shutil.rmtree(target_path)
        
    target_train.mkdir(parents=True, exist_ok=True)
    target_val.mkdir(parents=True, exist_ok=True)

    # 5. Copiar (Usando Hardlinks para economizar espaço e bater de forma instantânea, com fallback visual)
    logger.info(f"📁 Populating {target_train}...")
    for f in train_files:
        dest = target_train / f.name
        try:
            os.link(src=f, dst=dest) # Hardlink (super veloz e zero disco adicional)
        except Exception:
            shutil.copy2(f, dest)

    logger.info(f"📁 Populating {target_val}...")
    for f in val_files:
        dest = target_val / f.name
        try:
            os.link(src=f, dst=dest)
        except Exception:
            shutil.copy2(f, dest)

    logger.info(f"🚀✅ SUCESSO! Specialized splits created safely at:\n   {target_path}")

if __name__ == "__main__":
    setup_logger("specialized_split_creator", "")
    if len(sys.argv) < 2:
        logger.error("Usage: python create_specialized_splits.py <caminho_para_splits_labelled_...>")
        sys.exit(1)
        
    source = sys.argv[1]
    create_specialized_splits(source)
