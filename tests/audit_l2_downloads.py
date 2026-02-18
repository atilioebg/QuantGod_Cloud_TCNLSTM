import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURAÇÃO DE AUDITORIA (Data QA)
# ============================================================================
TARGET_YEAR = 2026  # Altere para o ano desejado (2023, 2024, 2025, 2026)
MIN_FILE_SIZE_KB = 100  # Limite para considerar o arquivo como suspeito/vazio

# Configuração de Datas Base (Regras de Negócio Bybit)
if TARGET_YEAR == 2023:
    START_DATE = datetime(2023, 1, 19)
else:
    START_DATE = datetime(TARGET_YEAR, 1, 1)

# Se o ano for o atual, o fim é ontem, caso contrário 31/12
CURRENT_DATE = datetime.now()
if TARGET_YEAR < CURRENT_DATE.year:
    END_DATE = datetime(TARGET_YEAR, 12, 31)
else:
    END_DATE = CURRENT_DATE - timedelta(days=1)

# Diretório de Origem
BASE_DIR = Path(fr"C:\Users\Atilio\Downloads\btcustd_L2_{TARGET_YEAR}")

# ============================================================================

def audit_l2_data():
    print(f"\n{'='*60}")
    print(f"📊 AUDITORIA DE INTEGRIDADE L2 - ANO {TARGET_YEAR}")
    print(f"Diretório: {BASE_DIR}")
    print(f"Período: {START_DATE.strftime('%d/%m/%Y')} até {END_DATE.strftime('%d/%m/%Y')}")
    print(f"{'='*60}\n")

    if not BASE_DIR.exists():
        print(f"❌ ERRO: O diretório {BASE_DIR} não existe.")
        return

    # 1. Gerar Range de Datas Ideal
    ideal_dates = []
    curr = START_DATE
    while curr <= END_DATE:
        ideal_dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    # Containers de Resultados
    valid_files = []
    missing_dates = []
    corrupted_files = []
    small_files = []

    # 2. Varredura e Validação
    print("🔍 Escaneando sequência e integridade...")
    for date_str in tqdm(ideal_dates, desc="Progresso", unit="dia"):
        # Tentar múltiplos padrões comuns (Bybit varia entre ob500, ob200 e outros)
        patterns = [
            f"{date_str}_BTCUSDT_ob200.data.zip",
            f"{date_str}_BTCUSDT_ob500.data.zip",
            f"{date_str}_BTCUSDT_0400.data.zip",
            f"{date_str}_BTCUSDT_ob400.data.zip"
        ]

        file_path = None
        filename = None
        
        # Tenta encontrar o arquivo existente entre os padrões
        for p in patterns:
            target = BASE_DIR / p
            if target.exists():
                filename = p
                file_path = target
                break
        
        # A. Checar Existência
        if not file_path:
            missing_dates.append(date_str)
            print(f"❌ AUSENTE: {date_str} (Nenhum padrão encontrado)")
            continue

        # B. Checar Tamanho (Anomalia de Download)
        size_kb = file_path.stat().st_size / 1024
        if size_kb < MIN_FILE_SIZE_KB:
            small_files.append(f"{filename} ({size_kb:.2f} KB)")
            print(f"📉 PEQUENO: {filename} ({size_kb:.2f} KB)")
            continue

        # C. Checar Integridade do ZIP
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # O comando testzip() verifica o CRC de cada arquivo e o header do ZIP
                bad_file = zf.testzip()
                if bad_file:
                    corrupted_files.append(f"{filename} (Interno: {bad_file})")
                    print(f"⚠️ CORROMPIDO: {filename}")
                else:
                    valid_files.append(filename)
        except zipfile.BadZipFile:
            corrupted_files.append(f"{filename} (Cabeçalho Corrompido)")
            print(f"💥 CORROMPIDO (Header): {filename}")
        except Exception as e:
            corrupted_files.append(f"{filename} (Erro Inesperado: {str(e)})")
            print(f"❓ ERRO: {filename} - {str(e)}")

    # 3. Relatório Final
    print(f"\n{'-'*60}")
    print("📋 RELATÓRIO FINAL DE DATA QUALITY ASSURANCE")
    print(f"{'-'*60}")
    
    print(f"✅ Arquivos Válidos: {len(valid_files)}")
    print(f"❌ Arquivos Ausentes: {len(missing_dates)}")
    print(f"⚠️ Arquivos Corrompidos: {len(corrupted_files)}")
    print(f"📉 Arquivos Suspeitos (Pequenos): {len(small_files)}")
    
    if missing_dates:
        print("\n📅 DATAS AUSENTES (Missing Sequence):")
        for d in missing_dates:
            print(f"  - {d}")

    if corrupted_files:
        print("\n💥 ARQUIVOS CORROMPIDOS (Recomenda-se Re-download):")
        for f in corrupted_files:
            print(f"  - {f}")

    if small_files:
        print("\n🤏 ARQUIVOS MUITO PEQUENOS (Possível erro no servidor Bybit/Download Zero):")
        for f in small_files:
            print(f"  - {f}")

    print(f"\n{'='*60}")
    if len(valid_files) == len(ideal_dates):
        print("🏆 INTEGRIDADE TOTAL: O dataset está completo e saudável!")
    else:
        print("💡 AÇÃO SUGERIDA: Corrija os erros listados antes de processar os dados.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    audit_l2_data()
