import optuna
import numpy as np
import scipy.stats as stats
import pandas as pd
from pathlib import Path

def calculate_deflated_metrics(db_path, study_name):
    # 1. Load Study
    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # 2. Extract Completed Trials
    trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not trials:
        print("No completed trials found.")
        return

    # Extrair valores da métrica principal (F1 Macro ou Direcional conforme configurado no return do objective)
    f1_values = [t.value for t in trials if t.value is not None]
    
    # Extrair especificamente o F1 Direcional (salvo nos user_attrs)
    f1_dir_values = [t.user_attrs.get("best_f1_dir", 0.0) for t in trials if "best_f1_dir" in t.user_attrs]

    if not f1_dir_values:
         # Fallback para os valores principais se os attrs não existirem
         f1_dir_values = f1_values

    # 3. Basic Statistics
    mean_f1 = np.mean(f1_dir_values)
    std_f1 = np.std(f1_dir_values)
    max_f1 = np.max(f1_dir_values)
    n_trials = len(f1_dir_values)

    # 4. Statistical Proof: Z-Score & P-Value (Normal Distribution Assumption)
    # Z = (X - mu) / sigma
    z_score = (max_f1 - mean_f1) / std_f1 if std_f1 > 0 else 0
    
    # Probabilidade de ser sorte em 1 tentativa
    p_single = 1 - stats.norm.cdf(z_score)
    
    # Bonferroni Correction: p_corrected = p_single * n_trials
    p_corrected = p_single * n_trials

    # 5. Robustness: Clumping (Top 10%)
    sorted_f1 = sorted(f1_dir_values, reverse=True)
    top_10 = sorted_f1[:10]
    
    # 6. Report
    print("="*60)
    print("📈 ESTATÍSTICA DE VALIDAÇÃO: QUANTGOD TRIAL 299")
    print("="*60)
    print(f"Total de Trials Analisados: {n_trials}")
    print(f"F1 Direcional Médio:       {mean_f1:.6f}")
    print(f"Desv. Padrão (Noise):      {std_f1:.6f}")
    print(f"F1 do Campeão (Trial 299): {max_f1:.6f}")
    print("-" * 30)
    print("TOP 10 F1 DIRECIONAL (CLUMPING CHECK):")
    for i, val in enumerate(top_10):
        print(f"  {i+1}º: {val:.6f}")
    print("-" * 30)
    print(f"Z-Score (Desvios da Média): {z_score:.2f} σ")
    print(f"P-Value (Single):          {p_single:.4e}")
    print(f"P-Value (Bonferroni Corr): {p_corrected:.4e}")
    print("-" * 30)
    
    # Analysis based on Top 10
    top_avg = np.mean(top_10)
    if top_avg > (mean_f1 + 1.2 * std_f1):
        print("✅ REGIME DE ALTA PERFORMANCE IDENTIFICADO")
        print("Existem múltiplos trials com F1 consistentemente alto.")
        print("Isso sugere que o 0.50 não é um outlier isolado, mas uma")
        print("região de hiperparâmetros que funciona.")
    else:
        print("⚠️ OUTLIER ISOLADO DETECTADO")
        print("A distância entre o 1º e os demais sugere sorte pontual.")
    print("="*60)

if __name__ == "__main__":
    DB_PATH = r"C:\Users\Atilio\Desktop\PROJETOS\PESSOAL\QuantGod_Cloud_TCNLSTM\optuna_finetune.db"
    # O nome do study deve bater com o master_config.yaml
    STUDY_NAME = "QUANTGOD_RECON_V009" 
    
    calculate_deflated_metrics(DB_PATH, STUDY_NAME)
