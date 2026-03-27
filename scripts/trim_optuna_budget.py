import optuna
import sys

def trim_study(db_path, study_name, budget):
    storage_url = f"sqlite:///{db_path}"
    
    try:
        # Carregar o estudo via storage
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # Identificar trials excedentes
    all_trials = study.get_trials()
    trials_to_delete = [t for t in all_trials if t.number >= budget]
    
    if not trials_to_delete:
        print(f"Nenhum trial acima de {budget-1} encontrado no banco.")
        return

    print(f"Total de trials no banco: {len(all_trials)}")
    print(f"Número de trials para deletar: {len(trials_to_delete)}")
    print(f"Trials que serão removidos: IDs {trials_to_delete[0].number} a {trials_to_delete[-1].number}")
    
    confirm = input(f"Tem certeza que deseja deletar esses {len(trials_to_delete)} trials excedentes? (y/n): ")
    if confirm.lower() == 'y':
        # v3.0: Usar a API oficial super-segura do Optuna 3.1+
        # Ela deleta tanto o trial quanto seus parâmetros/user_attrs/etc automaticamente.
        try:
            for t in trials_to_delete:
                study.delete_trial(t.number)
                print(f"Deletado Trial {t.number}")
            print(f"✅ Limpeza concluida. O banco agora tem exatamente {len(study.get_trials())} trials.")
        except AttributeError:
             # Caso a versão seja muito antiga, tenta via SQL direto (Plano C)
             print("❌ API study.delete_trial não encontrada. Tentando limpeza via SQL...")
             import sqlite3
             conn = sqlite3.connect(db_path)
             cursor = conn.cursor()
             # Deletar de todas as tabelas relacionadas
             cursor.execute("SELECT trial_id FROM trials WHERE number >= ?", (budget,))
             ids = [row[0] for row in cursor.fetchall()]
             if ids:
                 for t_id in ids:
                     cursor.execute("DELETE FROM trial_params WHERE trial_id = ?", (t_id,))
                     cursor.execute("DELETE FROM trial_values WHERE trial_id = ?", (t_id,))
                     cursor.execute("DELETE FROM trial_user_attributes WHERE trial_id = ?", (t_id,))
                     cursor.execute("DELETE FROM trial_system_attributes WHERE trial_id = ?", (t_id,))
                     cursor.execute("DELETE FROM trials WHERE trial_id = ?", (t_id,))
                 conn.commit()
                 print(f"✅ Limpeza SQL concluida para {len(ids)} trials.")
             conn.close()
    else:
        print("Operação cancelada.")

if __name__ == "__main__":
    DB_PATH = "optuna_finetune.db"
    STUDY_NAME = "QUANTGOD_RECON_V009"
    BUDGET = 400
    trim_study(DB_PATH, STUDY_NAME, BUDGET)
