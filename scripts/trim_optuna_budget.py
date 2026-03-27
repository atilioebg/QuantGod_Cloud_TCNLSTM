import optuna
import sys

def trim_study(db_path, study_name, budget):
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # Encontrar trials excedentes (número >= budget, assumindo que começa no 0)
    trials_to_delete = [t for t in study.trials if t.number >= budget]
    
    if not trials_to_delete:
        print(f"Nenhum trial acima de {budget-1} encontrado no banco.")
        return

    print(f"Total de trials no banco: {len(study.trials)}")
    print(f"Número de trials para deletar: {len(trials_to_delete)}")
    print(f"Trials afetados: {trials_to_delete[0].number} até {trials_to_delete[-1].number}")
    
    confirm = input(f"Tem certeza que deseja deletar esses {len(trials_to_delete)} trials excedentes? (y/n): ")
    if confirm.lower() == 'y':
        for t in trials_to_delete:
            study._storage.delete_trial(t._trial_id)
            print(f"Deletado Trial {t.number}")
        print("✅ Limpeza concluida. O banco agora tem exatamente 400 trials (ID 0 a 399).")
    else:
        print("Operação cancelada.")

if __name__ == "__main__":
    DB_PATH = "optuna_finetune.db"
    STUDY_NAME = "QUANTGOD_RECON_V009"
    BUDGET = 400
    trim_study(DB_PATH, STUDY_NAME, BUDGET)
