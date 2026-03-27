import optuna
import sys

def delete_last_n_trials(db_path, study_name, n):
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # Pegar todos os trials ordenados por número
    trials = sorted(study.trials, key=lambda t: t.number)
    
    if not trials:
        print("No trials found in the study.")
        return

    # Identificar os últimos N trials
    to_delete = trials[-n:]
    trial_numbers = [t.number for t in to_delete]
    
    print(f"Total trials: {len(trials)}")
    print(f"Trials to delete: {trial_numbers}")
    
    confirm = input(f"Are you sure you want to delete these {len(to_delete)} trials? (y/n): ")
    if confirm.lower() == 'y':
        for t_num in trial_numbers:
            # Nota: delete_trial não existe diretamente no objeto study em versões antigas, 
            # mas podemos usar o storage.
            study._storage.delete_trial(to_delete[trial_numbers.index(t_num)]._trial_id)
            print(f"Deleted trial {t_num}")
        print("Done.")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    DB_PATH = "optuna_finetune.db"
    STUDY_NAME = "QUANTGOD_RECON_V009"
    N = 30
    delete_last_n_trials(DB_PATH, STUDY_NAME, N)
