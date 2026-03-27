import optuna
from optuna.storages import RDBStorage

def trim_study(db_path, study_name, budget):
    storage_url = f"sqlite:///{db_path}"
    # v2.0: Usar RDBStorage diretamente para evitar o wrapper _CachedStorage do Study
    storage = RDBStorage(storage_url)
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
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
    
    confirm = input(f"Tem certeza que deseja deletar esses {len(trials_to_delete)} trials excedentes? (y/n): ")
    if confirm.lower() == 'y':
        for t in trials_to_delete:
            # Deletar via storage direto usando o ID interno
            storage.delete_trial(t._trial_id)
            print(f"Deletado Trial {t.number}")
        print(f"✅ Limpeza concluida. O banco agora tem exatamente {len(study.get_trials())} trials.")
    else:
        print("Operação cancelada.")

if __name__ == "__main__":
    DB_PATH = "optuna_finetune.db"
    STUDY_NAME = "QUANTGOD_RECON_V009"
    BUDGET = 400
    trim_study(DB_PATH, STUDY_NAME, BUDGET)
