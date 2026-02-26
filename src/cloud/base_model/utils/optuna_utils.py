import optuna
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def export_landscape_to_csv(db_path, study_name, output_path):
    """
    Exports the Optuna study trials to a formatted CSV file with parameter importance.
    This version for the project includes full precision formatting and specific column naming rules.
    """
    # Force sqlite prefix if not present
    if not db_path.startswith("sqlite:///"):
        db_url = f"sqlite:///{db_path}"
    else:
        db_url = db_path
    
    # Extract filename for existence check if it was a URL
    db_filename = db_path.replace("sqlite:///", "")
    if not os.path.exists(db_filename):
        logger.warning(f"Database not found at {db_filename}. Skipping landscape export.")
        return False

    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
        
        # Identify importance of parameters
        try:
            importance = optuna.importance.get_param_importances(study)
            sorted_params = [f"params_{p}" for p in importance.keys()]
        except Exception as e:
            logger.warning(f"Could not calculate importance: {e}")
            sorted_params = [col for col in study.trials_dataframe().columns if col.startswith('params_')]

        # Get the dataframe
        df = study.trials_dataframe(multi_index=False)
        
        # 1. Rename specific metric columns and capitalize others to match user style
        rename_rules = {
            'number': 'Number',
            'value': 'F1_macro',
            'user_attrs_best_f1_dir': 'F1_dir',
            'state': 'State'
        }
        df = df.rename(columns={k: v for k, v in rename_rules.items() if k in df.columns})
        
        # Capitalize Params_ prefix
        df.columns = [c.replace('params_', 'Params_') for c in df.columns]
        sorted_params = [p.replace('params_', 'Params_') for p in sorted_params]

        # 2. Identify the new primary metric column name for sorting
        main_metric = 'F1_macro' if 'F1_macro' in df.columns else 'value'

        # 3. Sort rows from highest value to lowest
        df = df.sort_values(by=main_metric, ascending=False)
        
        # 4. Remove unwanted time/date columns
        cols_to_remove = ['duration', 'datetime_start', 'datetime_complete']
        df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])
        
        # 5. ENSURE NUMERIC PRECISION: Force conversion to numeric for metrics and params
        cols_to_fix = ['F1_macro', 'F1_dir'] + [c for c in df.columns if c.startswith('Params_')]
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Format float columns to 8 decimal places
        float_cols = df.select_dtypes(include=['float64', 'float32']).columns
        df[float_cols] = df[float_cols].round(8)
        
        # 7. Reorder columns
        fixed_cols = ['Number', 'F1_macro', 'F1_dir', 'State']
        actual_fixed = [c for c in fixed_cols if c in df.columns]
        actual_params = [p for p in sorted_params if p in df.columns]
        remaining_cols = [c for c in df.columns if c not in actual_fixed and c not in actual_params]
        
        new_column_order = actual_fixed + actual_params + remaining_cols
        df = df[new_column_order]
        
        # Write to CSV with fixed point notation for all floats
        df.to_csv(output_path, index=False, float_format='%.8f')
        return True
    except Exception as e:
        logger.error(f"Error exporting Optuna landscape: {e}")
        return False
