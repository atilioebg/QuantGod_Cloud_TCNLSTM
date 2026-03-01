import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate_integrity(df: pd.DataFrame, name: str = "Dataset", feature_list: list = None) -> dict:
        """
        Performs basic integrity checks and returns a structured quality report.
        Discriminates between [DNN_INPUT], [XGB_ONLY] and [RAW_DATA] for lineage transparency.
        
        Args:
            df: The dataframe to validate.
            name: Label for the dataset.
            feature_list: The list of official features (from feature_names or auditor_features).
        """
        logger.info(f"--- Validating {name} ---")
        
        feature_list = feature_list or []
        expected_count = len(feature_list)
        
        report = {
            'is_valid': True,
            'nan_count': 0,
            'inf_count': 0,
            'dead_features': [],
            'dead_features_lineage': {},
            'high_tail_count': 0,
            'high_tails_detail': {},
            'stale_data_detected': False,
            'max_gap_minutes': 0.0,
            'shape_integrity': True,
            'lineage_summary': {
                '[DNN_INPUT]': 0,
                '[XGB_ONLY]': 0,
                '[RAW_DATA]': 0
            }
        }

        # 0. Check for Empty Dataset (FATAL)
        if df.empty:
            logger.error(f"❌ ARCHITECTURE INTEGRITY FAILURE: Dataset {name} is empty.")
            report['is_valid'] = False
            report['shape_integrity'] = False
            report['integrity_comment'] = "Empty dataset (Level 1 Healing failed or no data found)."
            return report

        # 0.1 Check for Duplicate Columns (FATAL)
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            msg = f"❌ FATAL: Dataset {name} has duplicate columns: {dupes}"
            logger.error(msg)
            raise ValueError(msg)
        
        # Lineage Mapping Logic (v4.6 Gold)
        def get_lineage(col):
            if col in feature_list:
                # XGB_ONLY features are usually logits or specialized model outputs
                if any(x in col.lower() for x in ['logit', 'prob_', 'prediction', 'conf_']):
                    return "[XGB_ONLY]"
                return "[DNN_INPUT]"
            return "[RAW_DATA]"

        # Populate Lineage Summary
        for col in df.columns:
            l = get_lineage(col)
            report['lineage_summary'][l] += 1

        # 0.1 Architecture Integrity Check (SHAPE)
        # We verify that ALL expected features are present.
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            logger.error(f"❌ ARCHITECTURE INTEGRITY FAILURE: Missing {len(missing_features)} features: {missing_features}")
            report['shape_integrity'] = False
            report['is_valid'] = False
        
        # 0.2 Ghost Feature Detection (Strict Governance)
        # Any column that looks like a model output/signal but isn't in feature_list
        ghost_patterns = ['logit', 'prob_', 'prediction', 'conf_']
        ghost_features = [c for c in df.columns if c not in feature_list and any(p in c.lower() for p in ghost_patterns)]
        if ghost_features:
            logger.error(f"❌ ARCHITECTURE INTEGRITY FAILURE: Detected {len(ghost_features)} unauthorized ghost features: {ghost_features}")
            report['shape_integrity'] = False
            report['is_valid'] = False
            report['integrity_comment'] = f"Ghost features detected: {len(ghost_features)}"

        if report['shape_integrity']:
            total_cols = len(df.columns)
            raw_count = total_cols - expected_count - (1 if 'close' in df.columns else 0)
            logger.info(f"✅ ARCHITECTURE INTEGRITY: {expected_count}/{expected_count} features present. "
                        f"Audit Scope: {total_cols} columns ({expected_count} Features + {raw_count} Raw/Price).")

        # 1. Check for NaNs
        report['nan_count'] = int(df.isna().sum().sum())
        if report['nan_count'] > 0:
            logger.warning(f"Found {report['nan_count']} NaN values in {name}")
            report['is_valid'] = False
        else:
            logger.info("No NaNs found.")

        # 2. Check for Infs
        report['inf_count'] = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
        if report['inf_count'] > 0:
            logger.warning(f"Found {report['inf_count']} Infinite values in {name}")
            report['is_valid'] = False
        else:
            logger.info("No Infinite values found.")

        # 3. Strict Monotonicity Enforcement (FATAL)
        if not df.index.is_monotonic_increasing:
            msg = f"❌ FATAL: Dataset {name} is NOT chronologically sorted! Corruption detected."
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.info("Chronological order verified.")

        # 4. Stale Data Check (Feed Lock Detection)
        if len(df) > 5:
            price_static = (df['close'].diff() == 0).rolling(5).sum() == 5 if 'close' in df.columns else False
            vol_static = (df['log_volume'].diff() == 0).rolling(5).sum() == 5 if 'log_volume' in df.columns else False
            stale_indices = df.index[price_static & vol_static] if 'log_volume' in df.columns else df.index[price_static] if 'close' in df.columns else pd.Index([])
            if not stale_indices.empty:
                logger.warning(f"⚠️ STALE DATA ALERT: Possible feed lock detected in {name}")
                report['stale_data_detected'] = True

        # 5. Cross-Scale Validation (Mathematical Consistency)
        if 'ofi' in df.columns and 'ofi_delta_1' in df.columns:
            reconstructed_delta = df['ofi'].diff(1).fillna(0)
            check_val = (df['ofi_delta_1'].fillna(0) - reconstructed_delta).abs()
            diff_check = float(check_val.max().max() if isinstance(check_val, pd.DataFrame) else check_val.max())
            if diff_check > 1e-7:
                 logger.warning(f"⚠️ CROSS-SCALE INCONSISTENCY: ofi_delta_1 drift detected ({diff_check})")

        # 6. Distribution Sanity
        ratio_features = [
            'kyle_lambda', 'vpin_min25', 'bid_deep_ratio', 'ask_deep_ratio',
            'bid_convexity', 'ask_convexity', 'book_asymmetry_v5', 'max_spread', 'ofi'
        ]
        active_features = [f for f in ratio_features if f in df.columns]
        for feat in active_features:
            p99 = df[feat].quantile(0.99)
            max_val = df[feat].max()
            if p99 > 1e-6 and max_val > 15 * p99:
                lineage = get_lineage(feat)
                logger.warning(f"☢️ CRITICAL OUTLIER: {lineage} {feat} max ({max_val:.4e}) is > 15x P99. Clipping might have failed!")
            elif p99 > 1e-6 and max_val > 10.1 * p99:
                logger.debug(f"✅ Clipping verified for {feat}.")

        # 7. Z-Score Intensity (Tail Check)
        audit_cols = [c for c in df.columns if c in feature_list or any(x in c for x in ['close', 'volume'])]
        numeric_df = df[audit_cols].select_dtypes(include=[np.number])
        z_scores = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-9)
        
        high_tail_mask = z_scores.abs() > 12
        report['high_tail_count'] = int(high_tail_mask.sum().sum())
        if report['high_tail_count'] > 0:
            extreme_counts = high_tail_mask.sum()
            for col, count in extreme_counts[extreme_counts > 0].items():
                lineage = get_lineage(col)
                report['high_tails_detail'][col] = {'count': int(count), 'lineage': lineage}
                logger.warning(f"⚠️ HIGH TAIL INTENSITY: {lineage} {col} has {count} points with Z-Score > 12.")

        # 8. Zero-Variance Detection (Dead Features)
        std_zero = numeric_df.std()
        report['dead_features'] = std_zero[std_zero == 0].index.tolist()
        if report['dead_features']:
             for col in report['dead_features']:
                 lineage = get_lineage(col)
                 report['dead_features_lineage'][col] = lineage
                 logger.warning(f"🧟 DEAD FEATURE: {lineage} {col} (Zero Variance detected)")
                 # Warning only as requested - not invalidating the dataset for this

        # 9. Time Gaps & Island Survivability (Island Split v4.6 Gold)
        # Check index continuity for abandonment report
        diffs = df.index.to_series().diff().dropna()
        max_idx_gap = float(diffs.max().total_seconds() / 60) if not diffs.empty else 0.0
        freq_min = diffs.median().total_seconds() / 60 if not diffs.empty else 1.0
        
        report['max_gap_minutes'] = max_idx_gap
        
        # --- Survival Rule Implementation ---
        target_df = df.copy() # We will prune islands from this copy if needed
        island_col = 'island_id' if 'island_id' in df.columns else None
        
        if island_col:
            islands = df.groupby(island_col)
            report['num_islands_generated'] = int(df[island_col].nunique())
            
            # Rule: Island must be > 120 minutes (Lookback context)
            survival_threshold_bars = 120 / freq_min
            valid_islands = []
            rows_retained = 0
            
            for island_id, group in islands:
                if len(group) >= survival_threshold_bars:
                    valid_islands.append(island_id)
                    rows_retained += len(group)
                else:
                    logger.warning(f"🏖️ [SURVIVAL] Island {island_id} rejected: {len(group)} bars < {survival_threshold_bars:.0f} (120min).")
            
            report['total_rows_retained'] = int(rows_retained)
            
            # Veredito de 'Abandon': Only INVALID if NO islands survive
            if not valid_islands:
                logger.error(f"❌ GAP ABANDONMENT: Zero islands survived the 120min criteria in {name}.")
                report['is_valid'] = False
                report['integrity_comment'] = "No valid data islands (>120min) found."
            else:
                # IMPORTANT: We need a way to pass back the PRUNED dataframe or signal the pipeline
                # For now, we update is_valid. The pipeline is responsible for saving the 'clean' version.
                # Since validate is read-only for the df, we signal the survivors.
                report['valid_island_ids'] = valid_islands
                if not report['is_valid']: # If it was invalid before (NaNs etc), stay invalid
                    pass 
                else:
                    logger.info(f"✅ Protocol Island Split: {len(valid_islands)}/{report['num_islands_generated']} islands survived. Total rows: {rows_retained}")

        # Final Gold Status (Status: VALID / INVALID / FIXED)
        is_healed = report.get('healed', False)
        status = 'VALID' if report['is_valid'] else 'INVALID'
        if is_healed and report['is_valid']:
            status = 'FIXED'
            
        logger.info(f"🏆 Gold Validation complete for {name}. Status: {status}")
        return report
