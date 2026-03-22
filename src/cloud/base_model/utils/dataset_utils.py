import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
import logging
from pathlib import Path
import gc
import os

logger = logging.getLogger(__name__)

class QuantGodLazyDataset(Dataset):
    """
    SOTA Big Data Streaming Loader (v10.25 Double-Chunking Edition).
    Suporta fatiamento aleatório tanto para Treino (samples_per_epoch) 
    quanto para Validação (samples_per_val_epoch).
    """
    def __init__(self, parquet_files: list, feature_cols: list, seq_len: int, config: dict, is_train: bool = True, scaler = None):
        self.parquet_files = sorted(parquet_files)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.config = config
        self.is_train = is_train
        self.scaler = scaler

        # ── SMART-SNIFF: Procura a config no labirinto do YAML ──────────
        opt_cfg = config.get('training_optimization', {})
        if not opt_cfg:
            opt_cfg = config.get('pre_processing', {}).get('training_optimization', {})
        if not opt_cfg:
            opt_cfg = config.get('training', {}).get('training_optimization', {})
        
        # ── Phase 1: Scan & Collect Indices ──────────
        all_file_indices = []
        all_local_indices = []
        
        label_for = 'Treino' if is_train else 'Val'
        logger.info(f"🔍 RAM-Safe Streaming Scan ({label_for}): {len(parquet_files)} arquivos...")
        
        for f_idx, pf in enumerate(self.parquet_files):
            meta = pl.scan_parquet(pf).select(['target', 'island_id']).collect()
            count = len(meta)
            targets = meta['target'].to_numpy()
            islands = meta['island_id'].to_numpy()
            
            valid_mask = (islands[:count - seq_len + 1] == islands[seq_len - 1:])
            valid_local_indices = np.where(valid_mask)[0]
            
            # ── [v10.26] CLASS SUBSAMPLING ──────────────────────────────
            if is_train and opt_cfg.get('use_class_subsampling', False):
                target_cls = int(opt_cfg.get('subsample_class_target', 1))
                keep_ratio = float(opt_cfg.get('subsample_keep_ratio', 0.2))
                seq_targets = targets[valid_local_indices + seq_len - 1]
                random_vals = np.random.rand(len(valid_local_indices))
                drop_mask = (seq_targets == target_cls) & (random_vals > keep_ratio)
                valid_local_indices = valid_local_indices[~drop_mask]

            if len(valid_local_indices) > 0:
                all_file_indices.append(np.full(len(valid_local_indices), f_idx, dtype=np.uint16))
                all_local_indices.append(valid_local_indices.astype(np.uint32))
            
            if f_idx % 300 == 0: gc.collect()

        # ── Phase 2: NumPy Concatenation ──────────
        if not all_file_indices:
             self.file_idx_map = np.array([], dtype=np.uint16)
             self.local_idx_map = np.array([], dtype=np.uint32)
        else:
             self.file_idx_map = np.concatenate(all_file_indices)
             self.local_idx_map = np.concatenate(all_local_indices)

        del all_file_indices; del all_local_indices; gc.collect()

        # ── [v10.27] DOUBLE-CHUNKING (The HPO Speed Hack) ──────────
        # Aplicamos chunking no Treino E na Validação se configurado.
        chunk_key = 'samples_per_epoch' if is_train else 'samples_per_val_epoch'
        
        # Se n_samples_val for null no config, tentamos um default de segurança para Optuna
        n_samples = opt_cfg.get(chunk_key)
        if not is_train and n_samples is None:
             # Se for validação e estamos no modo epoch_chunking, limitamos a 50k por segurança no teste
             if opt_cfg.get('use_epoch_chunking', False):
                  n_samples = 50000 

        if n_samples is not None:
            n_samples = int(n_samples)
            n_current = len(self.file_idx_map)
            if n_samples < n_current:
                sample_indices = np.random.choice(n_current, n_samples, replace=False)
                self.file_idx_map = self.file_idx_map[sample_indices]
                self.local_idx_map = self.local_idx_map[sample_indices]
                logger.info(f"✂️ Double-Chunking ({label_for}): Expondo fatias de {n_samples:,} amostras aleatórias.")

        # ── [v10.30] PRE-LOADING (The RAM Accelerator) ─────────────────────
        # Se o dataset for pequeno o suficiente, carregamos tudo no ram uma única vez.
        self.pre_loaded_X = None
        self.pre_loaded_y = None
        
        # Threshold de segurança: 100k samples (~600MB RAM)
        n_final = len(self.file_idx_map)
        if n_final > 0 and n_final <= 100000:
            logger.info(f"🚀 Pre-loading {n_final:,} samples into RAM (Lightning Mode)...")
            X_data = np.zeros((n_final, self.seq_len, len(self.feature_cols)), dtype=np.float32)
            y_data = np.zeros(n_final, dtype=np.int64)
            
            # Agrupamos por arquivo para otimizar leitura do disco
            from collections import defaultdict
            file_to_indices = defaultdict(list)
            for i in range(n_final):
                file_to_indices[self.file_idx_map[i]].append(i)
            
            for f_idx, s_indices in file_to_indices.items():
                pf = self.parquet_files[f_idx]
                df_all = pl.read_parquet(pf, columns=self.feature_cols + ['target']).fill_nan(0).fill_null(0)
                # v10.32: Multi-row optimization for pre-loading directly to NumPy
                raw_X_np = df_all.select(self.feature_cols).to_numpy()
                if self.scaler:
                    raw_X_np = self.scaler.transform(raw_X_np).astype(np.float32)
                
                for i in s_indices:
                    l_idx = int(self.local_idx_map[i])
                    X_data[i] = raw_X_np[l_idx : l_idx + self.seq_len]
                    y_data[i] = df_all['target'][l_idx + self.seq_len - 1]
            
            self.pre_loaded_X = torch.from_numpy(X_data)
            self.pre_loaded_y = torch.from_numpy(y_data)
            logger.info(f"✅ RAM-Safe Streaming Dataset: {n_final:,} sequences indexed and PRE-LOADED.")
        # ── [v10.35] K-FOLD COMPATIBILITY ──────────────────────────────────
        # Adicionamos global_indices para permitir que scripts de K-Fold (Phase 2) 
        # façam o subset do dataset sem precisar de re-escaneamento.
        self.global_indices = np.arange(len(self.file_idx_map))

        logger.info(f"✅ RAM-Safe Streaming Dataset: {n_final:,} sequences indexed (Mode: {'Lightning' if self.pre_loaded_X is not None else 'Lazy'}).")

    def __len__(self): return len(self.global_indices)
    def __getitem__(self, idx):
        # Route through global_indices for K-Fold subsetting
        g_idx = self.global_indices[idx]
        
        if self.pre_loaded_X is not None:
            return self.pre_loaded_X[g_idx], self.pre_loaded_y[g_idx]
            
        f_idx = self.file_idx_map[g_idx]; l_idx = int(self.local_idx_map[g_idx])
        df_slice = pl.read_parquet(self.parquet_files[f_idx], columns=self.feature_cols + ['target'], n_rows=self.seq_len, row_index_offset=l_idx).fill_nan(0).fill_null(0)
        X = df_slice.select(self.feature_cols).to_numpy()
        if self.scaler:
            X = self.scaler.transform(X).astype(np.float32)
        y = df_slice.select('target')[self.seq_len - 1, 0]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)
