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
    def __init__(self, parquet_files: list, feature_cols: list, seq_len: int, config: dict, is_train: bool = True):
        self.parquet_files = sorted(parquet_files)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.config = config
        self.is_train = is_train

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

        logger.info(f"✅ RAM-Safe Streaming Dataset: {len(self.file_idx_map):,} sequências indexadas.")

    def __len__(self): return len(self.file_idx_map)
    def __getitem__(self, idx):
        f_idx = self.file_idx_map[idx]; l_idx = int(self.local_idx_map[idx])
        df_slice = pl.read_parquet(self.parquet_files[f_idx], columns=self.feature_cols + ['target'], n_rows=self.seq_len, row_index_offset=l_idx)
        X = df_slice.select(self.feature_cols).to_numpy().astype(np.float32)
        y = df_slice.select('target')[self.seq_len - 1, 0]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)
