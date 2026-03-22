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
    SOTA Big Data Streaming Loader (v10.5).
    Mapeia todos os Parquets (1,151 arquivos) no disco.
    Permite: Subsampling de Classes, Epoch-Chunking (Fatia Representativa) e Island Protection.
    """
    def __init__(self, parquet_files: list, feature_cols: list, seq_len: int, config: dict, is_train: bool = True):
        self.parquet_files = sorted(parquet_files)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.config = config
        self.is_train = is_train

        opt_cfg = config.get('training_optimization', {})
        if not opt_cfg:
            # Fallback if config structure is different
            opt_cfg = config.get('training', {}).get('training_optimization', {})

        self.global_indices = []
        self.total_raw_rows = 0
        
        logger.info(f"🔍 Streaming Scan do Dataset ({'Treino' if is_train else 'Val'}): {len(parquet_files)} arquivos...")
        
        for f_idx, pf in enumerate(self.parquet_files):
            # Scan rápido apenas dos comprimentos (metadata)
            meta = pl.scan_parquet(pf).select(['target', 'island_id']).collect()
            count = len(meta)
            
            targets = meta['target'].to_numpy()
            islands = meta['island_id'].to_numpy()
            
            # Filtro de Island Protection (Lookback não cruza GAP)
            valid_mask = (islands[:count - seq_len + 1] == islands[seq_len - 1:])
            valid_local_indices = np.where(valid_mask)[0]
            
            # ── [v10.6] CLASS SUBSAMPLING (Memory Optimization) ──────────
            if is_train and opt_cfg.get('use_class_subsampling', False):
                target_cls = opt_cfg.get('subsample_class_target', 1)
                keep_ratio = opt_cfg.get('subsample_keep_ratio', 0.2)
                
                # Pegamos o target na posição final da sequência
                seq_targets = targets[valid_local_indices + seq_len - 1]
                
                # Sorteio aleatório para a classe alvo
                random_vals = np.random.rand(len(valid_local_indices))
                drop_mask = (seq_targets == target_cls) & (random_vals > keep_ratio)
                valid_local_indices = valid_local_indices[~drop_mask]

            # Registrar mapeamento: global_idx -> (file_idx, local_idx)
            for l_idx in valid_local_indices:
                self.global_indices.append((f_idx, l_idx))
            
            self.total_raw_rows += count

        # ── [v10.7] EPOCH-CHUNKING (Global Sample Sweep) ──────────
        if is_train and opt_cfg.get('use_epoch_chunking', False):
            n_samples = opt_cfg.get('samples_per_epoch', 5000000)
            if n_samples < len(self.global_indices):
                selected_indices = np.random.choice(len(self.global_indices), n_samples, replace=False)
                self.global_indices = [self.global_indices[i] for i in selected_indices]
                logger.info(f"✂️ Epoch-Chunking Ativo: Reduzindo {len(self.global_indices)} -> {n_samples} amostras/época.")

        logger.info(f"✅ Streaming Dataset carregado: {len(self.global_indices):,} sequências válidas.")

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        f_idx, l_idx = self.global_indices[idx]
        pf = self.parquet_files[f_idx]
        
        # Leitura sob demanda (Lazy) usando slice do Polars (mmap-fast)
        df_slice = pl.read_parquet(
            pf, 
            columns=self.feature_cols + ['target'],
            n_rows=self.seq_len,
            row_index_offset=l_idx
        )
        
        X = df_slice.select(self.feature_cols).to_numpy().astype(np.float32)
        y = df_slice.select('target')[self.seq_len - 1, 0] # Label na última barra do frame
        
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)
