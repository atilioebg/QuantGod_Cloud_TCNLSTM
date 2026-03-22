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
    SOTA Big Data Streaming Loader (v10.8 Sniper-RAM Edition).
    Mapeia todos os Parquets (1,151 arquivos) no disco usando NumPy Structured Arrays.
    Economia: Reduz o consumo do índice de 22GB para ~2.6GB para 447M de linhas.
    """
    def __init__(self, parquet_files: list, feature_cols: list, seq_len: int, config: dict, is_train: bool = True):
        self.parquet_files = sorted(parquet_files)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.config = config
        self.is_train = is_train

        opt_cfg = config.get('training_optimization', {})
        if not opt_cfg:
            opt_cfg = config.get('training', {}).get('training_optimization', {})

        # ── [v10.9] Memory-Safe Phase 1: Scan & Collect Indices in Chunks ──────────
        all_file_indices = []
        all_local_indices = []
        
        logger.info(f"🔍 RAM-Safe Streaming Scan ({'Treino' if is_train else 'Val'}): {len(parquet_files)} arquivos...")
        
        for f_idx, pf in enumerate(self.parquet_files):
            # Scan rápido apenas dos comprimentos (metadata)
            meta = pl.scan_parquet(pf).select(['target', 'island_id']).collect()
            count = len(meta)
            
            targets = meta['target'].to_numpy()
            islands = meta['island_id'].to_numpy()
            
            # Filtro de Island Protection (Lookback não cruza GAP)
            valid_mask = (islands[:count - seq_len + 1] == islands[seq_len - 1:])
            valid_local_indices = np.where(valid_mask)[0]
            
            # ── [v10.10] CLASS SUBSAMPLING (Memory Optimization) ──────────
            if is_train and opt_cfg.get('use_class_subsampling', False):
                target_cls = opt_cfg.get('subsample_class_target', 1)
                keep_ratio = opt_cfg.get('subsample_keep_ratio', 0.2)
                
                # Pegamos o target na posição final da sequência
                seq_targets = targets[valid_local_indices + seq_len - 1]
                
                # Sorteio aleatório para a classe alvo
                random_vals = np.random.rand(len(valid_local_indices))
                drop_mask = (seq_targets == target_cls) & (random_vals > keep_ratio)
                valid_local_indices = valid_local_indices[~drop_mask]

            # Registrar em pedaços NumPy para evitar overhead de listas Python
            if len(valid_local_indices) > 0:
                all_file_indices.append(np.full(len(valid_local_indices), f_idx, dtype=np.uint16))
                all_local_indices.append(valid_local_indices.astype(np.uint32))
            
            if f_idx % 200 == 0:
                gc.collect()

        # ── [v10.11] Phase 2: NumPy Concatenation (Shared Memory Friendly) ──────────
        if not all_file_indices:
             self.file_idx_map = np.array([], dtype=np.uint16)
             self.local_idx_map = np.array([], dtype=np.uint32)
        else:
             self.file_idx_map = np.concatenate(all_file_indices)
             self.local_idx_map = np.concatenate(all_local_indices)

        # Limpeza agressiva de RAM intermediária
        del all_file_indices
        del all_local_indices
        gc.collect()

        # ── [v10.12] EPOCH-CHUNKING (Global Sample Sweep) ──────────
        if is_train and opt_cfg.get('use_epoch_chunking', False):
            n_samples = opt_cfg.get('samples_per_epoch', 5000000)
            n_current = len(self.file_idx_map)
            if n_samples < n_current:
                # Sorteia os índices ordinais
                sample_indices = np.random.choice(n_current, n_samples, replace=False)
                self.file_idx_map = self.file_idx_map[sample_indices]
                self.local_idx_map = self.local_idx_map[sample_indices]
                logger.info(f"✂️ Epoch-Chunking Ativo: Expondo {n_samples:,} amostras aleatórias/época.")

        logger.info(f"✅ RAM-Safe Streaming Dataset: {len(self.file_idx_map):,} sequências indexadas.")

    def __len__(self):
        return len(self.file_idx_map)

    def __getitem__(self, idx):
        f_idx = self.file_idx_map[idx]
        l_idx = self.local_idx_map[idx]
        pf = self.parquet_files[f_idx]
        
        # Leitura sob demanda (Lazy) usando mmap-fast slice
        df_slice = pl.read_parquet(
            pf, 
            columns=self.feature_cols + ['target'],
            n_rows=self.seq_len,
            row_index_offset=int(l_idx)
        )
        
        X = df_slice.select(self.feature_cols).to_numpy().astype(np.float32)
        y = df_slice.select('target')[self.seq_len - 1, 0] # Label na última barra do frame
        
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)
