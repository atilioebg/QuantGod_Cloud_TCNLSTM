import polars as pl
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EventSampler:
    """
    Motor de amostragem avançada para barras baseadas em eventos (Dollar, Tick, Information).
    Implementa as regras de CoT (Anti-Leakage, Tick de Transbordo, Temporal Island Continuity).
    """

    def __init__(self, etl_cfg: dict):
        self._etl_cfg = etl_cfg
        
        self.dollar_threshold = float(etl_cfg.get("dollar_threshold_usd", 100000.0))
        self.tick_threshold = int(etl_cfg.get("tick_threshold", 1000))
        self.info_threshold = float(etl_cfg.get("information_threshold_ofi", 50.0))
        self.island_gap_min = float(etl_cfg.get("island_gap_minutes", 5.0))
        
        legacy_resample_min = int(max(1, etl_cfg.get("resample_min", 1)))
        short_min = int(etl_cfg.get("delta_short_min", 5))
        long_min = int(etl_cfg.get("delta_long_min", 30))
        
        self.ds = max(1, short_min // legacy_resample_min)
        self.dl = max(1, long_min // legacy_resample_min)
        self.ds_lbl = str(short_min)
        self.dl_lbl = str(long_min)
        
        self.vpin_window_min = int(etl_cfg.get("vpin_window_min", 25))
        self.vpin_bars = max(1, self.vpin_window_min // legacy_resample_min)
        self.vpin_lbl = f"vpin_min{self.vpin_window_min}"

        self.spread_zscore_window = max(1, int(etl_cfg.get("spread_zscore_window_min", 60)) // legacy_resample_min)
        self.levels = int(etl_cfg.get("levels", 200))
        
        self.book_asym_depth = int(etl_cfg.get("book_asymmetry_depth", 5))
        self.deep_book_start = int(etl_cfg.get("deep_book_start", 50))
        self.cn = int(etl_cfg.get("convexity_near_end", 10))
        self.cf = int(etl_cfg.get("convexity_far_end", 20))
        
        self.audit_report = {
            "type": "integrated",
            "generated_bars": 0,
            "islands": 0,
            "total_usd_volume": 0.0,
            "total_raw_volume": 0.0
        }

    def compute_event_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Gera as barras agrupando os dados de trade e orderbook com base na regra de limiar.
        Implementa a regra "Tick de Transbordo" e Isolamento de Ilhas.
        Gera uma barra integrada sempre que QUALQUER UM dos limiares (Dollar, Tick, OFI) 
        for rompido.
        """
        logger.info(f"Initiating INTEGRATED sampler (Dollar={self.dollar_threshold}, Tick={self.tick_threshold}, Info={self.info_threshold})")
        df = df.sort("ts")

        # ── 1. Temporal Island Continuty ──────────────────────────────
        gap_ms = int(self.island_gap_min * 60 * 1000)
        df = df.with_columns(
            (pl.col("ts").diff() > gap_ms).fill_null(False).alias("is_new_island")
        )
        df = df.with_columns(pl.col("is_new_island").cum_sum().alias("island_id"))
        
        # Guarda raw volume para integridade
        if "usd_volume" in df.columns:
            self.audit_report["total_raw_volume"] = df["usd_volume"].sum()

        # ── 2. Gatilhos de Amostragem ─────────────────────
        # Tick: 1.0 per execution
        df = df.with_columns(pl.lit(1.0).alias("__tick_unit__"))
        
        # Info (OFI): absolute OFI
        if "ofi" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("ofi")) 
        df = df.with_columns(pl.col("ofi").abs().alias("abs_ofi"))
        
        # ── 3. Indexacao Atômica (Transbordo Triplo CoT v8.0) ──────────────────
        # Computa os IDs parciais de cada tipo de barra
        df = df.with_columns([
            (pl.col("usd_volume").cum_sum().shift(1).fill_null(0.0).over("island_id") // self.dollar_threshold).cast(pl.Int32).alias("dollar_id"),
            (pl.col("__tick_unit__").cum_sum().shift(1).fill_null(0.0).over("island_id") // self.tick_threshold).cast(pl.Int32).alias("tick_id"),
            (pl.col("abs_ofi").cum_sum().shift(1).fill_null(0.0).over("island_id") // self.info_threshold).cast(pl.Int32).alias("info_id")
        ])
        
        # Nova barra é formada se HOUVER QUALQUER rompimento de ID em relacao à row anterior
        # Isso significa que a barra atual agrupa todos os eventos ocorridos ANTES de QUALQUER 
        # um dos triggers exceder o threshold. (Union of events).
        df = df.with_columns([
            (pl.col("dollar_id") != pl.col("dollar_id").shift(1).over("island_id")).alias("_d_brk"),
            (pl.col("tick_id") != pl.col("tick_id").shift(1).over("island_id")).alias("_t_brk"),
            (pl.col("info_id") != pl.col("info_id").shift(1).over("island_id")).alias("_i_brk")
        ])
        
        df = df.with_columns([
            (pl.col("_d_brk") | pl.col("_t_brk") | pl.col("_i_brk")).fill_null(False).alias("is_new_bar")
        ])
        
        # Cumsum para obter um único bar_id unificado, mantendo a integridade multi-disparo temporal
        df = df.with_columns(pl.col("is_new_bar").cum_sum().over("island_id").alias("bar_id"))

        # Cleanup das var auxiliares
        df = df.drop(["__tick_unit__", "abs_ofi", "dollar_id", "tick_id", "info_id", "_d_brk", "_t_brk", "_i_brk", "is_new_bar"])

        # ── 4. Agregacao do Motor de Barras ───────────────────────────
        ob_cols_raw = [c for c in df.columns if ('bid_' in c or 'ask_' in c) and not c.endswith(('_slope', '_rdi'))]
        ob_agg = [pl.col(c).last().alias(c) for c in ob_cols_raw]

        aggs = [
            pl.col("ts").last().alias("ts"),          # Anti-Leakage: ms de fechamento da barra
            pl.col("usd_volume").sum().alias("bar_usd_volume") if "usd_volume" in df.columns else pl.lit(0.0).alias("bar_usd_volume"),
            pl.col("size").sum().alias("bar_btc_volume") if "size" in df.columns else pl.lit(0.0).alias("bar_btc_volume"),
            pl.len().alias("tick_count"),
        ]

        # OHLC e Preco Ponderado (se existirem trades p/ formar vela)
        if "price" in df.columns and "size" in df.columns:
            aggs.extend([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                ((pl.col("price") * pl.col("size")).sum() / (pl.col("size").sum() + 1e-9)).alias("vwap")
            ])
        elif "micro_price" in df.columns:
            # Fallback seguro caso não haja data de trades no formato
            aggs.extend([
                pl.col("micro_price").first().alias("open"),
                pl.col("micro_price").max().alias("high"),
                pl.col("micro_price").min().alias("low"),
                pl.col("micro_price").last().alias("close"),
                pl.col("micro_price").mean().alias("vwap")
            ])

        # L2 Aggregations legadas q precisam transpor pro modelo final
        l2_aggs = []
        if "spread" in df.columns: l2_aggs.extend([pl.col("spread").max().alias("max_spread"), pl.col("spread").mean().alias("mean_spread")])
        if "micro_price" in df.columns: l2_aggs.append(pl.col("micro_price").std().alias("volatility"))
        if "obi_l0" in df.columns: l2_aggs.append(pl.col("obi_l0").mean().alias("mean_obi"))
        if "deep_obi_5" in df.columns: l2_aggs.append(pl.col("deep_obi_5").mean().alias("mean_deep_obi"))
        if "ofi" in df.columns: l2_aggs.append(pl.col("ofi").sum().alias("ofi"))
        if "micro_price_momentum" in df.columns: l2_aggs.append(pl.col("micro_price_momentum").sum().alias("micro_price_momentum"))
        if "bid_slope" in df.columns: l2_aggs.append(pl.col("bid_slope").mean().alias("mean_bid_slope"))
        if "mean_ask_slope" in df.columns: l2_aggs.append(pl.col("ask_slope").mean().alias("mean_ask_slope"))
        if "bid_rdi" in df.columns: l2_aggs.append(pl.col("bid_rdi").mean().alias("bid_rdi"))
        if "ask_rdi" in df.columns: l2_aggs.append(pl.col("ask_rdi").mean().alias("ask_rdi"))
        if "pressure_ratio" in df.columns: l2_aggs.append(pl.col("pressure_ratio").mean().alias("pressure_ratio"))
        if "__stale_l2__" in df.columns: l2_aggs.append(pl.col("__stale_l2__").max().alias("__stale_l2__"))

        aggs.extend(l2_aggs)
        aggs.extend(ob_agg)

        resampled = df.group_by(["island_id", "bar_id"], maintain_order=True).agg(aggs)

        self.audit_report["generated_bars"] = len(resampled)
        self.audit_report["islands"] = resampled["island_id"].n_unique()
        if "bar_usd_volume" in resampled.columns:
            self.audit_report["total_usd_volume"] = resampled["bar_usd_volume"].sum()
            
        logger.info(f"Sampler applied. Bars: {len(resampled)} | Islands: {self.audit_report['islands']}")
        return resampled

    def apply_feature_engineering_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula os deltas, rollings e caracteristicas institucionais a partir
        das barras puras geradas. (Traducao direta da L2Transformer).
        """
        # Limpeza rapida de NaNs criticos iniciais (ex. a primeira linha nao ter ref de diff)
        df = df.with_columns(pl.col("close").fill_nan(0.0).fill_null(0.0))
        
        df = df.with_columns(
            pl.from_epoch("ts", time_unit="ms").alias("datetime")
        ).sort("datetime")
        
        df = df.with_columns(
            np.log1p(pl.col("tick_count").cast(pl.Float64)).alias("log_volume")
        )

        prev_close = pl.col("close").shift(1).over("island_id")
        
        # O preco zero aqui ja e tratado, mas mantemos leniencia matematica
        df = df.with_columns([
            (pl.col("close") / pl.col("open").map_elements(lambda x: x if x != 0 else 1e-9, return_dtype=pl.Float64)).log().alias("body"),
            ((pl.col("high") - pl.max_horizontal("open", "close")) / (prev_close + 1e-9)).alias("upper_wick"),
            ((pl.min_horizontal("open", "close") - pl.col("low")) / (prev_close + 1e-9)).alias("lower_wick"),
            ((pl.col("close") / (prev_close + 1e-9)).log()).alias("log_ret_close"),
        ])

        # Se nao houver as variaveis originarias L2 por causa de CSV incompleto de teste,
        # criamos zeros pra n quebrar as math ops subsequentes (OFI, RDI, etc)
        missing = [c for c in ["ofi", "bid_rdi", "ask_rdi"] if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(0.0).alias(c) for c in missing])

        ds = self.ds
        dl = self.dl
        ds_l = self.ds_lbl
        dl_l = self.dl_lbl

        df = df.with_columns([
            pl.col("ofi").diff(ds).over("island_id").alias(f"ofi_delta_{ds_l}"),
            pl.col("bid_rdi").diff(ds).over("island_id").alias(f"bid_rdi_delta_{ds_l}"),
            pl.col("ask_rdi").diff(ds).over("island_id").alias(f"ask_rdi_delta_{ds_l}"),
            pl.col("close").pct_change(ds).over("island_id").alias(f"micro_price_delta_{ds_l}"),
            pl.col("ofi").diff(dl).over("island_id").alias(f"ofi_delta_{dl_l}"),
            pl.col("bid_rdi").diff(dl).over("island_id").alias(f"bid_rdi_delta_{dl_l}"),
            pl.col("ask_rdi").diff(dl).over("island_id").alias(f"ask_rdi_delta_{dl_l}"),
            pl.col("close").pct_change(dl).over("island_id").alias(f"micro_price_delta_{dl_l}"),
        ])

        n_asym = self.book_asym_depth
        dbs = self.deep_book_start
        cn = self.cn
        cf = self.cf

        sb_n = sum(pl.col(f"bid_{i}_s") for i in range(n_asym) if f"bid_{i}_s" in df.columns)
        sa_n = sum(pl.col(f"ask_{i}_s") for i in range(n_asym) if f"ask_{i}_s" in df.columns)
        sb_deep = sum(pl.col(f"bid_{i}_s") for i in range(dbs, self.levels) if f"bid_{i}_s" in df.columns)
        sa_deep = sum(pl.col(f"ask_{i}_s") for i in range(dbs, self.levels) if f"ask_{i}_s" in df.columns)
        sb0 = sum(pl.col(f"bid_{i}_s") for i in range(1, cn + 1) if f"bid_{i}_s" in df.columns)
        sb1 = sum(pl.col(f"bid_{i}_s") for i in range(cn + 1, cf + 1) if f"bid_{i}_s" in df.columns)
        sa0 = sum(pl.col(f"ask_{i}_s") for i in range(1, cn + 1) if f"ask_{i}_s" in df.columns)
        sa1 = sum(pl.col(f"ask_{i}_s") for i in range(cn + 1, cf + 1) if f"ask_{i}_s" in df.columns)

        if not isinstance(sb_n, pl.Expr): sb_n, sa_n, sb_deep, sa_deep, sb0, sb1, sa0, sa1 = (pl.lit(0.0) for _ in range(8))

        if "max_spread" not in df.columns:
            df = df.with_columns([(pl.lit(0.0)).alias("max_spread")])

        # Rolling math
        df = df.with_columns([
            ((sb_n + 1e-9) / (sa_n + 1e-9)).log().alias("book_asymmetry_v5"),
            (pl.col("max_spread").rolling_mean(self.spread_zscore_window, min_periods=1).over("island_id")).alias("_rm_s"),
            (pl.col("max_spread").rolling_std(self.spread_zscore_window, min_periods=1).over("island_id")).alias("_rs_s"),
            (pl.col("ofi").abs().rolling_sum(self.vpin_bars, min_periods=1).over("island_id")).alias("_ofi_roll"),
            (sb_n + sa_n).alias("_stot"),
            (sb_deep).alias("_sbd"), (sa_deep).alias("_sad"),
            (sb_n).alias("_sb_n"), (sa_n).alias("_sa_n"),
            (sb0 + 1e-9).alias("_sb0"), (sb1 + 1e-9).alias("_sb1"),
            (sa0 + 1e-9).alias("_sa0"), (sa1 + 1e-9).alias("_sa1")
        ])

        vpin_col = self.vpin_lbl
        df = df.with_columns([
            ((pl.col("max_spread") - pl.col("_rm_s")) / (pl.col("_rs_s") + 1e-9)).alias("spread_zscore_60"),
            (pl.col("_ofi_roll") / (pl.col("_stot") + 1e-9)).alias(vpin_col),
            (pl.col(f"micro_price_delta_{ds_l}") / (pl.col(f"ofi_delta_{ds_l}").abs() + 1e-9)).alias("kyle_lambda"),
            (pl.col("_sbd") / (pl.col("_sb_n") + 1e-9)).alias("bid_deep_ratio"),
            (pl.col("_sad") / (pl.col("_sa_n") + 1e-9)).alias("ask_deep_ratio"),
            (pl.col("_sb0") / pl.col("_sb1")).alias("bid_convexity"),
            (pl.col("_sa0") / pl.col("_sa1")).alias("ask_convexity"),
        ])

        helper = ["_rm_s", "_rs_s", "_ofi_roll", "_stot", "_sbd", "_sad", "_sb_n", "_sa_n", "_sb0", "_sb1", "_sa0", "_sa1"]
        df = df.drop([c for c in helper if c in df.columns])

        # Cleanup and finalize
        sniper_cols = [
            f"ofi_delta_{ds_l}", f"ofi_delta_{dl_l}", f"bid_rdi_delta_{ds_l}", f"bid_rdi_delta_{dl_l}",
            f"ask_rdi_delta_{ds_l}", f"ask_rdi_delta_{dl_l}", f"micro_price_delta_{ds_l}", f"micro_price_delta_{dl_l}",
            "book_asymmetry_v5", "spread_zscore_60", vpin_col, "kyle_lambda", "bid_deep_ratio", "ask_deep_ratio",
            "bid_convexity", "ask_convexity", "body", "upper_wick", "lower_wick", "log_ret_close", "vwap"
        ]
        
        for c in sniper_cols:
            if c in df.columns:
                df = df.with_columns(
                    pl.col(c).fill_nan(0.0).fill_null(0.0)
                    .map_elements(lambda x: 0.0 if (x == float('inf') or x == float('-inf')) else x, return_dtype=pl.Float64)
                    .alias(c)
                )
                
        # Fill mean features
        means = ["mean_spread", "mean_obi", "mean_deep_obi", "mean_bid_slope", "mean_ask_slope", "pressure_ratio"]
        for c in means:
            if c in df.columns:
                df = df.with_columns(pl.col(c).fill_nan(0.0).fill_null(0.0).alias(c))

        return df


