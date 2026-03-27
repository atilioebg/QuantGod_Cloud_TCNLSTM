import polars as pl
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM Symmetric Filter — De Prado (Advances in Financial Machine Learning)
# ─────────────────────────────────────────────────────────────────────────────
# Desenhado para ser Numba-ready (@njit compatible):
#   - Opera exclusivamente sobre np.ndarray (sem listas, dicts ou objetos Python)
#   - Loop sequencial simples (path-dependent por natureza, não paralelizável)
#   - Para ativar Numba no futuro: adicionar '@numba.njit' na linha anterior à def
#
# Upgrade para Numba (uma linha de mudança):
#   1. pip install numba
#   2. from numba import njit
#   3. @njit  ←── adicionar aqui
#      def _cusum_loop(...):
# ─────────────────────────────────────────────────────────────────────────────
def _cusum_loop(prices: np.ndarray, h_arr: np.ndarray) -> np.ndarray:
    """
    Filtro CUSUM Simétrico Adaptativo — De Prado (AFML, Cap. 17 + Cap. 3).

    [v9.0 AFML-Aligned] h_arr é um array de limiares adaptativos por tick,
    calculado via EWMA rolante da volatilidade (não mais estático).
    Isso garante que o filtro se adapta a mudanças de regime (alta/baixa vol).

    Upgrade para Numba (uma linha de mudança):
      1. pip install numba
      2. from numba import njit
      3. @njit  ←── adicionar aqui
         def _cusum_loop(...):

    Args:
        prices: Array de log-retornos (np.float64). Shape: (N,)
        h_arr:  Array de limiares adaptativos por tick. Shape: (N,)

    Returns:
        event_flags: Array booleano Shape (N,). True nos índices de disparo.
    """
    n = len(prices)
    event_flags = np.zeros(n, dtype=np.bool_)
    s_pos = 0.0
    s_neg = 0.0

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        h_i = h_arr[i]

        if s_pos >= h_i:
            s_pos = 0.0
            event_flags[i] = True
        elif s_neg <= -h_i:
            s_neg = 0.0
            event_flags[i] = True

    return event_flags


class EventSampler:
    """
    Motor de amostragem avançada para barras baseadas em eventos.
    Suporta três modos configuráveis via master_config.yaml (sampling_mode):
      'trigger' — Trigger Bars (Dollar/Tick/Information/Time). Padrão IID.
      'cusum'   — Somente Filtro CUSUM Simétrico de De Prado.
      'both'    — CUSUM define janelas de interesse; Trigger constrói barras IID dentro delas.
    """

    def __init__(self, etl_cfg: dict):
        self._etl_cfg = etl_cfg

        # ── [AMOSTRAGEM] Modo e Limiares ──
        self.sampling_mode        = str(etl_cfg.get("sampling_mode", "trigger"))
        self.dollar_threshold     = float(etl_cfg.get("dollar_threshold_usd", 100000.0))
        self.tick_threshold       = int(etl_cfg.get("tick_threshold", 1000))
        self.info_threshold       = float(etl_cfg.get("information_threshold_ofi", 50.0))
        self.island_gap_min       = float(etl_cfg.get("island_gap_minutes", 5.0))

        # ── [CUSUM] Parâmetros ──
        self.adaptive_h           = bool(etl_cfg.get("adaptive_h", True))
        self.cusum_h_factor       = float(etl_cfg.get("cusum_h_factor", 1.5))
        self.cusum_vol_span       = int(etl_cfg.get("cusum_vol_span", 100))

        # ── [RESAMPLE] Frequência e Features ──
        resample_freq = str(etl_cfg.get("resample_freq", "5min"))
        freq_min = int(resample_freq.replace('min', '').replace('m', '').replace('T', ''))
        self.time_threshold_ms = freq_min * 60 * 1000

        legacy_resample_min = int(max(1, etl_cfg.get("resample_min", 1)))
        short_min = int(etl_cfg.get("delta_short_min", 5))
        long_min  = int(etl_cfg.get("delta_long_min", 30))

        self.ds     = max(1, short_min // legacy_resample_min)
        self.dl     = max(1, long_min  // legacy_resample_min)
        self.ds_lbl = str(short_min)
        self.dl_lbl = str(long_min)

        self.vpin_window_min      = int(etl_cfg.get("vpin_window_min", 25))
        self.vpin_bars            = max(1, self.vpin_window_min // legacy_resample_min)
        self.vpin_lbl             = f"vpin_min{self.vpin_window_min}"
        self.spread_zscore_window = max(1, int(etl_cfg.get("spread_zscore_window_min", 60)) // legacy_resample_min)
        self.levels               = int(etl_cfg.get("levels", 200))
        self.book_asym_depth      = int(etl_cfg.get("book_asymmetry_depth", 5))
        self.deep_book_start      = int(etl_cfg.get("deep_book_start", 50))
        self.cn                   = int(etl_cfg.get("convexity_near_end", 10))
        self.cf                   = int(etl_cfg.get("convexity_far_end", 20))
        self.clipping_cfg         = etl_cfg.get("clipping", {"enabled": False})

        # ── [AUDITORIA] Relatório de Execução ──
        self.audit_report = {
            "type": "integrated",
            "sampling_mode": self.sampling_mode,
            "generated_bars": 0,
            "islands": 0,
            "total_usd_volume": 0.0,
            "total_raw_volume": 0.0,
            "cusum_events_fired": 0,
            "cusum_rejection_rate": 0.0,
            "max_gap_before": 0.0,
            "max_gap_after": 0.0,
            "healed": False
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Método Público Principal
    # ─────────────────────────────────────────────────────────────────────────

    def compute_event_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Gera barras IID a partir do DataFrame de ticks (L2 + Trades merged).
        Bifurca o processamento baseado em self.sampling_mode.
        """
        logger.info(f"EventSampler iniciado. Modo: '{self.sampling_mode}' | "
                    f"Dollar={self.dollar_threshold} | Tick={self.tick_threshold} | "
                    f"Info={self.info_threshold} | Time={self.time_threshold_ms}ms")

        df = df.sort("ts")
        df = self._apply_island_split(df)

        if "usd_volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("usd_volume"))

        self.audit_report["total_raw_volume"] = float(df["usd_volume"].sum())

        if self.sampling_mode == "trigger":
            return self._run_trigger_bars(df)

        elif self.sampling_mode == "cusum":
            cusum_mask = self._compute_cusum_mask(df)
            self.audit_report["cusum_events_fired"] = int(cusum_mask.sum())
            total = len(cusum_mask)
            self.audit_report["cusum_rejection_rate"] = round(
                1.0 - (self.audit_report["cusum_events_fired"] / total) if total > 0 else 0.0, 4)
            logger.info(f"CUSUM: {self.audit_report['cusum_events_fired']} eventos disparados "
                        f"({self.audit_report['cusum_rejection_rate']*100:.1f}% de ticks filtrados)")
            # Em modo cusum puro, cada evento CUSUM define uma barra unitária
            df = df.with_columns(cusum_mask.alias("__cusum_event__"))
            df = df.with_columns(pl.col("__cusum_event__").cum_sum().over("island_id").alias("bar_id"))
            return self._aggregate_bars(df)

        elif self.sampling_mode == "both":
            cusum_mask = self._compute_cusum_mask(df)
            self.audit_report["cusum_events_fired"] = int(cusum_mask.sum())
            total = len(cusum_mask)
            self.audit_report["cusum_rejection_rate"] = round(
                1.0 - (self.audit_report["cusum_events_fired"] / total) if total > 0 else 0.0, 4)
            logger.info(f"CUSUM (both): {self.audit_report['cusum_events_fired']} eventos. "
                        f"Trigger IID será aplicado dentro de cada janela CUSUM.")
            # CUSUM cria um regime_id que isola blocos de atividade direcional
            df = df.with_columns(cusum_mask.alias("__cusum_event__"))
            df = df.with_columns(
                pl.col("__cusum_event__").cum_sum().over("island_id").alias("cusum_regime_id")
            )
            return self._run_trigger_bars(df, extra_group_key="cusum_regime_id")

        else:
            logger.warning(f"sampling_mode '{self.sampling_mode}' desconhecido. Usando 'trigger'.")
            return self._run_trigger_bars(df)

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos Privados
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_island_split(self, df: pl.DataFrame) -> pl.DataFrame:
        """Quebra séries temporais em ilhas quando gaps excedem island_gap_minutes."""
        if df.is_empty(): return df

        # Calcular gap máximo antes do processamento (para auditoria)
        diffs_ms = df["ts"].diff().fill_null(0)
        self.audit_report["max_gap_before"] = round(float(diffs_ms.max() / 1000 / 60), 2)

        gap_ms = int(self.island_gap_min * 60 * 1000)
        df = df.with_columns(
            (pl.col("ts").diff() > gap_ms).fill_null(False).alias("is_new_island")
        )
        # Em barras event-driven, não "curamos" o tempo, apenas isolamos regimes.
        # Portanto max_gap_after será igual a max_gap_before na série bruta, 
        # mas as Ilhas garantem que os indicadores (EWMA, OFI) não "vazem" entre os gaps.
        self.audit_report["max_gap_after"] = self.audit_report["max_gap_before"]
        
        return df.with_columns(pl.col("is_new_island").cum_sum().alias("island_id"))

    def _compute_cusum_mask(self, df: pl.DataFrame) -> pl.Series:
        """
        Aplica _cusum_loop sobre os retornos logarítmicos do DataFrame.
        Retorna uma pl.Series booleana com True nos ticks de eventos CUSUM.

        [v9.0 AFML-Aligned] O limiar h é agora ADAPTATIVO:
          h[i] = cusum_h_factor * ewma_std_rolling(log_ret)[i]
        onde a EWMA é calculada tick-a-tick ao longo de TODA a série,
        não apenas no warmup inicial. Isso permite que o filtro se adapte
        a mudanças de regime de volatilidade (De Prado, AFML Cap. 3).
        """
        # Obter log-retornos calculados estritamente DENTRO de cada ilha
        # Isso evita que um pulo de preço durante um gap gere um evento CUSUM falso.
        price_col = "close" if "close" in df.columns else ("micro_price" if "micro_price" in df.columns else "price")

        if price_col in df.columns:
            log_ret = (
                df.select(
                    pl.col(price_col).log().diff().over("island_id").fill_null(0.0)
                ).to_series().to_numpy()
            )
        else:
            logger.warning("CUSUM: nenhuma coluna de preço encontrada. Retornando máscara vazia.")
            return pl.Series([False] * len(df))

        # Substituir NaN por 0 (ticks sem retorno válido)
        log_ret = np.nan_to_num(log_ret, nan=0.0)
        n = len(log_ret)

        # ── [AFML v9.0] Cálculo de h[i] (Adaptativo ou Estático) ──────────────
        alpha = 2.0 / (self.cusum_vol_span + 1.0)
        ewma_mean = 0.0
        running_var = 0.0
        h_arr = np.zeros(n, dtype=np.float64)

        if self.adaptive_h:
            # h[i] evolui com a volatilidade local tick-a-tick
            for i in range(n):
                val = log_ret[i]
                ewma_mean  = alpha * val + (1.0 - alpha) * ewma_mean
                running_var = alpha * (val - ewma_mean) ** 2 + (1.0 - alpha) * running_var
                h_arr[i]   = self.cusum_h_factor * np.sqrt(max(running_var, 1e-16))
        else:
            # h é estático: calculamos a volatilidade inicial (warmup) e fixamos
            warmup = min(n, self.cusum_vol_span)
            initial_vol = np.std(log_ret[:warmup]) if warmup > 1 else 0.0001
            h_fixed = self.cusum_h_factor * initial_vol
            h_arr[:] = h_fixed
            logger.info(f"CUSUM: Modo Estático ativado. h fixado em {h_fixed:.6f}")

        logger.debug(
            f"CUSUM Adaptativo: h_min={h_arr.min():.6f} | h_max={h_arr.max():.6f} | "
            f"h_mean={h_arr.mean():.6f} (factor={self.cusum_h_factor})"
        )

        # Chamar o loop puro NumPy com array de limiares adaptativos (Numba-ready)
        flags = _cusum_loop(log_ret, h_arr)
        return pl.Series(flags)

    def _run_trigger_bars(self, df: pl.DataFrame, extra_group_key: str = None) -> pl.DataFrame:
        """
        Gera barras por acúmulo de limiares (Dollar/Tick/Information/Time).
        Suporta uma chave de grupo extra (cusum_regime_id no modo 'both').
        """
        # ── Gatilho: Tick unit ──
        df = df.with_columns(pl.lit(1.0).alias("__tick_unit__"))

        # ── Gatilho: OFI ──
        if "ofi" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("ofi"))
        df = df.with_columns(pl.col("ofi").abs().alias("abs_ofi"))

        # ── IDs parciais por tipo de bar (via integer division do acúmulo) ──
        group_key = "island_id"
        df = df.with_columns([
            (pl.col("usd_volume").cum_sum().shift(1).fill_null(0.0).over(group_key) // self.dollar_threshold).cast(pl.Int32).alias("dollar_id"),
            (pl.col("__tick_unit__").cum_sum().shift(1).fill_null(0.0).over(group_key) // self.tick_threshold).cast(pl.Int32).alias("tick_id"),
            (pl.col("abs_ofi").cum_sum().shift(1).fill_null(0.0).over(group_key) // self.info_threshold).cast(pl.Int32).alias("info_id"),
            (pl.col("ts") // self.time_threshold_ms).cast(pl.Int64).alias("time_id"),
        ])

        df = df.with_columns([
            (pl.col("dollar_id") != pl.col("dollar_id").shift(1).over(group_key)).alias("_d_brk"),
            (pl.col("tick_id")   != pl.col("tick_id").shift(1).over(group_key)).alias("_t_brk"),
            (pl.col("info_id")   != pl.col("info_id").shift(1).over(group_key)).alias("_i_brk"),
            (pl.col("time_id")   != pl.col("time_id").shift(1).over(group_key)).alias("_time_brk"),
        ])

        df = df.with_columns(
            (pl.col("_d_brk") | pl.col("_t_brk") | pl.col("_i_brk") | pl.col("_time_brk"))
            .fill_null(False).alias("is_new_bar")
        )
        df = df.with_columns(pl.col("is_new_bar").cum_sum().over(group_key).alias("bar_id"))

        df = df.drop([c for c in ["__tick_unit__", "abs_ofi", "dollar_id", "tick_id",
                                   "info_id", "time_id", "_d_brk", "_t_brk", "_i_brk",
                                   "_time_brk", "is_new_bar"] if c in df.columns])

        return self._aggregate_bars(df, extra_group_key=extra_group_key)

    def _aggregate_bars(self, df: pl.DataFrame, extra_group_key: str = None) -> pl.DataFrame:
        """
        Agrega ticks em barras OHLC + features L2. Usa [island_id, bar_id] como chave.
        Se extra_group_key fornecido (ex: cusum_regime_id), inclui na chave de grupo.
        """
        ob_cols_raw = [c for c in df.columns if ('bid_' in c or 'ask_' in c) and not c.endswith(('_slope', '_rdi'))]
        ob_agg = [pl.col(c).last().alias(c) for c in ob_cols_raw]

        aggs = [
            pl.col("ts").last().alias("ts"),
            pl.col("usd_volume").sum().alias("bar_usd_volume") if "usd_volume" in df.columns else pl.lit(0.0).alias("bar_usd_volume"),
            pl.col("size").sum().alias("bar_btc_volume") if "size" in df.columns else pl.lit(0.0).alias("bar_btc_volume"),
            pl.len().alias("tick_count"),
        ]

        if "price" in df.columns and "size" in df.columns:
            aggs.extend([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                ((pl.col("price") * pl.col("size")).sum() / (pl.col("size").sum() + 1e-9)).alias("vwap"),
            ])
        elif "micro_price" in df.columns:
            aggs.extend([
                pl.col("micro_price").first().alias("open"),
                pl.col("micro_price").max().alias("high"),
                pl.col("micro_price").min().alias("low"),
                pl.col("micro_price").last().alias("close"),
                pl.col("micro_price").mean().alias("vwap"),
            ])

        l2_aggs = []
        if "spread"              in df.columns: l2_aggs.extend([pl.col("spread").max().alias("max_spread"), pl.col("spread").mean().alias("mean_spread")])
        if "micro_price"         in df.columns: l2_aggs.append(pl.col("micro_price").std().alias("volatility"))
        if "obi_l0"              in df.columns: l2_aggs.append(pl.col("obi_l0").mean().alias("mean_obi"))
        if "deep_obi_5"          in df.columns: l2_aggs.append(pl.col("deep_obi_5").mean().alias("mean_deep_obi"))
        if "ofi"                 in df.columns: l2_aggs.append(pl.col("ofi").sum().alias("ofi"))
        if "micro_price_momentum" in df.columns: l2_aggs.append(pl.col("micro_price_momentum").sum().alias("micro_price_momentum"))
        if "bid_slope"           in df.columns: l2_aggs.append(pl.col("bid_slope").mean().alias("mean_bid_slope"))
        if "ask_slope"           in df.columns: l2_aggs.append(pl.col("ask_slope").mean().alias("mean_ask_slope"))
        if "bid_rdi"             in df.columns: l2_aggs.append(pl.col("bid_rdi").mean().alias("bid_rdi"))
        if "ask_rdi"             in df.columns: l2_aggs.append(pl.col("ask_rdi").mean().alias("ask_rdi"))
        if "pressure_ratio"      in df.columns: l2_aggs.append(pl.col("pressure_ratio").mean().alias("pressure_ratio"))
        if "__stale_l2__"        in df.columns: l2_aggs.append(pl.col("__stale_l2__").max().alias("__stale_l2__"))

        aggs.extend(l2_aggs)
        aggs.extend(ob_agg)

        group_keys = ["island_id", "bar_id"]
        if extra_group_key and extra_group_key in df.columns:
            group_keys.insert(1, extra_group_key)

        resampled = df.group_by(group_keys, maintain_order=True).agg(aggs)

        self.audit_report["generated_bars"]   = len(resampled)
        self.audit_report["islands"]          = resampled["island_id"].n_unique()
        if "bar_usd_volume" in resampled.columns:
            self.audit_report["total_usd_volume"] = float(resampled["bar_usd_volume"].sum())

        logger.info(f"Barras geradas: {len(resampled)} | Ilhas: {self.audit_report['islands']}")
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
            (sa0 + 1e-9).alias("_sa0"), (sa1 + 1e-9).alias("_sa1"),
        ])

        # ── [BOOK SKEWNESS] Pearson Moment (L200 Depth) ──
        # Optimized to avoid creating 200+ intermediate columns
        bid_cols = [f"bid_{i}_s" for i in range(self.levels) if f"bid_{i}_s" in df.columns]
        ask_cols = [f"ask_{i}_s" for i in range(self.levels) if f"ask_{i}_s" in df.columns]
        
        if bid_cols and ask_cols:
            b_mean = pl.mean_horizontal(bid_cols)
            b_m2 = pl.sum_horizontal([(pl.col(c) - b_mean).pow(2) for c in bid_cols]) / len(bid_cols)
            b_m3 = pl.sum_horizontal([(pl.col(c) - b_mean).pow(3) for c in bid_cols]) / len(bid_cols)
            
            a_mean = pl.mean_horizontal(ask_cols)
            a_m2 = pl.sum_horizontal([(pl.col(c) - a_mean).pow(2) for c in ask_cols]) / len(ask_cols)
            a_m3 = pl.sum_horizontal([(pl.col(c) - a_mean).pow(3) for c in ask_cols]) / len(ask_cols)
            
            df = df.with_columns([
                (b_m3 / (b_m2.pow(1.5) + 1e-9)).alias("book_skew_bid"),
                (a_m3 / (a_m2.pow(1.5) + 1e-9)).alias("book_skew_ask"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0.0).alias("book_skew_bid"),
                pl.lit(0.0).alias("book_skew_ask"),
            ])

        vpin_col = self.vpin_lbl
        df = df.with_columns([
            ((pl.col("max_spread").clip(upper_bound=100.0) - pl.col("_rm_s")) / (pl.col("_rs_s") + 1e-7)).alias("spread_zscore_60"),
            (pl.col("_ofi_roll") / (pl.col("_stot") + 1e-4)).clip(upper_bound=100.0).alias(vpin_col),
            (pl.col(f"micro_price_delta_{ds_l}") / (pl.col(f"ofi_delta_{ds_l}").abs() + 1e-4)).clip(upper_bound=5.0).alias("kyle_lambda"),
            (pl.col("_sbd") / (pl.col("_sb_n") + 1e-4)).alias("bid_deep_ratio"),
            (pl.col("_sad") / (pl.col("_sa_n") + 1e-4)).alias("ask_deep_ratio"),
            (pl.col("_sb0") / (pl.col("_sb1") + 1e-4)).clip(upper_bound=250.0).alias("bid_convexity"),
            (pl.col("_sa0") / (pl.col("_sa1") + 1e-4)).clip(upper_bound=250.0).alias("ask_convexity"),
            pl.col("body").clip(lower_bound=-0.05, upper_bound=0.05),
            pl.col("upper_wick").clip(upper_bound=0.01),
            pl.col("lower_wick").clip(upper_bound=0.01),
        ])

        helper = ["_rm_s", "_rs_s", "_ofi_roll", "_stot", "_sbd", "_sad", "_sb_n", "_sa_n", "_sb0", "_sb1", "_sa0", "_sa1"]
        df = df.drop([c for c in helper if c in df.columns])

        # ── [PRODUCTION CLIPPING] ──
        if self.clipping_cfg.get("enabled", False):
            target_cols = self.clipping_cfg.get("target_columns", [])
            multiplier = self.clipping_cfg.get("p99_multiplier", 15)
            floor = self.clipping_cfg.get("noise_floor", 1e-7)
            
            for col in target_cols:
                if col in df.columns:
                    # Calculate P99 for dynamic clipping (Production Parity)
                    p99 = df[col].abs().quantile(0.99)
                    if p99 > floor:
                        limit = p99 * multiplier
                        df = df.with_columns(pl.col(col).clip(lower_bound=-limit, upper_bound=limit))
                        logger.debug(f"[clipping] {col} clipped at {limit:.4e} (P99={p99:.4e} x{multiplier})")

        # Final catch-all for any remaining NaNs across all columns (Price, Features, Raw)
        df = df.fill_nan(0.0).fill_null(0.0)

        # ── [GOLD MEMORY OPTIMIZATION] ──────────────────────────────────────────
        # Drop all raw book level columns (bid_0_p, ask_0_s, etc) to save RAM.
        # These are only needed for the derivations above.
        # We protect the derivations and snippets ending in _skew, _ratio, _convexity, etc.
        raw_book_cols = [c for c in df.columns if ('bid_' in c or 'ask_' in c) and ('_p' in c or '_s' in c) 
                         and not c.endswith(('_delta_5', '_delta_30', '_deep_ratio', '_convexity', '_skew_bid', '_skew_ask'))]
        if raw_book_cols:
            logger.info(f"💾 Memory Optimization: Dropping {len(raw_book_cols)} raw book columns.")
            df = df.drop(raw_book_cols)

        return df


