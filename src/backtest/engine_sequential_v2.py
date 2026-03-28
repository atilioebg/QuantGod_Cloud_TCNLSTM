import polars as pl
import pandas as pd
import numpy as np
import yaml
import logging
import json
import gc
from pathlib import Path
from tqdm import tqdm
from src.cloud.base_model.utils.path_utils import get_labelled_dir
from src.cloud.execution.inference_service import InferenceService

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SequentialBacktestEngineV2:
    def __init__(self, config_path: str):
        # 1. Carregar Configurações
        with open("src/cloud/base_model/configs/master_config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        with open(config_path, 'r') as f:
            backtest_cfg = yaml.safe_load(f)
        
        self.simulation_cfg = backtest_cfg['simulation']
        
        # 2. Inicializar Serviço de Inferência (Resolução de Caminhos V3.5 Autoritária)
        # O InferenceService agora trava a âncora diretamente da auditoria
        self.inference = InferenceService(self.config)
        
        # 3. Parâmetros da Simulação
        self.initial_capital = self.simulation_cfg['initial_capital']
        self.balance = self.initial_capital
        self.leverage = self.simulation_cfg['leverage']
        self.trading_fee = self.simulation_cfg['trading_fee']
        self.auditor_threshold = self.simulation_cfg['auditor_threshold']
        
        # Sincronia de Volatilidade (Prado Multipliers)
        self.pt_mult = self.simulation_cfg.get('pt_multiplier', 0.65)
        self.sl_mult = self.simulation_cfg.get('sl_multiplier', 0.33)
        self.vol_span = self.simulation_cfg.get('vol_span', 144)
        
        # Horizonte de Saída (Barreira Vertical) - Sincronizado com o Labelling
        self.exit_horizon_ms = self.simulation_cfg.get('horizon_minutes', 15) * 60 * 1000
        
        # Resultados
        self.trades = []
        self.output_dir = Path(self.config['pipeline_paths']['local_data_root']) / "backtest_results_sequential_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_sequential_backtest(self):
        data_dir = Path(get_labelled_dir(self.config))
        # Busca recursiva para encontrar parquets em subpastas (val/train)
        parquet_files = sorted(list(data_dir.rglob("*.parquet")))

        # [VERIFICAÇÃO V2.5] Filtro flexível para 21 de Março (hífen ou underscore)
        target_date_h = "2026-03-21"
        target_date_u = "2026_03_21"
        
        parquet_files = [f for f in parquet_files if target_date_h in f.name or target_date_u in f.name]
        
        if not parquet_files:
            logger.error(f"❌ No labelled parquet files found for {target_date_h}/{target_date_u} in {data_dir}")
            # Diagnóstico: listar o que foi encontrado (primeiros 3) para ajudar o usuário
            all_found = sorted(list(data_dir.rglob("*.parquet")))[:3]
            if all_found:
                logger.info(f"🔍 Filenames found in dir (sample): {[f.name for f in all_found]}")
            return

        logger.info(f"🚀 [V2.5 Twin] Starting Event-Driven Backtest on {len(parquet_files)} day(s)...")
        
        for pf in tqdm(parquet_files, desc="Processing Days"):
            self.execute_single_parquet(pf)

    def execute_single_parquet(self, pf: Path):
        try:
            # Polars para leitura rápida, Pandas para o loop complexo
            df = pl.read_parquet(pf).to_pandas()
            
            # --- [V2.4] Preparação de Arrays para Performance ---
            timestamps = df['ts'].values
            prices = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # --- [V2.4] Configurações do Twin Event Sampler ---
            event_cfg = self.simulation_cfg.get('event_sampling', {})
            dollar_thresh = event_cfg.get('dollar_threshold_usd', 500000.0)
            tick_thresh   = event_cfg.get('tick_threshold', 5000)
            ofi_thresh    = event_cfg.get('information_threshold_ofi', 100.0)
            time_thresh_s = event_cfg.get('resample_freq_min', 5) * 60
            
            # --- [V2.4] Estado do Acumulador de Eventos ---
            acc_dollar, acc_ticks, acc_ofi = 0.0, 0, 0.0
            last_bar_ts = timestamps[0] / 1000
            last_bar_price = prices[0]
            virtual_bar_rets = []
            
            # Vol âncora inicial: usamos a do primeiro segundo como fallback até termos 5 barras virtuais
            anchor_vol = 0.0003 
            
            busy_until = 0 # Milissegundos
            S = self.inference.seq_len # Janela de entrada do modelo
            
            # --- LOOP PRINCIPAL SEGUNDO A SEGUNDO ---
            for idx in range(S, len(df)):
                curr_price = prices[idx]
                curr_ts_ms = timestamps[idx]
                curr_ts_s  = curr_ts_ms / 1000
                
                # 1. Acumular Atividade (Twin logic com o Pre-Processamento)
                # 'bar_usd_volume' e 'ofi' são colunas nativas do parquet labelled
                acc_dollar += df['bar_usd_volume'].iloc[idx] if 'bar_usd_volume' in df.columns else (df['usd_volume'].iloc[idx] if 'usd_volume' in df.columns else 0)
                acc_ticks += 1
                acc_ofi += abs(df['ofi'].iloc[idx]) if 'ofi' in df.columns else 0
                time_elapsed = curr_ts_s - last_bar_ts
                
                # Check Trigger de Barra Virtual (Idêntico ao EventSampler)
                if (acc_dollar >= dollar_thresh or acc_ticks >= tick_thresh or 
                    acc_ofi >= ofi_thresh or time_elapsed >= time_thresh_s):
                    
                    # Calcula o retorno log entre esta barra virtual e a anterior
                    v_ret = np.log(curr_price / last_bar_price)
                    virtual_bar_rets.append(v_ret)
                    
                    # Mantemos histórico suficiente para o span (Prado recomenda > 2x span)
                    if len(virtual_bar_rets) > self.vol_span * 2: 
                        virtual_bar_rets.pop(0)
                    
                    if len(virtual_bar_rets) > 5:
                        # Cálculo real da Volatilidade de Eventos conforme Labelling
                        anchor_vol = pd.Series(virtual_bar_rets).ewm(span=self.vol_span).std().iloc[-1]
                    
                    # Reset accumulators para a próxima barra virtual
                    acc_dollar, acc_ticks, acc_ofi = 0.0, 0, 0.0
                    last_bar_ts, last_bar_price = curr_ts_s, curr_price

                # 2. Lógica de Pulo: se já estamos em um trade, apenas acumulamos dados mas não abrimos outro
                if curr_ts_ms < busy_until:
                    continue

                # 3. Inferência do Modelo (InferenceStack: Foundation + Specialists + Auditor)
                # Passamos o dataframe fatiado com seq_len para a inferência
                sig, conf = self.inference.predict_batch(df.iloc[idx:idx+1])
                sig, conf = sig[0], conf[0]
                
                # Auditor aprova o sinal? (Threshold 0.50)
                if sig in [0, 2] and conf >= self.auditor_threshold:
                    entry_price = curr_price
                    entry_ts = curr_ts_ms
                    
                    # Barrier Scale Fix: Usamos a anchor_vol (Volatilidade de Eventos)
                    # Isso dá ao trade o espaço de manobra que ele teve no treino.
                    curr_vol = max(anchor_vol, 1e-7)
                    
                    if sig == 2: # LONG
                        target_tp = entry_price * np.exp(curr_vol * self.pt_mult)
                        target_sl = entry_price * np.exp(-curr_vol * self.sl_mult)
                    else: # SHORT
                        target_tp = entry_price * np.exp(-curr_vol * self.pt_mult)
                        target_sl = entry_price * np.exp(curr_vol * self.sl_mult)
                    
                    # --- [TRIPLE BARRIER SEARCH] ---
                    # Procuramos segundo a segundo qual barreira será tocada primeiro
                    exit_idx = idx + 1
                    outcome = "TIME"
                    exit_price = entry_price
                    
                    while exit_idx < len(prices):
                        fwd_ts = timestamps[exit_idx]
                        
                        # Barreira 1: Vertical (Tempo Limite)
                        if (fwd_ts - entry_ts) > self.exit_horizon_ms:
                            outcome = "TIME"; exit_price = prices[exit_idx]; break
                        
                        # Barreira 2 e 3: TP e SL (Horizontal)
                        if sig == 2: # LONG
                            if highs[exit_idx] >= target_tp:
                                outcome = "TP"; exit_price = target_tp; break
                            if lows[exit_idx] <= target_sl:
                                outcome = "SL"; exit_price = target_sl; break
                        else: # SHORT
                            if lows[exit_idx] <= target_tp:
                                outcome = "TP"; exit_price = target_tp; break
                            if highs[exit_idx] >= target_sl:
                                outcome = "SL"; exit_price = target_sl; break
                        exit_idx += 1
                    
                    # Se o dia acabou sem tocar barreiras
                    if exit_idx >= len(prices):
                        exit_idx = len(prices) - 1; exit_price = prices[exit_idx]; outcome = "DAY_CLOSE"
                    
                    # --- Cálculos do Trade Consolidado ---
                    trade_return = (exit_price - entry_price) / entry_price if sig == 2 else (entry_price - exit_price) / entry_price
                    net_return = trade_return - (self.trading_fee * 2) # Fee de compra e venda
                    
                    old_balance = self.balance
                    self.balance *= (1 + net_return * self.leverage)
                    
                    duration_min = (timestamps[exit_idx] - entry_ts) / 60000
                    busy_until = timestamps[exit_idx] # Silêncio até a saída do trade
                    
                    self.trades.append({
                        "entry_ts": pd.to_datetime(entry_ts, unit='ms'),
                        "exit_ts": pd.to_datetime(timestamps[exit_idx], unit='ms'),
                        "side": "BUY" if sig == 2 else "SELL",
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "outcome": outcome,
                        "net_return": float(net_return),
                        "duration_min": float(duration_min),
                        "balance": float(self.balance)
                    })

            # Gestão de memória por dia
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Error processing day {pf.name}: {e}")

    def print_report(self, df_trades, final_balance):
        if df_trades.empty:
            logger.warning("No trades recorded.")
            return

        total_days = (df_trades['entry_ts'].max() - df_trades['entry_ts'].min()).total_seconds() / (60 * 60 * 24)
        total_days = max(total_days, 0.1) # Avoid zero division for single day
        
        wins = df_trades[df_trades['net_return'] > 0]
        
        report = {
            "initial_capital": self.initial_capital,
            "final_balance": float(final_balance),
            "roi_pct": float((final_balance / self.initial_capital - 1) * 100),
            "total_trades": len(df_trades),
            "win_rate": float(len(wins) / len(df_trades)),
            "avg_duration_min": float(df_trades['duration_min'].mean()),
            "trades_per_day": float(len(df_trades) / total_days)
        }
        
        logger.info("============== [V2.4 Twin] REALISTIC SEQUENTIAL REPORT ==============")
        print(json.dumps(report, indent=4))
        
        # Salva o log detalhado de trades
        report_path = self.output_dir / "trades_sequential_v2.csv"
        df_trades.to_csv(report_path, index=False)
        logger.info(f"✅ Full trade journal saved to {report_path}")

if __name__ == "__main__":
    import os
    config_path = os.path.join("src", "backtest", "backtest_config_cloud.yaml")
    engine = SequentialBacktestEngineV2(config_path)
    engine.run_sequential_backtest()
