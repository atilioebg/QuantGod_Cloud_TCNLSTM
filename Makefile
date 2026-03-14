# =============================================================================
# QUANTGOD TCN+LSTM — Pipeline Orchestration Makefile
# =============================================================================
# Versão: 5.0 (ASCII Safe)
# =============================================================================

.PHONY: help install-cpu install-gpu clean-logs clean-results clean-deep pull train-manager train-full etl lab split monitor

# Variáveis globais
PYTHON = python3
PIP = pip3
PYTHONPATH_VAL = $(shell pwd)
EXPORT_PYPATH = export PYTHONPATH="$${PYTHONPATH}:$(PYTHONPATH_VAL)"

help:
	@echo "============================================================================="
	@echo "QUANTGOD PIPELINE — COMANDOS DISPONIVEIS"
	@echo "============================================================================="
	@echo "SETUP:"
	@echo "  make install-cpu      Instala ambiente VENV para CPU"
	@echo "  make install-gpu      Instala ambiente VENV para GPU (CUDA 12.1+)"
	@echo "  make pull             Sincroniza do Repo (quant_god_full_v2)"
	@echo ""
	@echo "LIMPEZA:"
	@echo "  make clean-logs       Apaga apenas os arquivos de log"
	@echo "  make clean-results    Apaga modelos locais, bancos Optuna e CSVs"
	@echo "  make clean-deep       LIMPEZA TOTAL (Dados, Logs, Modelos, Optuna)"
	@echo ""
	@echo "EXECUCAO:"
	@echo "  make train-manager    Roda o Manager (Fluxo automatico v5.0)"
	@echo "  make train-full       Executa manualmente TODAS as fases (ETL -> OOS)"
	@echo "  make etl              Executa apenas o Pre-processamento"
	@echo "  make lab              Executa o Labelling"
	@echo "  make split            Executa o Split de datasets"
	@echo ""
	@echo "UTILITARIOS:"
	@echo "  make monitor          Visualiza o log de treino em tempo real"
	@echo "============================================================================="

# ── [SETUP] ──────────────────────────────────────────────────────────────────

install-cpu:
	$(PYTHON) -m venv venv_cpu
	. venv_cpu/bin/activate && $(PIP) install --upgrade pip
	. venv_cpu/bin/activate && $(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	. venv_cpu/bin/activate && $(PIP) install -r requirements.txt
	@echo "[OK] Ambiente CPU pronto. Use 'source venv_cpu/bin/activate'"

install-gpu:
	$(PYTHON) -m venv venv_gpu
	. venv_gpu/bin/activate && $(PIP) install --upgrade pip
	. venv_gpu/bin/activate && $(PIP) install -r requirements.txt
	@echo "[OK] Ambiente GPU pronto. Use 'source venv_gpu/bin/activate'"

pull:
	git pull origin quant_god_full_v2

# ── [LIMPEZA] ───────────────────────────────────────────────────────────────

clean-logs:
	rm -rf logs/*
	@echo "[OK] Logs limpos."

clean-results:
	rm -rf RESULTADOS_* optuna_tcn_lstm_*.db *.csv best_params.json best_dir_params.json
	@echo "[OK] Resultados e bases Optuna removidos."

clean-deep:
	@echo "[WARNING] Iniciando limpeza profunda..."
	rm -rf data/L2/pre_processed/* data/L2/labelled/* data/L2/specialized/*
	rm -rf data/auditor/* logs/* RESULTADOS_* optuna_tcn_lstm_*.db
	rm -rf src/cloud/base_model/otimizacao/best_params.json src/cloud/base_model/otimizacao/best_dir_params.json
	$(PYTHON) src/cloud/base_model/utils/transfer.py --clean
	@echo "[OK] Sistema resetado para estado inicial."

# ── [EXECUCAO] ──────────────────────────────────────────────────────────────

train-manager:
	$(EXPORT_PYPATH) && $(PYTHON) src/cloud/base_model/otimizacao/run_manager.py

etl:
	$(EXPORT_PYPATH) && $(PYTHON) src/cloud/base_model/pre_processamento/orchestration/run_pipeline.py

lab:
	$(EXPORT_PYPATH) && $(PYTHON) src/cloud/base_model/labelling/run_labelling.py

split:
	$(EXPORT_PYPATH) && $(PYTHON) src/cloud/base_model/treino/split_dataset.py

train-full: etl lab split train-manager
	@echo "[FINISH] Ciclo completo executado."

# ── [MONITORAMENTO] ─────────────────────────────────────────────────────────

monitor:
	@cat $$(ls -t logs/run_manager/*.log | head -n 1)
