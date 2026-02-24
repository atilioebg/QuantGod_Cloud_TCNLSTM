# 🛠️ 1. Setup & Environment

> **Target Audience**: DevOps, MLOps, Data Scientists.
> **Pipeline**: ETL → Labelling → Optuna → Training → XGBoost → Live

---

## 🖥️ Hardware Requirements

| Etapa | CPU | RAM | GPU | Notes |
|:---|:---:|:---:|:---:|:---|
| ETL (`run_pipeline.py`) | 14+ vCPUs | 16 GB | ❌ | CPU-bound — book reconstruction, ob500 é mais pesado |
| Labelling (`run_labelling.py`) | 14+ vCPUs | 8 GB | ❌ | 14 workers paralelos por padrão |
| Optuna (`run_optuna.py`) | 4 vCPUs | 16 GB | ✅ RTX 3090+ | LSTM+TCN requer VRAM ≥ 8GB |
| Training (`run_training.py`) | 4 vCPUs | 16 GB | ✅ RTX 3090+ | seq_len=720, batch=256 → ~4GB VRAM |
| XGBoost (`train_xgboost.py`) | 8+ vCPUs | 32 GB | ❌ | K=5 folds × treinamento do base model |
| Live Inference | 2 vCPUs | 4 GB | Opcional | ~1 predição/minuto |

**Instância RunPod recomendada para treino completo:** RTX 4090 (24GB) ou A100 (40GB).

---

## 🐍 Configuração do Ambiente

### 1. Clonar e criar virtualenv

```bash
# Clonar
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git .

# Criar e ativar virtualenv
python -m venv venv
source venv/bin/activate          # Linux/RunPod
# ou
venv\Scripts\Activate.ps1         # Windows PowerShell
```

### 2. Instalar PyTorch (CUDA 12.x — RunPod/GPU)

> **⚠️ CRÍTICO:** Sempre especifique a versão CUDA explicitamente. Nunca use `pip install torch` sem `--index-url`.

```bash
pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verificar instalação:**
```python
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | Device: NVIDIA RTX 4090
```

### 3. Instalar demais dependências

```bash
pip install -r requirements.txt
```

**Dependências principais:**

| Biblioteca | Versão | Uso |
|:---|:---|:---|
| `torch` | ≥ 2.1 | Base model (TCN+LSTM) |
| `xgboost` | ≥ 2.0 | Auditor model |
| `polars` | ≥ 0.19 | ETL + Labelling (engine rápida) |
| `pandas` | ≥ 2.0 | Compatibilidade e testes |
| `scikit-learn` | ≥ 1.3 | StandardScaler + TimeSeriesSplit |
| `optuna` | ≥ 3.4 | Hyperparameter search |
| `numpy` | ≥ 1.24 | Feature engineering |
| `pyyaml` | ≥ 6.0 | Carregamento de configs |
| `tqdm` | ≥ 4.0 | Barras de progresso no ETL/labelling |

### 4. Configurar PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:/workspace   # Linux
# ou no Windows:
$env:PYTHONPATH = "C:\Users\Atilio\Desktop\PROJETOS\PESSOAL\QuantGod_Cloud"
```

---

## 🔌 Configuração do rclone (Acesso ao Google Drive)

O pipeline lê dados brutos do Google Drive via `rclone mount`. Sem isso, o ETL não encontra os ZIPs.

### Windows (dev local)

```powershell
# Use o rclone.exe já incluído na raiz do projeto
.\rclone.exe mount drive: Z: --vfs-cache-mode full --config rclone.conf
# ⚠️ Mantenha esta janela aberta enquanto trabalhar
```

### Linux (RunPod)

```bash
# Instalar rclone (se necessário)
curl https://rclone.org/install.sh | sudo bash

# Migrar config do repositório para o sistema
mkdir -p /root/.config/rclone/
cp /workspace/rclone.conf /root/.config/rclone/rclone.conf

# Criar ponto de montagem e montar em background
mkdir -p /workspace/gdrive
rclone mount drive: /workspace/gdrive --vfs-cache-mode full --allow-other &

# Validar conexão
rclone lsd drive:
```

### Ajustar path no config

Edite `src/cloud/base_model/pre_processamento/configs/cloud_config.yaml`:
```yaml
# LOCAL (Windows)
rclone_mount: "Z:/PROJETOS/BTC_USDT_L2_2023_2026"

# RUNPOD (Linux)
rclone_mount: "/workspace/gdrive/PROJETOS/BTC_USDT_L2_2023_2026"
```

---

## 🔒 Segurança — Tokens e Credenciais

> **⚠️ NUNCA commite tokens no git.** O arquivo `runpod_token.txt` está no `.gitignore`.

Arquivos ignorados por padrão (`.gitignore`):
```
runpod_token.txt
*.token
*.secret
```

Para autenticar no GitHub a partir do RunPod:
```bash
git clone https://github.com/atilioebg/QuantGod_Cloud_TCNLSTM.git .
# Quando solicitado:
# User: atilioebg
# Password: <SEU GITHUB TOKEN> (não a senha da conta)
```

---

## ✅ Checklist Pré-Execução

```bash
# 1. Ambiente ativo
python -c "import torch; print('Torch OK:', torch.__version__)"

# 2. CUDA disponível (GPU)
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'"

# 3. rclone montado (verificar que os ZIPs estão acessíveis)
ls /workspace/gdrive/PROJETOS/BTC_USDT_L2_2023_2026/2024/ | head -5

# 4. PYTHONPATH correto
python -c "from src.cloud.base_model.models.model import Hybrid_TCN_LSTM; print('Import OK')"

# 5. Testes de integridade (sem dados, sem GPU — < 30s)
venv/bin/python -m pytest tests/test_config_integrity.py tests/test_meta_features.py -v
```
