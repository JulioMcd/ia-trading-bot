# 🛡️ ML Trading Bot - Sistema Seguro
## Documentação Completa para Produção

### 🚀 Visão Geral

Este sistema implementa **TODAS as melhorias críticas** identificadas no bot original:

- ❌ **MARTINGALE REMOVIDO** - Substituído por Kelly Criterion científico
- ✅ **RISK MANAGEMENT AVANÇADO** - Circuit breakers e controles de risco
- ✅ **ML AVANÇADO** - 35+ features, ensemble de modelos, validação temporal
- ✅ **SEGURANÇA COMPLETA** - API key, rate limiting, CORS seguro
- ✅ **KELLY CRITERION** - Position sizing matemático otimizado
- ✅ **MONITORAMENTO** - Logs, métricas, alertas em tempo real

---

## 📋 Índice

1. [Instalação e Setup](#instalação-e-setup)
2. [Configuração de Segurança](#configuração-de-segurança)
3. [Sistema de Risk Management](#sistema-de-risk-management)
4. [Kelly Criterion](#kelly-criterion)
5. [Sistema ML Avançado](#sistema-ml-avançado)
6. [API e Segurança](#api-e-segurança)
7. [Deployment para Produção](#deployment-para-produção)
8. [Monitoramento](#monitoramento)
9. [Troubleshooting](#troubleshooting)

---

## 🔧 Instalação e Setup

### 1. Pré-requisitos

```bash
# Python 3.8+ obrigatório
python --version  # Deve ser 3.8+

# Git para clonar repositório
git --version
```

### 2. Instalação

```bash
# 1. Clonar/baixar arquivos
mkdir ml-trading-secure
cd ml-trading-secure

# 2. Criar ambiente virtual
python -m venv venv

# 3. Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Executar setup inicial
python init_script.py
```

### 3. Configuração Inicial

```bash
# Executar configuração segura
python secure_config.py

# Verificar sistema
python init_script.py --check-env

# Executar migrações
python migrations.py
```

---

## 🔐 Configuração de Segurança

### 1. Variáveis de Ambiente Obrigatórias

Crie arquivo `.env`:

```bash
# === SEGURANÇA OBRIGATÓRIA ===
API_KEY=seu_api_key_seguro_aqui_32_chars
API_KEY_REQUIRED=true
ENVIRONMENT=production

# === DERIV API ===
DERIV_TOKEN=seu_token_deriv_aqui

# === RISK MANAGEMENT ===
MAX_DAILY_LOSS_PCT=3.0
MAX_DRAWDOWN_PCT=10.0
MAX_POSITION_SIZE_PCT=1.5
CIRCUIT_BREAKER_ENABLED=true
KELLY_ENABLED=true

# === ML AVANÇADO ===
ENSEMBLE_ENABLED=true
TEMPORAL_VALIDATION=true
MIN_CONFIDENCE_THRESHOLD=0.70

# === CORS (Especificar domínios) ===
CORS_ORIGINS=https://seu-dominio.com,http://localhost:3000

# === RATE LIMITING ===
RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=60

# === BACKUP ===
AUTO_BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=4

# === LOGGING ===
LOG_LEVEL=INFO
ALERTS_ENABLED=true
```

### 2. Gerar API Key Segura

```python
# Executar para gerar nova API key
python -c "
from secure_config import SecureConfigManager
config = SecureConfigManager()
print('Nova API Key:', config.generate_api_key())
"
```

### 3. Checklist de Segurança

```bash
# Executar checklist antes de produção
python production_deployment.py --check-only
```

**Deve mostrar 100% PASS para produção!**

---

## 🛡️ Sistema de Risk Management

### 1. Limites Automáticos

O sistema **BLOQUEIA AUTOMATICAMENTE** trades se:

- **Perda diária > 3%** do saldo inicial
- **Drawdown > 10%** do pico
- **Position size > 1.5%** do saldo
- **5+ perdas consecutivas**
- **3+ trades simultâneos**
- **Win rate < 30%** (após 20 trades)

### 2. Circuit Breakers

**Ativação automática quando:**
- Perda de 3% em 1 hora
- Drawdown de 10%
- 5 perdas consecutivas
- Risk score > 90

**Reset manual:**
```javascript
// No frontend
resetCircuitBreaker()
```

**Reset via API:**
```bash
curl -X POST http://localhost:8000/risk/reset \
  -H "X-API-Key: sua_api_key"
```

### 3. Monitoramento de Risco

```python
# Verificar métricas de risco
from risk_manager import risk_manager

# Status atual
status = risk_manager.get_dashboard_data()
print(f"Risk Level: {status['risk_level']}")
print(f"Risk Score: {status['risk_score']}/100")

# Verificar se pode fazer trade
can_trade = risk_manager.validate_trade(
    stake=10.0, 
    balance=1000.0, 
    trades=[]
)
print(f"Pode fazer trade: {can_trade['allowed']}")
```

---

## 💰 Kelly Criterion

### 1. Como Funciona

O sistema calcula **automaticamente** o tamanho ótimo da posição:

```
Kelly Fraction = (b × p - q) / b

Onde:
- b = Avg Win / Avg Loss
- p = Win Rate
- q = 1 - Win Rate
```

### 2. Limitações de Segurança

- **Máximo 25%** do saldo (Quarter Kelly)
- **Mínimo $0.35** (limite Deriv)
- **Risk-adjusted** com fator 0.8
- **Respeitam limites** do Risk Manager

### 3. Uso Prático

```javascript
// O sistema calcula automaticamente
calculateKellyPosition()

// Mostra no frontend:
// "💰 Posição Ótima: $2.45"
// "🎯 Kelly Fraction: 2.1% do saldo"
```

### 4. API Kelly

```bash
# Obter posição ótima
curl http://localhost:8000/kelly/optimal \
  -H "X-API-Key: sua_api_key"

# Resposta:
{
  "optimal_stake": 2.45,
  "kelly_fraction": 0.021,
  "confidence": 0.65,
  "win_rate": 0.58
}
```

---

## 🧠 Sistema ML Avançado

### 1. Arquitetura

**Ensemble de 5+ Modelos:**
- Random Forest
- Gradient Boosting  
- Logistic Regression
- SVM
- Neural Network
- XGBoost (opcional)

**35+ Features Avançadas:**
- Momentum (5, 10, 20 períodos)
- Volatilidade realizada
- RSI e divergências
- MACD
- Bandas de Bollinger
- Support/Resistance
- Horários de mercado
- Correlações
- Features de risco

### 2. Validação Temporal

✅ **NUNCA mistura dados futuros no treino**
- TimeSeriesSplit para validação
- Features criadas apenas com dados passados
- Teste em dados "não vistos"

### 3. API ML

```bash
# Análise ML
curl -X POST http://localhost:8000/ml/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sua_api_key" \
  -d '{
    "symbol": "R_50",
    "current_price": 1000.0,
    "risk_metrics": {...}
  }'

# Predição Ensemble
curl -X POST http://localhost:8000/ml/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sua_api_key" \
  -d '{
    "symbol": "R_50",
    "stake": 2.0,
    "duration": "5t"
  }'

# Estatísticas
curl http://localhost:8000/ml/stats \
  -H "X-API-Key: sua_api_key"
```

### 4. Retreinamento

```python
# Automático a cada 30 trades
# Manual via API:
import requests

response = requests.post(
    "http://localhost:8000/ml/train",
    headers={"X-API-Key": "sua_api_key"}
)
```

---

## 🔒 API e Segurança

### 1. Autenticação

**Todos os endpoints requerem API Key:**

```bash
curl -H "X-API-Key: sua_api_key_aqui" \
  http://localhost:8000/endpoint
```

### 2. Rate Limiting

- **60 requests/minuto** por IP
- **1000 requests/hora** por IP
- Bloqueio automático por 1 hora se exceder

### 3. CORS Seguro

```python
# Apenas origens específicas em produção
CORS_ORIGINS = [
    "https://seu-dominio.com",
    "https://app.seu-site.com"
]

# NUNCA usar "*" em produção!
```

### 4. Headers de Segurança

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
```

### 5. Endpoints Principais

```bash
# Health Check
GET /health

# Trading
POST /trade/execute
GET /trade/history
POST /trade/stop

# Risk Management  
GET /risk/status
POST /risk/reset
GET /risk/limits

# ML
POST /ml/predict
POST /ml/analyze
GET /ml/stats
POST /ml/train

# Kelly
GET /kelly/optimal
GET /kelly/stats
```

---

## 🚀 Deployment para Produção

### 1. Checklist Pré-Deploy

```bash
# OBRIGATÓRIO antes de produção
python production_deployment.py --check-only

# Deve mostrar:
# ✅ API Key configurada
# ✅ Rate Limiting ativo  
# ✅ CORS configurado
# ✅ Risk Management ativo
# ✅ Circuit Breakers ativos
# ✅ Kelly Criterion ativo
# ✅ ML Temporal Validation
# ✅ Backup automático
# ✅ Logs de segurança
# ✅ Headers de segurança
# 
# 🔐 Score de Segurança: 100% (10/10)
# ✅ Checklist de segurança aprovado
```

### 2. Deploy Local

```bash
# Desenvolvimento
python production_deployment.py

# Ou usando uvicorn diretamente
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Deploy Render.com

```yaml
# render.yaml
services:
  - type: web
    name: ml-trading-secure
    env: python
    buildCommand: |
      pip install -r requirements.txt &&
      python init_script.py --no-tests
    startCommand: python production_deployment.py
    plan: free
    healthCheckPath: /health
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: API_KEY_REQUIRED  
        value: true
      - key: KELLY_ENABLED
        value: true
```

### 4. Deploy Heroku

```bash
# Procfile
web: python production_deployment.py

# Deploy
git add .
git commit -m "Deploy sistema seguro"
git push heroku main
```

### 5. Deploy Docker

```dockerfile
FROM python:3.11-slim

WORK