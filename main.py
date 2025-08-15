# deploy_automation.py - Scripts de deploy automatizado

import os
import subprocess
import requests
import time
import json
from datetime import datetime

class TradingBotDeployer:
    """Automatiza o deploy do Trading Bot no Render"""
    
    def __init__(self):
        self.render_api_key = os.getenv('RENDER_API_KEY')
        self.github_repo = os.getenv('GITHUB_REPO')
        self.base_url = "https://api.render.com/v1"
        
    def create_project_structure(self):
        """Cria estrutura completa do projeto"""
        print("🚀 Criando estrutura do projeto...")
        
        # Criar diretórios
        directories = [
            "trading-bot-ml",
            "trading-bot-ml/api",
            "trading-bot-ml/frontend", 
            "trading-bot-ml/docs",
            "trading-bot-ml/tests",
            "trading-bot-ml/data",
            "trading-bot-ml/models"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ Criado: {directory}")
        
        # Criar arquivos principais
        self.create_main_files()
        self.create_requirements()
        self.create_docker_files()
        self.create_documentation()
        
    def create_main_files(self):
        """Cria arquivos principais da aplicação"""
        print("📄 Criando arquivos principais...")
        
        # main.py (API principal)
        main_content = '''# main.py - Trading Bot API Principal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Importar módulos
from advanced_ml_features import AdvancedMLFeatures
from backtesting_system import BacktestingEngine

app = FastAPI(
    title="Trading Bot ML API",
    description="API completa com Machine Learning para trading automatizado",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "🤖 Trading Bot ML API",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "Machine Learning completo",
            "Decisões autônomas",
            "Backtesting avançado",
            "Análise técnica completa"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        
        with open("trading-bot-ml/main.py", "w") as f:
            f.write(main_content)
        
        print("  ✅ main.py criado")
    
    def create_requirements(self):
        """Cria arquivo requirements.txt completo"""
        print("📦 Criando requirements.txt...")
        
        requirements = '''# Trading Bot ML Requirements
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
websockets==12.0
requests==2.31.0
python-multipart==0.0.6
pydantic==2.5.0

# Análise Técnica
TA-Lib==0.4.28

# Machine Learning Avançado
xgboost==2.0.2
lightgbm==4.1.0
catboost==1.2.2

# Visualização e Análise
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0

# Base de Dados
sqlite3

# Utilitários
python-dotenv==1.0.0
schedule==1.2.0

# Deploy
gunicorn==21.2.0
'''
        
        with open("trading-bot-ml/requirements.txt", "w") as f:
            f.write(requirements)
        
        print("  ✅ requirements.txt criado")
    
    def create_docker_files(self):
        """Cria arquivos Docker"""
        print("🐳 Criando arquivos Docker...")
        
        dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    wget \\
    build-essential \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Instalar TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \\
    tar -xzf ta-lib-0.4.0-src.tar.gz && \\
    cd ta-lib/ && \\
    ./configure --prefix=/usr && \\
    make && \\
    make install && \\
    cd .. && \\
    rm -rf ta-lib*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios necessários
RUN mkdir -p /app/data /app/models /app/logs

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
'''
        
        with open("trading-bot-ml/Dockerfile", "w") as f:
            f.write(dockerfile)
        
        # Docker Compose para desenvolvimento
        docker_compose = '''version: '3.8'

services:
  trading-bot-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    
  # Banco de dados (opcional)
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
'''
        
        with open("trading-bot-ml/docker-compose.yml", "w") as f:
            f.write(docker_compose)
        
        print("  ✅ Arquivos Docker criados")
    
    def create_documentation(self):
        """Cria documentação completa"""
        print("📚 Criando documentação...")
        
        readme = '''# 🤖 Trading Bot com Machine Learning Total

> Bot de trading completamente autônomo com Machine Learning avançado para Deriv API

## 🌟 Características Principais

### 🧠 Machine Learning Avançado
- **Decisões 100% Autônomas**: IA decide quando, como e quanto operar
- **Análise Técnica Completa**: 50+ indicadores técnicos automatizados
- **Reconhecimento de Padrões**: Head & Shoulders, Triângulos, Breakouts
- **Detecção de Anomalias**: Identificação automática de condições anômalas
- **Regime de Mercado**: Bull, Bear, Sideways automaticamente identificados

### ⚡ Funcionalidades Avançadas
- **Backtesting Completo**: Teste estratégias com dados históricos
- **Stake Dinâmico**: Cálculo automático baseado em risco e confiança
- **Timeframe Inteligente**: Seleção automática baseada na volatilidade
- **Anti-Loop**: Sistema robusto contra loops de erro
- **Cooling Period**: Pausas inteligentes após perdas

### 🎯 Performance
- **Taxa de Acerto**: Otimizada via Machine Learning
- **Gerenciamento de Risco**: Automático e inteligente
- **Análise de Sentimento**: Baseada em trades recentes
- **Fibonacci Automático**: Níveis calculados automaticamente

## 🚀 Deploy Rápido

### 1. Render (Recomendado)
```bash
# 1. Fork este repositório
# 2. Conecte no Render.com
# 3. Configure as variáveis:
#    - ENVIRONMENT=production
#    - PORT=8000
# 4. Deploy automático!
```

### 2. Docker Local
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/trading-bot-ml
cd trading-bot-ml

# Execute com Docker
docker-compose up -d

# API disponível em http://localhost:8000
```

### 3. Instalação Manual
```bash
# Instalar dependências
pip install -r requirements.txt

# Instalar TA-Lib (necessário)
# Ubuntu/Debian:
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make && sudo make install

# Windows: baixar binários pré-compilados

# Executar API
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🎮 Como Usar

### 1. Configurar Frontend
```html
<!-- URL da sua API no Render -->
<input id="mlApiUrl" value="https://sua-api.onrender.com">
```

### 2. Conectar Deriv API
```javascript
// Token da Deriv (app.deriv.com/account/api-token)
const token = "seu_token_aqui";
```

### 3. Ativar Automação Total
```javascript
// Bot toma TODAS as decisões automaticamente
startFullAutomation();
```

## 📊 Endpoints da API

### Machine Learning
- `POST /signal` - Obter sinal de trading
- `POST /auto-decision` - Decisão automática completa
- `POST /risk-assessment` - Avaliação de risco
- `POST /market-analysis` - Análise completa do mercado

### Funcionalidades Avançadas
- `POST /advanced-analysis` - Análise com 50+ indicadores
- `POST /smart-timeframe` - Timeframe inteligente
- `POST /dynamic-stake` - Stake dinâmico
- `POST /backtest` - Backtesting completo

### Sistema
- `GET /health` - Status da API
- `GET /model-status` - Status dos modelos ML
- `GET /features` - Lista de funcionalidades

## 🔧 Configuração Avançada

### Variáveis de Ambiente
```bash
# Produção
ENVIRONMENT=production
PORT=8000
LOG_LEVEL=INFO

# Desenvolvimento
ENVIRONMENT=development
DEBUG=true
```

### Parâmetros de Trading
```python
# Configurar no código
CONFIG = {
    'MIN_STAKE': 0.35,
    'MAX_STAKE': 50.0,
    'AUTO_DECISION_INTERVAL': 30000,  # 30 segundos
    'RISK_TOLERANCE': 'medium'
}
```

## 📈 Backtesting

### Teste Rápido (30 dias)
```bash
curl -X GET "https://sua-api.onrender.com/backtest/quick"
```

### Teste Personalizado
```bash
curl -X POST "https://sua-api.onrender.com/backtest" \\
  -H "Content-Type: application/json" \\
  -d '{
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "initial_balance": 1000,
    "symbol": "R_50"
  }'
```

## ⚠️ Avisos Importantes

### Risco
- **Trading envolve risco**: Nunca opere mais do que pode perder
- **Teste primeiro**: Use conta demo antes da real
- **Monitore sempre**: IA é auxiliar, não substitui supervisão

### Segurança
- **Token seguro**: Nunca compartilhe seu token da Deriv
- **Permissões mínimas**: Use apenas permissões necessárias
- **Monitoramento**: Acompanhe todas as operações

## 🆘 Suporte

### Problemas Comuns
1. **"ML API não conecta"**: Verifique URL da API
2. **"Token inválido"**: Regenere token na Deriv
3. **"Modelos não treinados"**: Aguarde 1-2 minutos após deploy

### Logs e Debug
```bash
# Verificar logs no Render
# Dashboard > Sua API > Logs

# Debug local
docker-compose logs -f trading-bot-api
```

## 🚀 Roadmap

- [ ] Integração com TradingView
- [ ] Mais exchanges (Binance, IQ Option)
- [ ] Mobile app
- [ ] Telegram bot
- [ ] Copy trading

## 📄 Licença

MIT License - Use como quiser, mas por sua conta e risco!

---

**⚡ Bot criado com IA avançada para máxima performance autônoma!**
'''
        
        with open("trading-bot-ml/README.md", "w") as f:
            f.write(readme)
        
        # API Documentation
        api_docs = '''# 📡 API Documentation

## Base URL
```
https://sua-api.onrender.com
```

## Authentication
Não é necessária autenticação para a API ML. A autenticação é feita diretamente com a Deriv via WebSocket.

## Endpoints

### 🧠 Machine Learning Core

#### GET /
Informações básicas da API

**Response:**
```json
{
  "message": "🤖 Trading Bot ML API",
  "version": "2.0.0",
  "status": "online",
  "features": ["Machine Learning completo", "..."]
}
```

#### POST /signal
Obtém sinal de trading inteligente

**Request:**
```json
{
  "symbol": "R_50",
  "current_price": 1000.50,
  "account_balance": 1000,
  "recent_trades": [],
  "market_data": []
}
```

**Response:**
```json
{
  "direction": "CALL",
  "confidence": 85.5,
  "timeframe": "5t",
  "entry_price": 1000.50,
  "reasoning": "RSI oversold + MACD bullish",
  "ml_status": "active"
}
```

#### POST /auto-decision
Decisão automática completa (recomendado)

**Response:**
```json
{
  "action": "trade",
  "direction": "CALL",
  "timeframe": "5t", 
  "stake": 5.00,
  "confidence": 87.2,
  "reasoning": "ML Analysis: RSI oversold, MACD bullish",
  "risk_assessment": {
    "risk_level": "medium",
    "risk_score": 45.2
  }
}
```

### 📊 Advanced Analysis

#### POST /advanced-analysis
Análise completa com 50+ indicadores

**Response:**
```json
{
  "indicators": {
    "rsi": 28.5,
    "macd": 0.5,
    "bb_upper": 1020.5,
    "bb_lower": 980.2,
    "stoch": 25.8,
    "adx": 45.2,
    "fib_levels": {...},
    "support_resistance": {...}
  },
  "anomalies": {
    "anomaly_detected": false,
    "anomaly_score": -0.2
  },
  "market_regime": {
    "regime": "bullish",
    "confidence": 0.75
  },
  "patterns": {
    "patterns": [
      {
        "name": "Double Bottom",
        "signal": "bullish", 
        "strength": 0.7
      }
    ]
  }
}
```

### 🎯 Smart Features

#### POST /smart-timeframe
Timeframe inteligente baseado na volatilidade

**Response:**
```json
{
  "type": "t",
  "duration": 5,
  "reasoning": "Volatilidade média - timeframe balanceado",
  "volatility": 0.025,
  "market_condition": "normal"
}
```

#### POST /dynamic-stake
Stake dinâmico baseado em múltiplos fatores

**Response:**
```json
{
  "stake": 7.50,
  "percentage": 1.5,
  "reasoning": "Confidence: 85%, Risk: medium, WinRate: 65%",
  "factors": {
    "confidence": 85.0,
    "win_rate": 65.0,
    "account_balance": 500.0
  }
}
```

### 📈 Backtesting

#### GET /backtest/quick
Backtesting rápido (30 dias simulados)

**Response:**
```json
{
  "status": "success",
  "type": "quick_backtest",
  "report": {
    "summary": {
      "initial_balance": 1000,
      "final_balance": 1150.50,
      "total_return": 15.05,
      "total_trades": 45,
      "win_rate": 67.8
    },
    "performance": {
      "total_pnl": 150.50,
      "max_drawdown": -25.80,
      "sharpe_ratio": 1.85
    },
    "recommendations": [
      "🟢 Excelente win rate - manter estratégia"
    ]
  }
}
```

#### POST /backtest
Backtesting personalizado

**Request:**
```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-01-31", 
  "symbol": "R_50",
  "initial_balance": 1000,
  "strategy_params": {}
}
```

### 📊 Performance & Monitoring

#### GET /performance/summary
Resumo de performance dos últimos 30 dias

**Response:**
```json
{
  "period": "30 days",
  "total_trades": 67,
  "winning_trades": 45,
  "win_rate": 67.16,
  "total_pnl": 245.80,
  "avg_pnl_per_trade": 3.67,
  "best_trade": 25.50,
  "worst_trade": -10.00
}
```

#### GET /system/status
Status completo do sistema

**Response:**
```json
{
  "api_status": "online",
  "ml_engine": {
    "trained": true,
    "models": {
      "direction_model": true,
      "confidence_model": true,
      "timeframe_model": true,
      "risk_model": true
    }
  },
  "advanced_features": {
    "available": true,
    "anomaly_detection": true,
    "pattern_recognition": true,
    "backtesting": true
  },
  "version": "2.0.0-advanced"
}
```

## Error Codes

- `200` - Success
- `400` - Bad Request (dados inválidos)
- `500` - Internal Server Error
- `501` - Not Implemented (funcionalidade não disponível)

## Rate Limits
- **Geral**: 100 requests/minuto
- **Backtesting**: 5 requests/minuto
- **Auto-decision**: 30 requests/minuto

## Status Codes
- `online` - Sistema funcionando
- `training` - Modelos sendo treinados
- `offline` - Sistema indisponível
- `warning` - Funcionando com limitações
'''
        
        with open("trading-bot-ml/docs/API.md", "w") as f:
            f.write(api_docs)
        
        print("  ✅ Documentação criada")
    
    def create_test_files(self):
        """Cria arquivos de teste"""
        print("🧪 Criando testes...")
        
        test_content = '''# test_api.py - Testes automatizados

import pytest
import requests
import json
from datetime import datetime

class TestTradingBotAPI:
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        
    def test_health_check(self):
        """Teste básico de saúde da API"""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
    def test_trading_signal(self):
        """Teste de sinal de trading"""
        data = {
            "symbol": "R_50",
            "current_price": 1000,
            "account_balance": 1000,
            "recent_trades": [],
            "market_data": []
        }
        
        response = requests.post(f"{self.base_url}/signal", json=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "direction" in result
        assert result["direction"] in ["CALL", "PUT"]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 100
        
    def test_auto_decision(self):
        """Teste de decisão automática"""
        data = {
            "symbol": "R_50",
            "current_price": 1000,
            "account_balance": 1000,
            "recent_trades": [],
            "market_data": []
        }
        
        response = requests.post(f"{self.base_url}/auto-decision", json=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "action" in result
        assert result["action"] in ["trade", "hold", "wait"]
        
    def test_quick_backtest(self):
        """Teste de backtesting rápido"""
        response = requests.get(f"{self.base_url}/backtest/quick")
        assert response.status_code == 200
        
        result = response.json()
        assert "report" in result
        assert "summary" in result["report"]

if __name__ == "__main__":
    # Executar testes
    tester = TestTradingBotAPI()
    
    print("🧪 Executando testes...")
    
    try:
        tester.test_health_check()
        print("  ✅ Health check OK")
        
        tester.test_trading_signal()
        print("  ✅ Trading signal OK")
        
        tester.test_auto_decision()
        print("  ✅ Auto decision OK")
        
        tester.test_quick_backtest()
        print("  ✅ Quick backtest OK")
        
        print("\\n🎉 Todos os testes passaram!")
        
    except Exception as e:
        print(f"\\n❌ Teste falhou: {e}")
'''
        
        with open("trading-bot-ml/tests/test_api.py", "w") as f:
            f.write(test_content)
        
        print("  ✅ Testes criados")
    
    def deploy_to_render(self):
        """Deploy automatizado no Render"""
        print("☁️ Iniciando deploy no Render...")
        
        if not self.render_api_key:
            print("❌ RENDER_API_KEY não configurada")
            return False
        
        # Configuração do serviço
        service_config = {
            "name": "trading-bot-ml-api",
            "type": "web_service",
            "repo": self.github_repo,
            "branch": "main",
            "runtime": "python",
            "buildCommand": "pip install -r requirements.txt",
            "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
            "envVars": [
                {"key": "ENVIRONMENT", "value": "production"},
                {"key": "LOG_LEVEL", "value": "INFO"}
            ],
            "region": "oregon",
            "plan": "starter"  # Plano gratuito
        }
        
        # Fazer requisição para criar serviço
        headers = {
            "Authorization": f"Bearer {self.render_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/services",
                headers=headers,
                json=service_config
            )
            
            if response.status_code == 201:
                service_data = response.json()
                service_url = service_data.get("serviceUrl")
                print(f"✅ Deploy realizado com sucesso!")
                print(f"🌐 URL: {service_url}")
                return True
            else:
                print(f"❌ Erro no deploy: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ Erro no deploy: {e}")
            return False
    
    def verify_deployment(self, url):
        """Verifica se o deployment está funcionando"""
        print(f"🔍 Verificando deployment em {url}...")
        
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{url}/health", timeout=10)
                if response.status_code == 200:
                    print("✅ API online e funcionando!")
                    
                    # Testar endpoints principais
                    self.test_deployed_api(url)
                    return True
                    
            except requests.exceptions.RequestException:
                pass
            
            print(f"  ⏳ Tentativa {attempt + 1}/{max_attempts}...")
            time.sleep(10)
        
        print("❌ API não respondeu após 5 minutos")
        return False
    
    def test_deployed_api(self, url):
        """Testa API deployada"""
        print("🧪 Testando endpoints da API deployada...")
        
        tests = [
            ("GET", "/", "Info básica"),
            ("GET", "/health", "Health check"),
            ("GET", "/model-status", "Status dos modelos"),
            ("GET", "/features", "Lista de features")
        ]
        
        for method, endpoint, description in tests:
            try:
                if method == "GET":
                    response = requests.get(f"{url}{endpoint}", timeout=30)
                
                if response.status_code == 200:
                    print(f"  ✅ {description}")
                else:
                    print(f"  ⚠️ {description} - Status: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ {description} - Erro: {e}")
    
    def run_full_deployment(self):
        """Executa deployment completo"""
        print("🚀 INICIANDO DEPLOYMENT COMPLETO DO TRADING BOT ML")
        print("=" * 60)
        
        # 1. Criar estrutura do projeto
        self.create_project_structure()
        
        # 2. Criar testes
        self.create_test_files()
        
        print("\n📁 Estrutura do projeto criada com sucesso!")
        print("\n📋 PRÓXIMOS PASSOS:")
        print("1. Faça upload dos arquivos para seu repositório GitHub")
        print("2. Acesse render.com e conecte seu repositório")
        print("3. Configure as variáveis de ambiente")
        print("4. Execute o deploy")
        print("\n🌐 Sua API estará disponível em: https://seu-app.onrender.com")
        
        return True

# Script principal
if __name__ == "__main__":
    deployer = TradingBotDeployer()
    
    print("🤖 TRADING BOT ML - DEPLOY AUTOMATIZADO")
    print("=" * 50)
    
    # Executar deployment completo
    success = deployer.run_full_deployment()
    
    if success:
        print("\n🎉 DEPLOYMENT COMPLETO!")
        print("📚 Consulte README.md para instruções detalhadas")
        print("📡 Consulte docs/API.md para documentação da API")
    else:
        print("\n❌ Erro no deployment")

# ==============================================
# SCRIPT DE MONITORAMENTO
# ==============================================

class APIMonitor:
    """Monitor de saúde da API em produção"""
    
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')
        
    def check_health(self):
        """Verifica saúde da API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "ml_engine": data.get("ml_engine", "unknown")
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_ml_models(self):
        """Verifica status dos modelos ML"""
        try:
            response = requests.get(f"{self.api_url}/model-status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "Não foi possível verificar modelos"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def test_trading_signal(self):
        """Testa geração de sinal"""
        try:
            data = {
                "symbol": "R_50",
                "current_price": 1000,
                "account_balance": 1000,
                "recent_trades": [],
                "market_data": []
            }
            
            response = requests.post(f"{self.api_url}/signal", json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "working",
                    "direction": result.get("direction"),
                    "confidence": result.get("confidence")
                }
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate_report(self):
        """Gera relatório completo de status"""
        print(f"📊 RELATÓRIO DE STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Health check
        health = self.check_health()
        print(f"🏥 Health: {health['status'].upper()}")
        if health['status'] == 'healthy':
            print(f"   ⏱️ Response Time: {health['response_time']:.2f}s")
        else:
            print(f"   ❌ Error: {health.get('error', 'Unknown')}")
        
        # ML Models
        models = self.check_ml_models()
        print(f"\\n🧠 ML Models: {'TRAINED' if models.get('is_trained') else 'TRAINING'}")
        
        # Trading Signal
        signal = self.test_trading_signal()
        print(f"\\n🎯 Trading Signal: {signal['status'].upper()}")
        if signal['status'] == 'working':
            print(f"   📈 Last Signal: {signal['direction']} ({signal['confidence']:.1f}%)")
        
        print("\\n" + "=" * 60)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "models": models,
            "signal": signal
        }

# Exemplo de uso do monitor
def monitor_api(api_url):
    """Monitora API continuamente"""
    monitor = APIMonitor(api_url)
    
    while True:
        try:
            report = monitor.generate_report()
            
            # Se tudo estiver funcionando, aguardar 5 minutos
            # Se houver problemas, aguardar 1 minuto
            if (report['health']['status'] == 'healthy' and 
                report['signal']['status'] == 'working'):
                time.sleep(300)  # 5 minutos
            else:
                time.sleep(60)   # 1 minuto
                
        except KeyboardInterrupt:
            print("\\n👋 Monitoramento interrompido")
            break
        except Exception as e:
            print(f"❌ Erro no monitoramento: {e}")
            time.sleep(60)

# Para usar: python deploy_automation.py monitor https://sua-api.onrender.com
import sys
if len(sys.argv) > 1 and sys.argv[1] == "monitor":
    if len(sys.argv) > 2:
        monitor_api(sys.argv[2])
    else:
        print("Uso: python deploy_automation.py monitor <URL_DA_API>")
