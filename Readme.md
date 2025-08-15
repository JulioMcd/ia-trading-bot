# 🤖 ML Trading Bot - IA para Trading Automático

Sistema de Machine Learning avançado para trading automático em derivados, com análise em tempo real e Martingale Inteligente.

## 🚀 Funcionalidades

### 🧠 Inteligência Artificial
- **Análise de Mercado**: ML analisa padrões e tendências
- **Sinais Automáticos**: Gera sinais CALL/PUT com alta precisão
- **Aprendizado Contínuo**: Modelo melhora com cada trade
- **Avaliação de Risco**: Sistema inteligente de gerenciamento

### 📊 Análise Técnica
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Análise de Volatilidade
- Padrões de Preço
- Volume Analysis

### 🎯 Martingale Inteligente
- Sistema anti-loop
- Cooling periods após perdas
- Análise obrigatória da IA
- Gerenciamento automático de risco

## 📡 API Endpoints

### `GET /`
Health check do sistema
```json
{
  "status": "online",
  "service": "ML Trading Bot",
  "model_trained": true
}
```

### `POST /analyze`
Análise completa do mercado
```json
{
  "symbol": "R_50",
  "currentPrice": 1234.56,
  "marketCondition": "neutral"
}
```

### `POST /signal`
Gerar sinal de trading
```json
{
  "direction": "CALL",
  "confidence": 87.3,
  "reasoning": "ML Model prediction",
  "timeframe": "5m"
}
```

### `POST /risk`
Avaliação de risco
```json
{
  "level": "medium",
  "message": "Risco moderado",
  "recommendation": "Operar com cautela"
}
```

### `POST /feedback`
Feedback de resultados (para aprendizado)
```json
{
  "direction": "CALL",
  "result": "win",
  "pnl": 1.85
}
```

## 🔧 Instalação

### 1. Deploy no Render

1. Faça fork deste repositório
2. Conecte sua conta GitHub ao Render
3. Crie novo Web Service
4. Selecione este repositório
5. Configure:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`

### 2. Configuração Local (Opcional)

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ml-trading-bot.git
cd ml-trading-bot

# Instale dependências
pip install -r requirements.txt

# Execute localmente
python app.py
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.9+**
- **Flask** - API Web Framework
- **Scikit-learn** - Machine Learning
- **NumPy & Pandas** - Processamento de dados
- **Random Forest** - Algoritmo principal
- **StandardScaler** - Normalização de dados

## 📈 Como Funciona

### 1. Coleta de Dados
- Recebe dados em tempo real do trading bot
- Armazena histórico de preços e trades
- Calcula indicadores técnicos

### 2. Processamento ML
- Extrai features técnicas dos dados
- Normaliza dados usando StandardScaler
- Aplica modelo Random Forest treinado

### 3. Geração de Sinais
- Prediz direção do mercado (CALL/PUT)
- Calcula nível de confiança
- Considera risco e Martingale

### 4. Aprendizado Contínuo
- Recebe feedback dos trades
- Re-treina modelo periodicamente
- Melhora precisão ao longo do tempo

## ⚙️ Configurações

### Variáveis de Ambiente
```
PYTHON_VERSION=3.9.16
FLASK_ENV=production
MODEL_VERSION=1.0
API_TIMEOUT=30
```

### Parâmetros do Modelo
- **Confidence Threshold**: 70%
- **Min Data Points**: 10
- **Max History**: 30 dias
- **Model**: Random Forest (100 trees)

## 🔒 Segurança

- Validação de entrada em todos endpoints
- Rate limiting automático
- Logs detalhados de todas operações
- Tratamento robusto de erros

## 📊 Métricas de Performance

O sistema monitora:
- Taxa de acerto dos sinais
- Tempo de resposta da API
- Precisão do modelo ML
- Volume de dados processados

## 🚨 Avisos Importantes

⚠️ **RISCO**: Trading envolve risco de perda financeira
⚠️ **DEMO**: Teste sempre em conta demo primeiro
⚠️ **RESPONSABILIDADE**: Use por sua própria conta e risco

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs no Render
2. Teste endpoints individualmente
3. Confirme configurações de rede

## 📄 Licença

Este projeto é de uso educacional. Não nos responsabilizamos por perdas financeiras.

---

**Desenvolvido com ❤️ para traders que querem automatizar suas operações com IA**
