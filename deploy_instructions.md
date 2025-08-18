# 🧠 Trading Bot com Machine Learning Real

## 📋 O que foi implementado

Este sistema implementa **Machine Learning REAL** para trading, não apenas simulação. Inclui:

### 🔧 Backend (Python + FastAPI)
- **Scikit-learn** para modelos ML reais
- **Random Forest, Gradient Boosting, Logistic Regression**
- **Banco SQLite** para armazenar dados de treino
- **Análise de padrões** baseada em dados históricos
- **API REST** completa para predições e análises
- **Treinamento automático** dos modelos

### 🎯 Frontend (HTML + JavaScript)
- **Interface moderna** com conexão à API ML real
- **Predições ML em tempo real** para trades
- **Estatísticas e métricas** dos modelos
- **Análise de padrões** identificados
- **Controles para treinar** modelos manualmente

## 🚀 Como fazer Deploy na Render

### Passo 1: Preparar os arquivos

1. Crie uma pasta no seu computador (ex: `trading-ml-bot`)
2. Coloque estes arquivos na pasta:
   - `main.py` (código do backend Python)
   - `requirements.txt` (dependências)
   - `Dockerfile` (opcional)
   - `start.sh` (script de inicialização)

### Passo 2: Fazer Deploy na Render

1. **Acesse**: https://render.com/
2. **Crie uma conta** (gratuita)
3. **Conecte seu GitHub**: 
   - Faça upload dos arquivos para um repositório GitHub
   - Ou use o zip upload da Render

4. **Criar Web Service**:
   - Clique em "New" > "Web Service"
   - Conecte seu repositório
   - Configure:
     ```
     Name: trading-ml-api
     Environment: Python 3
     Build Command: pip install -r requirements.txt
     Start Command: chmod +x start.sh && ./start.sh
     ```

5. **Variáveis de Ambiente** (opcional):
   ```
   PORT=8000
   PYTHON_VERSION=3.11.0
   ```

6. **Deploy**: Clique em "Create Web Service"

### Passo 3: Configurar o Frontend

1. Após o deploy, você terá uma URL como:
   ```
   https://trading-ml-api-xxx.onrender.com
   ```

2. **Abra o arquivo HTML** do frontend
3. **Configure a URL da API** no campo "URL da API ML"
4. **Teste a conexão** com o botão "Testar Conexão"

### Passo 4: Usar o Sistema

1. **Obtenha seu token Deriv**:
   - Acesse: https://app.deriv.com/account/api-token
   - Crie um token com permissões de trading
   - Use apenas em conta DEMO inicialmente

2. **Conecte os sistemas**:
   - Cole a URL da API ML
   - Cole seu token Deriv
   - Clique em "Conectar Deriv + ML Real"

3. **Use o ML**:
   - O sistema começará coletando dados dos seus trades
   - Após ~50 trades, os modelos ML serão treinados automaticamente
   - Use os botões "CALL (ML)" e "PUT (ML)" para trades com predição ML

## 📊 Como funciona o ML

### Coleta de Dados
O sistema coleta automaticamente:
- **Dados do trade**: símbolo, direção, valor, duração
- **Contexto de mercado**: preço atual, volatilidade, tendência
- **Resultados**: vitória/derrota, lucro/prejuízo
- **Padrões temporais**: horário do trade, sequências de resultados

### Modelos ML
Treina 3 modelos simultaneamente:
- **Random Forest**: Para padrões complexos
- **Gradient Boosting**: Para refinamento de predições
- **Logistic Regression**: Para interpretabilidade

### Predições
Para cada trade, o ML analisa:
- **Probabilidade de vitória** baseada no histórico
- **Padrões similares** no passado
- **Contexto atual** do mercado
- **Recomendação**: Executar, Evitar ou Neutro

### Aprendizado Contínuo
- **Retreina automaticamente** a cada 50 novos trades
- **Identifica padrões** de sucesso e erro
- **Adapta estratégias** baseado nos resultados
- **Melhora continuamente** a precisão

## 🔧 Estrutura dos Arquivos

```
trading-ml-bot/
├── main.py              # Backend FastAPI + ML
├── requirements.txt     # Dependências Python
├── Dockerfile          # Container (opcional)
├── start.sh            # Script de inicialização
├── README.md           # Este arquivo
└── frontend.html       # Interface web
```

## 📈 Endpoints da API

### Principais endpoints:
- `GET /health` - Status da API e modelos
- `POST /trade/save` - Salvar dados de trade
- `POST /ml/predict` - Obter predição ML
- `POST /ml/analyze` - Análise de mercado
- `POST /ml/train` - Forçar retreinamento
- `GET /ml/stats` - Estatísticas dos modelos
- `GET /ml/patterns` - Padrões identificados

## ⚠️ Importante

### Segurança
- **Use apenas conta DEMO** inicialmente
- **Nunca compartilhe** seu token da API
- **Teste bem** antes de usar dinheiro real

### Performance
- O **plano gratuito** da Render pode ter limitações
- Para uso intensivo, considere o **plano pago**
- **Dados são persistidos** no SQLite local

### Limitações do Plano Gratuito
- **Sleep após 15min** de inatividade
- **750 horas/mês** de uso
- **RAM limitada** para modelos grandes

## 🆘 Troubleshooting

### API não conecta
1. Verifique se a URL está correta
2. Teste endpoint `/health` no browser
3. Confira logs do deploy na Render

### Modelos não treinam
1. Certifique-se de ter dados suficientes (50+ trades)
2. Verifique logs no console do navegador
3. Force retreinamento com botão "Treinar Modelos"

### Predições imprecisas
1. **Normal no início** - precisa de mais dados
2. **Melhora com o tempo** - aprendizado contínuo
3. **Revise estratégias** baseado nas estatísticas

## 🎯 Próximos Passos

1. **Colete dados** fazendo trades manuais inicialmente
2. **Monitore estatísticas** de precisão dos modelos
3. **Ajuste parâmetros** conforme necessário
4. **Implemente melhorias** baseado nos padrões encontrados

## 📞 Suporte

- **Logs da Render**: Dashboard > Service > Logs
- **Console do navegador**: F12 > Console
- **Testes manuais**: Use Postman ou curl para testar endpoints

---

## ✅ Checklist de Deploy

- [ ] Arquivos criados e organizados
- [ ] Deploy feito na Render
- [ ] URL da API funcionando
- [ ] Frontend configurado com URL correta
- [ ] Token Deriv obtido (conta demo)
- [ ] Conexão Deriv + ML testada
- [ ] Primeiro trade executado
- [ ] Dados sendo salvos no ML
- [ ] Estatísticas sendo atualizadas

**🎉 Parabéns! Seu sistema de ML real está funcionando!**