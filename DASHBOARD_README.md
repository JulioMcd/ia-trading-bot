# 🚀 Dashboard Quantum Trading Bot

Dashboard HTML interativo para sua API de Trading com IA Quantum.

## 📋 Funcionalidades

- ✅ **Login Simulado**: Crie uma conta com nome e saldo inicial
- 💰 **Visualização de Saldo**: Acompanhe seu saldo em tempo real
- 📊 **Previsões Quantum**: Obtenha análises de IA para suas operações
- 🎯 **Recomendações Inteligentes**: BUY, SELL, HOLD com níveis de confiança
- 📈 **Histórico Completo**: Todas as operações registradas
- 📉 **Estatísticas**: Win rate, lucro total, perdas consecutivas
- ⚛️ **Métricas Quantum**: Vantagem quantum, score, dimensionalidade

## 🚀 Como Usar

### 1. Iniciar a API

Certifique-se de que sua API está rodando na porta 5001:

```bash
# Opção 1: Flask desenvolvimento
python app.py

# Opção 2: Gunicorn (produção)
gunicorn -c gunicorn.conf.py app:app
```

### 2. Abrir o Dashboard

Abra o arquivo `trading_dashboard.html` no seu navegador:

```bash
# Linux/Mac
open trading_dashboard.html

# Windows
start trading_dashboard.html

# Ou use um servidor HTTP local (recomendado)
python -m http.server 8000
# Depois acesse: http://localhost:8000/trading_dashboard.html
```

### 3. Criar Conta

1. Digite seu nome de trader
2. Defina o saldo inicial (ex: 10000)
3. Clique em "Iniciar Sessão"

### 4. Fazer uma Operação

1. Preencha os dados da operação:
   - **Valor da Aposta**: Quanto você quer investir
   - **Preço de Entrada**: Preço atual do ativo (ex: 50000 para Bitcoin)
   - **Preço de Saída**: Preço estimado de saída (ex: 51000)
   - **Duração**: Tempo da operação em segundos (ex: 60)
   - **Volatilidade**: Entre 0 e 1 (ex: 0.02 = 2%)
   - **Nível Martingale**: Nível da progressão (1 = primeira tentativa)

2. Clique em "🔮 Obter Previsão Quantum"

3. A IA irá analisar e retornar:
   - Probabilidades de vitória (Quantum, Tradicional, Ensemble)
   - PnL previsto
   - Recomendação (BUY/SELL/HOLD)
   - Nível de confiança
   - Métricas quantum

4. Execute a operação:
   - **Simular Vitória**: Adiciona lucro (85% da aposta)
   - **Simular Perda**: Subtrai a aposta do saldo

### 5. Acompanhar Resultados

- Veja seu saldo atualizado em tempo real
- Consulte estatísticas: Win Rate, Lucro Total, etc.
- Revise o histórico completo de operações

## 🎨 Interface

### Dashboard Principal

- **Header**: Status da API (online/offline)
- **Minha Conta**: Saldo e estatísticas
- **Nova Operação**: Formulário de trading
- **Previsão Quantum**: Resultados da IA
- **Histórico**: Lista de todas as operações

### Códigos de Cores

- 🟢 **Verde**: Vitórias, BUY, STRONG_BUY
- 🔴 **Vermelho**: Perdas, SELL, STRONG_SELL
- 🟡 **Amarelo**: HOLD
- 🔵 **Roxo**: Elementos quantum

## 📊 Entendendo as Previsões

### Probabilidades

- **Quantum**: Modelo com feature maps quânticos
- **Tradicional**: Modelo clássico de ML
- **Ensemble**: Combinação de múltiplos modelos

### Recomendações

- **STRONG_BUY** 🚀: Alta confiança, compre!
- **BUY** 📈: Confiança moderada, compre
- **HOLD** ⏸️: Aguarde melhor momento
- **SELL** 📉: Confiança moderada, venda
- **STRONG_SELL** ⚠️: Alta confiança, venda!

### Alertas de Risco

- ⚠️ **Deriva de mercado detectada**: Padrões mudaram
- ⚠️ **Baixa confiança**: Modelo incerto
- ⚠️ **PnL negativo esperado**: Operação arriscada

## 💾 Persistência

Os dados são salvos automaticamente no **LocalStorage** do navegador:
- Saldo atual
- Histórico de operações
- Estatísticas
- Sessão ativa

Para resetar, use o botão "Encerrar Sessão" ou limpe o cache do navegador.

## 🔧 Configuração Avançada

### Alterar URL da API

Edite no arquivo `trading_dashboard.html`:

```javascript
const API_URL = 'http://localhost:5001'; // Altere aqui
```

### Personalizar Payout

Por padrão, vitórias pagam 85% da aposta. Para alterar:

```javascript
// Na função executeTrade()
pnl = stake * 0.85; // Altere 0.85 para o valor desejado
```

## 📱 Responsividade

O dashboard é totalmente responsivo e funciona em:
- 💻 Desktop
- 📱 Tablet
- 📱 Smartphone

## ⚠️ Avisos Importantes

1. **Simulação**: Este dashboard simula operações. Não executa trades reais.
2. **Segurança**: A API não tem autenticação. Use apenas em ambiente local/confiável.
3. **Dados**: Os dados são salvos apenas no navegador (localStorage).
4. **IA**: As previsões são baseadas em modelos de ML, não são garantias.

## 🐛 Problemas Comuns

### API Offline

```
Problema: Indicador vermelho "API Offline"
Solução: Verifique se app.py está rodando na porta 5001
```

### Erro CORS

```
Problema: Erro de CORS no console
Solução: CORS já está configurado. Verifique se a API está rodando.
```

### Dados Não Salvam

```
Problema: Dados são perdidos ao recarregar
Solução: Verifique se localStorage está habilitado no navegador
```

## 📚 Documentação da API

Para mais detalhes sobre os endpoints da API, consulte:
- `/` - Status da API
- `/predict-quantum` - Obter previsões
- `/quantum-status` - Status do sistema quantum
- `/train-quantum` - Treinar modelo

## 🎯 Próximos Passos

1. Adicionar gráficos de performance
2. Implementar estratégias automáticas
3. Sistema de alertas
4. Integração com exchanges reais
5. Autenticação real

## 📄 Licença

Use livremente para seus projetos de trading!

---

**Desenvolvido com ❤️ para Quantum Trading**
