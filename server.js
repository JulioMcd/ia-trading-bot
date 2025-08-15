const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Rota principal - serve o HTML do trading bot
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Rotas da API para IA (simuladas para teste)
app.post('/api/analyze', (req, res) => {
    // Simula análise da IA
    setTimeout(() => {
        res.json({
            message: `Análise do ${req.body.symbol || 'mercado'}: Volatilidade ${(Math.random() * 100).toFixed(1)}%`,
            volatility: Math.random() * 100,
            trend: Math.random() > 0.5 ? 'bullish' : 'bearish',
            confidence: 70 + Math.random() * 25,
            timestamp: new Date().toISOString()
        });
    }, 1000 + Math.random() * 2000);
});

app.post('/api/signal', (req, res) => {
    // Simula sinal de trading da IA
    setTimeout(() => {
        const confidence = 70 + Math.random() * 25;
        const direction = Math.random() > 0.5 ? 'CALL' : 'PUT';
        
        res.json({
            direction: direction,
            confidence: confidence,
            reasoning: 'Baseado em padrões de mercado e análise técnica',
            timeframe: '5m',
            entry_price: req.body.currentPrice || 1000,
            timestamp: new Date().toISOString()
        });
    }, 1000 + Math.random() * 2000);
});

app.post('/api/risk', (req, res) => {
    // Simula avaliação de risco da IA
    setTimeout(() => {
        const martingaleLevel = req.body.martingaleLevel || 0;
        let level = 'medium';
        let message = 'Risco normal';
        let recommendation = 'Continuar operando';
        
        if (martingaleLevel > 4) {
            level = 'high';
            message = `Risco alto - Martingale nível ${martingaleLevel}`;
            recommendation = 'Considerar pausa ou redução';
        } else if (martingaleLevel > 2) {
            level = 'medium';
            message = `Risco moderado - Martingale nível ${martingaleLevel}`;
            recommendation = 'Operar com cautela';
        }
        
        res.json({
            level: level,
            message: message,
            score: Math.random() * 100,
            recommendation: recommendation,
            timestamp: new Date().toISOString()
        });
    }, 1000 + Math.random() * 2000);
});

// Rota de health check
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        timestamp: new Date().toISOString(),
        service: 'Trading Bot IA'
    });
});

app.listen(PORT, () => {
    console.log(`🚀 Trading Bot IA servidor rodando na porta ${PORT}`);
    console.log(`🌐 Acesse: http://localhost:${PORT}`);
});
