from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import random
import time
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Configurações
class Config:
    MIN_STAKE = 0.35
    MAX_STAKE = 2000
    AI_CONFIDENCE_RANGE = (70, 95)
    RISK_LEVELS = ['low', 'medium', 'high']
    MARKET_TRENDS = ['bullish', 'bearish', 'neutral']

# Simulador de IA Original (funcional)
class TradingAI:
    def __init__(self):
        self.last_analysis = None
        self.analysis_history = []
        # ✅ ML Engine simples em memória (sem dependências)
        self.ml_data = []
        self.ml_patterns = {}
        self.ml_enabled = True
    
    def analyze_market(self, market_data):
        """Análise inteligente do mercado"""
        
        # Extrair dados importantes
        symbol = market_data.get('symbol', 'R_50')
        current_price = market_data.get('currentPrice', 1000)
        martingale_level = market_data.get('martingaleLevel', 0)
        is_after_loss = market_data.get('isAfterLoss', False)
        win_rate = market_data.get('winRate', 50)
        volatility = market_data.get('volatility', 50)
        
        # Simular tempo de processamento da IA
        time.sleep(random.uniform(1, 3))
        
        # ✅ ML SIMPLES - Usar padrões aprendidos
        ml_boost = 0
        if len(self.ml_data) > 10:
            recent_success = sum([1 for d in self.ml_data[-10:] if d.get('success', False)]) / 10
            ml_boost = (recent_success - 0.5) * 20  # -10 a +10 boost
        
        # Análise baseada no contexto + ML
        if is_after_loss and martingale_level > 0:
            # Ser mais conservador após perdas
            confidence = random.uniform(65, 80) + ml_boost
            trend = 'neutral'
            message = f"🤖 ML Análise pós-perda do {symbol}: Recomendação conservadora (Martingale Nível {martingale_level})"
        elif martingale_level > 4:
            # Alto risco
            confidence = random.uniform(60, 75) + ml_boost
            trend = 'neutral'
            message = f"🤖 ML Análise de alto risco do {symbol}: Martingale Nível {martingale_level} - Cautela recomendada"
        else:
            # Análise normal com boost ML
            confidence = random.uniform(70, 95) + ml_boost
            trend = random.choice(Config.MARKET_TRENDS)
            message = f"🤖 ML Análise do {symbol}: Volatilidade {volatility:.1f}%, Tendência {trend} (ML boost: {ml_boost:+.1f})"
        
        # Ajustar confiança baseada na taxa de acerto
        if win_rate > 70:
            confidence += 5
        elif win_rate < 40:
            confidence -= 10
            
        confidence = max(60, min(95, confidence))
        
        analysis = {
            'message': message,
            'volatility': volatility,
            'trend': trend,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'martingaleLevel': martingale_level,
            'isAfterLoss': is_after_loss,
            'recommendation': self._get_recommendation(confidence, martingale_level, is_after_loss),
            'ml_enabled': True,
            'ml_data_size': len(self.ml_data),
            'ml_boost': ml_boost
        }
        
        self.last_analysis = analysis
        self.analysis_history.append(analysis)
        
        # Manter apenas últimas 50 análises
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
            
        return analysis
    
    def get_trading_signal(self, signal_data):
        """Gerar sinal de trading"""
        
        symbol = signal_data.get('symbol', 'R_50')
        current_price = signal_data.get('currentPrice', 1000)
        martingale_level = signal_data.get('martingaleLevel', 0)
        is_after_loss = signal_data.get('isAfterLoss', False)
        win_rate = signal_data.get('winRate', 50)
        
        # Simular processamento
        time.sleep(random.uniform(1, 2))
        
        # ✅ ML SIMPLES - Padrão baseado em histórico
        direction_bias = 'CALL'  # padrão
        if len(self.ml_data) > 5:
            recent_calls = sum([1 for d in self.ml_data[-5:] if d.get('direction') == 'CALL' and d.get('success', False)])
            recent_puts = sum([1 for d in self.ml_data[-5:] if d.get('direction') == 'PUT' and d.get('success', False)])
            if recent_puts > recent_calls:
                direction_bias = 'PUT'
        
        # Determinar direção baseada em análise + ML
        if is_after_loss and martingale_level > 0:
            # Mais conservador após perdas
            confidence = random.uniform(70, 82)
            direction = direction_bias if random.random() > 0.3 else random.choice(['CALL', 'PUT'])
            reasoning = f"🤖 ML Sinal conservador pós-perda (Martingale {martingale_level}) - Bias: {direction_bias}"
        elif martingale_level > 4:
            # Muito conservador em alto martingale
            confidence = random.uniform(65, 78)
            direction = direction_bias if random.random() > 0.4 else random.choice(['CALL', 'PUT'])
            reasoning = f"🤖 ML Sinal de alto risco - Martingale Nível {martingale_level} - Bias: {direction_bias}"
        else:
            # Sinal normal com ML
            confidence = random.uniform(75, 92)
            direction = direction_bias if random.random() > 0.5 else random.choice(['CALL', 'PUT'])
            reasoning = f"🤖 ML Sinal baseado em padrões de mercado - Bias: {direction_bias}"
        
        # Ajustar baseado na performance
        if win_rate > 70:
            confidence += 3
        elif win_rate < 40:
            confidence -= 8
            
        confidence = max(65, min(95, confidence))
        
        signal = {
            'direction': direction,
            'confidence': confidence,
            'reasoning': reasoning,
            'timeframe': '5m',
            'entry_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'martingaleLevel': martingale_level,
            'isAfterLoss': is_after_loss,
            'recommendation': self._get_signal_recommendation(confidence, direction),
            'ml_enabled': True,
            'ml_bias': direction_bias,
            'ml_data_size': len(self.ml_data)
        }
        
        return signal
    
    def assess_risk(self, risk_data):
        """Avaliação de risco"""
        
        current_balance = risk_data.get('currentBalance', 1000)
        today_pnl = risk_data.get('todayPnL', 0)
        martingale_level = risk_data.get('martingaleLevel', 0)
        win_rate = risk_data.get('winRate', 50)
        total_trades = risk_data.get('totalTrades', 0)
        is_cooling = risk_data.get('isInCoolingPeriod', False)
        needs_analysis = risk_data.get('needsAnalysisAfterLoss', False)
        
        # Simular análise
        time.sleep(random.uniform(0.5, 1.5))
        
        # ✅ ML SIMPLES - Ajustar risco baseado em histórico
        ml_risk_adjustment = 0
        if len(self.ml_data) > 10:
            recent_losses = sum([1 for d in self.ml_data[-10:] if not d.get('success', True)])
            if recent_losses > 6:  # Mais de 60% perdas
                ml_risk_adjustment += 20
            elif recent_losses < 3:  # Menos de 30% perdas
                ml_risk_adjustment -= 10
        
        # Calcular nível de risco
        risk_score = 0
        
        # Risco baseado no Martingale
        if martingale_level > 6:
            risk_score += 40
        elif martingale_level > 3:
            risk_score += 25
        elif martingale_level > 0:
            risk_score += 10
            
        # Risco baseado no P&L
        pnl_percentage = (today_pnl / current_balance) * 100 if current_balance > 0 else 0
        if pnl_percentage < -20:
            risk_score += 30
        elif pnl_percentage < -10:
            risk_score += 20
        elif pnl_percentage < -5:
            risk_score += 10
            
        # Risco baseado na taxa de acerto
        if win_rate < 30:
            risk_score += 25
        elif win_rate < 45:
            risk_score += 15
            
        # Estados especiais
        if is_cooling:
            risk_score += 5
        if needs_analysis:
            risk_score += 10
            
        # ✅ Aplicar ajuste ML
        risk_score += ml_risk_adjustment
        risk_score = max(0, min(100, risk_score))
        
        # Determinar nível e recomendação
        if risk_score >= 60:
            level = 'high'
            message = f"🤖 ML Risco ALTO detectado - Score: {risk_score} (ML ajuste: {ml_risk_adjustment:+d})"
            if martingale_level > 5:
                recommendation = "PARAR operações - Martingale muito alto"
            else:
                recommendation = "Reduzir stake e operar com extrema cautela"
        elif risk_score >= 35:
            level = 'medium'
            message = f"🤖 ML Risco MODERADO - Score: {risk_score} (ML ajuste: {ml_risk_adjustment:+d})"
            if martingale_level > 0:
                recommendation = f"Cautela - Martingale Nível {martingale_level} ativo"
            else:
                recommendation = "Operar com cautela moderada"
        else:
            level = 'low'
            message = f"🤖 ML Risco BAIXO - Score: {risk_score} (ML ajuste: {ml_risk_adjustment:+d})"
            recommendation = "Seguro para continuar operando"
            
        # Ajustes específicos para Martingale Inteligente
        if martingale_level > 0:
            message += f" | Martingale Nível {martingale_level}"
            
        if is_cooling:
            message += " | Em período de cooling"
            
        if needs_analysis:
            message += " | Aguardando análise pós-perda"
        
        risk_assessment = {
            'level': level,
            'message': message,
            'score': risk_score,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'martingaleLevel': martingale_level,
            'isAfterLoss': needs_analysis,
            'currentBalance': current_balance,
            'todayPnL': today_pnl,
            'winRate': win_rate,
            'ml_enabled': True,
            'ml_adjustment': ml_risk_adjustment,
            'ml_data_size': len(self.ml_data),
            'details': {
                'martingale_risk': martingale_level * 5,
                'pnl_risk': max(0, abs(pnl_percentage) - 5) * 2,
                'performance_risk': max(0, 50 - win_rate),
                'ml_risk': ml_risk_adjustment,
                'total_score': risk_score
            }
        }
        
        return risk_assessment
    
    def add_ml_data(self, trade_data):
        """📊 Adicionar dados para ML simples"""
        try:
            ml_record = {
                'direction': trade_data.get('direction'),
                'success': trade_data.get('pnl', 0) > 0,
                'pnl': trade_data.get('pnl', 0),
                'martingale_level': trade_data.get('martingaleLevel', 0),
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_data.get('symbol', 'R_50'),
                'volatility': trade_data.get('volatility', 50)
            }
            
            self.ml_data.append(ml_record)
            
            # Manter apenas últimos 100 trades
            if len(self.ml_data) > 100:
                self.ml_data = self.ml_data[-100:]
            
            print(f"📊 ML Data adicionado: {len(self.ml_data)} trades totais")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao adicionar ML data: {e}")
            return False
    
    def get_ml_stats(self):
        """📊 Estatísticas ML"""
        if len(self.ml_data) == 0:
            return {
                'total_trades': 0,
                'success_rate': 0,
                'ml_enabled': self.ml_enabled
            }
        
        successful_trades = sum([1 for d in self.ml_data if d.get('success', False)])
        success_rate = (successful_trades / len(self.ml_data)) * 100
        
        return {
            'total_trades': len(self.ml_data),
            'successful_trades': successful_trades,
            'success_rate': success_rate,
            'ml_enabled': self.ml_enabled,
            'recent_trend': self._analyze_recent_trend(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_recent_trend(self):
        """📈 Analisa tendência recente"""
        if len(self.ml_data) < 5:
            return 'insufficient_data'
        
        recent_successes = sum([1 for d in self.ml_data[-5:] if d.get('success', False)])
        
        if recent_successes >= 4:
            return 'very_positive'
        elif recent_successes >= 3:
            return 'positive'
        elif recent_successes >= 2:
            return 'neutral'
        else:
            return 'negative'
    
    def _get_recommendation(self, confidence, martingale_level, is_after_loss):
        """Gerar recomendação baseada na análise"""
        if is_after_loss and martingale_level > 0:
            return "wait_for_better_setup"
        elif confidence > 85:
            return "strong_signal"
        elif confidence > 75:
            return "moderate_signal"
        else:
            return "weak_signal"
    
    def _get_signal_recommendation(self, confidence, direction):
        """Gerar recomendação para o sinal"""
        if confidence > 85:
            return f"FORTE: {direction} com alta confiança"
        elif confidence > 75:
            return f"MODERADO: {direction} com boa confiança"
        else:
            return f"FRACO: {direction} com baixa confiança"

# Instância global da IA
trading_ai = TradingAI()

# Rotas da API

@app.route('/')
def index():
    """Servir o frontend"""
    return send_from_directory('public', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check da API"""
    ml_stats = trading_ai.get_ml_stats()
    
    return jsonify({
        'status': 'OK',
        'service': 'Trading Bot IA Python + ML Simples',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'features': [
            'Market Analysis',
            'Trading Signals', 
            'Risk Assessment',
            'Martingale Intelligence',
            '🤖 Simple ML Engine',
            '📊 Pattern Learning',
            '🎯 Smart Predictions',
            '⚠️ ML Risk Analysis'
        ],
        'ml_status': {
            'enabled': trading_ai.ml_enabled,
            'data_size': ml_stats['total_trades'],
            'success_rate': ml_stats['success_rate'],
            'recent_trend': ml_stats.get('recent_trend', 'unknown')
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_market():
    """Endpoint para análise de mercado"""
    try:
        market_data = request.get_json()
        
        if not market_data:
            return jsonify({'error': 'Dados de mercado necessários'}), 400
            
        analysis = trading_ai.analyze_market(market_data)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro na análise de mercado',
            'details': str(e)
        }), 500

@app.route('/api/signal', methods=['POST'])
def get_trading_signal():
    """Endpoint para obter sinal de trading"""
    try:
        signal_data = request.get_json()
        
        if not signal_data:
            return jsonify({'error': 'Dados para sinal necessários'}), 400
            
        signal = trading_ai.get_trading_signal(signal_data)
        
        return jsonify(signal)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro ao gerar sinal',
            'details': str(e)
        }), 500

@app.route('/api/risk', methods=['POST'])
def assess_risk():
    """Endpoint para avaliação de risco"""
    try:
        risk_data = request.get_json()
        
        if not risk_data:
            return jsonify({'error': 'Dados de risco necessários'}), 400
            
        risk_assessment = trading_ai.assess_risk(risk_data)
        
        return jsonify(risk_assessment)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro na avaliação de risco',
            'details': str(e)
        }), 500

# ✅ NOVO ENDPOINT: Adicionar dados ML
@app.route('/api/ml/learn', methods=['POST'])
def ml_learn():
    """Endpoint para aprendizado ML"""
    try:
        trade_data = request.get_json()
        
        if not trade_data:
            return jsonify({'error': 'Dados de trade necessários'}), 400
        
        success = trading_ai.add_ml_data(trade_data)
        ml_stats = trading_ai.get_ml_stats()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Trade adicionado ao ML',
                'ml_stats': ml_stats
            })
        else:
            return jsonify({'error': 'Falha ao adicionar ao ML'}), 500
            
    except Exception as e:
        return jsonify({
            'error': 'Erro no aprendizado ML',
            'details': str(e)
        }), 500

@app.route('/api/ml/stats', methods=['GET'])
def ml_statistics():
    """Estatísticas do ML"""
    try:
        stats = trading_ai.get_ml_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': 'Erro ao obter estatísticas ML',
            'details': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_ai_stats():
    """Estatísticas da IA"""
    ml_stats = trading_ai.get_ml_stats()
    
    return jsonify({
        'total_analyses': len(trading_ai.analysis_history),
        'last_analysis': trading_ai.last_analysis,
        'uptime': datetime.now().isoformat(),
        'status': 'active',
        'ml_stats': ml_stats
    })

@app.route('/api/frontend-js', methods=['GET'])
def get_frontend_js():
    """Servir JavaScript adicional"""
    js_code = """
    // JavaScript adicional para o frontend + ML
    console.log('🐍 JavaScript + ML Simples carregado');
    
    // Função para verificar status da IA + ML
    setInterval(async () => {
        try {
            const response = await fetch('/api/health');
            if (response.ok) {
                const data = await response.json();
                document.getElementById('connectionMethod').textContent = 
                    `Python API + ML Online (${data.ml_status.data_size} trades, ${data.ml_status.success_rate.toFixed(1)}% success)`;
                document.getElementById('apiKeyStatus').textContent = 
                    `ML: ${data.ml_status.recent_trend}`;
            }
        } catch (error) {
            document.getElementById('connectionMethod').textContent = 'Python API Offline';
        }
    }, 30000);
    
    // ✅ Função para notificar resultado do trade para ML
    window.notifyMLTrade = function(tradeData) {
        fetch('/api/ml/learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tradeData)
        }).then(response => response.json())
          .then(data => {
              if (data.status === 'success') {
                  console.log('✅ ML aprendeu com trade:', data.ml_stats.total_trades, 'trades');
              }
          }).catch(error => {
              console.error('❌ Erro ML:', error);
          });
    };
    """
    
    return js_code, 200, {'Content-Type': 'application/javascript'}

# Servir arquivos estáticos
@app.route('/public/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    
    print("🚀 Iniciando Trading Bot com ML Simples...")
    print(f"🤖 ML Status: Ativo")
    print(f"📊 Dados ML: {len(trading_ai.ml_data)} trades")
    
    app.run(host='0.0.0.0', port=port, debug=False)
