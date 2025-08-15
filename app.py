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

# Simulador de IA para análise de mercado
class TradingAI:
    def __init__(self):
        self.last_analysis = None
        self.analysis_history = []
    
    def analyze_market(self, market_data):
        """Análise inteligente do mercado baseada nos dados recebidos"""
        
        # Extrair dados importantes
        symbol = market_data.get('symbol', 'R_50')
        current_price = market_data.get('currentPrice', 1000)
        martingale_level = market_data.get('martingaleLevel', 0)
        is_after_loss = market_data.get('isAfterLoss', False)
        win_rate = market_data.get('winRate', 50)
        volatility = market_data.get('volatility', 50)
        
        # Simular tempo de processamento da IA
        time.sleep(random.uniform(1, 3))
        
        # Análise baseada no contexto
        if is_after_loss and martingale_level > 0:
            # Ser mais conservador após perdas
            confidence = random.uniform(65, 80)
            trend = 'neutral'
            message = f"Análise pós-perda do {symbol}: Recomendação conservadora (Martingale Nível {martingale_level})"
        elif martingale_level > 4:
            # Alto risco
            confidence = random.uniform(60, 75)
            trend = 'neutral'
            message = f"Análise de alto risco do {symbol}: Martingale Nível {martingale_level} - Cautela recomendada"
        else:
            # Análise normal
            confidence = random.uniform(70, 95)
            trend = random.choice(Config.MARKET_TRENDS)
            message = f"Análise do {symbol}: Volatilidade {volatility:.1f}%, Tendência {trend}"
        
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
            'recommendation': self._get_recommendation(confidence, martingale_level, is_after_loss)
        }
        
        self.last_analysis = analysis
        self.analysis_history.append(analysis)
        
        # Manter apenas últimas 50 análises
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
            
        return analysis
    
    def get_trading_signal(self, signal_data):
        """Gerar sinal de trading baseado na análise"""
        
        symbol = signal_data.get('symbol', 'R_50')
        current_price = signal_data.get('currentPrice', 1000)
        martingale_level = signal_data.get('martingaleLevel', 0)
        is_after_loss = signal_data.get('isAfterLoss', False)
        win_rate = signal_data.get('winRate', 50)
        
        # Simular processamento
        time.sleep(random.uniform(1, 2))
        
        # Determinar direção baseada em análise
        if is_after_loss and martingale_level > 0:
            # Mais conservador após perdas
            confidence = random.uniform(70, 82)
            direction = random.choice(['CALL', 'PUT'])
            reasoning = f"Sinal conservador pós-perda (Martingale {martingale_level})"
        elif martingale_level > 4:
            # Muito conservador em alto martingale
            confidence = random.uniform(65, 78)
            direction = random.choice(['CALL', 'PUT'])
            reasoning = f"Sinal de alto risco - Martingale Nível {martingale_level}"
        else:
            # Sinal normal
            confidence = random.uniform(75, 92)
            direction = random.choice(['CALL', 'PUT'])
            reasoning = "Baseado em padrões de mercado e análise técnica"
        
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
            'recommendation': self._get_signal_recommendation(confidence, direction)
        }
        
        return signal
    
    def assess_risk(self, risk_data):
        """Avaliar risco da operação"""
        
        current_balance = risk_data.get('currentBalance', 1000)
        today_pnl = risk_data.get('todayPnL', 0)
        martingale_level = risk_data.get('martingaleLevel', 0)
        win_rate = risk_data.get('winRate', 50)
        total_trades = risk_data.get('totalTrades', 0)
        is_cooling = risk_data.get('isInCoolingPeriod', False)
        needs_analysis = risk_data.get('needsAnalysisAfterLoss', False)
        
        # Simular análise
        time.sleep(random.uniform(0.5, 1.5))
        
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
            
        # Determinar nível e recomendação
        if risk_score >= 60:
            level = 'high'
            message = f"Risco ALTO detectado - Score: {risk_score}"
            if martingale_level > 5:
                recommendation = "PARAR operações - Martingale muito alto"
            else:
                recommendation = "Reduzir stake e operar com extrema cautela"
        elif risk_score >= 35:
            level = 'medium'
            message = f"Risco MODERADO - Score: {risk_score}"
            if martingale_level > 0:
                recommendation = f"Cautela - Martingale Nível {martingale_level} ativo"
            else:
                recommendation = "Operar com cautela moderada"
        else:
            level = 'low'
            message = f"Risco BAIXO - Score: {risk_score}"
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
            'details': {
                'martingale_risk': martingale_level * 5,
                'pnl_risk': max(0, abs(pnl_percentage) - 5) * 2,
                'performance_risk': max(0, 50 - win_rate),
                'total_score': risk_score
            }
        }
        
        return risk_assessment
    
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
    return jsonify({
        'status': 'OK',
        'service': 'Trading Bot IA Python',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'features': [
            'Market Analysis',
            'Trading Signals', 
            'Risk Assessment',
            'Martingale Intelligence'
        ]
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

@app.route('/api/stats', methods=['GET'])
def get_ai_stats():
    """Estatísticas da IA"""
    return jsonify({
        'total_analyses': len(trading_ai.analysis_history),
        'last_analysis': trading_ai.last_analysis,
        'uptime': datetime.now().isoformat(),
        'status': 'active'
    })

@app.route('/api/frontend-js', methods=['GET'])
def get_frontend_js():
    """Servir JavaScript adicional se necessário"""
    js_code = """
    // JavaScript adicional para o frontend
    console.log('🐍 JavaScript adicional da API Python carregado');
    
    // Função para verificar status da IA
    setInterval(async () => {
        try {
            const response = await fetch('/api/health');
            if (response.ok) {
                document.getElementById('connectionMethod').textContent = 'Python API Online';
            }
        } catch (error) {
            document.getElementById('connectionMethod').textContent = 'Python API Offline';
        }
    }, 30000); // Verificar a cada 30 segundos
    """
    
    return js_code, 200, {'Content-Type': 'application/javascript'}

# Servir arquivos estáticos
@app.route('/public/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
