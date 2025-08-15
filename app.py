from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import random
import time
from datetime import datetime
import json
import threading

# Importar módulos ML
from ml_engine import ml_engine
from ml_api import setup_ml_api

app = Flask(__name__)
CORS(app)

# Configurações
class Config:
    MIN_STAKE = 0.35
    MAX_STAKE = 2000
    AI_CONFIDENCE_RANGE = (70, 95)
    RISK_LEVELS = ['low', 'medium', 'high']
    MARKET_TRENDS = ['bullish', 'bearish', 'neutral']

# Simulador de IA Original + ML Integration
class TradingAI:
    def __init__(self):
        self.last_analysis = None
        self.analysis_history = []
        self.ml_integration = True  # ✅ Nova flag para ML
    
    def analyze_market(self, market_data):
        """Análise inteligente do mercado com ML integrado"""
        
        # ✅ INTEGRAÇÃO ML - Usar ML Engine quando disponível
        if self.ml_integration and len(ml_engine.historical_data) > 10:
            try:
                # Obter predição ML
                ml_prediction = ml_engine.predict_direction(market_data)
                
                # Obter análise de risco ML
                ml_risk = ml_engine.analyze_risk(market_data, [])
                
                # Combinar análise tradicional com ML
                analysis = {
                    'message': f"🤖 ML Análise: {ml_prediction['direction']} com {ml_prediction['confidence']:.1f}% confiança",
                    'volatility': market_data.get('volatility', 50),
                    'trend': self._ml_to_trend(ml_prediction['direction']),
                    'confidence': ml_prediction['confidence'],
                    'timestamp': datetime.now().isoformat(),
                    'symbol': market_data.get('symbol', 'R_50'),
                    'martingaleLevel': market_data.get('martingaleLevel', 0),
                    'isAfterLoss': market_data.get('isAfterLoss', False),
                    'recommendation': 'strong_signal' if ml_prediction['confidence'] > 85 else 'moderate_signal',
                    'ml_enabled': True,
                    'ml_risk_level': ml_risk['level'],
                    'ml_reasoning': ml_prediction['reasoning']
                }
                
                print(f"🤖 ML Análise integrada: {ml_prediction['direction']} ({ml_prediction['confidence']:.1f}%)")
                
            except Exception as e:
                print(f"❌ Erro ML, usando análise tradicional: {e}")
                analysis = self._traditional_analysis(market_data)
                
        else:
            # Análise tradicional quando ML não disponível
            analysis = self._traditional_analysis(market_data)
        
        self.last_analysis = analysis
        self.analysis_history.append(analysis)
        
        # Manter apenas últimas 50 análises
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
            
        return analysis
    
    def get_trading_signal(self, signal_data):
        """Gerar sinal de trading com ML integrado"""
        
        # ✅ INTEGRAÇÃO ML - Priorizar ML quando disponível
        if self.ml_integration and len(ml_engine.historical_data) > 10:
            try:
                # Obter sinal ML
                ml_signal = ml_engine.predict_direction(signal_data)
                
                # Obter otimização Martingale ML
                martingale_level = signal_data.get('martingaleLevel', 0)
                recent_performance = {
                    'win_rate': signal_data.get('winRate', 50),
                    'total_trades': 10,
                    'consecutive_losses': martingale_level,
                    'avg_pnl': 0,
                    'max_loss': martingale_level * 2,
                    'time_since_last_win': 60
                }
                
                ml_martingale = ml_engine.optimize_martingale(martingale_level, recent_performance)
                
                # Combinar sinal ML com otimização
                signal = {
                    'direction': ml_signal['direction'],
                    'confidence': ml_signal['confidence'],
                    'reasoning': f"🤖 ML: {ml_signal['reasoning']} | Martingale: {ml_martingale['action']}",
                    'timeframe': '5m',
                    'entry_price': signal_data.get('currentPrice', 1000),
                    'timestamp': datetime.now().isoformat(),
                    'symbol': signal_data.get('symbol', 'R_50'),
                    'martingaleLevel': martingale_level,
                    'isAfterLoss': signal_data.get('isAfterLoss', False),
                    'recommendation': self._get_ml_signal_recommendation(ml_signal['confidence'], ml_signal['direction']),
                    'ml_enabled': True,
                    'ml_martingale_action': ml_martingale['action'],
                    'optimal_stake': self._calculate_ml_stake(signal_data, ml_signal['confidence'])
                }
                
                print(f"🤖 ML Sinal integrado: {ml_signal['direction']} ({ml_signal['confidence']:.1f}%)")
                
            except Exception as e:
                print(f"❌ Erro ML signal, usando tradicional: {e}")
                signal = self._traditional_signal(signal_data)
                
        else:
            # Sinal tradicional quando ML não disponível
            signal = self._traditional_signal(signal_data)
        
        return signal
    
    def assess_risk(self, risk_data):
        """Avaliação de risco com ML integrado"""
        
        # ✅ INTEGRAÇÃO ML - Usar análise ML quando disponível
        if self.ml_integration and len(ml_engine.historical_data) > 10:
            try:
                # Obter análise de risco ML
                ml_risk = ml_engine.analyze_risk(risk_data, ml_engine.historical_data[-10:])
                
                # Enriquecer com análise tradicional
                traditional_risk = self._traditional_risk(risk_data)
                
                # Combinar ambas as análises
                risk_assessment = {
                    'level': ml_risk['level'],
                    'message': f"🤖 ML: {ml_risk['message']} | Tradicional: {traditional_risk['message']}",
                    'score': ml_risk['score'],
                    'recommendation': ml_risk['recommendation'],
                    'timestamp': datetime.now().isoformat(),
                    'martingaleLevel': risk_data.get('martingaleLevel', 0),
                    'isAfterLoss': risk_data.get('needsAnalysisAfterLoss', False),
                    'currentBalance': risk_data.get('currentBalance', 1000),
                    'todayPnL': risk_data.get('todayPnL', 0),
                    'winRate': risk_data.get('winRate', 50),
                    'ml_enabled': True,
                    'ml_risk_factors': ml_risk.get('risk_factors', []),
                    'combined_score': (ml_risk['score'] + traditional_risk['score']) / 2,
                    'details': {
                        'ml_analysis': ml_risk,
                        'traditional_analysis': traditional_risk
                    }
                }
                
                print(f"🤖 ML Risco integrado: {ml_risk['level']} (Score: {ml_risk['score']:.1f})")
                
            except Exception as e:
                print(f"❌ Erro ML risk, usando tradicional: {e}")
                risk_assessment = self._traditional_risk(risk_data)
                
        else:
            # Análise tradicional quando ML não disponível
            risk_assessment = self._traditional_risk(risk_data)
        
        return risk_assessment
    
    def _ml_to_trend(self, direction):
        """Converte direção ML para tendência"""
        return 'bullish' if direction == 'CALL' else 'bearish'
    
    def _get_ml_signal_recommendation(self, confidence, direction):
        """Gera recomendação baseada no ML"""
        if confidence > 90:
            return f"MUITO FORTE: {direction} com {confidence:.1f}% confiança"
        elif confidence > 80:
            return f"FORTE: {direction} com {confidence:.1f}% confiança"
        elif confidence > 70:
            return f"MODERADO: {direction} com {confidence:.1f}% confiança"
        else:
            return f"FRACO: {direction} com {confidence:.1f}% confiança"
    
    def _calculate_ml_stake(self, signal_data, confidence):
        """Calcula stake ótimo baseado em ML"""
        base_stake = signal_data.get('accountBalance', 1000) * 0.02  # 2% do saldo
        
        # Ajustar baseado na confiança ML
        confidence_multiplier = confidence / 100
        optimal_stake = base_stake * confidence_multiplier
        
        # Limitar entre min e max
        return max(Config.MIN_STAKE, min(optimal_stake, Config.MAX_STAKE))
    
    def _traditional_analysis(self, market_data):
        """Análise tradicional (fallback)"""
        symbol = market_data.get('symbol', 'R_50')
        martingale_level = market_data.get('martingaleLevel', 0)
        is_after_loss = market_data.get('isAfterLoss', False)
        win_rate = market_data.get('winRate', 50)
        volatility = market_data.get('volatility', 50)
        
        # Simular tempo de processamento
        time.sleep(random.uniform(1, 3))
        
        # Análise baseada no contexto
        if is_after_loss and martingale_level > 0:
            confidence = random.uniform(65, 80)
            trend = 'neutral'
            message = f"Análise pós-perda do {symbol}: Recomendação conservadora (Martingale Nível {martingale_level})"
        elif martingale_level > 4:
            confidence = random.uniform(60, 75)
            trend = 'neutral'
            message = f"Análise de alto risco do {symbol}: Martingale Nível {martingale_level} - Cautela recomendada"
        else:
            confidence = random.uniform(70, 95)
            trend = random.choice(Config.MARKET_TRENDS)
            message = f"Análise do {symbol}: Volatilidade {volatility:.1f}%, Tendência {trend}"
        
        # Ajustar confiança baseada na taxa de acerto
        if win_rate > 70:
            confidence += 5
        elif win_rate < 40:
            confidence -= 10
            
        confidence = max(60, min(95, confidence))
        
        return {
            'message': message,
            'volatility': volatility,
            'trend': trend,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'martingaleLevel': martingale_level,
            'isAfterLoss': is_after_loss,
            'recommendation': 'wait_for_better_setup' if is_after_loss else 'continue_normal',
            'ml_enabled': False
        }
    
    def _traditional_signal(self, signal_data):
        """Sinal tradicional (fallback)"""
        symbol = signal_data.get('symbol', 'R_50')
        current_price = signal_data.get('currentPrice', 1000)
        martingale_level = signal_data.get('martingaleLevel', 0)
        is_after_loss = signal_data.get('isAfterLoss', False)
        win_rate = signal_data.get('winRate', 50)
        
        time.sleep(random.uniform(1, 2))
        
        if is_after_loss and martingale_level > 0:
            confidence = random.uniform(70, 82)
            direction = random.choice(['CALL', 'PUT'])
            reasoning = f"Sinal conservador pós-perda (Martingale {martingale_level})"
        elif martingale_level > 4:
            confidence = random.uniform(65, 78)
            direction = random.choice(['CALL', 'PUT'])
            reasoning = f"Sinal de alto risco - Martingale Nível {martingale_level}"
        else:
            confidence = random.uniform(75, 92)
            direction = random.choice(['CALL', 'PUT'])
            reasoning = "Baseado em padrões de mercado e análise técnica"
        
        if win_rate > 70:
            confidence += 3
        elif win_rate < 40:
            confidence -= 8
            
        confidence = max(65, min(95, confidence))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'reasoning': reasoning,
            'timeframe': '5m',
            'entry_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'martingaleLevel': martingale_level,
            'isAfterLoss': is_after_loss,
            'recommendation': f"Tradicional: {direction} com {confidence:.1f}% confiança",
            'ml_enabled': False
        }
    
    def _traditional_risk(self, risk_data):
        """Análise de risco tradicional (fallback)"""
        current_balance = risk_data.get('currentBalance', 1000)
        today_pnl = risk_data.get('todayPnL', 0)
        martingale_level = risk_data.get('martingaleLevel', 0)
        win_rate = risk_data.get('winRate', 50)
        total_trades = risk_data.get('totalTrades', 0)
        is_cooling = risk_data.get('isInCoolingPeriod', False)
        needs_analysis = risk_data.get('needsAnalysisAfterLoss', False)
        
        time.sleep(random.uniform(0.5, 1.5))
        
        # Calcular nível de risco
        risk_score = 0
        
        if martingale_level > 6:
            risk_score += 40
        elif martingale_level > 3:
            risk_score += 25
        elif martingale_level > 0:
            risk_score += 10
            
        pnl_percentage = (today_pnl / current_balance) * 100 if current_balance > 0 else 0
        if pnl_percentage < -20:
            risk_score += 30
        elif pnl_percentage < -10:
            risk_score += 20
        elif pnl_percentage < -5:
            risk_score += 10
            
        if win_rate < 30:
            risk_score += 25
        elif win_rate < 45:
            risk_score += 15
            
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
            
        if martingale_level > 0:
            message += f" | Martingale Nível {martingale_level}"
            
        if is_cooling:
            message += " | Em período de cooling"
            
        if needs_analysis:
            message += " | Aguardando análise pós-perda"
        
        return {
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
            'ml_enabled': False
        }

# Instância global da IA integrada
trading_ai = TradingAI()

# ✅ CONFIGURAR ML API
ml_api = setup_ml_api(app)

# Rotas originais da API mantidas + Integração ML

@app.route('/')
def index():
    """Servir o frontend"""
    return send_from_directory('public', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check da API com status ML"""
    return jsonify({
        'status': 'OK',
        'service': 'Trading Bot IA Python + ML',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'features': [
            'Market Analysis',
            'Trading Signals', 
            'Risk Assessment',
            'Martingale Intelligence',
            '🤖 Machine Learning Engine',
            '📊 Continuous Learning',
            '🎯 ML Predictions',
            '⚠️ ML Risk Analysis'
        ],
        'ml_status': {
            'enabled': trading_ai.ml_integration,
            'models_trained': len([m for m in ml_engine.models.values() if m is not None]),
            'training_data_size': len(ml_engine.historical_data),
            'accuracy': ml_engine.metrics['accuracy']
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_market():
    """Endpoint para análise de mercado com ML"""
    try:
        market_data = request.get_json()
        
        if not market_data:
            return jsonify({'error': 'Dados de mercado necessários'}), 400
            
        # ✅ Usar análise integrada (ML + Tradicional)
        analysis = trading_ai.analyze_market(market_data)
        
        # ✅ Adicionar dados ao ML para aprendizado contínuo
        if market_data.get('symbol'):
            ml_engine.add_trade_data({
                'symbol': market_data.get('symbol'),
                'timestamp': datetime.now().isoformat(),
                'analysis_confidence': analysis.get('confidence', 50),
                'market_data': market_data
            })
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro na análise de mercado',
            'details': str(e)
        }), 500

@app.route('/api/signal', methods=['POST'])
def get_trading_signal():
    """Endpoint para obter sinal de trading com ML"""
    try:
        signal_data = request.get_json()
        
        if not signal_data:
            return jsonify({'error': 'Dados para sinal necessários'}), 400
            
        # ✅ Usar sinal integrado (ML + Tradicional)
        signal = trading_ai.get_trading_signal(signal_data)
        
        return jsonify(signal)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro ao gerar sinal',
            'details': str(e)
        }), 500

@app.route('/api/risk', methods=['POST'])
def assess_risk():
    """Endpoint para avaliação de risco com ML"""
    try:
        risk_data = request.get_json()
        
        if not risk_data:
            return jsonify({'error': 'Dados de risco necessários'}), 400
            
        # ✅ Usar análise de risco integrada (ML + Tradicional)
        risk_assessment = trading_ai.assess_risk(risk_data)
        
        return jsonify(risk_assessment)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro na avaliação de risco',
            'details': str(e)
        }), 500

# ✅ NOVO ENDPOINT: Notificação de resultado de trade para ML
@app.route('/api/trade-result', methods=['POST'])
def notify_trade_result():
    """Notifica resultado de trade para aprendizado ML"""
    try:
        trade_result = request.get_json()
        
        if not trade_result:
            return jsonify({'error': 'Dados do trade necessários'}), 400
        
        # Adicionar trade ao ML para aprendizado
        success = ml_engine.add_trade_data(trade_result)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Trade adicionado ao ML para aprendizado',
                'ml_data_size': len(ml_engine.historical_data),
                'ml_accuracy': ml_engine.metrics['accuracy']
            })
        else:
            return jsonify({'error': 'Falha ao adicionar trade ao ML'}), 500
            
    except Exception as e:
        return jsonify({
            'error': 'Erro ao notificar resultado',
            'details': str(e)
        }), 500

# ✅ NOVO ENDPOINT: Dashboard ML
@app.route('/api/ml-dashboard', methods=['GET'])
def ml_dashboard():
    """Dashboard com estatísticas ML"""
    try:
        dashboard_data = {
            'ml_engine_status': {
                'enabled': trading_ai.ml_integration,
                'training_data_size': len(ml_engine.historical_data),
                'models_status': {
                    name: 'trained' if model is not None else 'not_trained'
                    for name, model in ml_engine.models.items()
                },
                'metrics': ml_engine.metrics,
                'last_training': ml_engine.metrics.get('last_training'),
                'config': ml_engine.config
            },
            'recent_predictions': ml_engine.performance_history[-10:] if ml_engine.performance_history else [],
            'traditional_ai_status': {
                'total_analyses': len(trading_ai.analysis_history),
                'last_analysis': trading_ai.last_analysis
            },
            'integration_status': 'active' if trading_ai.ml_integration else 'disabled',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Erro no dashboard ML',
            'details': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_ai_stats():
    """Estatísticas da IA + ML"""
    return jsonify({
        'traditional_ai': {
            'total_analyses': len(trading_ai.analysis_history),
            'last_analysis': trading_ai.last_analysis,
            'uptime': datetime.now().isoformat(),
            'status': 'active'
        },
        'ml_engine': {
            'enabled': trading_ai.ml_integration,
            'training_data': len(ml_engine.historical_data),
            'accuracy': ml_engine.metrics['accuracy'],
            'models_trained': len([m for m in ml_engine.models.values() if m is not None]),
            'last_training': ml_engine.metrics.get('last_training')
        },
        'integration_status': 'full' if trading_ai.ml_integration else 'traditional_only'
    })

@app.route('/api/frontend-js', methods=['GET'])
def get_frontend_js():
    """Servir JavaScript adicional com integração ML"""
    js_code = """
    // JavaScript adicional para o frontend + ML
    console.log('🤖 JavaScript ML integrado carregado');
    
    // Função para verificar status da IA + ML
    setInterval(async () => {
        try {
            const response = await fetch('/api/health');
            if (response.ok) {
                const data = await response.json();
                document.getElementById('connectionMethod').textContent = 
                    `Python API + ML Online (${data.ml_status.training_data_size} trades)`;
                
                // Atualizar accuracy ML se disponível
                if (data.ml_status.accuracy > 0) {
                    document.getElementById('apiKeyStatus').textContent = 
                        `ML Accuracy: ${(data.ml_status.accuracy * 100).toFixed(1)}%`;
                }
            }
        } catch (error) {
            document.getElementById('connectionMethod').textContent = 'API Offline';
        }
    }, 30000);
    
    // ✅ NOVA FUNÇÃO: Notificar resultado do trade para ML
    window.notifyTradeResult = function(tradeData) {
        fetch('/api/trade-result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tradeData)
        }).then(response => response.json())
          .then(data => {
              if (data.status === 'success') {
                  console.log('✅ Trade adicionado ao ML:', data.ml_data_size, 'trades');
              }
          }).catch(error => {
              console.error('❌ Erro ao notificar ML:', error);
          });
    };
    
    // ✅ NOVA FUNÇÃO: Obter dashboard ML
    window.getMLDashboard = async function() {
        try {
            const response = await fetch('/api/ml-dashboard');
            const data = await response.json();
            console.log('🤖 ML Dashboard:', data);
            return data;
        } catch (error) {
            console.error('❌ Erro ML Dashboard:', error);
            return null;
        }
    };
    """
    
    return js_code, 200, {'Content-Type': 'application/javascript'}

# Servir arquivos estáticos
@app.route('/public/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    
    print("🚀 Iniciando Trading Bot com ML Engine integrado...")
    print(f"🤖 ML Status: {'Ativo' if trading_ai.ml_integration else 'Inativo'}")
    print(f"📊 Dados ML: {len(ml_engine.historical_data)} trades")
    print(f"🎯 Accuracy: {ml_engine.metrics['accuracy']:.2f}")
    
    app.run(host='0.0.0.0', port=port, debug=False)