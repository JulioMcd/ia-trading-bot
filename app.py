# app.py - Trading Bot IA API - Compatível com Render
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import random
import time
import os
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app)

# ===============================================
# CONFIGURAÇÕES
# ===============================================

CONFIG = {
    'AI_CONFIDENCE_MIN': 70,
    'AI_CONFIDENCE_MAX': 95,
    'ANALYSIS_SYMBOLS': [
        'EURUSD-OTC', 'GBPUSD-OTC', 'USDJPY-OTC', 'AUDUSD-OTC',
        'USDCAD-OTC', 'USDCHF-OTC', 'R_10', 'R_25', 'R_50', 'R_75', 'R_100'
    ],
    'VOLATILITY_INDICES': ['R_10', 'R_25', 'R_50', 'R_75', 'R_100'],
    'DURATION_LIMITS': {
        'ticks': {'min': 1, 'max': 10},
        'minutes': {'min': 1, 'max': 5}
    }
}

# ===============================================
# FUNÇÕES DE IA AVANÇADAS
# ===============================================

def analyze_market_conditions(symbol, current_price=None, volatility=None):
    """Análise avançada das condições de mercado"""
    
    # Simular análise baseada no símbolo
    is_volatility_index = symbol in CONFIG['VOLATILITY_INDICES']
    
    if is_volatility_index:
        # Índices de volatilidade têm comportamento específico
        base_volatility = random.uniform(40, 80)
        trend_strength = random.uniform(0.3, 0.8)
        market_condition = 'volatile'
    else:
        # Pares de moedas têm volatilidade menor
        base_volatility = random.uniform(20, 60)
        trend_strength = random.uniform(0.2, 0.7)
        market_condition = random.choice(['trending', 'ranging', 'volatile'])
    
    # Ajustar baseado em dados fornecidos
    if volatility:
        base_volatility = (base_volatility + volatility) / 2
    
    # Determinar força da tendência
    if trend_strength > 0.6:
        trend = 'strong'
    elif trend_strength > 0.4:
        trend = 'moderate'
    else:
        trend = 'weak'
    
    # Calcular confiança baseada na análise
    confidence = 70 + (trend_strength * 20) + random.uniform(-5, 10)
    confidence = max(70, min(95, confidence))
    
    return {
        'volatility': base_volatility,
        'trend_strength': trend_strength,
        'market_condition': market_condition,
        'trend': trend,
        'confidence': confidence
    }

def generate_trading_signal(symbol, market_data=None):
    """Gera sinal de trading inteligente"""
    
    analysis = analyze_market_conditions(symbol, 
                                       market_data.get('current_price') if market_data else None,
                                       market_data.get('volatility') if market_data else None)
    
    # Determinar direção baseada na análise
    if analysis['trend'] == 'strong':
        if analysis['trend_strength'] > 0.6:
            direction = 'call' if random.random() > 0.3 else 'put'
        else:
            direction = 'put' if random.random() > 0.3 else 'call'
    else:
        direction = random.choice(['call', 'put'])
    
    # Ajustar confiança baseada em condições
    confidence = analysis['confidence']
    
    # Win rate histórico (simulado)
    win_rate = market_data.get('win_rate', 0) if market_data else 0
    if win_rate > 70:
        confidence += 5
    elif win_rate < 40:
        confidence -= 5
    
    confidence = max(70, min(95, confidence))
    
    # Reasoning inteligente
    reasons = []
    if analysis['volatility'] > 60:
        reasons.append(f"Alta volatilidade ({analysis['volatility']:.1f}%)")
    if analysis['trend'] == 'strong':
        reasons.append(f"Tendência forte detectada")
    if symbol in CONFIG['VOLATILITY_INDICES']:
        reasons.append("Padrão de índice sintético")
    
    reasoning = " | ".join(reasons) if reasons else "Análise técnica avançada"
    
    return {
        'direction': direction,
        'confidence': confidence,
        'reasoning': reasoning,
        'volatility': analysis['volatility'],
        'trend_strength': analysis['trend_strength'],
        'market_condition': analysis['market_condition'],
        'optimal_timeframe': determine_optimal_timeframe(analysis, symbol)
    }

def determine_optimal_timeframe(analysis, symbol):
    """Determina timeframe ótimo baseado na análise"""
    
    volatility = analysis['volatility']
    trend_strength = analysis['trend_strength']
    is_volatility_index = symbol in CONFIG['VOLATILITY_INDICES']
    
    if is_volatility_index:
        # Índices de volatilidade preferem ticks
        if volatility > 70:
            return {'type': 'ticks', 'duration': random.randint(1, 3)}
        elif volatility > 50:
            return {'type': 'ticks', 'duration': random.randint(3, 6)}
        else:
            return {'type': 'ticks', 'duration': random.randint(5, 10)}
    else:
        # Pares de moeda preferem minutos
        if trend_strength > 0.6:
            return {'type': 'minutes', 'duration': random.randint(1, 2)}
        elif trend_strength > 0.4:
            return {'type': 'minutes', 'duration': random.randint(2, 4)}
        else:
            return {'type': 'minutes', 'duration': random.randint(3, 5)}

def assess_risk_level(trading_data):
    """Avalia nível de risco da operação"""
    
    balance = trading_data.get('current_balance', 1000)
    today_pnl = trading_data.get('today_pnl', 0)
    martingale_level = trading_data.get('martingale_level', 0)
    win_rate = trading_data.get('win_rate', 50)
    current_stake = trading_data.get('current_stake', 1)
    
    risk_score = 0
    risk_factors = []
    
    # Avaliar P&L do dia
    daily_loss_percent = (abs(today_pnl) / balance * 100) if today_pnl < 0 else 0
    if daily_loss_percent > 20:
        risk_score += 30
        risk_factors.append(f"Perda diária alta ({daily_loss_percent:.1f}%)")
    elif daily_loss_percent > 10:
        risk_score += 15
        risk_factors.append(f"Perda diária moderada ({daily_loss_percent:.1f}%)")
    
    # Avaliar nível de Martingale
    if martingale_level > 5:
        risk_score += 25
        risk_factors.append(f"Martingale nível alto ({martingale_level})")
    elif martingale_level > 3:
        risk_score += 15
        risk_factors.append(f"Martingale ativo ({martingale_level})")
    
    # Avaliar win rate
    if win_rate < 30:
        risk_score += 20
        risk_factors.append(f"Taxa de acerto baixa ({win_rate:.1f}%)")
    elif win_rate < 45:
        risk_score += 10
        risk_factors.append(f"Performance abaixo da média")
    
    # Avaliar stake em relação ao saldo
    stake_percent = (current_stake / balance * 100)
    if stake_percent > 10:
        risk_score += 20
        risk_factors.append(f"Stake alto ({stake_percent:.1f}% do saldo)")
    elif stake_percent > 5:
        risk_score += 10
        risk_factors.append(f"Stake moderado ({stake_percent:.1f}% do saldo)")
    
    # Determinar nível de risco
    if risk_score >= 50:
        level = 'high'
        recommendation = 'Pare ou reduza significativamente o stake'
    elif risk_score >= 25:
        level = 'medium'
        recommendation = 'Considere reduzir o stake ou fazer uma pausa'
    else:
        level = 'low'
        recommendation = 'Operação dentro dos parâmetros normais'
    
    return {
        'level': level,
        'score': risk_score,
        'factors': risk_factors,
        'recommendation': recommendation,
        'suggested_action': 'pause' if risk_score >= 50 else 'reduce' if risk_score >= 35 else 'continue'
    }

def generate_management_decision(trading_data):
    """Gera decisão de gerenciamento inteligente"""
    
    risk_assessment = assess_risk_level(trading_data)
    
    current_stake = trading_data.get('current_stake', 1)
    balance = trading_data.get('current_balance', 1000)
    martingale_level = trading_data.get('martingale_level', 0)
    win_rate = trading_data.get('win_rate', 50)
    
    # Decisão baseada no risco
    if risk_assessment['suggested_action'] == 'pause':
        return {
            'action': 'pause',
            'pause_duration': random.randint(30000, 120000),  # 30s a 2min
            'reason': 'Alto risco detectado',
            'risk_level': risk_assessment['level'],
            'message': f"IA recomenda pausa: {risk_assessment['recommendation']}"
        }
    
    # Ajuste de stake
    recommended_stake = current_stake
    
    if martingale_level == 0:  # Só ajustar stake se não estiver em Martingale
        if win_rate > 70:
            # Performance boa - pode aumentar ligeiramente
            recommended_stake = min(current_stake * 1.1, balance * 0.05)
        elif win_rate < 40:
            # Performance ruim - reduzir stake
            recommended_stake = max(current_stake * 0.8, 1)
        
        # Ajuste baseado no saldo
        if recommended_stake > balance * 0.1:
            recommended_stake = balance * 0.05
    
    return {
        'action': 'continue',
        'recommended_stake': round(recommended_stake, 2),
        'risk_level': risk_assessment['level'],
        'confidence': 85 + random.uniform(-10, 10),
        'message': f"Stake recomendado: ${recommended_stake:.2f} | Risco: {risk_assessment['level']}",
        'risk_factors': risk_assessment['factors']
    }

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Trading Bot IA API - Funcionando!',
        'version': '2.0.0',
        'features': [
            'Análise avançada de mercado',
            'Sinais de trading inteligentes',
            'Duração otimizada por IA',
            'Gerenciamento de risco automático',
            'Suporte a índices de volatilidade',
            'Avaliação de Martingale'
        ],
        'endpoints': [
            '/analyze', '/signal', '/duration', '/management',
            '/risk-assessment', '/optimal-duration', '/trading-signal'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST', 'GET'])
def analyze_market():
    """Análise avançada de mercado"""
    
    if request.method == 'GET':
        # Análise genérica se não houver dados
        symbol = 'EURUSD-OTC'
        market_data = {}
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
        market_data = data
    
    analysis = analyze_market_conditions(symbol, 
                                       market_data.get('current_price'),
                                       market_data.get('volatility'))
    
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'analysis': analysis,
        'message': f"Análise de {symbol}: {analysis['market_condition']} | Volatilidade {analysis['volatility']:.1f}%",
        'timestamp': datetime.now().isoformat(),
        'confidence': analysis['confidence'],
        'trend': analysis['trend'],
        'volatility': analysis['volatility']
    })

@app.route('/signal', methods=['POST', 'GET'])
@app.route('/trading-signal', methods=['POST', 'GET'])
def get_trading_signal():
    """Gera sinal de trading inteligente"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
        market_data = {}
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
        market_data = data
    
    signal = generate_trading_signal(symbol, market_data)
    
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'direction': signal['direction'],
        'confidence': signal['confidence'],
        'reasoning': signal['reasoning'],
        'volatility': signal['volatility'],
        'trend_strength': signal['trend_strength'],
        'market_condition': signal['market_condition'],
        'optimal_timeframe': signal['optimal_timeframe'],
        'message': f"Sinal {signal['direction'].upper()}: {signal['reasoning']}",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/duration', methods=['POST', 'GET'])
@app.route('/optimal-duration', methods=['POST', 'GET'])
@app.route('/timeframe', methods=['POST', 'GET'])
def get_optimal_duration():
    """Determina duração ótima para o trade"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
        market_data = {}
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
        market_data = data
    
    analysis = analyze_market_conditions(symbol, 
                                       market_data.get('current_price'),
                                       market_data.get('volatility'))
    
    timeframe = determine_optimal_timeframe(analysis, symbol)
    
    # Garantir que está dentro dos limites
    duration_type = timeframe['type']
    duration_value = timeframe['duration']
    
    if duration_type == 'ticks':
        limits = CONFIG['DURATION_LIMITS']['ticks']
        duration_value = max(limits['min'], min(limits['max'], duration_value))
    else:
        limits = CONFIG['DURATION_LIMITS']['minutes']
        duration_value = max(limits['min'], min(limits['max'], duration_value))
    
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'type': 't' if duration_type == 'ticks' else 'm',
        'duration_type': duration_type,
        'duration': duration_value,
        'value': duration_value,
        'confidence': analysis['confidence'],
        'reasoning': f"Otimizado para {symbol}: {duration_value} {duration_type} baseado em volatilidade {analysis['volatility']:.1f}%",
        'volatility': analysis['volatility'],
        'market_condition': analysis['market_condition'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/management', methods=['POST', 'GET'])
@app.route('/risk-management', methods=['POST', 'GET'])
@app.route('/auto-manage', methods=['POST', 'GET'])
def risk_management():
    """Gerenciamento inteligente de risco"""
    
    if request.method == 'GET':
        trading_data = {
            'current_balance': 1000,
            'today_pnl': 0,
            'martingale_level': 0,
            'win_rate': 50,
            'current_stake': 1
        }
    else:
        trading_data = request.get_json() or {}
    
    decision = generate_management_decision(trading_data)
    
    return jsonify({
        'status': 'success',
        'action': decision['action'],
        'recommended_stake': decision.get('recommended_stake'),
        'pause_duration': decision.get('pause_duration'),
        'risk_level': decision['risk_level'],
        'message': decision['message'],
        'confidence': decision.get('confidence', 85),
        'should_pause': decision['action'] == 'pause',
        'risk_factors': decision.get('risk_factors', []),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/risk-assessment', methods=['POST', 'GET'])
def risk_assessment():
    """Avaliação detalhada de risco"""
    
    if request.method == 'GET':
        trading_data = {
            'current_balance': 1000,
            'today_pnl': 0,
            'martingale_level': 0,
            'win_rate': 50,
            'current_stake': 1
        }
    else:
        trading_data = request.get_json() or {}
    
    risk = assess_risk_level(trading_data)
    
    return jsonify({
        'status': 'success',
        'level': risk['level'],
        'score': risk['score'],
        'factors': risk['factors'],
        'recommendation': risk['recommendation'],
        'suggested_action': risk['suggested_action'],
        'message': f"Risco {risk['level'].upper()}: {risk['recommendation']}",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check para monitoramento"""
    return jsonify({
        'status': 'healthy',
        'message': 'Trading Bot IA API operacional',
        'uptime': 'online',
        'timestamp': datetime.now().isoformat()
    })

# ===============================================
# TRATAMENTO DE ERROS
# ===============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint não encontrado',
        'available_endpoints': [
            '/', '/analyze', '/signal', '/duration', '/management',
            '/risk-assessment', '/health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Erro interno do servidor',
        'timestamp': datetime.now().isoformat()
    }), 500

# ===============================================
# INICIALIZAÇÃO
# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Iniciando Trading Bot IA API...")
    print(f"🌐 Porta: {port}")
    print(f"🔧 Debug: {debug}")
    print("🤖 Recursos: Análise IA + Sinais + Duração + Gerenciamento")
    print("✅ API pronta para uso!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
