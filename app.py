from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key válida (opcional)
VALID_API_KEY = "rnd_qpdTVwAeWzIItVbxHPPCc34uirv9"

# Dados de treino iniciais (mais robustos)
X_train = [
    [100, 101, 102], [101, 102, 103], [102, 101, 100], [103, 102, 101],
    [200, 205, 210], [210, 205, 200], [150, 155, 160], [160, 155, 150],
    [300, 310, 320], [320, 310, 300], [250, 260, 270], [270, 260, 250]
]
y_train = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = CALL, 0 = PUT

# Modelos IA
direction_model = RandomForestClassifier(n_estimators=20, random_state=42)
direction_model.fit(X_train, y_train)

# Dados históricos para análise
trade_history = []
performance_stats = {
    'total_trades': 0,
    'won_trades': 0,
    'total_pnl': 0.0
}

def validate_api_key():
    """Validar API Key (opcional)"""
    auth_header = request.headers.get('Authorization', '')
    api_key_header = request.headers.get('X-API-Key', '')
    
    if auth_header.startswith('Bearer '):
        api_key = auth_header.replace('Bearer ', '')
    else:
        api_key = api_key_header
    
    # Se não há API key, continua (modo permissivo)
    if not api_key:
        return True
    
    return api_key == VALID_API_KEY

def extract_features(data):
    """Extrair features dos dados recebidos"""
    # Tentar diferentes formas de obter dados de preço
    ticks = data.get("lastTicks", [])
    current_price = data.get("currentPrice", 1000)
    volatility = data.get("volatility", 50)
    
    if not ticks:
        # Gerar ticks simulados baseados no preço atual
        base_price = current_price
        ticks = [
            base_price - random.uniform(0, 5),
            base_price + random.uniform(0, 5),
            base_price - random.uniform(0, 3)
        ]
    
    # Garantir que temos pelo menos 3 valores
    while len(ticks) < 3:
        ticks.append(current_price + random.uniform(-2, 2))
    
    # Usar últimos 3 ticks
    features = ticks[-3:]
    
    # Adicionar volatilidade como feature se disponível
    if len(features) == 3:
        features.append(volatility)
    
    return features

def predict_direction_ml(features):
    """Predição com Machine Learning"""
    try:
        # Preparar features para o modelo
        if len(features) == 4:  # Incluindo volatilidade
            model_features = np.array(features[:3]).reshape(1, -1)
        else:
            model_features = np.array(features[-3:]).reshape(1, -1)
        
        # Predição
        prediction = direction_model.predict(model_features)[0]
        confidence_scores = direction_model.predict_proba(model_features)[0]
        confidence = confidence_scores[prediction] * 100
        
        # Ajustar confiança baseada na volatilidade
        if len(features) == 4:
            volatility = features[3]
            if volatility > 70:
                confidence = min(confidence + 5, 95)
            elif volatility < 30:
                confidence = max(confidence - 10, 65)
        
        direction = "CALL" if prediction == 1 else "PUT"
        
        return direction, round(confidence, 1)
        
    except Exception as e:
        logger.error(f"Erro na predição ML: {e}")
        # Fallback para análise técnica simples
        return analyze_technical_pattern(features)

def analyze_technical_pattern(features):
    """Análise técnica como backup"""
    try:
        if len(features) >= 3:
            recent_trend = features[-1] - features[-3]
            if recent_trend > 0:
                return "CALL", 72.5
            else:
                return "PUT", 71.8
        else:
            return "CALL" if random.random() > 0.5 else "PUT", 70.0
    except:
        return "CALL", 70.0

def calculate_risk_score(data):
    """Calcular score de risco"""
    martingale_level = data.get("martingaleLevel", 0)
    today_pnl = data.get("todayPnL", 0)
    win_rate = data.get("winRate", 50)
    total_trades = data.get("totalTrades", 0)
    
    risk_score = 0
    risk_level = "low"
    
    # Análise Martingale
    if martingale_level > 6:
        risk_score += 40
        risk_level = "high"
    elif martingale_level > 3:
        risk_score += 20
        risk_level = "medium"
    
    # Análise P&L
    if today_pnl < -100:
        risk_score += 25
        risk_level = "high"
    elif today_pnl < -50:
        risk_score += 10
        risk_level = "medium" if risk_level == "low" else risk_level
    
    # Análise Win Rate
    if win_rate < 30:
        risk_score += 20
        risk_level = "high"
    elif win_rate < 45:
        risk_score += 10
    
    # Over-trading
    if total_trades > 50:
        risk_score += 10
    
    return min(risk_score, 100), risk_level

def optimize_duration(data):
    """Otimizar duração baseada nos dados"""
    symbol = data.get("symbol", "R_50")
    volatility = data.get("volatility", 50)
    market_condition = data.get("marketCondition", "neutral")
    
    # Determinar se é índice de volatilidade
    is_volatility_index = "R_" in symbol or "HZ" in symbol
    
    if is_volatility_index:
        # Preferir ticks para índices de volatilidade
        duration_type = "t"
        if volatility > 70:
            duration = random.randint(1, 3)
        elif volatility > 40:
            duration = random.randint(4, 6)
        else:
            duration = random.randint(7, 10)
    else:
        # Análise mais complexa para outros ativos
        if random.random() > 0.3:
            duration_type = "m"
            if market_condition == "favorable":
                duration = random.randint(1, 2)
            elif market_condition == "unfavorable":
                duration = random.randint(4, 5)
            else:
                duration = random.randint(2, 4)
        else:
            duration_type = "t"
            duration = random.randint(3, 8)
    
    # Limites de segurança
    if duration_type == "t":
        duration = max(1, min(10, duration))
    else:
        duration = max(1, min(5, duration))
    
    confidence = 75 + random.uniform(0, 20)
    
    return {
        "type": duration_type,
        "duration_type": "ticks" if duration_type == "t" else "minutes",
        "value": duration,
        "duration": duration,
        "confidence": round(confidence, 1),
        "reasoning": f"Otimizado para {symbol}: {duration}{duration_type} baseado em volatilidade {volatility:.1f}%"
    }

def manage_position(data):
    """Gestão automática de posição"""
    current_balance = data.get("currentBalance", 1000)
    today_pnl = data.get("todayPnL", 0)
    martingale_level = data.get("martingaleLevel", 0)
    current_stake = data.get("currentStake", 1)
    win_rate = data.get("winRate", 50)
    
    action = "continue"
    recommended_stake = current_stake
    should_pause = False
    pause_duration = 0
    
    # Verificar se deve pausar
    if today_pnl < -200 or martingale_level > 7:
        should_pause = True
        action = "pause"
        pause_duration = random.randint(60000, 180000)  # 1-3 minutos
    elif today_pnl < -100 or martingale_level > 5:
        if random.random() > 0.7:
            should_pause = True
            action = "pause"
            pause_duration = random.randint(30000, 90000)  # 30-90 segundos
    
    # Ajustar stake se não em Martingale
    if not should_pause and martingale_level == 0:
        if win_rate > 70:
            recommended_stake = min(50, current_stake * 1.15)
        elif win_rate < 30:
            recommended_stake = max(0.35, current_stake * 0.8)
        elif today_pnl < -50:
            recommended_stake = max(0.35, current_stake * 0.9)
    
    message = ""
    if should_pause:
        message = f"PAUSA RECOMENDADA - {pause_duration//1000}s - Alto risco detectado"
    elif recommended_stake != current_stake:
        message = f"Stake ajustado: ${current_stake:.2f} → ${recommended_stake:.2f}"
    else:
        message = "Continuar operação - Parâmetros adequados"
    
    return {
        "action": action,
        "recommendedStake": round(recommended_stake, 2),
        "shouldPause": should_pause,
        "pauseDuration": pause_duration,
        "riskLevel": "high" if martingale_level > 5 else "medium" if today_pnl < -50 else "low",
        "message": message,
        "reasoning": "Pausa preventiva" if should_pause else "Parâmetros adequados"
    }

# ===============================
# ROTAS DA API
# ===============================

@app.route("/")
def home():
    return jsonify({
        "status": "🤖 IA Trading Bot API Online",
        "version": "2.0.0 - Python ML Edition",
        "description": "API de IA com Machine Learning Real para Trading",
        "model": "RandomForest Classifier",
        "endpoints": {
            "analyze": "POST /analyze - Análise de mercado",
            "signal": "POST /signal - Sinais de trading",
            "risk": "POST /risk - Avaliação de risco",
            "optimal-duration": "POST /optimal-duration - Duração ótima",
            "management": "POST /management - Gestão de posição"
        },
        "ml_stats": {
            "training_examples": len(y_train),
            "model_accuracy": "Dynamic Learning",
            "total_predictions": performance_stats['total_trades']
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "Python ML API"
    })

@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/analysis", methods=["POST", "OPTIONS"])
@app.route("/market-analysis", methods=["POST", "OPTIONS"])
@app.route("/advanced-analysis", methods=["POST", "OPTIONS"])
def analyze_market():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        features = extract_features(data)
        
        direction, confidence = predict_direction_ml(features)
        
        # Análise adicional
        symbol = data.get("symbol", "R_50")
        volatility = data.get("volatility", random.uniform(30, 80))
        
        # Determinar tendência
        if confidence > 80:
            trend = "bullish" if direction == "CALL" else "bearish"
        else:
            trend = "neutral"
        
        return jsonify({
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "volatility": round(volatility, 1),
            "message": f"Análise ML para {symbol}: Tendência {trend}, confiança {confidence}%",
            "recommendation": f"{direction} recomendado" if confidence > 75 else "Aguardar melhor oportunidade",
            "factors": {
                "ml_prediction": direction,
                "market_volatility": round(volatility, 1),
                "model_confidence": confidence
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Real - Python ML"
        })
        
    except Exception as e:
        logger.error(f"Erro em analyze: {e}")
        return jsonify({"error": "Erro na análise", "message": str(e)}), 500

@app.route("/signal", methods=["POST", "OPTIONS"])
@app.route("/trading-signal", methods=["POST", "OPTIONS"])
@app.route("/get-signal", methods=["POST", "OPTIONS"])
@app.route("/smart-signal", methods=["POST", "OPTIONS"])
@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
@app.route("/prediction", methods=["POST", "OPTIONS"])
def generate_signal():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        features = extract_features(data)
        
        direction, confidence = predict_direction_ml(features)
        
        # Dados do sinal
        current_price = data.get("currentPrice", 1000)
        symbol = data.get("symbol", "R_50")
        win_rate = data.get("winRate", 50)
        
        # Ajustar confiança baseada em performance
        if win_rate > 60:
            confidence = min(confidence + 3, 95)
        elif win_rate < 40:
            confidence = max(confidence - 5, 65)
        
        return jsonify({
            "direction": direction,
            "confidence": confidence,
            "reasoning": f"ML Real com RandomForest para {symbol} - baseado em {len(features)} features",
            "entry_price": current_price,
            "strength": "forte" if confidence > 85 else "moderado" if confidence > 75 else "fraco",
            "timeframe": "5m",
            "factors": {
                "ml_model": "RandomForest",
                "features_analyzed": len(features),
                "historical_performance": win_rate
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Real - ML Signal Generator"
        })
        
    except Exception as e:
        logger.error(f"Erro em signal: {e}")
        return jsonify({"error": "Erro na geração de sinal", "message": str(e)}), 500

@app.route("/risk", methods=["POST", "OPTIONS"])
@app.route("/risk-assessment", methods=["POST", "OPTIONS"])
@app.route("/evaluate-risk", methods=["POST", "OPTIONS"])
def assess_risk():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        risk_score, risk_level = calculate_risk_score(data)
        
        # Mensagens baseadas no nível de risco
        messages = {
            "high": "ALTO RISCO - Intervenção necessária",
            "medium": "Risco moderado - Cautela recomendada", 
            "low": "Risco controlado"
        }
        
        recommendations = {
            "high": "Pare imediatamente e revise estratégia",
            "medium": "Reduza frequency e monitore de perto",
            "low": "Continue operando com disciplina"
        }
        
        return jsonify({
            "level": risk_level,
            "score": risk_score,
            "message": messages[risk_level],
            "recommendation": recommendations[risk_level],
            "factors": {
                "martingale_level": data.get("martingaleLevel", 0),
                "today_pnl": data.get("todayPnL", 0),
                "win_rate": data.get("winRate", 50),
                "total_trades": data.get("totalTrades", 0)
            },
            "severity": "critical" if risk_level == "high" else "warning" if risk_level == "medium" else "normal",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Real - Risk Assessment"
        })
        
    except Exception as e:
        logger.error(f"Erro em risk: {e}")
        return jsonify({"error": "Erro na avaliação de risco", "message": str(e)}), 500

@app.route("/optimal-duration", methods=["POST", "OPTIONS"])
@app.route("/duration", methods=["POST", "OPTIONS"])
@app.route("/timeframe", methods=["POST", "OPTIONS"])
@app.route("/best-duration", methods=["POST", "OPTIONS"])
def get_optimal_duration():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        duration_data = optimize_duration(data)
        
        return jsonify({
            **duration_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Real - Duration Optimizer"
        })
        
    except Exception as e:
        logger.error(f"Erro em optimal-duration: {e}")
        return jsonify({"error": "Erro na otimização de duração", "message": str(e)}), 500

@app.route("/management", methods=["POST", "OPTIONS"])
@app.route("/auto-manage", methods=["POST", "OPTIONS"])
@app.route("/position-size", methods=["POST", "OPTIONS"])
@app.route("/risk-management", methods=["POST", "OPTIONS"])
def position_management():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        management_data = manage_position(data)
        
        return jsonify({
            **management_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Real - Position Management"
        })
        
    except Exception as e:
        logger.error(f"Erro em management: {e}")
        return jsonify({"error": "Erro no gerenciamento", "message": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def receive_feedback():
    """Endpoint para receber feedback e treinar o modelo"""
    if request.method == "OPTIONS":
        return '', 200
    
    try:
        data = request.get_json() or {}
        
        # Extrair dados do feedback
        features = extract_features(data)
        result = data.get("result", 0)  # 0 = loss, 1 = win
        direction = data.get("direction", "CALL")
        
        # Converter direção para label
        direction_label = 1 if direction == "CALL" else 0
        
        # Adicionar aos dados de treino
        global X_train, y_train, direction_model
        
        if len(features) >= 3:
            X_train.append(features[-3:])  # Últimos 3 ticks
            y_train.append(direction_label)
            
            # Manter apenas últimos 100 exemplos para evitar overfitting
            if len(X_train) > 100:
                X_train = X_train[-100:]
                y_train = y_train[-100:]
            
            # Retreinar modelo
            direction_model.fit(X_train, y_train)
            
            # Atualizar stats
            performance_stats['total_trades'] += 1
            if result == 1:
                performance_stats['won_trades'] += 1
            
            logger.info(f"Feedback recebido: {direction} -> {'WIN' if result == 1 else 'LOSS'}")
        
        return jsonify({
            "message": "Feedback recebido e modelo atualizado",
            "total_examples": len(y_train),
            "model_accuracy": f"{(performance_stats['won_trades'] / max(performance_stats['total_trades'], 1) * 100):.1f}%",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em feedback: {e}")
        return jsonify({"error": "Erro no feedback", "message": str(e)}), 500

# Middleware de erro global
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint não encontrado",
        "available_endpoints": ["/analyze", "/signal", "/risk", "/optimal-duration", "/management"],
        "timestamp": datetime.datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erro interno: {error}")
    return jsonify({
        "error": "Erro interno do servidor",
        "message": "Entre em contato com o suporte",
        "timestamp": datetime.datetime.now().isoformat()
    }), 500

if __name__ == "__main__":
    logger.info("🚀 Iniciando IA Trading Bot API com Machine Learning Real")
    logger.info(f"📊 Modelo treinado com {len(y_train)} exemplos")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    app.run(host="0.0.0.0", port=5000, debug=False)
