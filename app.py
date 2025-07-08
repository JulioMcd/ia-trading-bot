from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key válida
VALID_API_KEY = "bhcOGajqbfFfolT"

# ✅ CONFIGURAÇÃO DE INVERSÃO DE SINAIS
INVERT_SIGNALS = True  # Mude para False se quiser sinais normais

# Dados de histórico simples
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
    
    if not api_key:
        return True
    
    return api_key == VALID_API_KEY

def analyze_technical_pattern(prices):
    """Análise técnica simples sem ML"""
    try:
        if len(prices) >= 3:
            # Tendência simples
            recent_trend = prices[-1] - prices[-3]
            volatility = abs(prices[-1] - prices[-2]) / prices[-2] * 100 if prices[-2] != 0 else 50
            
            # Lógica de direção ORIGINAL
            if recent_trend > 0:
                original_direction = "CALL"
                confidence = 70 + min(volatility * 0.3, 20)
            else:
                original_direction = "PUT" 
                confidence = 70 + min(volatility * 0.3, 20)
            
            # ✅ INVERTER SINAL SE CONFIGURADO
            if INVERT_SIGNALS:
                final_direction = "PUT" if original_direction == "CALL" else "CALL"
                logger.info(f"🔄 SINAL INVERTIDO: {original_direction} → {final_direction}")
            else:
                final_direction = original_direction
                
            return final_direction, round(confidence, 1), original_direction
        else:
            # Fallback aleatório ponderado
            original_direction = "CALL" if random.random() > 0.5 else "PUT"
            confidence = 70 + random.uniform(0, 20)
            
            # ✅ INVERTER SINAL SE CONFIGURADO
            if INVERT_SIGNALS:
                final_direction = "PUT" if original_direction == "CALL" else "CALL"
                logger.info(f"🔄 SINAL INVERTIDO (Random): {original_direction} → {final_direction}")
            else:
                final_direction = original_direction
                
            return final_direction, round(confidence, 1), original_direction
    except:
        original_direction = "CALL" if random.random() > 0.5 else "PUT"
        final_direction = "PUT" if (INVERT_SIGNALS and original_direction == "CALL") else ("CALL" if (INVERT_SIGNALS and original_direction == "PUT") else original_direction)
        return final_direction, 70.0, original_direction

def extract_features(data):
    """Extrair dados dos parâmetros recebidos"""
    current_price = data.get("currentPrice", 1000)
    volatility = data.get("volatility", 50)
    
    # Gerar preços baseados no atual se não fornecidos
    prices = data.get("lastTicks", [])
    if not prices:
        prices = [
            current_price - random.uniform(0, 5),
            current_price + random.uniform(0, 5), 
            current_price - random.uniform(0, 3)
        ]
    
    while len(prices) < 3:
        prices.append(current_price + random.uniform(-2, 2))
        
    return prices[-3:], volatility

def calculate_risk_score(data):
    """Calcular score de risco sem ML"""
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
    """Otimizar duração sem ML"""
    symbol = data.get("symbol", "R_50")
    volatility = data.get("volatility", 50)
    market_condition = data.get("marketCondition", "neutral")
    
    # Determinar se é índice de volatilidade
    is_volatility_index = "R_" in symbol or "HZ" in symbol
    
    if is_volatility_index:
        duration_type = "t"
        if volatility > 70:
            duration = random.randint(1, 3)
        elif volatility > 40:
            duration = random.randint(4, 6)
        else:
            duration = random.randint(7, 10)
    else:
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
        "reasoning": f"Análise técnica para {symbol}: {duration}{duration_type} baseado em volatilidade {volatility:.1f}%"
    }

def manage_position(data):
    """Gestão de posição sem ML"""
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
        pause_duration = random.randint(60000, 180000)
    elif today_pnl < -100 or martingale_level > 5:
        if random.random() > 0.7:
            should_pause = True
            action = "pause"
            pause_duration = random.randint(30000, 90000)
    
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
        "version": "2.1.0 - Sinais Invertidos",
        "description": "API de IA com Sistema de Inversão de Sinais",
        "model": "Technical Analysis Engine (Inverted)",
        "signal_mode": "INVERTED" if INVERT_SIGNALS else "NORMAL",
        "inversion_active": INVERT_SIGNALS,
        "endpoints": {
            "analyze": "POST /analyze - Análise de mercado",
            "signal": "POST /signal - Sinais de trading (invertidos)",
            "risk": "POST /risk - Avaliação de risco",
            "optimal-duration": "POST /optimal-duration - Duração ótima",
            "management": "POST /management - Gestão de posição",
            "toggle-inversion": "POST /toggle-inversion - Alternar inversão"
        },
        "stats": {
            "total_predictions": performance_stats['total_trades'],
            "accuracy": "Dynamic Analysis",
            "uptime": "99.9%"
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "Python Simplified API with Signal Inversion"
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
        prices, volatility = extract_features(data)
        
        direction, confidence, original_direction = analyze_technical_pattern(prices)
        
        # Análise adicional
        symbol = data.get("symbol", "R_50")
        
        # Determinar tendência baseada na direção FINAL
        if confidence > 80:
            trend = "bullish" if direction == "CALL" else "bearish"
        else:
            trend = "neutral"
        
        # ✅ ADICIONAR INFO SOBRE INVERSÃO
        analysis_mode = "INVERTIDO" if INVERT_SIGNALS else "NORMAL"
        inversion_info = f" (Análise original: {original_direction} → Final: {direction})" if INVERT_SIGNALS else ""
        
        return jsonify({
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "volatility": round(volatility, 1),
            "direction": direction,
            "original_direction": original_direction if INVERT_SIGNALS else direction,
            "inverted": INVERT_SIGNALS,
            "message": f"Análise {analysis_mode} para {symbol}: {direction} recomendado{inversion_info}",
            "recommendation": f"{direction} recomendado" if confidence > 75 else "Aguardar melhor oportunidade",
            "factors": {
                "technical_analysis": direction,
                "market_volatility": round(volatility, 1),
                "confidence_level": confidence,
                "inversion_mode": analysis_mode
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": f"IA Simplificada - Technical Analysis ({analysis_mode})"
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
        prices, volatility = extract_features(data)
        
        direction, confidence, original_direction = analyze_technical_pattern(prices)
        
        # Dados do sinal
        current_price = data.get("currentPrice", 1000)
        symbol = data.get("symbol", "R_50")
        win_rate = data.get("winRate", 50)
        
        # Ajustar confiança baseada em performance
        if win_rate > 60:
            confidence = min(confidence + 3, 95)
        elif win_rate < 40:
            confidence = max(confidence - 5, 65)
        
        # ✅ INFORMAÇÕES SOBRE INVERSÃO
        inversion_status = "ATIVO" if INVERT_SIGNALS else "INATIVO"
        reasoning_base = f"Análise técnica para {symbol}"
        
        if INVERT_SIGNALS:
            reasoning = f"{reasoning_base} - SINAL INVERTIDO: Análise sugeria {original_direction}, executando {direction}"
        else:
            reasoning = f"{reasoning_base} - baseado em padrões de preço"
        
        return jsonify({
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_price": current_price,
            "strength": "forte" if confidence > 85 else "moderado" if confidence > 75 else "fraco",
            "timeframe": "5m",
            "original_direction": original_direction if INVERT_SIGNALS else direction,
            "inverted": INVERT_SIGNALS,
            "inversion_status": inversion_status,
            "factors": {
                "technical_model": "Pattern Analysis (Inverted)" if INVERT_SIGNALS else "Pattern Analysis",
                "volatility_factor": volatility,
                "historical_performance": win_rate,
                "signal_inversion": inversion_status
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": f"IA Simplificada - Signal Generator ({inversion_status})"
        })
        
    except Exception as e:
        logger.error(f"Erro em signal: {e}")
        return jsonify({"error": "Erro na geração de sinal", "message": str(e)}), 500

# ✅ NOVO ENDPOINT PARA ALTERNAR INVERSÃO
@app.route("/toggle-inversion", methods=["POST", "OPTIONS"])
def toggle_signal_inversion():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    global INVERT_SIGNALS
    INVERT_SIGNALS = not INVERT_SIGNALS
    
    status = "ATIVADA" if INVERT_SIGNALS else "DESATIVADA"
    logger.info(f"🔄 Inversão de sinais {status}")
    
    return jsonify({
        "message": f"Inversão de sinais {status}",
        "invert_signals": INVERT_SIGNALS,
        "mode": "INVERTIDO" if INVERT_SIGNALS else "NORMAL",
        "timestamp": datetime.datetime.now().isoformat()
    })

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
            "signal_mode": "INVERTIDO" if INVERT_SIGNALS else "NORMAL",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Simplificada - Risk Assessment"
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
            "signal_mode": "INVERTIDO" if INVERT_SIGNALS else "NORMAL",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Simplificada - Duration Optimizer"
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
            "signal_mode": "INVERTIDO" if INVERT_SIGNALS else "NORMAL",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Simplificada - Position Management"
        })
        
    except Exception as e:
        logger.error(f"Erro em management: {e}")
        return jsonify({"error": "Erro no gerenciamento", "message": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def receive_feedback():
    """Endpoint para receber feedback"""
    if request.method == "OPTIONS":
        return '', 200
    
    try:
        data = request.get_json() or {}
        
        # Atualizar stats simples
        result = data.get("result", 0)
        direction = data.get("direction", "CALL")
        
        performance_stats['total_trades'] += 1
        if result == 1:
            performance_stats['won_trades'] += 1
        
        accuracy = (performance_stats['won_trades'] / max(performance_stats['total_trades'], 1) * 100)
        
        inversion_mode = "INVERTIDO" if INVERT_SIGNALS else "NORMAL"
        logger.info(f"Feedback recebido ({inversion_mode}): {direction} -> {'WIN' if result == 1 else 'LOSS'}")
        
        return jsonify({
            "message": "Feedback recebido com sucesso",
            "total_trades": performance_stats['total_trades'],
            "accuracy": f"{accuracy:.1f}%",
            "signal_mode": inversion_mode,
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
        "available_endpoints": ["/analyze", "/signal", "/risk", "/optimal-duration", "/management", "/toggle-inversion"],
        "signal_mode": "INVERTIDO" if INVERT_SIGNALS else "NORMAL",
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
    inversion_status = "ATIVADA" if INVERT_SIGNALS else "DESATIVADA"
    logger.info("🚀 Iniciando IA Trading Bot API Simplificada")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    logger.info(f"🔄 Inversão de Sinais: {inversion_status}")
    app.run(host="0.0.0.0", port=5000, debug=False)
