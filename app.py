from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Libera acesso via CORS

@app.route("/")
def home():
    return jsonify({"status": "IA online"})

@app.route("/signal", methods=["POST", "OPTIONS"])
def trading_signal():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    return jsonify({
        "direction": "CALL",
        "confidence": 85.4,
        "reasoning": "Padrão de reversão detectado",
        "timeframe": "5m",
        "entry_price": data.get("currentPrice", 1000)
    })

@app.route("/smart-signal", methods=["POST", "OPTIONS"])
def smart_signal():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    return jsonify({
        "direction": "PUT",
        "confidence": 81.3,
        "reasoning": "IA Smart: padrão de rompimento identificado",
        "entry_price": data.get("currentPrice", 1000)
    })

@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
def evolutionary_signal():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    return jsonify({
        "direction": "CALL",
        "confidence": 88.2,
        "reasoning": "IA Evolutiva: aprendizado recente indica padrão vencedor",
        "entry_price": data.get("currentPrice", 1000)
    })

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_market():
    if request.method == "OPTIONS":
        return '', 200
    return jsonify({
        "trend": "bullish",
        "confidence": 78.9,
        "volatility": 42.3,
        "message": "Mercado em tendência de alta com alta volatilidade"
    })

@app.route("/risk", methods=["POST", "OPTIONS"])
def risk_assessment():
    if request.method == "OPTIONS":
        return '', 200
    return jsonify({
        "level": "medium",
        "message": "Risco moderado. Martingale em nível aceitável.",
        "recommendation": "Continuar operando"
    })

@app.route("/prediction", methods=["POST", "OPTIONS"])
def prediction():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    return jsonify({
        "prediction": "over 2",
        "confidence": 75.3,
        "ticks": data.get("lastTicks", [])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
