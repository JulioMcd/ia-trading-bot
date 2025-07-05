from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permitir acesso de outras origens, como seu HTML

@app.route("/")
def home():
    return jsonify({"status": "IA online"})

@app.route("/signal", methods=["POST", "OPTIONS"])
def trading_signal():
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
    data = request.get_json()
    return jsonify({
        "direction": "PUT",
        "confidence": 81.3,
        "reasoning": "IA Smart: padrão de rompimento identificado",
        "entry_price": data.get("currentPrice", 1000)
    })

@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
def evolutionary_signal():
    data = request.get_json()
    return jsonify({
        "direction": "CALL",
        "confidence": 88.2,
        "reasoning": "IA Evolutiva: aprendizado recente indica padrão vencedor",
        "entry_price": data.get("currentPrice", 1000)
    })

@app.route("/analyze", methods=["POST"])
def analyze_market():
    return jsonify({
        "trend": "bullish",
        "confidence": 78.9,
        "volatility": 42.3,
        "message": "Mercado em tendência de alta com alta volatilidade"
    })

@app.route("/risk", methods=["POST"])
def risk_assessment():
    return jsonify({
        "level": "medium",
        "message": "Risco moderado. Martingale em nível aceitável.",
        "recommendation": "Continuar operando"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
