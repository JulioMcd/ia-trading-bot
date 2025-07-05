from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Libera o acesso ao backend para o frontend

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
