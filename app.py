
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)
CORS(app)

# Simulação de dados de treino (normalmente viria de histórico real)
X_train = [[0, 1, 2], [1, 2, 3], [3, 2, 1], [2, 1, 0]]
y_train = [1, 1, 0, 0]  # 1 = CALL, 0 = PUT

# Modelo IA real (Random Forest)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

@app.route("/")
def home():
    return jsonify({"status": "IA real online"})

@app.route("/smart-signal", methods=["POST", "OPTIONS"])
def smart_signal():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    ticks = data.get("lastTicks", [1, 2, 3])[-3:]  # usa últimos 3 valores
    current_price = data.get("currentPrice", 1000)

    # Previsão com modelo IA real
    try:
        features = np.array(ticks).reshape(1, -1)
        prediction = model.predict(features)[0]
        direction = "CALL" if prediction == 1 else "PUT"
        confidence = model.predict_proba(features)[0][prediction] * 100
    except:
        direction = "WAIT"
        confidence = 0.0

    return jsonify({
        "direction": direction,
        "confidence": round(confidence, 2),
        "reasoning": "Previsão com IA real baseada nos últimos ticks",
        "entry_price": current_price
    })

@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    ticks = data.get("lastTicks", [1, 2, 3])[-3:]
    result = data.get("result")  # 1 = WIN, 0 = LOSS

    # Aprendizado online simples (adiciona novo exemplo e re-treina)
    global X_train, y_train, model
    X_train.append(ticks)
    y_train.append(result)
    model.fit(X_train, y_train)

    return jsonify({
        "message": "Feedback recebido e modelo atualizado",
        "total_examples": len(y_train)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
