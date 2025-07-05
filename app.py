
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)
CORS(app)

# Dados de treino simulados
X_train = [[0, 1, 2], [1, 2, 3], [3, 2, 1], [2, 1, 0]]
y_train = [1, 1, 0, 0]  # 1 = CALL, 0 = PUT

# Modelo IA real (Random Forest)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

def predict_direction(ticks):
    try:
        features = np.array(ticks[-3:]).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction] * 100
        direction = "CALL" if prediction == 1 else "PUT"
        return direction, round(confidence, 2)
    except:
        return "WAIT", 0.0

@app.route("/")
def home():
    return jsonify({"status": "IA real online"})

@app.route("/smart-signal", methods=["POST", "OPTIONS"])
@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
@app.route("/prediction", methods=["POST", "OPTIONS"])
@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/advanced-analysis", methods=["POST", "OPTIONS"])
def all_signals():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    ticks = data.get("lastTicks", [1, 2, 3])
    current_price = data.get("currentPrice", 1000)

    direction, confidence = predict_direction(ticks)

    return jsonify({
        "direction": direction,
        "confidence": confidence,
        "reasoning": "IA real com RandomForest baseada nos últimos ticks",
        "entry_price": current_price
    })

@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    ticks = data.get("lastTicks", [1, 2, 3])
    result = data.get("result", 0)

    global X_train, y_train, model
    X_train.append(ticks[-3:])
    y_train.append(result)
    model.fit(X_train, y_train)

    return jsonify({
        "message": "Feedback recebido e IA atualizada",
        "total_examples": len(y_train)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
