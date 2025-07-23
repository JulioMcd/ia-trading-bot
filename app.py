from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# LOGIN SIMULADO
# =========================
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')
    account_type = data.get('account_type')

    if not email or not password:
        return jsonify(success=False, error="Email e senha são obrigatórios."), 400

    # Simula login bem-sucedido e retorna saldo aleatório
    print(f"🔐 Login recebido: {email} | Tipo: {account_type}")

    return jsonify({
        "success": True,
        "balance": round(random.uniform(900, 1500), 2)
    })

# =========================
# HEALTH CHECK
# =========================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Trading Bot IA API operacional',
        'timestamp': datetime.now().isoformat()
    })

# =========================
# ERROS
# =========================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint não encontrado',
        'available_endpoints': [
            '/', '/api/login', '/health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Erro interno no servidor',
        'timestamp': datetime.now().isoformat()
    }), 500

# =========================
# START
# =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("🚀 Iniciando Trading Bot IA API...")
    print(f"🌐 Porta: {port}")
    print(f"🔧 Debug: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
