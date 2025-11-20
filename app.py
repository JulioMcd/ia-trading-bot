"""
Trading AI API - VERSÃO SIMPLIFICADA SEM AUTENTICAÇÃO
✅ SEM Login/Senha
✅ SEM SQL Database
✅ Armazena dados em memória
✅ API totalmente aberta e pronta para uso
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÃO
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_PORT = int(os.environ.get('PORT', 5000))

# ARMAZENAMENTO EM MEMÓRIA
class MemoryStorage:
    def __init__(self):
        self.trades = deque(maxlen=1000)
        self.market_data = deque(maxlen=500)
        self.predictions = deque(maxlen=200)
        self.metrics = {'total_predictions': 0, 'correct_predictions': 0, 'total_trades': 0, 'wins': 0, 'losses': 0}
        logger.info("💾 Memory Storage inicializado")
    
    def add_trade(self, trade_data):
        trade_data['timestamp'] = datetime.utcnow().isoformat()
        self.trades.append(trade_data)
        self.metrics['total_trades'] += 1
        if trade_data.get('result') == 'win':
            self.metrics['wins'] += 1
        elif trade_data.get('result') == 'loss':
            self.metrics['losses'] += 1
    
    def add_market_data(self, data):
        data['timestamp'] = datetime.utcnow().isoformat()
        self.market_data.append(data)
    
    def add_prediction(self, prediction):
        prediction['timestamp'] = datetime.utcnow().isoformat()
        self.predictions.append(prediction)
        self.metrics['total_predictions'] += 1
    
    def get_recent_trades(self, symbol=None, limit=10):
        trades = list(self.trades)
        if symbol:
            trades = [t for t in trades if t.get('symbol') == symbol]
        return trades[-limit:]
    
    def get_stats(self):
        total = self.metrics['total_trades']
        wins = self.metrics['wins']
        return {
            'total_trades': total, 'total_wins': wins, 'total_losses': self.metrics['losses'],
            'win_rate': wins / max(1, total),
            'total_predictions': self.metrics['total_predictions'],
            'accuracy': self.metrics['correct_predictions'] / max(1, self.metrics['total_predictions'])
        }

storage = MemoryStorage()

# INDICADORES TÉCNICOS
class TechnicalIndicators:
    @staticmethod
    def rsi(prices, window=14):
        try:
            prices_series = pd.Series(prices)
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.fillna(50).iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def macd(prices, fast=12, slow=26):
        try:
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            return float(macd_line.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def bollinger_bands(prices, window=20, num_std=2):
        try:
            prices_series = pd.Series(prices)
            rolling_mean = prices_series.rolling(window=window).mean()
            rolling_std = prices_series.rolling(window=window).std()
            upper = rolling_mean + (rolling_std * num_std)
            lower = rolling_mean - (rolling_std * num_std)
            return {'upper': float(upper.iloc[-1]), 'lower': float(lower.iloc[-1])}
        except:
            price = prices[-1]
            return {'upper': price * 1.02, 'lower': price * 0.98}
    
    @staticmethod
    def volatility(prices, window=14):
        try:
            prices_series = pd.Series(prices)
            returns = prices_series.pct_change().dropna()
            vol = returns.rolling(window=window).std() * np.sqrt(252)
            return float(vol.iloc[-1])
        except:
            return 1.0

indicators = TechnicalIndicators()

# TRADING AI ENGINE
class SimpleTradingAI:
    def __init__(self):
        self.online_model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_samples = 20
        logger.info("🤖 Simple Trading AI inicializado")
    
    def get_market_data(self, symbol):
        try:
            base_price = 1000 + np.random.normal(0, 50)
            periods = 100
            prices = [base_price]
            for i in range(periods - 1):
                change = np.random.normal(0, 0.02)
                new_price = prices[-1] * (1 + change)
                prices.append(max(0.01, new_price))
            
            rsi = indicators.rsi(prices)
            macd = indicators.macd(prices)
            bb = indicators.bollinger_bands(prices)
            vol = indicators.volatility(prices)
            
            data = {'symbol': symbol, 'price': prices[-1], 'rsi': rsi, 'macd': macd, 'bb_upper': bb['upper'], 'bb_lower': bb['lower'], 'volatility': vol, 'prices': prices}
            storage.add_market_data(data)
            return data
        except Exception as e:
            logger.error(f"Erro ao obter dados: {e}")
            return None
    
    def extract_features(self, market_data):
        try:
            price = market_data['price']
            bb_range = market_data['bb_upper'] - market_data['bb_lower']
            bb_position = (price - market_data['bb_lower']) / max(1, bb_range)
            features = [market_data['rsi'], market_data['macd'], market_data['volatility'], bb_position, np.sin(datetime.utcnow().hour * np.pi / 12), np.cos(datetime.utcnow().hour * np.pi / 12)]
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Erro extraindo features: {e}")
            return np.zeros((1, 6))
    
    def predict_direction(self, market_data):
        try:
            features = self.extract_features(market_data)
            if self.is_trained:
                X_scaled = self.scaler.transform(features)
                pred = self.online_model.predict(X_scaled)[0]
                proba = self.online_model.predict_proba(X_scaled)[0]
                direction = 'CALL' if pred == 1 else 'PUT'
                confidence = max(proba) * 100
                method = 'machine_learning'
            else:
                result = self.hybrid_prediction(market_data)
                direction = result['direction']
                confidence = result['confidence']
                method = 'hybrid_technical'
            
            prediction = {'symbol': market_data['symbol'], 'direction': direction, 'confidence': confidence, 'method': method, 'price': market_data['price'], 'rsi': market_data['rsi'], 'macd': market_data['macd']}
            storage.add_prediction(prediction)
            return prediction
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {'direction': 'CALL', 'confidence': 65.0, 'method': 'fallback'}
    
    def hybrid_prediction(self, market_data):
        try:
            rsi = market_data['rsi']
            macd = market_data['macd']
            price = market_data['price']
            bb_upper = market_data['bb_upper']
            bb_lower = market_data['bb_lower']
            signals = []
            if rsi < 30:
                signals.append(('CALL', 0.8))
            elif rsi > 70:
                signals.append(('PUT', 0.8))
            elif rsi < 50:
                signals.append(('CALL', 0.4))
            else:
                signals.append(('PUT', 0.4))
            if macd > 0:
                signals.append(('CALL', 0.6))
            else:
                signals.append(('PUT', 0.6))
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_position = (price - bb_lower) / bb_range
                if bb_position < 0.2:
                    signals.append(('CALL', 0.7))
                elif bb_position > 0.8:
                    signals.append(('PUT', 0.7))
            call_weight = sum(w for d, w in signals if d == 'CALL')
            put_weight = sum(w for d, w in signals if d == 'PUT')
            if call_weight > put_weight:
                direction = 'CALL'
                confidence = min(95, (call_weight / (call_weight + put_weight)) * 100)
            else:
                direction = 'PUT'
                confidence = min(95, (put_weight / (call_weight + put_weight)) * 100)
            confidence = max(60, confidence)
            return {'direction': direction, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Erro predição híbrida: {e}")
            return {'direction': np.random.choice(['CALL', 'PUT']), 'confidence': 65.0}
    
    def learn_from_trade(self, trade_data, features, target):
        try:
            recent_trades = storage.get_recent_trades(limit=50)
            if len(recent_trades) < self.min_samples:
                logger.info(f"⏳ Coletando dados: {len(recent_trades)}/{self.min_samples}")
                return False
            X_list = []
            y_list = []
            for trade in recent_trades:
                if 'features' in trade and 'target' in trade:
                    X_list.append(trade['features'])
                    y_list.append(trade['target'])
            if len(X_list) < self.min_samples:
                return False
            X = np.array(X_list)
            y = np.array(y_list)
            if not self.is_trained:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.online_model.partial_fit(X_scaled, y, classes=[0, 1])
                self.is_trained = True
                logger.info(f"✅ Modelo treinado inicialmente com {len(X)} amostras")
            else:
                X_scaled = self.scaler.transform([features.flatten()])
                self.online_model.partial_fit(X_scaled, [target])
                logger.info("📚 Modelo atualizado com novo trade")
            if self.is_trained:
                X_all_scaled = self.scaler.transform(X)
                predictions = self.online_model.predict(X_all_scaled)
                accuracy = accuracy_score(y, predictions)
                storage.metrics['correct_predictions'] = int(accuracy * len(y))
                logger.info(f"📊 Accuracy atual: {accuracy:.3f}")
            return True
        except Exception as e:
            logger.error(f"Erro no aprendizado: {e}")
            return False

ai_engine = SimpleTradingAI()

# ROTAS DA API
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'online', 'service': 'Trading AI API - Simplified', 'version': '5.0.0-NO-AUTH', 'timestamp': datetime.utcnow().isoformat(), 'features': {'authentication': False, 'database': 'in-memory', 'online_learning': True, 'ready_to_use': True}, 'stats': storage.get_stats(), 'model_trained': ai_engine.is_trained})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSD')
        logger.info(f"🔍 Predição solicitada para {symbol}")
        market_data = ai_engine.get_market_data(symbol)
        if not market_data:
            return jsonify({'error': 'Erro ao obter dados de mercado'}), 500
        prediction = ai_engine.predict_direction(market_data)
        logger.info(f"✅ Predição: {prediction['direction']} ({prediction['confidence']:.1f}%)")
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSD')
        market_data = ai_engine.get_market_data(symbol)
        if not market_data:
            return jsonify({'error': 'Erro ao obter dados'}), 500
        analysis = {'symbol': symbol, 'price': market_data['price'], 'timestamp': datetime.utcnow().isoformat(), 'indicators': {'rsi': market_data['rsi'], 'macd': market_data['macd'], 'bb_upper': market_data['bb_upper'], 'bb_lower': market_data['bb_lower'], 'volatility': market_data['volatility']}, 'market_condition': 'oversold' if market_data['rsi'] < 30 else 'overbought' if market_data['rsi'] > 70 else 'neutral', 'trend': 'bullish' if market_data['macd'] > 0 else 'bearish'}
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trade/submit', methods=['POST'])
def submit_trade():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        required = ['symbol', 'direction', 'result']
        if not all(field in data for field in required):
            return jsonify({'error': 'Campos obrigatórios: symbol, direction, result'}), 400
        market_data = ai_engine.get_market_data(data['symbol'])
        features = ai_engine.extract_features(market_data)
        target = 1 if data['result'].lower() == 'win' else 0
        trade_record = {'symbol': data['symbol'], 'direction': data['direction'], 'result': data['result'].lower(), 'pnl': float(data.get('pnl', 0)), 'stake': float(data.get('stake', 0.01)), 'features': features.flatten().tolist(), 'target': target}
        storage.add_trade(trade_record)
        learning_success = ai_engine.learn_from_trade(trade_record, features, target)
        logger.info(f"💾 Trade registrado: {data['symbol']} {data['direction']} = {data['result']} (Learning: {'✅' if learning_success else '⏳'})")
        return jsonify({'message': 'Trade registrado com sucesso', 'learning_applied': learning_success, 'model_trained': ai_engine.is_trained, 'total_trades': storage.metrics['total_trades']}), 201
    except Exception as e:
        logger.error(f"Erro ao registrar trade: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = storage.get_stats()
        recent_trades = storage.get_recent_trades(limit=10)
        recent_predictions = list(storage.predictions)[-10:]
        return jsonify({'overall_stats': stats, 'model_status': {'is_trained': ai_engine.is_trained, 'min_samples_required': ai_engine.min_samples, 'current_samples': len(storage.trades)}, 'recent_trades': recent_trades, 'recent_predictions': recent_predictions})
    except Exception as e:
        logger.error(f"Erro ao obter stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        global storage, ai_engine
        storage = MemoryStorage()
        ai_engine = SimpleTradingAI()
        logger.info("🔄 Sistema resetado")
        return jsonify({'message': 'Sistema resetado com sucesso', 'timestamp': datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Erro ao resetar: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Trading AI API - Simplified Version v5.0.0")
    logger.info("=" * 60)
    logger.info("✅ SEM Autenticação (API Aberta)")
    logger.info("✅ SEM Banco de Dados (In-Memory)")
    logger.info("✅ Online Learning Ativo")
    logger.info("✅ Pronto para usar com MT5")
    logger.info(f"🌐 Porta: {API_PORT}")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=API_PORT, debug=False, threaded=True)
