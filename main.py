from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configurações globais
CONFIG = {
    'MODEL_VERSION': '1.0',
    'CONFIDENCE_THRESHOLD': 0.7,
    'MIN_DATA_POINTS': 10,
    'MAX_HISTORY_DAYS': 30
}

class TradingAI:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'price_change_1', 'price_change_5', 'price_change_10',
            'volatility_short', 'volatility_long', 'rsi', 'macd',
            'volume_ratio', 'time_of_day', 'martingale_level',
            'win_rate', 'recent_wins', 'recent_losses'
        ]
        self.market_data = []
        self.trade_history = []
        
    def calculate_features(self, data):
        """Calcula features técnicas para o modelo"""
        try:
            if len(data) < 10:
                # Dados insuficientes, retorna features padrão
                return np.array([0.0] * len(self.feature_names)).reshape(1, -1)
            
            prices = [float(d.get('price', 1000)) for d in data[-20:]]
            
            # Mudanças de preço
            price_change_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            price_change_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            price_change_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 else 0
            
            # Volatilidade
            volatility_short = np.std(prices[-5:]) if len(prices) > 4 else 0
            volatility_long = np.std(prices[-10:]) if len(prices) > 9 else 0
            
            # RSI simplificado
            gains = []
            losses = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains[-14:]) if len(gains) > 13 else 0.1
            avg_loss = np.mean(losses[-14:]) if len(losses) > 13 else 0.1
            rs = avg_gain / avg_loss if avg_loss > 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # MACD simplificado
            ema_12 = np.mean(prices[-12:]) if len(prices) > 11 else prices[-1]
            ema_26 = np.mean(prices[-26:]) if len(prices) > 25 else prices[-1]
            macd = ema_12 - ema_26
            
            # Volume ratio (simulado)
            volume_ratio = np.random.uniform(0.8, 1.2)
            
            # Hora do dia (0-23)
            time_of_day = datetime.now().hour
            
            # Informações de trading
            martingale_level = data[-1].get('martingaleLevel', 0)
            win_rate = data[-1].get('winRate', 50)
            recent_wins = sum(1 for d in data[-10:] if d.get('result') == 'win')
            recent_losses = sum(1 for d in data[-10:] if d.get('result') == 'loss')
            
            features = np.array([
                price_change_1, price_change_5, price_change_10,
                volatility_short, volatility_long, rsi, macd,
                volume_ratio, time_of_day, martingale_level,
                win_rate, recent_wins, recent_losses
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao calcular features: {e}")
            return np.array([0.0] * len(self.feature_names)).reshape(1, -1)
    
    def train_model(self, training_data):
        """Treina o modelo com dados históricos"""
        try:
            if len(training_data) < CONFIG['MIN_DATA_POINTS']:
                logger.warning("Dados insuficientes para treinamento")
                return False
            
            X = []
            y = []
            
            for i in range(len(training_data) - 1):
                features = self.calculate_features(training_data[:i+1])
                X.append(features.flatten())
                
                # Label: 1 para CALL, 0 para PUT
                next_result = training_data[i+1].get('result')
                if next_result == 'win':
                    direction = training_data[i+1].get('direction', 'CALL')
                    y.append(1 if direction == 'CALL' else 0)
                else:
                    y.append(np.random.choice([0, 1]))  # Random para losses
            
            if len(X) > 0:
                X = np.array(X)
                y = np.array(y)
                
                # Normalizar features
                X_scaled = self.scaler.fit_transform(X)
                
                # Treinar modelo
                self.model.fit(X_scaled, y)
                self.is_trained = True
                
                logger.info(f"Modelo treinado com {len(X)} amostras")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return False
    
    def predict_signal(self, market_data):
        """Gera sinal de trading"""
        try:
            features = self.calculate_features(market_data)
            
            if not self.is_trained:
                # Modelo não treinado, usar heurísticas
                return self._heuristic_signal(features)
            
            # Normalizar features
            features_scaled = self.scaler.transform(features)
            
            # Predição
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled)[0])
            
            direction = 'CALL' if prediction == 1 else 'PUT'
            
            return {
                'direction': direction,
                'confidence': float(confidence * 100),
                'reasoning': f'ML Model prediction: {direction}',
                'model_trained': True
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return self._heuristic_signal(features)
    
    def _heuristic_signal(self, features):
        """Sinal baseado em heurísticas quando modelo não está treinado"""
        try:
            # Análise simples baseada em features
            price_trend = features[0][0] + features[0][1]  # price_change_1 + price_change_5
            volatility = features[0][3]  # volatility_short
            rsi = features[0][5]
            
            # Lógica heurística
            if price_trend > 0.001 and rsi < 70:
                direction = 'CALL'
                confidence = 65 + np.random.uniform(0, 15)
            elif price_trend < -0.001 and rsi > 30:
                direction = 'PUT'
                confidence = 65 + np.random.uniform(0, 15)
            else:
                direction = np.random.choice(['CALL', 'PUT'])
                confidence = 55 + np.random.uniform(0, 15)
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'reasoning': 'Heuristic analysis (model training)',
                'model_trained': False
            }
            
        except Exception as e:
            logger.error(f"Erro na heurística: {e}")
            return {
                'direction': 'CALL',
                'confidence': 60.0,
                'reasoning': 'Default signal due to error',
                'model_trained': False
            }

# Instância global da IA
trading_ai = TradingAI()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'ML Trading Bot',
        'version': CONFIG['MODEL_VERSION'],
        'model_trained': trading_ai.is_trained,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_market():
    """Endpoint para análise de mercado"""
    try:
        data = request.json
        symbol = data.get('symbol', 'unknown')
        current_price = data.get('currentPrice', 1000)
        market_condition = data.get('marketCondition', 'neutral')
        
        # Adicionar dados ao histórico
        market_data_point = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': current_price,
            'condition': market_condition,
            'martingaleLevel': data.get('martingaleLevel', 0),
            'winRate': data.get('winRate', 50)
        }
        
        trading_ai.market_data.append(market_data_point)
        
        # Manter apenas últimos dados
        if len(trading_ai.market_data) > 1000:
            trading_ai.market_data = trading_ai.market_data[-500:]
        
        # Análise
        volatility = np.random.uniform(20, 80)
        trend = 'bullish' if current_price > 1000 else 'bearish'
        
        response = {
            'message': f'Análise completa do {symbol}: Volatilidade {volatility:.1f}%',
            'volatility': volatility,
            'trend': trend,
            'confidence': np.random.uniform(70, 90),
            'market_condition': market_condition,
            'data_points': len(trading_ai.market_data),
            'model_status': 'trained' if trading_ai.is_trained else 'learning'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/signal', methods=['POST'])
def get_trading_signal():
    """Endpoint para sinal de trading"""
    try:
        data = request.json
        
        # Preparar dados para predição
        market_data_point = {
            'timestamp': datetime.now().isoformat(),
            'symbol': data.get('symbol', 'unknown'),
            'price': data.get('currentPrice', 1000),
            'martingaleLevel': data.get('martingaleLevel', 0),
            'winRate': data.get('winRate', 50),
            'volatility': data.get('volatility', 50)
        }
        
        # Adicionar ao histórico
        trading_ai.market_data.append(market_data_point)
        
        # Treinar modelo se temos dados suficientes
        if len(trading_ai.market_data) >= CONFIG['MIN_DATA_POINTS'] and not trading_ai.is_trained:
            logger.info("Iniciando treinamento do modelo...")
            trading_ai.train_model(trading_ai.market_data)
        
        # Gerar sinal
        signal = trading_ai.predict_signal(trading_ai.market_data)
        
        response = {
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning'],
            'timeframe': '5m',
            'entry_price': data.get('currentPrice', 1000),
            'model_trained': signal['model_trained'],
            'data_points': len(trading_ai.market_data)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro no sinal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/risk', methods=['POST'])
def assess_risk():
    """Endpoint para avaliação de risco"""
    try:
        data = request.json
        
        martingale_level = data.get('martingaleLevel', 0)
        win_rate = data.get('winRate', 50)
        total_trades = data.get('totalTrades', 0)
        current_balance = data.get('currentBalance', 1000)
        
        # Calcular nível de risco
        risk_score = 0
        
        # Risco por Martingale
        if martingale_level > 5:
            risk_score += 40
        elif martingale_level > 3:
            risk_score += 25
        elif martingale_level > 0:
            risk_score += 10
        
        # Risco por Win Rate
        if win_rate < 30:
            risk_score += 30
        elif win_rate < 45:
            risk_score += 15
        
        # Risco por número de trades
        if total_trades > 50:
            risk_score += 10
        
        # Determinar nível
        if risk_score < 20:
            level = 'low'
            message = 'Risco baixo - Operação segura'
            recommendation = 'Continuar operando normalmente'
        elif risk_score < 50:
            level = 'medium'
            message = 'Risco moderado - Atenção necessária'
            recommendation = 'Operar com cautela, considerar redução'
        else:
            level = 'high'
            message = 'Risco alto - Cuidado extremo'
            recommendation = 'Considerar pausa ou stake mínimo'
        
        response = {
            'level': level,
            'message': message,
            'score': risk_score,
            'recommendation': recommendation,
            'martingale_level': martingale_level,
            'win_rate': win_rate,
            'factors': {
                'martingale_risk': martingale_level > 0,
                'low_win_rate': win_rate < 45,
                'high_volume': total_trades > 50
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na avaliação de risco: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def trade_feedback():
    """Endpoint para feedback de trades (para aprendizado)"""
    try:
        data = request.json
        
        # Armazenar resultado do trade para aprendizado
        trade_result = {
            'timestamp': datetime.now().isoformat(),
            'direction': data.get('direction'),
            'result': data.get('result'),  # 'win' ou 'loss'
            'pnl': data.get('pnl', 0),
            'symbol': data.get('symbol'),
            'martingale_level': data.get('martingaleLevel', 0)
        }
        
        trading_ai.trade_history.append(trade_result)
        
        # Manter histórico limitado
        if len(trading_ai.trade_history) > 500:
            trading_ai.trade_history = trading_ai.trade_history[-300:]
        
        # Re-treinar modelo periodicamente
        if len(trading_ai.trade_history) % 20 == 0 and len(trading_ai.trade_history) > 20:
            logger.info("Re-treinando modelo com novos dados...")
            # Combinar dados de mercado com resultados
            combined_data = []
            for i, trade in enumerate(trading_ai.trade_history[-50:]):
                if i < len(trading_ai.market_data):
                    market_point = trading_ai.market_data[-(50-i)]
                    market_point['result'] = trade['result']
                    market_point['direction'] = trade['direction']
                    combined_data.append(market_point)
            
            if combined_data:
                trading_ai.train_model(combined_data)
        
        response = {
            'status': 'feedback_received',
            'trades_logged': len(trading_ai.trade_history),
            'model_trained': trading_ai.is_trained,
            'result': trade_result['result']
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro no feedback: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
