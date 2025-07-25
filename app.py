from flask import Flask, request, jsonify
import requests
import random
import os
import time
import logging
from datetime import datetime, timedelta
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 📊 Configurações
SUPPORTED_ASSETS = [
    'USOUSD-OTC', 'US100-OTC', 'USDZAR-OTC', 'USDTRY-OTC',
    'USDTHB-OTC', 'USDSGD-OTC', 'USDSEK-OTC', 'USDPLN-OTC',
    'USDNOK-OTC', 'USDMXN-OTC', 'USDJPY', 'EURUSD-OTC',
    'GBPUSD-OTC', 'AUDUSD-OTC'
]

class TradingSignalGenerator:
    """Gerador de sinais de trading inteligente sem dependências pesadas"""
    
    def __init__(self):
        self.market_data_cache = {}
        self.last_signals = {}
        
    def get_market_sentiment(self, asset):
        """Simula análise de sentimento do mercado"""
        # Simular diferentes cenários baseados no ativo
        if 'USD' in asset and 'OTC' in asset:
            # Forex tem tendências mais estáveis
            base_volatility = random.uniform(0.5, 1.5)
        elif 'US100' in asset:
            # Índices têm volatilidade média
            base_volatility = random.uniform(1.0, 2.5)
        elif 'USO' in asset:
            # Commodities têm alta volatilidade
            base_volatility = random.uniform(1.5, 3.0)
        else:
            base_volatility = random.uniform(0.8, 2.0)
            
        return base_volatility
    
    def calculate_rsi_simulation(self, asset):
        """Simula cálculo de RSI baseado no ativo"""
        # Simular RSI com tendências realistas
        hour = datetime.now().hour
        
        # Horários de maior volatilidade (sessões de trading)
        if 8 <= hour <= 10 or 14 <= hour <= 16:  # Aberturas de mercado
            rsi = random.uniform(25, 75)  # Mais movimento
        else:
            rsi = random.uniform(35, 65)  # Mais estável
            
        return rsi
    
    def calculate_macd_simulation(self, asset):
        """Simula MACD"""
        signals = ['bullish', 'bearish', 'neutral']
        weights = [0.35, 0.35, 0.30]  # Distribuição realista
        return random.choices(signals, weights=weights)[0]
    
    def calculate_bollinger_simulation(self, asset):
        """Simula Bollinger Bands"""
        positions = ['overbought', 'oversold', 'middle']
        # Mercado passa mais tempo no meio
        weights = [0.15, 0.15, 0.70]
        return random.choices(positions, weights=weights)[0]
    
    def get_time_factor(self):
        """Fator baseado no horário para aumentar realismo"""
        hour = datetime.now().hour
        
        # Horários de alta atividade
        if 8 <= hour <= 10 or 14 <= hour <= 16:
            return 1.2  # Aumenta confiança
        elif 22 <= hour or hour <= 6:
            return 0.8  # Diminui confiança (baixa liquidez)
        else:
            return 1.0
    
    def generate_signal(self, asset):
        """Gera um sinal completo para o ativo"""
        try:
            # Validar ativo
            if asset not in SUPPORTED_ASSETS:
                return self._error_response(f"Asset {asset} not supported")
            
            # Simular análise técnica
            rsi = self.calculate_rsi_simulation(asset)
            macd = self.calculate_macd_simulation(asset)
            bollinger = self.calculate_bollinger_simulation(asset)
            volatility = self.get_market_sentiment(asset)
            time_factor = self.get_time_factor()
            
            # Lógica de decisão inteligente
            score_call = 0
            score_put = 0
            
            # RSI Analysis
            if rsi < 30:
                score_call += 25  # Oversold = BUY
            elif rsi > 70:
                score_put += 25   # Overbought = SELL
            elif 40 <= rsi <= 60:
                score_call += 10
                score_put += 10
            
            # MACD Analysis
            if macd == 'bullish':
                score_call += 20
            elif macd == 'bearish':
                score_put += 20
            else:
                score_call += 5
                score_put += 5
            
            # Bollinger Bands Analysis
            if bollinger == 'oversold':
                score_call += 20
            elif bollinger == 'overbought':
                score_put += 20
            else:
                score_call += 10
                score_put += 10
            
            # Volatility factor
            if volatility > 2.0:
                # High volatility = reduce confidence
                score_call = max(0, score_call - 10)
                score_put = max(0, score_put - 10)
            elif volatility < 1.0:
                # Low volatility = increase confidence
                score_call += 5
                score_put += 5
            
            # Apply time factor
            score_call *= time_factor
            score_put *= time_factor
            
            # Determine direction and confidence
            if score_call > score_put:
                direction = 'call'
                confidence = min(95, 60 + (score_call - score_put))
            elif score_put > score_call:
                direction = 'put'
                confidence = min(95, 60 + (score_put - score_call))
            else:
                # Tie-breaker with slight randomness
                direction = random.choice(['call', 'put'])
                confidence = random.randint(65, 75)
            
            # Optimal timeframe based on volatility
            if volatility > 2.5:
                duration = 1  # High volatility = short timeframe
            elif volatility > 1.5:
                duration = 2  # Medium volatility = medium timeframe
            else:
                duration = 3  # Low volatility = longer timeframe
            
            # Generate reasoning
            reasoning_parts = []
            if rsi < 30:
                reasoning_parts.append("RSI oversold signal")
            elif rsi > 70:
                reasoning_parts.append("RSI overbought signal")
            
            if macd != 'neutral':
                reasoning_parts.append(f"MACD {macd} trend")
            
            if bollinger != 'middle':
                reasoning_parts.append(f"Bollinger {bollinger}")
            
            if volatility > 2.0:
                reasoning_parts.append("high volatility detected")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Mixed technical signals"
            
            # Current price simulation (realistic for each asset type)
            if 'USD' in asset and asset != 'USOUSD-OTC':
                current_price = round(random.uniform(0.8, 1.5), 5)
            elif 'US100' in asset:
                current_price = round(random.uniform(15000, 16000), 2)
            elif 'USOUSD' in asset:
                current_price = round(random.uniform(70, 90), 2)
            else:
                current_price = round(random.uniform(0.5, 2.0), 5)
            
            # Price change simulation
            price_change = round(random.uniform(-2.0, 2.0), 2)
            
            # Trend analysis
            if score_call > score_put + 15:
                trend = 'uptrend'
            elif score_put > score_call + 15:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Build response
            signal_data = {
                'status': 'success',
                'direction': direction,
                'confidence': round(confidence),
                'reasoning': reasoning,
                'signal_score': f"{round(score_call)}-{round(score_put)}",
                'optimal_timeframe': {
                    'type': 'minutes',
                    'duration': duration
                },
                'market_analysis': {
                    'current_price': current_price,
                    'price_change_percent': price_change,
                    'volatility': round(volatility, 2),
                    'trend': trend
                },
                'technical_indicators': {
                    'rsi': round(rsi, 1),
                    'macd_signal': macd,
                    'bollinger_position': bollinger,
                    'stochastic_signal': random.choice(['overbought', 'oversold', 'neutral']),
                    'ema_signal': random.choice(['bullish', 'bearish', 'neutral'])
                },
                'timestamp': datetime.now().isoformat(),
                'symbol': asset,
                'api_version': '2.0-lite'
            }
            
            # Cache last signal
            self.last_signals[asset] = signal_data
            
            logger.info(f"Signal generated for {asset}: {direction} ({confidence}%)")
            return signal_data
            
        except Exception as e:
            logger.error(f"Error generating signal for {asset}: {e}")
            return self._error_response(f"Error generating signal: {str(e)}")
    
    def _error_response(self, message):
        """Retorna resposta de erro padronizada"""
        return {
            'status': 'error',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

# Instância do gerador
signal_generator = TradingSignalGenerator()

@app.route('/', methods=['GET'])
def home():
    """Endpoint home da API"""
    return jsonify({
        'status': 'online',
        'message': 'Trading Bot API v2.0-lite - Simplified Analysis',
        'features': [
            'Multi-asset analysis',
            'Technical indicators simulation',
            'Real-time signal generation',
            'Risk management',
            'No heavy dependencies'
        ],
        'supported_assets': SUPPORTED_ASSETS,
        'total_assets': len(SUPPORTED_ASSETS),
        'timestamp': datetime.now().isoformat(),
        'version': '2.0-lite'
    })

@app.route('/signal', methods=['POST'])
def get_signal():
    """Endpoint principal para obter sinais de trading"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Symbol parameter is required',
                'example': {'symbol': 'EURUSD-OTC'}
            }), 400
        
        symbol = data['symbol']
        
        # Gerar sinal
        signal = signal_generator.generate_signal(symbol)
        
        return jsonify(signal)
        
    except Exception as e:
        logger.error(f"Error in /signal endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'OK',
        'version': '2.0-lite',
        'memory_usage': 'low',
        'dependencies': 'minimal'
    })

@app.route('/assets', methods=['GET'])
def list_assets():
    """Lista ativos suportados"""
    return jsonify({
        'status': 'success',
        'supported_assets': SUPPORTED_ASSETS,
        'total_assets': len(SUPPORTED_ASSETS),
        'categories': {
            'forex_otc': [a for a in SUPPORTED_ASSETS if 'USD' in a and 'OTC' in a and a != 'USOUSD-OTC'],
            'commodities': ['USOUSD-OTC'],
            'indices': ['US100-OTC'],
            'forex_regular': ['USDJPY']
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test/<symbol>', methods=['GET'])
def test_signal(symbol):
    """Endpoint de teste para um símbolo específico"""
    try:
        if symbol not in SUPPORTED_ASSETS:
            return jsonify({
                'status': 'error',
                'message': f'Symbol {symbol} not supported',
                'supported_assets': SUPPORTED_ASSETS
            }), 400
        
        # Gerar sinal de teste
        signal = signal_generator.generate_signal(symbol)
        
        return jsonify({
            'status': 'test_success',
            'signal': signal,
            'note': 'This is a test signal for development purposes'
        })
        
    except Exception as e:
        logger.error(f"Error in test endpoint for {symbol}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Test failed',
            'details': str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def get_batch_signals():
    """Endpoint para múltiplos sinais simultâneos"""
    try:
        data = request.get_json()
        
        if not data or 'symbols' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Symbols array parameter is required',
                'example': {'symbols': ['EURUSD-OTC', 'USDJPY', 'US100-OTC']}
            }), 400
        
        symbols = data['symbols']
        
        if not isinstance(symbols, list) or len(symbols) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Symbols must be a non-empty array'
            }), 400
        
        if len(symbols) > 5:
            return jsonify({
                'status': 'error',
                'message': 'Maximum 5 symbols per batch request'
            }), 400
        
        # Gerar sinais para todos os símbolos
        signals = {}
        for symbol in symbols:
            if symbol in SUPPORTED_ASSETS:
                signals[symbol] = signal_generator.generate_signal(symbol)
            else:
                signals[symbol] = {
                    'status': 'error',
                    'message': f'Symbol {symbol} not supported'
                }
        
        return jsonify({
            'status': 'success',
            'signals': signals,
            'processed_count': len(signals),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Batch processing failed',
            'details': str(e)
        }), 500

# Configurações para o Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Trading API v2.0-lite on port {port}")
    logger.info(f"Supported assets: {len(SUPPORTED_ASSETS)}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
