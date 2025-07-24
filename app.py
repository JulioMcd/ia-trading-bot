from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os

app = Flask(__name__)
CORS(app)

# ===============================================
# IA REAL ULTRA SIMPLES (SEM PANDAS)
# ===============================================

class SimpleTechnicalAnalysis:
    def __init__(self):
        self.symbol_mapping = {
            'EURUSD-OTC': 'EURUSD=X',
            'GBPUSD-OTC': 'GBPUSD=X',  
            'USDJPY-OTC': 'USDJPY=X',
            'AUDUSD-OTC': 'AUDUSD=X',
            'USDCAD-OTC': 'USDCAD=X',
            'USDCHF-OTC': 'USDCHF=X',
            'EURJPY-OTC': 'EURJPY=X',
            'EURGBP-OTC': 'EURGBP=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD'
        }
    
    def get_real_data(self, symbol):
        """Obtém dados reais usando yfinance"""
        try:
            yahoo_symbol = self.symbol_mapping.get(symbol, symbol)
            print(f"📊 Buscando dados REAIS para {yahoo_symbol}...")
            
            # Criar ticker
            ticker = yf.Ticker(yahoo_symbol)
            
            # Obter dados das últimas 48 horas com intervalos de 5 min
            hist = ticker.history(period="2d", interval="5m")
            
            if hist.empty:
                print(f"❌ Sem dados para {symbol}")
                return None
            
            # Converter para listas simples
            prices = hist['Close'].tolist()
            highs = hist['High'].tolist()  
            lows = hist['Low'].tolist()
            volumes = hist['Volume'].tolist() if 'Volume' in hist.columns else [0] * len(prices)
            
            if len(prices) < 20:
                print(f"⚠️ Poucos dados ({len(prices)}) para {symbol}")
                return None
            
            print(f"✅ {len(prices)} velas REAIS obtidas para {symbol}")
            print(f"📈 Preço atual: {prices[-1]:.5f}")
            print(f"📊 Variação: {((prices[-1] - prices[-2]) / prices[-2] * 100):.2f}%")
            
            return {
                'prices': prices,
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'current_price': prices[-1],
                'symbol': symbol
            }
            
        except Exception as e:
            print(f"❌ Erro ao obter dados para {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI real"""
        try:
            if len(prices) < period + 1:
                return 50
            
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return max(0, min(100, rsi))
            
        except:
            return 50
    
    def calculate_sma(self, prices, period):
        """Média móvel simples"""
        try:
            if len(prices) < period:
                return prices[-1]
            return sum(prices[-period:]) / period
        except:
            return prices[-1] if prices else 0
    
    def calculate_volatility(self, prices, highs, lows):
        """Calcula volatilidade real"""
        try:
            if len(prices) < 2:
                return 1.0
            
            # True Range simplificado
            current_price = prices[-1]
            prev_price = prices[-2]
            current_high = highs[-1]
            current_low = lows[-1]
            
            tr = max(
                current_high - current_low,
                abs(current_high - prev_price),
                abs(current_low - prev_price)
            )
            
            volatility_percent = (tr / current_price) * 100
            return max(0.1, volatility_percent)
            
        except:
            return 1.0
    
    def analyze_trend(self, prices):
        """Análise de tendência simples mas real"""
        try:
            if len(prices) < 20:
                return {'trend': 'sideways', 'strength': 0.5}
            
            # Médias móveis
            sma_5 = sum(prices[-5:]) / 5
            sma_10 = sum(prices[-10:]) / 10
            sma_20 = sum(prices[-20:]) / 20
            current = prices[-1]
            
            # Análise de tendência
            trend_score = 0
            
            if current > sma_5 > sma_10 > sma_20:
                trend_score = 2  # Forte alta
            elif current > sma_5 > sma_10:
                trend_score = 1  # Alta moderada
            elif current < sma_5 < sma_10 < sma_20:
                trend_score = -2  # Forte baixa
            elif current < sma_5 < sma_10:
                trend_score = -1  # Baixa moderada
            else:
                trend_score = 0  # Lateral
            
            if trend_score >= 1:
                trend = 'bullish'
            elif trend_score <= -1:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            strength = abs(trend_score) / 2
            
            return {
                'trend': trend,
                'strength': strength,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20
            }
            
        except:
            return {'trend': 'sideways', 'strength': 0.5}
    
    def generate_signal(self, symbol):
        """Gera sinal baseado em análise técnica real"""
        try:
            print(f"\n🤖 Iniciando análise REAL para {symbol}")
            
            # Obter dados reais
            data = self.get_real_data(symbol)
            if not data:
                return self._error_response("Dados não disponíveis")
            
            prices = data['prices']
            highs = data['highs']
            lows = data['lows']
            current_price = data['current_price']
            
            # Calcular indicadores REAIS
            rsi = self.calculate_rsi(prices)
            sma_10 = self.calculate_sma(prices, 10)
            sma_20 = self.calculate_sma(prices, 20)
            volatility = self.calculate_volatility(prices, highs, lows)
            trend_analysis = self.analyze_trend(prices)
            
            print(f"📊 RSI real: {rsi:.1f}")
            print(f"📈 SMA10: {sma_10:.5f} | SMA20: {sma_20:.5f}")
            print(f"📉 Volatilidade: {volatility:.2f}%")
            print(f"📊 Tendência: {trend_analysis['trend']}")
            
            # Gerar sinal baseado em indicadores reais
            signal_score = 0
            reasons = []
            
            # RSI Analysis
            if rsi < 30:
                signal_score += 2
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signal_score -= 2
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif 45 <= rsi <= 55:
                reasons.append(f"RSI neutro ({rsi:.1f})")
            
            # Price vs SMA Analysis
            if current_price > sma_10 > sma_20:
                signal_score += 1.5
                reasons.append("Preço acima das médias")
            elif current_price < sma_10 < sma_20:
                signal_score -= 1.5
                reasons.append("Preço abaixo das médias")
            
            # Trend Analysis
            if trend_analysis['trend'] == 'bullish':
                signal_score += trend_analysis['strength']
                reasons.append("Tendência de alta")
            elif trend_analysis['trend'] == 'bearish':
                signal_score -= trend_analysis['strength']
                reasons.append("Tendência de baixa")
            
            # Volatility Analysis
            if volatility > 2.0:
                reasons.append(f"Alta volatilidade ({volatility:.1f}%)")
            elif volatility < 0.5:
                reasons.append(f"Baixa volatilidade ({volatility:.1f}%)")
            
            # Price momentum (últimas 3 velas)
            if len(prices) >= 3:
                recent_change = ((prices[-1] - prices[-3]) / prices[-3]) * 100
                if recent_change > 0.1:
                    signal_score += 0.5
                    reasons.append(f"Momentum positivo ({recent_change:.2f}%)")
                elif recent_change < -0.1:
                    signal_score -= 0.5
                    reasons.append(f"Momentum negativo ({recent_change:.2f}%)")
            
            # Determinar direção e confiança
            if signal_score >= 1:
                direction = "call"
                confidence = min(95, 70 + abs(signal_score) * 10)
            elif signal_score <= -1:
                direction = "put"
                confidence = min(95, 70 + abs(signal_score) * 10)
            else:
                direction = "call" if signal_score > 0 else "put"
                confidence = max(60, 70 - abs(1 - abs(signal_score)) * 15)
            
            # Timeframe baseado na volatilidade real
            if volatility > 2.0:
                timeframe = {"type": "minutes", "duration": 1}
            elif volatility > 1.0:
                timeframe = {"type": "minutes", "duration": 2}
            else:
                timeframe = {"type": "minutes", "duration": 3}
            
            reasoning = " | ".join(reasons[:3]) if reasons else "Análise técnica baseada em dados reais"
            
            print(f"✅ Sinal: {direction.upper()}")
            print(f"🎯 Confiança: {confidence:.1f}%")
            print(f"📝 Razões: {reasoning}")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'direction': direction,
                'confidence': round(confidence, 1),
                'signal_score': round(signal_score, 2),
                'reasoning': reasoning,
                'market_analysis': {
                    'current_price': round(current_price, 5),
                    'price_change_percent': round(((prices[-1] - prices[-2]) / prices[-2] * 100), 2) if len(prices) > 1 else 0,
                    'volatility': round(volatility, 2),
                    'trend': trend_analysis['trend'],
                    'trend_strength': round(trend_analysis['strength'], 2),
                    'candles_analyzed': len(prices)
                },
                'technical_indicators': {
                    'rsi': round(rsi, 1),
                    'sma_10': round(sma_10, 5),
                    'sma_20': round(sma_20, 5),
                    'price_vs_sma10': 'above' if current_price > sma_10 else 'below',
                    'price_vs_sma20': 'above' if current_price > sma_20 else 'below'
                },
                'optimal_timeframe': timeframe,
                'data_source': 'Yahoo Finance Real-Time',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Erro geral: {e}")
            return self._error_response(f"Erro: {str(e)}")
    
    def _error_response(self, message):
        return {
            'status': 'error',
            'message': message,
            'direction': 'call',
            'confidence': 50,
            'reasoning': 'Análise indisponível - usando fallback',
            'timestamp': datetime.now().isoformat()
        }

# ===============================================
# INSTÂNCIA GLOBAL
# ===============================================

analyzer = SimpleTechnicalAnalysis()

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': '🤖 IA TRADING REAL - Ultra Simples',
        'version': '4.0.0 - COMPATIBLE',
        'features': [
            '✅ Yahoo Finance - Dados reais',
            '✅ RSI calculado com dados reais',
            '✅ Médias móveis reais (SMA 10, 20)',
            '✅ Análise de tendência verdadeira',
            '✅ Volatilidade calculada (True Range)',
            '✅ 100% compatível com Render',
            '✅ Sem dependências problemáticas'
        ],
        'supported_symbols': list(analyzer.symbol_mapping.keys()),
        'data_source': 'Yahoo Finance Real Data',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signal', methods=['POST', 'GET'])
@app.route('/trading-signal', methods=['POST', 'GET'])
def get_signal():
    """Endpoint principal - IA REAL"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
    
    print(f"\n🔄 Nova requisição REAL para {symbol}")
    
    # Validar símbolo
    if symbol not in analyzer.symbol_mapping:
        return jsonify({
            'status': 'error',
            'message': f'Símbolo {symbol} não suportado',
            'supported_symbols': list(analyzer.symbol_mapping.keys())
        }), 400
    
    # Gerar sinal REAL
    result = analyzer.generate_signal(symbol)
    return jsonify(result)

@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    """Análise detalhada"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
    
    try:
        data = analyzer.get_real_data(symbol)
        if not data:
            return jsonify({'status': 'error', 'message': 'Dados indisponíveis'}), 500
        
        prices = data['prices']
        current_price = data['current_price']
        
        rsi = analyzer.calculate_rsi(prices)
        sma_10 = analyzer.calculate_sma(prices, 10)
        sma_20 = analyzer.calculate_sma(prices, 20)
        volatility = analyzer.calculate_volatility(data['prices'], data['highs'], data['lows'])
        trend = analyzer.analyze_trend(prices)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'market_data': {
                'current_price': round(current_price, 5),
                'candles_analyzed': len(prices),
                'volatility': round(volatility, 2)
            },
            'indicators': {
                'rsi': round(rsi, 1),
                'sma_10': round(sma_10, 5),
                'sma_20': round(sma_20, 5)
            },
            'trend_analysis': trend,
            'data_source': 'Yahoo Finance',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': '🟢 IA REAL Ultra Simples Online',
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'data_source': 'Yahoo Finance',
        'timestamp': datetime.now().isoformat()
    })

# ===============================================
# INICIALIZAÇÃO
# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 IA REAL Ultra Simples Iniciando...")
    print("📊 Fonte: Yahoo Finance")
    print("⚙️ Indicadores: RSI, SMA, Volatilidade, Tendência")
    print("✅ 100% compatível com Python 3.13")
    print("🎯 Análise técnica real sem dependências problemáticas!")
    print(f"🌐 Porta: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
