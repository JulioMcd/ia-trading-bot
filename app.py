from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import os
from typing import Dict, List, Tuple, Optional

app = Flask(__name__)
CORS(app)

# ===============================================
# CLASSE DE ANÁLISE TÉCNICA REAL (SEM TA-LIB)
# ===============================================

class RealTechnicalAnalysis:
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
            'AUDCAD-OTC': 'AUDCAD=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD'
        }
    
    def get_real_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Obtém dados reais do mercado usando Yahoo Finance"""
        try:
            yahoo_symbol = self.symbol_mapping.get(symbol, symbol)
            print(f"📊 Buscando dados reais para {yahoo_symbol}...")
            
            # Baixar dados dos últimos 5 dias com intervalos de 5 minutos
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period="5d", interval="5m")
            
            if data.empty:
                print(f"❌ Dados não encontrados para {symbol}")
                # Tentar período menor
                data = ticker.history(period="1d", interval="1m")
            
            if len(data) < 20:
                print(f"⚠️ Poucos dados ({len(data)} velas) para {symbol}")
                return None
                
            print(f"✅ Obtidos {len(data)} dados reais para {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Erro ao obter dados para {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI manualmente"""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))
        except:
            return 50
    
    def calculate_sma(self, prices, period):
        """Calcula Média Móvel Simples"""
        try:
            if len(prices) < period:
                return prices[-1]
            return np.mean(prices[-period:])
        except:
            return prices[-1] if len(prices) > 0 else 0
    
    def calculate_ema(self, prices, period):
        """Calcula Média Móvel Exponencial"""
        try:
            if len(prices) < period:
                return prices[-1]
            
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
        except:
            return prices[-1] if len(prices) > 0 else 0
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcula Bandas de Bollinger"""
        try:
            if len(prices) < period:
                middle = np.mean(prices)
                std = np.std(prices)
            else:
                middle = np.mean(prices[-period:])
                std = np.std(prices[-period:])
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return upper, middle, lower
        except:
            current = prices[-1] if len(prices) > 0 else 1
            return current * 1.01, current, current * 0.99
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        try:
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)
            
            macd_line = ema_fast - ema_slow
            
            # Simular signal line (seria EMA do MACD)
            signal_line = macd_line * 0.8  # Simplificado
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except:
            return 0, 0, 0
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calcula indicadores técnicos reais"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            
            current_price = close[-1]
            
            # RSI
            rsi = self.calculate_rsi(close)
            
            # Médias Móveis
            sma_10 = self.calculate_sma(close, 10)
            sma_20 = self.calculate_sma(close, 20)
            ema_14 = self.calculate_ema(close, 14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Volatilidade (True Range simplificado)
            if len(high) > 1:
                tr = max(
                    high[-1] - low[-1],
                    abs(high[-1] - close[-2]),
                    abs(low[-1] - close[-2])
                )
                atr = tr  # Simplificado
            else:
                atr = high[-1] - low[-1]
            
            # Stochastic simplificado
            if len(high) >= 14:
                lowest_low = min(low[-14:])
                highest_high = max(high[-14:])
                if highest_high != lowest_low:
                    k_percent = 100 * (current_price - lowest_low) / (highest_high - lowest_low)
                else:
                    k_percent = 50
            else:
                k_percent = 50
            
            return {
                'current_price': current_price,
                'price_change': close[-1] - close[-2] if len(close) > 1 else 0,
                'price_change_percent': ((close[-1] - close[-2]) / close[-2] * 100) if len(close) > 1 else 0,
                'rsi': rsi,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'ema_14': ema_14,
                'bollinger': {
                    'upper': bb_upper,
                    'middle': bb_middle,
                    'lower': bb_lower
                },
                'macd': {
                    'line': macd_line,
                    'signal': macd_signal,
                    'histogram': macd_histogram
                },
                'atr': atr,
                'stochastic_k': k_percent,
                'volatility_percent': (atr / current_price) * 100
            }
            
        except Exception as e:
            print(f"❌ Erro no cálculo de indicadores: {e}")
            return None
    
    def analyze_trend(self, indicators: Dict) -> Dict:
        """Analisa tendência baseada em indicadores reais"""
        trend_signals = []
        trend_strength = 0
        
        current_price = indicators['current_price']
        
        # Análise de Médias Móveis
        if current_price > indicators['sma_10'] > indicators['sma_20']:
            trend_signals.append("Tendência de alta (SMAs)")
            trend_strength += 1
        elif current_price < indicators['sma_10'] < indicators['sma_20']:
            trend_signals.append("Tendência de baixa (SMAs)")
            trend_strength -= 1
        
        # Análise EMA
        if current_price > indicators['ema_14']:
            trend_signals.append("Preço acima da EMA")
            trend_strength += 0.5
        else:
            trend_signals.append("Preço abaixo da EMA")
            trend_strength -= 0.5
        
        # Análise MACD
        macd = indicators['macd']
        if macd['line'] > macd['signal']:
            trend_signals.append("MACD bullish")
            trend_strength += 1
        else:
            trend_signals.append("MACD bearish")
            trend_strength -= 1
        
        # Determinar tendência final
        if trend_strength >= 1.5:
            trend = "bullish"
        elif trend_strength <= -1.5:
            trend = "bearish"
        else:
            trend = "sideways"
        
        return {
            'trend': trend,
            'strength': abs(trend_strength),
            'signals': trend_signals,
            'confidence': min(95, abs(trend_strength) * 25 + 60)
        }
    
    def generate_trading_signal(self, symbol: str) -> Dict:
        """Gera sinal de trading baseado em análise técnica real"""
        try:
            print(f"🤖 Iniciando análise REAL para {symbol}...")
            
            # Obter dados reais
            data = self.get_real_market_data(symbol)
            if data is None:
                return self._generate_error_response("Dados de mercado não disponíveis")
            
            # Calcular indicadores
            indicators = self.calculate_technical_indicators(data)
            if indicators is None:
                return self._generate_error_response("Erro no cálculo de indicadores")
            
            print(f"📊 Preço atual: {indicators['current_price']:.5f}")
            print(f"📈 RSI: {indicators['rsi']:.1f}")
            print(f"📊 Mudança: {indicators['price_change_percent']:.2f}%")
            
            # Analisar tendência
            trend_analysis = self.analyze_trend(indicators)
            
            # Gerar sinais baseados em múltiplos indicadores
            signal_score = 0
            signal_reasons = []
            
            # RSI Analysis
            rsi = indicators['rsi']
            if rsi < 30:
                signal_score += 2
                signal_reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signal_score -= 2
                signal_reasons.append(f"RSI overbought ({rsi:.1f})")
            elif 45 <= rsi <= 55:
                signal_reasons.append(f"RSI neutro ({rsi:.1f})")
            
            # Bollinger Bands Analysis
            bb = indicators['bollinger']
            current_price = indicators['current_price']
            if current_price <= bb['lower']:
                signal_score += 1.5
                signal_reasons.append("Preço na banda inferior")
            elif current_price >= bb['upper']:
                signal_score -= 1.5
                signal_reasons.append("Preço na banda superior")
            
            # Stochastic Analysis
            stoch_k = indicators['stochastic_k']
            if stoch_k < 20:
                signal_score += 1
                signal_reasons.append(f"Stochastic oversold ({stoch_k:.1f})")
            elif stoch_k > 80:
                signal_score -= 1
                signal_reasons.append(f"Stochastic overbought ({stoch_k:.1f})")
            
            # Price Change Analysis
            price_change = indicators['price_change_percent']
            if price_change < -0.5:
                signal_score += 0.5
                signal_reasons.append(f"Queda recente ({price_change:.2f}%)")
            elif price_change > 0.5:
                signal_score -= 0.5
                signal_reasons.append(f"Alta recente ({price_change:.2f}%)")
            
            # Trend Confirmation
            if trend_analysis['trend'] == 'bullish':
                signal_score += trend_analysis['strength'] * 0.5
                signal_reasons.append("Tendência bullish confirmada")
            elif trend_analysis['trend'] == 'bearish':
                signal_score -= trend_analysis['strength'] * 0.5
                signal_reasons.append("Tendência bearish confirmada")
            
            # Determinar direção e confiança
            if signal_score >= 1:
                direction = "call"
                confidence = min(95, 70 + abs(signal_score) * 8)
            elif signal_score <= -1:
                direction = "put"
                confidence = min(95, 70 + abs(signal_score) * 8)
            else:
                direction = "call" if signal_score > 0 else "put"
                confidence = max(60, 70 - abs(1 - abs(signal_score)) * 10)
            
            # Determinar timeframe baseado na volatilidade
            volatility = indicators['volatility_percent']
            if volatility > 2.0:
                timeframe = {"type": "minutes", "duration": 1}
            elif volatility > 1.0:
                timeframe = {"type": "minutes", "duration": 2}
            else:
                timeframe = {"type": "minutes", "duration": 3}
            
            reasoning = " | ".join(signal_reasons[:3]) if signal_reasons else "Análise técnica multi-indicador"
            
            print(f"✅ Sinal gerado: {direction.upper()} com {confidence:.1f}% confiança")
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
                    'price_change': round(indicators['price_change'], 5),
                    'price_change_percent': round(indicators['price_change_percent'], 2),
                    'volatility': round(volatility, 2),
                    'trend': trend_analysis['trend'],
                    'trend_strength': round(trend_analysis['strength'], 2)
                },
                'technical_indicators': {
                    'rsi': round(rsi, 1),
                    'macd_signal': 'bullish' if indicators['macd']['line'] > indicators['macd']['signal'] else 'bearish',
                    'bollinger_position': self._get_bollinger_position(current_price, bb),
                    'stochastic_signal': 'oversold' if stoch_k < 20 else 'overbought' if stoch_k > 80 else 'neutral',
                    'price_vs_sma20': 'above' if current_price > indicators['sma_20'] else 'below'
                },
                'optimal_timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance Real Data'
            }
            
        except Exception as e:
            print(f"❌ Erro na geração de sinal: {e}")
            return self._generate_error_response(f"Erro interno: {str(e)}")
    
    def _get_bollinger_position(self, price: float, bb: Dict) -> str:
        """Determina posição nas Bandas de Bollinger"""
        if price >= bb['upper']:
            return "upper_band"
        elif price <= bb['lower']:
            return "lower_band"
        elif price > bb['middle']:
            return "above_middle"
        else:
            return "below_middle"
    
    def _generate_error_response(self, error_msg: str) -> Dict:
        """Gera resposta de erro padronizada"""
        return {
            'status': 'error',
            'message': error_msg,
            'direction': 'call',
            'confidence': 50,
            'reasoning': 'Dados insuficientes - usando fallback',
            'timestamp': datetime.now().isoformat()
        }

# ===============================================
# INSTÂNCIA GLOBAL
# ===============================================

analyzer = RealTechnicalAnalysis()

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': '🤖 IA DE TRADING REAL - Análise Técnica Verdadeira',
        'version': '3.0.0 - REAL DATA',
        'features': [
            '✅ Yahoo Finance - Dados reais de mercado',
            '✅ RSI, MACD, Bollinger Bands calculados',
            '✅ Análise de tendência multi-indicador',
            '✅ Stochastic e volatilidade real',
            '✅ Sinais baseados em dados verdadeiros',
            '✅ Sem dependências complexas (TA-Lib free)'
        ],
        'supported_symbols': list(analyzer.symbol_mapping.keys()),
        'data_source': 'Yahoo Finance Real-Time Data',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signal', methods=['POST', 'GET'])
@app.route('/trading-signal', methods=['POST', 'GET'])
def get_trading_signal():
    """Endpoint principal para sinais de trading REAIS"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
    
    print(f"\n🔄 Nova requisição para {symbol}")
    
    # Validar símbolo
    if symbol not in analyzer.symbol_mapping:
        return jsonify({
            'status': 'error',
            'message': f'Símbolo {symbol} não suportado',
            'supported_symbols': list(analyzer.symbol_mapping.keys())
        }), 400
    
    # Gerar sinal real
    signal_data = analyzer.generate_trading_signal(symbol)
    
    return jsonify(signal_data)

@app.route('/analyze', methods=['POST', 'GET'])
def analyze_market():
    """Análise detalhada do mercado"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
    
    try:
        market_data = analyzer.get_real_market_data(symbol)
        if market_data is None:
            return jsonify({'status': 'error', 'message': 'Dados não disponíveis'}), 500
        
        indicators = analyzer.calculate_technical_indicators(market_data)
        if indicators is None:
            return jsonify({'status': 'error', 'message': 'Erro no cálculo'}), 500
        
        trend_analysis = analyzer.analyze_trend(indicators)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'market_data': {
                'current_price': round(indicators['current_price'], 5),
                'price_change_percent': round(indicators['price_change_percent'], 2),
                'volatility': round(indicators['volatility_percent'], 2),
                'candles_analyzed': len(market_data)
            },
            'technical_indicators': {
                'rsi': round(indicators['rsi'], 1),
                'sma_10': round(indicators['sma_10'], 5),
                'sma_20': round(indicators['sma_20'], 5),
                'bollinger_upper': round(indicators['bollinger']['upper'], 5),
                'bollinger_lower': round(indicators['bollinger']['lower'], 5),
                'stochastic_k': round(indicators['stochastic_k'], 1)
            },
            'trend_analysis': trend_analysis,
            'data_source': 'Yahoo Finance',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro na análise: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': '🟢 IA REAL Online',
        'data_source': 'Yahoo Finance',
        'timestamp': datetime.now().isoformat()
    })

# ===============================================
# INICIALIZAÇÃO
# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Iniciando IA REAL de Trading...")
    print("📊 Fonte de dados: Yahoo Finance")
    print("⚙️ Indicadores: RSI, MACD, Bollinger, SMA, EMA")
    print("🎯 Análise técnica verdadeira ativa!")
    print(f"🌐 Porta: {port}")
    print("✅ IA REAL pronta!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
