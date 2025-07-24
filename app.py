from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import random
import math

app = Flask(__name__)
CORS(app)

# ===============================================
# IA ESPECÍFICA PARA OTC DA IQ OPTION
# ===============================================

class IQOptionOTCAnalysis:
    def __init__(self):
        # Configurações específicas dos OTCs da IQ Option
        self.otc_config = {
            'EURUSD-OTC': {
                'base_price': 1.08,
                'volatility': 0.008,  # 0.8% volatilidade típica
                'trend_factor': 0.0001,
                'support_resistance': [1.06, 1.10],
                'trading_hours': '24/7',
                'spread': 0.00003
            },
            'GBPUSD-OTC': {
                'base_price': 1.25,
                'volatility': 0.012,
                'trend_factor': 0.0002,
                'support_resistance': [1.22, 1.28],
                'trading_hours': '24/7',
                'spread': 0.00004
            },
            'USDJPY-OTC': {
                'base_price': 110.0,
                'volatility': 0.015,
                'trend_factor': 0.02,
                'support_resistance': [108.0, 112.0],
                'trading_hours': '24/7',
                'spread': 0.003
            },
            'AUDUSD-OTC': {
                'base_price': 0.67,
                'volatility': 0.010,
                'trend_factor': 0.0001,
                'support_resistance': [0.65, 0.69],
                'trading_hours': '24/7',
                'spread': 0.00004
            },
            'USDCAD-OTC': {
                'base_price': 1.35,
                'volatility': 0.009,
                'trend_factor': 0.0001,
                'support_resistance': [1.32, 1.38],
                'trading_hours': '24/7',
                'spread': 0.00004
            },
            'USDCHF-OTC': {
                'base_price': 0.92,
                'volatility': 0.008,
                'trend_factor': 0.0001,
                'support_resistance': [0.90, 0.94],
                'trading_hours': '24/7',
                'spread': 0.00003
            },
            'EURJPY-OTC': {
                'base_price': 118.0,
                'volatility': 0.011,
                'trend_factor': 0.015,
                'support_resistance': [116.0, 120.0],
                'trading_hours': '24/7',
                'spread': 0.004
            },
            'EURGBP-OTC': {
                'base_price': 0.86,
                'volatility': 0.007,
                'trend_factor': 0.0001,
                'support_resistance': [0.84, 0.88],
                'trading_hours': '24/7',
                'spread': 0.00003
            },
            'AUDCAD-OTC': {
                'base_price': 0.91,
                'volatility': 0.009,
                'trend_factor': 0.0001,
                'support_resistance': [0.89, 0.93],
                'trading_hours': '24/7',
                'spread': 0.00004
            }
        }
    
    def generate_otc_data(self, symbol, num_candles=100):
        """Gera dados realísticos para OTC da IQ Option"""
        try:
            if symbol not in self.otc_config:
                print(f"⚠️ Símbolo {symbol} não configurado, usando padrão EURUSD-OTC")
                config = self.otc_config['EURUSD-OTC']
            else:
                config = self.otc_config[symbol]
            
            print(f"🎯 Gerando dados OTC para {symbol}")
            print(f"📊 Configuração: Preço base {config['base_price']}, Volatilidade {config['volatility']*100:.1f}%")
            
            base_price = config['base_price']
            volatility = config['volatility']
            trend_factor = config['trend_factor']
            support, resistance = config['support_resistance']
            
            # Gerar timestamp base (últimas 2 horas)
            now = datetime.now()
            start_time = now - timedelta(hours=2)
            
            prices = []
            highs = []
            lows = []
            volumes = []
            timestamps = []
            
            current_price = base_price + random.uniform(-0.01, 0.01)
            
            # Simular tendência intraday
            hour = datetime.now().hour
            if 8 <= hour <= 12:  # Manhã europeia - mais volátil
                session_volatility = volatility * 1.3
                trend_bias = 0.0002
            elif 13 <= hour <= 17:  # Tarde europeia/manhã americana
                session_volatility = volatility * 1.5
                trend_bias = -0.0001
            elif 18 <= hour <= 22:  # Tarde americana
                session_volatility = volatility * 1.2
                trend_bias = 0.0001
            else:  # Asiático/noite
                session_volatility = volatility * 0.8
                trend_bias = 0
            
            for i in range(num_candles):
                # Timestamp para cada vela (1 minuto)
                candle_time = start_time + timedelta(minutes=i)
                timestamps.append(candle_time)
                
                # Movimento browniano com tendência e mean reversion
                drift = trend_bias + random.uniform(-trend_factor, trend_factor)
                
                # Mean reversion - volta para preço base
                mean_reversion = (base_price - current_price) * 0.001
                
                # Support/Resistance
                if current_price <= support:
                    bounce_factor = 0.0005
                elif current_price >= resistance:
                    bounce_factor = -0.0005
                else:
                    bounce_factor = 0
                
                # Movimento final
                total_drift = drift + mean_reversion + bounce_factor
                noise = session_volatility * random.gauss(0, 1)
                change = total_drift + noise
                
                new_price = current_price * (1 + change)
                
                # Gerar high e low realísticos
                high_offset = abs(random.gauss(0, session_volatility * 0.3))
                low_offset = abs(random.gauss(0, session_volatility * 0.3))
                
                candle_high = new_price * (1 + high_offset)
                candle_low = new_price * (1 - low_offset)
                
                # Garantir que price esteja entre high e low
                candle_high = max(candle_high, new_price)
                candle_low = min(candle_low, new_price)
                
                prices.append(new_price)
                highs.append(candle_high)
                lows.append(candle_low)
                volumes.append(random.randint(100, 1000))  # Volume simulado
                
                current_price = new_price
            
            print(f"✅ Gerados {len(prices)} velas OTC para {symbol}")
            print(f"📈 Preço atual: {current_price:.5f}")
            print(f"📊 Variação: {((current_price - prices[0]) / prices[0] * 100):+.2f}%")
            
            return {
                'prices': prices,
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'timestamps': timestamps,
                'current_price': current_price,
                'symbol': symbol,
                'data_source': f'OTC Simulado IQ Option - Sessão {self._get_session_name()}',
                'config': config
            }
            
        except Exception as e:
            print(f"❌ Erro ao gerar dados OTC: {e}")
            return None
    
    def _get_session_name(self):
        """Determina sessão de trading atual"""
        hour = datetime.now().hour
        if 8 <= hour <= 12:
            return "Europeia (Manhã)"
        elif 13 <= hour <= 17:
            return "Euro-Americana"
        elif 18 <= hour <= 22:
            return "Americana"
        else:
            return "Asiática"
    
    def calculate_rsi(self, prices, period=14):
        """RSI otimizado para OTC"""
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
        """SMA"""
        try:
            if len(prices) < period:
                return prices[-1]
            return sum(prices[-period:]) / period
        except:
            return prices[-1] if prices else 0
    
    def calculate_ema(self, prices, period):
        """EMA para OTC"""
        try:
            if len(prices) < period:
                return sum(prices) / len(prices)
            
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
        except:
            return prices[-1] if prices else 0
    
    def calculate_volatility_otc(self, prices, highs, lows):
        """Volatilidade específica para OTC"""
        try:
            if len(prices) < 2:
                return 1.0
            
            # ATR simplificado para OTC
            tr_values = []
            for i in range(1, min(14, len(prices))):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i-1]),
                    abs(lows[i] - prices[i-1])
                )
                tr_values.append(tr)
            
            atr = sum(tr_values) / len(tr_values) if tr_values else (highs[-1] - lows[-1])
            volatility_percent = (atr / prices[-1]) * 100
            
            return max(0.1, min(5.0, volatility_percent))
            
        except:
            return 1.0
    
    def analyze_otc_trend(self, prices, symbol):
        """Análise de tendência específica para OTC"""
        try:
            if len(prices) < 20:
                return {'trend': 'sideways', 'strength': 0.5}
            
            # Médias específicas para OTC (mais sensíveis)
            sma_5 = self.calculate_sma(prices, 5)
            sma_10 = self.calculate_sma(prices, 10) 
            sma_20 = self.calculate_sma(prices, 20)
            ema_8 = self.calculate_ema(prices, 8)
            current = prices[-1]
            
            # Análise de momentum OTC
            momentum_short = ((prices[-1] - prices[-5]) / prices[-5]) * 100 if len(prices) >= 5 else 0
            momentum_medium = ((prices[-1] - prices[-10]) / prices[-10]) * 100 if len(prices) >= 10 else 0
            
            # Score de tendência
            trend_score = 0
            signals = []
            
            # Análise das médias
            if current > sma_5 > sma_10 > sma_20:
                trend_score += 2
                signals.append("Médias alinhadas alta")
            elif current < sma_5 < sma_10 < sma_20:
                trend_score -= 2
                signals.append("Médias alinhadas baixa")
            elif current > ema_8:
                trend_score += 0.5
                signals.append("Acima da EMA")
            else:
                trend_score -= 0.5
                signals.append("Abaixo da EMA")
            
            # Momentum
            if momentum_short > 0.05:
                trend_score += 1
                signals.append("Momentum positivo")
            elif momentum_short < -0.05:
                trend_score -= 1
                signals.append("Momentum negativo")
            
            # Support/Resistance para OTC
            config = self.otc_config.get(symbol, self.otc_config['EURUSD-OTC'])
            support, resistance = config['support_resistance']
            
            if current <= support * 1.002:  # Próximo ao suporte
                trend_score += 0.5
                signals.append("Próximo ao suporte")
            elif current >= resistance * 0.998:  # Próximo à resistência
                trend_score -= 0.5
                signals.append("Próximo à resistência")
            
            # Classificar tendência
            if trend_score >= 1.5:
                trend = 'bullish'
            elif trend_score <= -1.5:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            strength = min(1.0, abs(trend_score) / 2)
            
            return {
                'trend': trend,
                'strength': strength,
                'score': trend_score,
                'signals': signals,
                'momentum_short': momentum_short,
                'momentum_medium': momentum_medium,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'ema_8': ema_8
            }
            
        except Exception as e:
            print(f"❌ Erro na análise de tendência: {e}")
            return {'trend': 'sideways', 'strength': 0.5}
    
    def generate_otc_signal(self, symbol):
        """Gera sinal específico para OTC da IQ Option"""
        try:
            print(f"\n🤖 Análise OTC IQ OPTION para {symbol}")
            
            # Gerar dados OTC realísticos
            data = self.generate_otc_data(symbol)
            if not data:
                return self._error_response("Falha ao gerar dados OTC")
            
            prices = data['prices']
            highs = data['highs']
            lows = data['lows']
            current_price = data['current_price']
            config = data['config']
            
            print(f"📊 Fonte: {data['data_source']}")
            print(f"💰 Preço atual: {current_price:.5f}")
            
            # Indicadores específicos para OTC
            rsi = self.calculate_rsi(prices, 14)
            rsi_fast = self.calculate_rsi(prices, 9)  # RSI mais rápido para OTC
            sma_10 = self.calculate_sma(prices, 10)
            sma_20 = self.calculate_sma(prices, 20)
            ema_8 = self.calculate_ema(prices, 8)
            volatility = self.calculate_volatility_otc(prices, highs, lows)
            trend_analysis = self.analyze_otc_trend(prices, symbol)
            
            print(f"🎯 RSI: {rsi:.1f} | RSI Fast: {rsi_fast:.1f}")
            print(f"📊 EMA8: {ema_8:.5f} | SMA10: {sma_10:.5f}")
            print(f"📈 Volatilidade OTC: {volatility:.2f}%")
            print(f"📊 Tendência: {trend_analysis['trend']} (força: {trend_analysis['strength']:.2f})")
            
            # Sistema de pontuação para OTC
            signal_score = 0
            reasons = []
            
            # RSI Analysis (duplo)
            if rsi < 30 and rsi_fast < 35:
                signal_score += 2.5
                reasons.append(f"RSI duplo oversold ({rsi:.1f}/{rsi_fast:.1f})")
            elif rsi > 70 and rsi_fast > 65:
                signal_score -= 2.5
                reasons.append(f"RSI duplo overbought ({rsi:.1f}/{rsi_fast:.1f})")
            elif rsi < 40:
                signal_score += 1
                reasons.append(f"RSI baixo ({rsi:.1f})")
            elif rsi > 60:
                signal_score -= 1
                reasons.append(f"RSI alto ({rsi:.1f})")
            
            # Price vs EMAs (importante para OTC)
            if current_price > ema_8 > sma_10:
                signal_score += 1.5
                reasons.append("Preço acima das médias rápidas")
            elif current_price < ema_8 < sma_10:
                signal_score -= 1.5
                reasons.append("Preço abaixo das médias rápidas")
            
            # Trend Analysis
            trend_score = trend_analysis['score']
            if trend_score > 1:
                signal_score += min(1.5, trend_score * 0.5)
                reasons.append(f"Tendência bullish ({trend_score:.1f})")
            elif trend_score < -1:
                signal_score -= min(1.5, abs(trend_score) * 0.5)
                reasons.append(f"Tendência bearish ({abs(trend_score):.1f})")
            
            # Support/Resistance OTC
            support, resistance = config['support_resistance']
            if current_price <= support * 1.003:
                signal_score += 1
                reasons.append("Próximo ao suporte OTC")
            elif current_price >= resistance * 0.997:
                signal_score -= 1
                reasons.append("Próximo à resistência OTC")
            
            # Volatility Analysis
            if volatility > 2.0:
                reasons.append(f"Alta volatilidade ({volatility:.1f}%)")
            elif volatility < 0.5:
                reasons.append(f"Baixa volatilidade ({volatility:.1f}%)")
            
            # Momentum OTC
            momentum = trend_analysis.get('momentum_short', 0)
            if momentum > 0.1:
                signal_score += 0.5
                reasons.append(f"Momentum positivo ({momentum:.2f}%)")
            elif momentum < -0.1:
                signal_score -= 0.5
                reasons.append(f"Momentum negativo ({momentum:.2f}%)")
            
            # Determinar direção e confiança (ajustado para OTC)
            if signal_score >= 1.5:
                direction = "call"
                confidence = min(92, 72 + abs(signal_score) * 8)
            elif signal_score <= -1.5:
                direction = "put"
                confidence = min(92, 72 + abs(signal_score) * 8)
            else:
                direction = "call" if signal_score > 0 else "put"
                confidence = max(65, 72 - abs(1.5 - abs(signal_score)) * 12)
            
            # Timeframe específico para OTC
            if volatility > 2.5:
                timeframe = {"type": "minutes", "duration": 1}
            elif volatility > 1.5:
                timeframe = {"type": "minutes", "duration": 2}
            elif volatility > 1.0:
                timeframe = {"type": "minutes", "duration": 3}
            else:
                timeframe = {"type": "minutes", "duration": 5}
            
            reasoning = " | ".join(reasons[:4]) if reasons else "Análise técnica OTC multi-indicador"
            
            print(f"✅ Sinal OTC: {direction.upper()}")
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
                    'price_change_percent': round(((prices[-1] - prices[-20]) / prices[-20] * 100), 2) if len(prices) > 20 else 0,
                    'volatility': round(volatility, 2),
                    'trend': trend_analysis['trend'],
                    'trend_strength': round(trend_analysis['strength'], 2),
                    'session': self._get_session_name(),
                    'support': support,
                    'resistance': resistance,
                    'candles_analyzed': len(prices)
                },
                'technical_indicators': {
                    'rsi': round(rsi, 1),
                    'rsi_fast': round(rsi_fast, 1),
                    'ema_8': round(ema_8, 5),
                    'sma_10': round(sma_10, 5),
                    'sma_20': round(sma_20, 5),
                    'momentum_short': round(momentum, 3),
                    'price_vs_ema8': 'above' if current_price > ema_8 else 'below',
                    'price_vs_support': 'near' if current_price <= support * 1.005 else 'far'
                },
                'optimal_timeframe': timeframe,
                'otc_config': {
                    'base_price': config['base_price'],
                    'expected_volatility': f"{config['volatility']*100:.1f}%",
                    'support_resistance': config['support_resistance']
                },
                'data_source': data['data_source'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Erro na análise OTC: {e}")
            return self._error_response(f"Erro: {str(e)}")
    
    def _error_response(self, message):
        return {
            'status': 'error',
            'message': message,
            'direction': 'call',
            'confidence': 50,
            'reasoning': 'Análise OTC indisponível',
            'timestamp': datetime.now().isoformat()
        }

# ===============================================
# INSTÂNCIA GLOBAL
# ===============================================

analyzer = IQOptionOTCAnalysis()

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': '🎯 IA ESPECÍFICA PARA OTC DA IQ OPTION',
        'version': '6.0.0 - OTC SPECIALIZED',
        'features': [
            '✅ Dados OTC realísticos da IQ Option',
            '✅ Configurações específicas por par OTC',
            '✅ RSI duplo (14 e 9 períodos)',
            '✅ Support/Resistance específicos OTC',
            '✅ Análise de sessões de trading',
            '✅ Volatilidade calculada para OTC',
            '✅ Momentum e médias otimizadas',
            '✅ Timeframes baseados em volatilidade OTC'
        ],
        'supported_otc_symbols': list(analyzer.otc_config.keys()),
        'current_session': analyzer._get_session_name(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signal', methods=['POST', 'GET'])
@app.route('/trading-signal', methods=['POST', 'GET'])
def get_signal():
    """Endpoint principal - IA OTC IQ OPTION"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
    
    print(f"\n🔄 Análise OTC para {symbol}")
    
    # Validar símbolo OTC
    if symbol not in analyzer.otc_config:
        return jsonify({
            'status': 'warning',
            'message': f'Símbolo {symbol} não tem configuração específica, usando padrão EURUSD-OTC',
            'supported_otc_symbols': list(analyzer.otc_config.keys())
        })
    
    # Gerar sinal OTC
    result = analyzer.generate_otc_signal(symbol)
    return jsonify(result)

@app.route('/otc-config/<symbol>')
def get_otc_config(symbol):
    """Retorna configuração específica do OTC"""
    if symbol in analyzer.otc_config:
        config = analyzer.otc_config[symbol]
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'config': config,
            'current_session': analyzer._get_session_name(),
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Configuração não encontrada para {symbol}',
            'available_symbols': list(analyzer.otc_config.keys())
        }), 404

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': '🟢 IA OTC IQ OPTION Online',
        'current_session': analyzer._get_session_name(),
        'otc_pairs_available': len(analyzer.otc_config),
        'timestamp': datetime.now().isoformat()
    })

# ===============================================
# INICIALIZAÇÃO
# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 IA OTC IQ OPTION Iniciando...")
    print("🎯 Especializada em ativos OTC da IQ Option")
    print("📊 Dados sintéticos baseados em padrões OTC reais")
    print("⚙️ RSI duplo, Support/Resistance, Sessões de trading")
    print(f"🕒 Sessão atual: {analyzer._get_session_name()}")
    print(f"💰 Pares OTC disponíveis: {len(analyzer.otc_config)}")
    print("✅ Nunca falha - dados sempre disponíveis!")
    print(f"🌐 Porta: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
