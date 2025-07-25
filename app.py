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
# IA RETRAÇÃO + MACHINE LEARNING
# ===============================================

class RetractionMLAnalysis:
    def __init__(self):
        # Configurações OTC da IQ Option
        self.otc_config = {
            'EURUSD-OTC': {'base_price': 1.08, 'volatility': 0.008, 'support_resistance': [1.06, 1.10]},
            'GBPUSD-OTC': {'base_price': 1.25, 'volatility': 0.012, 'support_resistance': [1.22, 1.28]},
            'USDJPY-OTC': {'base_price': 110.0, 'volatility': 0.015, 'support_resistance': [108.0, 112.0]},
            'AUDUSD-OTC': {'base_price': 0.67, 'volatility': 0.010, 'support_resistance': [0.65, 0.69]},
            'USDCAD-OTC': {'base_price': 1.35, 'volatility': 0.009, 'support_resistance': [1.32, 1.38]},
            'USDCHF-OTC': {'base_price': 0.92, 'volatility': 0.008, 'support_resistance': [0.90, 0.94]},
            'EURJPY-OTC': {'base_price': 118.0, 'volatility': 0.011, 'support_resistance': [116.0, 120.0]},
            'EURGBP-OTC': {'base_price': 0.86, 'volatility': 0.007, 'support_resistance': [0.84, 0.88]},
            'AUDCAD-OTC': {'base_price': 0.91, 'volatility': 0.009, 'support_resistance': [0.89, 0.93]}
        }
        
        # 🧠 SISTEMA DE MACHINE LEARNING
        self.ml_memory = {
            # Histórico de performance por símbolo
            'symbol_performance': {},
            
            # Pesos dos indicadores (ajustados por ML)
            'indicator_weights': {
                'retraction_signal': 1.0,      # Peso do sinal de retração
                'rsi_oversold': 0.8,           # Peso do RSI oversold
                'rsi_overbought': 0.8,         # Peso do RSI overbought
                'support_resistance': 0.6,     # Peso support/resistance
                'momentum': 0.4,               # Peso do momentum
                'volatility_filter': 0.3,      # Peso do filtro de volatilidade
                'session_factor': 0.2          # Peso da sessão de trading
            },
            
            # Parâmetros adaptativos
            'adaptive_params': {
                'confidence_adjustment': 0.0,   # Ajuste de confiança baseado em performance
                'retraction_threshold': 0.0001, # Limiar mínimo para considerar retração
                'min_confidence': 70,           # Confiança mínima
                'max_confidence': 95,           # Confiança máxima
                'learning_rate': 0.1            # Taxa de aprendizado
            },
            
            # Estatísticas globais
            'global_stats': {
                'total_signals': 0,
                'total_wins': 0,
                'total_losses': 0,
                'win_rate': 0.0,
                'last_updated': datetime.now().isoformat()
            }
        }
    
    def generate_otc_data(self, symbol, num_candles=50):
        """Gera dados OTC com padrões de retração"""
        try:
            config = self.otc_config.get(symbol, self.otc_config['EURUSD-OTC'])
            
            base_price = config['base_price']
            volatility = config['volatility']
            
            prices = []
            opens = []
            highs = []
            lows = []
            closes = []
            
            current_price = base_price + random.uniform(-0.01, 0.01)
            
            # Gerar velas com padrões de retração
            for i in range(num_candles):
                # Determinar se será vela verde ou vermelha
                # Inserir padrões de retração propositalmente
                if i > 0:
                    prev_direction = "green" if closes[-1] > opens[-1] else "red"
                    
                    # 60% chance de retração (padrão que queremos detectar)
                    if random.random() < 0.6:
                        # Retração: direção oposta à vela anterior
                        if prev_direction == "green":
                            direction = "red"
                        else:
                            direction = "green"
                    else:
                        # Continuação
                        direction = prev_direction
                else:
                    direction = random.choice(["green", "red"])
                
                # Gerar OHLC baseado na direção
                open_price = current_price
                
                if direction == "green":
                    # Vela verde (alta)
                    change = random.uniform(0.0001, volatility * 2)
                    close_price = open_price * (1 + change)
                    high_price = close_price * (1 + random.uniform(0, volatility * 0.5))
                    low_price = open_price * (1 - random.uniform(0, volatility * 0.3))
                else:
                    # Vela vermelha (baixa)
                    change = random.uniform(0.0001, volatility * 2)
                    close_price = open_price * (1 - change)
                    high_price = open_price * (1 + random.uniform(0, volatility * 0.3))
                    low_price = close_price * (1 - random.uniform(0, volatility * 0.5))
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                prices.append(close_price)  # Para compatibilidade
                
                current_price = close_price
            
            print(f"✅ Dados OTC gerados para {symbol}: {len(closes)} velas")
            print(f"📈 Última vela: Abertura {opens[-1]:.5f} → Fechamento {closes[-1]:.5f}")
            
            return {
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'prices': prices,  # Para compatibilidade
                'current_price': closes[-1],
                'symbol': symbol,
                'data_source': 'OTC com Padrões de Retração'
            }
            
        except Exception as e:
            print(f"❌ Erro ao gerar dados: {e}")
            return None
    
    def retraction_strategy(self, opens, closes):
        """
        🎯 ESTRATÉGIA DE RETRAÇÃO DE VELA
        
        Lógica: Se a vela anterior foi vermelha (baixa), próxima operação é CALL (alta)
                Se a vela anterior foi verde (alta), próxima operação é PUT (baixa)
        """
        try:
            if len(opens) < 2 or len(closes) < 2:
                return None, "Dados insuficientes para retração"
            
            # Analisar vela anterior
            prev_open = opens[-2]
            prev_close = closes[-2]
            
            # Calcular tamanho da vela
            candle_size = abs(prev_close - prev_open)
            price_pct_change = (candle_size / prev_open) * 100
            
            # Determinar direção da vela anterior
            if prev_close > prev_open:
                prev_direction = "green"  # Vela verde (alta)
                retraction_signal = "put"  # Apostamos na retração (baixa)
                signal_reason = f"Vela verde anterior (+{price_pct_change:.3f}%) → Retração PUT"
            elif prev_close < prev_open:
                prev_direction = "red"    # Vela vermelha (baixa)
                retraction_signal = "call" # Apostamos na retração (alta)
                signal_reason = f"Vela vermelha anterior (-{price_pct_change:.3f}%) → Retração CALL"
            else:
                prev_direction = "doji"   # Doji (empate)
                retraction_signal = "call" if random.random() > 0.5 else "put"
                signal_reason = "Doji anterior → Sinal neutro"
            
            return {
                'signal': retraction_signal,
                'prev_direction': prev_direction,
                'candle_size_pct': price_pct_change,
                'reason': signal_reason,
                'confidence_base': min(80, 60 + (price_pct_change * 1000))  # Maior confiança em velas maiores
            }, signal_reason
            
        except Exception as e:
            print(f"❌ Erro na estratégia de retração: {e}")
            return None, "Erro na análise"
    
    def calculate_rsi(self, prices, period=14):
        """RSI"""
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
    
    def ml_adjust_signal(self, symbol, base_signal, indicators):
        """
        🧠 MACHINE LEARNING - Ajusta sinal baseado no aprendizado
        """
        try:
            # Obter performance histórica do símbolo
            symbol_perf = self.ml_memory['symbol_performance'].get(symbol, {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.5,
                'last_signals': []
            })
            
            # Obter pesos adaptativos
            weights = self.ml_memory['indicator_weights']
            params = self.ml_memory['adaptive_params']
            
            # Calcular score ajustado por ML
            ml_score = 0
            ml_reasons = []
            
            # 1. Sinal base de retração (peso principal)
            retraction_confidence = base_signal.get('confidence_base', 70)
            ml_score += retraction_confidence * weights['retraction_signal']
            ml_reasons.append(f"Retração base: {retraction_confidence:.1f}")
            
            # 2. RSI com peso adaptativo
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                rsi_boost = (30 - rsi) * weights['rsi_oversold']
                ml_score += rsi_boost
                ml_reasons.append(f"RSI oversold boost: +{rsi_boost:.1f}")
            elif rsi > 70:
                rsi_penalty = (rsi - 70) * weights['rsi_overbought']
                ml_score -= rsi_penalty
                ml_reasons.append(f"RSI overbought penalty: -{rsi_penalty:.1f}")
            
            # 3. Support/Resistance
            if indicators.get('near_support', False):
                sr_boost = 10 * weights['support_resistance']
                ml_score += sr_boost
                ml_reasons.append(f"Near support: +{sr_boost:.1f}")
            elif indicators.get('near_resistance', False):
                sr_penalty = 10 * weights['support_resistance']
                ml_score -= sr_penalty
                ml_reasons.append(f"Near resistance: -{sr_penalty:.1f}")
            
            # 4. Ajuste baseado na performance histórica
            if symbol_perf['total_signals'] > 5:  # Só ajustar após histórico mínimo
                perf_adjustment = (symbol_perf['win_rate'] - 0.5) * 20  # -10 a +10
                ml_score += perf_adjustment * params['learning_rate']
                ml_reasons.append(f"Performance histórica: {perf_adjustment:+.1f}")
            
            # 5. Ajuste global de confiança
            ml_score += params['confidence_adjustment']
            
            # Normalizar para confiança (50-95%)
            final_confidence = max(params['min_confidence'], 
                                 min(params['max_confidence'], ml_score))
            
            # Decidir se usar o sinal ou inverter baseado no ML
            original_signal = base_signal['signal']
            
            # Se performance muito ruim, considerar inverter estratégia
            if symbol_perf['total_signals'] > 10 and symbol_perf['win_rate'] < 0.3:
                if random.random() < 0.3:  # 30% chance de inverter
                    ml_signal = "put" if original_signal == "call" else "call"
                    ml_reasons.append("❗ ML inverteu sinal (baixa performance)")
                else:
                    ml_signal = original_signal
            else:
                ml_signal = original_signal
            
            return {
                'signal': ml_signal,
                'confidence': round(final_confidence, 1),
                'ml_reasons': ml_reasons,
                'ml_score': round(ml_score, 2),
                'symbol_performance': symbol_perf,
                'weights_used': {k: round(v, 2) for k, v in weights.items()}
            }
            
        except Exception as e:
            print(f"❌ Erro no ajuste ML: {e}")
            return {
                'signal': base_signal['signal'],
                'confidence': base_signal.get('confidence_base', 70),
                'ml_reasons': ['Erro no ML - usando base'],
                'ml_score': 0
            }
    
    def update_ml_feedback(self, symbol, predicted_signal, actual_result):
        """
        📚 APRENDIZADO - Atualiza ML baseado no resultado real
        
        actual_result: 'win', 'loss', ou 'tie'
        """
        try:
            print(f"📚 ML Learning: {symbol} - Sinal {predicted_signal} = {actual_result}")
            
            # Atualizar performance do símbolo
            if symbol not in self.ml_memory['symbol_performance']:
                self.ml_memory['symbol_performance'][symbol] = {
                    'total_signals': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5, 'last_signals': []
                }
            
            perf = self.ml_memory['symbol_performance'][symbol]
            perf['total_signals'] += 1
            
            if actual_result == 'win':
                perf['wins'] += 1
                # Reforçar pesos que funcionaram
                self._reinforce_weights(True)
            elif actual_result == 'loss':
                perf['losses'] += 1
                # Penalizar pesos que falharam
                self._reinforce_weights(False)
            
            # Recalcular win rate
            perf['win_rate'] = perf['wins'] / perf['total_signals'] if perf['total_signals'] > 0 else 0.5
            
            # Manter histórico dos últimos 20 sinais
            perf['last_signals'].append({
                'signal': predicted_signal,
                'result': actual_result,
                'timestamp': datetime.now().isoformat()
            })
            
            if len(perf['last_signals']) > 20:
                perf['last_signals'] = perf['last_signals'][-20:]
            
            # Atualizar estatísticas globais
            global_stats = self.ml_memory['global_stats']
            global_stats['total_signals'] += 1
            
            if actual_result == 'win':
                global_stats['total_wins'] += 1
            elif actual_result == 'loss':
                global_stats['total_losses'] += 1
            
            global_stats['win_rate'] = global_stats['total_wins'] / global_stats['total_signals'] if global_stats['total_signals'] > 0 else 0.5
            global_stats['last_updated'] = datetime.now().isoformat()
            
            # Ajustar parâmetros adaptativos baseado na performance global
            self._adapt_parameters()
            
            print(f"📊 {symbol}: {perf['wins']}W/{perf['losses']}L = {perf['win_rate']*100:.1f}%")
            print(f"🌍 Global: {global_stats['total_wins']}W/{global_stats['total_losses']}L = {global_stats['win_rate']*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no feedback ML: {e}")
            return False
    
    def _reinforce_weights(self, success):
        """Ajusta pesos baseado no sucesso/falha"""
        try:
            weights = self.ml_memory['indicator_weights']
            learning_rate = self.ml_memory['adaptive_params']['learning_rate']
            
            if success:
                # Reforçar todos os pesos ligeiramente
                for key in weights:
                    weights[key] = min(2.0, weights[key] + learning_rate * 0.1)
            else:
                # Reduzir pesos ligeiramente
                for key in weights:
                    weights[key] = max(0.1, weights[key] - learning_rate * 0.1)
            
        except Exception as e:
            print(f"❌ Erro ao ajustar pesos: {e}")
    
    def _adapt_parameters(self):
        """Adapta parâmetros baseado na performance global"""
        try:
            global_stats = self.ml_memory['global_stats']
            params = self.ml_memory['adaptive_params']
            
            win_rate = global_stats['win_rate']
            
            # Ajustar confiança baseado na performance
            if win_rate > 0.7:
                params['confidence_adjustment'] = min(10, params['confidence_adjustment'] + 0.5)
            elif win_rate < 0.4:
                params['confidence_adjustment'] = max(-10, params['confidence_adjustment'] - 0.5)
            
            # Ajustar learning rate
            if global_stats['total_signals'] > 50:
                params['learning_rate'] = max(0.05, params['learning_rate'] * 0.99)  # Reduzir gradualmente
            
        except Exception as e:
            print(f"❌ Erro ao adaptar parâmetros: {e}")
    
    def generate_retraction_signal(self, symbol):
        """
        🎯 GERA SINAL DE RETRAÇÃO COM MACHINE LEARNING
        """
        try:
            print(f"\n🤖 Análise de RETRAÇÃO + ML para {symbol}")
            
            # Gerar dados OTC
            data = self.generate_otc_data(symbol)
            if not data:
                return self._error_response("Falha ao gerar dados OTC")
            
            opens = data['opens']
            closes = data['closes']
            highs = data['highs']
            lows = data['lows']
            current_price = data['current_price']
            
            # 🎯 ESTRATÉGIA DE RETRAÇÃO
            retraction_result, retraction_reason = self.retraction_strategy(opens, closes)
            if not retraction_result:
                return self._error_response("Falha na estratégia de retração")
            
            print(f"📊 {retraction_reason}")
            
            # Calcular indicadores complementares
            rsi = self.calculate_rsi(closes)
            
            # Verificar support/resistance
            config = self.otc_config.get(symbol, self.otc_config['EURUSD-OTC'])
            support, resistance = config['support_resistance']
            
            near_support = current_price <= support * 1.005
            near_resistance = current_price >= resistance * 0.995
            
            indicators = {
                'rsi': rsi,
                'near_support': near_support,
                'near_resistance': near_resistance,
                'current_price': current_price
            }
            
            # 🧠 AJUSTE POR MACHINE LEARNING
            ml_result = self.ml_adjust_signal(symbol, retraction_result, indicators)
            
            final_signal = ml_result['signal']
            final_confidence = ml_result['confidence']
            
            print(f"🎯 Sinal final: {final_signal.upper()}")
            print(f"📊 Confiança ML: {final_confidence}%")
            print(f"🧠 Razões ML: {' | '.join(ml_result['ml_reasons'][:3])}")
            
            # Timeframe baseado na confiança
            if final_confidence >= 85:
                timeframe = {"type": "minutes", "duration": 1}
            elif final_confidence >= 75:
                timeframe = {"type": "minutes", "duration": 2}
            else:
                timeframe = {"type": "minutes", "duration": 3}
            
            reasoning = f"{retraction_reason} | ML Score: {ml_result['ml_score']}"
            
            return {
                'status': 'success',
                'symbol': symbol,
                'direction': final_signal,
                'confidence': final_confidence,
                'signal_score': ml_result['ml_score'],
                'reasoning': reasoning,
                'strategy': 'Retração de Vela + Machine Learning',
                'retraction_analysis': {
                    'prev_candle_direction': retraction_result['prev_direction'],
                    'candle_size_percent': round(retraction_result['candle_size_pct'], 4),
                    'retraction_signal': retraction_result['signal'],
                    'base_confidence': retraction_result['confidence_base']
                },
                'ml_analysis': {
                    'ml_reasons': ml_result['ml_reasons'],
                    'weights_used': ml_result['weights_used'],
                    'symbol_performance': ml_result['symbol_performance'],
                    'learning_active': True
                },
                'market_analysis': {
                    'current_price': round(current_price, 5),
                    'rsi': round(rsi, 1),
                    'near_support': near_support,
                    'near_resistance': near_resistance,
                    'support_level': support,
                    'resistance_level': resistance
                },
                'optimal_timeframe': timeframe,
                'data_source': data['data_source'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return self._error_response(f"Erro: {str(e)}")
    
    def _error_response(self, message):
        return {
            'status': 'error',
            'message': message,
            'direction': 'call',
            'confidence': 50,
            'reasoning': 'Análise indisponível',
            'timestamp': datetime.now().isoformat()
        }

# ===============================================
# INSTÂNCIA GLOBAL
# ===============================================

analyzer = RetractionMLAnalysis()

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def home():
    global_stats = analyzer.ml_memory['global_stats']
    return jsonify({
        'status': 'success',
        'message': '🧠 IA RETRAÇÃO + MACHINE LEARNING',
        'version': '7.0.0 - RETRACTION + ML',
        'strategy': 'Retração de Vela + Aprendizado de Máquina',
        'features': [
            '🎯 Estratégia de Retração de Vela',
            '🧠 Machine Learning que aprende com erros',
            '📊 Ajuste automático de pesos dos indicadores',
            '📈 Performance histórica por símbolo',
            '⚙️ Parâmetros adaptativos',
            '🔄 Feedback loop de aprendizado',
            '📚 Memória de últimas 20 operações por par'
        ],
        'ml_stats': {
            'total_signals': global_stats['total_signals'],
            'global_win_rate': f"{global_stats['win_rate']*100:.1f}%",
            'learning_active': True,
            'last_updated': global_stats['last_updated']
        },
        'supported_symbols': list(analyzer.otc_config.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signal', methods=['POST', 'GET'])
@app.route('/trading-signal', methods=['POST', 'GET'])
def get_signal():
    """Endpoint principal - RETRAÇÃO + ML"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
    
    print(f"\n🔄 Análise RETRAÇÃO + ML para {symbol}")
    
    # Gerar sinal com ML
    result = analyzer.generate_retraction_signal(symbol)
    return jsonify(result)

@app.route('/feedback', methods=['POST'])
def ml_feedback():
    """
    📚 Endpoint para FEEDBACK do Machine Learning
    
    Envie o resultado da operação para a IA aprender:
    {
        "symbol": "EURUSD-OTC",
        "predicted_signal": "call",
        "actual_result": "win"  // "win", "loss", ou "tie"
    }
    """
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol')
        predicted_signal = data.get('predicted_signal')
        actual_result = data.get('actual_result')
        
        if not all([symbol, predicted_signal, actual_result]):
            return jsonify({
                'status': 'error',
                'message': 'Parâmetros obrigatórios: symbol, predicted_signal, actual_result'
            }), 400
        
        if actual_result not in ['win', 'loss', 'tie']:
            return jsonify({
                'status': 'error',
                'message': 'actual_result deve ser: win, loss, ou tie'
            }), 400
        
        # Atualizar ML
        success = analyzer.update_ml_feedback(symbol, predicted_signal, actual_result)
        
        if success:
            # Retornar estatísticas atualizadas
            symbol_perf = analyzer.ml_memory['symbol_performance'].get(symbol, {})
            global_stats = analyzer.ml_memory['global_stats']
            
            return jsonify({
                'status': 'success',
                'message': f'ML atualizado: {symbol} - {predicted_signal} = {actual_result}',
                'learning_updated': True,
                'symbol_performance': {
                    'symbol': symbol,
                    'win_rate': f"{symbol_perf.get('win_rate', 0)*100:.1f}%",
                    'total_signals': symbol_perf.get('total_signals', 0),
                    'wins': symbol_perf.get('wins', 0),
                    'losses': symbol_perf.get('losses', 0)
                },
                'global_performance': {
                    'global_win_rate': f"{global_stats['win_rate']*100:.1f}%",
                    'total_signals': global_stats['total_signals'],
                    'total_wins': global_stats['total_wins'],
                    'total_losses': global_stats['total_losses']
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Falha ao atualizar ML'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro no feedback: {str(e)}'
        }), 500

@app.route('/ml-stats')
def ml_stats():
    """Estatísticas detalhadas do Machine Learning"""
    try:
        ml_memory = analyzer.ml_memory
        
        return jsonify({
            'status': 'success',
            'ml_statistics': {
                'global_stats': ml_memory['global_stats'],
                'symbol_performance': ml_memory['symbol_performance'],
                'indicator_weights': ml_memory['indicator_weights'],
                'adaptive_params': ml_memory['adaptive_params']
            },
            'learning_summary': {
                'total_symbols_learned': len(ml_memory['symbol_performance']),
                'best_performing_symbol': max(ml_memory['symbol_performance'].items(), 
                                            key=lambda x: x[1].get('win_rate', 0))[0] if ml_memory['symbol_performance'] else None,
                'learning_active': True
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao obter estatísticas: {str(e)}'
        }), 500

@app.route('/health')
def health():
    global_stats = analyzer.ml_memory['global_stats']
    return jsonify({
        'status': 'healthy',
        'message': '🟢 IA RETRAÇÃO + ML Online',
        'strategy': 'Retração de Vela + Machine Learning',
        'ml_active': True,
        'global_win_rate': f"{global_stats['win_rate']*100:.1f}%",
        'total_signals_processed': global_stats['total_signals'],
        'timestamp': datetime.now().isoformat()
    })

# ===============================================
# INICIALIZAÇÃO
# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 IA RETRAÇÃO + MACHINE LEARNING Iniciando...")
    print("🎯 Estratégia: Retração de Vela")
    print("🧠 Machine Learning: Aprende com acertos e erros")
    print("📊 Ajuste automático de pesos e parâmetros")
    print("🔄 Sistema de feedback ativo")
    print("📚 Memória de performance por símbolo")
    print("✅ Aprendizado contínuo ativado!")
    print(f"🌐 Porta: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
