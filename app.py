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
# IA RETRAÇÃO + MARTINGALE + MACHINE LEARNING
# ===============================================

class RetractionMartingaleML:
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
        
        # 🎯 CONFIGURAÇÕES DO MARTINGALE
        self.martingale_config = {
            'max_levels': 5,              # Máximo 5 níveis de Martingale
            'multiplier': 2.0,            # Multiplicador (2x = Martingale clássico)
            'base_amount': 1.0,           # Valor base inicial
            'max_total_risk': 31.0,       # Risco máximo total (1+2+4+8+16 = 31)
            'stop_loss_percent': 20.0,    # Stop loss: 20% do saldo
            'safety_balance_percent': 5.0, # Manter 5% do saldo como segurança
            'recovery_mode': True,        # Modo recuperação ativo
            'smart_exit': True            # Saída inteligente do Martingale
        }
        
        # 🧠 SISTEMA DE MACHINE LEARNING + MARTINGALE
        self.ml_memory = {
            # Performance por símbolo
            'symbol_performance': {},
            
            # Pesos dos indicadores
            'indicator_weights': {
                'retraction_signal': 1.0,
                'rsi_oversold': 0.8,
                'rsi_overbought': 0.8,
                'support_resistance': 0.6,
                'momentum': 0.4,
                'volatility_filter': 0.3,
                'martingale_factor': 0.7    # Novo: peso do fator Martingale
            },
            
            # 🎯 PARÂMETROS DO MARTINGALE INTELIGENTE
            'martingale_intelligence': {
                'success_rate_by_level': {   # Taxa de sucesso por nível
                    1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5
                },
                'best_exit_level': 2,        # Melhor nível para sair do Martingale
                'avoid_martingale_sessions': [], # Sessões onde evitar Martingale
                'preferred_martingale_pairs': [], # Pares que funcionam melhor
                'martingale_confidence_threshold': 75, # Confiança mín. para Martingale
                'consecutive_losses_limit': 3,  # Limite de perdas consecutivas
                'martingale_pause_time': 0,    # Tempo de pausa após stop loss
                'dynamic_multiplier': 2.0      # Multiplicador dinâmico
            },
            
            # Parâmetros adaptativos gerais
            'adaptive_params': {
                'confidence_adjustment': 0.0,
                'retraction_threshold': 0.0001,
                'min_confidence': 70,
                'max_confidence': 95,
                'learning_rate': 0.1
            },
            
            # 📊 ESTATÍSTICAS DO MARTINGALE
            'martingale_stats': {
                'total_martingale_sequences': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'max_level_reached': 0,
                'total_recovered_amount': 0.0,
                'total_lost_amount': 0.0,
                'martingale_win_rate': 0.0,
                'average_recovery_level': 0.0,
                'last_martingale_result': None
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
        
        # 🎯 ESTADO ATUAL DO MARTINGALE (por símbolo)
        self.current_martingale_state = {}
    
    def initialize_martingale_state(self, symbol):
        """Inicializa estado do Martingale para um símbolo"""
        if symbol not in self.current_martingale_state:
            self.current_martingale_state[symbol] = {
                'is_active': False,
                'current_level': 0,
                'next_amount': self.martingale_config['base_amount'],
                'total_invested': 0.0,
                'sequence_start_time': None,
                'last_direction': None,
                'consecutive_losses': 0,
                'recovery_target': 0.0,
                'pause_until': None
            }
    
    def calculate_martingale_amount(self, symbol, current_balance):
        """
        🎯 CALCULA VALOR DO MARTINGALE INTELIGENTE
        """
        try:
            self.initialize_martingale_state(symbol)
            state = self.current_martingale_state[symbol]
            config = self.martingale_config
            
            # Verificar se está em pausa
            if state['pause_until'] and datetime.now() < state['pause_until']:
                return {
                    'action': 'pause',
                    'amount': config['base_amount'],
                    'level': 0,
                    'reason': 'Martingale em pausa após stop loss'
                }
            
            # Se não está em Martingale, usar valor base
            if not state['is_active']:
                return {
                    'action': 'start_fresh',
                    'amount': config['base_amount'],
                    'level': 1,
                    'reason': 'Nova sequência iniciada'
                }
            
            # Verificar limites de segurança
            next_level = state['current_level'] + 1
            
            if next_level > config['max_levels']:
                return {
                    'action': 'stop_loss',
                    'amount': config['base_amount'],
                    'level': 0,
                    'reason': f'Limite máximo de {config["max_levels"]} níveis atingido'
                }
            
            # Calcular próximo valor
            multiplier = self.ml_memory['martingale_intelligence']['dynamic_multiplier']
            next_amount = config['base_amount'] * (multiplier ** (next_level - 1))
            
            # Verificar se o saldo suporta
            safety_reserve = current_balance * (config['safety_balance_percent'] / 100)
            available_balance = current_balance - safety_reserve
            
            if next_amount > available_balance:
                return {
                    'action': 'insufficient_balance',
                    'amount': config['base_amount'],
                    'level': 0,
                    'reason': f'Saldo insuficiente. Necessário: {next_amount:.2f}, Disponível: {available_balance:.2f}'
                }
            
            # Verificar stop loss percentual
            total_potential_loss = state['total_invested'] + next_amount
            stop_loss_limit = current_balance * (config['stop_loss_percent'] / 100)
            
            if total_potential_loss > stop_loss_limit:
                return {
                    'action': 'stop_loss',
                    'amount': config['base_amount'],
                    'level': 0,
                    'reason': f'Stop loss ativado. Perda potencial: {total_potential_loss:.2f}'
                }
            
            return {
                'action': 'continue_martingale',
                'amount': next_amount,
                'level': next_level,
                'reason': f'Martingale nível {next_level}: {next_amount:.2f} USD'
            }
            
        except Exception as e:
            print(f"❌ Erro no cálculo Martingale: {e}")
            return {
                'action': 'error',
                'amount': self.martingale_config['base_amount'],
                'level': 0,
                'reason': 'Erro no cálculo'
            }
    
    def should_use_martingale(self, symbol, signal_confidence, market_conditions):
        """
        🧠 ML DECIDE SE DEVE USAR MARTINGALE
        """
        try:
            martingale_intelligence = self.ml_memory['martingale_intelligence']
            
            # Fatores para decidir usar Martingale
            use_score = 0
            reasons = []
            
            # 1. Confiança do sinal
            if signal_confidence >= martingale_intelligence['martingale_confidence_threshold']:
                use_score += 2
                reasons.append(f"Alta confiança ({signal_confidence}%)")
            else:
                use_score -= 1
                reasons.append(f"Baixa confiança ({signal_confidence}%)")
            
            # 2. Performance histórica do símbolo no Martingale
            symbol_perf = self.ml_memory['symbol_performance'].get(symbol, {})
            if symbol_perf.get('martingale_win_rate', 0) > 0.6:
                use_score += 1
                reasons.append("Bom histórico Martingale")
            elif symbol_perf.get('martingale_win_rate', 0) < 0.4:
                use_score -= 1
                reasons.append("Histórico Martingale ruim")
            
            # 3. Sessão de trading atual
            current_hour = datetime.now().hour
            if current_hour in [9, 10, 14, 15, 20, 21]:  # Horários de maior volume
                use_score += 0.5
                reasons.append("Horário de alto volume")
            
            # 4. Condições de mercado
            if market_conditions.get('near_support_resistance', False):
                use_score += 1
                reasons.append("Próximo a suporte/resistência")
            
            if market_conditions.get('low_volatility', False):
                use_score += 0.5
                reasons.append("Baixa volatilidade")
            
            # 5. Limite de perdas consecutivas
            state = self.current_martingale_state.get(symbol, {})
            if state.get('consecutive_losses', 0) >= martingale_intelligence['consecutive_losses_limit']:
                use_score -= 3
                reasons.append(f"Muitas perdas consecutivas ({state['consecutive_losses']})")
            
            # Decisão final
            should_use = use_score >= 1.5
            
            return {
                'should_use': should_use,
                'score': use_score,
                'reasons': reasons,
                'confidence': min(95, max(50, signal_confidence + (use_score * 5)))
            }
            
        except Exception as e:
            print(f"❌ Erro na decisão Martingale: {e}")
            return {'should_use': False, 'score': 0, 'reasons': ['Erro na análise']}
    
    def update_martingale_state(self, symbol, result, amount_used, level_used):
        """
        📊 ATUALIZA ESTADO DO MARTINGALE APÓS RESULTADO
        """
        try:
            self.initialize_martingale_state(symbol)
            state = self.current_martingale_state[symbol]
            martingale_stats = self.ml_memory['martingale_stats']
            
            if result == 'win':
                if state['is_active']:
                    # Recuperação bem-sucedida no Martingale
                    recovered_amount = state['total_invested'] + (amount_used * 0.85)  # 85% payout típico
                    
                    print(f"✅ MARTINGALE WIN! Recuperou {recovered_amount:.2f} USD em nível {level_used}")
                    
                    # Atualizar estatísticas
                    martingale_stats['successful_recoveries'] += 1
                    martingale_stats['total_recovered_amount'] += recovered_amount
                    martingale_stats['max_level_reached'] = max(martingale_stats['max_level_reached'], level_used)
                    
                    # Atualizar taxa de sucesso por nível
                    intelligence = self.ml_memory['martingale_intelligence']
                    current_rate = intelligence['success_rate_by_level'].get(level_used, 0.5)
                    intelligence['success_rate_by_level'][level_used] = min(0.95, current_rate + 0.05)
                    
                    # Resetar estado
                    state['is_active'] = False
                    state['current_level'] = 0
                    state['total_invested'] = 0.0
                    state['consecutive_losses'] = 0
                    state['next_amount'] = self.martingale_config['base_amount']
                else:
                    # Win simples (não Martingale)
                    state['consecutive_losses'] = 0
                
            elif result == 'loss':
                if state['is_active']:
                    # Continuar Martingale
                    state['current_level'] = level_used
                    state['total_invested'] += amount_used
                    state['consecutive_losses'] += 1
                    
                    # Verificar se deve parar
                    if level_used >= self.martingale_config['max_levels']:
                        print(f"❌ MARTINGALE STOP LOSS! Perdeu {state['total_invested']:.2f} USD")
                        
                        # Atualizar estatísticas de falha
                        martingale_stats['failed_recoveries'] += 1
                        martingale_stats['total_lost_amount'] += state['total_invested']
                        
                        # Pausar Martingale por 30 minutos
                        state['pause_until'] = datetime.now() + timedelta(minutes=30)
                        
                        # Resetar estado
                        state['is_active'] = False
                        state['current_level'] = 0
                        state['total_invested'] = 0.0
                        state['next_amount'] = self.martingale_config['base_amount']
                else:
                    # Primeira perda - iniciar Martingale
                    state['is_active'] = True
                    state['current_level'] = level_used
                    state['total_invested'] = amount_used
                    state['consecutive_losses'] = 1
                    state['sequence_start_time'] = datetime.now()
            
            # Recalcular estatísticas gerais do Martingale
            total_sequences = martingale_stats['successful_recoveries'] + martingale_stats['failed_recoveries']
            if total_sequences > 0:
                martingale_stats['martingale_win_rate'] = martingale_stats['successful_recoveries'] / total_sequences
            
            # Atualizar performance do símbolo
            self._update_symbol_martingale_performance(symbol, result, level_used)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao atualizar estado Martingale: {e}")
            return False
    
    def _update_symbol_martingale_performance(self, symbol, result, level):
        """Atualiza performance do Martingale por símbolo"""
        try:
            if symbol not in self.ml_memory['symbol_performance']:
                self.ml_memory['symbol_performance'][symbol] = {
                    'total_signals': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5,
                    'martingale_sequences': 0, 'martingale_wins': 0, 'martingale_win_rate': 0.5,
                    'best_martingale_level': 1, 'last_signals': []
                }
            
            perf = self.ml_memory['symbol_performance'][symbol]
            
            # Se foi resultado de Martingale
            if self.current_martingale_state.get(symbol, {}).get('is_active', False) or result == 'win':
                if result == 'win' and level > 1:  # Recovery no Martingale
                    perf['martingale_sequences'] += 1
                    perf['martingale_wins'] += 1
                    perf['best_martingale_level'] = level if level < perf.get('best_martingale_level', 5) else perf['best_martingale_level']
                elif result == 'loss' and level >= self.martingale_config['max_levels']:  # Falha no Martingale
                    perf['martingale_sequences'] += 1
            
            # Recalcular win rate do Martingale
            if perf['martingale_sequences'] > 0:
                perf['martingale_win_rate'] = perf['martingale_wins'] / perf['martingale_sequences']
            
        except Exception as e:
            print(f"❌ Erro ao atualizar performance do símbolo: {e}")
    
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
                if i > 0:
                    prev_direction = "green" if closes[-1] > opens[-1] else "red"
                    
                    # 65% chance de retração
                    if random.random() < 0.65:
                        direction = "red" if prev_direction == "green" else "green"
                    else:
                        direction = prev_direction
                else:
                    direction = random.choice(["green", "red"])
                
                # Gerar OHLC
                open_price = current_price
                
                if direction == "green":
                    change = random.uniform(0.0001, volatility * 2)
                    close_price = open_price * (1 + change)
                    high_price = close_price * (1 + random.uniform(0, volatility * 0.5))
                    low_price = open_price * (1 - random.uniform(0, volatility * 0.3))
                else:
                    change = random.uniform(0.0001, volatility * 2)
                    close_price = open_price * (1 - change)
                    high_price = open_price * (1 + random.uniform(0, volatility * 0.3))
                    low_price = close_price * (1 - random.uniform(0, volatility * 0.5))
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                prices.append(close_price)
                
                current_price = close_price
            
            return {
                'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes,
                'prices': prices, 'current_price': closes[-1], 'symbol': symbol,
                'data_source': 'OTC com Padrões de Retração + Martingale'
            }
            
        except Exception as e:
            print(f"❌ Erro ao gerar dados: {e}")
            return None
    
    def retraction_strategy(self, opens, closes):
        """Estratégia de retração de vela"""
        try:
            if len(opens) < 2 or len(closes) < 2:
                return None, "Dados insuficientes"
            
            prev_open = opens[-2]
            prev_close = closes[-2]
            
            candle_size = abs(prev_close - prev_open)
            price_pct_change = (candle_size / prev_open) * 100
            
            if prev_close > prev_open:
                prev_direction = "green"
                retraction_signal = "put"
                signal_reason = f"Vela verde anterior (+{price_pct_change:.3f}%) → Retração PUT"
            elif prev_close < prev_open:
                prev_direction = "red"
                retraction_signal = "call"
                signal_reason = f"Vela vermelha anterior (-{price_pct_change:.3f}%) → Retração CALL"
            else:
                prev_direction = "doji"
                retraction_signal = "call" if random.random() > 0.5 else "put"
                signal_reason = "Doji anterior → Sinal neutro"
            
            return {
                'signal': retraction_signal,
                'prev_direction': prev_direction,
                'candle_size_pct': price_pct_change,
                'reason': signal_reason,
                'confidence_base': min(85, 65 + (price_pct_change * 1000))
            }, signal_reason
            
        except Exception as e:
            print(f"❌ Erro na estratégia: {e}")
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
    
    def generate_signal_with_martingale(self, symbol, current_balance=1000):
        """
        🎯 GERA SINAL COM MARTINGALE + ML
        """
        try:
            print(f"\n🤖 Análise RETRAÇÃO + MARTINGALE + ML para {symbol}")
            
            # Gerar dados OTC
            data = self.generate_otc_data(symbol)
            if not data:
                return self._error_response("Falha ao gerar dados")
            
            opens, closes = data['opens'], data['closes']
            current_price = data['current_price']
            
            # Estratégia de retração
            retraction_result, retraction_reason = self.retraction_strategy(opens, closes)
            if not retraction_result:
                return self._error_response("Falha na retração")
            
            print(f"📊 {retraction_reason}")
            
            # Indicadores complementares
            rsi = self.calculate_rsi(closes)
            config = self.otc_config.get(symbol, self.otc_config['EURUSD-OTC'])
            support, resistance = config['support_resistance']
            
            # Condições de mercado para Martingale
            market_conditions = {
                'near_support_resistance': (current_price <= support * 1.005) or (current_price >= resistance * 0.995),
                'low_volatility': config['volatility'] < 0.01,
                'rsi_extreme': rsi < 30 or rsi > 70
            }
            
            # 🎯 CALCULAR MARTINGALE
            martingale_calc = self.calculate_martingale_amount(symbol, current_balance)
            martingale_decision = self.should_use_martingale(symbol, retraction_result['confidence_base'], market_conditions)
            
            # Determinar valor final e nível
            if martingale_calc['action'] in ['pause', 'stop_loss', 'insufficient_balance']:
                final_amount = self.martingale_config['base_amount']
                martingale_level = 0
                martingale_active = False
                martingale_info = martingale_calc['reason']
            elif martingale_decision['should_use'] and martingale_calc['action'] in ['continue_martingale', 'start_fresh']:
                final_amount = martingale_calc['amount']
                martingale_level = martingale_calc['level']
                martingale_active = True
                martingale_info = f"Martingale Nível {martingale_level}: {final_amount:.2f} USD"
            else:
                final_amount = self.martingale_config['base_amount']
                martingale_level = 1
                martingale_active = False
                martingale_info = "Martingale não recomendado pelo ML"
            
            # Ajustar confiança com base no Martingale
            base_confidence = retraction_result['confidence_base']
            if martingale_active and martingale_level > 1:
                # Mais confiante em Martingale (precisa recuperar)
                final_confidence = min(95, base_confidence + (martingale_level * 5))
                confidence_reason = f"Boosted +{martingale_level * 5}% (Martingale recovery)"
            else:
                final_confidence = base_confidence
                confidence_reason = "Base confidence"
            
            # Timeframe baseado no Martingale
            if martingale_level >= 3:
                timeframe = {"type": "minutes", "duration": 1}  # Mais agressivo em níveis altos
            elif martingale_level == 2:
                timeframe = {"type": "minutes", "duration": 2}
            else:
                timeframe = {"type": "minutes", "duration": 3}
            
            print(f"🎯 Sinal: {retraction_result['signal'].upper()}")
            print(f"💰 Valor: {final_amount:.2f} USD (Nível {martingale_level})")
            print(f"📊 Confiança: {final_confidence:.1f}% ({confidence_reason})")
            print(f"🎰 Martingale: {'ATIVO' if martingale_active else 'INATIVO'}")
            
            # Estado atual do Martingale
            current_state = self.current_martingale_state.get(symbol, {})
            martingale_stats = self.ml_memory['martingale_stats']
            
            reasoning = f"{retraction_reason} | Martingale L{martingale_level}: {final_amount:.2f} USD"
            
            return {
                'status': 'success',
                'symbol': symbol,
                'direction': retraction_result['signal'],
                'confidence': round(final_confidence, 1),
                'reasoning': reasoning,
                'strategy': 'Retração + Martingale + Machine Learning',
                
                # Análise de retração
                'retraction_analysis': {
                    'prev_candle_direction': retraction_result['prev_direction'],
                    'candle_size_percent': round(retraction_result['candle_size_pct'], 4),
                    'retraction_signal': retraction_result['signal'],
                    'base_confidence': retraction_result['confidence_base']
                },
                
                # 🎰 INFORMAÇÕES DO MARTINGALE
                'martingale_analysis': {
                    'is_active': martingale_active,
                    'current_level': martingale_level,
                    'amount_to_invest': final_amount,
                    'martingale_action': martingale_calc['action'],
                    'martingale_reason': martingale_info,
                    'should_use_ml_decision': martingale_decision['should_use'],
                    'ml_score': martingale_decision['score'],
                    'ml_reasons': martingale_decision['reasons'],
                    'total_invested_in_sequence': current_state.get('total_invested', 0.0),
                    'consecutive_losses': current_state.get('consecutive_losses', 0),
                    'recovery_potential': final_amount * 0.85 if martingale_active else 0
                },
                
                # Estatísticas do Martingale
                'martingale_stats': {
                    'total_sequences': martingale_stats['total_martingale_sequences'],
                    'success_rate': f"{martingale_stats['martingale_win_rate']*100:.1f}%",
                    'successful_recoveries': martingale_stats['successful_recoveries'],
                    'failed_recoveries': martingale_stats['failed_recoveries'],
                    'max_level_reached': martingale_stats['max_level_reached'],
                    'total_recovered': round(martingale_stats['total_recovered_amount'], 2),
                    'total_lost': round(martingale_stats['total_lost_amount'], 2)
                },
                
                # Análise de mercado
                'market_analysis': {
                    'current_price': round(current_price, 5),
                    'rsi': round(rsi, 1),
                    'near_support_resistance': market_conditions['near_support_resistance'],
                    'support_level': support,
                    'resistance_level': resistance,
                    'market_conditions': market_conditions
                },
                
                'optimal_timeframe': timeframe,
                'data_source': data['data_source'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return self._error_response(f"Erro: {str(e)}")
    
    def update_ml_feedback_with_martingale(self, symbol, predicted_signal, actual_result, amount_used, martingale_level):
        """
        📚 FEEDBACK ML + MARTINGALE
        """
        try:
            print(f"📚 ML + Martingale Learning: {symbol} - L{martingale_level} {predicted_signal} = {actual_result}")
            
            # Atualizar estado do Martingale
            self.update_martingale_state(symbol, actual_result, amount_used, martingale_level)
            
            # Atualizar performance geral (mesmo código anterior)
            if symbol not in self.ml_memory['symbol_performance']:
                self.ml_memory['symbol_performance'][symbol] = {
                    'total_signals': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5, 'last_signals': [],
                    'martingale_sequences': 0, 'martingale_wins': 0, 'martingale_win_rate': 0.5
                }
            
            perf = self.ml_memory['symbol_performance'][symbol]
            perf['total_signals'] += 1
            
            if actual_result == 'win':
                perf['wins'] += 1
                self._reinforce_weights(True)
            elif actual_result == 'loss':
                perf['losses'] += 1
                self._reinforce_weights(False)
            
            perf['win_rate'] = perf['wins'] / perf['total_signals'] if perf['total_signals'] > 0 else 0.5
            
            # Atualizar estatísticas globais
            global_stats = self.ml_memory['global_stats']
            global_stats['total_signals'] += 1
            
            if actual_result == 'win':
                global_stats['total_wins'] += 1
            elif actual_result == 'loss':
                global_stats['total_losses'] += 1
            
            global_stats['win_rate'] = global_stats['total_wins'] / global_stats['total_signals'] if global_stats['total_signals'] > 0 else 0.5
            global_stats['last_updated'] = datetime.now().isoformat()
            
            print(f"📊 {symbol}: {perf['wins']}W/{perf['losses']}L = {perf['win_rate']*100:.1f}%")
            print(f"🎰 Martingale: {perf.get('martingale_wins', 0)}/{perf.get('martingale_sequences', 0)} = {perf.get('martingale_win_rate', 0)*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no feedback: {e}")
            return False
    
    def _reinforce_weights(self, success):
        """Ajusta pesos ML"""
        try:
            weights = self.ml_memory['indicator_weights']
            learning_rate = self.ml_memory['adaptive_params']['learning_rate']
            
            adjustment = learning_rate * 0.1
            if success:
                for key in weights:
                    weights[key] = min(2.0, weights[key] + adjustment)
            else:
                for key in weights:
                    weights[key] = max(0.1, weights[key] - adjustment)
                    
        except Exception as e:
            print(f"❌ Erro ao ajustar pesos: {e}")
    
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

analyzer = RetractionMartingaleML()

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def home():
    global_stats = analyzer.ml_memory['global_stats']
    martingale_stats = analyzer.ml_memory['martingale_stats']
    
    return jsonify({
        'status': 'success',
        'message': '🎰 IA RETRAÇÃO + MARTINGALE + MACHINE LEARNING',
        'version': '8.0.0 - RETRACTION + MARTINGALE + ML',
        'strategy': 'Retração de Vela + Martingale Inteligente + Aprendizado',
        'features': [
            '🎯 Estratégia de Retração de Vela',
            '🎰 Martingale Inteligente com ML',
            '🧠 Machine Learning que aprende quando usar Martingale',
            '💰 Gestão de risco dinâmica',
            '🛡️ Stop Loss automático',
            '📊 Controle de níveis máximos',
            '⏸️ Pausa automática após perdas',
            '📈 Estatísticas detalhadas de recuperação'
        ],
        'martingale_config': {
            'max_levels': analyzer.martingale_config['max_levels'],
            'multiplier': analyzer.martingale_config['multiplier'],
            'max_total_risk': analyzer.martingale_config['max_total_risk'],
            'stop_loss_percent': analyzer.martingale_config['stop_loss_percent']
        },
        'ml_stats': {
            'total_signals': global_stats['total_signals'],
            'global_win_rate': f"{global_stats['win_rate']*100:.1f}%",
            'martingale_sequences': martingale_stats['total_martingale_sequences'],
            'martingale_success_rate': f"{martingale_stats['martingale_win_rate']*100:.1f}%",
            'total_recovered': martingale_stats['total_recovered_amount'],
            'learning_active': True
        },
        'supported_symbols': list(analyzer.otc_config.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signal', methods=['POST', 'GET'])
@app.route('/trading-signal', methods=['POST', 'GET'])
def get_signal():
    """Endpoint principal - RETRAÇÃO + MARTINGALE + ML"""
    
    if request.method == 'GET':
        symbol = 'EURUSD-OTC'
        current_balance = 1000
    else:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD-OTC')
        current_balance = data.get('current_balance', 1000)
    
    print(f"\n🔄 Análise RETRAÇÃO + MARTINGALE + ML para {symbol}")
    
    # Gerar sinal com Martingale
    result = analyzer.generate_signal_with_martingale(symbol, current_balance)
    return jsonify(result)

@app.route('/feedback', methods=['POST'])
def ml_martingale_feedback():
    """
    📚 Endpoint para FEEDBACK do ML + MARTINGALE
    
    {
        "symbol": "EURUSD-OTC",
        "predicted_signal": "call", 
        "actual_result": "win",
        "amount_used": 2.0,
        "martingale_level": 2
    }
    """
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol')
        predicted_signal = data.get('predicted_signal')
        actual_result = data.get('actual_result')
        amount_used = data.get('amount_used', 1.0)
        martingale_level = data.get('martingale_level', 1)
        
        if not all([symbol, predicted_signal, actual_result]):
            return jsonify({
                'status': 'error',
                'message': 'Parâmetros obrigatórios: symbol, predicted_signal, actual_result, amount_used, martingale_level'
            }), 400
        
        if actual_result not in ['win', 'loss', 'tie']:
            return jsonify({
                'status': 'error',
                'message': 'actual_result deve ser: win, loss, ou tie'
            }), 400
        
        # Atualizar ML com Martingale
        success = analyzer.update_ml_feedback_with_martingale(
            symbol, predicted_signal, actual_result, amount_used, martingale_level
        )
        
        if success:
            # Retornar estatísticas
            symbol_perf = analyzer.ml_memory['symbol_performance'].get(symbol, {})
            global_stats = analyzer.ml_memory['global_stats']
            martingale_stats = analyzer.ml_memory['martingale_stats']
            martingale_state = analyzer.current_martingale_state.get(symbol, {})
            
            return jsonify({
                'status': 'success',
                'message': f'ML + Martingale atualizado: {symbol} - L{martingale_level} {predicted_signal} = {actual_result}',
                'learning_updated': True,
                
                'symbol_performance': {
                    'symbol': symbol,
                    'win_rate': f"{symbol_perf.get('win_rate', 0)*100:.1f}%",
                    'total_signals': symbol_perf.get('total_signals', 0),
                    'martingale_win_rate': f"{symbol_perf.get('martingale_win_rate', 0)*100:.1f}%"
                },
                
                'martingale_status': {
                    'is_active': martingale_state.get('is_active', False),
                    'current_level': martingale_state.get('current_level', 0),
                    'consecutive_losses': martingale_state.get('consecutive_losses', 0),
                    'total_invested': martingale_state.get('total_invested', 0.0),
                    'next_amount': martingale_state.get('next_amount', analyzer.martingale_config['base_amount'])
                },
                
                'global_martingale_stats': {
                    'success_rate': f"{martingale_stats['martingale_win_rate']*100:.1f}%",
                    'total_sequences': martingale_stats['total_martingale_sequences'],
                    'successful_recoveries': martingale_stats['successful_recoveries'],
                    'failed_recoveries': martingale_stats['failed_recoveries'],
                    'total_recovered': round(martingale_stats['total_recovered_amount'], 2),
                    'total_lost': round(martingale_stats['total_lost_amount'], 2)
                },
                
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Falha ao atualizar ML + Martingale'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro no feedback: {str(e)}'
        }), 500

@app.route('/martingale-status/<symbol>')
def martingale_status(symbol):
    """Status atual do Martingale para um símbolo"""
    try:
        analyzer.initialize_martingale_state(symbol)
        state = analyzer.current_martingale_state[symbol]
        martingale_stats = analyzer.ml_memory['martingale_stats']
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'martingale_state': {
                'is_active': state['is_active'],
                'current_level': state['current_level'],
                'next_amount': state['next_amount'],
                'total_invested': state['total_invested'],
                'consecutive_losses': state['consecutive_losses'],
                'pause_until': state['pause_until'].isoformat() if state['pause_until'] else None
            },
            'global_martingale_stats': martingale_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao obter status: {str(e)}'
        }), 500

@app.route('/health')
def health():
    global_stats = analyzer.ml_memory['global_stats']
    martingale_stats = analyzer.ml_memory['martingale_stats']
    
    return jsonify({
        'status': 'healthy',
        'message': '🟢 IA RETRAÇÃO + MARTINGALE + ML Online',
        'strategy': 'Retração + Martingale Inteligente + Machine Learning',
        'ml_active': True,
        'martingale_active': True,
        'global_win_rate': f"{global_stats['win_rate']*100:.1f}%",
        'martingale_success_rate': f"{martingale_stats['martingale_win_rate']*100:.1f}%",
        'total_signals_processed': global_stats['total_signals'],
        'timestamp': datetime.now().isoformat()
    })

# ===============================================
# INICIALIZAÇÃO
# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 IA RETRAÇÃO + MARTINGALE + MACHINE LEARNING Iniciando...")
    print("🎯 Estratégia: Retração de Vela")
    print("🎰 Martingale: Inteligente com ML")
    print("🧠 Machine Learning: Aprende quando usar Martingale")
    print("💰 Gestão de Risco: Stop Loss automático")
    print("📊 Controle: Máximo 5 níveis, 20% stop loss")
    print("⏸️ Segurança: Pausa automática após perdas")
    print("🛡️ Proteção: 5% do saldo sempre preservado")
    print("✅ Sistema completo ativado!")
    print(f"🌐 Porta: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
