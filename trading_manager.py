import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingManager:
    def __init__(self):
        # Configurações de gerenciamento de banca
        self.initial_balance = 100.0  # Saldo inicial
        self.current_balance = self.initial_balance
        self.risk_per_trade = 0.02  # 2% do saldo por trade
        self.max_risk_per_day = 0.10  # 10% do saldo por dia
        
        # Controle de tempo entre operações
        self.last_trade_time = None
        self.min_time_between_trades = 60  # 1 minuto em segundos
        self.min_timeframe = 15  # Mínimo de 15 ticks
        
        # Configurações de análise técnica
        self.volatility_threshold = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }
        
        # Parâmetros dos indicadores ajustáveis por volatilidade
        self.indicator_params = {
            'rsi': {
                'low': {'period': 14, 'overbought': 70, 'oversold': 30},
                'medium': {'period': 10, 'overbought': 75, 'oversold': 25},
                'high': {'period': 7, 'overbought': 80, 'oversold': 20}
            },
            'ma': {
                'low': {'fast': 5, 'slow': 20},
                'medium': {'fast': 3, 'slow': 15},
                'high': {'fast': 2, 'slow': 10}
            }
        }
        
        # Configurações de Stop
        self.stop_gain = 0.15  # 15% de lucro diário
        self.stop_loss = -0.10  # 10% de perda diária
        
        # Configurações de Martingale
        self.martingale_factor = 2.0  # Multiplicador para martingale
        self.max_martingale = 3  # Número máximo de martingales
        self.current_martingale = 0
        
        # Controle de resultados
        self.daily_results = []
        self.consecutive_losses = 0
        self.daily_profit = 0.0
        
        # Data de referência para controle diário
        self.current_trading_day = datetime.now().date()
        
        # Níveis de confiança para ajuste de stake
        self.confidence_levels = {
            'very_high': 85,  # Aumenta stake em 50%
            'high': 75,      # Aumenta stake em 25%
            'medium': 65,    # Stake normal
            'low': 55        # Diminui stake em 25%
        }

    def reset_daily_stats(self):
        """Reseta estatísticas diárias"""
        self.daily_profit = 0.0
        self.consecutive_losses = 0
        self.current_martingale = 0
        self.daily_results = []
        self.current_trading_day = datetime.now().date()

    def check_new_day(self):
        """Verifica se é um novo dia de trading"""
        current_date = datetime.now().date()
        if current_date != self.current_trading_day:
            self.reset_daily_stats()
            return True
        return False

    def calculate_position_size(self, confidence, symbol_volatility):
        """Calcula o tamanho da posição baseado na confiança e volatilidade"""
        # Verificar limites diários
        if self.daily_profit <= self.stop_loss * self.initial_balance:
            logger.warning("Stop Loss diário atingido")
            return 0.0
        
        if self.daily_profit >= self.stop_gain * self.initial_balance:
            logger.warning("Stop Gain diário atingido")
            return 0.0

        # Base stake
        base_stake = self.current_balance * self.risk_per_trade

        # Ajuste por confiança
        if confidence >= self.confidence_levels['very_high']:
            confidence_multiplier = 1.5
        elif confidence >= self.confidence_levels['high']:
            confidence_multiplier = 1.25
        elif confidence >= self.confidence_levels['medium']:
            confidence_multiplier = 1.0
        else:
            confidence_multiplier = 0.75

        # Ajuste por volatilidade
        volatility_multiplier = 1.0
        if symbol_volatility > 2.0:
            volatility_multiplier = 0.7
        elif symbol_volatility < 0.5:
            volatility_multiplier = 1.2

        # Ajuste por martingale
        if self.current_martingale > 0:
            martingale_multiplier = self.martingale_factor ** self.current_martingale
        else:
            martingale_multiplier = 1.0

        # Cálculo final do stake
        stake = base_stake * confidence_multiplier * volatility_multiplier * martingale_multiplier

        # Limites de segurança
        max_stake = self.current_balance * 0.1  # Máximo de 10% do saldo por operação
        stake = min(stake, max_stake)
        stake = max(stake, 1.0)  # Stake mínimo de 1.0

        return round(stake, 2)

    def process_trade_result(self, result, stake, pnl):
        """Processa o resultado de um trade"""
        self.check_new_day()
        
        self.current_balance += pnl
        self.daily_profit += pnl
        self.last_trade_time = datetime.now()  # Atualiza tempo da última operação
        self.daily_results.append({
            'result': result,
            'stake': stake,
            'pnl': pnl,
            'balance': self.current_balance,
            'timestamp': datetime.now().isoformat()
        })

        if result == 'win':
            self.consecutive_losses = 0
            self.current_martingale = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses <= self.max_martingale:
                self.current_martingale = self.consecutive_losses
            else:
                self.current_martingale = 0  # Reset após máximo de martingales

        return {
            'current_balance': self.current_balance,
            'daily_profit': self.daily_profit,
            'daily_profit_percentage': (self.daily_profit / self.initial_balance) * 100,
            'can_trade': self.can_trade(),
            'martingale_level': self.current_martingale
        }

    def can_trade(self):
        """Verifica se pode realizar novas operações"""
        # Verificar stop loss
        if self.daily_profit <= self.stop_loss * self.initial_balance:
            return False
        
        # Verificar stop gain
        if self.daily_profit >= self.stop_gain * self.initial_balance:
            return False
        
        # Verificar risco diário máximo
        daily_risk = abs(sum(trade['pnl'] for trade in self.daily_results if trade['pnl'] < 0))
        if daily_risk >= self.max_risk_per_day * self.initial_balance:
            return False
        
        return True

    def analyze_market_conditions(self, signals, timeframe=None, last_trade_time=None, price_data=None, candles=None):
        """Analisa condições de mercado para determinar melhor momento de entrada"""
        current_time = datetime.now()
        
        # Verifica se estamos no início do minuto (primeiros 10 segundos)
        if current_time.second > 10:
            return False, 0.0, "Aguardando início do próximo minuto para entrada"
            
        # Verifica se o timeframe é adequado
        if timeframe and timeframe < self.min_timeframe:
            return False, 0.0, "Timeframe muito curto"

        # Verifica tempo mínimo entre operações
        if last_trade_time:
            time_diff = (current_time - last_trade_time).total_seconds()
            if time_diff < self.min_time_between_trades:
                return False, 0.0, "Aguardando tempo mínimo entre operações"

        # Análise de volatilidade e ajuste de parâmetros
        if price_data:
            volatility_level = self.analyze_volatility(price_data)
            adjusted_params = self.get_adjusted_parameters(volatility_level)
        else:
            volatility_level = 'medium'
            adjusted_params = self.get_adjusted_parameters('medium')

        # Análise de Price Action
        if candles:
            price_action_signal, pa_strength = self.analyze_price_action(candles)
        else:
            price_action_signal, pa_strength = False, 0.0

        total_signals = len(signals)
        if total_signals == 0:
            return False, 0.0, "Sem sinais disponíveis"

        # Análise dos sinais com base na volatilidade
        threshold = 0.8 if volatility_level == 'high' else 0.7 if volatility_level == 'medium' else 0.6
        positive_signals = sum(1 for signal in signals if signal['strength'] > threshold)
        signal_strength = sum(signal['strength'] for signal in signals) / total_signals

        # Combina sinais técnicos com price action
        combined_strength = (signal_strength + pa_strength) / 2 if candles else signal_strength

        # Condições para entrada adaptadas à volatilidade
        min_positive_ratio = 0.8 if volatility_level == 'high' else 0.7
        min_strength = 0.7 if volatility_level == 'high' else 0.65

        is_good_entry = (
            (positive_signals / total_signals) >= min_positive_ratio and
            combined_strength >= min_strength and
            self.consecutive_losses < 2 and
            (not candles or price_action_signal)
        )

        message = f"Volatilidade: {volatility_level}, " + \
                 ("Condições favoráveis" if is_good_entry else "Condições desfavoráveis")
        
        return is_good_entry, combined_strength, message

    def analyze_volatility(self, price_data):
        """Analisa a volatilidade do mercado e retorna os parâmetros ajustados"""
        if len(price_data) < 20:
            return 'medium'  # volatilidade padrão se não houver dados suficientes
            
        # Calcula a volatilidade usando desvio padrão normalizado
        prices = np.array(price_data)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(len(returns))
        
        if volatility > self.volatility_threshold['high']:
            return 'high'
        elif volatility < self.volatility_threshold['low']:
            return 'low'
        return 'medium'
    
    def get_adjusted_parameters(self, volatility_level):
        """Retorna parâmetros dos indicadores ajustados pela volatilidade"""
        return {
            'rsi': self.indicator_params['rsi'][volatility_level],
            'ma': self.indicator_params['ma'][volatility_level]
        }
    
    def analyze_price_action(self, candles):
        """Análise baseada em price action"""
        if len(candles) < 3:
            return False, 0.0
            
        # Identificação de padrões de price action
        last_candles = candles[-3:]
        
        # Verifica tendência
        trend = 'up' if all(c['close'] > c['open'] for c in last_candles) else \
               'down' if all(c['close'] < c['open'] for c in last_candles) else \
               'sideways'
               
        # Calcula força do sinal baseado em padrões
        signal_strength = 0.0
        
        # Doji
        if abs(last_candles[-1]['close'] - last_candles[-1]['open']) < 0.1 * (last_candles[-1]['high'] - last_candles[-1]['low']):
            signal_strength += 0.3
            
        # Hammer ou Shooting Star
        body = abs(last_candles[-1]['close'] - last_candles[-1]['open'])
        upper_wick = last_candles[-1]['high'] - max(last_candles[-1]['close'], last_candles[-1]['open'])
        lower_wick = min(last_candles[-1]['close'], last_candles[-1]['open']) - last_candles[-1]['low']
        
        if lower_wick > 2 * body and upper_wick < 0.2 * body:  # Hammer
            signal_strength += 0.4
        elif upper_wick > 2 * body and lower_wick < 0.2 * body:  # Shooting Star
            signal_strength += 0.4
            
        return trend != 'sideways', signal_strength
    
    def get_trading_stats(self):
        """Retorna estatísticas de trading"""
        total_trades = len(self.daily_results)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'daily_profit': 0,
                'current_balance': self.current_balance
            }

        wins = sum(1 for trade in self.daily_results if trade['result'] == 'win')
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': (wins / total_trades) * 100,
            'daily_profit': self.daily_profit,
            'daily_profit_percentage': (self.daily_profit / self.initial_balance) * 100,
            'current_balance': self.current_balance,
            'martingale_level': self.current_martingale,
            'can_trade': self.can_trade()
        }
