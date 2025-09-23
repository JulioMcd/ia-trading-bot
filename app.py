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

    def analyze_market_conditions(self, signals):
        """Analisa condições de mercado para determinar melhor momento de entrada"""
        total_signals = len(signals)
        if total_signals == 0:
            return False, 0.0

        positive_signals = sum(1 for signal in signals if signal['strength'] > 0.5)
        signal_strength = sum(signal['strength'] for signal in signals) / total_signals

        # Condições ideais para entrada
        is_good_entry = (positive_signals / total_signals) >= 0.7 and signal_strength >= 0.6

        return is_good_entry, signal_strength

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
