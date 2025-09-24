import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import logging
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuração do Flask
app = Flask(__name__)
CORS(app)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
API_PORT = int(os.environ.get('PORT', 5000))
DATABASE_URL = 'trading_stats_online.db'

class DerivSymbolManager:
    """Gerenciador específico para símbolos e timeframes do Deriv"""
    
    def __init__(self):
        self.symbol_configs = {
            # Volatility Indices
            'R_10': {'base_price': 1000, 'volatility': 0.10, 'tick_size': 0.001},
            'R_25': {'base_price': 1000, 'volatility': 0.25, 'tick_size': 0.001},
            'R_50': {'base_price': 1000, 'volatility': 0.50, 'tick_size': 0.001},
            'R_75': {'base_price': 1000, 'volatility': 0.75, 'tick_size': 0.001},
            'R_100': {'base_price': 1000, 'volatility': 1.00, 'tick_size': 0.001},
            
            # Volatility Indices (1s)
            '1HZ10V': {'base_price': 1000, 'volatility': 0.10, 'tick_size': 0.001, 'high_freq': True},
            '1HZ25V': {'base_price': 1000, 'volatility': 0.25, 'tick_size': 0.001, 'high_freq': True},
            '1HZ50V': {'base_price': 1000, 'volatility': 0.50, 'tick_size': 0.001, 'high_freq': True},
            '1HZ75V': {'base_price': 1000, 'volatility': 0.75, 'tick_size': 0.001, 'high_freq': True},
            '1HZ100V': {'base_price': 1000, 'volatility': 1.00, 'tick_size': 0.001, 'high_freq': True},
            '1HZ150V': {'base_price': 1000, 'volatility': 1.50, 'tick_size': 0.001, 'high_freq': True},
            '1HZ200V': {'base_price': 1000, 'volatility': 2.00, 'tick_size': 0.001, 'high_freq': True},
            '1HZ250V': {'base_price': 1000, 'volatility': 2.50, 'tick_size': 0.001, 'high_freq': True},
            
            # Jump Indices
            'JD10': {'base_price': 1000, 'volatility': 0.10, 'jump_factor': 0.10},
            'JD25': {'base_price': 1000, 'volatility': 0.25, 'jump_factor': 0.25},
            'JD50': {'base_price': 1000, 'volatility': 0.50, 'jump_factor': 0.50},
            'JD75': {'base_price': 1000, 'volatility': 0.75, 'jump_factor': 0.75},
            'JD100': {'base_price': 1000, 'volatility': 1.00, 'jump_factor': 1.00},
            
            # Crash/Boom
            'CRASH300': {'base_price': 8000, 'volatility': 3.0, 'crash_prob': 0.003},
            'CRASH500': {'base_price': 8000, 'volatility': 5.0, 'crash_prob': 0.002},
            'CRASH600': {'base_price': 8000, 'volatility': 6.0, 'crash_prob': 0.00167},
            'CRASH900': {'base_price': 8000, 'volatility': 9.0, 'crash_prob': 0.00111},
            'CRASH1000': {'base_price': 8000, 'volatility': 10.0, 'crash_prob': 0.001},
            'BOOM300': {'base_price': 1000, 'volatility': 3.0, 'boom_prob': 0.003},
            'BOOM500': {'base_price': 1000, 'volatility': 5.0, 'boom_prob': 0.002},
            'BOOM600': {'base_price': 1000, 'volatility': 6.0, 'boom_prob': 0.00167},
            'BOOM900': {'base_price': 1000, 'volatility': 9.0, 'boom_prob': 0.00111},
            'BOOM1000': {'base_price': 1000, 'volatility': 10.0, 'boom_prob': 0.001},
            
            # Step Indices
            'STPRAN': {'base_price': 1000, 'volatility': 1.0, 'step_size': 0.1},
            'STPRAN200': {'base_price': 1000, 'volatility': 2.0, 'step_size': 0.2},
            'STPRAN500': {'base_price': 1000, 'volatility': 5.0, 'step_size': 0.5},
            
            # Market Indices
            'RDBEAR': {'base_price': 5000, 'volatility': 1.5, 'trend_bias': -0.1},
            'RDBULL': {'base_price': 5000, 'volatility': 1.5, 'trend_bias': 0.1},
        }
        
        # Mapeamento de timeframes do Deriv
        self.timeframe_mapping = {
            't': {  # Ticks
                1: {'seconds': 1, 'description': '1 tick', 'min_confidence': 85},
                2: {'seconds': 2, 'description': '2 ticks', 'min_confidence': 82},
                3: {'seconds': 3, 'description': '3 ticks', 'min_confidence': 80},
                4: {'seconds': 4, 'description': '4 ticks', 'min_confidence': 78},
                5: {'seconds': 5, 'description': '5 ticks', 'min_confidence': 75},
                6: {'seconds': 6, 'description': '6 ticks', 'min_confidence': 73},
                7: {'seconds': 7, 'description': '7 ticks', 'min_confidence': 72},
                8: {'seconds': 8, 'description': '8 ticks', 'min_confidence': 70},
                9: {'seconds': 9, 'description': '9 ticks', 'min_confidence': 68},
                10: {'seconds': 10, 'description': '10 ticks', 'min_confidence': 65}
            },
            'm': {  # Minutos
                1: {'seconds': 60, 'description': '1 min', 'min_confidence': 65},
                2: {'seconds': 120, 'description': '2 mins', 'min_confidence': 62},
                3: {'seconds': 180, 'description': '3 mins', 'min_confidence': 60},
                4: {'seconds': 240, 'description': '4 mins', 'min_confidence': 58},
                5: {'seconds': 300, 'description': '5 mins', 'min_confidence': 55},
                10: {'seconds': 600, 'description': '10 mins', 'min_confidence': 50},
                15: {'seconds': 900, 'description': '15 mins', 'min_confidence': 50},
                30: {'seconds': 1800, 'description': '30 mins', 'min_confidence': 50},
                60: {'seconds': 3600, 'description': '1 hora', 'min_confidence': 50}
            }
        }
    
    def get_symbol_config(self, symbol):
        """Retorna configuração específica do símbolo"""
        return self.symbol_configs.get(symbol, {
            'base_price': 1000,
            'volatility': 1.0,
            'tick_size': 0.001
        })
    
    def get_optimal_timeframe(self, symbol, confidence, market_conditions):
        """Determina timeframe ótimo baseado no símbolo e condições"""
        config = self.get_symbol_config(symbol)
        volatility = config.get('volatility', 1.0)
        
        # Para símbolos de alta frequência (1s), preferir ticks
        if config.get('high_freq', False):
            if confidence >= 80:
                return {'type': 't', 'duration': 3}
            elif confidence >= 70:
                return {'type': 't', 'duration': 5}
            else:
                return {'type': 't', 'duration': 7}
        
        # Para símbolos de alta volatilidade, usar timeframes maiores
        elif volatility >= 2.0:
            if confidence >= 75:
                return {'type': 'm', 'duration': 2}
            elif confidence >= 65:
                return {'type': 'm', 'duration': 3}
            else:
                return {'type': 'm', 'duration': 5}
        
        # Para volatilidade normal
        else:
            if confidence >= 80:
                return {'type': 't', 'duration': 5}
            elif confidence >= 70:
                return {'type': 'm', 'duration': 1}
            elif confidence >= 60:
                return {'type': 'm', 'duration': 2}
            else:
                return {'type': 'm', 'duration': 3}
    
    def validate_timeframe(self, duration_type, duration):
        """Valida se o timeframe é suportado"""
        if duration_type not in self.timeframe_mapping:
            return False
        return duration in self.timeframe_mapping[duration_type]
    
    def get_timeframe_confidence_threshold(self, duration_type, duration):
        """Retorna threshold de confiança mínima para o timeframe"""
        if not self.validate_timeframe(duration_type, duration):
            return 70
        
        return self.timeframe_mapping[duration_type][duration]['min_confidence']


class EnhancedTradingAI:
    def __init__(self):
        # Gerenciador de símbolos Deriv
        self.deriv_manager = DerivSymbolManager()
        
        # Gerenciador de Trading
        from trading_manager import TradingManager
        self.trading_manager = TradingManager()
        
        # Sistema de ML (mantido do código original)
        self.offline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.online_model = SGDClassifier(
            loss='log_loss', 
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            max_iter=1000
        )
        self.passive_model = PassiveAggressiveClassifier(
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        self.scaler = StandardScaler()
        self.online_scaler = StandardScaler()
        
        # Estados do sistema
        self.offline_trained = False
        self.online_initialized = False
        self.passive_initialized = False
        
        # Buffers para inicialização
        self.feature_buffer = []
        self.target_buffer = []
        self.min_samples_init = 20
        
        # Métricas e controle
        self.online_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': [],
            'last_10_trades': [],
            'learning_updates': 0
        }
        
        # Configurações de IA automatizada
        self.full_ai_mode = {
            'active': False,
            'auto_symbol_selection': False,
            'auto_timeframe_selection': True,
            'auto_stake_management': True,
            'auto_analysis_cycle': 30,  # segundos
            'min_confidence_threshold': 75,
            'preferred_symbols': ['R_50', 'R_25', 'R_100'],
            'risk_level': 'medium'
        }
        
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            # Tabela principal de trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    direction TEXT,
                    stake REAL,
                    duration_type TEXT,
                    duration_value INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    result TEXT,
                    pnl REAL,
                    confidence REAL,
                    timeframe_used TEXT,
                    ai_mode TEXT,
                    features TEXT,
                    market_analysis TEXT
                )
            ''')
            
            # Tabela de configurações de sessão
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    duration_type TEXT,
                    duration_value INTEGER,
                    stake REAL,
                    ai_mode_active BOOLEAN,
                    session_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Banco de dados inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco: {e}")
    
    def get_market_data_for_symbol(self, symbol):
        """Gera dados de mercado específicos para símbolos Deriv"""
        try:
            config = self.deriv_manager.get_symbol_config(symbol)
            base_price = config['base_price']
            volatility = config['volatility']
            
            # Simular série temporal realista
            periods = 100
            prices = [base_price]
            
            # Seed baseado no símbolo para consistência
            np.random.seed(hash(symbol) % 2**32)
            
            for i in range(periods - 1):
                # Movimento base com volatilidade específica
                drift = np.random.normal(0, volatility * 0.001)
                
                # Mean reversion para manter preços estáveis
                mean_reversion = (base_price - prices[-1]) * 0.0001
                
                # Ruído específico do ativo
                noise = np.random.normal(0, volatility * 0.01)
                
                # Fatores especiais por tipo de ativo
                special_factor = 0
                if 'CRASH' in symbol:
                    # Simulação de crashes ocasionais
                    if np.random.random() < config.get('crash_prob', 0.001):
                        special_factor = -volatility * 0.1
                elif 'BOOM' in symbol:
                    # Simulação de booms ocasionais
                    if np.random.random() < config.get('boom_prob', 0.001):
                        special_factor = volatility * 0.1
                elif 'JD' in symbol:
                    # Jump diffusion
                    if np.random.random() < 0.01:  # 1% chance de jump
                        special_factor = np.random.choice([-1, 1]) * config.get('jump_factor', 0.5) * 0.01
                
                new_price = prices[-1] * (1 + drift + mean_reversion + noise + special_factor)
                prices.append(max(0.01, new_price))
            
            # Calcular indicadores técnicos
            close_prices = np.array(prices)
            
            # RSI
            delta = np.diff(close_prices)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # MACD simplificado
            ema12 = np.mean(close_prices[-12:])
            ema26 = np.mean(close_prices[-26:]) if len(close_prices) >= 26 else np.mean(close_prices)
            macd = ema12 - ema26
            
            # Bollinger Bands
            sma20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.mean(close_prices)
            std20 = np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.std(close_prices)
            bb_upper = sma20 + (2 * std20)
            bb_lower = sma20 - (2 * std20)
            
            # Tendência
            recent_prices = close_prices[-10:]
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            trend_strength = abs(trend_slope / close_prices[-1]) * 1000  # Normalizado
            
            if trend_slope > close_prices[-1] * 0.001:
                trend = 'bullish'
            elif trend_slope < -close_prices[-1] * 0.001:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'symbol': symbol,
                'price': float(close_prices[-1]),
                'rsi': float(rsi),
                'macd': float(macd),
                'bb_upper': float(bb_upper),
                'bb_lower': float(bb_lower),
                'volatility': volatility,
                'trend_strength': float(trend_strength),
                'trend': trend,
                'volume': 1000.0,
                'timestamp': datetime.now().isoformat(),
                'symbol_type': self.get_symbol_type(symbol),
                'recommended_timeframes': self.get_recommended_timeframes(symbol, volatility)
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados para {symbol}: {e}")
            return self.get_fallback_data(symbol)
    
    def get_symbol_type(self, symbol):
        """Identifica o tipo do símbolo"""
        if symbol.startswith('R_'):
            return 'volatility_index'
        elif symbol.startswith('1HZ'):
            return 'high_frequency_volatility'
        elif 'CRASH' in symbol or 'BOOM' in symbol:
            return 'crash_boom'
        elif symbol.startswith('JD'):
            return 'jump_diffusion'
        elif 'STEP' in symbol:
            return 'step_index'
        elif symbol in ['RDBEAR', 'RDBULL']:
            return 'market_index'
        else:
            return 'unknown'
    
    def get_recommended_timeframes(self, symbol, volatility):
        """Retorna timeframes recomendados para o símbolo"""
        symbol_type = self.get_symbol_type(symbol)
        
        if symbol_type == 'high_frequency_volatility':
            return ['3t', '5t', '7t', '1m']
        elif symbol_type == 'crash_boom':
            return ['2m', '3m', '5m', '10m']
        elif volatility <= 0.25:
            return ['5t', '7t', '1m', '2m']
        elif volatility <= 0.75:
            return ['1m', '2m', '3m', '5m']
        else:
            return ['2m', '3m', '5m', '10m']
    
    def get_fallback_data(self, symbol):
        """Dados de fallback"""
        config = self.deriv_manager.get_symbol_config(symbol)
        return {
            'symbol': symbol,
            'price': config['base_price'] + np.random.normal(0, 10),
            'rsi': 50.0,
            'macd': 0.0,
            'bb_upper': config['base_price'] * 1.02,
            'bb_lower': config['base_price'] * 0.98,
            'volatility': config['volatility'],
            'trend_strength': 0.0,
            'trend': 'neutral',
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat(),
            'symbol_type': self.get_symbol_type(symbol)
        }
    
    def predict_with_user_config(self, symbol, duration_type, duration_value, stake=None, ai_mode=False):
        """Predição respeitando configurações do usuário"""
        try:
            # Verificar se pode fazer novos trades
            if not self.trading_manager.can_trade():
                return {
                    'error': 'Trading bloqueado - Stop diário atingido',
                    'should_trade': False,
                    'trading_stats': self.trading_manager.get_trading_stats()
                }

            # Validar timeframe
            if not self.deriv_manager.validate_timeframe(duration_type, duration_value):
                return {
                    'error': f'Timeframe inválido: {duration_value}{duration_type}',
                    'should_trade': False
                }
            
            # Obter dados de mercado
            market_data = self.get_market_data_for_symbol(symbol)
            
            # Calcular confiança mínima necessária para o timeframe
            min_confidence = self.deriv_manager.get_timeframe_confidence_threshold(duration_type, duration_value)
            
            # Fazer análise técnica
            analysis = self.analyze_market_conditions(market_data)
            
            # Calcular confiança final
            base_confidence = analysis['confidence']
            
            # Ajustar confiança baseada no símbolo e timeframe
            symbol_multiplier = self.get_symbol_confidence_multiplier(symbol)
            timeframe_multiplier = self.get_timeframe_confidence_multiplier(duration_type, duration_value)
            
            final_confidence = base_confidence * symbol_multiplier * timeframe_multiplier
            final_confidence = max(50, min(95, final_confidence))
            
            # Análise de sinais de mercado para melhor momento de entrada
            market_signals = [
                {'type': 'rsi', 'strength': 1.0 if 30 <= market_data['rsi'] <= 70 else 0.5},
                {'type': 'trend', 'strength': 0.8 if market_data['trend'] != 'neutral' else 0.3},
                {'type': 'volatility', 'strength': 0.7 if market_data['volatility'] <= 1.5 else 0.4}
            ]
            
            is_good_entry, entry_quality = self.trading_manager.analyze_market_conditions(market_signals)
            
            # Decidir se deve operar
            should_trade = final_confidence >= min_confidence and is_good_entry
            
            # Calcular stake ideal considerando gerenciamento de risco
            if stake is None:
                stake = self.trading_manager.calculate_position_size(
                    final_confidence, 
                    market_data['volatility']
                )
            
            # Resultado
            result = {
                'should_trade': should_trade,
                'direction': analysis['direction'],
                'confidence': final_confidence,
                'base_confidence': base_confidence,
                'min_confidence_required': min_confidence,
                'symbol': symbol,
                'timeframe': f"{duration_value}{duration_type}",
                'stake': stake,
                'market_analysis': analysis,
                'symbol_type': market_data['symbol_type'],
                'recommendation': self.get_trading_recommendation(final_confidence, min_confidence),
                'ai_mode_active': ai_mode,
                'timestamp': datetime.now().isoformat(),
                'trading_stats': self.trading_manager.get_trading_stats(),
                'entry_quality': {
                    'is_good_entry': is_good_entry,
                    'entry_score': entry_quality,
                    'best_entry_time': self.get_best_entry_time(market_data)
                },
                'risk_management': {
                    'martingale_level': self.trading_manager.current_martingale,
                    'current_balance': self.trading_manager.current_balance,
                    'daily_profit': self.trading_manager.daily_profit,
                    'stop_gain_target': self.trading_manager.initial_balance * self.trading_manager.stop_gain,
                    'stop_loss_limit': self.trading_manager.initial_balance * self.trading_manager.stop_loss
                }
            }
            
            # Log detalhado
            logger.info(f"ANÁLISE: {symbol} {duration_value}{duration_type} | "
                       f"Direção: {analysis['direction']} | "
                       f"Confiança: {final_confidence:.1f}% (min: {min_confidence}%) | "
                       f"Operar: {'SIM' if should_trade else 'NÃO'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'error': str(e),
                'should_trade': False,
                'symbol': symbol,
                'timeframe': f"{duration_value}{duration_type}"
            }
    
    def analyze_market_conditions(self, market_data):
        """Análise técnica das condições de mercado"""
        try:
            rsi = market_data['rsi']
            macd = market_data['macd']
            price = market_data['price']
            bb_upper = market_data['bb_upper']
            bb_lower = market_data['bb_lower']
            trend = market_data['trend']
            volatility = market_data['volatility']
            
            # Sinais de trading
            signals = []
            
            # RSI signals
            if rsi < 30:
                signals.append(('CALL', 0.9))
            elif rsi < 40:
                signals.append(('CALL', 0.7))
            elif rsi > 70:
                signals.append(('PUT', 0.9))
            elif rsi > 60:
                signals.append(('PUT', 0.7))
            
            # MACD signals
            if macd > 0:
                signals.append(('CALL', 0.6))
            else:
                signals.append(('PUT', 0.6))
            
            # Bollinger Bands
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            if bb_position < 0.2:
                signals.append(('CALL', 0.8))
            elif bb_position > 0.8:
                signals.append(('PUT', 0.8))
            
            # Trend signals
            if trend == 'bullish':
                signals.append(('CALL', 0.6))
            elif trend == 'bearish':
                signals.append(('PUT', 0.6))
            
            # Volatility adjustment
            vol_factor = 1.0
            if volatility > 2.0:
                vol_factor = 0.8  # Reduz confiança em alta volatilidade
            elif volatility < 0.5:
                vol_factor = 1.1  # Aumenta confiança em baixa volatilidade
            
            # Combinar sinais
            call_weight = sum(weight for direction, weight in signals if direction == 'CALL')
            put_weight = sum(weight for direction, weight in signals if direction == 'PUT')
            
            if call_weight > put_weight:
                direction = 'CALL'
                confidence = (call_weight / (call_weight + put_weight + 0.1)) * 100 * vol_factor
            else:
                direction = 'PUT'
                confidence = (put_weight / (call_weight + put_weight + 0.1)) * 100 * vol_factor
            
            return {
                'direction': direction,
                'confidence': max(60, min(90, confidence)),
                'rsi_signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
                'trend_signal': trend,
                'bb_position': bb_position,
                'volatility_factor': vol_factor,
                'signal_strength': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return {
                'direction': 'CALL' if np.random.random() > 0.5 else 'PUT',
                'confidence': 65.0,
                'error': str(e)
            }
    
    def get_symbol_confidence_multiplier(self, symbol):
        """Multiplicador de confiança baseado no símbolo"""
        symbol_type = self.get_symbol_type(symbol)
        config = self.deriv_manager.get_symbol_config(symbol)
        volatility = config.get('volatility', 1.0)
        
        if symbol_type == 'high_frequency_volatility':
            return 0.9  # Ligeiramente mais difícil
        elif symbol_type == 'crash_boom':
            return 0.8  # Mais difícil devido aos eventos especiais
        elif volatility <= 0.25:
            return 1.1  # Mais fácil em baixa volatilidade
        elif volatility >= 1.5:
            return 0.85  # Mais difícil em alta volatilidade
        else:
            return 1.0
    
    def get_timeframe_confidence_multiplier(self, duration_type, duration_value):
        """Multiplicador de confiança baseado no timeframe"""
        if duration_type == 't':
            # Ticks são mais difíceis
            if duration_value <= 3:
                return 0.8
            elif duration_value <= 5:
                return 0.9
            else:
                return 0.95
        else:
            # Minutos são mais previsíveis
            if duration_value == 1:
                return 1.0
            elif duration_value <= 3:
                return 1.05
            else:
                return 1.1
    
    def calculate_optimal_stake(self, confidence, symbol, market_data):
        """Calcula stake ótimo baseado na confiança e condições"""
        base_stake = 1.0
        
        # Ajustar por confiança
        if confidence >= 85:
            stake_multiplier = 2.0
        elif confidence >= 75:
            stake_multiplier = 1.5
        elif confidence >= 65:
            stake_multiplier = 1.0
        else:
            stake_multiplier = 0.5
        
        # Ajustar por volatilidade
        volatility = market_data.get('volatility', 1.0)
        if volatility > 2.0:
            stake_multiplier *= 0.7
        elif volatility < 0.5:
            stake_multiplier *= 1.2
        
        stake = base_stake * stake_multiplier
        return round(max(0.35, min(50.0, stake)), 2)
    
    def get_trading_recommendation(self, confidence, min_confidence):
        """Gera recomendação de trading"""
        if confidence >= min_confidence + 10:
            return "Sinal forte - Recomendado operar"
        elif confidence >= min_confidence:
            return "Sinal moderado - Pode operar"
        elif confidence >= min_confidence - 5:
            return "Sinal fraco - Aguardar melhor momento"
        else:
            return "Sinal insuficiente - Não operar"
            
    def get_best_entry_time(self, market_data):
        """Determina o melhor momento para entrada na vela"""
        try:
            # Análise de volatilidade e tendência
            volatility = market_data['volatility']
            trend = market_data['trend']
            trend_strength = market_data.get('trend_strength', 0.5)
            
            # Momento ideal baseado no tipo de ativo e condições
            if market_data['symbol_type'] == 'high_frequency_volatility':
                # Para ativos de alta frequência, entrar mais cedo
                entry_percentage = 0.1  # 10% do início da vela
            elif market_data['symbol_type'] == 'crash_boom':
                # Para crash/boom, aguardar confirmação inicial
                entry_percentage = 0.2  # 20% do início da vela
            else:
                # Para outros ativos, base no trend_strength
                if trend_strength > 0.7:
                    entry_percentage = 0.15  # Tendência forte = entrada mais rápida
                else:
                    entry_percentage = 0.25  # Aguardar mais confirmação
            
            # Ajuste por volatilidade
            if volatility > 1.5:
                entry_percentage += 0.05  # Aguardar mais em alta volatilidade
            
            # Converter para milissegundos (considerando vela de 1 minuto como base)
            entry_time_ms = int(60000 * entry_percentage)  # 60000ms = 1 minuto
            
            return {
                'entry_percentage': entry_percentage,
                'entry_time_ms': entry_time_ms,
                'reason': f"Entrada otimizada para {market_data['symbol_type']} com volatilidade {volatility:.2f}",
                'recommendation': "Entrada no início da vela" if entry_percentage <= 0.15 else "Aguardar confirmação inicial"
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular tempo de entrada: {e}")
            return {
                'entry_percentage': 0.2,
                'entry_time_ms': 12000,
                'reason': "Usando configuração padrão devido a erro",
                'recommendation': "Entrada após 12 segundos do início da vela"
            }
    
    def full_ai_analysis(self, preferred_symbols=None):
        """Análise completa automatizada da IA"""
        try:
            symbols_to_analyze = preferred_symbols or self.full_ai_mode['preferred_symbols']
            
            best_opportunity = None
            best_confidence = 0
            
            analyses = []
            
            # Analisar cada símbolo
            for symbol in symbols_to_analyze:
                # Obter dados de mercado
                market_data = self.get_market_data_for_symbol(symbol)
                
                # Selecionar timeframe ótimo
                optimal_tf = self.deriv_manager.get_optimal_timeframe(
                    symbol, 75, market_data
                )
                
                # Fazer análise
                analysis = self.predict_with_user_config(
                    symbol, 
                    optimal_tf['type'], 
                    optimal_tf['duration'],
                    ai_mode=True
                )
                
                analysis['symbol'] = symbol
                analysis['market_data'] = market_data
                analyses.append(analysis)
                
                # Verificar se é a melhor oportunidade
                if (analysis.get('should_trade', False) and 
                    analysis.get('confidence', 0) > best_confidence):
                    best_opportunity = analysis
                    best_confidence = analysis['confidence']
            
            # Resultado da análise completa
            result = {
                'full_ai_active': True,
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols_to_analyze),
                'best_opportunity': best_opportunity,
                'all_analyses': analyses,
                'recommendation': self.get_full_ai_recommendation(best_opportunity, analyses),
                'market_overview': self.generate_market_overview(analyses)
            }
            
            logger.info(f"IA COMPLETA: Analisou {len(symbols_to_analyze)} símbolos. "
                       f"Melhor: {best_opportunity['symbol'] if best_opportunity else 'Nenhum'} "
                       f"({best_confidence:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise completa: {e}")
            return {
                'error': str(e),
                'full_ai_active': False
            }
    
    def get_full_ai_recommendation(self, best_opportunity, all_analyses):
        """Gera recomendação da IA completa"""
        if not best_opportunity:
            return {
                'action': 'wait',
                'reason': 'Nenhuma oportunidade com confiança suficiente encontrada',
                'next_analysis_in': 30
            }
        
        confidence = best_opportunity.get('confidence', 0)
        
        if confidence >= 85:
            return {
                'action': 'trade_strong',
                'reason': f"Oportunidade excelente detectada em {best_opportunity['symbol']}",
                'confidence': confidence,
                'trade_config': best_opportunity
            }
        elif confidence >= 75:
            return {
                'action': 'trade_moderate',
                'reason': f"Boa oportunidade em {best_opportunity['symbol']}",
                'confidence': confidence,
                'trade_config': best_opportunity
            }
        else:
            return {
                'action': 'wait',
                'reason': 'Aguardar condições mais favoráveis',
                'best_confidence': confidence
            }
    
    def generate_market_overview(self, analyses):
        """Gera overview do mercado"""
        try:
            total = len(analyses)
            tradeable = sum(1 for a in analyses if a.get('should_trade', False))
            avg_confidence = np.mean([a.get('confidence', 0) for a in analyses])
            
            # Análise por direção
            calls = sum(1 for a in analyses if a.get('direction') == 'CALL')
            puts = sum(1 for a in analyses if a.get('direction') == 'PUT')
            
            # Análise por volatilidade
            high_vol = sum(1 for a in analyses 
                          if a.get('market_data', {}).get('volatility', 0) > 1.0)
            
            return {
                'total_symbols': total,
                'tradeable_opportunities': tradeable,
                'average_confidence': round(avg_confidence, 1),
                'market_bias': 'BULLISH' if calls > puts else 'BEARISH' if puts > calls else 'NEUTRAL',
                'high_volatility_symbols': high_vol,
                'market_condition': 'FAVORABLE' if tradeable > total * 0.3 else 'UNFAVORABLE'
            }
            
        except Exception as e:
            logger.error(f"Erro no overview: {e}")
            return {'error': str(e)}


# Instância global
trading_ai = EnhancedTradingAI()

# ====================================
# ROTAS DA API CORRIGIDAS
# ====================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Enhanced Deriv Trading AI',
        'version': '4.0.0',
        'timestamp': datetime.now().isoformat(),
        'supported_symbols': list(trading_ai.deriv_manager.symbol_configs.keys()),
        'supported_timeframes': {
            'ticks': list(trading_ai.deriv_manager.timeframe_mapping['t'].keys()),
            'minutes': list(trading_ai.deriv_manager.timeframe_mapping['m'].keys())
        },
        'full_ai_mode': trading_ai.full_ai_mode
    })

@app.route('/analyze', methods=['POST'])
def analyze_symbol():
    """Análise específica para símbolo e timeframe do usuário"""
    try:
        data = request.get_json()
        
        # Parâmetros obrigatórios
        symbol = data.get('symbol', 'R_50')
        duration_type = data.get('duration_type', 't')
        duration_value = int(data.get('duration', 5))
        
        # Parâmetros opcionais
        stake = data.get('stake')
        ai_mode = data.get('ai_mode', False)
        
        # Validar entrada
        if not symbol or not duration_type or not duration_value:
            return jsonify({
                'error': 'Parâmetros obrigatórios: symbol, duration_type, duration',
                'example': {
                    'symbol': 'R_50',
                    'duration_type': 't',
                    'duration': 5,
                    'stake': 1.0
                }
            }), 400
        
        # Fazer análise
        result = trading_ai.predict_with_user_config(
            symbol, duration_type, duration_value, stake, ai_mode
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/full-ai-analysis', methods=['POST'])
def full_ai_analysis():
    """Análise completa automatizada da IA"""
    try:
        data = request.get_json() or {}
        preferred_symbols = data.get('preferred_symbols')
        
        result = trading_ai.full_ai_analysis(preferred_symbols)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na análise completa: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/symbol-info/<symbol>', methods=['GET'])
def get_symbol_info(symbol):
    """Informações detalhadas sobre um símbolo"""
    try:
        config = trading_ai.deriv_manager.get_symbol_config(symbol)
        market_data = trading_ai.get_market_data_for_symbol(symbol)
        
        return jsonify({
            'symbol': symbol,
            'config': config,
            'current_market_data': market_data,
            'symbol_type': trading_ai.get_symbol_type(symbol),
            'recommended_timeframes': trading_ai.get_recommended_timeframes(symbol, config.get('volatility', 1.0)),
            'optimal_timeframe': trading_ai.deriv_manager.get_optimal_timeframe(symbol, 75, market_data)
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter info do símbolo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/market-overview', methods=['GET'])
def market_overview():
    """Overview geral do mercado para todos os símbolos"""
    try:
        symbols = list(trading_ai.deriv_manager.symbol_configs.keys())[:10]  # Limitar a 10
        analyses = []
        
        for symbol in symbols:
            try:
                market_data = trading_ai.get_market_data_for_symbol(symbol)
                optimal_tf = trading_ai.deriv_manager.get_optimal_timeframe(symbol, 70, market_data)
                
                quick_analysis = {
                    'symbol': symbol,
                    'price': market_data['price'],
                    'trend': market_data['trend'],
                    'volatility': market_data['volatility'],
                    'rsi': market_data['rsi'],
                    'optimal_timeframe': f"{optimal_tf['duration']}{optimal_tf['type']}"
                }
                analyses.append(quick_analysis)
                
            except Exception as e:
                logger.error(f"Erro ao analisar {symbol}: {e}")
                continue
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(analyses),
            'market_data': analyses,
            'overall_sentiment': trading_ai.generate_market_overview([{'market_data': a, 'direction': 'CALL'} for a in analyses])
        })
        
    except Exception as e:
        logger.error(f"Erro no overview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/report-trade', methods=['POST'])
def report_trade():
    """Reporta resultado de trade"""
    try:
        data = request.get_json()
        
        # Validação
        required = ['symbol', 'direction', 'result', 'entry_price', 'stake']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Campo obrigatório: {field}'}), 400
        
        # Processar resultado no gerenciador de trading
        pnl = data.get('pnl', 0)
        trading_stats = trading_ai.trading_manager.process_trade_result(
            data['result'],
            data['stake'],
            pnl
        )
        
        # Salvar no banco
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, direction, stake, duration_type, duration_value,
             entry_price, exit_price, result, pnl, confidence, timeframe_used, ai_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            data['symbol'],
            data['direction'],
            data['stake'],
            data.get('duration_type', 't'),
            data.get('duration_value', 5),
            data['entry_price'],
            data.get('exit_price', data['entry_price']),
            data['result'],
            pnl,
            data.get('confidence', 0),
            data.get('timeframe_used', '5t'),
            data.get('ai_mode', False)
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Trade reportado com sucesso',
            'trade_id': cursor.lastrowid,
            'trading_stats': trading_stats,
            'next_trade': {
                'can_trade': trading_stats['can_trade'],
                'martingale_level': trading_stats['martingale_level'],
                'recommended_stake': trading_ai.trading_manager.calculate_position_size(
                    data.get('confidence', 70),
                    data.get('volatility', 1.0)
                ) if trading_stats['can_trade'] else 0.0
            }
        })
        
    except Exception as e:
        logger.error(f"Erro ao reportar trade: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Estatísticas do sistema"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        
        # Estatísticas básicas
        trades_df = pd.read_sql_query('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100', conn)
        
        if len(trades_df) == 0:
            return jsonify({
                'total_trades': 0,
                'message': 'Nenhum trade registrado ainda'
            })
        
        wins = len(trades_df[trades_df['result'] == 'win'])
        losses = len(trades_df[trades_df['result'] == 'loss'])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Estatísticas por símbolo
        symbol_stats = {}
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            symbol_wins = len(symbol_trades[symbol_trades['result'] == 'win'])
            symbol_stats[symbol] = {
                'total': len(symbol_trades),
                'wins': symbol_wins,
                'win_rate': symbol_wins / len(symbol_trades) if len(symbol_trades) > 0 else 0
            }
        
        # Estatísticas por timeframe
        timeframe_stats = {}
        for tf in trades_df['timeframe_used'].unique():
            if tf:
                tf_trades = trades_df[trades_df['timeframe_used'] == tf]
                tf_wins = len(tf_trades[tf_trades['result'] == 'win'])
                timeframe_stats[tf] = {
                    'total': len(tf_trades),
                    'wins': tf_wins,
                    'win_rate': tf_wins / len(tf_trades) if len(tf_trades) > 0 else 0
                }
        
        conn.close()
        
        return jsonify({
            'total_trades': len(trades_df),
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate * 100, 2),
            'symbol_performance': symbol_stats,
            'timeframe_performance': timeframe_stats,
            'recent_trades': trades_df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Erro nas estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Sistema de Trading Deriv rodando na porta {API_PORT}")
    logger.info(f"Símbolos suportados: {len(trading_ai.deriv_manager.symbol_configs)}")
    logger.info("APIs principais: /analyze, /full-ai-analysis, /symbol-info, /market-overview")
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
