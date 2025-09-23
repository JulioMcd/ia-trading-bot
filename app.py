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

class BankrollManager:
    """Gerenciador Avançado de Banca com Proteção Anti-Sequência de Perdas"""
    
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.min_balance_threshold = initial_balance * 0.2  # 20% do capital inicial
        self.max_risk_per_trade = 0.02  # 2% máximo por trade
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses_allowed = 3
        self.recovery_mode = False
        self.high_risk_mode = False
        self.last_trades_results = []
        
        # Configurações de Stake Dinâmico
        self.base_stake = initial_balance * 0.01  # 1% da banca inicial
        self.min_stake = 1.0
        self.max_stake = initial_balance * 0.05  # 5% máximo
        
        # Martingale Controlado
        self.martingale_enabled = True
        self.martingale_multiplier = 2.1
        self.max_martingale_level = 2  # Máximo 2 níveis apenas
        self.current_martingale_level = 0
        
        # Histórico para análise
        self.performance_window = []
        self.risk_assessment = {
            'market_volatility': 'normal',
            'recent_accuracy': 0.5,
            'confidence_threshold': 70.0,
            'danger_signals': []
        }
    
    def analyze_recent_performance(self, last_n_trades=10):
        """Analisa performance recente para ajustar estratégia"""
        if len(self.last_trades_results) < 3:
            return {
                'trend': 'insufficient_data',
                'win_rate': 0.5,
                'risk_level': 'medium'
            }
        
        recent = self.last_trades_results[-last_n_trades:]
        wins = sum(1 for t in recent if t['result'] == 'win')
        win_rate = wins / len(recent)
        
        # Detectar padrões perigosos
        consecutive_losses = 0
        for trade in reversed(recent):
            if trade['result'] == 'loss':
                consecutive_losses += 1
            else:
                break
        
        # Avaliar tendência
        if consecutive_losses >= 3:
            trend = 'dangerous_losing_streak'
            risk_level = 'critical'
        elif win_rate < 0.3:
            trend = 'poor_performance'
            risk_level = 'high'
        elif win_rate < 0.5:
            trend = 'below_average'
            risk_level = 'medium_high'
        elif win_rate < 0.7:
            trend = 'average'
            risk_level = 'medium'
        else:
            trend = 'good_performance'
            risk_level = 'low'
        
        return {
            'trend': trend,
            'win_rate': win_rate,
            'risk_level': risk_level,
            'consecutive_losses': consecutive_losses
        }
    
    def calculate_optimal_stake(self, confidence, market_conditions):
        """Calcula stake ótimo baseado em múltiplos fatores"""
        performance = self.analyze_recent_performance()
        
        # Base stake inicial
        stake = self.base_stake
        
        # Ajustar por confiança
        if confidence >= 85:
            stake *= 1.5  # Aumenta 50% para alta confiança
        elif confidence >= 75:
            stake *= 1.2
        elif confidence >= 65:
            stake *= 1.0
        elif confidence >= 55:
            stake *= 0.8
        else:
            stake *= 0.5  # Reduz significativamente para baixa confiança
        
        # Ajustar por performance recente
        if performance['trend'] == 'dangerous_losing_streak':
            stake *= 0.3  # Redução drástica
            self.recovery_mode = True
            logger.warning("🚨 MODO RECUPERAÇÃO ATIVADO - Stake reduzido drasticamente")
        elif performance['trend'] == 'poor_performance':
            stake *= 0.5
        elif performance['trend'] == 'below_average':
            stake *= 0.7
        elif performance['trend'] == 'good_performance':
            stake *= 1.3
        
        # Ajustar por volatilidade do mercado
        volatility = market_conditions.get('volatility', 1.0)
        if volatility > 3.0:
            stake *= 0.5  # Alta volatilidade = menor stake
        elif volatility > 2.0:
            stake *= 0.7
        elif volatility < 0.5:
            stake *= 1.2  # Baixa volatilidade = pode aumentar
        
        # Proteção de capital
        if self.current_balance < self.initial_balance * 0.5:
            stake *= 0.5  # Proteção quando perdeu 50% do capital
            logger.warning("⚠️ Proteção de capital ativada - Stake reduzido")
        
        # Limites absolutos
        stake = max(self.min_stake, min(stake, self.max_stake))
        
        # Nunca arriscar mais que o permitido
        max_allowed = self.current_balance * self.max_risk_per_trade
        stake = min(stake, max_allowed)
        
        return round(stake, 2)
    
    def should_use_martingale(self, last_trade_result):
        """Decide se deve usar martingale baseado em condições"""
        if not self.martingale_enabled:
            return False, 0
        
        # Não usar martingale se:
        # 1. Em modo recuperação
        if self.recovery_mode:
            return False, 0
        
        # 2. Muitas perdas consecutivas
        if self.consecutive_losses >= self.max_consecutive_losses_allowed:
            logger.warning("❌ Martingale desativado - muitas perdas consecutivas")
            return False, 0
        
        # 3. Banca muito baixa
        if self.current_balance < self.initial_balance * 0.3:
            logger.warning("❌ Martingale desativado - banca muito baixa")
            return False, 0
        
        # 4. Já está no nível máximo
        if self.current_martingale_level >= self.max_martingale_level:
            logger.warning("❌ Martingale no nível máximo")
            return False, 0
        
        # Se última foi perda, incrementar nível
        if last_trade_result == 'loss':
            self.current_martingale_level += 1
            return True, self.current_martingale_level
        else:
            self.current_martingale_level = 0
            return False, 0
    
    def update_balance(self, trade_result, pnl):
        """Atualiza saldo e métricas após trade"""
        self.current_balance += pnl
        
        # Atualizar consecutivos
        if trade_result == 'win':
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if self.recovery_mode and self.consecutive_wins >= 3:
                self.recovery_mode = False
                logger.info("✅ Modo recuperação desativado - 3 vitórias consecutivas")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Adicionar ao histórico
        self.last_trades_results.append({
            'result': trade_result,
            'pnl': pnl,
            'balance': self.current_balance,
            'timestamp': datetime.now()
        })
        
        # Manter apenas últimos 50 trades
        if len(self.last_trades_results) > 50:
            self.last_trades_results.pop(0)
        
        # Log do status
        emoji = "✅" if trade_result == 'win' else "❌"
        logger.info(f"{emoji} Trade: {trade_result.upper()} | PnL: ${pnl:.2f} | "
                   f"Saldo: ${self.current_balance:.2f} | "
                   f"Consecutivos: W{self.consecutive_wins}/L{self.consecutive_losses}")


class TimeFrameSelector:
    """Seletor Inteligente de TimeFrame baseado em condições de mercado"""
    
    def __init__(self):
        self.timeframes = {
            '15s': {'duration': 15, 'volatility_range': (0, 0.5), 'confidence_min': 80},
            '30s': {'duration': 30, 'volatility_range': (0.3, 1.0), 'confidence_min': 75},
            '1m': {'duration': 60, 'volatility_range': (0.5, 1.5), 'confidence_min': 70},
            '2m': {'duration': 120, 'volatility_range': (1.0, 2.0), 'confidence_min': 65},
            '3m': {'duration': 180, 'volatility_range': (1.5, 2.5), 'confidence_min': 60},
            '5m': {'duration': 300, 'volatility_range': (2.0, 5.0), 'confidence_min': 55}
        }
        
        self.last_timeframe_results = []
    
    def select_optimal_timeframe(self, market_conditions, confidence, trend_strength):
        """Seleciona timeframe ótimo baseado em múltiplos fatores"""
        volatility = market_conditions.get('volatility', 1.0)
        rsi = market_conditions.get('rsi', 50)
        trend = market_conditions.get('trend', 'neutral')
        
        # Análise de volatilidade
        if volatility < 0.5:
            # Baixa volatilidade - timeframes curtos funcionam melhor
            if confidence >= 80:
                selected = '15s'
            elif confidence >= 75:
                selected = '30s'
            else:
                selected = '1m'
        
        elif volatility < 1.5:
            # Volatilidade média
            if confidence >= 75:
                selected = '30s'
            elif confidence >= 70:
                selected = '1m'
            else:
                selected = '2m'
        
        elif volatility < 2.5:
            # Volatilidade alta
            if confidence >= 70:
                selected = '1m'
            elif confidence >= 65:
                selected = '2m'
            else:
                selected = '3m'
        
        else:
            # Volatilidade muito alta - timeframes longos para segurança
            if confidence >= 65:
                selected = '2m'
            elif confidence >= 60:
                selected = '3m'
            else:
                selected = '5m'
        
        # Ajustes baseados em RSI
        if rsi < 20 or rsi > 80:
            # Condições extremas - aumentar timeframe
            current_duration = self.timeframes[selected]['duration']
            for tf, info in self.timeframes.items():
                if info['duration'] > current_duration * 1.5:
                    selected = tf
                    break
        
        # Ajustes baseados em tendência
        if trend in ['strong_bullish', 'strong_bearish']:
            # Tendência forte - pode usar timeframes menores
            if selected in ['3m', '5m'] and confidence >= 70:
                selected = '2m'
        elif trend == 'choppy':
            # Mercado instável - aumentar timeframe
            if selected in ['15s', '30s']:
                selected = '1m'
        
        # Verificar performance do timeframe anterior
        if self.last_timeframe_results:
            recent_tf_performance = self.analyze_timeframe_performance()
            if recent_tf_performance['success_rate'] < 0.4:
                # Performance ruim - mudar estratégia
                logger.info(f"📊 Mudando timeframe devido a baixa performance")
                if selected in ['15s', '30s']:
                    selected = '2m'
                elif selected == '1m':
                    selected = '3m'
        
        logger.info(f"⏱️ TimeFrame selecionado: {selected} | "
                   f"Volatilidade: {volatility:.2f} | "
                   f"Confiança: {confidence:.1f}%")
        
        return selected
    
    def analyze_timeframe_performance(self, last_n=10):
        """Analisa performance dos últimos timeframes usados"""
        if len(self.last_timeframe_results) < 3:
            return {'success_rate': 0.5, 'best_timeframe': None}
        
        recent = self.last_timeframe_results[-last_n:]
        
        # Agrupar por timeframe
        tf_performance = {}
        for result in recent:
            tf = result['timeframe']
            if tf not in tf_performance:
                tf_performance[tf] = {'wins': 0, 'total': 0}
            
            tf_performance[tf]['total'] += 1
            if result['result'] == 'win':
                tf_performance[tf]['wins'] += 1
        
        # Calcular taxas de sucesso
        best_tf = None
        best_rate = 0
        total_wins = 0
        total_trades = 0
        
        for tf, stats in tf_performance.items():
            rate = stats['wins'] / stats['total']
            if rate > best_rate:
                best_rate = rate
                best_tf = tf
            total_wins += stats['wins']
            total_trades += stats['total']
        
        overall_rate = total_wins / total_trades if total_trades > 0 else 0.5
        
        return {
            'success_rate': overall_rate,
            'best_timeframe': best_tf,
            'performance_by_tf': tf_performance
        }
    
    def record_result(self, timeframe, result):
        """Registra resultado do timeframe usado"""
        self.last_timeframe_results.append({
            'timeframe': timeframe,
            'result': result,
            'timestamp': datetime.now()
        })
        
        # Manter apenas últimos 30 resultados
        if len(self.last_timeframe_results) > 30:
            self.last_timeframe_results.pop(0)


class TechnicalIndicators:
    """Classe para calcular indicadores técnicos sem dependências externas"""
    
    @staticmethod
    def rsi(prices, window=14):
        """Calcula RSI"""
        try:
            prices_series = pd.Series(prices)
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.fillna(50).iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        try:
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return float(macd_line.iloc[-1]) if len(macd_line) > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def bollinger_bands(prices, window=20, num_std=2):
        """Calcula Bollinger Bands"""
        try:
            prices_series = pd.Series(prices)
            rolling_mean = prices_series.rolling(window=window).mean()
            rolling_std = prices_series.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            return {
                'upper': float(upper_band.iloc[-1]) if len(upper_band) > 0 else prices[-1] * 1.02,
                'lower': float(lower_band.iloc[-1]) if len(lower_band) > 0 else prices[-1] * 0.98
            }
        except:
            return {
                'upper': prices[-1] * 1.02,
                'lower': prices[-1] * 0.98
            }
    
    @staticmethod
    def volatility(prices, window=14):
        """Calcula volatilidade (ATR simplificado)"""
        try:
            prices_series = pd.Series(prices)
            returns = prices_series.pct_change().dropna()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Anualizada
            return float(volatility.iloc[-1]) if len(volatility) > 0 else 1.0
        except:
            return 1.0
    
    @staticmethod
    def trend_strength(prices, window=20):
        """Calcula força da tendência"""
        try:
            prices_series = pd.Series(prices)
            sma = prices_series.rolling(window=window).mean()
            
            if len(sma) < window:
                return 0.0
            
            # Calcular ângulo da tendência
            recent_sma = sma.iloc[-5:].values
            if len(recent_sma) < 2:
                return 0.0
            
            trend = (recent_sma[-1] - recent_sma[0]) / recent_sma[0]
            return float(np.clip(trend * 100, -100, 100))
        except:
            return 0.0


class EnhancedTradingAI:
    def __init__(self):
        # Sistema de ML
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
        
        # Gerenciadores especializados
        self.bankroll_manager = BankrollManager(initial_balance=1000.0)
        self.timeframe_selector = TimeFrameSelector()
        self.indicators = TechnicalIndicators()
        
        # Métricas e controle
        self.online_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': [],
            'last_10_trades': [],
            'learning_updates': 0
        }
        
        # Sistema de detecção de padrões perigosos
        self.danger_patterns = {
            'consecutive_losses': 0,
            'last_5_accuracy': 0.5,
            'volatility_spike': False,
            'trend_reversal': False,
            'confidence_drop': False
        }
        
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados com colunas adicionais"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            # Tabela de trades aprimorada
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    direction TEXT,
                    stake REAL,
                    duration TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    result TEXT,
                    pnl REAL,
                    martingale_level INTEGER,
                    market_conditions TEXT,
                    features TEXT,
                    online_updated BOOLEAN DEFAULT 0,
                    prediction_confidence REAL,
                    model_used TEXT,
                    learning_iteration INTEGER,
                    data_type TEXT DEFAULT 'real',
                    market_scenario TEXT,
                    bankroll_before REAL,
                    bankroll_after REAL,
                    risk_level TEXT,
                    timeframe_used TEXT
                )
            ''')
            
            # Tabela de métricas online
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS online_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    model_type TEXT,
                    accuracy REAL,
                    total_samples INTEGER,
                    recent_performance TEXT,
                    adaptation_rate REAL
                )
            ''')
            
            # Tabela de gerenciamento de risco
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    bankroll REAL,
                    consecutive_losses INTEGER,
                    consecutive_wins INTEGER,
                    risk_level TEXT,
                    recovery_mode BOOLEAN,
                    max_drawdown REAL,
                    current_drawdown REAL
                )
            ''')
            
            # Tabela de estatísticas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    best_streak INTEGER,
                    worst_streak INTEGER,
                    martingale_usage TEXT,
                    online_accuracy REAL,
                    adaptation_score REAL,
                    avg_stake REAL,
                    max_drawdown REAL
                )
            ''')
            
            # Tabela de dados de mercado
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    price REAL,
                    volume REAL,
                    rsi REAL,
                    macd REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    volatility REAL,
                    trend_strength REAL,
                    data_type TEXT DEFAULT 'real',
                    market_scenario TEXT,
                    synthetic_params TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Banco de dados inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco: {e}")
    
    def detect_danger_patterns(self, market_data, recent_trades):
        """Detecta padrões perigosos que podem levar a perdas sequenciais"""
        dangers = []
        risk_score = 0
        
        # 1. Verificar perdas consecutivas
        if recent_trades:
            losses = 0
            for trade in reversed(recent_trades[-5:]):
                if trade.get('result') == 'loss':
                    losses += 1
                else:
                    break
            
            if losses >= 3:
                dangers.append('consecutive_losses_detected')
                risk_score += 50
            elif losses >= 2:
                risk_score += 30
        
        # 2. Verificar volatilidade extrema
        volatility = market_data.get('volatility', 1.0)
        if volatility > 3.0:
            dangers.append('extreme_volatility')
            risk_score += 40
        elif volatility > 2.0:
            risk_score += 20
        
        # 3. Verificar RSI extremo
        rsi = market_data.get('rsi', 50)
        if rsi < 20 or rsi > 80:
            dangers.append('extreme_rsi')
            risk_score += 30
        elif rsi < 30 or rsi > 70:
            risk_score += 15
        
        # 4. Verificar mudança brusca de tendência
        if 'trend_strength' in market_data:
            trend = market_data['trend_strength']
            if abs(trend) > 50:
                dangers.append('strong_trend_reversal')
                risk_score += 25
        
        # 5. Verificar accuracy recente
        if self.online_metrics['last_10_trades']:
            recent_correct = sum(1 for t in self.online_metrics['last_10_trades'][-5:] if t['correct'])
            recent_accuracy = recent_correct / min(5, len(self.online_metrics['last_10_trades']))
            
            if recent_accuracy < 0.3:
                dangers.append('poor_recent_accuracy')
                risk_score += 40
            elif recent_accuracy < 0.5:
                risk_score += 20
        
        # Classificar nível de risco
        if risk_score >= 80:
            risk_level = 'CRITICAL'
        elif risk_score >= 60:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MEDIUM_HIGH'
        elif risk_score >= 20:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'dangers': dangers,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'should_trade': risk_score < 80,  # Não operar em risco crítico
            'confidence_penalty': min(30, risk_score / 2)  # Penalidade na confiança
        }
    
    def get_market_data(self, symbol, force_scenario=None):
        """Obtém dados de mercado com análise aprimorada"""
        try:
            # Para índices sintéticos Deriv
            if symbol.startswith(('R_', '1HZ', 'CRASH', 'BOOM', 'JD', 'STEP')):
                scenario = force_scenario or self.get_market_scenario(symbol)
                return self.get_synthetic_data(symbol, scenario)
            
            # Para outros símbolos
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1m")
                
                if not data.empty:
                    return self.process_market_data(data, symbol, data_type='real')
            except Exception as e:
                logger.warning(f"yfinance falhou para {symbol}: {e}")
                pass
            
            # Fallback para sintético
            scenario = force_scenario or 'normal'
            return self.get_synthetic_data(symbol, scenario)
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado: {e}")
            return self.get_fallback_data(symbol, data_type='synthetic', scenario='error_fallback')
    
    def get_synthetic_data(self, symbol, scenario='normal'):
        """Gera dados sintéticos mais realistas"""
        try:
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            
            volatility_map = {
                'R_10': 0.1, 'R_25': 0.25, 'R_50': 0.5, 'R_75': 0.75, 'R_100': 1.0,
                '1HZ10V': 0.1, '1HZ25V': 0.25, '1HZ50V': 0.5, '1HZ75V': 0.75, '1HZ100V': 1.0,
                'CRASH300': 3.0, 'CRASH500': 5.0, 'CRASH1000': 10.0,
                'BOOM300': 3.0, 'BOOM500': 5.0, 'BOOM1000': 10.0
            }
            
            volatility = volatility_map.get(symbol, 0.5)
            base_price = 1000 + np.random.normal(0, 50)
            
            periods = 100
            prices = [base_price]
            
            for i in range(periods - 1):
                drift = np.random.normal(0, volatility * 0.01)
                mean_reversion = (1000 - prices[-1]) * 0.001
                noise = np.random.normal(0, volatility * 0.1)
                
                new_price = prices[-1] * (1 + drift + mean_reversion + noise)
                prices.append(max(0.01, new_price))
            
            df = pd.DataFrame({
                'Close': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'Volume': [np.random.randint(1000, 10000) for _ in prices]
            })
            
            return self.process_market_data(df, symbol, data_type='synthetic', scenario=scenario)
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados sintéticos: {e}")
            return self.get_fallback_data(symbol)
    
    def process_market_data(self, data, symbol, data_type='real', scenario=None, synthetic_params=None):
        """Processa dados com análise aprimorada"""
        try:
            close_prices = data['Close'].values
            
            # Calcular todos os indicadores
            rsi = self.indicators.rsi(close_prices)
            macd = self.indicators.macd(close_prices)
            bb = self.indicators.bollinger_bands(close_prices)
            volatility = self.indicators.volatility(close_prices)
            trend_strength = self.indicators.trend_strength(close_prices)
            
            # Determinar tendência
            if trend_strength > 20:
                trend = 'strong_bullish'
            elif trend_strength > 10:
                trend = 'bullish'
            elif trend_strength < -20:
                trend = 'strong_bearish'
            elif trend_strength < -10:
                trend = 'bearish'
            elif volatility > 2.0:
                trend = 'choppy'
            else:
                trend = 'neutral'
            
            current_data = {
                'symbol': symbol,
                'price': float(close_prices[-1]),
                'rsi': rsi,
                'macd': macd,
                'bb_upper': bb['upper'],
                'bb_lower': bb['lower'],
                'volatility': volatility,
                'trend_strength': trend_strength,
                'trend': trend,
                'volume': float(data['Volume'].values[-1]) if 'Volume' in data.columns else 1000.0,
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'market_scenario': scenario,
                'synthetic_params': json.dumps(synthetic_params) if synthetic_params else None
            }
            
            self.save_market_data(current_data)
            
            logger.info(f"📊 Dados {data_type.upper()}: {symbol} | "
                       f"Preço: ${current_data['price']:.2f} | "
                       f"RSI: {rsi:.1f} | Vol: {volatility:.2f} | "
                       f"Tendência: {trend}")
            
            return current_data
            
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            return self.get_fallback_data(symbol, data_type=data_type, scenario=scenario)
    
    def get_fallback_data(self, symbol, data_type='synthetic', scenario='fallback'):
        """Dados de fallback em caso de erro"""
        return {
            'symbol': symbol,
            'price': 1000.0 + np.random.normal(0, 10),
            'rsi': 50.0,
            'macd': 0.0,
            'bb_upper': 1020.0,
            'bb_lower': 980.0,
            'volatility': 1.0,
            'trend_strength': 0.0,
            'trend': 'neutral',
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'market_scenario': scenario
        }
    
    def save_market_data(self, data):
        """Salva dados de mercado no banco"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data 
                (timestamp, symbol, price, volume, rsi, macd, bb_upper, bb_lower, 
                 volatility, trend_strength, data_type, market_scenario, synthetic_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'], data['symbol'], data['price'], data['volume'],
                data['rsi'], data['macd'], data['bb_upper'], data['bb_lower'],
                data['volatility'], data.get('trend_strength', 0),
                data.get('data_type', 'real'), data.get('market_scenario'),
                data.get('synthetic_params')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
    
    def extract_features(self, market_data, trade_history=None):
        """Extrai features para ML"""
        try:
            features = []
            
            # Features de mercado
            features.extend([
                market_data['rsi'],
                market_data['macd'],
                market_data['volatility'],
                (market_data['price'] - market_data['bb_lower']) / 
                (market_data['bb_upper'] - market_data['bb_lower']) if market_data['bb_upper'] != market_data['bb_lower'] else 0.5,
                market_data.get('trend_strength', 0) / 100.0  # Normalizado
            ])
            
            # Features temporais
            now = datetime.now()
            features.extend([
                now.hour / 24.0,
                now.weekday() / 6.0,
                (now.minute % 60) / 60.0
            ])
            
            # Features de histórico
            if trade_history and len(trade_history) > 0:
                recent_trades = trade_history[-10:]
                win_rate = sum(1 for t in recent_trades if t.get('result') == 'win') / len(recent_trades)
                avg_pnl = np.mean([t.get('pnl', 0) for t in recent_trades])
                features.extend([win_rate, avg_pnl / 100.0])
            else:
                features.extend([0.5, 0.0])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erro ao extrair features: {e}")
            return np.array([50, 0, 1, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0]).reshape(1, -1)
    
    def predict_with_risk_management(self, market_data, trade_history=None):
        """Predição principal com gerenciamento de risco integrado"""
        try:
            # Detectar padrões perigosos
            danger_analysis = self.detect_danger_patterns(market_data, trade_history)
            
            # Se risco crítico, não operar
            if not danger_analysis['should_trade']:
                logger.warning(f"⛔ OPERAÇÃO BLOQUEADA - Risco {danger_analysis['risk_level']}")
                logger.warning(f"Perigos detectados: {', '.join(danger_analysis['dangers'])}")
                return {
                    'should_trade': False,
                    'reason': f"Risco {danger_analysis['risk_level']} - {', '.join(danger_analysis['dangers'])}",
                    'risk_level': danger_analysis['risk_level']
                }
            
            # Fazer predição
            features = self.extract_features(market_data, trade_history)
            prediction_result = self.get_best_prediction(features, market_data)
            
            # Ajustar confiança baseada em risco
            adjusted_confidence = max(50, prediction_result['confidence'] - danger_analysis['confidence_penalty'])
            
            # Calcular stake ótimo
            optimal_stake = self.bankroll_manager.calculate_optimal_stake(
                adjusted_confidence,
                market_data
            )
            
            # Selecionar timeframe ótimo
            optimal_timeframe = self.timeframe_selector.select_optimal_timeframe(
                market_data,
                adjusted_confidence,
                market_data.get('trend_strength', 0)
            )
            
            # Verificar Martingale
            use_martingale = False
            martingale_level = 0
            
            if trade_history and len(trade_history) > 0:
                last_result = trade_history[-1].get('result', 'win')
                use_martingale, martingale_level = self.bankroll_manager.should_use_martingale(last_result)
                
                if use_martingale:
                    optimal_stake *= self.bankroll_manager.martingale_multiplier
                    logger.info(f"🎲 Martingale Nível {martingale_level} - Stake ajustado para ${optimal_stake:.2f}")
            
            # Resultado final
            result = {
                'should_trade': True,
                'direction': prediction_result['direction'],
                'confidence': adjusted_confidence,
                'original_confidence': prediction_result['confidence'],
                'stake': optimal_stake,
                'timeframe': optimal_timeframe,
                'martingale_active': use_martingale,
                'martingale_level': martingale_level,
                'risk_level': danger_analysis['risk_level'],
                'dangers_detected': danger_analysis['dangers'],
                'method': prediction_result['method'],
                'bankroll': self.bankroll_manager.current_balance,
                'market_conditions': {
                    'rsi': market_data['rsi'],
                    'volatility': market_data['volatility'],
                    'trend': market_data.get('trend', 'neutral')
                }
            }
            
            # Log detalhado
            logger.info(f"🎯 SINAL: {result['direction']} | "
                       f"Confiança: {adjusted_confidence:.1f}% | "
                       f"Stake: ${optimal_stake:.2f} | "
                       f"TimeFrame: {optimal_timeframe} | "
                       f"Risco: {danger_analysis['risk_level']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição com gerenciamento: {e}")
            return {
                'should_trade': False,
                'reason': 'Erro no sistema',
                'error': str(e)
            }
    
    def get_best_prediction(self, features, market_data):
        """Sistema de predição ensemble"""
        try:
            predictions = {}
            
            # Modelo offline
            if self.offline_trained:
                try:
                    features_scaled = self.scaler.transform(features)
                    offline_pred = self.offline_model.predict(features_scaled)[0]
                    offline_proba = self.offline_model.predict_proba(features_scaled)[0]
                    predictions['offline'] = {
                        'prediction': offline_pred,
                        'confidence': max(offline_proba)
                    }
                except:
                    pass
            
            # Modelo online SGD
            if self.online_initialized:
                try:
                    features_scaled = self.online_scaler.transform(features)
                    online_pred = self.online_model.predict(features_scaled)[0]
                    online_proba = self.online_model.predict_proba(features_scaled)[0]
                    predictions['online_sgd'] = {
                        'prediction': online_pred,
                        'confidence': max(online_proba)
                    }
                except:
                    pass
            
            # Modelo Passive Aggressive
            if self.passive_initialized:
                try:
                    features_scaled = self.online_scaler.transform(features)
                    passive_pred = self.passive_model.predict(features_scaled)[0]
                    decision = self.passive_model.decision_function(features_scaled)[0]
                    passive_confidence = 1 / (1 + np.exp(-abs(decision)))
                    predictions['passive'] = {
                        'prediction': passive_pred,
                        'confidence': passive_confidence
                    }
                except:
                    pass
            
            # Escolher melhor predição
            if predictions:
                # Voting system com peso por confiança
                weighted_call = 0
                weighted_put = 0
                total_weight = 0
                
                for model_name, pred in predictions.items():
                    weight = pred['confidence']
                    if pred['prediction'] == 1:
                        weighted_call += weight
                    else:
                        weighted_put += weight
                    total_weight += weight
                
                if weighted_call > weighted_put:
                    direction = 'CALL'
                    confidence = (weighted_call / total_weight) * 100 if total_weight > 0 else 60
                else:
                    direction = 'PUT'
                    confidence = (weighted_put / total_weight) * 100 if total_weight > 0 else 60
                
                return {
                    'direction': direction,
                    'confidence': min(95, confidence),
                    'method': 'ensemble_voting',
                    'all_predictions': predictions
                }
            
            # Fallback para análise técnica
            return self.technical_analysis_prediction(market_data)
            
        except Exception as e:
            logger.error(f"Erro no ensemble: {e}")
            return self.technical_analysis_prediction(market_data)
    
    def technical_analysis_prediction(self, market_data):
        """Predição baseada em análise técnica"""
        try:
            rsi = market_data['rsi']
            macd = market_data['macd']
            price = market_data['price']
            bb_upper = market_data['bb_upper']
            bb_lower = market_data['bb_lower']
            trend = market_data.get('trend', 'neutral')
            
            signals = []
            
            # RSI
            if rsi < 30:
                signals.append(('CALL', 0.8))
            elif rsi > 70:
                signals.append(('PUT', 0.8))
            elif rsi < 40:
                signals.append(('CALL', 0.6))
            elif rsi > 60:
                signals.append(('PUT', 0.6))
            
            # MACD
            if macd > 0:
                signals.append(('CALL', 0.7))
            else:
                signals.append(('PUT', 0.7))
            
            # Bollinger Bands
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            if bb_position < 0.2:
                signals.append(('CALL', 0.9))
            elif bb_position > 0.8:
                signals.append(('PUT', 0.9))
            
            # Tendência
            if trend in ['strong_bullish', 'bullish']:
                signals.append(('CALL', 0.6))
            elif trend in ['strong_bearish', 'bearish']:
                signals.append(('PUT', 0.6))
            
            # Combinar sinais
            call_weight = sum(weight for direction, weight in signals if direction == 'CALL')
            put_weight = sum(weight for direction, weight in signals if direction == 'PUT')
            
            if call_weight > put_weight:
                direction = 'CALL'
                confidence = min(90, (call_weight / (call_weight + put_weight)) * 100)
            else:
                direction = 'PUT'
                confidence = min(90, (put_weight / (call_weight + put_weight)) * 100)
            
            return {
                'direction': direction,
                'confidence': max(60, confidence),
                'method': 'technical_analysis'
            }
            
        except Exception as e:
            logger.error(f"Erro na análise técnica: {e}")
            return {
                'direction': 'CALL' if np.random.random() > 0.5 else 'PUT',
                'confidence': 60.0,
                'method': 'random_fallback'
            }
    
    def update_online_model(self, features, target):
        """Atualização incremental do modelo online"""
        try:
            features_flat = features.flatten()
            
            if not self.online_initialized:
                self.feature_buffer.append(features_flat)
                self.target_buffer.append(target)
                
                if len(self.feature_buffer) >= self.min_samples_init:
                    self.initialize_online_models()
                
                return False
            
            # Atualização incremental
            X_scaled = self.online_scaler.transform([features_flat])
            
            self.online_model.partial_fit(X_scaled, [target])
            self.passive_model.partial_fit(X_scaled, [target])
            
            # Atualizar métricas
            self.online_metrics['total_predictions'] += 1
            self.online_metrics['learning_updates'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na atualização online: {e}")
            return False
    
    def initialize_online_models(self):
        """Inicializa modelos online"""
        try:
            if len(self.feature_buffer) < self.min_samples_init:
                return False
            
            X = np.array(self.feature_buffer)
            y = np.array(self.target_buffer)
            
            self.online_scaler.fit(X)
            X_scaled = self.online_scaler.transform(X)
            
            self.online_model.partial_fit(X_scaled, y, classes=[0, 1])
            self.passive_model.partial_fit(X_scaled, y, classes=[0, 1])
            
            self.online_initialized = True
            self.passive_initialized = True
            
            # Limpar buffers
            self.feature_buffer = []
            self.target_buffer = []
            
            logger.info("✅ Modelos online inicializados")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelos: {e}")
            return False
    
    def train_offline_model(self):
        """Treina modelo offline com dados históricos"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE result IN ('win', 'loss')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', conn)
            
            if len(trades_df) < 50:
                logger.info("Dados insuficientes para treinar modelo offline")
                conn.close()
                return False
            
            X = []
            y = []
            
            for _, trade in trades_df.iterrows():
                try:
                    features = json.loads(trade['features']) if trade['features'] else []
                    if len(features) >= 9:
                        X.append(features[:10])  # Usar apenas primeiras 10 features
                        y.append(1 if trade['result'] == 'win' else 0)
                except:
                    continue
            
            if len(X) < 50:
                logger.info("Features insuficientes")
                conn.close()
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.offline_model.fit(X_train_scaled, y_train)
            
            accuracy = self.offline_model.score(X_test_scaled, y_test)
            logger.info(f"✅ Modelo offline treinado - Accuracy: {accuracy:.3f}")
            
            self.offline_trained = True
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {e}")
            return False

# Instância global
trading_ai = EnhancedTradingAI()

# ====================================
# ROTAS DA API
# ====================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Enhanced Trading AI with Risk Management',
        'version': '3.0.0',
        'timestamp': datetime.now().isoformat(),
        'bankroll': {
            'current': trading_ai.bankroll_manager.current_balance,
            'initial': trading_ai.bankroll_manager.initial_balance,
            'consecutive_losses': trading_ai.bankroll_manager.consecutive_losses,
            'recovery_mode': trading_ai.bankroll_manager.recovery_mode
        },
        'models': {
            'offline_trained': trading_ai.offline_trained,
            'online_initialized': trading_ai.online_initialized
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de predição com gerenciamento completo"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'R_50')
        trade_history = data.get('trade_history', [])
        
        # Obter dados de mercado
        market_data = trading_ai.get_market_data(symbol)
        
        # Fazer predição com gerenciamento de risco
        result = trading_ai.predict_with_risk_management(market_data, trade_history)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return jsonify({'error': str(e), 'should_trade': False}), 500

@app.route('/report_trade', methods=['POST'])
def report_trade():
    """Reporta resultado de trade e atualiza sistemas"""
    try:
        data = request.get_json()
        
        # Validação
        required = ['symbol', 'direction', 'result', 'entry_price', 'stake']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Campo obrigatório: {field}'}), 400
        
        # Calcular PnL
        stake = data['stake']
        result = data['result'].lower()
        pnl = stake * 0.85 if result == 'win' else -stake
        
        # Atualizar banca
        trading_ai.bankroll_manager.update_balance(result, pnl)
        
        # Atualizar timeframe selector
        if 'timeframe_used' in data:
            trading_ai.timeframe_selector.record_result(data['timeframe_used'], result)
        
        # Obter dados de mercado
        market_data = trading_ai.get_market_data(data['symbol'])
        
        # Extrair features e atualizar modelo online
        features = trading_ai.extract_features(market_data, data.get('trade_history', []))
        target = 1 if result == 'win' else 0
        trading_ai.update_online_model(features, target)
        
        # Salvar no banco
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, direction, stake, duration, entry_price, exit_price,
             result, pnl, martingale_level, features, bankroll_before, bankroll_after,
             risk_level, timeframe_used, prediction_confidence, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            data['symbol'],
            data['direction'],
            stake,
            data.get('duration', '1m'),
            data['entry_price'],
            data.get('exit_price', data['entry_price']),
            result,
            pnl,
            data.get('martingale_level', 0),
            json.dumps(features.flatten().tolist()),
            trading_ai.bankroll_manager.current_balance - pnl,
            trading_ai.bankroll_manager.current_balance,
            data.get('risk_level', 'MEDIUM'),
            data.get('timeframe_used', '1m'),
            data.get('confidence', 0),
            data.get('model_used', 'unknown')
        ))
        
        conn.commit()
        conn.close()
        
        # Salvar métricas de risco periodicamente
        if trading_ai.online_metrics['learning_updates'] % 5 == 0:
            save_risk_metrics()
        
        return jsonify({
            'message': 'Trade reportado com sucesso',
            'bankroll': {
                'current': trading_ai.bankroll_manager.current_balance,
                'pnl': pnl,
                'consecutive_losses': trading_ai.bankroll_manager.consecutive_losses,
                'recovery_mode': trading_ai.bankroll_manager.recovery_mode
            },
            'performance': trading_ai.bankroll_manager.analyze_recent_performance()
        })
        
    except Exception as e:
        logger.error(f"Erro ao reportar trade: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Estatísticas detalhadas do sistema"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        
        # Estatísticas gerais
        trades_df = pd.read_sql_query('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100', conn)
        
        if len(trades_df) == 0:
            return jsonify({
                'total_trades': 0,
                'bankroll': trading_ai.bankroll_manager.current_balance
            })
        
        wins = len(trades_df[trades_df['result'] == 'win'])
        losses = len(trades_df[trades_df['result'] == 'loss'])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        
        # Performance por timeframe
        tf_stats = {}
        if 'timeframe_used' in trades_df.columns:
            for tf in trades_df['timeframe_used'].unique():
                if tf:
                    tf_trades = trades_df[trades_df['timeframe_used'] == tf]
                    tf_wins = len(tf_trades[tf_trades['result'] == 'win'])
                    tf_total = len(tf_trades)
                    tf_stats[tf] = {
                        'total': tf_total,
                        'wins': tf_wins,
                        'win_rate': tf_wins / tf_total if tf_total > 0 else 0
                    }
        
        conn.close()
        
        return jsonify({
            'total_trades': len(trades_df),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'bankroll': {
                'current': trading_ai.bankroll_manager.current_balance,
                'initial': trading_ai.bankroll_manager.initial_balance,
                'roi': ((trading_ai.bankroll_manager.current_balance - 
                         trading_ai.bankroll_manager.initial_balance) / 
                        trading_ai.bankroll_manager.initial_balance * 100)
            },
            'risk_status': {
                'consecutive_losses': trading_ai.bankroll_manager.consecutive_losses,
                'recovery_mode': trading_ai.bankroll_manager.recovery_mode,
                'recent_performance': trading_ai.bankroll_manager.analyze_recent_performance()
            },
            'timeframe_performance': tf_stats,
            'models': {
                'offline_trained': trading_ai.offline_trained,
                'online_initialized': trading_ai.online_initialized,
                'online_updates': trading_ai.online_metrics['learning_updates']
            }
        })
        
    except Exception as e:
        logger.error(f"Erro nas estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Treina modelos offline"""
    try:
        success = trading_ai.train_offline_model()
        return jsonify({
            'success': success,
            'message': 'Modelo treinado com sucesso' if success else 'Dados insuficientes'
        })
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        return jsonify({'error': str(e)}), 500

def save_risk_metrics():
    """Salva métricas de risco no banco"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_management 
            (timestamp, bankroll, consecutive_losses, consecutive_wins,
             risk_level, recovery_mode, max_drawdown, current_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            trading_ai.bankroll_manager.current_balance,
            trading_ai.bankroll_manager.consecutive_losses,
            trading_ai.bankroll_manager.consecutive_wins,
            'HIGH' if trading_ai.bankroll_manager.recovery_mode else 'NORMAL',
            trading_ai.bankroll_manager.recovery_mode,
            0,  # Calcular drawdown máximo se necessário
            (trading_ai.bankroll_manager.initial_balance - 
             trading_ai.bankroll_manager.current_balance) / 
            trading_ai.bankroll_manager.initial_balance * 100
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Erro ao salvar métricas: {e}")

if __name__ == '__main__':
    # Tentar treinar modelo na inicialização
    try:
        logger.info("Inicializando sistema de trading...")
        trading_ai.train_offline_model()
    except:
        pass
    
    logger.info(f"🚀 Sistema de Trading Avançado rodando na porta {API_PORT}")
    logger.info(f"💰 Banca inicial: ${trading_ai.bankroll_manager.initial_balance}")
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
