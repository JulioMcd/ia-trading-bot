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
# MUDANÇA: Modelos que suportam online learning
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

class OnlineTradingAI:
    def __init__(self):
        # SISTEMA HÍBRIDO: Offline + Online Learning
        self.offline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # MODELOS ONLINE LEARNING
        self.online_model = SGDClassifier(
            loss='log_loss', 
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            max_iter=1000
        )
        
        # Modelo alternativo para comparação
        self.passive_model = PassiveAggressiveClassifier(
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        self.scaler = StandardScaler()
        self.online_scaler = StandardScaler()
        
        # CONTROLES DE ESTADO
        self.offline_trained = False
        self.online_initialized = False
        self.passive_initialized = False
        
        # BUFFERS PARA INICIALIZAÇÃO ONLINE
        self.feature_buffer = []
        self.target_buffer = []
        self.min_samples_init = 20
        
        # MÉTRICAS EM TEMPO REAL
        self.online_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': [],
            'last_10_trades': [],
            'learning_updates': 0
        }
        
        self.indicators = TechnicalIndicators()
        self.init_database()
        
    def init_database(self):
        """Inicializa o banco de dados com colunas para online learning"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            # Tabela de trades com campos para online learning
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
                    market_scenario TEXT
                )
            ''')
            
            # Tabela para métricas de online learning
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
            
            # Demais tabelas mantidas...
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
                    adaptation_score REAL
                )
            ''')
            
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
                    data_type TEXT DEFAULT 'real',
                    market_scenario TEXT,
                    synthetic_params TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Banco de dados de online learning inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco: {e}")
    
    def get_market_data(self, symbol, force_scenario=None):
        """Obtém dados de mercado em tempo real com laboratório sintético"""
        try:
            # Para índices sintéticos Deriv, usar laboratório sintético
            if symbol.startswith(('R_', '1HZ', 'CRASH', 'BOOM', 'JD', 'STEP')):
                scenario = force_scenario or self.get_market_scenario(symbol)
                return self.get_synthetic_data(symbol, scenario)
            
            # Para outros símbolos, tentar yfinance primeiro
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1m")
                
                if not data.empty:
                    return self.process_market_data(data, symbol, data_type='real')
            except Exception as e:
                logger.warning(f"yfinance falhou para {symbol}: {e}")
                pass
                
            # Fallback para laboratório sintético
            scenario = force_scenario or 'normal'
            logger.info(f"Usando laboratório sintético para {symbol} (cenário: {scenario})")
            return self.get_synthetic_data(symbol, scenario)
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado: {e}")
            return self.get_fallback_data(symbol, data_type='synthetic', scenario='error_fallback')
    
    def get_synthetic_data(self, symbol, scenario='normal'):
        """Gera dados sintéticos baseados em padrões de mercado reais"""
        try:
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            
            # Mapear volatilidade por símbolo
            volatility_map = {
                'R_10': 0.1, 'R_25': 0.25, 'R_50': 0.5, 'R_75': 0.75, 'R_100': 1.0,
                '1HZ10V': 0.1, '1HZ25V': 0.25, '1HZ50V': 0.5, '1HZ75V': 0.75, '1HZ100V': 1.0,
                'CRASH300': 3.0, 'CRASH500': 5.0, 'CRASH1000': 10.0,
                'BOOM300': 3.0, 'BOOM500': 5.0, 'BOOM1000': 10.0
            }
            
            volatility = volatility_map.get(symbol, 0.5)
            base_price = 1000 + np.random.normal(0, 50)
            
            # Gerar série temporal realística
            periods = 100
            prices = [base_price]
            
            for i in range(periods - 1):
                # Movimento browniano com drift e mean reversion
                drift = np.random.normal(0, volatility * 0.01)
                mean_reversion = (1000 - prices[-1]) * 0.001  # Leve mean reversion
                noise = np.random.normal(0, volatility * 0.1)
                
                new_price = prices[-1] * (1 + drift + mean_reversion + noise)
                prices.append(max(0.01, new_price))  # Evitar preços negativos
            
            # Criar DataFrame simulado
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
        """Processa dados de mercado e calcula indicadores técnicos"""
        try:
            close_prices = data['Close'].values
            high_prices = data['High'].values if 'High' in data.columns else close_prices
            low_prices = data['Low'].values if 'Low' in data.columns else close_prices
            volume = data['Volume'].values if 'Volume' in data.columns else [1000] * len(close_prices)
            
            # Calcular indicadores técnicos
            rsi = self.indicators.rsi(close_prices)
            macd = self.indicators.macd(close_prices)
            bb = self.indicators.bollinger_bands(close_prices)
            volatility = self.indicators.volatility(close_prices)
            
            current_data = {
                'symbol': symbol,
                'price': float(close_prices[-1]),
                'rsi': rsi,
                'macd': macd,
                'macd_signal': 0.0,  # Simplificado
                'bb_upper': bb['upper'],
                'bb_lower': bb['lower'],
                'volatility': volatility,
                'volume': float(volume[-1]) if len(volume) > 0 else 1000.0,
                'timestamp': datetime.now().isoformat(),
                # NOVOS CAMPOS PARA IDENTIFICAÇÃO
                'data_type': data_type,
                'market_scenario': scenario,
                'synthetic_params': json.dumps(synthetic_params) if synthetic_params else None
            }
            
            # Salvar no banco com identificação de tipo
            self.save_market_data(current_data)
            
            # Log diferenciado para dados sintéticos
            if data_type == 'synthetic':
                logger.info(f"DADOS SINTÉTICOS {scenario}: {symbol} - "
                           f"Preço: {current_data['price']:.4f}, "
                           f"RSI: {rsi:.1f}, Vol: {volatility:.3f}")
            else:
                logger.info(f"DADOS REAIS: {symbol} - "
                           f"Preço: {current_data['price']:.4f}, "
                           f"RSI: {rsi:.1f}, Vol: {volatility:.3f}")
            
            return current_data
            
        except Exception as e:
            logger.error(f"Erro no processamento de dados: {e}")
            return self.get_fallback_data(symbol, data_type=data_type, scenario=scenario)
    
    def get_fallback_data(self, symbol, data_type='synthetic', scenario='fallback'):
        """Dados de fallback em caso de erro"""
        return {
            'symbol': symbol,
            'price': 1000.0 + np.random.normal(0, 10),
            'rsi': 45.0 + np.random.normal(0, 10),
            'macd': np.random.normal(0, 0.5),
            'macd_signal': 0.0,
            'bb_upper': 1020.0,
            'bb_lower': 980.0,
            'volatility': 1.0 + np.random.normal(0, 0.3),
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'market_scenario': scenario,
            'synthetic_params': json.dumps({'source': 'fallback_emergency'})
        }
    
    def save_market_data(self, data):
        """Salva dados de mercado no banco com identificação de tipo"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data 
                (timestamp, symbol, price, volume, rsi, macd, bb_upper, bb_lower, volatility,
                 data_type, market_scenario, synthetic_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'], data['symbol'], data['price'], data['volume'],
                data['rsi'], data['macd'], data['bb_upper'], data['bb_lower'], 
                data['volatility'], data.get('data_type', 'real'), 
                data.get('market_scenario'), data.get('synthetic_params')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de mercado: {e}")
    
    def get_market_scenario(self, symbol, force_scenario=None):
        """
        Determina qual cenário usar baseado em condições ou força um cenário específico
        """
        if force_scenario:
            return force_scenario
        
        # Auto-detecção de cenário baseado no histórico recente
        try:
            conn = sqlite3.connect(DATABASE_URL)
            recent_data = pd.read_sql_query('''
                SELECT price, volatility, timestamp FROM market_data 
                WHERE symbol = ? AND data_type = 'real'
                ORDER BY timestamp DESC LIMIT 10
            ''', conn, params=[symbol])
            conn.close()
            
            if len(recent_data) < 5:
                return 'normal'
            
            # Analisar tendência recente
            price_change = (recent_data['price'].iloc[0] - recent_data['price'].iloc[-1]) / recent_data['price'].iloc[-1]
            avg_volatility = recent_data['volatility'].mean()
            
            if avg_volatility > 2.0:
                return 'high_volatility'
            elif price_change > 0.05:
                return 'bull_market'
            elif price_change < -0.05:
                return 'bear_market'
            elif avg_volatility < 0.3:
                return 'low_volatility'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Erro na detecção de cenário: {e}")
            return 'normal'
    
    def extract_features(self, market_data, trade_history=None):
        """Extrai features para o modelo de ML"""
        try:
            features = []
            
            # Features de mercado
            features.extend([
                market_data['rsi'],
                market_data['macd'],
                market_data['volatility'],
                (market_data['price'] - market_data['bb_lower']) / (market_data['bb_upper'] - market_data['bb_lower']) if market_data['bb_upper'] != market_data['bb_lower'] else 0.5
            ])
            
            # Features temporais
            now = datetime.now()
            features.extend([
                now.hour / 24.0,  # Hora do dia normalizada
                now.weekday() / 6.0,  # Dia da semana normalizado
                (now.minute % 60) / 60.0  # Minuto normalizado
            ])
            
            # Features de histórico
            if trade_history and len(trade_history) > 0:
                recent_trades = trade_history[-10:]  # Últimos 10 trades
                win_rate = sum(1 for t in recent_trades if t.get('result') == 'win') / len(recent_trades)
                avg_pnl = np.mean([t.get('pnl', 0) for t in recent_trades])
                features.extend([win_rate, avg_pnl / 100.0])  # Normalizar PnL
            else:
                features.extend([0.5, 0.0])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erro ao extrair features: {e}")
            return np.array([50, 0, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0]).reshape(1, -1)
    
    def initialize_online_model(self):
        """Inicializa modelo online com dados do buffer"""
        try:
            if len(self.feature_buffer) < self.min_samples_init:
                logger.info(f"Buffer insuficiente: {len(self.feature_buffer)}/{self.min_samples_init}")
                return False
            
            X = np.array(self.feature_buffer)
            y = np.array(self.target_buffer)
            
            # Inicializar scaler online
            self.online_scaler.fit(X)
            X_scaled = self.online_scaler.transform(X)
            
            # Inicializar modelos online
            self.online_model.partial_fit(X_scaled, y, classes=[0, 1])
            self.passive_model.partial_fit(X_scaled, y, classes=[0, 1])
            
            self.online_initialized = True
            self.passive_initialized = True
            
            # Calcular accuracy inicial
            predictions = self.online_model.predict(X_scaled)
            initial_accuracy = accuracy_score(y, predictions)
            
            logger.info(f"ONLINE LEARNING INICIALIZADO!")
            logger.info(f"Amostras: {len(X)}, Accuracy inicial: {initial_accuracy:.3f}")
            
            # Limpar buffer
            self.feature_buffer = []
            self.target_buffer = []
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar online learning: {e}")
            return False
    
    def update_online_model(self, features, target):
        """Atualização incremental do modelo a cada trade"""
        try:
            features_flat = features.flatten()
            
            # Se não inicializado, adicionar ao buffer
            if not self.online_initialized:
                self.feature_buffer.append(features_flat)
                self.target_buffer.append(target)
                
                logger.info(f"Adicionado ao buffer: {len(self.feature_buffer)}/{self.min_samples_init}")
                
                # Tentar inicializar quando buffer estiver cheio
                if len(self.feature_buffer) >= self.min_samples_init:
                    self.initialize_online_model()
                
                return False
            
            # Modelo já inicializado - atualização incremental
            X_scaled = self.online_scaler.transform([features_flat])
            
            # Fazer predição antes da atualização (para métricas)
            prediction_before = self.online_model.predict(X_scaled)[0]
            confidence_before = max(self.online_model.predict_proba(X_scaled)[0])
            
            # ATUALIZAÇÃO INCREMENTAL
            self.online_model.partial_fit(X_scaled, [target])
            self.passive_model.partial_fit(X_scaled, [target])
            
            # Atualizar métricas
            self.online_metrics['total_predictions'] += 1
            self.online_metrics['learning_updates'] += 1
            
            is_correct = (prediction_before == target)
            if is_correct:
                self.online_metrics['correct_predictions'] += 1
            
            # Manter histórico dos últimos 10 trades
            self.online_metrics['last_10_trades'].append({
                'prediction': prediction_before,
                'actual': target,
                'correct': is_correct,
                'confidence': confidence_before
            })
            
            if len(self.online_metrics['last_10_trades']) > 10:
                self.online_metrics['last_10_trades'].pop(0)
            
            # Calcular accuracy dos últimos 10
            recent_correct = sum(1 for t in self.online_metrics['last_10_trades'] if t['correct'])
            recent_accuracy = recent_correct / len(self.online_metrics['last_10_trades'])
            self.online_metrics['recent_accuracy'].append(recent_accuracy)
            
            if len(self.online_metrics['recent_accuracy']) > 50:
                self.online_metrics['recent_accuracy'].pop(0)
            
            # Log da atualização
            overall_accuracy = self.online_metrics['correct_predictions'] / self.online_metrics['total_predictions']
            
            result_emoji = "✅" if is_correct else "❌"
            target_name = "WIN" if target == 1 else "LOSS"
            
            logger.info(f"ONLINE UPDATE #{self.online_metrics['learning_updates']}")
            logger.info(f"{result_emoji} Predição: {'WIN' if prediction_before == 1 else 'LOSS'}, "
                       f"Real: {target_name}, Confiança: {confidence_before:.3f}")
            logger.info(f"Accuracy Geral: {overall_accuracy:.3f}, "
                       f"Últimos 10: {recent_accuracy:.3f}")
            
            # Salvar métricas no banco periodicamente
            if self.online_metrics['learning_updates'] % 10 == 0:
                self.save_online_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na atualização online: {e}")
            return False
    
    def get_best_prediction(self, market_data, trade_history=None):
        """Combina predições de múltiplos modelos"""
        try:
            features = self.extract_features(market_data, trade_history)
            
            predictions = {}
            
            # 1. Modelo offline (se treinado)
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
            
            # 2. Modelo online SGD
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
            
            # 3. Modelo Passive Aggressive
            if self.passive_initialized:
                try:
                    features_scaled = self.online_scaler.transform(features)
                    passive_pred = self.passive_model.predict(features_scaled)[0]
                    # Passive Aggressive não tem predict_proba, usar confidence baseada em decision_function
                    decision = self.passive_model.decision_function(features_scaled)[0]
                    passive_confidence = 1 / (1 + np.exp(-abs(decision)))  # Sigmoid do decision
                    predictions['passive'] = {
                        'prediction': passive_pred,
                        'confidence': passive_confidence
                    }
                except:
                    pass
            
            # Escolher melhor predição
            if predictions:
                # Preferir modelo online se disponível e confiante
                if 'online_sgd' in predictions and predictions['online_sgd']['confidence'] > 0.6:
                    best = predictions['online_sgd']
                    method = 'online_sgd'
                elif 'passive' in predictions and predictions['passive']['confidence'] > 0.6:
                    best = predictions['passive']
                    method = 'passive_aggressive'
                elif 'offline' in predictions:
                    best = predictions['offline']
                    method = 'offline_random_forest'
                else:
                    best = list(predictions.values())[0]
                    method = list(predictions.keys())[0]
                
                direction = 'CALL' if best['prediction'] == 1 else 'PUT'
                confidence = best['confidence'] * 100
                
                logger.info(f"Melhor predição via {method}: {direction} ({confidence:.1f}%)")
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'method': f'best_of_ensemble_{method}',
                    'all_predictions': predictions
                }
            
            # Fallback para híbrido
            return self.hybrid_prediction(market_data)
            
        except Exception as e:
            logger.error(f"Erro na predição ensemble: {e}")
            return self.hybrid_prediction(market_data)
    
    def save_online_metrics(self):
        """Salva métricas de online learning no banco"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            overall_accuracy = (self.online_metrics['correct_predictions'] / 
                              self.online_metrics['total_predictions']) if self.online_metrics['total_predictions'] > 0 else 0
            
            recent_performance = {
                'last_10_accuracy': sum(1 for t in self.online_metrics['last_10_trades'] if t['correct']) / max(1, len(self.online_metrics['last_10_trades'])),
                'avg_confidence': np.mean([t['confidence'] for t in self.online_metrics['last_10_trades']]) if self.online_metrics['last_10_trades'] else 0,
                'learning_updates': self.online_metrics['learning_updates']
            }
            
            cursor.execute('''
                INSERT INTO online_metrics 
                (timestamp, model_type, accuracy, total_samples, recent_performance, adaptation_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                'sgd_online',
                overall_accuracy,
                self.online_metrics['total_predictions'],
                json.dumps(recent_performance),
                len(self.online_metrics['recent_accuracy']) / 50.0  # Taxa de adaptação
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Métricas online salvas: {overall_accuracy:.3f} accuracy")
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
    
    def train_offline_model(self):
        """Treina modelo offline com dados históricos"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            
            # Obter dados de trades
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE result IN ('win', 'loss')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', conn)
            
            if len(trades_df) < 50:  # Dados insuficientes
                logger.info("Dados insuficientes para treinar modelo offline")
                conn.close()
                return False
            
            # Preparar features e targets
            X = []
            y = []
            
            for _, trade in trades_df.iterrows():
                try:
                    features = json.loads(trade['features']) if trade['features'] else []
                    if len(features) == 9:  # Verificar se tem o número correto de features
                        X.append(features)
                        y.append(1 if trade['result'] == 'win' else 0)
                except:
                    continue
            
            if len(X) < 50:
                logger.info("Features insuficientes para treinar modelo offline")
                conn.close()
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Treinar modelo offline
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.offline_model.fit(X_train_scaled, y_train)
            
            # Avaliar modelo
            accuracy = self.offline_model.score(X_test_scaled, y_test)
            logger.info(f"MODELO OFFLINE treinado com acurácia: {accuracy:.3f}")
            
            self.offline_trained = True
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo offline: {e}")
            return False
    
    def predict_direction(self, market_data, trade_history=None):
        """Prediz direção usando sistema híbrido online/offline"""
        # Usar sistema ensemble com online learning
        return self.get_best_prediction(market_data, trade_history)
    
    def hybrid_prediction(self, market_data):
        """Predição híbrida usando análise técnica + padrões"""
        try:
            rsi = market_data['rsi']
            macd = market_data['macd']
            price = market_data['price']
            bb_upper = market_data['bb_upper']
            bb_lower = market_data['bb_lower']
            volatility = market_data['volatility']
            
            signals = []
            
            # Análise RSI
            if rsi < 30:
                signals.append(('CALL', 0.7))  # Sobrevenda
            elif rsi > 70:
                signals.append(('PUT', 0.7))   # Sobrecompra
            
            # Análise MACD
            if macd > 0:
                signals.append(('CALL', 0.6))
            else:
                signals.append(('PUT', 0.6))
            
            # Análise Bollinger Bands
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            if bb_position < 0.2:
                signals.append(('CALL', 0.8))  # Próximo da banda inferior
            elif bb_position > 0.8:
                signals.append(('PUT', 0.8))   # Próximo da banda superior
            
            # Análise de volatilidade
            if volatility > 2.0:
                # Alta volatilidade favorece reversões
                if rsi > 60:
                    signals.append(('PUT', 0.5))
                elif rsi < 40:
                    signals.append(('CALL', 0.5))
            
            # Combinar sinais
            call_weight = sum(weight for direction, weight in signals if direction == 'CALL')
            put_weight = sum(weight for direction, weight in signals if direction == 'PUT')
            
            if call_weight > put_weight:
                direction = 'CALL'
                confidence = min(95, (call_weight / (call_weight + put_weight)) * 100) if (call_weight + put_weight) > 0 else 65
            else:
                direction = 'PUT'
                confidence = min(95, (put_weight / (call_weight + put_weight)) * 100) if (call_weight + put_weight) > 0 else 65
            
            # Ajustar confiança baseada na volatilidade
            if volatility > 3.0:
                confidence *= 0.8  # Reduzir confiança em alta volatilidade
            
            return {
                'direction': direction,
                'confidence': max(60, confidence),  # Mínimo de 60%
                'method': 'hybrid_technical_analysis'
            }
            
        except Exception as e:
            logger.error(f"Erro na predição híbrida: {e}")
            return {
                'direction': 'CALL' if np.random.random() > 0.5 else 'PUT',
                'confidence': 65.0,
                'method': 'fallback_random'
            }

# Instância global da IA com Online Learning
trading_ai = OnlineTradingAI()

# ====================================
# ROTAS DA API
# ====================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Trading AI API - Online Learning',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'offline_trained': trading_ai.offline_trained,
            'online_initialized': trading_ai.online_initialized,
            'passive_initialized': trading_ai.passive_initialized
        },
        'online_metrics': {
            'total_predictions': trading_ai.online_metrics['total_predictions'],
            'accuracy': (trading_ai.online_metrics['correct_predictions'] / 
                        max(1, trading_ai.online_metrics['total_predictions'])),
            'learning_updates': trading_ai.online_metrics['learning_updates']
        },
        'database': 'connected'
    })

@app.route('/analyze', methods=['POST'])
def analyze_market():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'R_100')
        
        # Obter dados de mercado
        market_data = trading_ai.get_market_data(symbol)
        
        # Análise detalhada
        analysis = {
            'symbol': symbol,
            'current_price': market_data['price'],
            'timestamp': market_data['timestamp'],
            'technical_indicators': {
                'rsi': market_data['rsi'],
                'macd': market_data['macd'],
                'bb_upper': market_data['bb_upper'],
                'bb_lower': market_data['bb_lower'],
                'volatility': market_data['volatility']
            },
            'market_condition': 'neutral',
            'volatility_level': 'medium',
            'trend': 'sideways',
            'online_learning_status': {
                'initialized': trading_ai.online_initialized,
                'total_updates': trading_ai.online_metrics['learning_updates'],
                'recent_accuracy': (sum(1 for t in trading_ai.online_metrics['last_10_trades'] if t['correct']) / 
                                  max(1, len(trading_ai.online_metrics['last_10_trades']))) if trading_ai.online_metrics['last_10_trades'] else 0
            }
        }
        
        # Determinar condições de mercado
        rsi = market_data['rsi']
        if rsi < 30:
            analysis['market_condition'] = 'oversold'
            analysis['trend'] = 'bullish_reversal'
        elif rsi > 70:
            analysis['market_condition'] = 'overbought'
            analysis['trend'] = 'bearish_reversal'
        elif market_data['macd'] > 0:
            analysis['trend'] = 'bullish'
        elif market_data['macd'] < 0:
            analysis['trend'] = 'bearish'
        
        # Nível de volatilidade
        if market_data['volatility'] > 2.0:
            analysis['volatility_level'] = 'high'
        elif market_data['volatility'] < 0.5:
            analysis['volatility_level'] = 'low'
        
        analysis['message'] = f"Análise técnica: RSI {rsi:.1f}, Volatilidade {market_data['volatility']:.2f}, Tendência {analysis['trend']}"
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'R_100')
        
        # Obter dados de mercado
        market_data = trading_ai.get_market_data(symbol)
        
        # Obter histórico de trades (opcional)
        trade_history = data.get('trade_history', [])
        
        # Fazer predição usando ensemble de modelos
        prediction = trading_ai.predict_direction(market_data, trade_history)
        
        return jsonify({
            'symbol': symbol,
            'prediction': prediction['direction'],
            'confidence': prediction['confidence'],
            'method': prediction['method'],
            'market_data': {
                'price': market_data['price'],
                'rsi': market_data['rsi'],
                'macd': market_data['macd'],
                'volatility': market_data['volatility']
            },
            'timestamp': datetime.now().isoformat(),
            'online_learning_active': trading_ai.online_initialized
        })
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/report_trade', methods=['POST'])
def report_trade():
    """Endpoint para reportar resultado de trades e atualizar online learning"""
    try:
        data = request.get_json()
        
        # Validar dados obrigatórios
        required_fields = ['symbol', 'direction', 'result', 'entry_price']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo obrigatório ausente: {field}'}), 400
        
        # Obter dados de mercado no momento do trade
        market_data = trading_ai.get_market_data(data['symbol'])
        
        # Extrair features
        features = trading_ai.extract_features(market_data, data.get('trade_history', []))
        
        # Converter resultado para target numérico
        target = 1 if data['result'].lower() == 'win' else 0
        
        # Atualizar modelo online
        trading_ai.update_online_model(features, target)
        
        # Salvar trade no banco
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': data['symbol'],
            'direction': data['direction'],
            'stake': data.get('stake', 0),
            'duration': data.get('duration', '1m'),
            'entry_price': data['entry_price'],
            'exit_price': data.get('exit_price', data['entry_price']),
            'result': data['result'],
            'pnl': data.get('pnl', 0),
            'martingale_level': data.get('martingale_level', 0),
            'market_conditions': json.dumps(market_data),
            'features': json.dumps(features.flatten().tolist()),
            'online_updated': True,
            'prediction_confidence': data.get('confidence', 0),
            'model_used': 'online_ensemble',
            'learning_iteration': trading_ai.online_metrics['learning_updates']
        }
        
        # Salvar no banco
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, direction, stake, duration, entry_price, exit_price, 
             result, pnl, martingale_level, market_conditions, features, online_updated,
             prediction_confidence, model_used, learning_iteration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_record['timestamp'], trade_record['symbol'], trade_record['direction'],
            trade_record['stake'], trade_record['duration'], trade_record['entry_price'],
            trade_record['exit_price'], trade_record['result'], trade_record['pnl'],
            trade_record['martingale_level'], trade_record['market_conditions'],
            trade_record['features'], trade_record['online_updated'],
            trade_record['prediction_confidence'], trade_record['model_used'],
            trade_record['learning_iteration']
        ))
        
        conn.commit()
        conn.close()
        
        # Resposta com status do aprendizado
        return jsonify({
            'message': 'Trade reportado e modelo atualizado',
            'trade_id': cursor.lastrowid,
            'online_learning_status': {
                'initialized': trading_ai.online_initialized,
                'total_updates': trading_ai.online_metrics['learning_updates'],
                'recent_accuracy': (sum(1 for t in trading_ai.online_metrics['last_10_trades'] if t['correct']) / 
                                  max(1, len(trading_ai.online_metrics['last_10_trades']))) if trading_ai.online_metrics['last_10_trades'] else 0,
                'overall_accuracy': trading_ai.online_metrics['correct_predictions'] / max(1, trading_ai.online_metrics['total_predictions'])
            }
        })
        
    except Exception as e:
        logger.error(f"Erro ao reportar trade: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Obter estatísticas de trading e online learning"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        
        # Estatísticas gerais
        trades_df = pd.read_sql_query('''
            SELECT * FROM trades 
            ORDER BY timestamp DESC
            LIMIT 1000
        ''', conn)
        
        if len(trades_df) == 0:
            return jsonify({
                'total_trades': 0,
                'win_rate': 0,
                'online_learning': {
                    'initialized': trading_ai.online_initialized,
                    'total_updates': 0,
                    'accuracy': 0
                }
            })
        
        # Calcular estatísticas
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['result'] == 'win'])
        losses = len(trades_df[trades_df['result'] == 'loss'])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
        
        # Estatísticas de online learning
        online_stats = {
            'initialized': trading_ai.online_initialized,
            'total_updates': trading_ai.online_metrics['learning_updates'],
            'total_predictions': trading_ai.online_metrics['total_predictions'],
            'correct_predictions': trading_ai.online_metrics['correct_predictions'],
            'overall_accuracy': trading_ai.online_metrics['correct_predictions'] / max(1, trading_ai.online_metrics['total_predictions']),
            'recent_accuracy': (sum(1 for t in trading_ai.online_metrics['last_10_trades'] if t['correct']) / 
                              max(1, len(trading_ai.online_metrics['last_10_trades']))) if trading_ai.online_metrics['last_10_trades'] else 0,
            'buffer_size': len(trading_ai.feature_buffer),
            'buffer_required': trading_ai.min_samples_init
        }
        
        # Últimos trades
        recent_trades = trades_df.head(10).to_dict('records') if len(trades_df) > 0 else []
        
        conn.close()
        
        return jsonify({
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'recent_trades': recent_trades,
            'online_learning': online_stats,
            'models': {
                'offline_trained': trading_ai.offline_trained,
                'online_initialized': trading_ai.online_initialized,
                'passive_initialized': trading_ai.passive_initialized
            }
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Endpoint para treinar modelos offline"""
    try:
        # Tentar treinar modelo offline
        success = trading_ai.train_offline_model()
        
        if success:
            return jsonify({
                'message': 'Modelo offline treinado com sucesso',
                'offline_trained': True,
                'online_initialized': trading_ai.online_initialized
            })
        else:
            return jsonify({
                'message': 'Dados insuficientes para treinar modelo offline',
                'offline_trained': False,
                'required_trades': 50
            })
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/reset_online', methods=['POST'])
def reset_online_learning():
    """Reset do sistema de online learning"""
    try:
        # Reset dos modelos online
        trading_ai.online_initialized = False
        trading_ai.passive_initialized = False
        
        # Limpar buffers
        trading_ai.feature_buffer = []
        trading_ai.target_buffer = []
        
        # Reset das métricas
        trading_ai.online_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': [],
            'last_10_trades': [],
            'learning_updates': 0
        }
        
        # Recriar modelos
        trading_ai.online_model = SGDClassifier(
            loss='log_loss', 
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            max_iter=1000
        )
        
        trading_ai.passive_model = PassiveAggressiveClassifier(
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        trading_ai.online_scaler = StandardScaler()
        
        logger.info("Sistema de online learning resetado")
        
        return jsonify({
            'message': 'Sistema de online learning resetado com sucesso',
            'status': {
                'online_initialized': False,
                'passive_initialized': False,
                'buffer_size': 0,
                'learning_updates': 0
            }
        })
        
    except Exception as e:
        logger.error(f"Erro no reset: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    # Tentar treinar modelo offline na inicialização
    try:
        logger.info("Tentando treinar modelo offline na inicialização...")
        trading_ai.train_offline_model()
    except Exception as e:
        logger.warning(f"Não foi possível treinar modelo offline: {e}")
    
    logger.info(f"Iniciando Trading AI API na porta {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
