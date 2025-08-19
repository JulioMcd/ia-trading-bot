#!/usr/bin/env python3
"""
Sistema de Machine Learning Avançado para Trading
Features robustas, validação temporal, ensemble de modelos
"""

import os
import json
import sqlite3
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """Sistema avançado de feature engineering"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = RobustScaler()  # Mais robusto a outliers
        self.is_fitted = False
        
    def create_technical_features(self, prices: np.ndarray, window: int = 20) -> Dict[str, float]:
        """Cria features técnicas avançadas"""
        if len(prices) < window:
            # Se não temos dados suficientes, retornar valores padrão
            return self._get_default_technical_features()
        
        features = {}
        
        # 1. Momentum e Tendência
        features['price_momentum_5'] = self._calculate_momentum(prices, 5)
        features['price_momentum_10'] = self._calculate_momentum(prices, 10)
        features['price_momentum_20'] = self._calculate_momentum(prices, min(20, len(prices)-1))
        
        # 2. Volatilidade
        features['realized_volatility'] = self._calculate_realized_volatility(prices)
        features['volatility_percentile'] = self._calculate_volatility_percentile(prices)
        features['price_acceleration'] = self._calculate_acceleration(prices)
        
        # 3. Médias Móveis
        features['sma_ratio_5'] = prices[-1] / np.mean(prices[-5:]) if len(prices) >= 5 else 1.0
        features['sma_ratio_10'] = prices[-1] / np.mean(prices[-10:]) if len(prices) >= 10 else 1.0
        features['ema_ratio'] = self._calculate_ema_ratio(prices)
        
        # 4. Bandas de Bollinger
        bb_features = self._calculate_bollinger_bands(prices)
        features.update(bb_features)
        
        # 5. RSI
        features['rsi'] = self._calculate_rsi(prices)
        features['rsi_divergence'] = self._calculate_rsi_divergence(prices)
        
        # 6. MACD
        macd_features = self._calculate_macd(prices)
        features.update(macd_features)
        
        # 7. Price Action
        features['price_range_norm'] = self._calculate_normalized_range(prices)
        features['support_resistance'] = self._calculate_support_resistance_strength(prices)
        
        return features
    
    def create_market_features(self, trade_data: Dict) -> Dict[str, float]:
        """Cria features relacionadas ao mercado"""
        features = {}
        
        # 1. Horário de trading (importante para volatilidade)
        timestamp = trade_data.get('timestamp', datetime.now().isoformat())
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00').replace('+00:00', ''))
        
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()
        features['is_weekend'] = 1.0 if dt.weekday() >= 5 else 0.0
        
        # 2. Sessões de mercado (overlaps são importantes)
        features['london_session'] = 1.0 if 8 <= dt.hour <= 16 else 0.0
        features['new_york_session'] = 1.0 if 13 <= dt.hour <= 21 else 0.0
        features['tokyo_session'] = 1.0 if 23 <= dt.hour or dt.hour <= 7 else 0.0
        features['session_overlap'] = features['london_session'] * features['new_york_session']
        
        # 3. Duration encoding
        duration_str = str(trade_data.get('duration', '5t'))
        features['duration_numeric'] = self._parse_duration(duration_str)
        features['is_tick_duration'] = 1.0 if 't' in duration_str.lower() else 0.0
        
        # 4. Symbol encoding
        symbol = trade_data.get('symbol', 'R_50')
        features['volatility_index'] = self._encode_volatility_index(symbol)
        features['is_crash_boom'] = 1.0 if 'CRASH' in symbol or 'BOOM' in symbol else 0.0
        features['is_jump_index'] = 1.0 if 'JD' in symbol else 0.0
        
        return features
    
    def create_risk_features(self, trade_data: Dict, recent_trades: List[Dict]) -> Dict[str, float]:
        """Cria features relacionadas ao risco"""
        features = {}
        
        # 1. Martingale level (risco exponencial)
        martingale_level = trade_data.get('martingale_level', 0)
        features['martingale_level'] = float(martingale_level)
        features['martingale_risk'] = 2 ** martingale_level  # Risco exponencial
        
        # 2. Stake features
        current_stake = trade_data.get('stake', 1.0)
        features['stake_normalized'] = current_stake / 10.0  # Normalizar
        features['stake_log'] = np.log1p(current_stake)
        
        # 3. Histórico recente
        if recent_trades:
            recent_pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
            features['recent_win_rate'] = len([p for p in recent_pnls if p > 0]) / len(recent_pnls)
            features['recent_avg_pnl'] = np.mean(recent_pnls)
            features['recent_volatility_pnl'] = np.std(recent_pnls) if len(recent_pnls) > 1 else 0
            features['consecutive_losses'] = self._count_consecutive_losses(recent_trades)
            features['max_recent_loss'] = min(recent_pnls) if recent_pnls else 0
        else:
            features['recent_win_rate'] = 0.5
            features['recent_avg_pnl'] = 0.0
            features['recent_volatility_pnl'] = 0.0
            features['consecutive_losses'] = 0.0
            features['max_recent_loss'] = 0.0
        
        # 4. Risk-adjusted features
        features['kelly_estimate'] = self._estimate_kelly_fraction(recent_trades)
        features['sharpe_estimate'] = self._estimate_sharpe_ratio(recent_trades)
        
        return features
    
    def create_all_features(self, trade_data: Dict, price_history: List[float], 
                           recent_trades: List[Dict]) -> np.ndarray:
        """Cria todas as features combinadas"""
        all_features = {}
        
        # 1. Features técnicas
        if price_history and len(price_history) > 0:
            prices = np.array(price_history)
            technical_features = self.create_technical_features(prices)
            all_features.update(technical_features)
        else:
            all_features.update(self._get_default_technical_features())
        
        # 2. Features de mercado
        market_features = self.create_market_features(trade_data)
        all_features.update(market_features)
        
        # 3. Features de risco
        risk_features = self.create_risk_features(trade_data, recent_trades)
        all_features.update(risk_features)
        
        # 4. Features de interação
        interaction_features = self._create_interaction_features(all_features)
        all_features.update(interaction_features)
        
        # Converter para array numpy
        if not self.feature_names:
            self.feature_names = sorted(all_features.keys())
        
        feature_vector = np.array([all_features.get(name, 0.0) for name in self.feature_names])
        
        return feature_vector.reshape(1, -1)
    
    def _get_default_technical_features(self) -> Dict[str, float]:
        """Features técnicas padrão quando não há dados suficientes"""
        return {
            'price_momentum_5': 0.0,
            'price_momentum_10': 0.0,
            'price_momentum_20': 0.0,
            'realized_volatility': 50.0,
            'volatility_percentile': 50.0,
            'price_acceleration': 0.0,
            'sma_ratio_5': 1.0,
            'sma_ratio_10': 1.0,
            'ema_ratio': 1.0,
            'bb_position': 0.5,
            'bb_width': 0.05,
            'bb_squeeze': 0.0,
            'rsi': 50.0,
            'rsi_divergence': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'price_range_norm': 0.02,
            'support_resistance': 0.5
        }
    
    def _calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calcula momentum de preços"""
        if len(prices) < period + 1:
            return 0.0
        return (prices[-1] / prices[-period-1] - 1) * 100
    
    def _calculate_realized_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """Calcula volatilidade realizada"""
        if len(prices) < 2:
            return 50.0
        
        returns = np.diff(np.log(prices))
        vol = np.std(returns) * np.sqrt(252) * 100  # Anualizada
        return min(200, max(1, vol))
    
    def _calculate_volatility_percentile(self, prices: np.ndarray, window: int = 50) -> float:
        """Calcula percentil da volatilidade atual"""
        if len(prices) < window:
            return 50.0
        
        vol_series = []
        for i in range(window, len(prices)):
            returns = np.diff(np.log(prices[i-window:i]))
            vol = np.std(returns)
            vol_series.append(vol)
        
        if not vol_series:
            return 50.0
        
        current_vol = np.std(np.diff(np.log(prices[-window:])))
        percentile = (np.sum(np.array(vol_series) < current_vol) / len(vol_series)) * 100
        
        return percentile
    
    def _calculate_acceleration(self, prices: np.ndarray) -> float:
        """Calcula aceleração do preço"""
        if len(prices) < 3:
            return 0.0
        
        returns = np.diff(np.log(prices))
        if len(returns) < 2:
            return 0.0
        
        acceleration = np.diff(returns)[-1]
        return acceleration * 10000  # Escalar para visualização
    
    def _calculate_ema_ratio(self, prices: np.ndarray, alpha: float = 0.1) -> float:
        """Calcula razão EMA/preço"""
        if len(prices) < 2:
            return 1.0
        
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return prices[-1] / ema if ema > 0 else 1.0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, num_std: float = 2) -> Dict[str, float]:
        """Calcula features das Bandas de Bollinger"""
        if len(prices) < window:
            return {'bb_position': 0.5, 'bb_width': 0.05, 'bb_squeeze': 0.0}
        
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        # Posição dentro das bandas (0 = banda inferior, 1 = banda superior)
        bb_position = (prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
        
        # Largura das bandas (normalizada)
        bb_width = (upper_band - lower_band) / sma if sma > 0 else 0.05
        
        # Squeeze indicator (bandas estreitas)
        bb_squeeze = 1.0 if bb_width < 0.02 else 0.0
        
        return {
            'bb_position': max(0, min(1, bb_position)),
            'bb_width': bb_width,
            'bb_squeeze': bb_squeeze
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_divergence(self, prices: np.ndarray) -> float:
        """Calcula divergência do RSI"""
        if len(prices) < 30:
            return 0.0
        
        # Simplificado: compara tendência do preço vs RSI
        price_trend = (prices[-1] - prices[-10]) / prices[-10]
        
        rsi_current = self._calculate_rsi(prices)
        rsi_past = self._calculate_rsi(prices[:-10])
        rsi_trend = rsi_current - rsi_past
        
        # Divergência: preço sobe mas RSI desce (ou vice-versa)
        divergence = np.sign(price_trend) != np.sign(rsi_trend)
        
        return 1.0 if divergence else 0.0
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calcula MACD"""
        if len(prices) < 26:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
        
        # EMA 12 e 26
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        macd_line = ema12 - ema26
        
        # Signal line (EMA 9 do MACD)
        # Simplificado para este exemplo
        signal_line = macd_line * 0.8  # Aproximação
        
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line / prices[-1] * 1000,  # Normalizado
            'macd_signal': signal_line / prices[-1] * 1000,
            'macd_histogram': histogram / prices[-1] * 1000
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calcula EMA"""
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_normalized_range(self, prices: np.ndarray, window: int = 10) -> float:
        """Calcula range normalizado"""
        if len(prices) < window:
            return 0.02
        
        recent_prices = prices[-window:]
        price_range = (np.max(recent_prices) - np.min(recent_prices))
        normalized_range = price_range / np.mean(recent_prices)
        
        return normalized_range
    
    def _calculate_support_resistance_strength(self, prices: np.ndarray) -> float:
        """Calcula força de suporte/resistência"""
        if len(prices) < 20:
            return 0.5
        
        current_price = prices[-1]
        
        # Encontrar níveis próximos ao preço atual
        tolerance = np.std(prices) * 0.5
        
        support_levels = []
        resistance_levels = []
        
        for i in range(len(prices) - 10):
            window = prices[i:i+10]
            min_price = np.min(window)
            max_price = np.max(window)
            
            if abs(current_price - min_price) < tolerance:
                support_levels.append(min_price)
            
            if abs(current_price - max_price) < tolerance:
                resistance_levels.append(max_price)
        
        # Força baseada no número de toques
        strength = (len(support_levels) + len(resistance_levels)) / 20
        
        return min(1.0, strength)
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string para número"""
        duration_str = str(duration_str).lower()
        
        if 't' in duration_str:
            return float(duration_str.replace('t', '').replace('ticks', ''))
        elif 'm' in duration_str:
            return float(duration_str.replace('m', '').replace('min', '')) * 60
        elif 'h' in duration_str:
            return float(duration_str.replace('h', '').replace('hour', '')) * 3600
        else:
            return float(duration_str)
    
    def _encode_volatility_index(self, symbol: str) -> float:
        """Codifica índice de volatilidade"""
        volatility_map = {
            'R_10': 10, 'R_25': 25, 'R_50': 50, 'R_75': 75, 'R_100': 100,
            '1HZ10V': 10, '1HZ25V': 25, '1HZ50V': 50, '1HZ75V': 75, '1HZ100V': 100,
            '1HZ150V': 150, '1HZ200V': 200, '1HZ250V': 250
        }
        
        return float(volatility_map.get(symbol, 50))
    
    def _count_consecutive_losses(self, trades: List[Dict]) -> float:
        """Conta perdas consecutivas"""
        if not trades:
            return 0.0
        
        consecutive = 0
        for trade in reversed(trades[-10:]):  # Últimos 10 trades
            if trade.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
        
        return float(consecutive)
    
    def _estimate_kelly_fraction(self, trades: List[Dict]) -> float:
        """Estima fração ótima do Kelly"""
        if not trades or len(trades) < 5:
            return 0.01
        
        pnls = [t.get('pnl', 0) for t in trades[-20:]]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        if not wins or not losses:
            return 0.01
        
        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.01
        
        b = avg_win / avg_loss
        kelly = (b * win_rate - (1 - win_rate)) / b
        
        return max(0, min(0.25, kelly))
    
    def _estimate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Estima Sharpe ratio"""
        if not trades or len(trades) < 5:
            return 0.0
        
        pnls = [t.get('pnl', 0) for t in trades[-20:]]
        
        if len(pnls) < 2:
            return 0.0
        
        mean_return = np.mean(pnls)
        std_return = np.std(pnls)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return
        
        return sharpe
    
    def _create_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Cria features de interação entre variáveis"""
        interactions = {}
        
        # Interações importantes identificadas
        
        # 1. Volatilidade x Momentum
        vol = features.get('realized_volatility', 50)
        momentum = features.get('price_momentum_5', 0)
        interactions['vol_momentum'] = (vol / 50) * abs(momentum)
        
        # 2. RSI x Volatilidade
        rsi = features.get('rsi', 50)
        interactions['rsi_vol'] = ((rsi - 50) / 50) * (vol / 50)
        
        # 3. Martingale x Win Rate
        martingale = features.get('martingale_level', 0)
        win_rate = features.get('recent_win_rate', 0.5)
        interactions['martingale_risk'] = martingale * (1 - win_rate)
        
        # 4. Time x Volatility (sessões importantes)
        hour = features.get('hour_of_day', 12)
        london_ny_overlap = 1 if 13 <= hour <= 16 else 0
        interactions['time_vol'] = london_ny_overlap * (vol / 50)
        
        # 5. Support/Resistance x Momentum
        sr_strength = features.get('support_resistance', 0.5)
        interactions['sr_momentum'] = sr_strength * abs(momentum)
        
        return interactions

class AdvancedMLTradingSystem:
    """Sistema ML avançado com ensemble e validação temporal"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.feature_engineer = AdvancedFeatureEngineering()
        self.models = {}
        self.ensemble_model = None
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_classif, k=20)
        self.is_trained = False
        self.last_training = None
        self.training_metrics = {}
        
        # Configurações
        self.min_training_samples = 100
        self.validation_split = 0.2
        self.use_ensemble = True
        
        # Inicializar
        self._init_database()
        self._load_models()
    
    def _init_database(self):
        """Inicializa tabelas necessárias"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de price history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices para performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_time ON price_history(symbol, timestamp)')
        
        conn.commit()
        conn.close()
    
    def train_models(self, force_retrain: bool = False) -> Dict:
        """Treina ensemble de modelos com validação temporal"""
        logger.info("🎓 Iniciando treinamento avançado de modelos ML...")
        
        try:
            # 1. Carregar dados
            df = self._load_training_data()
            
            if len(df) < self.min_training_samples:
                logger.warning(f"⚠️ Dados insuficientes: {len(df)} < {self.min_training_samples}")
                return self._train_with_synthetic_data()
            
            # 2. Preparar features
            X, y, feature_names = self._prepare_advanced_features(df)
            
            if X.shape[0] < self.min_training_samples:
                logger.warning("⚠️ Features insuficientes após processamento")
                return self._train_with_synthetic_data()
            
            # 3. Validação temporal
            train_results = self._train_with_temporal_validation(X, y, feature_names)
            
            # 4. Criar ensemble
            if self.use_ensemble and len(self.models) >= 3:
                self._create_ensemble_model()
            
            # 5. Salvar modelos
            self._save_models()
            
            self.is_trained = True
            self.last_training = {
                'timestamp': datetime.now().isoformat(),
                'samples': X.shape[0],
                'features': X.shape[1],
                'models_trained': len(train_results),
                'best_accuracy': max(train_results.values()) if train_results else 0,
                'ensemble_created': self.ensemble_model is not None
            }
            
            logger.info(f"✅ Treinamento concluído: {len(train_results)} modelos")
            return train_results
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return {}
    
    def _load_training_data(self) -> pd.DataFrame:
        """Carrega dados para treinamento"""
        conn = sqlite3.connect(self.db_path)
        
        # Query otimizada com JOIN para price history
        query = '''
            SELECT t.*, ph.price as historical_price
            FROM trades t
            LEFT JOIN price_history ph ON t.symbol = ph.symbol 
                AND datetime(ph.timestamp) BETWEEN 
                    datetime(t.timestamp, '-1 hour') AND datetime(t.timestamp)
            WHERE t.outcome IS NOT NULL 
            ORDER BY t.timestamp DESC 
            LIMIT 2000
        '''
        
        try:
            df = pd.read_sql_query(query, conn)
            logger.info(f"📊 Dados carregados: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def _prepare_advanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepara features avançadas para treinamento"""
        feature_matrix = []
        targets = []
        
        for idx, row in df.iterrows():
            try:
                # Preparar dados do trade
                trade_data = {
                    'timestamp': row.get('timestamp', datetime.now().isoformat()),
                    'symbol': row.get('symbol', 'R_50'),
                    'duration': row.get('duration', '5t'),
                    'stake': row.get('stake', 1.0),
                    'martingale_level': row.get('martingale_level', 0),
                    'entry_price': row.get('entry_price', 1000.0)
                }
                
                # Obter histórico de preços
                price_history = self._get_price_history_for_trade(trade_data)
                
                # Obter trades recentes
                recent_trades = self._get_recent_trades_for_training(row.get('timestamp'), df)
                
                # Criar features
                features = self.feature_engineer.create_all_features(
                    trade_data=trade_data,
                    price_history=price_history,
                    recent_trades=recent_trades
                )
                
                # Target
                target = 1 if row.get('outcome') == 'won' else 0
                
                feature_matrix.append(features.flatten())
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao processar linha {idx}: {e}")
                continue
        
        if not feature_matrix:
            raise ValueError("Nenhuma feature válida criada")
        
        X = np.vstack(feature_matrix)
        y = np.array(targets)
        
        logger.info(f"🔧 Features criadas: {X.shape[1]} features, {X.shape[0]} amostras")
        
        return X, y, self.feature_engineer.feature_names
    
    def _train_with_temporal_validation(self, X: np.ndarray, y: np.ndarray, 
                                       feature_names: List[str]) -> Dict:
        """Treina com validação temporal (não misturar dados futuros)"""
        
        # 1. Preprocessamento
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Seleção de features
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # 3. Split temporal (últimos 20% para validação)
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_val = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"🔄 Split temporal: {len(X_train)} treino, {len(X_val)} validação")
        
        # 4. Definir modelos
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=20,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(50, 25),
                learning_rate_init=0.01,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # 5. Treinar cada modelo
        results = {}
        
        for name, model in models_config.items():
            try:
                logger.info(f"🔄 Treinando {name}...")
                
                # Treinar
                model.fit(X_train, y_train)
                
                # Validar
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Métricas
                accuracy = accuracy_score(y_val, y_pred)
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                # Cross-validation temporal
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = cross_val_score(model, X_selected, y, cv=tscv, scoring='accuracy')
                
                # Salvar modelo
                self.models[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'trained_at': datetime.now().isoformat(),
                    'samples': len(X_train)
                }
                
                results[name] = accuracy
                
                logger.info(f"✅ {name}: Acc={accuracy:.3f}, AUC={auc_score:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"❌ Erro treinando {name}: {e}")
                continue
        
        return results
    
    def _create_ensemble_model(self):
        """Cria modelo ensemble com voting"""
        try:
            # Selecionar melhores modelos
            model_list = []
            
            for name, model_data in self.models.items():
                if model_data['accuracy'] > 0.45:  # Apenas modelos decentes
                    model_list.append((name, model_data['model']))
            
            if len(model_list) >= 2:
                self.ensemble_model = VotingClassifier(
                    estimators=model_list,
                    voting='soft'  # Usa probabilidades
                )
                
                logger.info(f"🎯 Ensemble criado com {len(model_list)} modelos")
            else:
                logger.warning("⚠️ Modelos insuficientes para ensemble")
                
        except Exception as e:
            logger.error(f"❌ Erro ao criar ensemble: {e}")
    
    def _train_with_synthetic_data(self) -> Dict:
        """Fallback: treina com dados sintéticos melhorados"""
        logger.info("🎲 Criando dados sintéticos melhorados...")
        
        n_samples = 500
        np.random.seed(42)
        
        # Criar features sintéticas mais realistas
        features = []
        targets = []
        
        for i in range(n_samples):
            # Simular condições de mercado variadas
            market_condition = np.random.choice(['bull', 'bear', 'sideways'], p=[0.3, 0.3, 0.4])
            
            # Features baseadas na condição do mercado
            if market_condition == 'bull':
                momentum = np.random.normal(2, 1)
                volatility = np.random.normal(30, 10)
                win_rate_bias = 0.6
            elif market_condition == 'bear':
                momentum = np.random.normal(-2, 1)
                volatility = np.random.normal(45, 15)
                win_rate_bias = 0.4
            else:  # sideways
                momentum = np.random.normal(0, 0.5)
                volatility = np.random.normal(25, 8)
                win_rate_bias = 0.5
            
            # Criar feature vector sintético
            feature_dict = {
                'price_momentum_5': momentum + np.random.normal(0, 0.5),
                'realized_volatility': max(5, volatility),
                'rsi': np.random.normal(50, 20),
                'martingale_level': np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05]),
                'recent_win_rate': np.random.beta(2, 2),  # Distribuição mais realista
                'hour_of_day': np.random.randint(0, 24),
                'volatility_index': np.random.choice([10, 25, 50, 75, 100]),
            }
            
            # Target baseado em lógica mais sofisticada
            score = (
                (feature_dict['price_momentum_5'] > 0) * 0.3 +
                (feature_dict['rsi'] < 30 or feature_dict['rsi'] > 70) * 0.2 +
                (feature_dict['recent_win_rate'] > 0.6) * 0.3 +
                (feature_dict['martingale_level'] == 0) * 0.2
            )
            
            win_prob = win_rate_bias + score * 0.3 + np.random.normal(0, 0.1)
            target = 1 if win_prob > 0.5 else 0
            
            # Preencher features faltantes com valores padrão
            default_features = self.feature_engineer._get_default_technical_features()
            for key, value in default_features.items():
                if key not in feature_dict:
                    feature_dict[key] = value
            
            # Ordenar features
            if not self.feature_engineer.feature_names:
                self.feature_engineer.feature_names = sorted(feature_dict.keys())
            
            feature_vector = np.array([feature_dict.get(name, 0.0) for name in self.feature_engineer.feature_names])
            
            features.append(feature_vector)
            targets.append(target)
        
        X = np.array(features)
        y = np.array(targets)
        
        return self._train_with_temporal_validation(X, y, self.feature_engineer.feature_names)
    
    def predict(self, trade_data: Dict, price_history: List[float] = None, 
                recent_trades: List[Dict] = None) -> Dict:
        """Faz predição usando ensemble ou melhor modelo"""
        try:
            if not self.models:
                return self._default_prediction("Nenhum modelo treinado")
            
            # Criar features
            if price_history is None:
                price_history = self._get_price_history_for_trade(trade_data)
            
            if recent_trades is None:
                recent_trades = []
            
            features = self.feature_engineer.create_all_features(
                trade_data=trade_data,
                price_history=price_history,
                recent_trades=recent_trades
            )
            
            # Preprocessar
            features_scaled = self.scaler.transform(features)
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Usar ensemble se disponível
            if self.ensemble_model is not None:
                prediction = self._predict_with_ensemble(features_selected)
                prediction['model_used'] = 'ensemble'
            else:
                prediction = self._predict_with_best_model(features_selected)
            
            # Adicionar confiança baseada em histórico
            prediction['confidence'] = self._adjust_confidence(prediction['confidence'], recent_trades)
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            return self._default_prediction(f"Erro: {str(e)}")
    
    def _predict_with_ensemble(self, features: np.ndarray) -> Dict:
        """Predição com ensemble"""
        try:
            probabilities = self.ensemble_model.predict_proba(features)[0]
            win_probability = probabilities[1]
            
            # Decisão baseada em threshold adaptativo
            threshold = 0.55  # Mais conservador
            
            if win_probability > threshold:
                prediction = "favor"
                reason = f"Ensemble recomenda trade (confiança: {win_probability:.1%})"
            elif win_probability < (1 - threshold):
                prediction = "avoid"
                reason = f"Ensemble sugere evitar (confiança perda: {1-win_probability:.1%})"
            else:
                prediction = "neutral"
                reason = f"Ensemble neutro (probabilidade: {win_probability:.1%})"
            
            return {
                "prediction": prediction,
                "confidence": float(max(win_probability, 1 - win_probability)),
                "win_probability": float(win_probability),
                "reason": reason,
                "ensemble_votes": self._get_ensemble_votes(features)
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no ensemble: {e}")
            return self._predict_with_best_model(features)
    
    def _predict_with_best_model(self, features: np.ndarray) -> Dict:
        """Predição com melhor modelo individual"""
        # Encontrar melhor modelo
        best_model_name = max(self.models.keys(), 
                             key=lambda x: self.models[x]['accuracy'])
        
        model_data = self.models[best_model_name]
        model = model_data['model']
        
        probabilities = model.predict_proba(features)[0]
        win_probability = probabilities[1]
        
        # Threshold baseado na accuracy do modelo
        base_threshold = 0.5
        accuracy_adjustment = (model_data['accuracy'] - 0.5) * 0.2
        threshold = base_threshold + accuracy_adjustment
        
        if win_probability > threshold:
            prediction = "favor"
            reason = f"Modelo {best_model_name} recomenda (acc: {model_data['accuracy']:.1%})"
        elif win_probability < (1 - threshold):
            prediction = "avoid"
            reason = f"Modelo {best_model_name} sugere evitar"
        else:
            prediction = "neutral"
            reason = f"Modelo {best_model_name} neutro"
        
        return {
            "prediction": prediction,
            "confidence": float(max(win_probability, 1 - win_probability)),
            "win_probability": float(win_probability),
            "model_used": best_model_name,
            "model_accuracy": model_data['accuracy'],
            "reason": reason
        }
    
    def _get_ensemble_votes(self, features: np.ndarray) -> Dict:
        """Obtém votos individuais do ensemble"""
        votes = {}
        
        for name, model_data in self.models.items():
            try:
                proba = model_data['model'].predict_proba(features)[0][1]
                votes[name] = {
                    'win_probability': float(proba),
                    'vote': 'favor' if proba > 0.5 else 'avoid',
                    'accuracy': model_data['accuracy']
                }
            except Exception as e:
                logger.warning(f"⚠️ Erro no voto {name}: {e}")
        
        return votes
    
    def _adjust_confidence(self, base_confidence: float, recent_trades: List[Dict]) -> float:
        """Ajusta confiança baseada no histórico recente"""
        if not recent_trades:
            return base_confidence
        
        # Fator baseado na performance recente
        recent_pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
        if recent_pnls:
            recent_win_rate = len([p for p in recent_pnls if p > 0]) / len(recent_pnls)
            
            # Reduzir confiança se performance recente for ruim
            if recent_win_rate < 0.3:
                base_confidence *= 0.8
            elif recent_win_rate > 0.7:
                base_confidence *= 1.1
        
        return min(1.0, base_confidence)
    
    def _default_prediction(self, reason: str) -> Dict:
        """Predição padrão quando não há modelos"""
        return {
            "prediction": "neutral",
            "confidence": 0.5,
            "win_probability": 0.5,
            "model_used": "default",
            "reason": reason
        }
    
    def _get_price_history_for_trade(self, trade_data: Dict) -> List[float]:
        """Obtém histórico de preços para um trade"""
        # Simplificado: retorna preços sintéticos por enquanto
        # Em produção, isso viria de uma API ou banco de dados
        
        base_price = trade_data.get('entry_price', 1000.0)
        
        # Simular 50 pontos de preço
        prices = []
        current_price = base_price
        
        for i in range(50):
            change = np.random.normal(0, 0.002)  # 0.2% volatilidade
            current_price *= (1 + change)
            prices.append(current_price)
        
        return prices
    
    def _get_recent_trades_for_training(self, current_timestamp: str, df: pd.DataFrame) -> List[Dict]:
        """Obtém trades recentes para um momento específico no treinamento"""
        try:
            current_dt = datetime.fromisoformat(current_timestamp)
            
            # Filtrar trades anteriores
            mask = pd.to_datetime(df['timestamp']) < current_dt
            recent_df = df[mask].tail(10)  # Últimos 10 trades
            
            trades = []
            for _, row in recent_df.iterrows():
                pnl = 0
                if row.get('outcome') == 'won':
                    pnl = row.get('stake', 1) * 0.8  # Aproximação
                elif row.get('outcome') == 'lost':
                    pnl = -row.get('stake', 1)
                
                trades.append({
                    'pnl': pnl,
                    'timestamp': row.get('timestamp'),
                    'outcome': row.get('outcome')
                })
            
            return trades
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao obter trades recentes: {e}")
            return []
    
    def _save_models(self):
        """Salva modelos treinados"""
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Salvar modelos individuais
            for name, model_data in self.models.items():
                filepath = models_dir / f"{name}_advanced.joblib"
                joblib.dump(model_data, filepath)
            
            # Salvar ensemble
            if self.ensemble_model is not None:
                ensemble_path = models_dir / "ensemble_model.joblib"
                joblib.dump(self.ensemble_model, ensemble_path)
            
            # Salvar preprocessors
            scaler_path = models_dir / "scaler_advanced.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            selector_path = models_dir / "feature_selector.joblib"
            joblib.dump(self.feature_selector, selector_path)
            
            # Salvar feature engineer
            feature_path = models_dir / "feature_engineer.joblib"
            joblib.dump(self.feature_engineer, feature_path)
            
            logger.info("💾 Modelos avançados salvos")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelos: {e}")
    
    def _load_models(self):
        """Carrega modelos salvos"""
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                return
            
            # Carregar modelos individuais
            for model_file in models_dir.glob("*_advanced.joblib"):
                name = model_file.stem.replace("_advanced", "")
                try:
                    model_data = joblib.load(model_file)
                    self.models[name] = model_data
                    logger.info(f"📂 Modelo {name} carregado")
                except Exception as e:
                    logger.warning(f"⚠️ Erro carregando {name}: {e}")
            
            # Carregar ensemble
            ensemble_path = models_dir / "ensemble_model.joblib"
            if ensemble_path.exists():
                try:
                    self.ensemble_model = joblib.load(ensemble_path)
                    logger.info("📂 Ensemble carregado")
                except Exception as e:
                    logger.warning(f"⚠️ Erro carregando ensemble: {e}")
            
            # Carregar preprocessors
            scaler_path = models_dir / "scaler_advanced.joblib"
            if scaler_path.exists():
                try:
                    self.scaler = joblib.load(scaler_path)
                except Exception as e:
                    logger.warning(f"⚠️ Erro carregando scaler: {e}")
            
            selector_path = models_dir / "feature_selector.joblib"
            if selector_path.exists():
                try:
                    self.feature_selector = joblib.load(selector_path)
                except Exception as e:
                    logger.warning(f"⚠️ Erro carregando feature selector: {e}")
            
            # Carregar feature engineer
            feature_path = models_dir / "feature_engineer.joblib"
            if feature_path.exists():
                try:
                    self.feature_engineer = joblib.load(feature_path)
                except Exception as e:
                    logger.warning(f"⚠️ Erro carregando feature engineer: {e}")
            
            if self.models:
                self.is_trained = True
                logger.info(f"✅ {len(self.models)} modelos avançados carregados")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
    
    def get_model_stats(self) -> Dict:
        """Retorna estatísticas dos modelos"""
        if not self.models:
            return {"error": "Nenhum modelo treinado"}
        
        stats = {
            "models_count": len(self.models),
            "ensemble_available": self.ensemble_model is not None,
            "last_training": self.last_training,
            "individual_models": {}
        }
        
        for name, model_data in self.models.items():
            stats["individual_models"][name] = {
                "accuracy": model_data.get('accuracy', 0),
                "auc_score": model_data.get('auc_score', 0),
                "cv_score": model_data.get('cv_mean', 0),
                "trained_at": model_data.get('trained_at'),
                "samples": model_data.get('samples', 0)
            }
        
        # Melhor modelo
        if self.models:
            best_model = max(self.models.keys(), 
                           key=lambda x: self.models[x].get('accuracy', 0))
            stats["best_model"] = {
                "name": best_model,
                "accuracy": self.models[best_model].get('accuracy', 0)
            }
        
        return stats

# Instância global
advanced_ml_system = AdvancedMLTradingSystem()

if __name__ == "__main__":
    print("🧠 Sistema ML Avançado para Trading")
    print("=" * 50)
    
    # Treinar modelos
    results = advanced_ml_system.train_models()
    print(f"Resultados do treinamento: {results}")
    
    # Teste de predição
    test_trade = {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'R_50',
        'duration': '5t',
        'stake': 2.0,
        'martingale_level': 0,
        'entry_price': 1000.0
    }
    
    prediction = advanced_ml_system.predict(test_trade)
    print(f"Predição de teste: {json.dumps(prediction, indent=2)}")
    
    # Estatísticas
    stats = advanced_ml_system.get_model_stats()
    print(f"Estatísticas: {json.dumps(stats, indent=2)}")
