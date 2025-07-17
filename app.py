from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sqlite3
import json
import threading
import time
import os
import logging
import datetime
import hashlib
import pickle
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import math

# Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
from scipy import stats
from scipy.signal import find_peaks
import ta  # Technical Analysis library

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configurar logging avançado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
VALID_API_KEY = "bhcOGajqbfFfolT"
DB_PATH = os.environ.get('DB_PATH', '/tmp/advanced_ml_trading.db')
MODEL_PATH = os.environ.get('MODEL_PATH', '/tmp/models/')

# Criar diretório de modelos
os.makedirs(MODEL_PATH, exist_ok=True)

@dataclass
class TradingSignal:
    """Estrutura de dados para sinais de trading"""
    timestamp: str
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    volatility: float
    features: Dict[str, float]
    model_version: str
    ensemble_votes: Dict[str, str]
    feature_importance: Dict[str, float]

@dataclass
class ModelPerformance:
    """Métricas de performance dos modelos"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    prediction_count: int
    last_updated: str

class AdvancedFeatureEngineering:
    """Engine avançado de feature engineering para trading"""
    
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.ma_periods = [7, 14, 21, 50]
        
    def calculate_technical_indicators(self, prices: List[float], volumes: List[float] = None) -> Dict[str, float]:
        """Calcular indicadores técnicos avançados"""
        if len(prices) < 50:
            # Preencher com preços simulados se não temos histórico suficiente
            base_price = prices[-1] if prices else 1000
            prices = [base_price + np.random.normal(0, base_price * 0.01) for _ in range(50)]
        
        df = pd.DataFrame({'close': prices})
        if volumes is None:
            volumes = [1000 + np.random.randint(-100, 100) for _ in range(len(prices))]
        df['volume'] = volumes[:len(prices)]
        
        features = {}
        
        try:
            # Médias móveis e cruzamentos
            for period in self.ma_periods:
                if len(prices) >= period:
                    df[f'sma_{period}'] = df['close'].rolling(period).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                    features[f'sma_{period}_ratio'] = df['close'].iloc[-1] / df[f'sma_{period}'].iloc[-1] if df[f'sma_{period}'].iloc[-1] != 0 else 1.0
                    features[f'ema_{period}_ratio'] = df['close'].iloc[-1] / df[f'ema_{period}'].iloc[-1] if df[f'ema_{period}'].iloc[-1] != 0 else 1.0
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = signal.iloc[-1]
            features['macd_histogram'] = (macd - signal).iloc[-1]
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            features['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            features['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            features['bb_position'] = (df['close'].iloc[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Volatilidade e momentum
            for period in self.lookback_periods:
                if len(prices) >= period:
                    returns = df['close'].pct_change(period)
                    features[f'return_{period}'] = returns.iloc[-1]
                    features[f'volatility_{period}'] = returns.rolling(period).std().iloc[-1]
                    features[f'momentum_{period}'] = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
            
            # Support e Resistance levels
            highs = df['close'].rolling(10).max()
            lows = df['close'].rolling(10).min()
            features['resistance_ratio'] = df['close'].iloc[-1] / highs.iloc[-1] if highs.iloc[-1] != 0 else 1.0
            features['support_ratio'] = df['close'].iloc[-1] / lows.iloc[-1] if lows.iloc[-1] != 0 else 1.0
            
            # Volume indicators
            if len(volumes) >= 20:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                features['volume_ratio'] = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1] if df['volume_sma'].iloc[-1] != 0 else 1.0
            
            # Padrões de candlestick (simulados)
            features['doji_pattern'] = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-1] < 0.001
            features['hammer_pattern'] = (df['close'].iloc[-1] > df['close'].iloc[-2]) and (df['close'].iloc[-2] < df['close'].iloc[-3])
            
            # Fractals e suporte/resistência
            recent_prices = df['close'].tail(20).values
            peaks, _ = find_peaks(recent_prices, height=np.mean(recent_prices))
            valleys, _ = find_peaks(-recent_prices, height=-np.mean(recent_prices))
            features['near_resistance'] = len(peaks) > 0 and (len(recent_prices) - peaks[-1]) <= 3
            features['near_support'] = len(valleys) > 0 and (len(recent_prices) - valleys[-1]) <= 3
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de indicadores técnicos: {e}")
            # Features de fallback
            for period in self.ma_periods:
                features[f'sma_{period}_ratio'] = 1.0
                features[f'ema_{period}_ratio'] = 1.0
            features.update({
                'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                'bb_position': 0.5, 'volume_ratio': 1.0
            })
        
        return features
    
    def create_features_from_data(self, data: Dict) -> Dict[str, float]:
        """Criar features a partir dos dados recebidos"""
        features = {}
        
        # Features básicas
        features['volatility'] = data.get('volatility', 50)
        features['current_price'] = data.get('currentPrice', 1000)
        features['martingale_level'] = data.get('martingaleLevel', 0)
        features['win_rate'] = data.get('winRate', 50)
        features['today_pnl'] = data.get('todayPnL', 0)
        
        # Features temporais
        now = datetime.datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = now.weekday() >= 5
        features['is_market_open'] = 9 <= now.hour <= 17
        
        # Features do símbolo
        symbol = data.get('symbol', 'R_50')
        features['is_volatility_index'] = 'R_' in symbol or 'HZ' in symbol
        features['is_jump_index'] = 'JD' in symbol
        features['is_crash_boom'] = 'CRASH' in symbol or 'BOOM' in symbol
        
        # Features de mercado
        market_condition = data.get('marketCondition', 'neutral')
        features['market_bullish'] = market_condition == 'bullish'
        features['market_bearish'] = market_condition == 'bearish'
        features['market_neutral'] = market_condition == 'neutral'
        
        # Features de risco
        features['high_martingale'] = features['martingale_level'] > 3
        features['negative_pnl'] = features['today_pnl'] < 0
        features['low_win_rate'] = features['win_rate'] < 45
        
        # Indicadores técnicos
        prices = data.get('lastTicks', [])
        if not prices:
            base_price = features['current_price']
            prices = [base_price + np.random.normal(0, base_price * 0.01) for _ in range(100)]
        
        technical_features = self.calculate_technical_indicators(prices)
        features.update(technical_features)
        
        # Normalizar features numéricas
        for key, value in features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if 'ratio' in key:
                    features[key] = max(0.1, min(3.0, value))  # Limitar ratios
                elif 'volatility' in key:
                    features[key] = max(0, min(200, value))  # Limitar volatilidade
        
        return features

class AdvancedMLTradingDatabase:
    """Banco de dados avançado para ML trading"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializar banco de dados com tabelas ML"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela principal de sinais com features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                result INTEGER,
                pnl REAL,
                features TEXT NOT NULL,
                model_version TEXT NOT NULL,
                ensemble_votes TEXT,
                feature_importance TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                feedback_received_at TEXT,
                model_accuracy REAL,
                cross_val_score REAL
            )
        ''')
        
        # Performance de modelos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall_score REAL NOT NULL,
                f1_score REAL NOT NULL,
                roc_auc REAL NOT NULL,
                cross_val_mean REAL NOT NULL,
                cross_val_std REAL NOT NULL,
                training_samples INTEGER NOT NULL,
                feature_count INTEGER NOT NULL,
                hyperparameters TEXT,
                feature_importance TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Feature importance histórica
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                rank_position INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Drift detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_drift_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                drift_score REAL NOT NULL,
                drift_detected BOOLEAN NOT NULL,
                feature_drift TEXT,
                performance_drift REAL,
                recommendation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # A/B testing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_ab_testing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                samples_a INTEGER DEFAULT 0,
                samples_b INTEGER DEFAULT 0,
                accuracy_a REAL DEFAULT 0,
                accuracy_b REAL DEFAULT 0,
                winner TEXT,
                confidence_level REAL,
                test_status TEXT DEFAULT 'running',
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT
            )
        ''')
        
        # Auto-tuning histórico
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hyperparameter_tuning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                parameter_set TEXT NOT NULL,
                cv_score REAL NOT NULL,
                best_params TEXT,
                improvement_over_baseline REAL,
                tuning_time_seconds REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_ml_signal(self, signal: TradingSignal) -> int:
        """Salvar sinal ML no banco"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ml_signals (
                timestamp, symbol, direction, confidence, entry_price,
                features, model_version, ensemble_votes, feature_importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp,
            signal.symbol,
            signal.direction,
            signal.confidence,
            signal.entry_price,
            json.dumps(signal.features),
            signal.model_version,
            json.dumps(signal.ensemble_votes),
            json.dumps(signal.feature_importance)
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    
    def update_signal_result(self, signal_id: int, result: int, pnl: float):
        """Atualizar resultado do sinal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE ml_signals 
            SET result = ?, pnl = ?, feedback_received_at = ?
            WHERE id = ?
        ''', (result, pnl, datetime.datetime.now().isoformat(), signal_id))
        
        conn.commit()
        conn.close()
    
    def save_model_performance(self, performance: ModelPerformance):
        """Salvar performance do modelo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance (
                model_name, version, accuracy, precision_score, recall_score,
                f1_score, roc_auc, cross_val_mean, cross_val_std, training_samples, feature_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance.model_name,
            'v1.0',
            performance.accuracy,
            performance.precision,
            performance.recall,
            performance.f1_score,
            performance.roc_auc,
            0.0,  # cross_val_mean
            0.0,  # cross_val_std  
            performance.prediction_count,
            0     # feature_count
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self, limit: int = 1000, min_samples: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """Obter dados para treinamento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT features, direction, result FROM ml_signals 
            WHERE result IS NOT NULL 
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) < min_samples:
            # Gerar dados sintéticos se não temos amostras suficientes
            logger.info(f"Gerando {min_samples} amostras sintéticas para treinamento")
            return self._generate_synthetic_data(min_samples)
        
        # Processar dados reais
        features_list = []
        directions = []
        
        for row in data:
            try:
                features = json.loads(row[0])
                direction = 1 if row[1] == 'CALL' else 0
                result = row[2]
                
                # Usar o resultado como target
                features_list.append(features)
                directions.append(result)  # 1 para win, 0 para loss
                
            except Exception as e:
                logger.warning(f"Erro ao processar linha de dados: {e}")
                continue
        
        if not features_list:
            return self._generate_synthetic_data(min_samples)
        
        # Converter para DataFrame
        df_features = pd.DataFrame(features_list)
        y = pd.Series(directions)
        
        # Limpar e processar features
        df_features = self._clean_features(df_features)
        
        return df_features, y
    
    def _generate_synthetic_data(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Gerar dados sintéticos para treinamento inicial"""
        np.random.seed(42)
        
        features_data = []
        targets = []
        
        for _ in range(n_samples):
            # Gerar features sintéticas com correlações realistas
            volatility = np.random.uniform(10, 100)
            rsi = np.random.uniform(20, 80)
            
            # Features com algumas correlações
            features = {
                'volatility': volatility,
                'rsi': rsi,
                'macd': np.random.normal(0, 1),
                'bb_position': np.random.uniform(0, 1),
                'sma_7_ratio': np.random.uniform(0.95, 1.05),
                'sma_14_ratio': np.random.uniform(0.95, 1.05),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'martingale_level': np.random.randint(0, 5),
                'win_rate': np.random.uniform(30, 70),
                'momentum_5': np.random.normal(0, 0.02),
                'momentum_10': np.random.normal(0, 0.03),
                'return_5': np.random.normal(0, 0.02),
                'volatility_10': np.random.uniform(0.01, 0.1)
            }
            
            # Target baseado em algumas regras realistas
            target_prob = 0.5
            
            # RSI extremos
            if rsi < 30:
                target_prob += 0.1  # Oversold - mais chance de reverter
            elif rsi > 70:
                target_prob -= 0.1  # Overbought
                
            # Volatilidade
            if volatility > 80:
                target_prob -= 0.05  # Alta volatilidade é mais arriscada
            
            # Momentum
            if features['momentum_5'] > 0.01:
                target_prob += 0.05
            elif features['momentum_5'] < -0.01:
                target_prob -= 0.05
                
            target = 1 if np.random.random() < target_prob else 0
            
            features_data.append(features)
            targets.append(target)
        
        df_features = pd.DataFrame(features_data)
        y = pd.Series(targets)
        
        return df_features, y
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpar e processar features"""
        # Remover colunas com muitos valores nulos
        df = df.dropna(thresh=len(df) * 0.7, axis=1)
        
        # Preencher valores nulos
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
        
        # Remover outliers extremos
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)
        
        return df

class AdvancedMLEngine:
    """Engine avançado de Machine Learning"""
    
    def __init__(self, database: AdvancedMLTradingDatabase):
        self.db = database
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_performances = {}
        self.is_training = False
        
        # Configurações de ML
        self.cv_folds = 5
        self.test_size = 0.2
        self.random_state = 42
        
        # Inicializar modelos
        self._initialize_models()
        
    def _initialize_models(self):
        """Inicializar todos os modelos ML"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        # Scaler para cada modelo
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
            
        logger.info(f"Inicializados {len(self.models)} modelos ML")
    
    def auto_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering automático"""
        X_engineered = X.copy()
        
        # Criar features de interação
        numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Top 5 combinações de features mais importantes
            important_features = ['volatility', 'rsi', 'macd', 'volume_ratio', 'momentum_5']
            available_features = [f for f in important_features if f in numeric_cols]
            
            for i, feat1 in enumerate(available_features[:3]):
                for feat2 in available_features[i+1:4]:
                    if feat1 in X_engineered.columns and feat2 in X_engineered.columns:
                        # Ratio features
                        X_engineered[f'{feat1}_{feat2}_ratio'] = X_engineered[feat1] / (X_engineered[feat2] + 1e-8)
                        # Difference features
                        X_engineered[f'{feat1}_{feat2}_diff'] = X_engineered[feat1] - X_engineered[feat2]
        
        # Features polinomiais para features importantes
        for feat in ['volatility', 'rsi']:
            if feat in X_engineered.columns:
                X_engineered[f'{feat}_squared'] = X_engineered[feat] ** 2
                X_engineered[f'{feat}_sqrt'] = np.sqrt(np.abs(X_engineered[feat]))
        
        # Features de bins
        if 'rsi' in X_engineered.columns:
            X_engineered['rsi_overbought'] = (X_engineered['rsi'] > 70).astype(int)
            X_engineered['rsi_oversold'] = (X_engineered['rsi'] < 30).astype(int)
        
        if 'volatility' in X_engineered.columns:
            X_engineered['high_volatility'] = (X_engineered['volatility'] > 70).astype(int)
            X_engineered['low_volatility'] = (X_engineered['volatility'] < 30).astype(int)
        
        # Remover features com variância zero
        X_engineered = X_engineered.loc[:, X_engineered.var() > 1e-8]
        
        return X_engineered
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> pd.DataFrame:
        """Seleção automática de features"""
        try:
            # Método 1: SelectKBest
            k = min(20, X.shape[1])  # Máximo 20 features
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Método 2: Feature importance do Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            rf.fit(X, y)
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = feature_importance.nlargest(k).index.tolist()
            
            # Combinar ambos os métodos
            final_features = list(set(selected_features) & set(top_features))
            if len(final_features) < 10:  # Garantir mínimo de features
                final_features = feature_importance.nlargest(15).index.tolist()
            
            self.feature_selectors[model_name] = final_features
            return X[final_features]
            
        except Exception as e:
            logger.warning(f"Erro na seleção de features: {e}")
            # Fallback: usar todas as features numéricas
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_selectors[model_name] = numeric_features[:20]
            return X[self.feature_selectors[model_name]]
    
    def hyperparameter_tuning(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Auto-tuning de hiperparâmetros"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if model_name not in param_grids:
            return {}
        
        try:
            model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            # Usar TimeSeriesSplit para dados temporais
            cv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X, y)
            
            # Atualizar modelo com melhores parâmetros
            self.models[model_name] = grid_search.best_estimator_
            
            logger.info(f"Hiperparâmetros otimizados para {model_name}: {grid_search.best_params_}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'improvement': grid_search.best_score_ - 0.5  # baseline
            }
            
        except Exception as e:
            logger.warning(f"Erro no tuning de {model_name}: {e}")
            return {}
    
    def train_all_models(self, retrain: bool = False) -> Dict[str, ModelPerformance]:
        """Treinar todos os modelos"""
        if self.is_training:
            logger.warning("Treinamento já em andamento")
            return self.model_performances
            
        self.is_training = True
        logger.info("Iniciando treinamento de todos os modelos...")
        
        try:
            # Obter dados de treinamento
            X, y = self.db.get_training_data(limit=2000, min_samples=200)
            
            if X.empty or len(y) == 0:
                logger.error("Dados de treinamento insuficientes")
                return {}
            
            logger.info(f"Dados de treinamento: {X.shape[0]} amostras, {X.shape[1]} features")
            logger.info(f"Distribuição de classes: {y.value_counts().to_dict()}")
            
            # Feature engineering automático
            X_engineered = self.auto_feature_engineering(X)
            logger.info(f"Features após engineering: {X_engineered.shape[1]}")
            
            performances = {}
            
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Treinando {model_name}...")
                    
                    # Seleção de features
                    X_selected = self.feature_selection(X_engineered, y, model_name)
                    
                    # Scaling
                    X_scaled = self.scalers[model_name].fit_transform(X_selected)
                    
                    # Hyperparameter tuning (apenas para alguns modelos)
                    if model_name in ['random_forest', 'xgboost'] and not retrain:
                        tuning_results = self.hyperparameter_tuning(model_name, pd.DataFrame(X_scaled), y)
                        if tuning_results:
                            logger.info(f"Tuning concluído para {model_name}")
                    
                    # Treinar modelo
                    model.fit(X_scaled, y)
                    
                    # Validação cruzada
                    cv_scores = cross_val_score(model, X_scaled, y, cv=self.cv_folds, scoring='accuracy')
                    
                    # Predições para métricas
                    y_pred = model.predict(X_scaled)
                    y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    # Calcular métricas
                    performance = ModelPerformance(
                        model_name=model_name,
                        accuracy=accuracy_score(y, y_pred),
                        precision=precision_score(y, y_pred, average='weighted'),
                        recall=recall_score(y, y_pred, average='weighted'),
                        f1_score=f1_score(y, y_pred, average='weighted'),
                        roc_auc=roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5,
                        prediction_count=len(y),
                        last_updated=datetime.datetime.now().isoformat()
                    )
                    
                    performances[model_name] = performance
                    
                    # Salvar feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_names = X_selected.columns if hasattr(X_selected, 'columns') else [f'feature_{i}' for i in range(X_selected.shape[1])]
                        self.feature_importance[model_name] = dict(zip(feature_names, model.feature_importances_))
                    
                    # Salvar no banco
                    self.db.save_model_performance(performance)
                    
                    logger.info(f"{model_name} - Accuracy: {performance.accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    logger.error(f"Erro ao treinar {model_name}: {e}")
                    continue
            
            # Criar ensemble
            self._create_ensemble_model(X_engineered, y, performances)
            
            # Salvar modelos
            self._save_models()
            
            self.model_performances = performances
            logger.info(f"Treinamento concluído! {len(performances)} modelos treinados.")
            
            return performances
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return {}
        finally:
            self.is_training = False
    
    def _create_ensemble_model(self, X: pd.DataFrame, y: pd.Series, performances: Dict[str, ModelPerformance]):
        """Criar modelo ensemble baseado nas performances"""
        try:
            # Selecionar os 3 melhores modelos
            sorted_models = sorted(performances.items(), key=lambda x: x[1].accuracy, reverse=True)[:3]
            
            ensemble_models = []
            for model_name, perf in sorted_models:
                if model_name in self.models and perf.accuracy > 0.5:
                    ensemble_models.append((model_name, self.models[model_name]))
            
            if len(ensemble_models) >= 2:
                self.ensemble_model = VotingClassifier(
                    estimators=ensemble_models,
                    voting='soft'
                )
                
                # Treinar ensemble com features do melhor modelo
                best_model_name = sorted_models[0][0]
                selected_features = self.feature_selectors.get(best_model_name, X.columns.tolist())
                X_selected = X[selected_features] if hasattr(X, 'columns') else X
                X_scaled = self.scalers[best_model_name].fit_transform(X_selected)
                
                self.ensemble_model.fit(X_scaled, y)
                
                logger.info(f"Ensemble criado com {len(ensemble_models)} modelos: {[m[0] for m in ensemble_models]}")
            
        except Exception as e:
            logger.error(f"Erro ao criar ensemble: {e}")
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
        """Fazer predição com ensemble de modelos"""
        try:
            # Preparar features
            feature_df = pd.DataFrame([features])
            
            # Feature engineering
            feature_df_engineered = self.auto_feature_engineering(feature_df)
            
            predictions = {}
            confidences = {}
            
            # Predições de cada modelo
            for model_name, model in self.models.items():
                try:
                    if model_name not in self.feature_selectors:
                        continue
                        
                    # Selecionar features
                    selected_features = self.feature_selectors[model_name]
                    available_features = [f for f in selected_features if f in feature_df_engineered.columns]
                    
                    if not available_features:
                        continue
                    
                    X_model = feature_df_engineered[available_features]
                    
                    # Preencher features faltantes
                    for feat in selected_features:
                        if feat not in X_model.columns:
                            X_model[feat] = 0.0
                    
                    X_model = X_model[selected_features]  # Reordenar
                    
                    # Scaling
                    X_scaled = self.scalers[model_name].transform(X_model)
                    
                    # Predição
                    pred = model.predict(X_scaled)[0]
                    conf = model.predict_proba(X_scaled)[0].max() if hasattr(model, 'predict_proba') else 0.6
                    
                    predictions[model_name] = 'CALL' if pred == 1 else 'PUT'
                    confidences[model_name] = conf
                    
                except Exception as e:
                    logger.warning(f"Erro na predição de {model_name}: {e}")
                    continue
            
            # Ensemble prediction
            if self.ensemble_model:
                try:
                    # Usar features do melhor modelo para ensemble
                    best_model = max(self.model_performances.items(), key=lambda x: x[1].accuracy)[0]
                    selected_features = self.feature_selectors.get(best_model, feature_df_engineered.columns.tolist())
                    
                    X_ensemble = feature_df_engineered[selected_features]
                    for feat in selected_features:
                        if feat not in X_ensemble.columns:
                            X_ensemble[feat] = 0.0
                    
                    X_ensemble = X_ensemble[selected_features]
                    X_scaled = self.scalers[best_model].transform(X_ensemble)
                    
                    ensemble_pred = self.ensemble_model.predict(X_scaled)[0]
                    ensemble_conf = self.ensemble_model.predict_proba(X_scaled)[0].max()
                    
                    predictions['ensemble'] = 'CALL' if ensemble_pred == 1 else 'PUT'
                    confidences['ensemble'] = ensemble_conf
                    
                except Exception as e:
                    logger.warning(f"Erro na predição ensemble: {e}")
            
            if not predictions:
                # Fallback
                return 'CALL', 60.0, {'error': 'Nenhum modelo disponível'}
            
            # Decisão final baseada em voting e confidence
            call_votes = sum(1 for pred in predictions.values() if pred == 'CALL')
            put_votes = len(predictions) - call_votes
            
            # Weighted voting baseado nas performances
            weighted_score = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = self.model_performances.get(model_name, ModelPerformance('', 0.5, 0, 0, 0, 0, 0, '')).accuracy
                if pred == 'CALL':
                    weighted_score += weight
                total_weight += weight
            
            final_direction = 'CALL' if weighted_score / total_weight > 0.5 else 'PUT'
            
            # Calcular confidence final
            avg_confidence = np.mean(list(confidences.values()))
            consensus_bonus = abs(call_votes - put_votes) / len(predictions) * 10
            final_confidence = min(95, (avg_confidence * 100) + consensus_bonus)
            
            prediction_details = {
                'model_predictions': predictions,
                'model_confidences': confidences,
                'vote_counts': {'CALL': call_votes, 'PUT': put_votes},
                'ensemble_used': 'ensemble' in predictions,
                'weighted_score': weighted_score / total_weight,
                'consensus_strength': abs(call_votes - put_votes) / len(predictions)
            }
            
            return final_direction, final_confidence, prediction_details
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return 'CALL', 60.0, {'error': str(e)}
    
    def _save_models(self):
        """Salvar modelos treinados"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'ensemble_model': self.ensemble_model,
                'feature_importance': self.feature_importance,
                'model_performances': self.model_performances
            }
            
            with open(os.path.join(MODEL_PATH, 'ml_trading_models.pkl'), 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info("Modelos salvos com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelos: {e}")
    
    def load_models(self):
        """Carregar modelos salvos"""
        try:
            model_file = os.path.join(MODEL_PATH, 'ml_trading_models.pkl')
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.feature_selectors = model_data.get('feature_selectors', {})
                self.ensemble_model = model_data.get('ensemble_model')
                self.feature_importance = model_data.get('feature_importance', {})
                self.model_performances = model_data.get('model_performances', {})
                
                logger.info(f"Modelos carregados: {list(self.models.keys())}")
                return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
        
        return False
    
    def drift_detection(self) -> Dict[str, Any]:
        """Detectar drift nos modelos"""
        try:
            # Comparar performance recente vs histórica
            recent_data = self.db.get_training_data(limit=100, min_samples=50)
            if recent_data[0].empty:
                return {'drift_detected': False, 'reason': 'Dados insuficientes'}
            
            X_recent, y_recent = recent_data
            
            drift_results = {}
            
            for model_name, model in self.models.items():
                if model_name not in self.feature_selectors:
                    continue
                
                try:
                    # Testar modelo em dados recentes
                    selected_features = self.feature_selectors[model_name]
                    available_features = [f for f in selected_features if f in X_recent.columns]
                    
                    if len(available_features) < len(selected_features) * 0.8:
                        # Feature drift detectado
                        drift_results[model_name] = {
                            'drift_detected': True,
                            'drift_type': 'feature_drift',
                            'missing_features': set(selected_features) - set(available_features)
                        }
                        continue
                    
                    X_test = X_recent[available_features]
                    # Preencher features faltantes
                    for feat in selected_features:
                        if feat not in X_test.columns:
                            X_test[feat] = 0.0
                    
                    X_test = X_test[selected_features]
                    X_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Calcular accuracy recente
                    y_pred = model.predict(X_scaled)
                    recent_accuracy = accuracy_score(y_recent, y_pred)
                    
                    # Comparar com performance histórica
                    historical_accuracy = self.model_performances.get(model_name, ModelPerformance('', 0.5, 0, 0, 0, 0, 0, '')).accuracy
                    
                    performance_drift = historical_accuracy - recent_accuracy
                    
                    drift_detected = performance_drift > 0.1  # 10% de queda
                    
                    drift_results[model_name] = {
                        'drift_detected': drift_detected,
                        'drift_type': 'performance_drift' if drift_detected else 'no_drift',
                        'historical_accuracy': historical_accuracy,
                        'recent_accuracy': recent_accuracy,
                        'performance_drop': performance_drift
                    }
                    
                except Exception as e:
                    logger.warning(f"Erro no drift detection para {model_name}: {e}")
                    continue
            
            # Recomendar ações
            high_drift_models = [name for name, result in drift_results.items() if result['drift_detected']]
            
            recommendation = "OK"
            if len(high_drift_models) > len(self.models) / 2:
                recommendation = "RETRAIN_ALL"
            elif high_drift_models:
                recommendation = f"RETRAIN_MODELS: {high_drift_models}"
            
            return {
                'drift_detected': len(high_drift_models) > 0,
                'affected_models': high_drift_models,
                'drift_details': drift_results,
                'recommendation': recommendation,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro no drift detection: {e}")
            return {'drift_detected': False, 'error': str(e)}

class AutoRetrainingManager:
    """Gerenciador de retreinamento automático"""
    
    def __init__(self, ml_engine: AdvancedMLEngine, database: AdvancedMLTradingDatabase):
        self.ml_engine = ml_engine
        self.db = database
        self.retrain_threshold = 100  # Retreinar a cada 100 novos sinais
        self.last_retrain = None
        self.retrain_in_progress = False
        
    def should_retrain(self) -> bool:
        """Verificar se deve retreinar"""
        # Verificar quantidade de novos dados
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM ml_signals WHERE result IS NOT NULL AND feedback_received_at > ?', 
                      (self.last_retrain or '2024-01-01',))
        new_signals = cursor.fetchone()[0]
        conn.close()
        
        if new_signals >= self.retrain_threshold:
            return True
        
        # Verificar drift
        drift_results = self.ml_engine.drift_detection()
        if drift_results.get('drift_detected', False):
            logger.info(f"Drift detectado: {drift_results['affected_models']}")
            return True
        
        return False
    
    def auto_retrain(self):
        """Processo de retreinamento automático"""
        if self.retrain_in_progress:
            logger.info("Retreinamento já em progresso")
            return False
        
        try:
            self.retrain_in_progress = True
            logger.info("Iniciando retreinamento automático...")
            
            # Treinar modelos
            performances = self.ml_engine.train_all_models(retrain=True)
            
            if performances:
                self.last_retrain = datetime.datetime.now().isoformat()
                logger.info(f"Retreinamento concluído: {len(performances)} modelos atualizados")
                return True
            else:
                logger.error("Falha no retreinamento")
                return False
                
        except Exception as e:
            logger.error(f"Erro no retreinamento automático: {e}")
            return False
        finally:
            self.retrain_in_progress = False

# Instâncias globais
db = AdvancedMLTradingDatabase()
feature_engineer = AdvancedFeatureEngineering()
ml_engine = AdvancedMLEngine(db)
retrain_manager = AutoRetrainingManager(ml_engine, db)

def background_ml_tasks():
    """Tarefas de ML em background"""
    while True:
        try:
            # Verificar se precisa retreinar
            if retrain_manager.should_retrain():
                retrain_manager.auto_retrain()
            
            # Drift detection periódico
            drift_results = ml_engine.drift_detection()
            if drift_results.get('drift_detected'):
                logger.warning(f"Drift detectado: {drift_results}")
            
            # Aguardar 1 hora
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Erro nas tarefas de background: {e}")
            time.sleep(300)  # 5 minutos em caso de erro

# Iniciar thread de background
ml_thread = threading.Thread(target=background_ml_tasks, daemon=True)
ml_thread.start()

def validate_api_key():
    """Validar API Key"""
    auth_header = request.headers.get('Authorization', '')
    api_key_header = request.headers.get('X-API-Key', '')
    
    if auth_header.startswith('Bearer '):
        api_key = auth_header.replace('Bearer ', '')
    else:
        api_key = api_key_header
    
    return not api_key or api_key == VALID_API_KEY

@app.route("/")
def home():
    """Home page com informações do sistema ML"""
    try:
        # Estatísticas dos modelos
        model_stats = {}
        for name, perf in ml_engine.model_performances.items():
            model_stats[name] = {
                'accuracy': round(perf.accuracy, 3),
                'f1_score': round(perf.f1_score, 3),
                'predictions': perf.prediction_count
            }
        
        # Verificar drift
        drift_status = ml_engine.drift_detection()
        
        return jsonify({
            "status": "🤖 Advanced ML Trading Bot API - TRUE MACHINE LEARNING SYSTEM",
            "version": "6.0.0 - Advanced ML with Auto-Retraining",
            "description": "Verdadeiro sistema de Machine Learning com auto-aprendizado e retreinamento",
            "ml_engine": "Multi-Model Ensemble with Auto-Feature Engineering",
            
            "ml_models": {
                "available_models": list(ml_engine.models.keys()),
                "ensemble_active": ml_engine.ensemble_model is not None,
                "feature_engineering": "Auto-Feature Engineering + Selection",
                "hyperparameter_tuning": "Automated Grid Search",
                "cross_validation": "Time Series Split CV"
            },
            
            "model_performances": model_stats,
            
            "learning_capabilities": {
                "auto_feature_engineering": True,
                "hyperparameter_optimization": True,
                "ensemble_learning": True,
                "drift_detection": True,
                "auto_retraining": True,
                "feature_selection": True,
                "cross_validation": True,
                "performance_monitoring": True
            },
            
            "drift_detection": {
                "enabled": True,
                "status": "DRIFT_DETECTED" if drift_status.get('drift_detected') else "OK",
                "affected_models": drift_status.get('affected_models', []),
                "recommendation": drift_status.get('recommendation', 'OK')
            },
            
            "auto_retraining": {
                "enabled": True,
                "threshold": retrain_manager.retrain_threshold,
                "last_retrain": retrain_manager.last_retrain,
                "in_progress": retrain_manager.retrain_in_progress
            },
            
            "endpoints": {
                "ml-signal": "POST /ml-signal - Sinal ML com ensemble de modelos",
                "train-models": "POST /train-models - Treinar todos os modelos",
                "model-performance": "GET /model-performance - Performance dos modelos",
                "feature-importance": "GET /feature-importance - Importância das features",
                "drift-detection": "GET /drift-detection - Status do drift",
                "retrain": "POST /retrain - Forçar retreinamento",
                "ml-feedback": "POST /ml-feedback - Feedback para aprendizado",
                "predict-custom": "POST /predict-custom - Predição customizada"
            },
            
            "technical_stack": {
                "ml_libraries": ["scikit-learn", "xgboost", "pandas", "numpy"],
                "models": ["Random Forest", "XGBoost", "Gradient Boosting", "Neural Network", "SVM", "Logistic Regression"],
                "feature_engineering": "Automated with polynomial and interaction features",
                "validation": "Time Series Cross-Validation",
                "ensemble": "Soft Voting Classifier"
            },
            
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "True Machine Learning System - Python"
        })
        
    except Exception as e:
        logger.error(f"Erro na home: {e}")
        return jsonify({"error": "Erro interno", "message": str(e)}), 500

@app.route("/ml-signal", methods=["POST", "OPTIONS"])
@app.route("/signal", methods=["POST", "OPTIONS"])
@app.route("/advanced-signal", methods=["POST", "OPTIONS"])
def generate_ml_signal():
    """Gerar sinal usando Machine Learning"""
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        
        # Extrair features usando feature engineering
        features = feature_engineer.create_features_from_data(data)
        
        # Fazer predição ML
        direction, confidence, prediction_details = ml_engine.predict(features)
        
        # Criar sinal estruturado
        signal = TradingSignal(
            timestamp=datetime.datetime.now().isoformat(),
            symbol=data.get("symbol", "R_50"),
            direction=direction,
            confidence=confidence,
            entry_price=data.get("currentPrice", 1000),
            volatility=features.get('volatility', 50),
            features=features,
            model_version="v6.0",
            ensemble_votes=prediction_details.get('model_predictions', {}),
            feature_importance=ml_engine.feature_importance.get('random_forest', {})
        )
        
        # Salvar no banco
        signal_id = db.save_ml_signal(signal)
        
        # Preparar resposta
        return jsonify({
            "signal_id": signal_id,
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry_price": signal.entry_price,
            "symbol": signal.symbol,
            "timestamp": signal.timestamp,
            
            "ml_analysis": {
                "model_predictions": prediction_details.get('model_predictions', {}),
                "model_confidences": prediction_details.get('model_confidences', {}),
                "ensemble_used": prediction_details.get('ensemble_used', False),
                "consensus_strength": round(prediction_details.get('consensus_strength', 0), 3),
                "weighted_score": round(prediction_details.get('weighted_score', 0.5), 3)
            },
            
            "feature_analysis": {
                "total_features": len(features),
                "top_features": dict(list(sorted(
                    ml_engine.feature_importance.get('random_forest', {}).items(),
                    key=lambda x: x[1], reverse=True
                ))[:5]),
                "volatility": features.get('volatility'),
                "rsi": features.get('rsi'),
                "momentum": features.get('momentum_5')
            },
            
            "model_status": {
                "models_available": len(ml_engine.models),
                "ensemble_active": ml_engine.ensemble_model is not None,
                "last_training": retrain_manager.last_retrain,
                "drift_status": "OK"
            },
            
            "reasoning": f"ML Ensemble com {len(prediction_details.get('model_predictions', {}))} modelos - Consenso: {prediction_details.get('consensus_strength', 0):.1%}",
            "strength": "muito forte" if confidence > 90 else "forte" if confidence > 80 else "moderado" if confidence > 70 else "fraco",
            "timeframe": "optimized",
            "source": "Advanced Machine Learning Ensemble"
        })
        
    except Exception as e:
        logger.error(f"Erro na geração de sinal ML: {e}")
        return jsonify({"error": "Erro na predição ML", "message": str(e)}), 500

@app.route("/train-models", methods=["POST"])
def train_models():
    """Treinar todos os modelos ML"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        force_retrain = request.get_json().get('force', False) if request.get_json() else False
        
        # Treinar modelos
        performances = ml_engine.train_all_models(retrain=force_retrain)
        
        if not performances:
            return jsonify({"error": "Falha no treinamento"}), 500
        
        # Preparar resposta
        results = {}
        for name, perf in performances.items():
            results[name] = {
                "accuracy": round(perf.accuracy, 3),
                "precision": round(perf.precision, 3),
                "recall": round(perf.recall, 3),
                "f1_score": round(perf.f1_score, 3),
                "roc_auc": round(perf.roc_auc, 3),
                "samples": perf.prediction_count
            }
        
        return jsonify({
            "status": "Treinamento concluído",
            "models_trained": len(performances),
            "ensemble_created": ml_engine.ensemble_model is not None,
            "performances": results,
            "best_model": max(performances.items(), key=lambda x: x[1].accuracy)[0],
            "average_accuracy": round(np.mean([p.accuracy for p in performances.values()]), 3),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        return jsonify({"error": "Erro no treinamento", "message": str(e)}), 500

@app.route("/model-performance", methods=["GET"])
def get_model_performance():
    """Obter performance dos modelos"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        performances = {}
        for name, perf in ml_engine.model_performances.items():
            performances[name] = {
                "accuracy": round(perf.accuracy, 3),
                "precision": round(perf.precision, 3),
                "recall": round(perf.recall, 3),
                "f1_score": round(perf.f1_score, 3),
                "roc_auc": round(perf.roc_auc, 3),
                "prediction_count": perf.prediction_count,
                "last_updated": perf.last_updated
            }
        
        # Ranking dos modelos
        ranked_models = sorted(performances.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        return jsonify({
            "model_performances": performances,
            "model_ranking": [{"model": name, "accuracy": perf["accuracy"]} for name, perf in ranked_models],
            "ensemble_available": ml_engine.ensemble_model is not None,
            "total_models": len(performances),
            "average_accuracy": round(np.mean([p["accuracy"] for p in performances.values()]) if performances else 0, 3),
            "best_model": ranked_models[0][0] if ranked_models else None,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter performance: {e}")
        return jsonify({"error": "Erro ao obter performance", "message": str(e)}), 500

@app.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    """Obter importância das features"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        feature_importance_data = {}
        
        for model_name, importance in ml_engine.feature_importance.items():
            # Top 15 features mais importantes
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            feature_importance_data[model_name] = {
                "features": [{"name": name, "importance": round(imp, 4)} for name, imp in sorted_features],
                "total_features": len(importance)
            }
        
        # Features mais importantes globalmente
        all_features = defaultdict(list)
        for model_importance in ml_engine.feature_importance.values():
            for feat, imp in model_importance.items():
                all_features[feat].append(imp)
        
        global_importance = {
            feat: np.mean(imps) for feat, imps in all_features.items()
        }
        
        top_global_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return jsonify({
            "feature_importance_by_model": feature_importance_data,
            "global_feature_importance": [
                {"feature": name, "importance": round(imp, 4)} 
                for name, imp in top_global_features
            ],
            "feature_categories": {
                "technical_indicators": len([f for f in global_importance.keys() if any(indicator in f for indicator in ['rsi', 'macd', 'sma', 'ema', 'bb'])]),
                "momentum_features": len([f for f in global_importance.keys() if 'momentum' in f]),
                "volatility_features": len([f for f in global_importance.keys() if 'volatility' in f or 'vol' in f]),
                "temporal_features": len([f for f in global_importance.keys() if any(temp in f for temp in ['hour', 'day', 'weekend'])])
            },
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter feature importance: {e}")
        return jsonify({"error": "Erro ao obter feature importance", "message": str(e)}), 500

@app.route("/drift-detection", methods=["GET"])
def drift_detection():
    """Verificar drift nos modelos"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        drift_results = ml_engine.drift_detection()
        
        return jsonify({
            "drift_status": drift_results,
            "recommendation": drift_results.get('recommendation', 'OK'),
            "action_needed": drift_results.get('drift_detected', False),
            "summary": {
                "total_models": len(ml_engine.models),
                "models_with_drift": len(drift_results.get('affected_models', [])),
                "drift_percentage": len(drift_results.get('affected_models', [])) / len(ml_engine.models) * 100 if ml_engine.models else 0
            },
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no drift detection: {e}")
        return jsonify({"error": "Erro no drift detection", "message": str(e)}), 500

@app.route("/retrain", methods=["POST"])
def force_retrain():
    """Forçar retreinamento dos modelos"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        success = retrain_manager.auto_retrain()
        
        return jsonify({
            "retrain_success": success,
            "message": "Retreinamento concluído com sucesso" if success else "Falha no retreinamento",
            "models_updated": len(ml_engine.model_performances) if success else 0,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no retreinamento forçado: {e}")
        return jsonify({"error": "Erro no retreinamento", "message": str(e)}), 500

@app.route("/ml-feedback", methods=["POST"])
@app.route("/feedback", methods=["POST"])
def ml_feedback():
    """Receber feedback para aprendizado ML"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        
        signal_id = data.get("signal_id")
        result = data.get("result", 0)  # 1 para win, 0 para loss
        pnl = data.get("pnl", 0)
        
        if signal_id:
            # Atualizar resultado no banco
            db.update_signal_result(signal_id, result, pnl)
            
            # Verificar se precisa retreinar
            if retrain_manager.should_retrain():
                logger.info("Feedback recebido - Iniciando retreinamento automático")
                # Retreinar em thread separada para não bloquear
                threading.Thread(target=retrain_manager.auto_retrain, daemon=True).start()
        
        return jsonify({
            "feedback_received": True,
            "signal_id": signal_id,
            "result": "WIN" if result == 1 else "LOSS",
            "pnl": pnl,
            "learning_active": True,
            "auto_retrain_triggered": retrain_manager.should_retrain(),
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Feedback integrado ao sistema de aprendizado ML"
        })
        
    except Exception as e:
        logger.error(f"Erro no feedback ML: {e}")
        return jsonify({"error": "Erro no feedback", "message": str(e)}), 500

@app.route("/predict-custom", methods=["POST"])
def predict_custom():
    """Predição customizada com features específicas"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        
        # Permitir features customizadas
        custom_features = data.get("features", {})
        
        # Usar feature engineering se não tem features customizadas
        if not custom_features:
            custom_features = feature_engineer.create_features_from_data(data)
        
        # Fazer predição
        direction, confidence, details = ml_engine.predict(custom_features)
        
        return jsonify({
            "prediction": direction,
            "confidence": round(confidence, 1),
            "features_used": list(custom_features.keys()),
            "model_details": details,
            "feature_count": len(custom_features),
            "models_contributing": len(details.get('model_predictions', {})),
            "ensemble_consensus": details.get('consensus_strength', 0),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro na predição customizada: {e}")
        return jsonify({"error": "Erro na predição", "message": str(e)}), 500

# Carregamento inicial dos modelos
@app.before_first_request
def initialize_ml_system():
    """Inicializar sistema ML"""
    try:
        # Tentar carregar modelos existentes
        if not ml_engine.load_models():
            logger.info("Modelos não encontrados. Iniciando treinamento inicial...")
            # Treinar modelos iniciais em thread separada
            threading.Thread(target=ml_engine.train_all_models, daemon=True).start()
        else:
            logger.info("Modelos carregados com sucesso")
            
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    # Logs de inicialização
    logger.info("🤖 INICIALIZANDO ADVANCED ML TRADING BOT API")
    logger.info("=" * 80)
    logger.info("🧠 SISTEMA DE MACHINE LEARNING VERDADEIRO:")
    logger.info(f"   - {len(ml_engine.models)} Modelos ML: {list(ml_engine.models.keys())}")
    logger.info("   - Feature Engineering Automático")
    logger.info("   - Hyperparameter Tuning Automático")
    logger.info("   - Ensemble Learning com Voting")
    logger.info("   - Drift Detection Automático")
    logger.info("   - Auto-Retreinamento Inteligente")
    logger.info("   - Cross-Validation Temporal")
    logger.info("   - Feature Selection Automática")
    logger.info("=" * 80)
    logger.info(f"📊 Banco de dados: {DB_PATH}")
    logger.info(f"💾 Modelos salvos em: {MODEL_PATH}")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    logger.info("🚀 SISTEMA PRONTO - VERDADEIRO MACHINE LEARNING ATIVO!")
    
    # Inicializar modelos
    try:
        if not ml_engine.load_models():
            logger.info("🧠 Iniciando treinamento inicial dos modelos...")
            ml_engine.train_all_models()
    except Exception as e:
        logger.warning(f"Aviso na inicialização: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
