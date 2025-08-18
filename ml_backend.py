from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import asyncio
import os
import time
import gc
import psutil
from contextlib import asynccontextmanager
from pathlib import Path

# Importar módulos locais (assumindo que estão no mesmo projeto)
try:
    from config import config, get_config
    from monitoring import monitor, log_ml_training_result, log_trading_session, MLMetrics, TradingMetrics
except ImportError:
    # Fallback se os módulos não estiverem disponíveis
    config = None
    monitor = None

# Configuração de logging avançado
def setup_advanced_logging():
    """Configura sistema de logging avançado"""
    if config and config.logging:
        log_config = config.logging
        
        # Criar diretório de logs
        log_dir = Path(log_config.log_file_path).parent
        log_dir.mkdir(exist_ok=True)
        
        # Configurar formatação
        formatter = logging.Formatter(log_config.format)
        
        # Logger principal
        logger = logging.getLogger("MLTrading")
        logger.setLevel(getattr(logging, log_config.level))
        
        # Handler para arquivo (se habilitado)
        if log_config.log_to_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.log_file_path,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_config.level))
            logger.addHandler(file_handler)
        
        # Handler para console (se habilitado)
        if log_config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, log_config.console_level))
            logger.addHandler(console_handler)
        
        # Configurar níveis para módulos específicos
        for module, level in log_config.module_levels.items():
            logging.getLogger(module).setLevel(getattr(logging, level))
            
        return logger
    else:
        # Configuração básica se config não estiver disponível
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("MLTrading")

# Sistema de autenticação
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verifica API key se a segurança estiver habilitada"""
    if config and config.security.api_key_required:
        if not credentials or credentials.credentials != config.security.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key inválida",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials

# Sistema de monitoramento de recursos
class ResourceMonitor:
    """Monitor de recursos do sistema"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.memory_warnings = 0
        
    def check_memory_usage(self):
        """Verifica uso de memória"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        max_memory = config.performance.max_memory_usage_mb if config else 512
        
        if memory_mb > max_memory * 0.9:  # 90% do limite
            self.memory_warnings += 1
            if self.memory_warnings % 10 == 0:  # Log a cada 10 avisos
                logger.warning(f"Alto uso de memória: {memory_mb:.1f}MB (limite: {max_memory}MB)")
                gc.collect()  # Forçar garbage collection
                
        return memory_mb
    
    def increment_request(self):
        """Incrementa contador de requests"""
        self.request_count += 1
        
    def get_stats(self):
        """Retorna estatísticas do sistema"""
        uptime = time.time() - self.start_time
        memory_mb = self.check_memory_usage()
        
        return {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "total_requests": self.request_count,
            "requests_per_hour": (self.request_count / uptime) * 3600 if uptime > 0 else 0,
            "memory_usage_mb": memory_mb,
            "memory_warnings": self.memory_warnings,
            "cpu_percent": psutil.cpu_percent(),
            "disk_usage_percent": psutil.disk_usage('.').percent
        }

resource_monitor = ResourceMonitor()

# Modelos Pydantic
class TradeData(BaseModel):
    id: str
    timestamp: str
    symbol: str
    direction: str
    stake: float
    duration: str
    entry_price: float
    exit_price: Optional[float] = None
    outcome: Optional[str] = None
    market_context: Dict[str, Any]
    martingale_level: int
    volatility: float
    trend: str

class MarketAnalysisRequest(BaseModel):
    symbol: str
    current_price: float
    timestamp: str
    trades: List[Dict]
    balance: float
    win_rate: float
    volatility: float
    market_condition: str
    martingale_level: int
    is_after_loss: bool
    ml_patterns: Optional[int] = 0
    ml_accuracy: Optional[float] = 0.0

class TradingSignalRequest(BaseModel):
    symbol: str
    current_price: float
    account_balance: float
    win_rate: float
    recent_trades: List[Dict]
    timestamp: str
    volatility: float
    market_condition: str
    martingale_level: int
    is_after_loss: bool
    ml_data: Optional[Dict] = None

class RiskAssessmentRequest(BaseModel):
    current_balance: float
    today_pnl: float
    martingale_level: int
    recent_trades: List[Dict]
    win_rate: float
    total_trades: int
    timestamp: str
    is_in_cooling_period: bool
    needs_analysis_after_loss: bool
    ml_risk: Optional[Dict] = None

# Classe principal do sistema ML
class MLTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.db_path = "trading_data.db"
        self.min_trades_for_training = 50
        self.pattern_confidence_threshold = 0.7
        self.initialize_database()
        self.load_models()
        
    def initialize_database(self):
        """Inicializa o banco de dados SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                stake REAL,
                duration TEXT,
                entry_price REAL,
                exit_price REAL,
                outcome TEXT,
                market_context TEXT,
                martingale_level INTEGER,
                volatility REAL,
                trend TEXT,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de padrões ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                occurrences INTEGER,
                success_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de métricas de performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                accuracy REAL,
                precision_call REAL,
                precision_put REAL,
                recall_call REAL,
                recall_put REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                training_data_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Base de dados inicializada")
        
    def save_trade_data(self, trade_data: TradeData):
        """Salva dados de trade no banco"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        features = self.extract_features(trade_data)
        
        cursor.execute('''
            INSERT OR REPLACE INTO trades 
            (id, timestamp, symbol, direction, stake, duration, entry_price, exit_price, 
             outcome, market_context, martingale_level, volatility, trend, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.id, trade_data.timestamp, trade_data.symbol, trade_data.direction,
            trade_data.stake, trade_data.duration, trade_data.entry_price, trade_data.exit_price,
            trade_data.outcome, json.dumps(trade_data.market_context), trade_data.martingale_level,
            trade_data.volatility, trade_data.trend, json.dumps(features)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Trade {trade_data.id} salvo no banco")
        
    def extract_features(self, trade_data: TradeData) -> Dict:
        """Extrai features para ML a partir dos dados de trade"""
        features = {
            'hour_of_day': datetime.fromisoformat(trade_data.timestamp.replace('Z', '+00:00')).hour,
            'volatility': trade_data.volatility,
            'martingale_level': trade_data.martingale_level,
            'stake_normalized': min(trade_data.stake / 10.0, 1.0),  # Normalizar stake
            'symbol_encoded': hash(trade_data.symbol) % 100,  # Encoding simples do símbolo
            'direction_encoded': 1 if trade_data.direction == 'CALL' else 0,
            'trend_encoded': {'bullish': 1, 'bearish': -1, 'neutral': 0}.get(trade_data.trend, 0),
            'duration_minutes': self.parse_duration_to_minutes(trade_data.duration),
            'entry_price_normalized': trade_data.entry_price / 1000.0,  # Normalizar preço
        }
        
        # Features do contexto de mercado
        if 'recent_results' in trade_data.market_context:
            recent_results = trade_data.market_context['recent_results']
            features['recent_wins'] = recent_results.count('won')
            features['recent_losses'] = recent_results.count('lost')
            features['recent_win_rate'] = features['recent_wins'] / max(len(recent_results), 1)
        
        return features
        
    def parse_duration_to_minutes(self, duration: str) -> float:
        """Converte duration para minutos"""
        if 't' in duration:
            return float(duration.replace('t', '')) * 0.1  # Ticks para minutos aproximado
        elif 'm' in duration:
            return float(duration.replace('m', ''))
        return 1.0
        
    def get_training_data(self) -> pd.DataFrame:
        """Obtém dados de treino do banco"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM trades 
            WHERE outcome IS NOT NULL 
            AND outcome IN ('won', 'lost')
            ORDER BY timestamp DESC
            LIMIT 10000
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return pd.DataFrame()
            
        # Processar features
        features_list = []
        labels = []
        
        for _, row in df.iterrows():
            try:
                features = json.loads(row['features'])
                features_list.append(features)
                labels.append(1 if row['outcome'] == 'won' else 0)
            except (json.JSONDecodeError, KeyError):
                continue
                
        if not features_list:
            return pd.DataFrame()
            
        features_df = pd.DataFrame(features_list)
        features_df['label'] = labels
        
        return features_df
        
    # Classe principal do sistema ML - MELHORADA
class AdvancedMLTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Configurações
        if config:
            self.db_path = config.database.path
            self.min_trades_for_training = config.ml.min_trades_for_training
            self.pattern_confidence_threshold = config.ml.pattern_confidence_threshold
            self.models_to_train = config.ml.models_to_train
            self.model_save_path = config.ml.model_save_path
        else:
            # Valores padrão
            self.db_path = "trading_data.db"
            self.min_trades_for_training = 50
            self.pattern_confidence_threshold = 0.7
            self.models_to_train = ["random_forest", "gradient_boosting", "logistic_regression"]
            self.model_save_path = "models"
            
        # Estatísticas
        self.training_stats = {}
        self.prediction_stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "last_accuracy": 0.0
        }
        
        # Cache para melhor performance
        self.feature_cache = {}
        self.pattern_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        self.initialize_database()
        self.load_models()
        
        logger.info("Sistema ML Avançado inicializado")
        
    def get_model_configs(self):
        """Retorna configurações dos modelos"""
        return {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'requires_scaling': False
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                },
                'requires_scaling': False
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000
                },
                'requires_scaling': True
            },
            'svm': {
                'class': SVC,
                'params': {
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': 42
                },
                'requires_scaling': True
            },
            'neural_network': {
                'class': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 500,
                    'random_state': 42
                },
                'requires_scaling': True
            }
        }
    
    def train_models_advanced(self):
        """Treinamento avançado de modelos com validação cruzada"""
        df = self.get_training_data()
        
        if len(df) < self.min_trades_for_training:
            logger.warning(f"Dados insuficientes para treino: {len(df)} < {self.min_trades_for_training}")
            return False
            
        # Preparar dados
        X = df.drop(['label'], axis=1)
        y = df['label']
        
        # Garantir que todas as colunas são numéricas
        X = X.select_dtypes(include=[np.number])
        self.feature_columns = X.columns.tolist()
        
        if len(X.columns) == 0:
            logger.error("Nenhuma feature numérica encontrada")
            return False
            
        # Split dos dados
        test_size = config.ml.test_size if config else 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if config and config.ml.stratify else None
        )
        
        # Configurações dos modelos
        model_configs = self.get_model_configs()
        
        # Treinar apenas modelos selecionados
        models_to_train = [m for m in self.models_to_train if m in model_configs]
        
        best_accuracy = 0
        best_model_name = None
        training_results = {}
        
        for model_name in models_to_train:
            try:
                logger.info(f"Treinando modelo: {model_name}")
                
                model_config = model_configs[model_name]
                model = model_config['class'](**model_config['params'])
                
                # Aplicar scaling se necessário
                if model_config['requires_scaling']:
                    scaler = StandardScaler()
                    X_train_processed = scaler.fit_transform(X_train)
                    X_test_processed = scaler.transform(X_test)
                    self.scalers[model_name] = scaler
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test
                
                # Treinar modelo
                start_time = time.time()
                model.fit(X_train_processed, y_train)
                training_time = time.time() - start_time
                
                # Predições
                y_pred = model.predict(X_test_processed)
                y_pred_proba = model.predict_proba(X_test_processed)
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Validação cruzada
                cv_folds = config.ml.cross_validation_folds if config else 5
                cv_scores = cross_val_score(model, X_train_processed, y_train, cv=cv_folds)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Feature importance (se disponível)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                    feature_importance = dict(zip(self.feature_columns, importance_scores))
                elif hasattr(model, 'coef_'):
                    importance_scores = np.abs(model.coef_[0])
                    feature_importance = dict(zip(self.feature_columns, importance_scores))
                
                # Salvar resultados
                training_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'confusion_matrix': conf_matrix.tolist(),
                    'feature_importance': feature_importance,
                    'total_predictions': len(y_test),
                    'correct_predictions': int(np.sum(y_pred == y_test)),
                    'training_size': len(X_train)
                }
                
                # Salvar modelo
                self.models[model_name] = model
                
                logger.info(f"Modelo {model_name}: Accuracy = {accuracy:.3f}, CV = {cv_mean:.3f}±{cv_std:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
                    
                # Log para monitoramento
                if monitor:
                    log_ml_training_result(model_name, accuracy, training_results[model_name])
                
            except Exception as e:
                logger.error(f"Erro ao treinar modelo {model_name}: {e}")
                continue
        
        # Salvar modelos e estatísticas
        if self.models:
            self.save_models()
            self.training_stats = training_results
            
            # Salvar métricas consolidadas
            self.save_training_metrics(best_model_name, best_accuracy, len(X_train))
            
            logger.info(f"Treinamento concluído. Melhor modelo: {best_model_name} (Accuracy: {best_accuracy:.3f})")
            return True
        else:
            logger.error("Nenhum modelo foi treinado com sucesso")
            return False
        
    def save_models(self):
        """Salva modelos treinados"""
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}.pkl')
            
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'models/scaler_{name}.pkl')
            
        # Salvar feature columns
        with open('models/feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f)
            
        logger.info("Modelos salvos")
        
    def load_models(self):
        """Carrega modelos salvos"""
        try:
            model_files = ['random_forest.pkl', 'gradient_boosting.pkl', 'logistic_regression.pkl']
            
            for model_file in model_files:
                model_path = f'models/{model_file}'
                if os.path.exists(model_path):
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
                    
            # Carregar scaler
            scaler_path = 'models/scaler_standard.pkl'
            if os.path.exists(scaler_path):
                self.scalers['standard'] = joblib.load(scaler_path)
                
            # Carregar feature columns
            features_path = 'models/feature_columns.json'
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
                    
            if self.models:
                logger.info(f"Modelos carregados: {list(self.models.keys())}")
            else:
                logger.info("Nenhum modelo salvo encontrado")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
            
    def predict_trade_outcome(self, features: Dict) -> Dict:
        """Faz predição do resultado do trade"""
        if not self.models or not self.feature_columns:
            return {
                'prediction': 'neutral',
                'confidence': 0.5,
                'model_used': 'none',
                'reason': 'Modelos não treinados'
            }
            
        try:
            # Preparar features
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
                
            X = np.array(feature_vector).reshape(1, -1)
            
            # Usar ensemble de modelos
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                if name == 'logistic_regression' and 'standard' in self.scalers:
                    X_scaled = self.scalers['standard'].transform(X)
                    pred_proba = model.predict_proba(X_scaled)[0]
                else:
                    pred_proba = model.predict_proba(X)[0]
                    
                predictions.append(pred_proba[1])  # Probabilidade de WIN
                confidences.append(max(pred_proba))
                
            # Média das predições
            avg_win_prob = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            # Decisão final
            if avg_win_prob > 0.6:
                prediction = 'favor'
            elif avg_win_prob < 0.4:
                prediction = 'avoid'
            else:
                prediction = 'neutral'
                
            return {
                'prediction': prediction,
                'confidence': avg_confidence,
                'win_probability': avg_win_prob,
                'model_used': 'ensemble',
                'models_count': len(self.models),
                'reason': f'Ensemble de {len(self.models)} modelos'
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'prediction': 'neutral',
                'confidence': 0.5,
                'model_used': 'error',
                'reason': str(e)
            }
            
    def analyze_patterns(self) -> Dict:
        """Analisa padrões nos dados"""
        df = self.get_training_data()
        
        if len(df) < 10:
            return {'patterns': [], 'total_trades': len(df)}
            
        patterns = []
        
        # Padrão 1: Performance por símbolo
        conn = sqlite3.connect(self.db_path)
        symbol_performance = pd.read_sql_query('''
            SELECT symbol, 
                   COUNT(*) as total_trades,
                   SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) as wins,
                   AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate
            FROM trades 
            WHERE outcome IS NOT NULL
            GROUP BY symbol
            HAVING COUNT(*) >= 5
            ORDER BY win_rate DESC
        ''', conn)
        
        for _, row in symbol_performance.iterrows():
            if row['win_rate'] > 0.7:
                patterns.append({
                    'type': 'symbol_success',
                    'description': f"Alto sucesso em {row['symbol']}",
                    'confidence': row['win_rate'],
                    'data': {'symbol': row['symbol'], 'trades': int(row['total_trades'])}
                })
            elif row['win_rate'] < 0.3:
                patterns.append({
                    'type': 'symbol_avoid',
                    'description': f"Baixo sucesso em {row['symbol']}",
                    'confidence': 1 - row['win_rate'],
                    'data': {'symbol': row['symbol'], 'trades': int(row['total_trades'])}
                })
                
        # Padrão 2: Performance por horário
        hourly_performance = pd.read_sql_query('''
            SELECT 
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                COUNT(*) as total_trades,
                AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate
            FROM trades 
            WHERE outcome IS NOT NULL
            GROUP BY hour
            HAVING COUNT(*) >= 5
            ORDER BY win_rate DESC
        ''', conn)
        
        for _, row in hourly_performance.iterrows():
            if row['win_rate'] > 0.7:
                patterns.append({
                    'type': 'time_success',
                    'description': f"Alto sucesso às {row['hour']:02d}h",
                    'confidence': row['win_rate'],
                    'data': {'hour': int(row['hour']), 'trades': int(row['total_trades'])}
                })
                
        # Padrão 3: Martingale performance
        martingale_performance = pd.read_sql_query('''
            SELECT 
                martingale_level,
                COUNT(*) as total_trades,
                AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate
            FROM trades 
            WHERE outcome IS NOT NULL
            GROUP BY martingale_level
            HAVING COUNT(*) >= 3
            ORDER BY martingale_level
        ''', conn)
        
        for _, row in martingale_performance.iterrows():
            if row['martingale_level'] > 0 and row['win_rate'] < 0.4:
                patterns.append({
                    'type': 'martingale_risk',
                    'description': f"Baixo sucesso no Martingale nível {row['martingale_level']}",
                    'confidence': 1 - row['win_rate'],
                    'data': {'level': int(row['martingale_level']), 'trades': int(row['total_trades'])}
                })
                
        conn.close()
        
        return {
            'patterns': patterns,
            'total_trades': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    def save_training_metrics(self, best_model: str, accuracy: float, training_size: int):
        """Salva métricas de treino"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ml_metrics 
            (model_type, accuracy, total_predictions, correct_predictions, training_data_size)
            VALUES (?, ?, ?, ?, ?)
        ''', (best_model, accuracy, training_size, int(accuracy * training_size), training_size))
        
        conn.commit()
        conn.close()
        
    def get_ml_stats(self) -> Dict:
        """Retorna estatísticas do ML"""
        conn = sqlite3.connect(self.db_path)
        
        # Estatísticas básicas
        basic_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) as total_wins,
                AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as overall_win_rate
            FROM trades 
            WHERE outcome IS NOT NULL
        ''', conn)
        
        # Última métrica de treino
        last_training = pd.read_sql_query('''
            SELECT * FROM ml_metrics 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', conn)
        
        # Padrões ativos
        patterns_count = pd.read_sql_query('''
            SELECT COUNT(*) as patterns_count FROM ml_patterns
        ''', conn)
        
        conn.close()
        
        stats = {
            'total_trades': int(basic_stats.iloc[0]['total_trades']) if len(basic_stats) > 0 else 0,
            'total_wins': int(basic_stats.iloc[0]['total_wins']) if len(basic_stats) > 0 else 0,
            'overall_win_rate': float(basic_stats.iloc[0]['overall_win_rate']) if len(basic_stats) > 0 else 0.0,
            'models_loaded': len(self.models),
            'patterns_count': int(patterns_count.iloc[0]['patterns_count']) if len(patterns_count) > 0 else 0,
            'last_training': None
        }
        
        if len(last_training) > 0:
            stats['last_training'] = {
                'model_type': last_training.iloc[0]['model_type'],
                'accuracy': float(last_training.iloc[0]['accuracy']),
                'training_size': int(last_training.iloc[0]['training_data_size']),
                'date': last_training.iloc[0]['created_at']
            }
            
            def extract_features_from_request(self, request: Dict) -> Dict:
        """Extrai features de uma requisição de predição"""
        features = {
            'hour_of_day': datetime.now().hour,
            'volatility': request.get('volatility', 50),
            'martingale_level': request.get('martingale_level', 0),
            'stake_normalized': min(request.get('stake', 1) / 10.0, 1.0),
            'symbol_encoded': hash(request.get('symbol', 'R_50')) % 100,
            'direction_encoded': 1 if request.get('direction') == 'CALL' else 0,
            'trend_encoded': {'bullish': 1, 'bearish': -1, 'neutral': 0}.get(request.get('trend', 'neutral'), 0),
            'duration_minutes': self.parse_duration_to_minutes(request.get('duration', '5t')),
            'entry_price_normalized': request.get('current_price', 1000) / 1000.0,
            'recent_wins': request.get('recent_wins', 0),
            'recent_losses': request.get('recent_losses', 0),
            'recent_win_rate': request.get('recent_win_rate', 0.5)
        }
        return features
    
    def get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """Obtém predição do cache"""
        if cache_key in self.feature_cache:
            cached_data = self.feature_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                cached_data['prediction']['cache_hit'] = True
                return cached_data['prediction']
            else:
                del self.feature_cache[cache_key]
        return None
    
    def cache_prediction(self, cache_key: str, prediction: Dict):
        """Armazena predição no cache"""
        self.feature_cache[cache_key] = {
            'timestamp': time.time(),
            'prediction': prediction.copy()
        }
        
        # Limpar cache antigo
        if len(self.feature_cache) > 1000:
            oldest_key = min(self.feature_cache.keys(), 
                           key=lambda k: self.feature_cache[k]['timestamp'])
            del self.feature_cache[oldest_key]
    
    def update_prediction_stats(self, trade_data: TradeData):
        """Atualiza estatísticas de predição"""
        if hasattr(trade_data, 'ml_prediction') and trade_data.ml_prediction:
            ml_pred = trade_data.ml_prediction
            if trade_data.outcome:
                # Verificar se predição estava correta
                prediction_correct = (
                    (ml_pred.get('prediction') == 'favor' and trade_data.outcome == 'won') or
                    (ml_pred.get('prediction') == 'avoid' and trade_data.outcome == 'lost')
                )
                
                if prediction_correct:
                    self.prediction_stats["correct_predictions"] += 1
                
                # Atualizar accuracy
                total = self.prediction_stats["total_predictions"]
                correct = self.prediction_stats["correct_predictions"]
                self.prediction_stats["last_accuracy"] = correct / max(total, 1)
    
    def clear_pattern_cache(self):
        """Limpa cache de padrões"""
        self.pattern_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        return {
            "feature_cache_size": len(self.feature_cache),
            "pattern_cache_size": len(self.pattern_cache),
            "cache_ttl_seconds": self.cache_ttl,
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache"""
        # Implementação simplificada
        if hasattr(self, '_cache_hits') and hasattr(self, '_cache_requests'):
            return self._cache_hits / max(self._cache_requests, 1)
        return 0.0
    
    def generate_training_report(self) -> Dict:
        """Gera relatório de treinamento"""
        return {
            "timestamp": datetime.now().isoformat(),
            "models_trained": len(self.models),
            "training_stats": self.training_stats,
            "feature_count": len(self.feature_columns),
            "data_quality_score": self._calculate_data_quality(),
            "recommendations": self._generate_training_recommendations()
        }
    
    def _calculate_data_quality(self) -> float:
        """Calcula score de qualidade dos dados"""
        df = self.get_training_data()
        if len(df) == 0:
            return 0.0
        
        # Fatores de qualidade
        size_score = min(len(df) / 1000, 1.0)  # Até 1000 trades = 100%
        balance_score = self._calculate_class_balance(df)
        completeness_score = self._calculate_completeness(df)
        
        return (size_score + balance_score + completeness_score) / 3
    
    def _calculate_class_balance(self, df: pd.DataFrame) -> float:
        """Calcula balanceamento das classes"""
        if 'label' not in df.columns:
            return 0.0
        
        class_counts = df['label'].value_counts()
        if len(class_counts) != 2:
            return 0.0
        
        minority_ratio = class_counts.min() / class_counts.max()
        return minority_ratio  # 1.0 = perfeitamente balanceado
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calcula completude dos dados"""
        total_cells = df.size
        non_null_cells = df.count().sum()
        return non_null_cells / max(total_cells, 1)
    
    def _generate_training_recommendations(self) -> List[str]:
        """Gera recomendações baseadas no treinamento"""
        recommendations = []
        
        df = self.get_training_data()
        
        if len(df) < 200:
            recommendations.append("Colete mais dados (recomendado: 200+ trades)")
        
        data_quality = self._calculate_data_quality()
        if data_quality < 0.7:
            recommendations.append("Melhore a qualidade dos dados")
        
        if len(self.models) < 3:
            recommendations.append("Considere treinar mais tipos de modelos")
        
        best_accuracy = max([stats.get('accuracy', 0) for stats in self.training_stats.values()] or [0])
        if best_accuracy < 0.6:
            recommendations.append("Accuracy baixa - revisar features ou estratégia")
        
        return recommendations

    def backup_models(self) -> bool:
        """Cria backup dos modelos"""
        try:
            backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiar modelos
            models_dir = Path(self.model_save_path)
            if models_dir.exists():
                import shutil
                shutil.copytree(models_dir, backup_dir / "models")
            
            # Backup do banco
            import shutil
            shutil.copy2(self.db_path, backup_dir / "trading_data.db")
            
            logger.info(f"Backup criado: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Erro no backup: {e}")
            return False

# Setup do logger
logger = setup_advanced_logging()

# Instância global do sistema ML
ml_system = AdvancedMLTradingSystem()

# Middleware para monitoramento de requests
@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()
    resource_monitor.increment_request()
    resource_monitor.check_memory_usage()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# Endpoints da API - MELHORADOS
@app.get("/")
async def root():
    """Endpoint raiz com informações do sistema"""
    system_stats = resource_monitor.get_stats()
    
    return {
        "message": "🧠 ML Trading API está funcionando!",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(ml_system.models),
        "environment": config.get_environment_summary() if config else "basic",
        "system_stats": system_stats,
        "features": {
            "advanced_ml": True,
            "monitoring": monitor is not None,
            "security": config.security.api_key_required if config else False,
            "cache": config.performance.enable_cache if config else False
        }
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde avançada"""
    stats = ml_system.get_ml_stats()
    system_stats = resource_monitor.get_stats()
    
    # Verificar saúde do sistema
    health_status = "healthy"
    issues = []
    
    # Verificar memória
    if system_stats["memory_usage_mb"] > (config.performance.max_memory_usage_mb * 0.9 if config else 460):
        health_status = "warning"
        issues.append("Alto uso de memória")
    
    # Verificar modelos
    if len(ml_system.models) == 0:
        health_status = "warning"
        issues.append("Nenhum modelo ML carregado")
    
    # Verificar dados
    if stats.get("total_trades", 0) == 0:
        issues.append("Nenhum dado de treino disponível")
    
    health_data = {
        "status": health_status,
        "timestamp": datetime.now().isoformat(),
        "issues": issues,
        "models_loaded": len(ml_system.models),
        "database_connected": True,
        "ml_stats": stats,
        "system_stats": system_stats,
        "uptime_hours": system_stats["uptime_hours"],
        "total_requests": system_stats["total_requests"]
    }
    
    # Log de monitoramento
    if monitor:
        monitor_health = monitor.health_check()
        health_data["monitoring"] = monitor_health
    
    return health_data

@app.post("/trade/save")
async def save_trade(
    trade_data: TradeData, 
    background_tasks: BackgroundTasks,
    credentials = Depends(verify_api_key)
):
    """Salva dados de trade com processamento avançado"""
    try:
        # Salvar trade no sistema ML
        ml_system.save_trade_data(trade_data)
        
        # Atualizar estatísticas de predição se aplicável
        if hasattr(trade_data, 'ml_prediction') and trade_data.ml_prediction:
            ml_system.update_prediction_stats(trade_data)
        
        # Processo em background
        def process_trade_background():
            try:
                # Verificar se deve retreinar
                if config and config.ml.auto_retrain_interval:
                    df = ml_system.get_training_data()
                    if len(df) % config.ml.auto_retrain_interval == 0 and len(df) >= ml_system.min_trades_for_training:
                        logger.info("Retreinando modelos automaticamente...")
                        ml_system.train_models_advanced()
                
                # Atualizar cache de padrões
                ml_system.clear_pattern_cache()
                
                # Log para monitoramento
                if monitor:
                    # Simular sessão de trading com o último trade
                    log_trading_session([trade_data.dict()])
                    
            except Exception as e:
                logger.error(f"Erro no processamento background: {e}")
        
        background_tasks.add_task(process_trade_background)
        
        return {
            "message": "Trade salvo com sucesso",
            "trade_id": trade_data.id,
            "ml_processing": "scheduled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao salvar trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict")
async def predict_trade_advanced(
    request: Dict,
    credentials = Depends(verify_api_key)
):
    """Predição ML avançada com cache e validação"""
    try:
        # Verificar cache se habilitado
        if config and config.performance.enable_cache:
            cache_key = json.dumps(request, sort_keys=True)
            cached_result = ml_system.get_cached_prediction(cache_key)
            if cached_result:
                logger.debug("Retornando predição do cache")
                return cached_result
        
        # Validar entrada
        required_fields = ['symbol', 'current_price', 'direction']
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"Campo obrigatório: {field}")
        
        # Converter request para features
        features = ml_system.extract_features_from_request(request)
        
        # Fazer predição
        prediction = ml_system.predict_trade_outcome(features)
        
        # Adicionar metadados
        prediction.update({
            "request_id": f"pred_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "model_versions": {name: getattr(model, '_sklearn_version', 'unknown') 
                             for name, model in ml_system.models.items()},
            "feature_count": len(features),
            "cache_hit": False
        })
        
        # Salvar no cache se habilitado
        if config and config.performance.enable_cache:
            ml_system.cache_prediction(cache_key, prediction)
        
        # Atualizar estatísticas
        ml_system.prediction_stats["total_predictions"] += 1
        
        return prediction
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/train")
async def train_models_endpoint(
    background_tasks: BackgroundTasks,
    force: bool = False,
    credentials = Depends(verify_api_key)
):
    """Endpoint para treinamento de modelos com opções avançadas"""
    try:
        # Verificar se deve treinar
        df = ml_system.get_training_data()
        
        if not force and len(df) < ml_system.min_trades_for_training:
            return {
                "message": f"Dados insuficientes para treino: {len(df)} < {ml_system.min_trades_for_training}",
                "current_data_size": len(df),
                "required_size": ml_system.min_trades_for_training,
                "suggestion": "Use force=true para treinar mesmo assim ou colete mais dados"
            }
        
        def train_models_background():
            try:
                start_time = time.time()
                logger.info("Iniciando treinamento de modelos...")
                
                success = ml_system.train_models_advanced()
                
                training_time = time.time() - start_time
                
                if success:
                    logger.info(f"Treinamento concluído em {training_time:.2f}s")
                    
                    # Gerar relatório de treinamento
                    if monitor:
                        report = ml_system.generate_training_report()
                        monitor.log_training_session(report)
                else:
                    logger.error("Treinamento falhou")
                    
            except Exception as e:
                logger.error(f"Erro no treinamento background: {e}")
        
        background_tasks.add_task(train_models_background)
        
        return {
            "message": "Treinamento iniciado em background",
            "data_size": len(df),
            "models_to_train": ml_system.models_to_train,
            "estimated_time_minutes": len(df) / 1000 * 2,  # Estimativa
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no endpoint de treinamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/stats")
async def get_ml_statistics_advanced(credentials = Depends(verify_api_key)):
    """Estatísticas ML avançadas"""
    try:
        stats = ml_system.get_ml_stats()
        patterns = ml_system.analyze_patterns()
        
        # Estatísticas de sistema
        system_stats = resource_monitor.get_stats()
        
        # Estatísticas de predição
        prediction_stats = ml_system.prediction_stats.copy()
        if prediction_stats["total_predictions"] > 0:
            prediction_stats["accuracy_rate"] = (
                prediction_stats["correct_predictions"] / prediction_stats["total_predictions"]
            )
        else:
            prediction_stats["accuracy_rate"] = 0.0
        
        # Estatísticas de treinamento
        training_stats = ml_system.training_stats
        
        comprehensive_stats = {
            "ml_stats": stats,
            "patterns": patterns,
            "prediction_performance": prediction_stats,
            "training_results": training_stats,
            "models_info": {
                "available": list(ml_system.models.keys()),
                "feature_count": len(ml_system.feature_columns),
                "feature_columns": ml_system.feature_columns
            },
            "system_performance": system_stats,
            "cache_stats": ml_system.get_cache_stats() if hasattr(ml_system, 'get_cache_stats') else {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Adicionar estatísticas de monitoramento se disponível
        if monitor:
            comprehensive_stats["monitoring"] = {
                "ml_performance": monitor.get_ml_performance_summary(7),
                "trading_performance": monitor.get_trading_performance_summary(7),
                "recent_alerts": monitor.get_recent_alerts(24)
            }
        
        return comprehensive_stats
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/dashboard")
async def monitoring_dashboard(credentials = Depends(verify_api_key)):
    """Dashboard de monitoramento"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Sistema de monitoramento não disponível")
    
    try:
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "system_health": monitor.health_check(),
            "daily_report": monitor.generate_daily_report(),
            "ml_performance": monitor.get_ml_performance_summary(7),
            "trading_performance": monitor.get_trading_performance_summary(7),
            "recent_alerts": monitor.get_recent_alerts(24),
            "system_stats": resource_monitor.get_stats()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Erro no dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_configuration(credentials = Depends(verify_api_key)):
    """Retorna configuração atual do sistema"""
    if not config:
        raise HTTPException(status_code=503, detail="Sistema de configuração não disponível")
    
    return {
        "environment_summary": config.get_environment_summary(),
        "ml_config": {
            "min_trades_for_training": config.ml.min_trades_for_training,
            "models_to_train": config.ml.models_to_train,
            "pattern_confidence_threshold": config.ml.pattern_confidence_threshold,
            "auto_retrain_interval": config.ml.auto_retrain_interval
        },
        "api_config": {
            "debug": config.api.debug,
            "rate_limit_enabled": config.api.rate_limit_enabled
        },
        "performance_config": {
            "cache_enabled": config.performance.enable_cache,
            "max_memory_mb": config.performance.max_memory_usage_mb
        },
@app.post("/ml/backup")
async def create_backup(
    background_tasks: BackgroundTasks,
    credentials = Depends(verify_api_key)
):
    """Cria backup dos modelos e dados"""
    def create_backup_background():
        try:
            success = ml_system.backup_models()
            if success:
                logger.info("Backup criado com sucesso")
            else:
                logger.error("Falha na criação do backup")
        except Exception as e:
            logger.error(f"Erro no backup: {e}")
    
    background_tasks.add_task(create_backup_background)
    
    return {
        "message": "Backup iniciado em background",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ml/optimize")
async def optimize_models(
    background_tasks: BackgroundTasks,
    credentials = Depends(verify_api_key)
):
    """Otimiza hiperparâmetros dos modelos"""
    def optimize_background():
        try:
            logger.info("Iniciando otimização de hiperparâmetros...")
            
            # Implementar otimização com GridSearchCV ou RandomizedSearchCV
            from sklearn.model_selection import RandomizedSearchCV
            
            df = ml_system.get_training_data()
            if len(df) < 100:
                logger.warning("Dados insuficientes para otimização")
                return
            
            X = df.drop(['label'], axis=1).select_dtypes(include=[np.number])
            y = df['label']
            
            # Parâmetros para otimização
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                }
            }
            
            optimized_models = {}
            
            for model_name, params in param_grids.items():
                if model_name in ml_system.models:
                    model = ml_system.models[model_name]
                    
                    search = RandomizedSearchCV(
                        model, params, n_iter=10, cv=3, 
                        scoring='accuracy', random_state=42, n_jobs=-1
                    )
                    
                    search.fit(X, y)
                    optimized_models[model_name] = {
                        'best_params': search.best_params_,
                        'best_score': search.best_score_,
                        'model': search.best_estimator_
                    }
            
            # Salvar modelos otimizados
            for model_name, result in optimized_models.items():
                ml_system.models[model_name] = result['model']
                logger.info(f"Modelo {model_name} otimizado: {result['best_score']:.3f}")
            
            ml_system.save_models()
            logger.info("Otimização concluída")
            
        except Exception as e:
            logger.error(f"Erro na otimização: {e}")
    
    background_tasks.add_task(optimize_background)
    
    return {
        "message": "Otimização de hiperparâmetros iniciada",
        "estimated_time_minutes": 10,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ml/feature-importance")
async def get_feature_importance(credentials = Depends(verify_api_key)):
    """Retorna importância das features"""
    try:
        importance_data = {}
        
        for model_name, model in ml_system.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                importance_data[model_name] = dict(zip(ml_system.feature_columns, importance_scores))
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_[0])
                importance_data[model_name] = dict(zip(ml_system.feature_columns, importance_scores))
        
        # Calcular importância média
        if importance_data:
            avg_importance = {}
            for feature in ml_system.feature_columns:
                scores = [importance_data[model].get(feature, 0) for model in importance_data.keys()]
                avg_importance[feature] = np.mean(scores)
            
            # Ordenar por importância
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "feature_importance_by_model": importance_data,
                "average_importance": dict(sorted_features),
                "top_features": sorted_features[:10],
                "total_features": len(ml_system.feature_columns),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "message": "Nenhum modelo com feature importance disponível",
                "available_models": list(ml_system.models.keys())
            }
            
    except Exception as e:
        logger.error(f"Erro ao obter feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/evaluate")
async def evaluate_models(credentials = Depends(verify_api_key)):
    """Avalia performance dos modelos com dados de teste"""
    try:
        df = ml_system.get_training_data()
        
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Dados insuficientes para avaliação")
        
        X = df.drop(['label'], axis=1).select_dtypes(include=[np.number])
        y = df['label']
        
        # Split para avaliação
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        evaluation_results = {}
        
        for model_name, model in ml_system.models.items():
            try:
                # Preparar dados baseado no tipo de modelo
                if model_name in ml_system.scalers:
                    scaler = ml_system.scalers[model_name]
                    X_test_processed = scaler.transform(X_test)
                else:
                    X_test_processed = X_test
                
                # Predições
                y_pred = model.predict(X_test_processed)
                y_pred_proba = model.predict_proba(X_test_processed)
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Métricas por classe
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                evaluation_results[model_name] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'confusion_matrix': conf_matrix.tolist(),
                    'classification_report': class_report,
                    'test_samples': len(X_test),
                    'prediction_confidence': {
                        'mean': float(np.mean(np.max(y_pred_proba, axis=1))),
                        'std': float(np.std(np.max(y_pred_proba, axis=1)))
                    }
                }
                
            except Exception as e:
                evaluation_results[model_name] = {'error': str(e)}
        
        # Ranking dos modelos
        model_ranking = sorted(
            [(name, result.get('accuracy', 0)) for name, result in evaluation_results.items() if 'error' not in result],
            key=lambda x: x[1], reverse=True
        )
        
        return {
            "evaluation_results": evaluation_results,
            "model_ranking": model_ranking,
            "best_model": model_ranking[0][0] if model_ranking else None,
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_data_size": len(X_test)
        }
        
    except Exception as e:
        logger.error(f"Erro na avaliação: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/export")
async def export_data(
    format: str = "json",
    days: int = 30,
    credentials = Depends(verify_api_key)
):
    """Exporta dados de trading"""
    try:
        # Obter dados dos últimos N dias
        since = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(ml_system.db_path)
        query = '''
            SELECT * FROM trades 
            WHERE created_at > ?
            ORDER BY created_at DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[since.isoformat()])
        conn.close()
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado no período")
        
        # Preparar dados para export
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "period_days": days,
                "total_records": len(df),
                "format": format
            },
            "data": df.to_dict('records')
        }
        
        if format.lower() == "json":
            return export_data
        elif format.lower() == "csv":
            from fastapi.responses import StreamingResponse
            import io
            
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=trading_data_{days}days.csv"}
            )
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use 'json' ou 'csv'")
            
    except Exception as e:
        logger.error(f"Erro na exportação: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/import")
async def import_data(
    background_tasks: BackgroundTasks,
    file_content: str,
    format: str = "json",
    credentials = Depends(verify_api_key)
):
    """Importa dados de trading"""
    def import_background():
        try:
            logger.info(f"Importando dados no formato {format}")
            
            if format.lower() == "json":
                data = json.loads(file_content)
                if isinstance(data, dict) and 'data' in data:
                    trades_data = data['data']
                else:
                    trades_data = data
            else:
                raise ValueError("Formato não suportado para importação")
            
            imported_count = 0
            
            for trade_data in trades_data:
                try:
                    # Converter para TradeData
                    trade = TradeData(**trade_data)
                    ml_system.save_trade_data(trade)
                    imported_count += 1
                except Exception as e:
                    logger.warning(f"Erro ao importar trade: {e}")
                    continue
            
            logger.info(f"Importação concluída: {imported_count} trades")
            
        except Exception as e:
            logger.error(f"Erro na importação: {e}")
    
    background_tasks.add_task(import_background)
    
    return {
        "message": "Importação iniciada em background",
        "format": format,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/logs")
async def get_system_logs(
    lines: int = 100,
    level: str = "INFO",
    credentials = Depends(verify_api_key)
):
    """Retorna logs do sistema"""
    try:
        log_file = config.logging.log_file_path if config else "logs/ml_trading.log"
        
        if not Path(log_file).exists():
            return {"message": "Arquivo de log não encontrado", "logs": []}
        
        # Ler últimas linhas do log
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # Filtrar por nível se especificado
        if level.upper() != "ALL":
            filtered_lines = [line for line in all_lines if level.upper() in line]
        else:
            filtered_lines = all_lines
        
        # Pegar últimas N linhas
        recent_lines = filtered_lines[-lines:] if lines > 0 else filtered_lines
        
        return {
            "logs": recent_lines,
            "total_lines": len(recent_lines),
            "log_level_filter": level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/cleanup")
async def cleanup_system(
    background_tasks: BackgroundTasks,
    days_old: int = 30,
    credentials = Depends(verify_api_key)
):
    """Limpa dados antigos do sistema"""
    def cleanup_background():
        try:
            logger.info(f"Iniciando limpeza de dados com mais de {days_old} dias")
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            conn = sqlite3.connect(ml_system.db_path)
            cursor = conn.cursor()
            
            # Contar registros antes da limpeza
            cursor.execute("SELECT COUNT(*) FROM trades WHERE created_at < ?", [cutoff_date.isoformat()])
            old_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ml_metrics WHERE created_at < ?", [cutoff_date.isoformat()])
            old_metrics = cursor.fetchone()[0]
            
            # Remover dados antigos
            cursor.execute("DELETE FROM trades WHERE created_at < ?", [cutoff_date.isoformat()])
            cursor.execute("DELETE FROM ml_metrics WHERE created_at < ?", [cutoff_date.isoformat()])
            
            # Limpar logs antigos
            cursor.execute("DELETE FROM alerts WHERE created_at < ? AND resolved = TRUE", [cutoff_date.isoformat()])
            
            conn.commit()
            conn.close()
            
            # Limpar cache
            ml_system.feature_cache.clear()
            ml_system.pattern_cache.clear()
            
            # Vacuum do banco
            conn = sqlite3.connect(ml_system.db_path)
            conn.execute("VACUUM")
            conn.close()
            
            logger.info(f"Limpeza concluída: {old_trades} trades, {old_metrics} métricas removidas")
            
        except Exception as e:
            logger.error(f"Erro na limpeza: {e}")
    
    background_tasks.add_task(cleanup_background)
    
    return {
        "message": f"Limpeza iniciada para dados com mais de {days_old} dias",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configurações do servidor
    host = config.api.host if config else "0.0.0.0"
    port = config.api.port if config else 8000
    debug = config.api.debug if config else False
    workers = config.api.workers if config else 1
    
    logger.info(f"🚀 Iniciando ML Trading API em {host}:{port}")
    logger.info(f"🧠 Modelos ML carregados: {len(ml_system.models)}")
    logger.info(f"📊 Monitoramento: {'Ativo' if monitor else 'Inativo'}")
    logger.info(f"🔒 Segurança: {'Ativa' if config and config.security.api_key_required else 'Inativa'}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        debug=debug,
        workers=workers if not debug else 1,
        access_log=True,
        log_level="info"
    )

# Inicialização do FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando sistema ML...")
    
    # Carregar modelos existentes
    ml_system.load_models()
    
    # Treinar modelos se houver dados suficientes
    try:
        df = ml_system.get_training_data()
        if len(df) >= ml_system.min_trades_for_training:
            logger.info("Retreinando modelos com dados existentes...")
            ml_system.train_models()
    except Exception as e:
        logger.warning(f"Erro no treino inicial: {e}")
    
    yield
    
    # Shutdown
    logger.info("Finalizando sistema ML...")

app = FastAPI(
    title="ML Trading API",
    description="API de Machine Learning para Trading Bot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints da API
@app.get("/")
async def root():
    return {
        "message": "ML Trading API está funcionando!",
        "version": "1.0.0",
        "models_loaded": len(ml_system.models),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    stats = ml_system.get_ml_stats()
    return {
        "status": "healthy",
        "models_loaded": len(ml_system.models),
        "database_connected": True,
        "ml_stats": stats,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/trade/save")
async def save_trade(trade_data: TradeData, background_tasks: BackgroundTasks):
    """Salva dados de trade e retreina modelo se necessário"""
    try:
        ml_system.save_trade_data(trade_data)
        
        # Retreinar modelo a cada 50 novos trades
        def maybe_retrain():
            df = ml_system.get_training_data()
            if len(df) % 50 == 0 and len(df) >= ml_system.min_trades_for_training:
                logger.info("Retreinando modelos...")
                ml_system.train_models()
                
        background_tasks.add_task(maybe_retrain)
        
        return {"message": "Trade salvo com sucesso", "trade_id": trade_data.id}
        
    except Exception as e:
        logger.error(f"Erro ao salvar trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict")
async def predict_trade(request: Dict):
    """Faz predição ML para um trade"""
    try:
        # Converter request para features
        features = {
            'hour_of_day': datetime.now().hour,
            'volatility': request.get('volatility', 50),
            'martingale_level': request.get('martingale_level', 0),
            'stake_normalized': min(request.get('stake', 1) / 10.0, 1.0),
            'symbol_encoded': hash(request.get('symbol', 'R_50')) % 100,
            'direction_encoded': 1 if request.get('direction') == 'CALL' else 0,
            'trend_encoded': {'bullish': 1, 'bearish': -1, 'neutral': 0}.get(request.get('trend', 'neutral'), 0),
            'duration_minutes': 1.0,
            'entry_price_normalized': request.get('current_price', 1000) / 1000.0,
            'recent_wins': request.get('recent_wins', 0),
            'recent_losses': request.get('recent_losses', 0),
            'recent_win_rate': request.get('recent_win_rate', 0.5)
        }
        
        prediction = ml_system.predict_trade_outcome(features)
        return prediction
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/analyze")
async def analyze_market(request: MarketAnalysisRequest):
    """Análise de mercado com ML"""
    try:
        patterns = ml_system.analyze_patterns()
        
        # Análise específica baseada no request
        analysis = {
            "message": f"Análise ML do {request.symbol}: Volatilidade {request.volatility:.1f}%",
            "volatility": request.volatility,
            "trend": "neutral",
            "confidence": 75 + np.random.random() * 20,
            "patterns_found": len(patterns['patterns']),
            "ml_recommendation": "continue",
            "risk_level": "medium"
        }
        
        # Ajustar recomendação baseado nos padrões
        if request.martingale_level > 3:
            analysis["confidence"] -= 15
            analysis["risk_level"] = "high"
            analysis["ml_recommendation"] = "caution"
            
        if request.is_after_loss:
            analysis["confidence"] -= 10
            analysis["message"] += " | Análise pós-perda: Aguardar setup melhor"
            
        return analysis
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/signal")
async def get_trading_signal(request: TradingSignalRequest):
    """Obtém sinal de trading baseado em ML"""
    try:
        # Preparar features para predição
        features = {
            'hour_of_day': datetime.now().hour,
            'volatility': request.volatility,
            'martingale_level': request.martingale_level,
            'stake_normalized': 0.1,  # Valor padrão
            'symbol_encoded': hash(request.symbol) % 100,
            'direction_encoded': 0.5,  # Neutro inicialmente
            'trend_encoded': 0,
            'duration_minutes': 1.0,
            'entry_price_normalized': request.current_price / 1000.0,
            'recent_wins': len([t for t in request.recent_trades if t.get('status') == 'won']),
            'recent_losses': len([t for t in request.recent_trades if t.get('status') == 'lost']),
            'recent_win_rate': request.win_rate / 100.0
        }
        
        # Testar ambas as direções
        features_call = features.copy()
        features_call['direction_encoded'] = 1
        
        features_put = features.copy()
        features_put['direction_encoded'] = 0
        
        prediction_call = ml_system.predict_trade_outcome(features_call)
        prediction_put = ml_system.predict_trade_outcome(features_put)
        
        # Escolher melhor direção
        if prediction_call['win_probability'] > prediction_put['win_probability']:
            direction = 'CALL'
            confidence = prediction_call['confidence'] * 100
            win_prob = prediction_call['win_probability']
        else:
            direction = 'PUT'
            confidence = prediction_put['confidence'] * 100
            win_prob = prediction_put['win_probability']
            
        # Ajustar confiança baseado no contexto
        if request.is_after_loss and request.martingale_level > 0:
            confidence = max(60, confidence - 10)
            
        return {
            "direction": direction,
            "confidence": confidence,
            "win_probability": win_prob,
            "reasoning": f"ML analysis: {win_prob:.1%} win probability",
            "model_used": "ensemble_ml",
            "martingale_aware": request.martingale_level > 0,
            "risk_adjusted": request.is_after_loss
        }
        
    except Exception as e:
        logger.error(f"Erro no sinal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/risk")
async def assess_risk(request: RiskAssessmentRequest):
    """Avaliação de risco com ML"""
    try:
        risk_score = 50  # Base
        
        # Fatores de risco
        if request.martingale_level > 4:
            risk_score += 30
        elif request.martingale_level > 2:
            risk_score += 15
            
        if request.win_rate < 40:
            risk_score += 20
            
        if request.today_pnl < -100:
            risk_score += 25
            
        # Análise ML de padrões de risco
        patterns = ml_system.analyze_patterns()
        risk_patterns = [p for p in patterns['patterns'] if 'risk' in p['type'] or 'avoid' in p['type']]
        
        if len(risk_patterns) > 2:
            risk_score += 15
            
        # Determinar nível de risco
        if risk_score >= 80:
            level = "high"
            recommendation = "Parar operações e reavaliar estratégia"
        elif risk_score >= 60:
            level = "medium"
            recommendation = "Operar com muita cautela"
        else:
            level = "low"
            recommendation = "Continuar operando normalmente"
            
        return {
            "level": level,
            "score": risk_score,
            "message": f"Risco {level.upper()} detectado (Score: {risk_score})",
            "recommendation": recommendation,
            "risk_factors": {
                "martingale_level": request.martingale_level,
                "win_rate": request.win_rate,
                "pnl_today": request.today_pnl,
                "risk_patterns": len(risk_patterns)
            },
            "ml_analysis": True
        }
        
    except Exception as e:
        logger.error(f"Erro na avaliação de risco: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/train")
async def train_models(background_tasks: BackgroundTasks):
    """Força retreinamento dos modelos"""
    def train():
        success = ml_system.train_models()
        logger.info(f"Retreinamento {'bem-sucedido' if success else 'falhou'}")
        
    background_tasks.add_task(train)
    
    return {"message": "Retreinamento iniciado em background"}

@app.get("/ml/stats")
async def get_ml_statistics():
    """Retorna estatísticas do sistema ML"""
    try:
        stats = ml_system.get_ml_stats()
        patterns = ml_system.analyze_patterns()
        
        return {
            "ml_stats": stats,
            "patterns": patterns,
            "models_available": list(ml_system.models.keys()),
            "feature_count": len(ml_system.feature_columns),
            "database_size": stats["total_trades"]
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/patterns")
async def get_patterns():
    """Retorna padrões identificados pelo ML"""
    try:
        patterns = ml_system.analyze_patterns()
        return patterns
        
    except Exception as e:
        logger.error(f"Erro ao obter padrões: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)