#!/usr/bin/env python3
"""
Aplicação Principal - ML Trading Bot
Sistema de Machine Learning para Trading Bot com FastAPI
"""

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

# Importar módulos locais
try:
    from config import config, get_config
    from monitoring import monitor, log_ml_training_result, log_trading_session, MLMetrics, TradingMetrics
except ImportError as e:
    print(f"Aviso: Não foi possível importar alguns módulos: {e}")
    config = None
    monitor = None

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MLTradingMain")

# Sistema de autenticação
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verifica API key se a segurança estiver habilitada"""
    if config and hasattr(config, 'security') and config.security.api_key_required:
        if not credentials or credentials.credentials != config.security.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key inválida",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials

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

# Sistema ML Principal
class MLTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Configurações
        if config and hasattr(config, 'database'):
            self.db_path = config.database.path
        else:
            self.db_path = "data/trading_data.db"
            
        if config and hasattr(config, 'ml'):
            self.min_trades_for_training = config.ml.min_trades_for_training
            self.pattern_confidence_threshold = config.ml.pattern_confidence_threshold
            self.models_to_train = config.ml.models_to_train
        else:
            self.min_trades_for_training = 50
            self.pattern_confidence_threshold = 0.7
            self.models_to_train = ["random_forest", "gradient_boosting", "logistic_regression"]
            
        # Estatísticas
        self.training_stats = {}
        self.prediction_stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "last_accuracy": 0.0
        }
        
        self.initialize_database()
        self.load_models()
        
        logger.info("Sistema ML inicializado")
        
    def initialize_database(self):
        """Inicializa o banco de dados SQLite"""
        # Criar diretório se não existir
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
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
            'stake_normalized': min(trade_data.stake / 10.0, 1.0),
            'symbol_encoded': hash(trade_data.symbol) % 100,
            'direction_encoded': 1 if trade_data.direction == 'CALL' else 0,
            'trend_encoded': {'bullish': 1, 'bearish': -1, 'neutral': 0}.get(trade_data.trend, 0),
            'duration_minutes': self.parse_duration_to_minutes(trade_data.duration),
            'entry_price_normalized': trade_data.entry_price / 1000.0,
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
            return float(duration.replace('t', '')) * 0.1
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
        
    def train_models(self) -> bool:
        """Treina os modelos ML"""
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinar modelos
        model_configs = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        trained_models = 0
        
        for model_name in self.models_to_train:
            if model_name in model_configs:
                try:
                    logger.info(f"Treinando modelo: {model_name}")
                    
                    model = model_configs[model_name]
                    
                    # Aplicar scaling para regressão logística
                    if model_name == 'logistic_regression':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        self.scalers[model_name] = scaler
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calcular métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Salvar modelo
                    self.models[model_name] = model
                    
                    logger.info(f"Modelo {model_name}: Accuracy = {accuracy:.3f}")
                    trained_models += 1
                    
                except Exception as e:
                    logger.error(f"Erro ao treinar modelo {model_name}: {e}")
                    continue
        
        if trained_models > 0:
            self.save_models()
            logger.info(f"Treinamento concluído: {trained_models} modelos treinados")
            return True
        else:
            logger.error("Nenhum modelo foi treinado com sucesso")
            return False
        
    def save_models(self):
        """Salva modelos treinados"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
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
            models_dir = Path('models')
            if not models_dir.exists():
                logger.info("Diretório de modelos não encontrado")
                return
                
            model_files = list(models_dir.glob('*.pkl'))
            
            for model_file in model_files:
                if model_file.name.startswith('scaler_'):
                    scaler_name = model_file.name.replace('scaler_', '').replace('.pkl', '')
                    self.scalers[scaler_name] = joblib.load(model_file)
                elif not model_file.name.startswith('scaler_'):
                    model_name = model_file.name.replace('.pkl', '')
                    self.models[model_name] = joblib.load(model_file)
                    
            # Carregar feature columns
            features_path = models_dir / 'feature_columns.json'
            if features_path.exists():
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
                if name == 'logistic_regression' and name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
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
        
        # Análise básica de padrões
        conn = sqlite3.connect(self.db_path)
        
        # Padrão 1: Performance por símbolo
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
                
        conn.close()
        
        return {
            'patterns': patterns,
            'total_trades': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
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
        
        conn.close()
        
        stats = {
            'total_trades': int(basic_stats.iloc[0]['total_trades']) if len(basic_stats) > 0 else 0,
            'total_wins': int(basic_stats.iloc[0]['total_wins']) if len(basic_stats) > 0 else 0,
            'overall_win_rate': float(basic_stats.iloc[0]['overall_win_rate']) if len(basic_stats) > 0 else 0.0,
            'models_loaded': len(self.models),
            'feature_count': len(self.feature_columns)
        }
        
        return stats

# Instância global do sistema ML
ml_system = MLTradingSystem()

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
    version="2.0.0",
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
        "message": "🧠 ML Trading API está funcionando!",
        "version": "2.0.0",
        "models_loaded": len(ml_system.models),
        "timestamp": datetime.now().isoformat(),
        "environment": config.get_environment_summary() if config else "basic"
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
    
    # Configurações do servidor
    host = config.api.host if config and hasattr(config, 'api') else "0.0.0.0"
    port = config.api.port if config and hasattr(config, 'api') else 8000
    debug = config.api.debug if config and hasattr(config, 'api') else False
    
    logger.info(f"🚀 Iniciando ML Trading API em {host}:{port}")
    logger.info(f"🧠 Modelos ML carregados: {len(ml_system.models)}")
    logger.info(f"📊 Monitoramento: {'Ativo' if monitor else 'Inativo'}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        debug=debug,
        access_log=True,
        log_level="info"
    )
