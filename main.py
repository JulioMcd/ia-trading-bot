#!/usr/bin/env python3
"""
API Principal do ML Trading Bot
FastAPI + Scikit-learn para predições em tempo real
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== MODELS PYDANTIC =====

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
    market_context: Optional[Dict] = None
    martingale_level: int = 0
    volatility: float = 0.0
    trend: str = "neutral"

class PredictionRequest(BaseModel):
    symbol: str
    current_price: float
    direction: str
    stake: float
    duration: str
    trend: str = "neutral"
    volatility: float = 0.0
    martingale_level: int = 0
    recent_wins: int = 0
    recent_losses: int = 0
    recent_win_rate: float = 0.5

class AnalysisRequest(BaseModel):
    symbol: str
    current_price: float
    timestamp: str
    trades: List[Dict] = []
    balance: float = 1000.0
    win_rate: float = 0.0
    volatility: float = 0.0
    market_condition: str = "neutral"
    martingale_level: int = 0
    is_after_loss: bool = False
    ml_patterns: int = 0
    ml_accuracy: float = 0.0

# ===== ML TRADING SYSTEM =====

class MLTradingSystem:
    """Sistema de Machine Learning para Trading"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.last_training = None
        self.feature_columns = [
            'current_price', 'volatility', 'martingale_level', 
            'recent_win_rate', 'stake', 'duration_numeric'
        ]
        
        # Criar diretórios
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        # Inicializar banco
        self.init_database()
        
        # Carregar modelos se existirem
        self.load_models()
        
        # Treinar se necessário
        if not self.is_trained:
            self.train_initial_models()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                stake REAL NOT NULL,
                duration TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                outcome TEXT,
                market_context TEXT,
                martingale_level INTEGER DEFAULT 0,
                volatility REAL DEFAULT 0,
                trend TEXT DEFAULT 'neutral',
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de métricas ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                accuracy REAL NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Banco de dados inicializado")
    
    def create_features(self, data: Dict) -> np.ndarray:
        """Cria features para o modelo ML"""
        try:
            # Extrair duration numérico
            duration_str = str(data.get('duration', '5'))
            duration_numeric = float(duration_str.replace('t', '').replace('ticks', ''))
            
            features = [
                float(data.get('current_price', 0)),
                float(data.get('volatility', 50)),  # Default médio
                int(data.get('martingale_level', 0)),
                float(data.get('recent_win_rate', 0.5)),
                float(data.get('stake', 1)),
                duration_numeric
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erro ao criar features: {e}")
            # Features padrão em caso de erro
            return np.array([1000, 50, 0, 0.5, 1, 5]).reshape(1, -1)
    
    def get_trade_data(self) -> pd.DataFrame:
        """Busca dados de trades do banco"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE outcome IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''', conn)
            
            if len(df) == 0:
                # Criar dados sintéticos para treinamento inicial
                logger.info("Criando dados sintéticos para treinamento inicial...")
                df = self.create_synthetic_data()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar trades: {e}")
            return self.create_synthetic_data()
        finally:
            conn.close()
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Cria dados sintéticos para treinamento inicial"""
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'id': [f'synthetic_{i}' for i in range(n_samples)],
            'current_price': np.random.uniform(800, 1200, n_samples),
            'volatility': np.random.uniform(20, 80, n_samples),
            'martingale_level': np.random.choice([0, 1, 2], n_samples),
            'recent_win_rate': np.random.uniform(0.3, 0.7, n_samples),
            'stake': np.random.uniform(1, 10, n_samples),
            'duration': np.random.choice([5, 7, 10], n_samples),
            'direction': np.random.choice(['CALL', 'PUT'], n_samples)
        }
        
        # Criar outcomes baseados em lógica simples
        outcomes = []
        for i in range(n_samples):
            # Lógica sintética: win_rate alta = mais chance de win
            win_prob = data['recent_win_rate'][i] * 0.8 + np.random.uniform(0, 0.4)
            outcome = 'won' if win_prob > 0.5 else 'lost'
            outcomes.append(outcome)
        
        data['outcome'] = outcomes
        
        return pd.DataFrame(data)
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepara dados para treinamento"""
        if len(df) < 10:
            raise ValueError("Dados insuficientes para treinamento")
        
        # Features
        features_list = []
        for _, row in df.iterrows():
            duration_numeric = float(str(row.get('duration', 5)).replace('t', '').replace('ticks', ''))
            
            features = [
                float(row.get('current_price', 1000)),
                float(row.get('volatility', 50)),
                int(row.get('martingale_level', 0)),
                float(row.get('recent_win_rate', 0.5)),
                float(row.get('stake', 1)),
                duration_numeric
            ]
            features_list.append(features)
        
        X = np.array(features_list)
        
        # Target: converter outcome para binário
        y = (df['outcome'] == 'won').astype(int)
        
        return X, y
    
    def train_models(self) -> Dict:
        """Treina todos os modelos ML"""
        logger.info("🎓 Iniciando treinamento dos modelos ML...")
        
        try:
            # Buscar dados
            df = self.get_trade_data()
            X, y = self.prepare_training_data(df)
            
            logger.info(f"📊 Dados de treinamento: {len(df)} trades")
            
            # Split dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Salvar scaler
            self.scalers['main'] = scaler
            
            # Modelos para treinar
            models_config = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42),
                'svm': SVC(probability=True, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
            }
            
            results = {}
            
            # Treinar cada modelo
            for name, model in models_config.items():
                try:
                    logger.info(f"Treinando {name}...")
                    
                    # Treinar
                    if name in ['svm', 'logistic_regression', 'neural_network']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Avaliar
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Salvar modelo
                    self.models[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'trained_at': datetime.now().isoformat(),
                        'samples': len(X_train)
                    }
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'samples': len(X_train)
                    }
                    
                    logger.info(f"✅ {name}: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    logger.error(f"❌ Erro treinando {name}: {e}")
                    continue
            
            # Salvar modelos
            self.save_models()
            
            self.is_trained = True
            self.last_training = {
                'timestamp': datetime.now().isoformat(),
                'models_trained': len(results),
                'best_accuracy': max([r['accuracy'] for r in results.values()]) if results else 0
            }
            
            logger.info(f"✅ Treinamento concluído: {len(results)} modelos")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return {}
    
    def train_initial_models(self):
        """Treina modelos iniciais com dados sintéticos"""
        logger.info("🎯 Treinamento inicial com dados sintéticos...")
        self.train_models()
    
    def save_models(self):
        """Salva modelos treinados"""
        try:
            for name, model_data in self.models.items():
                model_path = f"models/{name}_model.joblib"
                joblib.dump(model_data, model_path)
            
            # Salvar scalers
            for name, scaler in self.scalers.items():
                scaler_path = f"models/{name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            
            logger.info("💾 Modelos salvos")
            
        except Exception as e:
            logger.error(f"❌ Erro salvando modelos: {e}")
    
    def load_models(self):
        """Carrega modelos salvos"""
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                return
            
            # Carregar modelos
            for model_file in models_dir.glob("*_model.joblib"):
                name = model_file.stem.replace("_model", "")
                try:
                    model_data = joblib.load(model_file)
                    self.models[name] = model_data
                    logger.info(f"📂 Modelo {name} carregado")
                except Exception as e:
                    logger.error(f"❌ Erro carregando {name}: {e}")
            
            # Carregar scalers
            for scaler_file in models_dir.glob("*_scaler.joblib"):
                name = scaler_file.stem.replace("_scaler", "")
                try:
                    scaler = joblib.load(scaler_file)
                    self.scalers[name] = scaler
                except Exception as e:
                    logger.error(f"❌ Erro carregando scaler {name}: {e}")
            
            if self.models:
                self.is_trained = True
                logger.info(f"✅ {len(self.models)} modelos carregados")
            
        except Exception as e:
            logger.error(f"❌ Erro carregando modelos: {e}")
    
    def predict(self, request_data: Dict) -> Dict:
        """Faz predição ML"""
        try:
            if not self.models:
                return {
                    "prediction": "neutral",
                    "confidence": 0.5,
                    "model_used": "none",
                    "reason": "Nenhum modelo treinado disponível"
                }
            
            # Criar features
            features = self.create_features(request_data)
            
            # Usar o melhor modelo disponível
            best_model_name = max(self.models.keys(), 
                                key=lambda x: self.models[x]['accuracy'])
            
            model_data = self.models[best_model_name]
            model = model_data['model']
            
            # Aplicar normalização se necessário
            if best_model_name in ['svm', 'logistic_regression', 'neural_network']:
                if 'main' in self.scalers:
                    features = self.scalers['main'].transform(features)
            
            # Predição
            prediction_proba = model.predict_proba(features)[0]
            prediction_binary = model.predict(features)[0]
            
            # Interpretar resultado
            confidence = max(prediction_proba)
            
            if prediction_binary == 1 and confidence > 0.6:
                prediction = "favor"
                reason = f"ML recomenda este trade com {confidence:.1%} confiança"
            elif prediction_binary == 0 and confidence > 0.6:
                prediction = "avoid"
                reason = f"ML recomenda evitar este trade ({confidence:.1%} confiança de perda)"
            else:
                prediction = "neutral"
                reason = f"ML neutro - confiança baixa ({confidence:.1%})"
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "win_probability": float(prediction_proba[1]),
                "model_used": best_model_name,
                "reason": reason,
                "model_accuracy": model_data['accuracy']
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            return {
                "prediction": "neutral",
                "confidence": 0.5,
                "model_used": "error",
                "reason": f"Erro na predição: {str(e)}"
            }
    
    def save_trade(self, trade_data: TradeData) -> bool:
        """Salva trade no banco de dados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades 
                (id, timestamp, symbol, direction, stake, duration, entry_price, 
                 exit_price, outcome, market_context, martingale_level, volatility, trend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.id,
                trade_data.timestamp,
                trade_data.symbol,
                trade_data.direction,
                trade_data.stake,
                trade_data.duration,
                trade_data.entry_price,
                trade_data.exit_price,
                trade_data.outcome,
                json.dumps(trade_data.market_context) if trade_data.market_context else None,
                trade_data.martingale_level,
                trade_data.volatility,
                trade_data.trend
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Trade {trade_data.id} salvo")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro salvando trade: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do sistema ML"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Estatísticas básicas
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE outcome = 'won'")
            total_wins = cursor.fetchone()[0]
            
            win_rate = total_wins / total_trades if total_trades > 0 else 0
            
            # Padrões simples
            patterns = self.analyze_patterns()
            
            conn.close()
            
            return {
                "ml_stats": {
                    "total_trades": total_trades,
                    "total_wins": total_wins,
                    "overall_win_rate": win_rate,
                    "models_loaded": len(self.models),
                    "last_training": self.last_training
                },
                "models_available": list(self.models.keys()),
                "patterns": {
                    "patterns": patterns,
                    "patterns_count": len(patterns)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas: {e}")
            return {
                "ml_stats": {
                    "total_trades": 0,
                    "total_wins": 0,
                    "overall_win_rate": 0,
                    "models_loaded": len(self.models),
                    "last_training": None
                },
                "models_available": [],
                "patterns": {"patterns": [], "patterns_count": 0}
            }
    
    def analyze_patterns(self) -> List[Dict]:
        """Analisa padrões simples nos dados"""
        try:
            df = self.get_trade_data()
            if len(df) < 10:
                return []
            
            patterns = []
            
            # Padrão 1: Performance por símbolo
            if 'symbol' in df.columns and 'outcome' in df.columns:
                symbol_stats = df.groupby('symbol')['outcome'].apply(
                    lambda x: (x == 'won').mean()
                ).to_dict()
                
                for symbol, win_rate in symbol_stats.items():
                    if win_rate > 0.6:
                        patterns.append({
                            "type": "symbol_performance",
                            "description": f"Símbolo {symbol} tem alta taxa de vitória",
                            "confidence": win_rate,
                            "data": {"symbol": symbol, "win_rate": win_rate}
                        })
            
            # Padrão 2: Performance por direção
            if 'direction' in df.columns:
                direction_stats = df.groupby('direction')['outcome'].apply(
                    lambda x: (x == 'won').mean()
                ).to_dict()
                
                for direction, win_rate in direction_stats.items():
                    if win_rate > 0.6:
                        patterns.append({
                            "type": "direction_bias",
                            "description": f"Direção {direction} tem melhor performance",
                            "confidence": win_rate,
                            "data": {"direction": direction, "win_rate": win_rate}
                        })
            
            return patterns[:10]  # Máximo 10 padrões
            
        except Exception as e:
            logger.error(f"❌ Erro analisando padrões: {e}")
            return []

# ===== INSTÂNCIA GLOBAL =====
ml_system = MLTradingSystem()

# ===== FASTAPI APP =====

app = FastAPI(
    title="ML Trading Bot API",
    description="API de Machine Learning para Trading Bot com Scikit-learn",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS =====

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "🧠 ML Trading Bot API",
        "version": "1.0.0",
        "status": "online",
        "models_loaded": len(ml_system.models),
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(ml_system.models),
        "is_trained": ml_system.is_trained,
        "ml_stats": ml_system.get_stats()["ml_stats"]
    }

@app.post("/ml/predict")
async def predict_trade(request: PredictionRequest):
    """Endpoint para predições ML"""
    try:
        request_data = request.dict()
        prediction = ml_system.predict(request_data)
        
        return JSONResponse(content=prediction)
        
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/save")
async def save_trade(trade: TradeData):
    """Salva dados de trade"""
    try:
        success = ml_system.save_trade(trade)
        
        if success:
            return {"message": "Trade salvo com sucesso", "trade_id": trade.id}
        else:
            raise HTTPException(status_code=500, detail="Erro ao salvar trade")
            
    except Exception as e:
        logger.error(f"❌ Erro salvando trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/stats")
async def get_ml_stats():
    """Retorna estatísticas ML"""
    try:
        stats = ml_system.get_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"❌ Erro nas estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/train")
async def train_models(background_tasks: BackgroundTasks):
    """Retreina modelos ML em background"""
    try:
        def train_in_background():
            logger.info("🎓 Iniciando treinamento em background...")
            ml_system.train_models()
            logger.info("✅ Treinamento concluído")
        
        background_tasks.add_task(train_in_background)
        
        return {
            "message": "Treinamento iniciado em background",
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"❌ Erro iniciando treinamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/analyze")
async def analyze_market(request: AnalysisRequest):
    """Análise de mercado ML"""
    try:
        # Análise simples baseada nos dados
        analysis = {
            "message": "Análise de mercado concluída",
            "recommendation": "neutral",
            "confidence": 0.7,
            "factors": [
                f"Win rate atual: {request.win_rate:.1f}%",
                f"Volatilidade: {request.volatility:.1f}",
                f"Condição do mercado: {request.market_condition}"
            ]
        }
        
        # Lógica simples de recomendação
        if request.win_rate > 60 and request.volatility < 50:
            analysis["recommendation"] = "favorable"
            analysis["message"] = "Condições favoráveis para trading"
        elif request.win_rate < 40 or request.volatility > 80:
            analysis["recommendation"] = "cautious"
            analysis["message"] = "Condições requerem cautela"
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/patterns")
async def get_patterns():
    """Retorna padrões identificados"""
    try:
        patterns = ml_system.analyze_patterns()
        
        return {
            "patterns": patterns,
            "total": len(patterns),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erro nos padrões: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== MAIN =====

if __name__ == "__main__":
    # Configurações
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Iniciando ML Trading Bot API em {host}:{port}")
    logger.info(f"📊 Modelos carregados: {len(ml_system.models)}")
    logger.info(f"🎯 Treinamento: {'OK' if ml_system.is_trained else 'Pendente'}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
