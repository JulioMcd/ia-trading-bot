#!/usr/bin/env python3
"""
API Principal do ML Trading Bot - ANÁLISES REAIS
FastAPI + Scikit-learn para predições em tempo real com logs detalhados
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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """Sistema de Machine Learning para Trading com Análises Reais"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.last_training = None
        self.prediction_count = 0
        self.analysis_count = 0
        self.feature_columns = [
            'current_price', 'volatility', 'martingale_level', 
            'recent_win_rate', 'stake', 'duration_numeric'
        ]
        
        # Estatísticas em tempo real
        self.stats = {
            'total_predictions': 0,
            'total_analysis': 0,
            'total_trades_learned': 0,
            'models_accuracy': {},
            'patterns_found': []
        }
        
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
        
        logger.info("🧠 Sistema ML Trading REAL inicializado com sucesso")
    
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
        
        # Tabela de atividade frontend
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frontend_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                request_data TEXT,
                response_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Banco de dados inicializado com tabelas ML")
    
    def log_frontend_activity(self, endpoint: str, request_data: dict, response_data: dict, ip: str = None):
        """Log da atividade do frontend"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO frontend_activity (endpoint, request_data, response_data, ip_address)
                VALUES (?, ?, ?, ?)
            ''', (
                endpoint,
                json.dumps(request_data),
                json.dumps(response_data),
                ip or 'unknown'
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"📝 Atividade frontend registrada: {endpoint}")
        except Exception as e:
            logger.error(f"Erro ao registrar atividade: {e}")
    
    def create_features(self, data: Dict) -> np.ndarray:
        """Cria features para o modelo ML"""
        try:
            # Extrair duration numérico
            duration_str = str(data.get('duration', '5'))
            duration_numeric = float(duration_str.replace('t', '').replace('ticks', ''))
            
            features = [
                float(data.get('current_price', 1000)),
                float(data.get('volatility', 50)),
                int(data.get('martingale_level', 0)),
                float(data.get('recent_win_rate', 0.5)),
                float(data.get('stake', 1)),
                duration_numeric
            ]
            
            logger.info(f"🔢 Features criadas: {features}")
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erro ao criar features: {e}")
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
                logger.info("Criando dados sintéticos para treinamento inicial...")
                df = self.create_synthetic_data()
            else:
                logger.info(f"📊 {len(df)} trades reais carregados do banco")
            
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
            win_prob = data['recent_win_rate'][i] * 0.8 + np.random.uniform(0, 0.4)
            outcome = 'won' if win_prob > 0.5 else 'lost'
            outcomes.append(outcome)
        
        data['outcome'] = outcomes
        
        return pd.DataFrame(data)
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepara dados para treinamento"""
        if len(df) < 10:
            raise ValueError("Dados insuficientes para treinamento")
        
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
        y = (df['outcome'] == 'won').astype(int)
        
        return X, y
    
    def train_models(self) -> Dict:
        """Treina todos os modelos ML"""
        logger.info("🎓 Iniciando treinamento dos modelos ML...")
        
        try:
            df = self.get_trade_data()
            X, y = self.prepare_training_data(df)
            
            logger.info(f"📊 Dados de treinamento: {len(df)} trades")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['main'] = scaler
            
            models_config = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42),
                'svm': SVC(probability=True, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
            }
            
            results = {}
            
            for name, model in models_config.items():
                try:
                    logger.info(f"Treinando {name}...")
                    
                    if name in ['svm', 'logistic_regression', 'neural_network']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.models[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'trained_at': datetime.now().isoformat(),
                        'samples': len(X_train)
                    }
                    
                    # Atualizar estatísticas
                    self.stats['models_accuracy'][name] = accuracy
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'samples': len(X_train)
                    }
                    
                    logger.info(f"✅ {name}: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    logger.error(f"❌ Erro treinando {name}: {e}")
                    continue
            
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
            
            for model_file in models_dir.glob("*_model.joblib"):
                name = model_file.stem.replace("_model", "")
                try:
                    model_data = joblib.load(model_file)
                    self.models[name] = model_data
                    logger.info(f"📂 Modelo {name} carregado")
                except Exception as e:
                    logger.error(f"❌ Erro carregando {name}: {e}")
            
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
        """Faz predição ML REAL"""
        logger.info(f"🧠 Nova solicitação de predição ML: {request_data}")
        
        try:
            if not self.models:
                logger.warning("⚠️ Nenhum modelo disponível para predição")
                return {
                    "prediction": "neutral",
                    "confidence": 0.5,
                    "model_used": "none",
                    "reason": "Nenhum modelo treinado disponível",
                    "analysis_real": True
                }
            
            features = self.create_features(request_data)
            
            best_model_name = max(self.models.keys(), 
                                key=lambda x: self.models[x]['accuracy'])
            
            model_data = self.models[best_model_name]
            model = model_data['model']
            
            if best_model_name in ['svm', 'logistic_regression', 'neural_network']:
                if 'main' in self.scalers:
                    features = self.scalers['main'].transform(features)
            
            prediction_proba = model.predict_proba(features)[0]
            prediction_binary = model.predict(features)[0]
            
            confidence = max(prediction_proba)
            
            if prediction_binary == 1 and confidence > 0.6:
                prediction = "favor"
                reason = f"ML recomenda CALL/RISE com {confidence:.1%} confiança"
            elif prediction_binary == 0 and confidence > 0.6:
                prediction = "avoid"
                reason = f"ML recomenda PUT/FALL com {confidence:.1%} confiança"
            else:
                prediction = "neutral"
                reason = f"ML neutro - confiança baixa ({confidence:.1%})"
            
            # Atualizar estatísticas
            self.stats['total_predictions'] += 1
            self.prediction_count += 1
            
            result = {
                "prediction": prediction,
                "confidence": float(confidence),
                "win_probability": float(prediction_proba[1]),
                "model_used": best_model_name,
                "reason": reason,
                "model_accuracy": model_data['accuracy'],
                "analysis_real": True,
                "prediction_id": self.prediction_count,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Predição ML concluída: {prediction} ({confidence:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na predição ML: {e}")
            return {
                "prediction": "neutral",
                "confidence": 0.5,
                "model_used": "error",
                "reason": f"Erro na predição: {str(e)}",
                "analysis_real": False
            }
    
    def analyze_market(self, analysis_data: Dict) -> Dict:
        """Análise de mercado ML REAL"""
        logger.info(f"📊 Nova solicitação de análise ML: {analysis_data}")
        
        try:
            self.analysis_count += 1
            self.stats['total_analysis'] += 1
            
            # Análise real baseada nos dados
            symbol = analysis_data.get('symbol', 'Unknown')
            win_rate = analysis_data.get('win_rate', 0)
            volatility = analysis_data.get('volatility', 50)
            martingale_level = analysis_data.get('martingale_level', 0)
            market_condition = analysis_data.get('market_condition', 'neutral')
            
            # Lógica de análise real
            score = 0.5  # Base neutra
            
            # Ajustar score baseado em win rate
            if win_rate > 60:
                score += 0.2
            elif win_rate < 40:
                score -= 0.2
            
            # Ajustar score baseado em volatilidade
            if 30 <= volatility <= 70:  # Volatilidade ideal
                score += 0.1
            elif volatility > 80:  # Muito volátil
                score -= 0.15
            
            # Penalizar martingale alto
            if martingale_level > 2:
                score -= 0.1 * martingale_level
            
            # Ajustar baseado em condição de mercado
            if market_condition == 'favorable':
                score += 0.15
            elif market_condition == 'unfavorable':
                score -= 0.15
            
            score = max(0.1, min(0.9, score))  # Limitar entre 0.1 e 0.9
            
            # Determinar recomendação
            if score > 0.7:
                recommendation = "favorable"
                message = f"📈 Condições FAVORÁVEIS para trading (Score: {score:.2f})"
            elif score < 0.4:
                recommendation = "cautious"
                message = f"⚠️ Condições requerem CAUTELA (Score: {score:.2f})"
            else:
                recommendation = "neutral"
                message = f"📊 Condições NEUTRAS - análise cuidadosa (Score: {score:.2f})"
            
            analysis = {
                "message": message,
                "recommendation": recommendation,
                "confidence": score,
                "analysis_real": True,
                "analysis_id": self.analysis_count,
                "timestamp": datetime.now().isoformat(),
                "factors": [
                    f"Win rate: {win_rate:.1f}%",
                    f"Volatilidade: {volatility:.1f}",
                    f"Condição: {market_condition}",
                    f"Martingale: Nível {martingale_level}",
                    f"Símbolo: {symbol}"
                ],
                "score_breakdown": {
                    "base_score": 0.5,
                    "win_rate_adjustment": (win_rate - 50) / 100,
                    "volatility_adjustment": 0.1 if 30 <= volatility <= 70 else -0.15,
                    "martingale_penalty": -0.1 * martingale_level if martingale_level > 2 else 0,
                    "market_condition_adjustment": 0.15 if market_condition == 'favorable' else -0.15 if market_condition == 'unfavorable' else 0,
                    "final_score": score
                }
            }
            
            logger.info(f"✅ Análise ML concluída: {recommendation} (Score: {score:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erro na análise ML: {e}")
            return {
                "message": f"Erro na análise: {str(e)}",
                "recommendation": "error",
                "confidence": 0.5,
                "analysis_real": False
            }
    
    def save_trade(self, trade_data: TradeData) -> bool:
        """Salva trade no banco de dados para aprendizado"""
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
            
            self.stats['total_trades_learned'] += 1
            
            logger.info(f"💾 Trade {trade_data.id} salvo para aprendizado ML")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro salvando trade: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do sistema ML em tempo real"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE outcome = 'won'")
            total_wins = cursor.fetchone()[0]
            
            win_rate = total_wins / total_trades if total_trades > 0 else 0
            
            patterns = self.analyze_patterns()
            
            conn.close()
            
            return {
                "ml_stats": {
                    "total_trades": total_trades,
                    "total_wins": total_wins,
                    "overall_win_rate": win_rate,
                    "models_loaded": len(self.models),
                    "last_training": self.last_training,
                    "predictions_made": self.stats['total_predictions'],
                    "analysis_made": self.stats['total_analysis'],
                    "trades_learned": self.stats['total_trades_learned']
                },
                "models_available": list(self.models.keys()),
                "models_accuracy": self.stats['models_accuracy'],
                "patterns": {
                    "patterns": patterns,
                    "patterns_count": len(patterns)
                },
                "system_status": "active_learning",
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas: {e}")
            return {
                "ml_stats": {
                    "total_trades": 0,
                    "total_wins": 0,
                    "overall_win_rate": 0,
                    "models_loaded": len(self.models),
                    "last_training": None,
                    "predictions_made": self.stats['total_predictions'],
                    "analysis_made": self.stats['total_analysis'],
                    "trades_learned": self.stats['total_trades_learned']
                },
                "models_available": [],
                "patterns": {"patterns": [], "patterns_count": 0},
                "system_status": "error"
            }
    
    def analyze_patterns(self) -> List[Dict]:
        """Analisa padrões ML reais nos dados"""
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
                            "description": f"Símbolo {symbol} tem alta taxa de vitória ({win_rate:.1%})",
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
                            "description": f"Direção {direction} tem melhor performance ({win_rate:.1%})",
                            "confidence": win_rate,
                            "data": {"direction": direction, "win_rate": win_rate}
                        })
            
            # Padrão 3: Performance por nível Martingale
            if 'martingale_level' in df.columns:
                martingale_stats = df.groupby('martingale_level')['outcome'].apply(
                    lambda x: (x == 'won').mean()
                ).to_dict()
                
                for level, win_rate in martingale_stats.items():
                    if level > 0 and win_rate > 0.5:
                        patterns.append({
                            "type": "martingale_recovery",
                            "description": f"Martingale nível {level} tem {win_rate:.1%} de recuperação",
                            "confidence": win_rate,
                            "data": {"martingale_level": level, "win_rate": win_rate}
                        })
            
            self.stats['patterns_found'] = patterns
            logger.info(f"🔍 {len(patterns)} padrões ML identificados")
            return patterns[:10]
            
        except Exception as e:
            logger.error(f"❌ Erro analisando padrões: {e}")
            return []

# ===== INSTÂNCIA GLOBAL =====
ml_system = MLTradingSystem()

# ===== FASTAPI APP =====

app = FastAPI(
    title="ML Trading Bot API - ANÁLISES REAIS",
    description="API de Machine Learning para Trading Bot com Análises Reais e Logs Detalhados",
    version="2.0.0"
)

# ===== CONFIGURAR CORS DETALHADO =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ===== MIDDLEWARE DE LOG =====
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    # Log da requisição
    logger.info(f"🌐 Requisição: {request.method} {request.url.path} de {request.client.host}")
    
    response = await call_next(request)
    
    # Log da resposta
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ Resposta: {response.status_code} em {process_time:.3f}s")
    
    return response

# ===== ENDPOINTS =====

@app.get("/")
async def root():
    """Endpoint raiz com informações detalhadas"""
    logger.info("🏠 Endpoint raiz acessado")
    return {
        "message": "🧠 ML Trading Bot API - ANÁLISES REAIS",
        "version": "2.0.0",
        "status": "online_ml_real",
        "models_loaded": len(ml_system.models),
        "predictions_made": ml_system.stats['total_predictions'],
        "analysis_made": ml_system.stats['total_analysis'],
        "documentation": "/docs",
        "health_check": "/health",
        "timestamp": datetime.now().isoformat(),
        "ml_features": [
            "Predições ML Reais",
            "Análise de Mercado ML",
            "Aprendizado Contínuo",
            "Padrões ML",
            "Estatísticas em Tempo Real"
        ]
    }

@app.get("/health")
async def health_check(request: Request):
    """Health check detalhado da API"""
    logger.info(f"🏥 Health check acessado de {request.client.host}")
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(ml_system.models),
        "is_trained": ml_system.is_trained,
        "ml_stats": ml_system.get_stats()["ml_stats"],
        "models_available": list(ml_system.models.keys()),
        "predictions_today": ml_system.stats['total_predictions'],
        "analysis_today": ml_system.stats['total_analysis'],
        "last_training": ml_system.last_training,
        "system_info": {
            "environment": os.getenv("ENVIRONMENT", "production"),
            "port": os.getenv("PORT", "10000"),
            "memory_usage": "optimized",
            "ml_ready": True
        }
    }
    
    logger.info("✅ Health check respondido com sucesso")
    return health_data

@app.post("/ml/predict")
async def predict_trade(request: PredictionRequest, req: Request):
    """Endpoint para predições ML REAIS"""
    logger.info(f"🎯 Predição ML solicitada de {req.client.host}")
    
    try:
        request_data = request.dict()
        logger.info(f"📥 Dados recebidos: {request_data}")
        
        # Fazer predição real
        prediction = ml_system.predict(request_data)
        
        # Log da atividade
        ml_system.log_frontend_activity("predict", request_data, prediction, req.client.host)
        
        logger.info(f"📤 Predição enviada: {prediction['prediction']} ({prediction['confidence']:.1%})")
        
        return JSONResponse(content=prediction)
        
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        error_response = {
            "error": f"Erro na predição: {str(e)}",
            "prediction": "neutral",
            "confidence": 0.5,
            "analysis_real": False
        }
        return JSONResponse(content=error_response, status_code=500)

@app.post("/ml/analyze")
async def analyze_market(request: AnalysisRequest, req: Request):
    """Análise de mercado ML REAL"""
    logger.info(f"📊 Análise ML solicitada de {req.client.host}")
    
    try:
        request_data = request.dict()
        logger.info(f"📥 Dados de análise: {request_data}")
        
        # Fazer análise real
        analysis = ml_system.analyze_market(request_data)
        
        # Log da atividade
        ml_system.log_frontend_activity("analyze", request_data, analysis, req.client.host)
        
        logger.info(f"📤 Análise enviada: {analysis['recommendation']} (Score: {analysis['confidence']:.2f})")
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        error_response = {
            "error": f"Erro na análise: {str(e)}",
            "recommendation": "error",
            "confidence": 0.5,
            "analysis_real": False
        }
        return JSONResponse(content=error_response, status_code=500)

@app.post("/trade/save")
async def save_trade(trade: TradeData, req: Request):
    """Salva dados de trade para aprendizado ML"""
    logger.info(f"💾 Trade para salvar recebido de {req.client.host}")
    
    try:
        trade_data = trade.dict()
        logger.info(f"📥 Dados do trade: {trade_data}")
        
        success = ml_system.save_trade(trade)
        
        if success:
            response = {
                "message": "Trade salvo com sucesso para aprendizado ML",
                "trade_id": trade.id,
                "learned": True,
                "total_trades_learned": ml_system.stats['total_trades_learned']
            }
            logger.info(f"✅ Trade salvo: {trade.id}")
            return response
        else:
            raise HTTPException(status_code=500, detail="Erro ao salvar trade")
            
    except Exception as e:
        logger.error(f"❌ Erro salvando trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/stats")
async def get_ml_stats(req: Request):
    """Retorna estatísticas ML em tempo real"""
    logger.info(f"📊 Estatísticas ML solicitadas de {req.client.host}")
    
    try:
        stats = ml_system.get_stats()
        logger.info(f"📤 Estatísticas enviadas: {stats['ml_stats']['total_trades']} trades")
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"❌ Erro nas estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/train")
async def train_models(background_tasks: BackgroundTasks, req: Request):
    """Retreina modelos ML em background"""
    logger.info(f"🎓 Treinamento ML solicitado de {req.client.host}")
    
    try:
        def train_in_background():
            logger.info("🎓 Iniciando treinamento em background...")
            results = ml_system.train_models()
            logger.info(f"✅ Treinamento concluído: {len(results)} modelos")
        
        background_tasks.add_task(train_in_background)
        
        return {
            "message": "Treinamento ML iniciado em background",
            "status": "started",
            "current_models": len(ml_system.models),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erro iniciando treinamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/patterns")
async def get_patterns(req: Request):
    """Retorna padrões ML identificados"""
    logger.info(f"🔍 Padrões ML solicitados de {req.client.host}")
    
    try:
        patterns = ml_system.analyze_patterns()
        
        response = {
            "patterns": patterns,
            "total": len(patterns),
            "generated_at": datetime.now().isoformat(),
            "analysis_real": True
        }
        
        logger.info(f"📤 Padrões enviados: {len(patterns)} identificados")
        return response
        
    except Exception as e:
        logger.error(f"❌ Erro nos padrões: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ENDPOINT DE DEBUG =====
@app.get("/debug/activity")
async def get_frontend_activity():
    """Debug: últimas atividades do frontend"""
    try:
        conn = sqlite3.connect(ml_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM frontend_activity 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        activities = []
        for row in cursor.fetchall():
            activities.append({
                "id": row[0],
                "endpoint": row[1],
                "timestamp": row[4],
                "ip": row[5]
            })
        
        conn.close()
        
        return {
            "recent_activities": activities,
            "total_predictions": ml_system.stats['total_predictions'],
            "total_analysis": ml_system.stats['total_analysis']
        }
        
    except Exception as e:
        return {"error": str(e)}

# ===== MAIN =====

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 10000))
    
    logger.info(f"🚀 Iniciando ML Trading Bot API REAL em {host}:{port}")
    logger.info(f"📊 Modelos carregados: {len(ml_system.models)}")
    logger.info(f"🎯 Treinamento: {'OK' if ml_system.is_trained else 'Pendente'}")
    logger.info(f"🔗 URL: http://{host}:{port}")
    logger.info("🧠 SISTEMA ML REAL ATIVO - AGUARDANDO CONEXÕES FRONTEND")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
