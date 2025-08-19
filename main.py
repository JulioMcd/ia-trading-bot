#!/usr/bin/env python3
"""
API Principal do ML Trading Bot com Persistência e Aprendizado Contínuo
FastAPI + Scikit-learn para predições em tempo real com dados históricos
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
    pnl: Optional[float] = None
    market_context: Optional[Dict] = None
    martingale_level: int = 0
    volatility: float = 0.0
    trend: str = "neutral"
    ml_prediction: Optional[Dict] = None
    confidence_score: Optional[float] = None

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

class HistoryRequest(BaseModel):
    symbol: Optional[str] = None
    limit: int = 100
    days: int = 30

# ===== ML TRADING SYSTEM MELHORADO =====

class MLTradingSystem:
    """Sistema de Machine Learning para Trading com Persistência"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.last_training = None
        self.historical_stats = {}
        self.symbol_patterns = {}
        
        self.feature_columns = [
            'current_price', 'volatility', 'martingale_level', 
            'recent_win_rate', 'stake', 'duration_numeric',
            'hour_of_day', 'day_of_week', 'symbol_encoded'
        ]
        
        # Criar diretórios
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("backups").mkdir(exist_ok=True)
        
        # Inicializar banco
        self.init_database()
        
        # Carregar estatísticas históricas
        self.load_historical_stats()
        
        # Carregar modelos se existirem
        self.load_models()
        
        # Treinar se necessário
        if not self.is_trained:
            self.train_initial_models()
    
    def init_database(self):
        """Inicializa o banco de dados com tabelas melhoradas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de trades melhorada
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
                pnl REAL DEFAULT 0,
                market_context TEXT,
                martingale_level INTEGER DEFAULT 0,
                volatility REAL DEFAULT 0,
                trend TEXT DEFAULT 'neutral',
                ml_prediction TEXT,
                confidence_score REAL,
                session_id TEXT,
                trade_source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de sessões de trading
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                initial_balance REAL DEFAULT 0,
                final_balance REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de estatísticas por símbolo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_stats (
                symbol TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                best_streak INTEGER DEFAULT 0,
                worst_streak INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de padrões ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                symbol TEXT,
                conditions TEXT NOT NULL,
                success_rate REAL NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices para performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Banco de dados inicializado com tabelas melhoradas")
    
    def load_historical_stats(self):
        """Carrega estatísticas históricas do banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Estatísticas gerais
            general_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) as total_wins,
                    AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as overall_win_rate,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM trades 
                WHERE outcome IS NOT NULL
            ''', conn)
            
            if len(general_stats) > 0:
                stats = general_stats.iloc[0]
                self.historical_stats = {
                    'total_trades': int(stats['total_trades']),
                    'total_wins': int(stats['total_wins']),
                    'overall_win_rate': float(stats['overall_win_rate'] or 0),
                    'total_pnl': float(stats['total_pnl'] or 0),
                    'avg_pnl': float(stats['avg_pnl'] or 0)
                }
            
            # Estatísticas por símbolo
            symbol_stats = pd.read_sql_query('''
                SELECT 
                    symbol,
                    COUNT(*) as trades,
                    AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(pnl) as avg_pnl
                FROM trades 
                WHERE outcome IS NOT NULL
                GROUP BY symbol
            ''', conn)
            
            self.symbol_patterns = {}
            for _, row in symbol_stats.iterrows():
                self.symbol_patterns[row['symbol']] = {
                    'trades': int(row['trades']),
                    'win_rate': float(row['win_rate']),
                    'avg_pnl': float(row['avg_pnl'])
                }
            
            conn.close()
            
            logger.info(f"📊 Estatísticas históricas carregadas: {self.historical_stats}")
            logger.info(f"🎯 Padrões por símbolo: {len(self.symbol_patterns)} símbolos")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar estatísticas: {e}")
            self.historical_stats = {'total_trades': 0, 'total_wins': 0, 'overall_win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0}
            self.symbol_patterns = {}
    
    def create_enhanced_features(self, data: Dict) -> np.ndarray:
        """Cria features melhoradas incluindo dados históricos"""
        try:
            # Features básicas
            duration_str = str(data.get('duration', '5'))
            duration_numeric = float(duration_str.replace('t', '').replace('ticks', ''))
            
            # Features temporais
            timestamp = datetime.now()
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Encoding do símbolo (simples)
            symbol = data.get('symbol', 'R_50')
            symbol_encoded = hash(symbol) % 100  # Hash simples
            
            # Features históricas do símbolo
            symbol_win_rate = 0.5
            if symbol in self.symbol_patterns:
                symbol_win_rate = self.symbol_patterns[symbol]['win_rate']
            
            features = [
                float(data.get('current_price', 0)),
                float(data.get('volatility', 50)),
                int(data.get('martingale_level', 0)),
                float(data.get('recent_win_rate', symbol_win_rate)),  # Usar histórico se disponível
                float(data.get('stake', 1)),
                duration_numeric,
                hour_of_day,
                day_of_week,
                symbol_encoded
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erro ao criar features: {e}")
            return np.array([1000, 50, 0, 0.5, 1, 5, 12, 1, 50]).reshape(1, -1)
    
    def get_trade_history(self, symbol: str = None, limit: int = 1000) -> pd.DataFrame:
        """Busca histórico de trades do banco"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT * FROM trades 
                WHERE outcome IS NOT NULL
            '''
            params = []
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if len(df) == 0:
                logger.info("Criando dados sintéticos para treinamento inicial...")
                df = self.create_synthetic_data()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar histórico: {e}")
            return self.create_synthetic_data()
        finally:
            conn.close()
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Cria dados sintéticos mais realistas"""
        np.random.seed(42)
        n_samples = 300
        
        symbols = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100']
        directions = ['CALL', 'PUT']
        
        data = {
            'id': [f'synthetic_{i}' for i in range(n_samples)],
            'symbol': np.random.choice(symbols, n_samples),
            'direction': np.random.choice(directions, n_samples),
            'current_price': np.random.uniform(800, 1200, n_samples),
            'volatility': np.random.uniform(20, 80, n_samples),
            'martingale_level': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'recent_win_rate': np.random.uniform(0.3, 0.7, n_samples),
            'stake': np.random.uniform(1, 10, n_samples),
            'duration': np.random.choice([3, 5, 7, 10], n_samples),
            'pnl': np.random.uniform(-10, 10, n_samples)
        }
        
        # Criar outcomes mais realistas baseados nas features
        outcomes = []
        for i in range(n_samples):
            base_prob = data['recent_win_rate'][i]
            
            # Ajustar por volatilidade (alta volatilidade = mais difícil)
            vol_factor = 1 - (data['volatility'][i] - 20) / 120  # 0.5 a 1.0
            
            # Ajustar por martingale (níveis altos = mais arriscado)
            mart_factor = 1 - (data['martingale_level'][i] * 0.1)
            
            # Probabilidade final
            win_prob = base_prob * vol_factor * mart_factor
            win_prob = max(0.2, min(0.8, win_prob))  # Limitar entre 20% e 80%
            
            outcome = 'won' if np.random.random() < win_prob else 'lost'
            outcomes.append(outcome)
            
            # Ajustar PnL baseado no outcome
            if outcome == 'won':
                data['pnl'][i] = abs(data['pnl'][i])
            else:
                data['pnl'][i] = -abs(data['pnl'][i])
        
        data['outcome'] = outcomes
        
        return pd.DataFrame(data)
    
    def prepare_enhanced_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepara dados para treinamento com features melhoradas"""
        if len(df) < 20:
            raise ValueError("Dados insuficientes para treinamento")
        
        features_list = []
        for _, row in df.iterrows():
            # Simular timestamp se não existir
            if 'timestamp' in row and pd.notna(row['timestamp']):
                try:
                    timestamp = pd.to_datetime(row['timestamp'])
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            duration_numeric = float(str(row.get('duration', 5)).replace('t', '').replace('ticks', ''))
            
            features = [
                float(row.get('current_price', 1000)),
                float(row.get('volatility', 50)),
                int(row.get('martingale_level', 0)),
                float(row.get('recent_win_rate', 0.5)),
                float(row.get('stake', 1)),
                duration_numeric,
                timestamp.hour,
                timestamp.weekday(),
                hash(str(row.get('symbol', 'R_50'))) % 100
            ]
            features_list.append(features)
        
        X = np.array(features_list)
        y = (df['outcome'] == 'won').astype(int)
        
        return X, y
    
    def train_models(self) -> Dict:
        """Treina modelos com dados históricos"""
        logger.info("🎓 Iniciando treinamento com dados históricos...")
        
        try:
            df = self.get_trade_history()
            X, y = self.prepare_enhanced_training_data(df)
            
            logger.info(f"📊 Dados de treinamento: {len(df)} trades")
            
            if len(df) < 20:
                logger.warning("⚠️ Poucos dados para treinamento robusto")
            
            # Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Normalizar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['main'] = scaler
            
            # Configuração de modelos melhorada
            models_config = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                'svm': SVC(
                    probability=True, 
                    gamma='scale',
                    random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    alpha=0.01,
                    random_state=42,
                    max_iter=1000
                )
            }
            
            results = {}
            
            for name, model in models_config.items():
                try:
                    logger.info(f"Treinando {name}...")
                    
                    if name in ['svm', 'logistic_regression', 'neural_network']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_proba = model.predict_proba(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3)
                    cv_mean = cv_scores.mean()
                    
                    self.models[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'cv_score': cv_mean,
                        'trained_at': datetime.now().isoformat(),
                        'samples': len(X_train),
                        'features': len(X_train[0])
                    }
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'cv_score': cv_mean,
                        'samples': len(X_train)
                    }
                    
                    logger.info(f"✅ {name}: {accuracy:.3f} accuracy, {cv_mean:.3f} CV")
                    
                except Exception as e:
                    logger.error(f"❌ Erro treinando {name}: {e}")
                    continue
            
            self.save_models()
            self.is_trained = True
            self.last_training = {
                'timestamp': datetime.now().isoformat(),
                'models_trained': len(results),
                'best_accuracy': max([r['accuracy'] for r in results.values()]) if results else 0,
                'data_size': len(df)
            }
            
            logger.info(f"✅ Treinamento concluído: {len(results)} modelos")
            
            # Atualizar estatísticas após treinamento
            self.load_historical_stats()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return {}
    
    def predict_enhanced(self, request_data: Dict) -> Dict:
        """Predição melhorada com contexto histórico"""
        try:
            if not self.models:
                return {
                    "prediction": "neutral",
                    "confidence": 0.5,
                    "model_used": "none",
                    "reason": "Nenhum modelo treinado disponível",
                    "historical_context": {}
                }
            
            # Usar features melhoradas
            features = self.create_enhanced_features(request_data)
            
            # Contexto histórico do símbolo
            symbol = request_data.get('symbol', 'R_50')
            historical_context = self.symbol_patterns.get(symbol, {
                'trades': 0, 'win_rate': 0.5, 'avg_pnl': 0
            })
            
            # Usar ensemble dos melhores modelos
            predictions = []
            confidences = []
            models_used = []
            
            # Pegar os 3 melhores modelos
            sorted_models = sorted(
                self.models.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )[:3]
            
            for name, model_data in sorted_models:
                try:
                    model = model_data['model']
                    
                    # Aplicar normalização se necessário
                    if name in ['svm', 'logistic_regression', 'neural_network']:
                        if 'main' in self.scalers:
                            model_features = self.scalers['main'].transform(features)
                        else:
                            continue
                    else:
                        model_features = features
                    
                    # Predição
                    pred_proba = model.predict_proba(model_features)[0]
                    pred_binary = model.predict(model_features)[0]
                    
                    predictions.append(pred_binary)
                    confidences.append(max(pred_proba))
                    models_used.append(name)
                    
                except Exception as e:
                    logger.error(f"Erro na predição {name}: {e}")
                    continue
            
            if not predictions:
                return {
                    "prediction": "neutral",
                    "confidence": 0.5,
                    "model_used": "error",
                    "reason": "Erro em todos os modelos",
                    "historical_context": historical_context
                }
            
            # Ensemble voting
            final_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
            avg_confidence = sum(confidences) / len(confidences)
            
            # Ajustar confiança com contexto histórico
            symbol_win_rate = historical_context.get('win_rate', 0.5)
            if symbol_win_rate > 0.6:
                avg_confidence *= 1.1  # Boost para símbolos com bom histórico
            elif symbol_win_rate < 0.4:
                avg_confidence *= 0.9  # Reduzir para símbolos com histórico ruim
            
            avg_confidence = min(1.0, avg_confidence)
            
            # Interpretar resultado
            if final_prediction == 1 and avg_confidence > 0.65:
                prediction = "favor"
                reason = f"Ensemble recomenda CALL/PUT com {avg_confidence:.1%} confiança"
            elif final_prediction == 0 and avg_confidence > 0.65:
                prediction = "avoid"
                reason = f"Ensemble recomenda evitar ({avg_confidence:.1%} confiança de perda)"
            else:
                prediction = "neutral"
                reason = f"Ensemble neutro - confiança baixa ({avg_confidence:.1%})"
            
            # Adicionar contexto histórico à razão
            if historical_context['trades'] > 10:
                reason += f" | Histórico {symbol}: {symbol_win_rate:.1%} win rate em {historical_context['trades']} trades"
            
            return {
                "prediction": prediction,
                "confidence": float(avg_confidence),
                "win_probability": float(final_prediction),
                "model_used": f"ensemble_{len(models_used)}",
                "models_in_ensemble": models_used,
                "reason": reason,
                "historical_context": historical_context,
                "training_data_size": self.historical_stats.get('total_trades', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            return {
                "prediction": "neutral",
                "confidence": 0.5,
                "model_used": "error",
                "reason": f"Erro na predição: {str(e)}",
                "historical_context": {}
            }
    
    def save_trade_enhanced(self, trade_data: TradeData) -> bool:
        """Salva trade com atualizações de estatísticas"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades 
                (id, timestamp, symbol, direction, stake, duration, entry_price, 
                 exit_price, outcome, pnl, market_context, martingale_level, 
                 volatility, trend, ml_prediction, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                trade_data.pnl,
                json.dumps(trade_data.market_context) if trade_data.market_context else None,
                trade_data.martingale_level,
                trade_data.volatility,
                trade_data.trend,
                json.dumps(trade_data.ml_prediction) if trade_data.ml_prediction else None,
                trade_data.confidence_score
            ))
            
            # Atualizar estatísticas do símbolo se trade finalizado
            if trade_data.outcome:
                self.update_symbol_stats(cursor, trade_data.symbol, trade_data.outcome, trade_data.pnl)
            
            conn.commit()
            conn.close()
            
            # Recarregar estatísticas se trade finalizado
            if trade_data.outcome:
                self.load_historical_stats()
            
            logger.info(f"💾 Trade {trade_data.id} salvo e estatísticas atualizadas")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro salvando trade: {e}")
            return False
    
    def update_symbol_stats(self, cursor, symbol: str, outcome: str, pnl: float):
        """Atualiza estatísticas do símbolo"""
        cursor.execute('''
            INSERT OR IGNORE INTO symbol_stats (symbol) VALUES (?)
        ''', (symbol,))
        
        cursor.execute('''
            UPDATE symbol_stats SET
                total_trades = total_trades + 1,
                winning_trades = winning_trades + ?,
                win_rate = CAST(winning_trades AS REAL) / total_trades,
                avg_pnl = (avg_pnl * (total_trades - 1) + ?) / total_trades,
                last_updated = CURRENT_TIMESTAMP
            WHERE symbol = ?
        ''', (1 if outcome == 'won' else 0, pnl or 0, symbol))
    
    def get_trade_history_for_frontend(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Retorna histórico formatado para o frontend"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    id, timestamp, symbol, direction, stake, duration,
                    entry_price, exit_price, outcome, pnl, martingale_level,
                    ml_prediction, confidence_score
                FROM trades 
                WHERE outcome IS NOT NULL
            '''
            params = []
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            trades = []
            
            for row in cursor.fetchall():
                trade = dict(zip(columns, row))
                
                # Parse JSON fields
                if trade['ml_prediction']:
                    try:
                        trade['ml_prediction'] = json.loads(trade['ml_prediction'])
                    except:
                        trade['ml_prediction'] = None
                
                trades.append(trade)
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"❌ Erro buscando histórico: {e}")
            return []
    
    def get_enhanced_stats(self) -> Dict:
        """Retorna estatísticas completas incluindo histórico"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Estatísticas gerais
            general = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) as total_wins,
                    AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades 
                WHERE outcome IS NOT NULL
            ''', conn)
            
            # Estatísticas por símbolo
            by_symbol = pd.read_sql_query('''
                SELECT 
                    symbol,
                    COUNT(*) as trades,
                    AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(pnl) as total_pnl
                FROM trades 
                WHERE outcome IS NOT NULL
                GROUP BY symbol
                ORDER BY trades DESC
            ''', conn)
            
            # Estatísticas recentes (últimos 7 dias)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            recent = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as trades_7d,
                    AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate_7d,
                    SUM(pnl) as pnl_7d
                FROM trades 
                WHERE outcome IS NOT NULL AND timestamp > ?
            ''', conn, params=[week_ago])
            
            # Performance dos modelos ML
            ml_trades = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as ml_influenced_trades,
                    AVG(CASE WHEN outcome = 'won' THEN 1.0 ELSE 0.0 END) as ml_win_rate,
                    AVG(confidence_score) as avg_confidence
                FROM trades 
                WHERE outcome IS NOT NULL AND ml_prediction IS NOT NULL
            ''', conn)
            
            conn.close()
            
            # Montar resposta
            stats = {
                "ml_stats": {
                    "total_trades": int(general.iloc[0]['total_trades']) if len(general) > 0 else 0,
                    "total_wins": int(general.iloc[0]['total_wins']) if len(general) > 0 else 0,
                    "overall_win_rate": float(general.iloc[0]['win_rate'] or 0) if len(general) > 0 else 0,
                    "total_pnl": float(general.iloc[0]['total_pnl'] or 0) if len(general) > 0 else 0,
                    "avg_pnl": float(general.iloc[0]['avg_pnl'] or 0) if len(general) > 0 else 0,
                    "best_trade": float(general.iloc[0]['best_trade'] or 0) if len(general) > 0 else 0,
                    "worst_trade": float(general.iloc[0]['worst_trade'] or 0) if len(general) > 0 else 0,
                    "models_loaded": len(self.models),
                    "last_training": self.last_training,
                    "is_trained": self.is_trained
                },
                "recent_performance": {
                    "trades_7d": int(recent.iloc[0]['trades_7d']) if len(recent) > 0 else 0,
                    "win_rate_7d": float(recent.iloc[0]['win_rate_7d'] or 0) if len(recent) > 0 else 0,
                    "pnl_7d": float(recent.iloc[0]['pnl_7d'] or 0) if len(recent) > 0 else 0
                },
                "ml_performance": {
                    "ml_influenced_trades": int(ml_trades.iloc[0]['ml_influenced_trades']) if len(ml_trades) > 0 else 0,
                    "ml_win_rate": float(ml_trades.iloc[0]['ml_win_rate'] or 0) if len(ml_trades) > 0 else 0,
                    "avg_confidence": float(ml_trades.iloc[0]['avg_confidence'] or 0) if len(ml_trades) > 0 else 0
                },
                "symbol_performance": by_symbol.to_dict('records') if len(by_symbol) > 0 else [],
                "models_available": list(self.models.keys()),
                "patterns": {
                    "patterns": self.analyze_enhanced_patterns(),
                    "patterns_count": len(self.analyze_enhanced_patterns())
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas: {e}")
            return self.get_basic_stats()
    
    def get_basic_stats(self) -> Dict:
        """Estatísticas básicas em caso de erro"""
        return {
            "ml_stats": {
                "total_trades": self.historical_stats.get('total_trades', 0),
                "total_wins": self.historical_stats.get('total_wins', 0),
                "overall_win_rate": self.historical_stats.get('overall_win_rate', 0),
                "total_pnl": self.historical_stats.get('total_pnl', 0),
                "models_loaded": len(self.models),
                "last_training": self.last_training,
                "is_trained": self.is_trained
            },
            "models_available": list(self.models.keys()),
            "patterns": {"patterns": [], "patterns_count": 0}
        }
    
    def analyze_enhanced_patterns(self) -> List[Dict]:
        """Análise de padrões melhorada"""
        try:
            df = self.get_trade_history(limit=500)
            if len(df) < 20:
                return []
            
            patterns = []
            
            # Padrão 1: Performance por símbolo
            symbol_stats = df.groupby('symbol').agg({
                'outcome': lambda x: (x == 'won').mean(),
                'pnl': 'mean',
                'id': 'count'
            }).rename(columns={'outcome': 'win_rate', 'id': 'count'})
            
            for symbol, stats in symbol_stats.iterrows():
                if stats['count'] >= 10 and stats['win_rate'] > 0.6:
                    patterns.append({
                        "type": "symbol_performance",
                        "description": f"Símbolo {symbol} tem alta performance ({stats['win_rate']:.1%})",
                        "confidence": stats['win_rate'],
                        "data": {
                            "symbol": symbol, 
                            "win_rate": stats['win_rate'],
                            "avg_pnl": stats['pnl'],
                            "trades": stats['count']
                        }
                    })
            
            # Padrão 2: Performance por horário
            df['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
            hourly_stats = df.groupby('hour')['outcome'].apply(lambda x: (x == 'won').mean())
            
            best_hours = hourly_stats[hourly_stats > 0.65].index.tolist()
            if best_hours:
                patterns.append({
                    "type": "time_pattern",
                    "description": f"Melhor performance nos horários: {best_hours}",
                    "confidence": hourly_stats[best_hours].mean(),
                    "data": {"best_hours": best_hours}
                })
            
            # Padrão 3: Performance por nível de Martingale
            if 'martingale_level' in df.columns:
                mart_stats = df.groupby('martingale_level')['outcome'].apply(lambda x: (x == 'won').mean())
                
                if 0 in mart_stats.index and mart_stats[0] > 0.6:
                    patterns.append({
                        "type": "martingale_pattern",
                        "description": f"Trades sem Martingale têm melhor performance ({mart_stats[0]:.1%})",
                        "confidence": mart_stats[0],
                        "data": {"level_0_win_rate": mart_stats[0]}
                    })
            
            # Padrão 4: Performance de ML
            ml_trades = df[df['ml_prediction'].notna()]
            if len(ml_trades) > 10:
                ml_win_rate = (ml_trades['outcome'] == 'won').mean()
                manual_trades = df[df['ml_prediction'].isna()]
                manual_win_rate = (manual_trades['outcome'] == 'won').mean() if len(manual_trades) > 0 else 0.5
                
                if ml_win_rate > manual_win_rate + 0.05:  # 5% melhor
                    patterns.append({
                        "type": "ml_advantage",
                        "description": f"Trades com ML têm melhor performance ({ml_win_rate:.1%} vs {manual_win_rate:.1%})",
                        "confidence": ml_win_rate,
                        "data": {
                            "ml_win_rate": ml_win_rate,
                            "manual_win_rate": manual_win_rate,
                            "ml_trades": len(ml_trades)
                        }
                    })
            
            return patterns[:10]
            
        except Exception as e:
            logger.error(f"❌ Erro analisando padrões: {e}")
            return []
    
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
    
    def train_initial_models(self):
        """Treina modelos iniciais"""
        logger.info("🎯 Treinamento inicial...")
        self.train_models()

# ===== INSTÂNCIA GLOBAL =====
ml_system = MLTradingSystem()

# ===== FASTAPI APP =====

app = FastAPI(
    title="ML Trading Bot API Enhanced",
    description="API de Machine Learning para Trading Bot com Persistência e Aprendizado Contínuo",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS MELHORADOS =====

@app.get("/")
async def root():
    return {
        "message": "🧠 ML Trading Bot API Enhanced",
        "version": "2.0.0",
        "status": "online",
        "models_loaded": len(ml_system.models),
        "historical_trades": ml_system.historical_stats.get('total_trades', 0),
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(ml_system.models),
        "is_trained": ml_system.is_trained,
        "historical_data": ml_system.historical_stats,
        "ml_stats": ml_system.get_enhanced_stats()["ml_stats"]
    }

@app.post("/ml/predict")
async def predict_trade(request: PredictionRequest):
    try:
        request_data = request.dict()
        prediction = ml_system.predict_enhanced(request_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/save")
async def save_trade(trade: TradeData):
    try:
        success = ml_system.save_trade_enhanced(trade)
        if success:
            return {"message": "Trade salvo com sucesso", "trade_id": trade.id}
        else:
            raise HTTPException(status_code=500, detail="Erro ao salvar trade")
    except Exception as e:
        logger.error(f"❌ Erro salvando trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/stats")
async def get_ml_stats():
    try:
        stats = ml_system.get_enhanced_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"❌ Erro nas estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trades/history")
async def get_trade_history(request: HistoryRequest):
    """Endpoint para buscar histórico de trades"""
    try:
        trades = ml_system.get_trade_history_for_frontend(
            symbol=request.symbol,
            limit=request.limit
        )
        
        return {
            "trades": trades,
            "total": len(trades),
            "symbol_filter": request.symbol,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Erro buscando histórico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/train")
async def train_models(background_tasks: BackgroundTasks):
    try:
        def train_in_background():
            logger.info("🎓 Iniciando retreinamento com dados históricos...")
            results = ml_system.train_models()
            logger.info(f"✅ Retreinamento concluído: {len(results)} modelos")
        
        background_tasks.add_task(train_in_background)
        
        return {
            "message": "Retreinamento iniciado com dados históricos",
            "status": "started",
            "current_data_size": ml_system.historical_stats.get('total_trades', 0)
        }
    except Exception as e:
        logger.error(f"❌ Erro iniciando treinamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/analyze")
async def analyze_market(request: AnalysisRequest):
    try:
        # Análise melhorada com contexto histórico
        symbol_stats = ml_system.symbol_patterns.get(request.symbol, {})
        overall_stats = ml_system.historical_stats
        
        analysis = {
            "message": "Análise de mercado com contexto histórico concluída",
            "recommendation": "neutral",
            "confidence": 0.7,
            "factors": [
                f"Win rate atual: {request.win_rate:.1f}%",
                f"Volatilidade: {request.volatility:.1f}",
                f"Condição do mercado: {request.market_condition}",
                f"Dados históricos: {overall_stats.get('total_trades', 0)} trades"
            ],
            "historical_context": {
                "symbol_stats": symbol_stats,
                "overall_stats": overall_stats
            }
        }
        
        # Lógica melhorada baseada em histórico
        symbol_win_rate = symbol_stats.get('win_rate', 0.5)
        overall_win_rate = overall_stats.get('overall_win_rate', 0.5)
        
        # Combinar dados atuais com histórico
        combined_win_rate = (request.win_rate / 100 + symbol_win_rate + overall_win_rate) / 3
        
        if combined_win_rate > 0.6 and request.volatility < 50:
            analysis["recommendation"] = "favorable"
            analysis["message"] = "Condições favoráveis baseadas em histórico e dados atuais"
            analysis["confidence"] = min(0.9, combined_win_rate + 0.1)
        elif combined_win_rate < 0.4 or request.volatility > 80:
            analysis["recommendation"] = "cautious"
            analysis["message"] = "Dados históricos sugerem cautela"
            analysis["confidence"] = max(0.3, 1 - combined_win_rate)
        
        return JSONResponse(content=analysis)
    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/patterns")
async def get_patterns():
    try:
        patterns = ml_system.analyze_enhanced_patterns()
        return {
            "patterns": patterns,
            "total": len(patterns),
            "historical_data_size": ml_system.historical_stats.get('total_trades', 0),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Erro nos padrões: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/backup")
async def create_backup():
    """Cria backup do banco de dados"""
    try:
        backup_path = f"backups/trading_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        # Copiar banco atual
        import shutil
        shutil.copy2(ml_system.db_path, backup_path)
        
        return {
            "message": "Backup criado com sucesso",
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Erro criando backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Iniciando ML Trading Bot API Enhanced em {host}:{port}")
    logger.info(f"📊 Modelos carregados: {len(ml_system.models)}")
    logger.info(f"📈 Dados históricos: {ml_system.historical_stats.get('total_trades', 0)} trades")
    logger.info(f"🎯 Treinamento: {'OK' if ml_system.is_trained else 'Pendente'}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
