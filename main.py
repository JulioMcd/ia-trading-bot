#!/usr/bin/env python3
"""
Sistema de Scalping Automatizado com IA - Otimizado para Deriv
Lógica específica dos índices sintéticos + Machine Learning avançado
"""

import os
import json
import sqlite3
import logging
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import time
import math
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.cluster import KMeans
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== ANÁLISE ESPECÍFICA DA DERIV =====

class DerivSyntheticAnalyzer:
    """Analisador especializado nos índices sintéticos da Deriv"""
    
    def __init__(self):
        # Configurações específicas por índice sintético
        self.synthetic_configs = {
            'R_10': {
                'volatility_base': 10,
                'jump_probability': 0.05,
                'reversal_tendency': 0.7,
                'momentum_decay': 0.8,
                'optimal_duration': 60,  # segundos
                'scalp_threshold': 0.15  # %
            },
            'R_25': {
                'volatility_base': 25,
                'jump_probability': 0.08,
                'reversal_tendency': 0.65,
                'momentum_decay': 0.75,
                'optimal_duration': 90,
                'scalp_threshold': 0.25
            },
            'R_50': {
                'volatility_base': 50,
                'jump_probability': 0.12,
                'reversal_tendency': 0.6,
                'momentum_decay': 0.7,
                'optimal_duration': 120,
                'scalp_threshold': 0.4
            },
            'R_75': {
                'volatility_base': 75,
                'jump_probability': 0.15,
                'reversal_tendency': 0.55,
                'momentum_decay': 0.65,
                'optimal_duration': 150,
                'scalp_threshold': 0.6
            },
            'R_100': {
                'volatility_base': 100,
                'jump_probability': 0.18,
                'reversal_tendency': 0.5,
                'momentum_decay': 0.6,
                'optimal_duration': 180,
                'scalp_threshold': 0.8
            }
        }
        
        self.price_memory = {}
        self.pattern_cache = {}
        self.volatility_cycles = {}
        
    def detect_deriv_jump_pattern(self, symbol: str, prices: List[float]) -> Dict:
        """Detecta padrões de jump específicos da Deriv"""
        if len(prices) < 10:
            return {'jump_detected': False, 'direction': 0, 'magnitude': 0}
        
        config = self.synthetic_configs.get(symbol, self.synthetic_configs['R_50'])
        
        # Calcular mudanças percentuais
        price_changes = np.diff(prices) / prices[:-1] * 100
        recent_changes = price_changes[-5:]  # Últimas 5 mudanças
        
        # Detectar jumps baseado na volatilidade do índice
        jump_threshold = config['volatility_base'] * 0.02  # 2% da volatilidade base
        
        significant_moves = np.abs(recent_changes) > jump_threshold
        if np.sum(significant_moves) >= 2:  # 2 ou mais jumps recentes
            # Direção predominante
            direction = np.sign(np.sum(recent_changes[significant_moves]))
            magnitude = np.mean(np.abs(recent_changes[significant_moves]))
            
            # Probabilidade de reversão baseada na tendência histórica
            reversal_prob = config['reversal_tendency']
            if magnitude > jump_threshold * 2:  # Jump muito grande
                reversal_prob += 0.2
            
            return {
                'jump_detected': True,
                'direction': int(direction),
                'magnitude': magnitude,
                'reversal_probability': reversal_prob,
                'suggested_action': 'reversal' if reversal_prob > 0.6 else 'momentum'
            }
        
        return {'jump_detected': False, 'direction': 0, 'magnitude': 0}
    
    def analyze_synthetic_cycle(self, symbol: str, prices: List[float], timestamps: List[datetime]) -> Dict:
        """Analisa o ciclo de volatilidade sintética"""
        if len(prices) < 20:
            return {'cycle_phase': 'unknown', 'confidence': 0.0}
        
        config = self.synthetic_configs.get(symbol, self.synthetic_configs['R_50'])
        
        # Calcular volatilidade realizada em janelas
        window_size = 10
        volatilities = []
        
        for i in range(window_size, len(prices)):
            window_prices = prices[i-window_size:i]
            returns = np.diff(np.log(window_prices))
            vol = np.std(returns) * np.sqrt(len(returns)) * 100
            volatilities.append(vol)
        
        if not volatilities:
            return {'cycle_phase': 'unknown', 'confidence': 0.0}
        
        current_vol = volatilities[-1]
        avg_vol = np.mean(volatilities)
        vol_trend = volatilities[-3:] if len(volatilities) >= 3 else [current_vol]
        
        # Identificar fase do ciclo
        if current_vol < avg_vol * 0.7:
            cycle_phase = 'low_volatility'
            # Em baixa volatilidade, aguardar breakout
            scalp_opportunity = 0.3
            strategy = 'wait_for_breakout'
        elif current_vol > avg_vol * 1.3:
            cycle_phase = 'high_volatility'
            # Alta volatilidade = oportunidade de scalp
            scalp_opportunity = 0.9
            strategy = 'aggressive_scalp'
        elif np.mean(vol_trend) > current_vol:
            cycle_phase = 'volatility_declining'
            # Volatilidade declinando = possível final de movimento
            scalp_opportunity = 0.6
            strategy = 'counter_trend'
        else:
            cycle_phase = 'volatility_rising'
            # Volatilidade subindo = seguir tendência
            scalp_opportunity = 0.8
            strategy = 'trend_follow'
        
        return {
            'cycle_phase': cycle_phase,
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'scalp_opportunity': scalp_opportunity,
            'strategy': strategy,
            'confidence': min(0.95, len(volatilities) / 20)
        }
    
    def calculate_optimal_entry(self, symbol: str, current_price: float, pattern_data: Dict) -> Dict:
        """Calcula ponto de entrada ótimo para scalping"""
        config = self.synthetic_configs.get(symbol, self.synthetic_configs['R_50'])
        
        # Calcular níveis de suporte/resistência dinâmicos
        volatility_range = current_price * (config['volatility_base'] / 10000)
        
        entry_data = {
            'optimal_entry': current_price,
            'stop_loss': current_price - (volatility_range * 2),
            'take_profit_1': current_price + (volatility_range * 1.5),  # Primeiro alvo
            'take_profit_2': current_price + (volatility_range * 3),    # Segundo alvo
            'duration_seconds': config['optimal_duration'],
            'confidence': pattern_data.get('confidence', 0.5)
        }
        
        # Ajustar baseado no padrão detectado
        if pattern_data.get('strategy') == 'aggressive_scalp':
            entry_data['take_profit_1'] = current_price + (volatility_range * 1.2)
            entry_data['duration_seconds'] = config['optimal_duration'] // 2
            
        elif pattern_data.get('strategy') == 'counter_trend':
            entry_data['stop_loss'] = current_price - (volatility_range * 1.5)
            entry_data['take_profit_1'] = current_price + (volatility_range * 2)
            
        return entry_data

class AdvancedMLScalper:
    """Sistema ML avançado para scalping com ensemble de modelos"""
    
    def __init__(self, db_path: str = "data/advanced_scalping.db"):
        self.db_path = db_path
        self.deriv_analyzer = DerivSyntheticAnalyzer()
        
        # Ensemble de modelos especializados
        self.entry_ensemble = None
        self.exit_ensemble = None
        self.risk_classifier = None
        self.pattern_clusterer = None
        
        # Scalers especializados
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.risk_scaler = RobustScaler()
        
        # Memória de padrões e performance
        self.pattern_memory = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        self.recent_trades = {}
        
        # Métricas de performance em tempo real
        self.live_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'avg_trade_duration': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Configurações adaptativas
        self.adaptive_config = {
            'confidence_threshold': 0.65,
            'risk_multiplier': 1.0,
            'scalp_aggressiveness': 0.7,
            'learning_rate': 0.1,
            'pattern_weight_decay': 0.95
        }
        
        Path("data").mkdir(exist_ok=True)
        self.init_advanced_database()
        self.load_models()
        
    def init_advanced_database(self):
        """Inicializa database com tabelas avançadas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de padrões detectados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                features TEXT NOT NULL,
                confidence REAL NOT NULL,
                market_context TEXT,
                outcome TEXT,
                success_rate REAL DEFAULT 0.0,
                pattern_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de trades com análise detalhada
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL DEFAULT 0,
                direction TEXT NOT NULL,
                stake REAL NOT NULL,
                duration_planned INTEGER,
                duration_actual INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0,
                pnl_percentage REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                entry_confidence REAL DEFAULT 0,
                exit_reason TEXT,
                market_pattern TEXT,
                risk_score REAL DEFAULT 0,
                volatility_at_entry REAL DEFAULT 0,
                model_predictions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de performance por padrão
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_performance (
                pattern_type TEXT PRIMARY KEY,
                total_occurrences INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_pnl REAL DEFAULT 0.0,
                avg_duration REAL DEFAULT 0.0,
                risk_adjusted_return REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Advanced database initialized")
    
    def create_advanced_features(self, symbol: str, prices: List[float], timestamps: List[datetime] = None) -> np.ndarray:
        """Cria features avançadas para ML"""
        try:
            if len(prices) < 20:
                # Features mínimas se dados insuficientes
                return np.zeros(25).reshape(1, -1)
            
            features = []
            
            # Features básicas de preço
            current_price = prices[-1]
            price_change_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            price_change_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            price_change_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 else 0
            
            # Features de volatilidade
            returns = np.diff(np.log(prices[-20:]))
            current_vol = np.std(returns) * 100
            vol_ma_5 = np.std(np.diff(np.log(prices[-5:]))) * 100 if len(prices) > 5 else current_vol
            vol_ratio = current_vol / vol_ma_5 if vol_ma_5 > 0 else 1.0
            
            # Features de momentum
            momentum_3 = np.mean(np.diff(prices[-3:])) if len(prices) > 3 else 0
            momentum_10 = np.mean(np.diff(prices[-10:])) if len(prices) > 10 else 0
            rsi = self.calculate_rsi(prices[-14:]) if len(prices) > 14 else 50
            
            # Features de tendência
            trend_short = np.polyfit(range(5), prices[-5:], 1)[0] if len(prices) >= 5 else 0
            trend_long = np.polyfit(range(20), prices[-20:], 1)[0] if len(prices) >= 20 else 0
            trend_strength = abs(trend_short) / (np.std(prices[-5:]) + 1e-8) if len(prices) >= 5 else 0
            
            # Features de suporte/resistência
            recent_high = max(prices[-10:]) if len(prices) >= 10 else current_price
            recent_low = min(prices[-10:]) if len(prices) >= 10 else current_price
            price_position = (current_price - recent_low) / (recent_high - recent_low + 1e-8)
            
            # Features específicas da Deriv
            jump_info = self.deriv_analyzer.detect_deriv_jump_pattern(symbol, prices)
            cycle_info = self.deriv_analyzer.analyze_synthetic_cycle(symbol, prices, timestamps or [])
            
            jump_magnitude = jump_info.get('magnitude', 0)
            jump_direction = jump_info.get('direction', 0)
            reversal_prob = jump_info.get('reversal_probability', 0.5)
            volatility_cycle = cycle_info.get('scalp_opportunity', 0.5)
            
            # Features temporais
            if timestamps:
                current_time = timestamps[-1]
                hour_sin = np.sin(2 * np.pi * current_time.hour / 24)
                hour_cos = np.cos(2 * np.pi * current_time.hour / 24)
                day_of_week = current_time.weekday() / 6
                minute_of_hour = current_time.minute / 59
            else:
                hour_sin = hour_cos = day_of_week = minute_of_hour = 0
            
            # Compilar features
            features = [
                current_price,           # 0
                price_change_1,         # 1
                price_change_5,         # 2
                price_change_10,        # 3
                current_vol,            # 4
                vol_ratio,              # 5
                momentum_3,             # 6
                momentum_10,            # 7
                rsi / 100,              # 8 - normalizado
                trend_short,            # 9
                trend_long,             # 10
                trend_strength,         # 11
                price_position,         # 12
                jump_magnitude,         # 13
                jump_direction,         # 14
                reversal_prob,          # 15
                volatility_cycle,       # 16
                hour_sin,               # 17
                hour_cos,               # 18
                day_of_week,            # 19
                minute_of_hour,         # 20
                len(prices) / 100,      # 21 - quantidade de dados
                np.mean(prices[-5:]),   # 22 - média recente
                np.std(prices[-5:]),    # 23 - desvio recente
                self.get_market_sentiment(prices)  # 24 - sentimento
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return np.zeros(25).reshape(1, -1)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
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
    
    def get_market_sentiment(self, prices: List[float]) -> float:
        """Calcula sentimento do mercado baseado em movimentos recentes"""
        if len(prices) < 10:
            return 0.5
        
        recent_moves = np.diff(prices[-10:])
        positive_moves = np.sum(recent_moves > 0)
        total_moves = len(recent_moves)
        
        sentiment = positive_moves / total_moves
        return sentiment
    
    def train_ensemble_models(self, training_data: pd.DataFrame) -> bool:
        """Treina ensemble de modelos especializados"""
        try:
            if len(training_data) < 100:
                logger.warning("Insufficient data for ensemble training")
                return False
            
            logger.info(f"Training ensemble with {len(training_data)} samples")
            
            # Preparar dados
            features_list = []
            entry_targets = []
            exit_targets = []
            risk_targets = []
            
            for _, row in training_data.iterrows():
                try:
                    # Criar features simuladas (em produção viriam do histórico real)
                    mock_prices = [row.get('entry_price', 1000) + np.random.normal(0, 5, 20)]
                    features = self.create_advanced_features(
                        row.get('symbol', 'R_50'),
                        mock_prices[0]
                    ).flatten()
                    
                    features_list.append(features)
                    
                    # Targets
                    pnl = row.get('pnl', 0)
                    entry_success = 1 if pnl > 0 else 0
                    exit_timing = min(1.0, row.get('duration_actual', 60) / 300)  # Normalizar 0-1
                    risk_level = 0 if abs(pnl) < 5 else 1 if abs(pnl) < 15 else 2  # Low, Medium, High
                    
                    entry_targets.append(entry_success)
                    exit_targets.append(exit_timing)
                    risk_targets.append(risk_level)
                    
                except Exception as e:
                    logger.warning(f"Error processing training sample: {e}")
                    continue
            
            if len(features_list) < 50:
                logger.warning("Too few valid samples for training")
                return False
            
            X = np.array(features_list)
            y_entry = np.array(entry_targets)
            y_exit = np.array(exit_targets)
            y_risk = np.array(risk_targets)
            
            # Normalizar features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Treinar modelo de entrada (Ensemble)
            entry_models = [
                GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                RandomForestClassifier(n_estimators=100, random_state=42),
                LogisticRegression(random_state=42, max_iter=1000)
            ]
            
            self.entry_ensemble = VotingClassifier(
                estimators=[
                    ('gb', entry_models[0]),
                    ('rf', entry_models[1]),
                    ('lr', entry_models[2])
                ],
                voting='soft'
            )
            
            self.entry_ensemble.fit(X_scaled, y_entry)
            
            # Treinar modelo de risco
            self.risk_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.risk_classifier.fit(X_scaled, y_risk)
            
            # Treinar clustering para padrões
            self.pattern_clusterer = KMeans(n_clusters=5, random_state=42)
            self.pattern_clusterer.fit(X_scaled)
            
            # Validação cruzada
            entry_scores = cross_val_score(self.entry_ensemble, X_scaled, y_entry, cv=5)
            risk_scores = cross_val_score(self.risk_classifier, X_scaled, y_risk, cv=5)
            
            logger.info(f"Entry model CV score: {entry_scores.mean():.3f} (+/- {entry_scores.std() * 2:.3f})")
            logger.info(f"Risk model CV score: {risk_scores.mean():.3f} (+/- {risk_scores.std() * 2:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            return False
    
    def make_intelligent_decision(self, symbol: str, prices: List[float], timestamps: List[datetime] = None) -> Dict:
        """Toma decisão inteligente usando ensemble de modelos"""
        try:
            # Criar features
            features = self.create_advanced_features(symbol, prices, timestamps)
            
            # Análise específica da Deriv
            jump_pattern = self.deriv_analyzer.detect_deriv_jump_pattern(symbol, prices)
            cycle_analysis = self.deriv_analyzer.analyze_synthetic_cycle(symbol, prices, timestamps or [])
            entry_analysis = self.deriv_analyzer.calculate_optimal_entry(
                symbol, prices[-1], cycle_analysis
            )
            
            # Predições dos modelos
            entry_confidence = 0.5
            risk_level = 1
            pattern_cluster = 0
            
            if self.entry_ensemble and self.risk_classifier:
                try:
                    features_scaled = self.feature_scaler.transform(features)
                    
                    # Predição de entrada
                    entry_proba = self.entry_ensemble.predict_proba(features_scaled)[0]
                    entry_confidence = max(entry_proba)
                    should_enter = entry_proba[1] > entry_proba[0]  # Probabilidade de sucesso > fracasso
                    
                    # Predição de risco
                    risk_proba = self.risk_classifier.predict_proba(features_scaled)[0]
                    risk_level = np.argmax(risk_proba)
                    
                    # Cluster do padrão
                    pattern_cluster = self.pattern_clusterer.predict(features_scaled)[0]
                    
                except Exception as e:
                    logger.warning(f"Model prediction error: {e}")
                    should_enter = False
            else:
                should_enter = False
            
            # Lógica híbrida: ML + Regras específicas da Deriv
            decision_factors = {
                'ml_confidence': entry_confidence,
                'jump_detected': jump_pattern.get('jump_detected', False),
                'jump_reversal_prob': jump_pattern.get('reversal_probability', 0.5),
                'volatility_opportunity': cycle_analysis.get('scalp_opportunity', 0.5),
                'strategy_type': cycle_analysis.get('strategy', 'wait'),
                'risk_level': risk_level,
                'pattern_cluster': pattern_cluster
            }
            
            # Calcular confiança final
            base_confidence = entry_confidence
            
            # Boost baseado em padrões da Deriv
            if jump_pattern.get('jump_detected'):
                base_confidence += 0.15
                
            if cycle_analysis.get('scalp_opportunity', 0) > 0.7:
                base_confidence += 0.1
                
            if cycle_analysis.get('strategy') == 'aggressive_scalp':
                base_confidence += 0.2
                
            # Penalidade por alto risco
            if risk_level == 2:  # Alto risco
                base_confidence -= 0.2
            
            final_confidence = min(0.95, max(0.05, base_confidence))
            
            # Determinar ação
            action = 'hold'
            direction = None
            
            if (should_enter and 
                final_confidence > self.adaptive_config['confidence_threshold'] and
                cycle_analysis.get('strategy') != 'wait_for_breakout'):
                
                # Determinar direção baseada em análise híbrida
                if jump_pattern.get('jump_detected'):
                    if jump_pattern.get('suggested_action') == 'reversal':
                        direction = 'put' if jump_pattern.get('direction') > 0 else 'call'
                    else:
                        direction = 'call' if jump_pattern.get('direction') > 0 else 'put'
                else:
                    # Usar momentum
                    momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
                    direction = 'call' if momentum > 0 else 'put'
                
                action = f'buy_{direction}'
            
            # Calcular parâmetros do trade
            current_price = prices[-1]
            stake = self.calculate_optimal_stake(final_confidence, risk_level)
            duration = entry_analysis.get('duration_seconds', 120)
            
            # Ajustar duração baseada no padrão
            if cycle_analysis.get('strategy') == 'aggressive_scalp':
                duration = min(duration, 90)
            
            decision = {
                'action': action,
                'direction': direction,
                'confidence': final_confidence,
                'entry_price': current_price,
                'stake': stake,
                'duration': duration,
                'take_profit': entry_analysis.get('take_profit_1', current_price * 1.01),
                'stop_loss': entry_analysis.get('stop_loss', current_price * 0.99),
                'reasoning': self.generate_reasoning(decision_factors, cycle_analysis),
                'risk_score': risk_level / 2,  # Normalizar 0-1
                'pattern_detected': cycle_analysis.get('cycle_phase', 'unknown'),
                'deriv_factors': {
                    'jump_info': jump_pattern,
                    'cycle_info': cycle_analysis,
                    'entry_info': entry_analysis
                }
            }
            
            # Salvar decisão para aprendizado
            self.record_decision(symbol, decision, features.flatten())
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making intelligent decision: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}',
                'risk_score': 1.0
            }
    
    def calculate_optimal_stake(self, confidence: float, risk_level: int) -> float:
        """Calcula stake ótimo baseado em confiança e risco"""
        base_stake = 5.0  # Stake base
        
        # Ajustar por confiança
        confidence_multiplier = min(2.0, max(0.2, confidence * 2))
        
        # Ajustar por risco
        risk_multipliers = [1.2, 1.0, 0.6]  # Low, Medium, High risk
        risk_multiplier = risk_multipliers[min(risk_level, 2)]
        
        # Ajustar por performance recente
        performance_multiplier = 1.0
        if self.live_metrics['win_rate'] > 0.7:
            performance_multiplier = 1.3
        elif self.live_metrics['win_rate'] < 0.4:
            performance_multiplier = 0.7
        
        stake = base_stake * confidence_multiplier * risk_multiplier * performance_multiplier
        
        return max(1.0, min(20.0, stake))  # Entre $1 e $20
    
    def generate_reasoning(self, factors: Dict, cycle_analysis: Dict) -> str:
        """Gera explicação da decisão"""
        reasons = []
        
        if factors['ml_confidence'] > 0.7:
            reasons.append(f"Alta confiança ML ({factors['ml_confidence']:.2f})")
        
        if factors['jump_detected']:
            reasons.append(f"Jump detectado (reversão: {factors['jump_reversal_prob']:.2f})")
        
        strategy = cycle_analysis.get('strategy', 'unknown')
        if strategy != 'wait':
            reasons.append(f"Estratégia: {strategy}")
        
        volatility_opp = factors.get('volatility_opportunity', 0)
        if volatility_opp > 0.7:
            reasons.append("Alta oportunidade de volatilidade")
        
        risk_level = factors.get('risk_level', 1)
        risk_labels = ['Baixo', 'Médio', 'Alto']
        reasons.append(f"Risco: {risk_labels[min(risk_level, 2)]}")
        
        return " | ".join(reasons) if reasons else "Análise padrão"
    
    def record_decision(self, symbol: str, decision: Dict, features: np.ndarray):
        """Registra decisão para aprendizado futuro"""
        try:
            pattern_hash = hash(tuple(features.round(4)))  # Hash das features
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO detected_patterns
                (timestamp, symbol, pattern_type, features, confidence, market_context, pattern_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                decision.get('pattern_detected', 'unknown'),
                json.dumps(features.tolist()),
                decision.get('confidence', 0),
                json.dumps(decision.get('deriv_factors', {})),
                str(pattern_hash)
            ))
            
            conn.commit()
            conn.close()
            
            # Adicionar à memória de padrões
            self.pattern_memory.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'decision': decision,
                'features': features,
                'pattern_hash': pattern_hash
            })
            
        except Exception as e:
            logger.error(f"Error recording decision: {e}")
    
    def update_performance_metrics(self, trade_result: Dict):
        """Atualiza métricas de performance em tempo real"""
        try:
            pnl = trade_result.get('pnl', 0)
            duration = trade_result.get('duration_actual', 0)
            
            self.live_metrics['total_trades'] += 1
            
            if pnl > 0:
                self.live_metrics['winning_trades'] += 1
            
            self.live_metrics['total_pnl'] += pnl
            
            # Calcular win rate
            if self.live_metrics['total_trades'] > 0:
                self.live_metrics['win_rate'] = (
                    self.live_metrics['winning_trades'] / 
                    self.live_metrics['total_trades']
                )
            
            # Adicionar ao histórico
            self.performance_history.append({
                'timestamp': datetime.now(),
                'pnl': pnl,
                'duration': duration,
                'cumulative_pnl': self.live_metrics['total_pnl']
            })
            
            # Calcular drawdown
            if len(self.performance_history) > 1:
                peak_pnl = max([h['cumulative_pnl'] for h in self.performance_history])
                current_pnl = self.live_metrics['total_pnl']
                self.live_metrics['current_drawdown'] = (peak_pnl - current_pnl) / max(peak_pnl, 1)
                
                if self.live_metrics['current_drawdown'] > self.live_metrics['max_drawdown']:
                    self.live_metrics['max_drawdown'] = self.live_metrics['current_drawdown']
            
            # Adaptar configurações baseadas na performance
            self.adapt_configuration()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def adapt_configuration(self):
        """Adapta configurações baseadas na performance"""
        try:
            win_rate = self.live_metrics['win_rate']
            drawdown = self.live_metrics['current_drawdown']
            
            # Ajustar threshold de confiança
            if win_rate > 0.75:
                # Alta performance - pode ser mais agressivo
                self.adaptive_config['confidence_threshold'] = max(0.5, self.adaptive_config['confidence_threshold'] - 0.05)
                self.adaptive_config['scalp_aggressiveness'] = min(1.0, self.adaptive_config['scalp_aggressiveness'] + 0.1)
            elif win_rate < 0.4 or drawdown > 0.15:
                # Baixa performance - ser mais conservador
                self.adaptive_config['confidence_threshold'] = min(0.8, self.adaptive_config['confidence_threshold'] + 0.1)
                self.adaptive_config['scalp_aggressiveness'] = max(0.3, self.adaptive_config['scalp_aggressiveness'] - 0.1)
            
            # Ajustar multiplicador de risco
            if drawdown > 0.2:
                self.adaptive_config['risk_multiplier'] = 0.5
            elif win_rate > 0.8:
                self.adaptive_config['risk_multiplier'] = min(1.5, self.adaptive_config['risk_multiplier'] + 0.1)
            
            logger.info(f"Config adapted - Threshold: {self.adaptive_config['confidence_threshold']:.2f}, Risk: {self.adaptive_config['risk_multiplier']:.2f}")
            
        except Exception as e:
            logger.error(f"Error adapting configuration: {e}")
    
    def save_models(self):
        """Salva modelos treinados"""
        try:
            Path("models").mkdir(exist_ok=True)
            
            if self.entry_ensemble:
                joblib.dump(self.entry_ensemble, "models/entry_ensemble.joblib")
            if self.risk_classifier:
                joblib.dump(self.risk_classifier, "models/risk_classifier.joblib")
            if self.pattern_clusterer:
                joblib.dump(self.pattern_clusterer, "models/pattern_clusterer.joblib")
                
            joblib.dump(self.feature_scaler, "models/feature_scaler.joblib")
            joblib.dump(self.adaptive_config, "models/adaptive_config.joblib")
            
            logger.info("Advanced models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Carrega modelos salvos"""
        try:
            if Path("models/entry_ensemble.joblib").exists():
                self.entry_ensemble = joblib.load("models/entry_ensemble.joblib")
            if Path("models/risk_classifier.joblib").exists():
                self.risk_classifier = joblib.load("models/risk_classifier.joblib")
            if Path("models/pattern_clusterer.joblib").exists():
                self.pattern_clusterer = joblib.load("models/pattern_clusterer.joblib")
            if Path("models/feature_scaler.joblib").exists():
                self.feature_scaler = joblib.load("models/feature_scaler.joblib")
            if Path("models/adaptive_config.joblib").exists():
                self.adaptive_config = joblib.load("models/adaptive_config.joblib")
                
            logger.info("Advanced models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Instância global do sistema avançado
ml_scalper = AdvancedMLScalper()

# ===== FASTAPI APPLICATION =====

app = FastAPI(
    title="Advanced AI Scalping Bot - Deriv Optimized",
    description="Sistema Avançado de Scalping com IA e Lógica Específica da Deriv",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "bot_name": "Advanced AI Scalping Bot",
        "version": "4.0.0",
        "deriv_optimized": True,
        "status": "operational",
        "models_status": {
            "entry_ensemble": ml_scalper.entry_ensemble is not None,
            "risk_classifier": ml_scalper.risk_classifier is not None,
            "pattern_clusterer": ml_scalper.pattern_clusterer is not None,
        },
        "live_metrics": ml_scalper.live_metrics,
        "adaptive_config": ml_scalper.adaptive_config
    }

@app.post("/scalping/analyze")
async def analyze_scalping_opportunity(
    symbol: str,
    prices: List[float],
    include_reasoning: bool = True
):
    """Analisa oportunidade de scalping usando IA avançada"""
    try:
        decision = ml_scalper.make_intelligent_decision(symbol, prices)
        
        response = {
            "symbol": symbol,
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
            "models_used": {
                "ml_ensemble": ml_scalper.entry_ensemble is not None,
                "risk_assessment": ml_scalper.risk_classifier is not None,
                "pattern_clustering": ml_scalper.pattern_clusterer is not None
            }
        }
        
        if not include_reasoning:
            response["decision"].pop("reasoning", None)
            response["decision"].pop("deriv_factors", None)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in scalping analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scalping/train")
async def train_models():
    """Treina modelos com dados históricos"""
    try:
        # Simular dados históricos para treinamento
        # Em produção, estes dados viriam do banco
        mock_data = []
        for i in range(200):
            mock_data.append({
                'symbol': np.random.choice(['R_10', 'R_25', 'R_50', 'R_75', 'R_100']),
                'entry_price': np.random.uniform(900, 1100),
                'pnl': np.random.normal(0, 5),
                'duration_actual': np.random.randint(30, 300)
            })
        
        training_df = pd.DataFrame(mock_data)
        success = ml_scalper.train_ensemble_models(training_df)
        
        if success:
            ml_scalper.save_models()
            return {
                "status": "success",
                "message": "Models trained successfully",
                "training_samples": len(training_df),
                "models_trained": ["entry_ensemble", "risk_classifier", "pattern_clusterer"]
            }
        else:
            return {
                "status": "failed",
                "message": "Training failed - insufficient data"
            }
            
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scalping/performance")
async def get_performance_metrics():
    """Retorna métricas de performance detalhadas"""
    try:
        return {
            "live_metrics": ml_scalper.live_metrics,
            "adaptive_config": ml_scalper.adaptive_config,
            "recent_patterns": len(ml_scalper.pattern_memory),
            "performance_history_size": len(ml_scalper.performance_history),
            "model_status": {
                "entry_ensemble": ml_scalper.entry_ensemble is not None,
                "risk_classifier": ml_scalper.risk_classifier is not None,
                "pattern_clusterer": ml_scalper.pattern_clusterer is not None
            },
            "deriv_configs": list(ml_scalper.deriv_analyzer.synthetic_configs.keys())
        }
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scalping/update_result")
async def update_trade_result(
    trade_id: str,
    pnl: float,
    duration_actual: int,
    exit_reason: str = "manual"
):
    """Atualiza resultado de trade para aprendizado"""
    try:
        trade_result = {
            'trade_id': trade_id,
            'pnl': pnl,
            'duration_actual': duration_actual,
            'exit_reason': exit_reason
        }
        
        ml_scalper.update_performance_metrics(trade_result)
        
        return {
            "status": "updated",
            "trade_id": trade_id,
            "updated_metrics": ml_scalper.live_metrics,
            "adaptive_changes": ml_scalper.adaptive_config
        }
        
    except Exception as e:
        logger.error(f"Error updating trade result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Interface HTML do dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Scalping Dashboard - Deriv Optimized</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .dashboard { 
                max-width: 1400px; margin: 0 auto; 
                display: grid; gap: 20px;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }
            .card { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; padding: 20px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                backdrop-filter: blur(10px);
            }
            .header { 
                text-align: center; margin-bottom: 30px;
                grid-column: 1 / -1;
            }
            .header h1 { 
                color: white; font-size: 2.5em; margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p { 
                color: rgba(255,255,255,0.9); font-size: 1.2em;
                margin: 10px 0;
            }
            .metric { 
                display: flex; justify-content: space-between; 
                align-items: center; padding: 12px 0; 
                border-bottom: 1px solid #eee;
            }
            .metric:last-child { border-bottom: none; }
            .metric-label { font-weight: 600; color: #555; }
            .metric-value { 
                font-weight: 700; padding: 5px 10px; 
                border-radius: 20px; color: white;
            }
            .positive { background: linear-gradient(135deg, #4CAF50, #45a049); }
            .negative { background: linear-gradient(135deg, #f44336, #d32f2f); }
            .neutral { background: linear-gradient(135deg, #2196F3, #1976d2); }
            .btn { 
                padding: 12px 20px; border: none; border-radius: 8px; 
                cursor: pointer; font-weight: 600; margin: 5px;
                transition: all 0.3s ease;
            }
            .btn-primary { 
                background: linear-gradient(135deg, #667eea, #764ba2); 
                color: white; 
            }
            .btn-primary:hover { transform: translateY(-2px); }
            .btn-success { background: #4CAF50; color: white; }
            .btn-danger { background: #f44336; color: white; }
            .status-indicator {
                width: 12px; height: 12px; border-radius: 50%;
                display: inline-block; margin-right: 8px;
            }
            .status-online { background: #4CAF50; }
            .status-offline { background: #f44336; }
            .analysis-section {
                background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 15px 0;
            }
            .symbol-selector {
                width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #ddd;
                margin: 10px 0;
            }
            #log { 
                height: 200px; overflow-y: auto; 
                background: #1a1a1a; color: #00ff00; 
                padding: 15px; border-radius: 8px; 
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>🤖 AI Scalping Bot</h1>
                <p>Otimizado para Deriv • Machine Learning Avançado</p>
                <div>
                    <span class="status-indicator status-online"></span>
                    Sistema Operacional
                </div>
            </div>

            <!-- Performance Card -->
            <div class="card">
                <h3>📊 Performance em Tempo Real</h3>
                <div id="performance-metrics">
                    <div class="metric">
                        <span class="metric-label">Total de Trades</span>
                        <span class="metric-value neutral" id="total-trades">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value neutral" id="win-rate">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">PnL Total</span>
                        <span class="metric-value neutral" id="total-pnl">$0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Drawdown Atual</span>
                        <span class="metric-value neutral" id="current-drawdown">0%</span>
                    </div>
                </div>
            </div>

            <!-- AI Models Status -->
            <div class="card">
                <h3>🧠 Status dos Modelos IA</h3>
                <div id="models-status">
                    <div class="metric">
                        <span class="metric-label">Ensemble de Entrada</span>
                        <span class="metric-value" id="entry-model">🔄</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Classificador de Risco</span>
                        <span class="metric-value" id="risk-model">🔄</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Clustering de Padrões</span>
                        <span class="metric-value" id="pattern-model">🔄</span>
                    </div>
                </div>
                <button class="btn btn-primary" onclick="trainModels()">
                    🎯 Treinar Modelos
                </button>
            </div>

            <!-- Análise de Mercado -->
            <div class="card">
                <h3>📈 Análise de Mercado</h3>
                <select class="symbol-selector" id="symbol-select">
                    <option value="R_10">Volatility 10 Index</option>
                    <option value="R_25">Volatility 25 Index</option>
                    <option value="R_50" selected>Volatility 50 Index</option>
                    <option value="R_75">Volatility 75 Index</option>
                    <option value="R_100">Volatility 100 Index</option>
                </select>
                
                <button class="btn btn-primary" onclick="analyzeMarket()">
                    🔍 Analisar Oportunidade
                </button>
                
                <div class="analysis-section" id="analysis-result" style="display:none;">
                    <h4>Resultado da Análise:</h4>
                    <div id="analysis-content"></div>
                </div>
            </div>

            <!-- Configurações Adaptativas -->
            <div class="card">
                <h3>⚙️ Configurações Adaptativas</h3>
                <div id="adaptive-config">
                    <div class="metric">
                        <span class="metric-label">Threshold Confiança</span>
                        <span class="metric-value neutral" id="confidence-threshold">65%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Multiplicador Risco</span>
                        <span class="metric-value neutral" id="risk-multiplier">1.0x</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Agressividade Scalp</span>
                        <span class="metric-value neutral" id="scalp-aggressiveness">70%</span>
                    </div>
                </div>
            </div>

            <!-- Log do Sistema -->
            <div class="card" style="grid-column: 1 / -1;">
                <h3>📋 Log do Sistema</h3>
                <div id="log"></div>
                <button class="btn btn-primary" onclick="clearLog()">
                    🗑️ Limpar Log
                </button>
            </div>
        </div>

        <script>
            // Estado global
            let isAnalyzing = false;
            let updateInterval;

            // Função para log
            function addLog(message) {
                const log = document.getElementById('log');
                const timestamp = new Date().toLocaleTimeString();
                log.innerHTML += `[${timestamp}] ${message}\\n`;
                log.scrollTop = log.scrollHeight;
            }

            function clearLog() {
                document.getElementById('log').innerHTML = '';
                addLog('Log limpo');
            }

            // Função para atualizar métricas
            async function updateMetrics() {
                try {
                    const response = await fetch('/scalping/performance');
                    const data = await response.json();

                    // Atualizar métricas de performance
                    document.getElementById('total-trades').textContent = data.live_metrics.total_trades;
                    
                    const winRate = (data.live_metrics.win_rate * 100).toFixed(1);
                    const winRateElement = document.getElementById('win-rate');
                    winRateElement.textContent = winRate + '%';
                    winRateElement.className = 'metric-value ' + (data.live_metrics.win_rate > 0.5 ? 'positive' : 'negative');

                    const pnl = data.live_metrics.total_pnl.toFixed(2);
                    const pnlElement = document.getElementById('total-pnl');
                    pnlElement.textContent = '$' + pnl;
                    pnlElement.className = 'metric-value ' + (data.live_metrics.total_pnl >= 0 ? 'positive' : 'negative');

                    const drawdown = (data.live_metrics.current_drawdown * 100).toFixed(1);
                    const drawdownElement = document.getElementById('current-drawdown');
                    drawdownElement.textContent = drawdown + '%';
                    drawdownElement.className = 'metric-value ' + (data.live_metrics.current_drawdown < 0.1 ? 'positive' : 'negative');

                    // Atualizar status dos modelos
                    document.getElementById('entry-model').textContent = data.model_status.entry_ensemble ? '✅' : '❌';
                    document.getElementById('risk-model').textContent = data.model_status.risk_classifier ? '✅' : '❌';
                    document.getElementById('pattern-model').textContent = data.model_status.pattern_clusterer ? '✅' : '❌';

                    // Atualizar configurações adaptativas
                    document.getElementById('confidence-threshold').textContent = (data.adaptive_config.confidence_threshold * 100).toFixed(0) + '%';
                    document.getElementById('risk-multiplier').textContent = data.adaptive_config.risk_multiplier.toFixed(1) + 'x';
                    document.getElementById('scalp-aggressiveness').textContent = (data.adaptive_config.scalp_aggressiveness * 100).toFixed(0) + '%';

                } catch (error) {
                    addLog('❌ Erro ao atualizar métricas: ' + error.message);
                }
            }

            // Função para analisar mercado
            async function analyzeMarket() {
                if (isAnalyzing) return;
                
                isAnalyzing = true;
                const symbol = document.getElementById('symbol-select').value;
                
                addLog(`🔍 Iniciando análise para ${symbol}...`);

                try {
                    // Simular dados de preço para análise
                    const mockPrices = [];
                    let basePrice = 1000;
                    
                    for (let i = 0; i < 30; i++) {
                        basePrice += (Math.random() - 0.5) * 10;
                        mockPrices.push(basePrice);
                    }

                    const response = await fetch('/scalping/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            symbol: symbol,
                            prices: mockPrices,
                            include_reasoning: true
                        })
                    });

                    const data = await response.json();
                    displayAnalysisResult(data);
                    
                    addLog(`✅ Análise concluída: ${data.decision.action} (${(data.decision.confidence * 100).toFixed(1)}% confiança)`);

                } catch (error) {
                    addLog('❌ Erro na análise: ' + error.message);
                } finally {
                    isAnalyzing = false;
                }
            }

            function displayAnalysisResult(data) {
                const resultDiv = document.getElementById('analysis-result');
                const contentDiv = document.getElementById('analysis-content');
                
                const decision = data.decision;
                const actionColor = decision.action === 'hold' ? 'neutral' : 
                                  decision.action.includes('call') ? 'positive' : 'negative';

                contentDiv.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Ação Recomendada</span>
                        <span class="metric-value ${actionColor}">${decision.action.toUpperCase()}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confiança</span>
                        <span class="metric-value neutral">${(decision.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Risco</span>
                        <span class="metric-value ${decision.risk_score > 0.7 ? 'negative' : decision.risk_score > 0.4 ? 'neutral' : 'positive'}">
                            ${(decision.risk_score * 100).toFixed(0)}%
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Stake Sugerido</span>
                        <span class="metric-value neutral">$${decision.stake.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Duração</span>
                        <span class="metric-value neutral">${decision.duration}s</span>
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 6px;">
                        <strong>Raciocínio:</strong><br>
                        ${decision.reasoning}
                    </div>
                `;
                
                resultDiv.style.display = 'block';
            }

            // Função para treinar modelos
            async function trainModels() {
                addLog('🎯 Iniciando treinamento dos modelos...');
                
                try {
                    const response = await fetch('/scalping/train', {
                        method: 'POST'
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        addLog(`✅ Modelos treinados com sucesso! (${result.training_samples} amostras)`);
                        updateMetrics(); // Atualizar status dos modelos
                    } else {
                        addLog(`❌ Falha no treinamento: ${result.message}`);
                    }
                    
                } catch (error) {
                    addLog('❌ Erro no treinamento: ' + error.message);
                }
            }

            // Inicialização
            document.addEventListener('DOMContentLoaded', function() {
                addLog('🚀 AI Scalping Bot iniciado');
                addLog('🤖 Sistema otimizado para Deriv');
                addLog('📊 Carregando métricas iniciais...');
                
                updateMetrics();
                
                // Atualizar métricas a cada 10 segundos
                updateInterval = setInterval(updateMetrics, 10000);
                
                addLog('✅ Dashboard pronto!');
            });

            // Cleanup ao sair da página
            window.addEventListener('beforeunload', function() {
                if (updateInterval) {
                    clearInterval(updateInterval);
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info("🚀 Iniciando Advanced AI Scalping Bot")
    logger.info("🎯 Características principais:")
    logger.info("   • Ensemble de modelos ML especializados")
    logger.info("   • Lógica específica dos índices sintéticos da Deriv")
    logger.info("   • Sistema de aprendizado adaptativo")
    logger.info("   • Análise de padrões avançada")
    logger.info("   • Risk management inteligente")
    logger.info("   • Interface web interativa")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
