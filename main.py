#!/usr/bin/env python3
"""
Sistema de Scalping Automatizado com IA para Trading Bot
Análise de padrões da Deriv + Machine Learning + Decisões automáticas
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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== MODELOS PYDANTIC =====

@dataclass
class MarketState:
    """Estado atual do mercado para análise"""
    symbol: str
    price: float
    timestamp: datetime
    volatility: float
    trend_direction: int  # -1, 0, 1
    momentum_strength: float
    volume_indicator: float
    support_resistance: Dict
    pattern_detected: str

class ScalpingDecision(BaseModel):
    action: str  # 'buy_call', 'buy_put', 'hold', 'exit'
    confidence: float
    entry_price: float
    target_profit: float
    stop_loss: float
    duration: int
    reasoning: str
    risk_score: float

class AutoTradeRequest(BaseModel):
    symbol: str
    balance: float
    max_stake: float = 10.0
    min_profit_percent: float = 5.0
    max_loss_percent: float = 30.0
    scalping_mode: str = "aggressive"  # conservative, moderate, aggressive
    
# ===== SISTEMA DE ANÁLISE DE PADRÕES DA DERIV =====

class DerivPatternAnalyzer:
    """Analisador de padrões específicos da Deriv"""
    
    def __init__(self):
        self.volatility_patterns = {
            'R_10': {'base_volatility': 10, 'jump_frequency': 0.1, 'trend_persistence': 0.3},
            'R_25': {'base_volatility': 25, 'jump_frequency': 0.15, 'trend_persistence': 0.35},
            'R_50': {'base_volatility': 50, 'jump_frequency': 0.2, 'trend_persistence': 0.4},
            'R_75': {'base_volatility': 75, 'jump_frequency': 0.25, 'trend_persistence': 0.45},
            'R_100': {'base_volatility': 100, 'jump_frequency': 0.3, 'trend_persistence': 0.5}
        }
        
        self.price_history = {}
        self.pattern_memory = {}
        
    def analyze_volatility_cycle(self, symbol: str, prices: List[float]) -> Dict:
        """Analisa o ciclo de volatilidade específico do símbolo"""
        if len(prices) < 20:
            return {'cycle_phase': 'unknown', 'volatility_score': 0.5}
        
        # Calcular volatilidade realizada
        returns = np.diff(np.log(prices))
        realized_vol = np.std(returns) * np.sqrt(len(returns))
        
        base_vol = self.volatility_patterns.get(symbol, {}).get('base_volatility', 50)
        vol_ratio = realized_vol / (base_vol / 10000)  # Normalizar
        
        # Detectar fase do ciclo
        if vol_ratio < 0.8:
            cycle_phase = 'low_volatility'
            opportunity_score = 0.3  # Baixa oportunidade
        elif vol_ratio > 1.2:
            cycle_phase = 'high_volatility'
            opportunity_score = 0.8  # Alta oportunidade para scalping
        else:
            cycle_phase = 'normal_volatility'
            opportunity_score = 0.6
        
        return {
            'cycle_phase': cycle_phase,
            'volatility_score': opportunity_score,
            'realized_volatility': realized_vol,
            'volatility_ratio': vol_ratio
        }
    
    def detect_deriv_patterns(self, symbol: str, prices: List[float], timestamps: List[datetime]) -> Dict:
        """Detecta padrões específicos dos índices da Deriv"""
        if len(prices) < 10:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        patterns = []
        
        # Padrão 1: Reversão após movimento extremo
        price_changes = np.diff(prices[-10:])
        if len(price_changes) >= 5:
            extreme_moves = np.abs(price_changes) > np.std(price_changes) * 2
            if np.sum(extreme_moves[-3:]) >= 2:  # 2 movimentos extremos recentes
                patterns.append({
                    'pattern': 'extreme_reversal_setup',
                    'confidence': 0.75,
                    'direction': 'opposite',
                    'reasoning': 'Movimentos extremos tendem a reverter na Deriv'
                })
        
        # Padrão 2: Consolidação antes de breakout
        recent_prices = prices[-20:]
        if len(recent_prices) >= 20:
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = np.mean(recent_prices)
            consolidation_ratio = price_range / avg_price
            
            if consolidation_ratio < 0.02:  # Consolidação apertada
                patterns.append({
                    'pattern': 'breakout_setup',
                    'confidence': 0.65,
                    'direction': 'momentum',
                    'reasoning': 'Consolidação indica possível breakout'
                })
        
        # Padrão 3: Padrão de tempo (horários específicos)
        if timestamps:
            current_hour = timestamps[-1].hour
            # Horários de maior volatilidade na Deriv
            volatile_hours = [8, 9, 13, 14, 16, 17, 20, 21]
            if current_hour in volatile_hours:
                patterns.append({
                    'pattern': 'high_volatility_time',
                    'confidence': 0.6,
                    'direction': 'any',
                    'reasoning': f'Horário {current_hour}h tem maior volatilidade'
                })
        
        # Retornar o padrão com maior confiança
        if patterns:
            best_pattern = max(patterns, key=lambda x: x['confidence'])
            return best_pattern
        
        return {'pattern': 'no_pattern', 'confidence': 0.0}

# ===== SISTEMA ML DE SCALPING AUTOMATIZADO =====

class AIScalpingSystem:
    """Sistema de IA para scalping automatizado"""
    
    def __init__(self, db_path: str = "data/ai_scalping.db"):
        self.db_path = db_path
        self.pattern_analyzer = DerivPatternAnalyzer()
        
        # Modelos especializados
        self.entry_model = None  # Modelo para decisão de entrada
        self.exit_model = None   # Modelo para decisão de saída
        self.risk_model = None   # Modelo para análise de risco
        self.scaler = StandardScaler()
        
        # Estado do sistema
        self.is_active = False
        self.current_trades = {}
        self.market_state = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        # Configurações de scalping
        self.scalping_config = {
            'min_profit_percent': 3.0,   # Mínimo 3% para sair
            'quick_exit_percent': 5.0,   # Saída rápida em 5%
            'safe_exit_percent': 15.0,   # Saída segura em 15%
            'stop_loss_percent': 25.0,   # Stop loss em 25%
            'max_trade_duration': 300,   # 5 minutos máximo
            'confidence_threshold': 0.65  # Confiança mínima para trade
        }
        
        # Histórico para análise
        self.price_history = {}
        self.decision_history = []
        
        # Inicializar
        Path("data").mkdir(exist_ok=True)
        self.init_database()
        self.load_models()
        
    def init_database(self):
        """Inicializa banco de dados para scalping"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de trades automatizados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stake REAL NOT NULL,
                duration_planned INTEGER,
                duration_actual INTEGER,
                pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                ai_confidence REAL,
                pattern_detected TEXT,
                exit_reason TEXT,
                market_state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de decisões da IA
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                market_features TEXT,
                outcome TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de padrões aprendidos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                symbol TEXT,
                features TEXT NOT NULL,
                success_rate REAL NOT NULL,
                occurrences INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized for AI Scalping")
    
    def create_market_features(self, market_data: Dict) -> np.ndarray:
        """Cria features para os modelos ML"""
        try:
            symbol = market_data.get('symbol', 'R_50')
            prices = market_data.get('prices', [1000])
            
            if len(prices) < 2:
                # Features padrão se dados insuficientes
                return np.array([1000, 0.5, 0, 0, 0, 0.5, 12, 1]).reshape(1, -1)
            
            # Features básicas
            current_price = prices[-1]
            price_change = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
            
            # Features de volatilidade
            if len(prices) >= 10:
                returns = np.diff(np.log(prices[-10:]))
                volatility = np.std(returns)
                momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            else:
                volatility = 0.05
                momentum = price_change
            
            # Features de tendência
            if len(prices) >= 5:
                recent_trend = np.polyfit(range(len(prices[-5:])), prices[-5:], 1)[0]
            else:
                recent_trend = 0
            
            # Features temporais
            current_time = datetime.now()
            hour_feature = np.sin(2 * np.pi * current_time.hour / 24)
            day_feature = current_time.weekday() / 6
            
            # Features específicas do símbolo
            symbol_volatility = self.pattern_analyzer.volatility_patterns.get(
                symbol, {}
            ).get('base_volatility', 50) / 100
            
            features = np.array([
                current_price,
                price_change,
                volatility,
                momentum,
                recent_trend,
                symbol_volatility,
                hour_feature,
                day_feature
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return np.array([1000, 0, 0.05, 0, 0, 0.5, 0, 0.5]).reshape(1, -1)
    
    def train_specialized_models(self, historical_data: pd.DataFrame) -> bool:
        """Treina modelos especializados para scalping"""
        try:
            if len(historical_data) < 50:
                logger.warning("Insufficient data for training specialized models")
                return False
            
            # Preparar dados para modelo de entrada
            entry_features = []
            entry_targets = []
            
            # Preparar dados para modelo de saída
            exit_features = []
            exit_targets = []
            
            for _, row in historical_data.iterrows():
                # Simular features de mercado
                market_features = self.create_market_features({
                    'symbol': row.get('symbol', 'R_50'),
                    'prices': [row.get('entry_price', 1000)]
                }).flatten()
                
                # Target para entrada (se deve entrar no trade)
                entry_success = 1 if row.get('pnl', 0) > 0 else 0
                entry_features.append(market_features)
                entry_targets.append(entry_success)
                
                # Target para saída (quando deve sair)
                if row.get('duration_actual'):
                    exit_timing = min(1.0, row.get('duration_actual', 60) / 300)  # Normalizar
                    exit_features.append(market_features)
                    exit_targets.append(exit_timing)
            
            # Treinar modelo de entrada
            if len(entry_features) >= 20:
                X_entry = np.array(entry_features)
                y_entry = np.array(entry_targets)
                
                # Normalizar features
                X_entry_scaled = self.scaler.fit_transform(X_entry)
                
                self.entry_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                self.entry_model.fit(X_entry_scaled, y_entry)
                
                # Modelo de risco (detecção de anomalias)
                self.risk_model = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self.risk_model.fit(X_entry_scaled)
                
                logger.info("Specialized models trained successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training specialized models: {e}")
            return False
    
    async def analyze_market_realtime(self, symbol: str, price_data: List[float]) -> MarketState:
        """Análise de mercado em tempo real"""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            # Adicionar novo preço
            self.price_history[symbol].extend(price_data)
            
            # Manter apenas últimos 100 pontos
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            prices = self.price_history[symbol]
            current_price = prices[-1]
            
            # Análise de volatilidade
            vol_analysis = self.pattern_analyzer.analyze_volatility_cycle(symbol, prices)
            
            # Detecção de padrões
            pattern_info = self.pattern_analyzer.detect_deriv_patterns(
                symbol, prices, [datetime.now()] * len(prices)
            )
            
            # Calcular momentum
            if len(prices) >= 10:
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:])
                momentum_strength = (short_ma - long_ma) / long_ma
                trend_direction = 1 if momentum_strength > 0.001 else -1 if momentum_strength < -0.001 else 0
            else:
                momentum_strength = 0
                trend_direction = 0
            
            # Calcular suporte e resistência
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                support = min(recent_prices)
                resistance = max(recent_prices)
            else:
                support = current_price * 0.99
                resistance = current_price * 1.01
            
            market_state = MarketState(
                symbol=symbol,
                price=current_price,
                timestamp=datetime.now(),
                volatility=vol_analysis['volatility_score'],
                trend_direction=trend_direction,
                momentum_strength=abs(momentum_strength),
                volume_indicator=vol_analysis['volatility_ratio'],
                support_resistance={'support': support, 'resistance': resistance},
                pattern_detected=pattern_info['pattern']
            )
            
            self.market_state[symbol] = market_state
            return market_state
            
        except Exception as e:
            logger.error(f"Error in realtime market analysis: {e}")
            return MarketState(
                symbol=symbol,
                price=price_data[-1] if price_data else 1000,
                timestamp=datetime.now(),
                volatility=0.5,
                trend_direction=0,
                momentum_strength=0,
                volume_indicator=1.0,
                support_resistance={'support': 990, 'resistance': 1010},
                pattern_detected='error'
            )
    
    async def make_scalping_decision(self, market_state: MarketState, balance: float) -> ScalpingDecision:
        """Toma decisão de scalping baseada em IA"""
        try:
            # Verificar se já tem trade ativo neste símbolo
            if market_state.symbol in self.current_trades:
                return await self.make_exit_decision(market_state)
            
            # Criar features para análise
            market_features = self.create_market_features({
                'symbol': market_state.symbol,
                'prices': self.price_history.get(market_state.symbol, [market_state.price])
            })
            
            # Análise de risco
            risk_score = 0.5
            if self.risk_model:
                try:
                    features_scaled = self.scaler.transform(market_features)
                    risk_prediction = self.risk_model.decision_function(features_scaled)[0]
                    risk_score = max(0.1, min(0.9, (risk_prediction + 1) / 2))
                except:
                    pass
            
            # Decisão de entrada usando modelo
            entry_confidence = 0.5
            should_enter = False
            
            if self.entry_model:
                try:
                    features_scaled = self.scaler.transform(market_features)
                    entry_proba = self.entry_model.predict_proba(features_scaled)[0]
                    entry_confidence = max(entry_proba)
                    should_enter = entry_confidence > self.scalping_config['confidence_threshold']
                except:
                    pass
            
            # Lógica heurística baseada em padrões da Deriv
            pattern_boost = 0
            direction_hint = 'hold'
            
            if market_state.pattern_detected == 'extreme_reversal_setup':
                pattern_boost = 0.2
                direction_hint = 'put' if market_state.trend_direction > 0 else 'call'
            elif market_state.pattern_detected == 'breakout_setup':
                pattern_boost = 0.15
                direction_hint = 'call' if market_state.momentum_strength > 0 else 'put'
            elif market_state.pattern_detected == 'high_volatility_time':
                pattern_boost = 0.1
                direction_hint = 'call' if market_state.trend_direction >= 0 else 'put'
            
            # Ajustar confiança com padrões
            final_confidence = min(0.95, entry_confidence + pattern_boost)
            
            # Calcular stake baseado no risco
            max_stake = min(balance * 0.02, 10.0)  # Máximo 2% do saldo
            stake = max_stake * (1 - risk_score)  # Reduzir stake se risco alto
            stake = max(1.0, stake)  # Mínimo $1
            
            # Decidir ação
            if should_enter and final_confidence > self.scalping_config['confidence_threshold']:
                if direction_hint == 'hold':
                    # Usar momentum se não há hint de direção
                    action = 'buy_call' if market_state.momentum_strength > 0 else 'buy_put'
                else:
                    action = f'buy_{direction_hint}'
                
                # Calcular targets
                price_volatility = market_state.price * market_state.volatility * 0.01
                target_profit = market_state.price + (price_volatility * 2)
                stop_loss = market_state.price - (price_volatility * 3)
                
                reasoning = f"Pattern: {market_state.pattern_detected}, Confidence: {final_confidence:.2f}, Risk: {risk_score:.2f}"
                
                decision = ScalpingDecision(
                    action=action,
                    confidence=final_confidence,
                    entry_price=market_state.price,
                    target_profit=target_profit,
                    stop_loss=stop_loss,
                    duration=180,  # 3 minutos
                    reasoning=reasoning,
                    risk_score=risk_score
                )
            else:
                decision = ScalpingDecision(
                    action='hold',
                    confidence=final_confidence,
                    entry_price=market_state.price,
                    target_profit=0,
                    stop_loss=0,
                    duration=0,
                    reasoning=f"Insufficient confidence: {final_confidence:.2f} < {self.scalping_config['confidence_threshold']}",
                    risk_score=risk_score
                )
            
            # Salvar decisão
            await self.save_ai_decision(market_state.symbol, decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making scalping decision: {e}")
            return ScalpingDecision(
                action='hold',
                confidence=0.0,
                entry_price=market_state.price,
                target_profit=0,
                stop_loss=0,
                duration=0,
                reasoning=f"Error in decision making: {str(e)}",
                risk_score=1.0
            )
    
    async def make_exit_decision(self, market_state: MarketState) -> ScalpingDecision:
        """Decisão de saída para trades ativos"""
        try:
            trade_info = self.current_trades.get(market_state.symbol)
            if not trade_info:
                return ScalpingDecision(action='hold', confidence=0.0, entry_price=0, target_profit=0, stop_loss=0, duration=0, reasoning="No active trade", risk_score=0.0)
            
            entry_price = trade_info['entry_price']
            entry_time = trade_info['entry_time']
            direction = trade_info['direction']  # 'call' or 'put'
            
            current_price = market_state.price
            time_in_trade = (datetime.now() - entry_time).total_seconds()
            
            # Calcular P&L atual
            if direction == 'call':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # put
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            should_exit = False
            exit_reason = ""
            
            # Regra 1: Tempo máximo
            if time_in_trade > self.scalping_config['max_trade_duration']:
                should_exit = True
                exit_reason = "Max time reached"
            
            # Regra 2: Stop loss
            elif pnl_percent <= -self.scalping_config['stop_loss_percent']:
                should_exit = True
                exit_reason = f"Stop loss: {pnl_percent:.1f}%"
            
            # Regra 3: Saída segura
            elif pnl_percent >= self.scalping_config['safe_exit_percent']:
                should_exit = True
                exit_reason = f"Safe exit: {pnl_percent:.1f}%"
            
            # Regra 4: Saída rápida com momentum negativo
            elif (pnl_percent >= self.scalping_config['quick_exit_percent'] and 
                  market_state.momentum_strength < 0.1 and time_in_trade > 30):
                should_exit = True
                exit_reason = f"Quick exit - weak momentum: {pnl_percent:.1f}%"
            
            # Regra 5: Padrão de reversão detectado
            elif (pnl_percent >= self.scalping_config['min_profit_percent'] and 
                  market_state.pattern_detected == 'extreme_reversal_setup'):
                should_exit = True
                exit_reason = f"Reversal pattern detected: {pnl_percent:.1f}%"
            
            if should_exit:
                decision = ScalpingDecision(
                    action='exit',
                    confidence=0.8,
                    entry_price=entry_price,
                    target_profit=current_price,
                    stop_loss=0,
                    duration=int(time_in_trade),
                    reasoning=exit_reason,
                    risk_score=0.3
                )
            else:
                decision = ScalpingDecision(
                    action='hold',
                    confidence=0.6,
                    entry_price=entry_price,
                    target_profit=current_price,
                    stop_loss=0,
                    duration=int(time_in_trade),
                    reasoning=f"Hold - P&L: {pnl_percent:.1f}%, Time: {time_in_trade:.0f}s",
                    risk_score=0.5
                )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making exit decision: {e}")
            return ScalpingDecision(action='hold', confidence=0.0, entry_price=0, target_profit=0, stop_loss=0, duration=0, reasoning=f"Exit error: {str(e)}", risk_score=1.0)
    
    async def save_ai_decision(self, symbol: str, decision: ScalpingDecision):
        """Salva decisão da IA para aprendizado"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            market_features = json.dumps({
                'symbol': symbol,
                'price': decision.entry_price,
                'confidence': decision.confidence,
                'risk_score': decision.risk_score
            })
            
            cursor.execute('''
                INSERT INTO ai_decisions 
                (timestamp, symbol, decision_type, action_taken, confidence, reasoning, market_features)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                'scalping',
                decision.action,
                decision.confidence,
                decision.reasoning,
                market_features
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving AI decision: {e}")
    
    async def execute_trade(self, symbol: str, decision: ScalpingDecision, websocket_client) -> bool:
        """Executa trade automaticamente"""
        try:
            if decision.action == 'hold':
                return False
            
            if decision.action == 'exit':
                # Executar saída
                trade_info = self.current_trades.get(symbol)
                if trade_info:
                    # Enviar comando de venda via WebSocket
                    sell_command = {
                        "sell": trade_info['contract_id'],
                        "price": decision.target_profit
                    }
                    await websocket_client.send(json.dumps(sell_command))
                    
                    # Remover trade ativo
                    del self.current_trades[symbol]
                    
                    logger.info(f"Exit trade executed for {symbol}: {decision.reasoning}")
                    return True
                return False
            
            # Executar entrada (buy_call ou buy_put)
            direction = decision.action.replace('buy_', '').upper()
            
            # Calcular stake baseado no risco
            balance = 1000  # Pegar do WebSocket
            max_stake = min(balance * 0.02, 10.0)
            stake = max(1.0, max_stake * (1 - decision.risk_score))
            
            buy_command = {
                "buy": 1,
                "price": stake,
                "parameters": {
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": direction,
                    "currency": "USD",
                    "duration": decision.duration,
                    "duration_unit": "s",
                    "symbol": symbol
                }
            }
            
            await websocket_client.send(json.dumps(buy_command))
            
            # Registrar trade ativo
            self.current_trades[symbol] = {
                'entry_price': decision.entry_price,
                'entry_time': datetime.now(),
                'direction': direction.lower(),
                'stake': stake,
                'contract_id': None,  # Será preenchido na resposta
                'decision': decision
            }
            
            logger.info(f"Entry trade executed for {symbol}: {direction} at {decision.entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def learn_from_outcome(self, trade_id: str, outcome: str, pnl: float):
        """Aprende com o resultado do trade"""
        try:
            # Buscar decisão original
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Atualizar resultado da decisão
            cursor.execute('''
                UPDATE ai_decisions 
                SET outcome = ?
                WHERE symbol = ? AND timestamp > ?
            ''', (f"{outcome}:{pnl:.2f}", trade_id, (datetime.now() - timedelta(minutes=10)).isoformat()))
            
            # Análise de padrão aprendido
            success = outcome == 'won' and pnl > 0
            
            if success:
                # Reforçar padrões que funcionaram
                cursor.execute('''
                    INSERT INTO learned_patterns 
                    (pattern_type, symbol, features, success_rate, occurrences)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT (pattern_type, symbol) DO UPDATE SET
                    success_rate = (success_rate * occurrences + 1) / (occurrences + 1),
                    occurrences = occurrences + 1
                ''', ('successful_scalp', trade_id, json.dumps({'pnl': pnl}), 1.0))
            
            conn.commit()
            conn.close()
            
            # Atualizar métricas
            self.performance_metrics['total_trades'] += 1
            if success:
                self.performance_metrics['winning_trades'] += 1
            
            self.performance_metrics['total_pnl'] += pnl
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
            
            logger.info(f"Learned from trade {trade_id}: {outcome}, PnL: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error learning from outcome: {e}")
    
    def save_models(self):
        """Salva modelos treinados"""
        try:
            Path("models").mkdir(exist_ok=True)
            
            if self.entry_model:
                joblib.dump(self.entry_model, "models/ai_entry_model.joblib")
            if self.exit_model:
                joblib.dump(self.exit_model, "models/ai_exit_model.joblib")
            if self.risk_model:
                joblib.dump(self.risk_model, "models/ai_risk_model.joblib")
            if self.scaler:
                joblib.dump(self.scaler, "models/ai_scaler.joblib")
                
            logger.info("AI models saved")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Carrega modelos salvos"""
        try:
            if Path("models/ai_entry_model.joblib").exists():
                self.entry_model = joblib.load("models/ai_entry_model.joblib")
            if Path("models/ai_exit_model.joblib").exists():
                self.exit_model = joblib.load("models/ai_exit_model.joblib")
            if Path("models/ai_risk_model.joblib").exists():
                self.risk_model = joblib.load("models/ai_risk_model.joblib")
            if Path("models/ai_scaler.joblib").exists():
                self.scaler = joblib.load("models/ai_scaler.joblib")
                
            logger.info("AI models loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# ===== SISTEMA PRINCIPAL =====

class AutoTradingEngine:
    """Engine principal de trading automatizado"""
    
    def __init__(self):
        self.ai_system = AIScalpingSystem()
        self.active_connections = {}
        self.is_running = False
        
    async def start_auto_trading(self, websocket_client, config: AutoTradeRequest):
        """Inicia trading automatizado"""
        self.is_running = True
        self.ai_system.is_active = True
        
        logger.info(f"Starting auto trading for {config.symbol}")
        
        while self.is_running and self.ai_system.is_active:
            try:
                # Simular dados de preço (em produção viria do WebSocket)
                current_price = np.random.uniform(990, 1010)
                price_data = [current_price]
                
                # Análise de mercado
                market_state = await self.ai_system.analyze_market_realtime(
                    config.symbol, price_data
                )
                
                # Decisão da IA
                decision = await self.ai_system.make_scalping_decision(
                    market_state, config.balance
                )
                
                # Executar trade se necessário
                if decision.action != 'hold':
                    success = await self.ai_system.execute_trade(
                        config.symbol, decision, websocket_client
                    )
                    if success:
                        logger.info(f"Trade executed: {decision.action} on {config.symbol}")
                
                # Aguardar antes da próxima análise
                await asyncio.sleep(5)  # Análise a cada 5 segundos
                
            except Exception as e:
                logger.error(f"Error in auto trading loop: {e}")
                await asyncio.sleep(10)
    
    def stop_auto_trading(self):
        """Para trading automatizado"""
        self.is_running = False
        self.ai_system.is_active = False
        logger.info("Auto trading stopped")

# ===== INSTÂNCIA GLOBAL =====
trading_engine = AutoTradingEngine()

# ===== FASTAPI APP =====

app = FastAPI(
    title="AI Scalping Trading Bot",
    description="Sistema de Trading Automatizado com IA e Scalping Inteligente",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "🤖 AI Scalping Trading Bot",
        "version": "3.0.0",
        "status": "online",
        "ai_active": trading_engine.ai_system.is_active,
        "models_loaded": {
            "entry_model": trading_engine.ai_system.entry_model is not None,
            "risk_model": trading_engine.ai_system.risk_model is not None,
        },
        "performance": trading_engine.ai_system.performance_metrics
    }

@app.post("/ai/start_auto_trading")
async def start_auto_trading(config: AutoTradeRequest, background_tasks: BackgroundTasks):
    """Inicia trading automatizado"""
    try:
        if trading_engine.is_running:
            return {"message": "Auto trading already running", "status": "active"}
        
        # Simular WebSocket client (em produção seria real)
        class MockWebSocketClient:
            async def send(self, message):
                logger.info(f"Sending to WebSocket: {message}")
        
        websocket_client = MockWebSocketClient()
        
        background_tasks.add_task(
            trading_engine.start_auto_trading,
            websocket_client,
            config
        )
        
        return {
            "message": "Auto trading started",
            "status": "active",
            "config": config.dict(),
            "ai_confidence_threshold": trading_engine.ai_system.scalping_config['confidence_threshold']
        }
    except Exception as e:
        logger.error(f"Error starting auto trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/stop_auto_trading")
async def stop_auto_trading():
    """Para trading automatizado"""
    trading_engine.stop_auto_trading()
    return {
        "message": "Auto trading stopped",
        "status": "stopped",
        "final_performance": trading_engine.ai_system.performance_metrics
    }

@app.get("/ai/status")
async def get_ai_status():
    """Status do sistema de IA"""
    return {
        "ai_active": trading_engine.ai_system.is_active,
        "auto_trading_running": trading_engine.is_running,
        "active_trades": len(trading_engine.ai_system.current_trades),
        "performance": trading_engine.ai_system.performance_metrics,
        "scalping_config": trading_engine.ai_system.scalping_config,
        "patterns_learned": len(trading_engine.ai_system.pattern_analyzer.pattern_memory)
    }

@app.post("/ai/manual_analysis")
async def manual_market_analysis(symbol: str, prices: List[float]):
    """Análise manual de mercado"""
    try:
        market_state = await trading_engine.ai_system.analyze_market_realtime(symbol, prices)
        decision = await trading_engine.ai_system.make_scalping_decision(market_state, 1000.0)
        
        return {
            "market_state": {
                "symbol": market_state.symbol,
                "price": market_state.price,
                "volatility": market_state.volatility,
                "trend": market_state.trend_direction,
                "momentum": market_state.momentum_strength,
                "pattern": market_state.pattern_detected
            },
            "ai_decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "risk_score": decision.risk_score
            }
        }
    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/performance")
async def get_performance_metrics():
    """Métricas de performance da IA"""
    return {
        "current_performance": trading_engine.ai_system.performance_metrics,
        "active_trades": {
            symbol: {
                "entry_time": info['entry_time'].isoformat(),
                "entry_price": info['entry_price'],
                "direction": info['direction'],
                "stake": info['stake']
            } for symbol, info in trading_engine.ai_system.current_trades.items()
        },
        "recent_decisions": len(trading_engine.ai_system.decision_history),
        "scalping_settings": trading_engine.ai_system.scalping_config
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting AI Scalping Trading Bot on {host}:{port}")
    logger.info("🧠 Features:")
    logger.info("   - Automated scalping decisions")
    logger.info("   - Real-time pattern recognition")
    logger.info("   - Risk management with ML")
    logger.info("   - Deriv-specific market analysis")
    logger.info("   - Continuous learning system")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
