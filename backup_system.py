#!/usr/bin/env python3
"""
Sistema Avançado de Estatísticas e Aprendizado para Trading Bot IA
Implementa salvamento robusto de dados e aprendizado por erros
"""

import os
import json
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import time
import pickle
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Estrutura completa para resultado de trade"""
    id: str
    timestamp: datetime
    symbol: str
    direction: str  # 'call' ou 'put'
    entry_price: float
    exit_price: float
    stake: float
    duration_planned: int
    duration_actual: int
    pnl: float
    pnl_percentage: float
    status: str  # 'won', 'lost', 'cancelled'
    
    # Dados contextuais
    market_conditions: Dict[str, Any]
    ai_confidence: float
    ai_reasoning: str
    entry_features: List[float]
    martingale_level: int
    
    # Análise pós-trade
    exit_reason: str
    error_type: Optional[str] = None
    lessons_learned: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['market_conditions'] = json.dumps(self.market_conditions)
        result['entry_features'] = json.dumps(self.entry_features)
        return result

@dataclass
class PerformanceMetrics:
    """Métricas de performance detalhadas"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    
    # Métricas por período
    daily_pnl: Dict[str, float] = None
    weekly_pnl: Dict[str, float] = None
    monthly_pnl: Dict[str, float] = None
    
    def __post_init__(self):
        if self.daily_pnl is None:
            self.daily_pnl = {}
        if self.weekly_pnl is None:
            self.weekly_pnl = {}
        if self.monthly_pnl is None:
            self.monthly_pnl = {}

class StatsDatabase:
    """Sistema robusto de persistência de dados"""
    
    def __init__(self, db_path: str = "data/trading_stats.db"):
        self.db_path = db_path
        Path("data").mkdir(exist_ok=True)
        self.init_database()
        
    def init_database(self):
        """Inicializa banco de dados com todas as tabelas necessárias"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela principal de trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                stake REAL NOT NULL,
                duration_planned INTEGER,
                duration_actual INTEGER,
                pnl REAL NOT NULL,
                pnl_percentage REAL NOT NULL,
                status TEXT NOT NULL,
                market_conditions TEXT,
                ai_confidence REAL,
                ai_reasoning TEXT,
                entry_features TEXT,
                martingale_level INTEGER DEFAULT 0,
                exit_reason TEXT,
                error_type TEXT,
                lessons_learned TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de métricas de performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metrics TEXT NOT NULL,
                period_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de padrões de erro identificados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                pattern_type TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                avg_loss REAL NOT NULL,
                market_conditions TEXT,
                features_involved TEXT,
                first_occurrence TEXT NOT NULL,
                last_occurrence TEXT NOT NULL,
                severity_score REAL DEFAULT 0.0,
                mitigation_strategy TEXT,
                is_resolved BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de estratégias de sucesso
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS success_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                pattern_type TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                avg_profit REAL NOT NULL,
                win_rate REAL NOT NULL,
                market_conditions TEXT,
                features_involved TEXT,
                first_occurrence TEXT NOT NULL,
                last_occurrence TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                replication_strategy TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de aprendizados e insights
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                data_points_analyzed INTEGER,
                confidence_level REAL,
                action_recommended TEXT,
                impact_score REAL DEFAULT 0.0,
                implementation_status TEXT DEFAULT 'pending',
                results_after_implementation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de configurações adaptativas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                config_type TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                reason TEXT,
                performance_before TEXT,
                performance_after TEXT,
                is_reverted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Advanced database initialized successfully")
    
    def save_trade(self, trade: TradeResult) -> bool:
        """Salva trade completo no banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            trade_dict = trade.to_dict()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                trade_dict['id'],
                trade_dict['timestamp'],
                trade_dict['symbol'],
                trade_dict['direction'],
                trade_dict['entry_price'],
                trade_dict['exit_price'],
                trade_dict['stake'],
                trade_dict['duration_planned'],
                trade_dict['duration_actual'],
                trade_dict['pnl'],
                trade_dict['pnl_percentage'],
                trade_dict['status'],
                trade_dict['market_conditions'],
                trade_dict['ai_confidence'],
                trade_dict['ai_reasoning'],
                trade_dict['entry_features'],
                trade_dict['martingale_level'],
                trade_dict['exit_reason'],
                trade_dict['error_type'],
                trade_dict['lessons_learned']
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    def get_trades(self, 
                   limit: int = 1000, 
                   symbol: str = None, 
                   start_date: datetime = None, 
                   end_date: datetime = None) -> List[Dict]:
        """Recupera trades com filtros"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

class ErrorPatternAnalyzer:
    """Analisador avançado de padrões de erro"""
    
    def __init__(self, stats_db: StatsDatabase):
        self.stats_db = stats_db
        self.error_clusterer = DBSCAN(eps=0.3, min_samples=3)
        self.scaler = StandardScaler()
        
    def analyze_loss_patterns(self, recent_days: int = 30) -> Dict[str, Any]:
        """Analisa padrões de perdas nos últimos dias"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=recent_days)
        
        # Buscar apenas trades perdedores
        trades = self.stats_db.get_trades(
            limit=5000, 
            start_date=start_date, 
            end_date=end_date
        )
        
        losing_trades = [t for t in trades if t['status'] == 'lost']
        
        if len(losing_trades) < 5:
            return {'error': 'Insufficient data for pattern analysis'}
        
        # Extrair features dos trades perdedores
        features_matrix = []
        trade_contexts = []
        
        for trade in losing_trades:
            try:
                entry_features = json.loads(trade['entry_features'])
                market_conditions = json.loads(trade['market_conditions'])
                
                # Criar vetor de características
                feature_vector = [
                    trade['ai_confidence'],
                    trade['pnl'],
                    trade['duration_actual'],
                    trade['martingale_level'],
                    len(entry_features),  # Complexidade dos dados
                    market_conditions.get('volatility', 0),
                    market_conditions.get('trend_strength', 0),
                    trade['stake']
                ]
                
                features_matrix.append(feature_vector)
                trade_contexts.append({
                    'id': trade['id'],
                    'symbol': trade['symbol'],
                    'direction': trade['direction'],
                    'timestamp': trade['timestamp'],
                    'ai_reasoning': trade['ai_reasoning'],
                    'exit_reason': trade['exit_reason']
                })
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error processing trade {trade['id']}: {e}")
                continue
        
        if len(features_matrix) < 3:
            return {'error': 'Insufficient valid data for clustering'}
        
        # Clustering de padrões de erro
        features_scaled = self.scaler.fit_transform(features_matrix)
        clusters = self.error_clusterer.fit_predict(features_scaled)
        
        # Analisar clusters encontrados
        pattern_analysis = {}
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise no DBSCAN
                continue
                
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_trades = [trade_contexts[i] for i in cluster_indices]
            cluster_features = [features_matrix[i] for i in cluster_indices]
            
            # Estatísticas do cluster
            avg_loss = np.mean([losing_trades[i]['pnl'] for i in cluster_indices])
            avg_confidence = np.mean([f[0] for f in cluster_features])
            
            # Identificar características comuns
            common_symbols = list(set([t['symbol'] for t in cluster_trades]))
            common_directions = list(set([t['direction'] for t in cluster_trades]))
            common_exit_reasons = list(set([t['exit_reason'] for t in cluster_trades if t['exit_reason']]))
            
            pattern_analysis[f'error_pattern_{cluster_id}'] = {
                'frequency': len(cluster_trades),
                'avg_loss': avg_loss,
                'avg_confidence': avg_confidence,
                'common_symbols': common_symbols,
                'common_directions': common_directions,
                'common_exit_reasons': common_exit_reasons,
                'severity_score': abs(avg_loss) * len(cluster_trades),
                'sample_trades': cluster_trades[:3],  # Amostras para análise
                'mitigation_suggestions': self._generate_mitigation_strategy(
                    cluster_features, cluster_trades
                )
            }
        
        return {
            'analysis_period': f'{recent_days} days',
            'total_losses_analyzed': len(losing_trades),
            'patterns_found': len(pattern_analysis),
            'patterns': pattern_analysis,
            'overall_insights': self._generate_overall_insights(pattern_analysis)
        }
    
    def _generate_mitigation_strategy(self, features: List[List], trades: List[Dict]) -> List[str]:
        """Gera estratégias de mitigação baseadas nos padrões"""
        strategies = []
        
        avg_confidence = np.mean([f[0] for f in features])
        avg_martingale = np.mean([f[3] for f in features])
        common_symbols = [t['symbol'] for t in trades]
        
        if avg_confidence > 0.8:
            strategies.append("Reduzir threshold de confiança - overconfidence detectada")
        
        if avg_martingale > 3:
            strategies.append("Implementar limite mais baixo de Martingale")
        
        if len(set(common_symbols)) == 1:
            strategies.append(f"Evitar temporariamente o símbolo {common_symbols[0]}")
        
        if len(strategies) == 0:
            strategies.append("Monitorar padrão emergente - coletar mais dados")
        
        return strategies
    
    def _generate_overall_insights(self, patterns: Dict) -> List[str]:
        """Gera insights gerais baseados em todos os padrões"""
        insights = []
        
        if not patterns:
            return ["Nenhum padrão de erro significativo detectado"]
        
        total_frequency = sum(p['frequency'] for p in patterns.values())
        most_severe = max(patterns.values(), key=lambda x: x['severity_score'])
        
        insights.append(f"Total de {total_frequency} perdas analisadas em {len(patterns)} padrões distintos")
        insights.append(f"Padrão mais severo: {most_severe['severity_score']:.2f} pontos de severidade")
        
        # Insights sobre símbolos
        all_symbols = []
        for pattern in patterns.values():
            all_symbols.extend(pattern['common_symbols'])
        
        from collections import Counter
        symbol_frequency = Counter(all_symbols)
        if symbol_frequency:
            most_problematic = symbol_frequency.most_common(1)[0]
            insights.append(f"Símbolo mais problemático: {most_problematic[0]} ({most_problematic[1]} padrões)")
        
        return insights

class SuccessPatternAnalyzer:
    """Analisador de padrões de sucesso para replicação"""
    
    def __init__(self, stats_db: StatsDatabase):
        self.stats_db = stats_db
        
    def analyze_winning_patterns(self, min_profit: float = 5.0, recent_days: int = 30) -> Dict[str, Any]:
        """Analisa padrões de trades vencedores para replicação"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=recent_days)
        
        trades = self.stats_db.get_trades(
            limit=5000, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # Filtrar apenas trades vencedores significativos
        winning_trades = [t for t in trades if t['status'] == 'won' and t['pnl'] >= min_profit]
        
        if len(winning_trades) < 5:
            return {'error': 'Insufficient winning trades for analysis'}
        
        # Agrupar por características similares
        patterns = {}
        
        for trade in winning_trades:
            try:
                # Criar chave do padrão baseada em características
                pattern_key = f"{trade['symbol']}_{trade['direction']}_{trade['martingale_level']}"
                
                if pattern_key not in patterns:
                    patterns[pattern_key] = {
                        'trades': [],
                        'total_profit': 0,
                        'win_count': 0,
                        'avg_confidence': 0,
                        'avg_duration': 0
                    }
                
                patterns[pattern_key]['trades'].append(trade)
                patterns[pattern_key]['total_profit'] += trade['pnl']
                patterns[pattern_key]['win_count'] += 1
                patterns[pattern_key]['avg_confidence'] += trade['ai_confidence']
                patterns[pattern_key]['avg_duration'] += trade['duration_actual']
                
            except Exception as e:
                logger.warning(f"Error processing winning trade: {e}")
                continue
        
        # Analisar padrões mais lucrativos
        success_analysis = {}
        
        for pattern_key, data in patterns.items():
            if data['win_count'] < 3:  # Mínimo de 3 trades para considerar padrão
                continue
            
            avg_profit = data['total_profit'] / data['win_count']
            avg_confidence = data['avg_confidence'] / data['win_count']
            avg_duration = data['avg_duration'] / data['win_count']
            
            success_analysis[pattern_key] = {
                'frequency': data['win_count'],
                'total_profit': data['total_profit'],
                'avg_profit': avg_profit,
                'avg_confidence': avg_confidence,
                'avg_duration': avg_duration,
                'profitability_score': avg_profit * data['win_count'],
                'sample_trades': data['trades'][:3],
                'replication_strategy': self._generate_replication_strategy(data['trades'])
            }
        
        # Ordenar por profitabilidade
        sorted_patterns = sorted(
            success_analysis.items(), 
            key=lambda x: x[1]['profitability_score'], 
            reverse=True
        )
        
        return {
            'analysis_period': f'{recent_days} days',
            'winning_trades_analyzed': len(winning_trades),
            'profitable_patterns': len(success_analysis),
            'top_patterns': dict(sorted_patterns[:5]),  # Top 5
            'optimization_suggestions': self._generate_optimization_suggestions(sorted_patterns)
        }
    
    def _generate_replication_strategy(self, trades: List[Dict]) -> List[str]:
        """Gera estratégias para replicar padrões de sucesso"""
        strategies = []
        
        # Analisar condições comuns
        symbols = [t['symbol'] for t in trades]
        directions = [t['direction'] for t in trades]
        confidences = [t['ai_confidence'] for t in trades]
        
        from collections import Counter
        
        # Símbolo mais comum
        common_symbol = Counter(symbols).most_common(1)[0]
        if common_symbol[1] / len(trades) > 0.7:
            strategies.append(f"Focar no símbolo {common_symbol[0]} - alta taxa de sucesso")
        
        # Direção predominante
        common_direction = Counter(directions).most_common(1)[0]
        if common_direction[1] / len(trades) > 0.6:
            strategies.append(f"Preferir direção {common_direction[0]} neste padrão")
        
        # Nível de confiança
        avg_confidence = np.mean(confidences)
        if avg_confidence > 0.8:
            strategies.append(f"Manter confiança alta (>{avg_confidence:.2f}) para este padrão")
        
        return strategies
    
    def _generate_optimization_suggestions(self, sorted_patterns: List) -> List[str]:
        """Gera sugestões de otimização baseadas nos padrões de sucesso"""
        suggestions = []
        
        if not sorted_patterns:
            return ["Coletar mais dados de trades vencedores"]
        
        best_pattern = sorted_patterns[0][1]
        suggestions.append(
            f"Aumentar frequência do padrão mais lucrativo "
            f"(Score: {best_pattern['profitability_score']:.2f})"
        )
        
        # Analisar todas as características dos melhores padrões
        top_3_patterns = sorted_patterns[:3]
        all_strategies = []
        
        for _, pattern_data in top_3_patterns:
            all_strategies.extend(pattern_data['replication_strategy'])
        
        # Estratégias mais comuns
        from collections import Counter
        common_strategies = Counter(all_strategies).most_common(3)
        
        for strategy, frequency in common_strategies:
            suggestions.append(f"Aplicar estratégia recorrente: {strategy}")
        
        return suggestions

class PerformanceTracker:
    """Sistema avançado de tracking de performance"""
    
    def __init__(self, stats_db: StatsDatabase):
        self.stats_db = stats_db
        self.current_metrics = PerformanceMetrics()
        self.performance_history = deque(maxlen=1000)
        
    def update_metrics(self, trade: TradeResult):
        """Atualiza métricas com novo trade"""
        self.current_metrics.total_trades += 1
        
        if trade.status == 'won':
            self.current_metrics.winning_trades += 1
            self.current_metrics.consecutive_wins += 1
            self.current_metrics.consecutive_losses = 0
            
            if self.current_metrics.consecutive_wins > self.current_metrics.max_consecutive_wins:
                self.current_metrics.max_consecutive_wins = self.current_metrics.consecutive_wins
                
            if trade.pnl > self.current_metrics.largest_win:
                self.current_metrics.largest_win = trade.pnl
                
        elif trade.status == 'lost':
            self.current_metrics.losing_trades += 1
            self.current_metrics.consecutive_losses += 1
            self.current_metrics.consecutive_wins = 0
            
            if self.current_metrics.consecutive_losses > self.current_metrics.max_consecutive_losses:
                self.current_metrics.max_consecutive_losses = self.current_metrics.consecutive_losses
                
            if trade.pnl < self.current_metrics.largest_loss:
                self.current_metrics.largest_loss = trade.pnl
        
        # Atualizar PnL total
        self.current_metrics.total_pnl += trade.pnl
        
        # Calcular métricas derivadas
        self._calculate_derived_metrics()
        
        # Atualizar métricas por período
        self._update_period_metrics(trade)
        
        # Salvar snapshot
        self._save_performance_snapshot()
    
    def _calculate_derived_metrics(self):
        """Calcula métricas derivadas"""
        if self.current_metrics.total_trades > 0:
            self.current_metrics.win_rate = (
                self.current_metrics.winning_trades / self.current_metrics.total_trades
            )
        
        if self.current_metrics.winning_trades > 0:
            self.current_metrics.average_win = (
                sum([h['pnl'] for h in self.performance_history if h['pnl'] > 0]) /
                self.current_metrics.winning_trades
            )
        
        if self.current_metrics.losing_trades > 0:
            self.current_metrics.average_loss = (
                sum([h['pnl'] for h in self.performance_history if h['pnl'] < 0]) /
                self.current_metrics.losing_trades
            )
        
        # Profit Factor
        total_wins = sum([h['pnl'] for h in self.performance_history if h['pnl'] > 0])
        total_losses = abs(sum([h['pnl'] for h in self.performance_history if h['pnl'] < 0]))
        
        if total_losses > 0:
            self.current_metrics.profit_factor = total_wins / total_losses
        
        # Drawdown
        self._calculate_drawdown()
        
        # Sharpe Ratio
        self._calculate_sharpe_ratio()
    
    def _calculate_drawdown(self):
        """Calcula drawdown atual e máximo"""
        if len(self.performance_history) < 2:
            return
        
        # Calcular equity curve
        equity_curve = []
        running_total = 0
        
        for trade in self.performance_history:
            running_total += trade['pnl']
            equity_curve.append(running_total)
        
        # Encontrar picos e drawdowns
        peak = equity_curve[0]
        max_dd = 0
        current_dd = 0
        
        for equity in equity_curve[1:]:
            if equity > peak:
                peak = equity
                current_dd = 0
            else:
                current_dd = (peak - equity) / max(peak, 1)
                if current_dd > max_dd:
                    max_dd = current_dd
        
        self.current_metrics.max_drawdown = max_dd
        self.current_metrics.current_drawdown = current_dd
    
    def _calculate_sharpe_ratio(self):
        """Calcula Sharpe Ratio"""
        if len(self.performance_history) < 10:
            return
        
        returns = [trade['pnl'] for trade in self.performance_history[-30:]]  # Últimos 30 trades
        
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                self.current_metrics.sharpe_ratio = avg_return / std_return
    
    def _update_period_metrics(self, trade: TradeResult):
        """Atualiza métricas por período (diário, semanal, mensal)"""
        trade_date = trade.timestamp.date()
        
        # PnL diário
        date_key = trade_date.isoformat()
        if date_key not in self.current_metrics.daily_pnl:
            self.current_metrics.daily_pnl[date_key] = 0
        self.current_metrics.daily_pnl[date_key] += trade.pnl
        
        # PnL semanal
        week_start = trade_date - timedelta(days=trade_date.weekday())
        week_key = week_start.isoformat()
        if week_key not in self.current_metrics.weekly_pnl:
            self.current_metrics.weekly_pnl[week_key] = 0
        self.current_metrics.weekly_pnl[week_key] += trade.pnl
        
        # PnL mensal
        month_key = f"{trade_date.year}-{trade_date.month:02d}"
        if month_key not in self.current_metrics.monthly_pnl:
            self.current_metrics.monthly_pnl[month_key] = 0
        self.current_metrics.monthly_pnl[month_key] += trade.pnl
        
        # Adicionar ao histórico
        self.performance_history.append({
            'timestamp': trade.timestamp,
            'pnl': trade.pnl,
            'cumulative_pnl': self.current_metrics.total_pnl,
            'trade_id': trade.id
        })
    
    def _save_performance_snapshot(self):
        """Salva snapshot das métricas no banco"""
        try:
            conn = sqlite3.connect(self.stats_db.db_path)
            cursor = conn.cursor()
            
            metrics_json = json.dumps(asdict(self.current_metrics), default=str)
            
            cursor.execute('''
                INSERT INTO performance_snapshots (timestamp, metrics, period_type)
                VALUES (?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics_json,
                'real_time'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")
    
    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Gera relatório completo de performance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Buscar trades do período
        period_trades = self.stats_db.get_trades(
            limit=10000,
            start_date=start_date,
            end_date=end_date
        )
        
        # Estatísticas do período
        period_pnl = sum([t['pnl'] for t in period_trades])
        period_wins = len([t for t in period_trades if t['status'] == 'won'])
        period_losses = len([t for t in period_trades if t['status'] == 'lost'])
        period_win_rate = period_wins / len(period_trades) if period_trades else 0
        
        # Análise por símbolo
        symbol_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
        for trade in period_trades:
            symbol = trade['symbol']
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade['pnl']
            if trade['status'] == 'won':
                symbol_stats[symbol]['wins'] += 1
        
        # Melhores e piores momentos
        best_day = max(self.current_metrics.daily_pnl.items(), key=lambda x: x[1]) if self.current_metrics.daily_pnl else ('N/A', 0)
        worst_day = min(self.current_metrics.daily_pnl.items(), key=lambda x: x[1]) if self.current_metrics.daily_pnl else ('N/A', 0)
        
        return {
            'period_analysis': {
                'days_analyzed': days,
                'total_trades': len(period_trades),
                'total_pnl': period_pnl,
                'win_rate': period_win_rate,
                'wins': period_wins,
                'losses': period_losses
            },
            'current_metrics': asdict(self.current_metrics),
            'symbol_breakdown': dict(symbol_stats),
            'best_worst_moments': {
                'best_day': {'date': best_day[0], 'pnl': best_day[1]},
                'worst_day': {'date': worst_day[0], 'pnl': worst_day[1]},
                'largest_win': self.current_metrics.largest_win,
                'largest_loss': self.current_metrics.largest_loss
            },
            'recent_performance_trend': self._analyze_recent_trend()
        }
    
    def _analyze_recent_trend(self) -> Dict[str, Any]:
        """Analisa tendência de performance recente"""
        if len(self.performance_history) < 10:
            return {'trend': 'insufficient_data'}
        
        recent_pnls = [trade['pnl'] for trade in list(self.performance_history)[-10:]]
        older_pnls = [trade['pnl'] for trade in list(self.performance_history)[-20:-10]]
        
        if not older_pnls:
            return {'trend': 'insufficient_data'}
        
        recent_avg = np.mean(recent_pnls)
        older_avg = np.mean(older_pnls)
        
        trend_direction = 'improving' if recent_avg > older_avg else 'declining'
        trend_strength = abs(recent_avg - older_avg) / max(abs(older_avg), 1)
        
        return {
            'trend': trend_direction,
            'strength': trend_strength,
            'recent_avg_pnl': recent_avg,
            'previous_avg_pnl': older_avg,
            'change_percentage': ((recent_avg - older_avg) / max(abs(older_avg), 1)) * 100
        }

# Classe principal que integra tudo
class IntelligentTradingSystem:
    """Sistema inteligente de trading com aprendizado contínuo"""
    
    def __init__(self):
        self.stats_db = StatsDatabase()
        self.error_analyzer = ErrorPatternAnalyzer(self.stats_db)
        self.success_analyzer = SuccessPatternAnalyzer(self.stats_db)
        self.performance_tracker = PerformanceTracker(self.stats_db)
        
        # Load previous state if exists
        self._load_system_state()
    
    def record_trade(self, trade_data: Dict) -> TradeResult:
        """Registra um trade completo e atualiza sistemas de aprendizado"""
        try:
            # Criar objeto TradeResult
            trade = TradeResult(
                id=trade_data['id'],
                timestamp=datetime.fromisoformat(trade_data['timestamp']) if isinstance(trade_data['timestamp'], str) else trade_data['timestamp'],
                symbol=trade_data['symbol'],
                direction=trade_data['direction'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                stake=trade_data['stake'],
                duration_planned=trade_data.get('duration_planned', 0),
                duration_actual=trade_data.get('duration_actual', 0),
                pnl=trade_data['pnl'],
                pnl_percentage=trade_data.get('pnl_percentage', 0),
                status=trade_data['status'],
                market_conditions=trade_data.get('market_conditions', {}),
                ai_confidence=trade_data.get('ai_confidence', 0.5),
                ai_reasoning=trade_data.get('ai_reasoning', ''),
                entry_features=trade_data.get('entry_features', []),
                martingale_level=trade_data.get('martingale_level', 0),
                exit_reason=trade_data.get('exit_reason', 'unknown'),
                error_type=trade_data.get('error_type'),
                lessons_learned=trade_data.get('lessons_learned')
            )
            
            # Salvar no banco
            success = self.stats_db.save_trade(trade)
            
            if success:
                # Atualizar métricas de performance
                self.performance_tracker.update_metrics(trade)
                
                # Trigger análises periódicas
                if self.performance_tracker.current_metrics.total_trades % 50 == 0:
                    self._run_periodic_analysis()
                
                logger.info(f"Trade {trade.id} recorded successfully")
                return trade
            else:
                logger.error(f"Failed to save trade {trade.id}")
                return None
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return None
    
    def _run_periodic_analysis(self):
        """Executa análises periódicas para insights automáticos"""
        try:
            # Análise de padrões de erro
            error_analysis = self.error_analyzer.analyze_loss_patterns()
            
            if 'patterns' in error_analysis:
                for pattern_name, pattern_data in error_analysis['patterns'].items():
                    # Salvar insights sobre erros
                    self._save_ai_insight(
                        insight_type='error_pattern',
                        description=f"Padrão de erro detectado: {pattern_name}",
                        confidence_level=min(pattern_data['severity_score'] / 100, 1.0),
                        action_recommended='; '.join(pattern_data['mitigation_suggestions'])
                    )
            
            # Análise de padrões de sucesso
            success_analysis = self.success_analyzer.analyze_winning_patterns()
            
            if 'top_patterns' in success_analysis:
                for pattern_name, pattern_data in success_analysis['top_patterns'].items():
                    # Salvar insights sobre sucessos
                    self._save_ai_insight(
                        insight_type='success_pattern',
                        description=f"Padrão de sucesso identificado: {pattern_name}",
                        confidence_level=min(pattern_data['profitability_score'] / 100, 1.0),
                        action_recommended='; '.join(pattern_data['replication_strategy'])
                    )
            
            logger.info("Periodic analysis completed")
            
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
    
    def _save_ai_insight(self, insight_type: str, description: str, 
                        confidence_level: float, action_recommended: str):
        """Salva insight da IA no banco"""
        try:
            conn = sqlite3.connect(self.stats_db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_insights 
                (timestamp, insight_type, description, confidence_level, action_recommended)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                insight_type,
                description,
                confidence_level,
                action_recommended
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving AI insight: {e}")
    
    def get_ai_recommendations(self) -> Dict[str, Any]:
        """Obtém recomendações atuais da IA baseadas no aprendizado"""
        try:
            # Buscar insights recentes
            conn = sqlite3.connect(self.stats_db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM ai_insights 
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY confidence_level DESC, timestamp DESC
                LIMIT 10
            ''')
            
            insights = cursor.fetchall()
            conn.close()
            
            # Análise atual de performance
            performance_report = self.performance_tracker.get_performance_report(days=7)
            
            # Gerar recomendações baseadas em dados
            recommendations = {
                'immediate_actions': [],
                'strategic_adjustments': [],
                'risk_warnings': [],
                'optimization_opportunities': []
            }
            
            # Processar insights
            for insight in insights:
                insight_type = insight[2]
                description = insight[3]
                confidence = insight[5]
                action = insight[6]
                
                if confidence > 0.7:
                    if insight_type == 'error_pattern':
                        recommendations['immediate_actions'].append({
                            'type': 'error_mitigation',
                            'description': description,
                            'action': action,
                            'confidence': confidence
                        })
                    elif insight_type == 'success_pattern':
                        recommendations['optimization_opportunities'].append({
                            'type': 'success_replication',
                            'description': description,
                            'action': action,
                            'confidence': confidence
                        })
            
            # Adicionar recomendações baseadas em performance
            current_metrics = performance_report['current_metrics']
            
            if current_metrics['win_rate'] < 0.4:
                recommendations['risk_warnings'].append({
                    'type': 'low_win_rate',
                    'description': f"Taxa de vitória baixa: {current_metrics['win_rate']:.1%}",
                    'action': 'Considerar pausa para análise ou ajuste de parâmetros',
                    'confidence': 0.9
                })
            
            if current_metrics['current_drawdown'] > 0.15:
                recommendations['risk_warnings'].append({
                    'type': 'high_drawdown',
                    'description': f"Drawdown alto: {current_metrics['current_drawdown']:.1%}",
                    'action': 'Reduzir tamanho de posições ou pausar trading',
                    'confidence': 0.95
                })
            
            if performance_report['recent_performance_trend']['trend'] == 'improving':
                recommendations['strategic_adjustments'].append({
                    'type': 'performance_improving',
                    'description': 'Tendência de melhoria detectada',
                    'action': 'Considerar aumento gradual de agressividade',
                    'confidence': 0.7
                })
            
            return {
                'recommendations': recommendations,
                'performance_summary': performance_report,
                'last_analysis': datetime.now().isoformat(),
                'system_health': self._assess_system_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return {'error': str(e)}
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Avalia saúde geral do sistema"""
        try:
            metrics = self.performance_tracker.current_metrics
            
            health_score = 0
            issues = []
            
            # Avaliar win rate
            if metrics.win_rate >= 0.6:
                health_score += 30
            elif metrics.win_rate >= 0.5:
                health_score += 20
            else:
                issues.append("Taxa de vitória abaixo do ideal")
            
            # Avaliar profit factor
            if metrics.profit_factor >= 1.5:
                health_score += 25
            elif metrics.profit_factor >= 1.0:
                health_score += 15
            else:
                issues.append("Profit factor baixo")
            
            # Avaliar drawdown
            if metrics.current_drawdown <= 0.05:
                health_score += 25
            elif metrics.current_drawdown <= 0.15:
                health_score += 15
            else:
                issues.append("Drawdown elevado")
            
            # Avaliar consistência
            if metrics.max_consecutive_losses <= 5:
                health_score += 20
            elif metrics.max_consecutive_losses <= 10:
                health_score += 10
            else:
                issues.append("Sequências de perda muito longas")
            
            health_status = 'excellent' if health_score >= 80 else \
                           'good' if health_score >= 60 else \
                           'fair' if health_score >= 40 else 'poor'
            
            return {
                'health_score': health_score,
                'health_status': health_status,
                'issues_identified': issues,
                'data_quality': 'good' if metrics.total_trades >= 100 else 'limited'
            }
            
        except Exception as e:
            return {'health_status': 'unknown', 'error': str(e)}
    
    def _load_system_state(self):
        """Carrega estado anterior do sistema"""
        try:
            # Carregar métricas mais recentes
            conn = sqlite3.connect(self.stats_db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT metrics FROM performance_snapshots 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            
            if result:
                metrics_data = json.loads(result[0])
                # Reconstruir métricas (conversão pode ser necessária)
                logger.info("Previous system state loaded")
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}")
    
    def _save_system_state(self):
        """Salva estado atual do sistema"""
        try:
            # Estado já é salvo automaticamente via performance_tracker
            pass
        except Exception as e:
            logger.error(f"Error saving system state: {e}")

# Instância global do sistema
intelligent_trading_system = IntelligentTradingSystem()
