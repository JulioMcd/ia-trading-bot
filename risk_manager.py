#!/usr/bin/env python3
"""
Sistema de Risk Management Avançado para Trading Bot
Implementa Kelly Criterion, Stop Loss Dinâmico, Circuit Breakers e Controles de Risco
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path
import math

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Métricas de risco em tempo real"""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    consecutive_losses: int = 0
    largest_loss: float = 0.0
    risk_score: float = 0.0

@dataclass
class RiskLimits:
    """Limites de risco configuráveis"""
    max_daily_loss_pct: float = 5.0      # Máximo 5% de perda diária
    max_weekly_loss_pct: float = 10.0    # Máximo 10% de perda semanal
    max_monthly_loss_pct: float = 20.0   # Máximo 20% de perda mensal
    max_drawdown_pct: float = 15.0       # Máximo 15% de drawdown
    max_position_size_pct: float = 2.0   # Máximo 2% por trade
    max_concurrent_trades: int = 3       # Máximo 3 trades simultâneos
    min_win_rate: float = 30.0           # Mínimo 30% win rate
    max_consecutive_losses: int = 5      # Máximo 5 perdas consecutivas
    circuit_breaker_threshold: float = 3.0  # Para tudo se perder 3% em 1 hora

class KellyCalculator:
    """Calculador de Kelly Criterion otimizado"""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        
    def calculate_kelly_fraction(self, trades: List[Dict]) -> float:
        """Calcula fração ótima do Kelly Criterion"""
        if not trades or len(trades) < 10:
            return 0.01  # Posição muito conservadora
        
        # Filtrar trades recentes
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_trades = [
            t for t in trades 
            if datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > cutoff_date
        ]
        
        if len(recent_trades) < 5:
            return 0.01
        
        # Calcular métricas
        wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
        
        if not wins or not losses:
            return 0.01
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = abs(np.mean([t['pnl'] for t in losses]))
        
        if avg_loss == 0:
            return 0.01
        
        # Kelly formula: f = (bp - q) / b
        # b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Aplicar limitações de segurança
        kelly_fraction = max(0, kelly_fraction)  # Não pode ser negativo
        kelly_fraction = min(0.25, kelly_fraction)  # Máximo 25% (Quarter Kelly)
        
        # Aplicar fatores de correção
        confidence_factor = min(1.0, len(recent_trades) / 20)  # Mais dados = mais confiança
        kelly_fraction *= confidence_factor
        
        return kelly_fraction
    
    def calculate_optimal_position_size(self, 
                                      balance: float, 
                                      trades: List[Dict],
                                      risk_limits: RiskLimits) -> float:
        """Calcula tamanho ótimo da posição"""
        kelly_fraction = self.calculate_kelly_fraction(trades)
        
        # Tamanho baseado em Kelly
        kelly_position = balance * kelly_fraction
        
        # Aplicar limites de risco
        max_position_by_limits = balance * (risk_limits.max_position_size_pct / 100)
        
        # Usar o menor entre Kelly e limites de risco
        optimal_size = min(kelly_position, max_position_by_limits)
        
        # Garantir mínimo viável
        min_stake = 0.35  # Mínimo da Deriv
        optimal_size = max(min_stake, optimal_size)
        
        return round(optimal_size, 2)

class CircuitBreaker:
    """Sistema de circuit breaker para parar trading em situações extremas"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.is_triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        self.cool_down_minutes = 60
        
    def check_triggers(self, balance: float, initial_balance: float, 
                      recent_trades: List[Dict]) -> bool:
        """Verifica se algum circuit breaker deve ser ativado"""
        
        # 1. Perda rápida em pouco tempo (3% em 1 hora)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_hour_trades = [
            t for t in recent_trades 
            if datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > one_hour_ago
        ]
        
        if recent_hour_trades:
            hour_pnl = sum(t.get('pnl', 0) for t in recent_hour_trades)
            hour_loss_pct = abs(hour_pnl) / initial_balance * 100
            
            if hour_loss_pct > 3.0:
                self._trigger("Perda rápida: {:.1f}% em 1 hora".format(hour_loss_pct))
                return True
        
        # 2. Drawdown excessivo
        current_drawdown_pct = ((initial_balance - balance) / initial_balance) * 100
        if current_drawdown_pct > 15.0:
            self._trigger("Drawdown excessivo: {:.1f}%".format(current_drawdown_pct))
            return True
        
        # 3. Muitas perdas consecutivas
        consecutive_losses = self._count_consecutive_losses(recent_trades)
        if consecutive_losses >= 7:
            self._trigger("Muitas perdas consecutivas: {}".format(consecutive_losses))
            return True
        
        # 4. Win rate muito baixo com muitos trades
        if len(recent_trades) >= 20:
            wins = len([t for t in recent_trades if t.get('pnl', 0) > 0])
            win_rate = wins / len(recent_trades) * 100
            
            if win_rate < 20:
                self._trigger("Win rate crítico: {:.1f}%".format(win_rate))
                return True
        
        return False
    
    def _trigger(self, reason: str):
        """Ativa o circuit breaker"""
        self.is_triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now()
        
        logger.critical(f"🚨 CIRCUIT BREAKER ATIVADO: {reason}")
        
        # Salvar evento no banco
        self._save_circuit_breaker_event(reason)
    
    def _count_consecutive_losses(self, trades: List[Dict]) -> int:
        """Conta perdas consecutivas recentes"""
        if not trades:
            return 0
        
        # Ordenar por timestamp (mais recente primeiro)
        sorted_trades = sorted(trades, 
                             key=lambda x: x.get('timestamp', '2023-01-01'), 
                             reverse=True)
        
        consecutive = 0
        for trade in sorted_trades:
            if trade.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def can_trade(self) -> Tuple[bool, str]:
        """Verifica se pode fazer trades"""
        if not self.is_triggered:
            return True, ""
        
        # Verificar se passou o período de cool down
        if self.trigger_time:
            cool_down_end = self.trigger_time + timedelta(minutes=self.cool_down_minutes)
            if datetime.now() > cool_down_end:
                self.reset()
                return True, ""
        
        time_remaining = cool_down_end - datetime.now()
        minutes_remaining = int(time_remaining.total_seconds() / 60)
        
        return False, f"Circuit breaker ativo: {self.trigger_reason} (Liberação em {minutes_remaining}min)"
    
    def reset(self):
        """Reset manual do circuit breaker"""
        self.is_triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        logger.info("🔄 Circuit breaker resetado")
    
    def _save_circuit_breaker_event(self, reason: str):
        """Salva evento de circuit breaker"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                INSERT INTO circuit_breaker_events (timestamp, reason)
                VALUES (?, ?)
            ''', (datetime.now().isoformat(), reason))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar evento circuit breaker: {e}")

class AdvancedRiskManager:
    """Sistema de Risk Management Avançado"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.risk_limits = RiskLimits()
        self.kelly_calculator = KellyCalculator()
        self.circuit_breaker = CircuitBreaker(db_path)
        self.current_metrics = RiskMetrics()
        self.initial_balance = 0.0
        self.session_start_balance = 0.0
        self.active_trades_count = 0
        
        # Inicializar banco
        self._init_risk_tables()
    
    def _init_risk_tables(self):
        """Inicializa tabelas de risco"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de snapshots de risco
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                weekly_pnl REAL DEFAULT 0,
                monthly_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                current_drawdown REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                consecutive_losses INTEGER DEFAULT 0,
                risk_score REAL DEFAULT 0,
                kelly_fraction REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de violações de risco
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                action_taken TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def set_initial_balance(self, balance: float):
        """Define saldo inicial para cálculos de risco"""
        self.initial_balance = balance
        self.session_start_balance = balance
        logger.info(f"💰 Saldo inicial definido: ${balance:.2f}")
    
    def update_balance(self, current_balance: float):
        """Atualiza saldo atual e recalcula métricas"""
        if self.initial_balance == 0:
            self.set_initial_balance(current_balance)
        
        self._update_risk_metrics(current_balance)
        self._save_risk_snapshot(current_balance)
    
    def can_place_trade(self, 
                       proposed_stake: float, 
                       current_balance: float,
                       recent_trades: List[Dict]) -> Tuple[bool, str]:
        """Verifica se pode executar trade baseado em todos os controles de risco"""
        
        # 1. Verificar circuit breaker
        can_trade_cb, cb_reason = self.circuit_breaker.can_trade()
        if not can_trade_cb:
            return False, f"🚨 {cb_reason}"
        
        # 2. Verificar se circuit breaker deve ser ativado
        if self.circuit_breaker.check_triggers(current_balance, self.initial_balance, recent_trades):
            return False, "🚨 Circuit breaker ativado automaticamente"
        
        # 3. Verificar limites de posição
        max_position = current_balance * (self.risk_limits.max_position_size_pct / 100)
        if proposed_stake > max_position:
            return False, f"💰 Stake muito alto: ${proposed_stake:.2f} > ${max_position:.2f} (max {self.risk_limits.max_position_size_pct}%)"
        
        # 4. Verificar perdas diárias
        daily_loss_pct = abs(self.current_metrics.daily_pnl) / self.session_start_balance * 100
        if daily_loss_pct >= self.risk_limits.max_daily_loss_pct:
            return False, f"📉 Limite diário atingido: {daily_loss_pct:.1f}% >= {self.risk_limits.max_daily_loss_pct}%"
        
        # 5. Verificar drawdown atual
        current_dd_pct = self.current_metrics.current_drawdown
        if current_dd_pct >= self.risk_limits.max_drawdown_pct:
            return False, f"📊 Drawdown limite atingido: {current_dd_pct:.1f}% >= {self.risk_limits.max_drawdown_pct}%"
        
        # 6. Verificar trades concorrentes
        if self.active_trades_count >= self.risk_limits.max_concurrent_trades:
            return False, f"⚡ Muitos trades simultâneos: {self.active_trades_count} >= {self.risk_limits.max_concurrent_trades}"
        
        # 7. Verificar perdas consecutivas
        if self.current_metrics.consecutive_losses >= self.risk_limits.max_consecutive_losses:
            return False, f"🔴 Muitas perdas consecutivas: {self.current_metrics.consecutive_losses} >= {self.risk_limits.max_consecutive_losses}"
        
        # 8. Verificar win rate mínimo (se tiver dados suficientes)
        if len(recent_trades) >= 20 and self.current_metrics.win_rate < self.risk_limits.min_win_rate:
            return False, f"📈 Win rate baixo: {self.current_metrics.win_rate:.1f}% < {self.risk_limits.min_win_rate}%"
        
        return True, "✅ Trade aprovado pelos controles de risco"
    
    def get_optimal_position_size(self, 
                                 current_balance: float, 
                                 recent_trades: List[Dict]) -> float:
        """Calcula tamanho ótimo da posição usando Kelly Criterion"""
        optimal_size = self.kelly_calculator.calculate_optimal_position_size(
            balance=current_balance,
            trades=recent_trades,
            risk_limits=self.risk_limits
        )
        
        logger.info(f"🎯 Posição ótima calculada: ${optimal_size:.2f}")
        return optimal_size
    
    def _update_risk_metrics(self, current_balance: float):
        """Atualiza métricas de risco"""
        if self.initial_balance == 0:
            return
        
        # PnL da sessão
        session_pnl = current_balance - self.session_start_balance
        self.current_metrics.daily_pnl = session_pnl
        
        # Drawdown atual
        peak_balance = max(self.session_start_balance, current_balance)
        current_dd = (peak_balance - current_balance) / peak_balance * 100
        self.current_metrics.current_drawdown = current_dd
        
        # Atualizar drawdown máximo
        self.current_metrics.max_drawdown = max(self.current_metrics.max_drawdown, current_dd)
        
        # Score de risco geral (0-100, onde 100 = risco máximo)
        risk_score = 0
        
        # Componente: Drawdown (peso 40%)
        dd_score = min(100, (current_dd / self.risk_limits.max_drawdown_pct) * 100) * 0.4
        
        # Componente: Perdas consecutivas (peso 30%)
        loss_score = min(100, (self.current_metrics.consecutive_losses / self.risk_limits.max_consecutive_losses) * 100) * 0.3
        
        # Componente: Win rate (peso 30%)
        wr_score = max(0, 100 - self.current_metrics.win_rate * 2) * 0.3
        
        risk_score = dd_score + loss_score + wr_score
        self.current_metrics.risk_score = min(100, risk_score)
    
    def process_trade_result(self, trade_result: Dict):
        """Processa resultado de trade para atualizar métricas"""
        pnl = trade_result.get('pnl', 0)
        
        if pnl < 0:
            self.current_metrics.consecutive_losses += 1
            self.current_metrics.largest_loss = min(self.current_metrics.largest_loss, pnl)
        else:
            self.current_metrics.consecutive_losses = 0
    
    def add_active_trade(self):
        """Incrementa contador de trades ativos"""
        self.active_trades_count += 1
    
    def remove_active_trade(self):
        """Decrementa contador de trades ativos"""
        self.active_trades_count = max(0, self.active_trades_count - 1)
    
    def _save_risk_snapshot(self, current_balance: float):
        """Salva snapshot das métricas de risco"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            kelly_fraction = self.kelly_calculator.calculate_kelly_fraction([])
            
            cursor.execute('''
                INSERT INTO risk_snapshots 
                (timestamp, balance, daily_pnl, max_drawdown, current_drawdown, 
                 win_rate, consecutive_losses, risk_score, kelly_fraction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                current_balance,
                self.current_metrics.daily_pnl,
                self.current_metrics.max_drawdown,
                self.current_metrics.current_drawdown,
                self.current_metrics.win_rate,
                self.current_metrics.consecutive_losses,
                self.current_metrics.risk_score,
                kelly_fraction
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar snapshot de risco: {e}")
    
    def get_risk_report(self) -> Dict:
        """Gera relatório completo de risco"""
        return {
            "risk_metrics": asdict(self.current_metrics),
            "risk_limits": asdict(self.risk_limits),
            "circuit_breaker": {
                "is_triggered": self.circuit_breaker.is_triggered,
                "trigger_reason": self.circuit_breaker.trigger_reason,
                "trigger_time": self.circuit_breaker.trigger_time.isoformat() if self.circuit_breaker.trigger_time else None
            },
            "active_trades": self.active_trades_count,
            "kelly_stats": {
                "recommended_position_pct": self.kelly_calculator.calculate_kelly_fraction([]) * 100
            },
            "overall_risk_level": self._get_risk_level(),
            "recommendations": self._get_risk_recommendations()
        }
    
    def _get_risk_level(self) -> str:
        """Determina nível de risco atual"""
        score = self.current_metrics.risk_score
        
        if score < 25:
            return "BAIXO"
        elif score < 50:
            return "MODERADO"
        elif score < 75:
            return "ALTO"
        else:
            return "CRÍTICO"
    
    def _get_risk_recommendations(self) -> List[str]:
        """Gera recomendações baseadas no risco atual"""
        recommendations = []
        
        if self.current_metrics.current_drawdown > 10:
            recommendations.append("🔴 Considere reduzir tamanho das posições")
        
        if self.current_metrics.consecutive_losses >= 3:
            recommendations.append("⚠️ Revise estratégia após perdas consecutivas")
        
        if self.current_metrics.win_rate < 40 and self.current_metrics.win_rate > 0:
            recommendations.append("📊 Win rate baixo - ajuste parâmetros de entry")
        
        if self.active_trades_count >= 2:
            recommendations.append("⚡ Evite trades adicionais simultâneos")
        
        if self.current_metrics.risk_score > 75:
            recommendations.append("🚨 Pare de fazer trades até situação melhorar")
        
        return recommendations

# Classe para integração fácil
class RiskManagerIntegration:
    """Wrapper para integração fácil com bot existente"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.risk_manager = AdvancedRiskManager(db_path)
        self.is_enabled = True
    
    def initialize(self, initial_balance: float):
        """Inicializa com saldo inicial"""
        self.risk_manager.set_initial_balance(initial_balance)
    
    def validate_trade(self, stake: float, balance: float, trades: List[Dict]) -> Dict:
        """Valida se pode fazer trade e retorna informações"""
        if not self.is_enabled:
            return {"allowed": True, "reason": "Risk management desabilitado", "optimal_size": stake}
        
        can_trade, reason = self.risk_manager.can_place_trade(stake, balance, trades)
        optimal_size = self.risk_manager.get_optimal_position_size(balance, trades)
        
        return {
            "allowed": can_trade,
            "reason": reason,
            "optimal_size": optimal_size,
            "risk_level": self.risk_manager._get_risk_level(),
            "risk_score": self.risk_manager.current_metrics.risk_score
        }
    
    def update_balance(self, balance: float):
        """Atualiza saldo"""
        self.risk_manager.update_balance(balance)
    
    def on_trade_opened(self):
        """Chama quando trade é aberto"""
        self.risk_manager.add_active_trade()
    
    def on_trade_closed(self, trade_result: Dict):
        """Chama quando trade é fechado"""
        self.risk_manager.remove_active_trade()
        self.risk_manager.process_trade_result(trade_result)
    
    def get_dashboard_data(self) -> Dict:
        """Retorna dados para dashboard"""
        return self.risk_manager.get_risk_report()
    
    def emergency_stop(self):
        """Para tudo imediatamente"""
        self.risk_manager.circuit_breaker._trigger("Parada manual de emergência")
    
    def reset_circuit_breaker(self):
        """Reset manual do circuit breaker"""
        self.risk_manager.circuit_breaker.reset()

# Instância global para uso
risk_manager = RiskManagerIntegration()

if __name__ == "__main__":
    # Exemplo de uso
    print("🛡️ Sistema de Risk Management Avançado")
    print("=" * 50)
    
    # Inicializar
    risk_manager.initialize(1000.0)
    
    # Simular alguns trades
    trades = [
        {"pnl": -10, "timestamp": datetime.now().isoformat()},
        {"pnl": 15, "timestamp": datetime.now().isoformat()},
        {"pnl": -8, "timestamp": datetime.now().isoformat()},
    ]
    
    # Validar trade
    validation = risk_manager.validate_trade(
        stake=50.0, 
        balance=950.0, 
        trades=trades
    )
    
    print(f"Validação: {validation}")
    
    # Obter dados do dashboard
    dashboard = risk_manager.get_dashboard_data()
    print(f"Dashboard: {json.dumps(dashboard, indent=2)}")
