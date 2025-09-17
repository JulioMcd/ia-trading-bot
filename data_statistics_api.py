from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import threading
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configurações
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'trading_data.db')
API_KEY = os.environ.get('API_KEY', 'rnd_qpdTVwAeWzIItVbxHPPCc34uirv9')

@dataclass
class TradeRecord:
    """Classe para representar um trade"""
    id: str
    timestamp: str
    symbol: str
    direction: str
    stake: float
    duration: str
    entry_price: float
    exit_price: Optional[float]
    status: str  # 'open', 'won', 'lost'
    pnl: float
    martingale_level: int
    features: Optional[Dict]
    market_conditions: Optional[Dict]
    ai_confidence: Optional[float]
    ai_analysis: Optional[Dict]

@dataclass
class PerformanceStats:
    """Estatísticas de performance"""
    total_trades: int
    won_trades: int
    lost_trades: int
    win_rate: float
    total_pnl: float
    best_trade: Optional[float]
    worst_trade: Optional[float]
    average_win: float
    average_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    streak_type: str  # 'win' or 'loss'

class TradingDatabase:
    """Classe para gerenciar o banco de dados"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Inicializa o banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                        status TEXT NOT NULL,
                        pnl REAL DEFAULT 0,
                        martingale_level INTEGER DEFAULT 0,
                        features TEXT,
                        market_conditions TEXT,
                        ai_confidence REAL,
                        ai_analysis TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de análises da IA
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        analysis TEXT NOT NULL,
                        context TEXT,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de sinais de trading
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de avaliações de risco
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        assessment TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de dados de mercado sintéticos
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de padrões identificados
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        outcome TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de configurações da IA
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Índices para performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)')
                
                conn.commit()
                logger.info("Banco de dados inicializado com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise
    
    def save_trade(self, trade: TradeRecord) -> bool:
        """Salva um trade no banco"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (id, timestamp, symbol, direction, stake, duration, entry_price, 
                     exit_price, status, pnl, martingale_level, features, 
                     market_conditions, ai_confidence, ai_analysis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.id, trade.timestamp, trade.symbol, trade.direction,
                    trade.stake, trade.duration, trade.entry_price, trade.exit_price,
                    trade.status, trade.pnl, trade.martingale_level,
                    json.dumps(trade.features) if trade.features else None,
                    json.dumps(trade.market_conditions) if trade.market_conditions else None,
                    trade.ai_confidence,
                    json.dumps(trade.ai_analysis) if trade.ai_analysis else None
                ))
                conn.commit()
                logger.info(f"Trade salvo: {trade.id}")
                return True
        except Exception as e:
            logger.error(f"Erro ao salvar trade: {e}")
            return False
    
    def get_trades(self, limit: int = 100, symbol: str = None, 
                   status: str = None, days_back: int = None) -> List[TradeRecord]:
        """Obtém trades do banco"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                if days_back:
                    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                    query += " AND timestamp > ?"
                    params.append(cutoff_date)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trade = TradeRecord(
                        id=row[0], timestamp=row[1], symbol=row[2], direction=row[3],
                        stake=row[4], duration=row[5], entry_price=row[6], exit_price=row[7],
                        status=row[8], pnl=row[9], martingale_level=row[10],
                        features=json.loads(row[11]) if row[11] else None,
                        market_conditions=json.loads(row[12]) if row[12] else None,
                        ai_confidence=row[13],
                        ai_analysis=json.loads(row[14]) if row[14] else None
                    )
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Erro ao obter trades: {e}")
            return []
    
    def calculate_performance_stats(self, days_back: int = 30) -> PerformanceStats:
        """Calcula estatísticas de performance"""
        try:
            trades = self.get_trades(limit=1000, days_back=days_back)
            
            if not trades:
                return PerformanceStats(
                    total_trades=0, won_trades=0, lost_trades=0, win_rate=0.0,
                    total_pnl=0.0, best_trade=None, worst_trade=None,
                    average_win=0.0, average_loss=0.0, max_consecutive_wins=0,
                    max_consecutive_losses=0, current_streak=0, streak_type='none'
                )
            
            completed_trades = [t for t in trades if t.status in ['won', 'lost']]
            won_trades = [t for t in completed_trades if t.status == 'won']
            lost_trades = [t for t in completed_trades if t.status == 'lost']
            
            total_trades = len(completed_trades)
            won_count = len(won_trades)
            lost_count = len(lost_trades)
            win_rate = (won_count / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl for t in completed_trades)
            best_trade = max((t.pnl for t in completed_trades), default=0)
            worst_trade = min((t.pnl for t in completed_trades), default=0)
            
            average_win = sum(t.pnl for t in won_trades) / len(won_trades) if won_trades else 0
            average_loss = sum(t.pnl for t in lost_trades) / len(lost_trades) if lost_trades else 0
            
            # Calcular streaks
            max_consecutive_wins, max_consecutive_losses, current_streak, streak_type = self._calculate_streaks(completed_trades)
            
            return PerformanceStats(
                total_trades=total_trades,
                won_trades=won_count,
                lost_trades=lost_count,
                win_rate=win_rate,
                total_pnl=total_pnl,
                best_trade=best_trade,
                worst_trade=worst_trade,
                average_win=average_win,
                average_loss=average_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                current_streak=current_streak,
                streak_type=streak_type
            )
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return PerformanceStats(
                total_trades=0, won_trades=0, lost_trades=0, win_rate=0.0,
                total_pnl=0.0, best_trade=None, worst_trade=None,
                average_win=0.0, average_loss=0.0, max_consecutive_wins=0,
                max_consecutive_losses=0, current_streak=0, streak_type='none'
            )
    
    def _calculate_streaks(self, trades: List[TradeRecord]) -> tuple:
        """Calcula streaks de vitórias e derrotas"""
        if not trades:
            return 0, 0, 0, 'none'
        
        # Ordenar por timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sorted_trades:
            if trade.status == 'won':
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.status == 'lost':
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Streak atual
        if current_wins > 0:
            return max_wins, max_losses, current_wins, 'win'
        elif current_losses > 0:
            return max_wins, max_losses, current_losses, 'loss'
        else:
            return max_wins, max_losses, 0, 'none'
    
    def save_analysis(self, symbol: str, analysis: Dict, context: Dict, timestamp: str) -> bool:
        """Salva análise da IA"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ai_analyses (symbol, analysis, context, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, json.dumps(analysis), json.dumps(context), timestamp))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Erro ao salvar análise: {e}")
            return False
    
    def save_signal(self, symbol: str, signal: Dict, timestamp: str) -> bool:
        """Salva sinal de trading"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_signals (symbol, signal, timestamp)
                    VALUES (?, ?, ?)
                ''', (symbol, json.dumps(signal), timestamp))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Erro ao salvar sinal: {e}")
            return False
    
    def save_risk_assessment(self, assessment: Dict, timestamp: str) -> bool:
        """Salva avaliação de risco"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO risk_assessments (assessment, timestamp)
                    VALUES (?, ?)
                ''', (json.dumps(assessment), timestamp))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Erro ao salvar avaliação de risco: {e}")
            return False
    
    def save_market_data(self, symbol: str, data: List[Dict]) -> bool:
        """Salva dados de mercado"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for point in data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, point['timestamp'], point['open'], point['high'],
                        point['low'], point['close'], point['volume']
                    ))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Erro ao salvar dados de mercado: {e}")
            return False
    
    def get_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Obtém dados de mercado"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (symbol, limit))
                
                rows = cursor.fetchall()
                return [
                    {
                        'timestamp': row[0],
                        'open': row[1],
                        'high': row[2],
                        'low': row[3],
                        'close': row[4],
                        'volume': row[5]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Erro ao obter dados de mercado: {e}")
            return []
    
    def get_learning_insights(self) -> Dict:
        """Gera insights de aprendizado baseados nos dados históricos"""
        try:
            trades = self.get_trades(limit=500)
            
            if len(trades) < 10:
                return {"message": "Dados insuficientes para insights", "insights": []}
            
            insights = []
            
            # Análise por símbolo
            symbol_performance = {}
            for trade in trades:
                if trade.symbol not in symbol_performance:
                    symbol_performance[trade.symbol] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                if trade.status == 'won':
                    symbol_performance[trade.symbol]['wins'] += 1
                elif trade.status == 'lost':
                    symbol_performance[trade.symbol]['losses'] += 1
                
                symbol_performance[trade.symbol]['total_pnl'] += trade.pnl
            
            # Melhor e pior símbolo
            best_symbol = max(symbol_performance.keys(), 
                            key=lambda s: symbol_performance[s]['total_pnl'])
            worst_symbol = min(symbol_performance.keys(), 
                             key=lambda s: symbol_performance[s]['total_pnl'])
            
            insights.append({
                "type": "symbol_performance",
                "message": f"Melhor desempenho: {best_symbol}, Pior: {worst_symbol}",
                "data": symbol_performance
            })
            
            # Análise por horário
            hourly_performance = {}
            for trade in trades:
                try:
                    hour = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00')).hour
                    if hour not in hourly_performance:
                        hourly_performance[hour] = {'wins': 0, 'losses': 0}
                    
                    if trade.status == 'won':
                        hourly_performance[hour]['wins'] += 1
                    elif trade.status == 'lost':
                        hourly_performance[hour]['losses'] += 1
                except:
                    continue
            
            best_hours = sorted(hourly_performance.keys(), 
                              key=lambda h: hourly_performance[h]['wins'] - hourly_performance[h]['losses'], 
                              reverse=True)[:3]
            
            insights.append({
                "type": "time_analysis",
                "message": f"Melhores horários para trading: {best_hours}",
                "data": hourly_performance
            })
            
            # Análise Martingale
            martingale_stats = {}
            for trade in trades:
                level = trade.martingale_level
                if level not in martingale_stats:
                    martingale_stats[level] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                if trade.status == 'won':
                    martingale_stats[level]['wins'] += 1
                elif trade.status == 'lost':
                    martingale_stats[level]['losses'] += 1
                
                martingale_stats[level]['total_pnl'] += trade.pnl
            
            insights.append({
                "type": "martingale_analysis",
                "message": "Análise de performance por nível de Martingale",
                "data": martingale_stats
            })
            
            # Padrões de direção
            direction_performance = {'CALL': {'wins': 0, 'losses': 0}, 'PUT': {'wins': 0, 'losses': 0}}
            for trade in trades:
                if trade.direction in direction_performance:
                    if trade.status == 'won':
                        direction_performance[trade.direction]['wins'] += 1
                    elif trade.status == 'lost':
                        direction_performance[trade.direction]['losses'] += 1
            
            insights.append({
                "type": "direction_analysis",
                "message": "Performance por direção de trade",
                "data": direction_performance
            })
            
            return {
                "message": f"Insights baseados em {len(trades)} trades",
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar insights: {e}")
            return {"message": "Erro ao gerar insights", "insights": []}

# Instância global do banco
db = TradingDatabase(DATABASE_PATH)

# Funções de geração de dados sintéticos para inicialização
def generate_synthetic_market_data():
    """Gera dados de mercado sintéticos para inicialização"""
    symbols = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100', '1HZ10V', '1HZ25V', '1HZ50V']
    
    for symbol in symbols:
        data_points = []
        current_time = datetime.now()
        base_price = 1000
        
        for i in range(100):
            timestamp = current_time - timedelta(minutes=i)
            
            # Simular movimento de preço
            change = np.random.normal(0, 0.02) * base_price
            base_price += change
            
            data_point = {
                'timestamp': timestamp.isoformat(),
                'open': base_price,
                'high': base_price * (1 + abs(np.random.normal(0, 0.01))),
                'low': base_price * (1 - abs(np.random.normal(0, 0.01))),
                'close': base_price,
                'volume': np.random.randint(1000, 10000)
            }
            data_points.append(data_point)
        
        db.save_market_data(symbol, data_points)

# Endpoints da API

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "Trading Data API",
        "version": "1.0.0",
        "database": "connected",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/trades', methods=['POST'])
def save_trade():
    """Salva um novo trade"""
    try:
        data = request.get_json()
        
        trade = TradeRecord(
            id=data.get('id', ''),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            symbol=data.get('symbol', ''),
            direction=data.get('direction', ''),
            stake=float(data.get('stake', 0)),
            duration=data.get('duration', ''),
            entry_price=float(data.get('entry_price', 0)),
            exit_price=data.get('exit_price'),
            status=data.get('status', 'open'),
            pnl=float(data.get('pnl', 0)),
            martingale_level=int(data.get('martingale_level', 0)),
            features=data.get('features'),
            market_conditions=data.get('market_conditions'),
            ai_confidence=data.get('ai_confidence'),
            ai_analysis=data.get('ai_analysis')
        )
        
        success = db.save_trade(trade)
        
        if success:
            return jsonify({"status": "success", "message": "Trade salvo com sucesso"})
        else:
            return jsonify({"status": "error", "message": "Erro ao salvar trade"}), 500
            
    except Exception as e:
        logger.error(f"Erro ao salvar trade: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/trades', methods=['GET'])
def get_trades():
    """Obtém trades"""
    try:
        limit = int(request.args.get('limit', 100))
        symbol = request.args.get('symbol')
        status = request.args.get('status')
        days_back = int(request.args.get('days_back', 30))
        
        trades = db.get_trades(limit=limit, symbol=symbol, status=status, days_back=days_back)
        
        return jsonify({
            "trades": [asdict(trade) for trade in trades],
            "count": len(trades)
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter trades: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/performance/stats', methods=['GET'])
def get_performance_stats():
    """Obtém estatísticas de performance"""
    try:
        days_back = int(request.args.get('days_back', 30))
        stats = db.calculate_performance_stats(days_back)
        
        return jsonify(asdict(stats))
        
    except Exception as e:
        logger.error(f"Erro ao calcular estatísticas: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/performance/history', methods=['GET'])
def get_performance_history():
    """Obtém histórico de performance para treinamento da IA"""
    try:
        trades = db.get_trades(limit=1000)
        completed_trades = [trade for trade in trades if trade.status in ['won', 'lost']]
        
        # Preparar dados para IA
        training_data = []
        for trade in completed_trades:
            trade_data = {
                'id': trade.id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'stake': trade.stake,
                'martingale_level': trade.martingale_level,
                'result': trade.status,
                'pnl': trade.pnl,
                'features': trade.features,
                'market_conditions': trade.market_conditions,
                'ai_confidence': trade.ai_confidence
            }
            training_data.append(trade_data)
        
        stats = db.calculate_performance_stats()
        
        return jsonify({
            "trades": training_data,
            "total_trades": stats.total_trades,
            "win_rate": stats.win_rate,
            "total_pnl": stats.total_pnl,
            "average_win": stats.average_win,
            "average_loss": stats.average_loss
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter histórico: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-analysis', methods=['POST'])
def save_analysis():
    """Salva análise da IA"""
    try:
        data = request.get_json()
        
        success = db.save_analysis(
            symbol=data.get('symbol', ''),
            analysis=data.get('analysis', {}),
            context=data.get('context', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
        
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"}), 500
            
    except Exception as e:
        logger.error(f"Erro ao salvar análise: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-signal', methods=['POST'])
def save_signal():
    """Salva sinal de trading"""
    try:
        data = request.get_json()
        
        success = db.save_signal(
            symbol=data.get('symbol', ''),
            signal=data.get('signal', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
        
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"}), 500
            
    except Exception as e:
        logger.error(f"Erro ao salvar sinal: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-risk-assessment', methods=['POST'])
def save_risk_assessment():
    """Salva avaliação de risco"""
    try:
        data = request.get_json()
        
        success = db.save_risk_assessment(
            assessment=data.get('assessment', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
        
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"}), 500
            
    except Exception as e:
        logger.error(f"Erro ao salvar avaliação: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/market-data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Obtém dados de mercado para um símbolo"""
    try:
        limit = int(request.args.get('limit', 100))
        data = db.get_market_data(symbol, limit)
        
        return jsonify({
            "symbol": symbol,
            "prices": data,
            "count": len(data)
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter dados de mercado: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/market-data', methods=['POST'])
def save_market_data():
    """Salva dados de mercado"""
    try:
        data = request.get_json()
        
        success = db.save_market_data(
            symbol=data.get('symbol', ''),
            data=data.get('data', [])
        )
        
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"}), 500
            
    except Exception as e:
        logger.error(f"Erro ao salvar dados de mercado: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/insights', methods=['GET'])
def get_learning_insights():
    """Obtém insights de aprendizado"""
    try:
        insights = db.get_learning_insights()
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Erro ao obter insights: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/best-worst-moments', methods=['GET'])
def get_best_worst_moments():
    """Obtém melhores e piores momentos"""
    try:
        trades = db.get_trades(limit=500)
        
        if not trades:
            return jsonify({
                "best_moments": [],
                "worst_moments": [],
                "message": "Nenhum dado disponível"
            })
        
        # Filtrar trades completados
        completed_trades = [t for t in trades if t.status in ['won', 'lost']]
        
        # Melhores momentos (maiores profits)
        best_trades = sorted([t for t in completed_trades if t.pnl > 0], 
                           key=lambda x: x.pnl, reverse=True)[:10]
        
        # Piores momentos (maiores perdas)
        worst_trades = sorted([t for t in completed_trades if t.pnl < 0], 
                            key=lambda x: x.pnl)[:10]
        
        best_moments = []
        for trade in best_trades:
            best_moments.append({
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "direction": trade.direction,
                "pnl": trade.pnl,
                "martingale_level": trade.martingale_level,
                "market_conditions": trade.market_conditions,
                "ai_analysis": trade.ai_analysis
            })
        
        worst_moments = []
        for trade in worst_trades:
            worst_moments.append({
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "direction": trade.direction,
                "pnl": trade.pnl,
                "martingale_level": trade.martingale_level,
                "market_conditions": trade.market_conditions,
                "ai_analysis": trade.ai_analysis
            })
        
        return jsonify({
            "best_moments": best_moments,
            "worst_moments": worst_moments,
            "analysis": {
                "best_avg_pnl": sum(t.pnl for t in best_trades) / len(best_trades) if best_trades else 0,
                "worst_avg_pnl": sum(t.pnl for t in worst_trades) / len(worst_trades) if worst_trades else 0,
                "best_count": len(best_trades),
                "worst_count": len(worst_trades)
            }
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter melhores/piores momentos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Obtém resumo analítico completo"""
    try:
        days_back = int(request.args.get('days_back', 30))
        
        # Estatísticas básicas
        stats = db.calculate_performance_stats(days_back)
        
        # Insights de aprendizado
        insights = db.get_learning_insights()
        
        # Trades recentes
        recent_trades = db.get_trades(limit=20)
        
        return jsonify({
            "period_days": days_back,
            "performance_stats": asdict(stats),
            "learning_insights": insights,
            "recent_activity": [asdict(trade) for trade in recent_trades[:5]],
            "summary": {
                "health_score": min(100, max(0, stats.win_rate)),
                "risk_level": "high" if stats.current_streak < -3 else "medium" if stats.current_streak < 0 else "low",
                "recommendation": self.get_recommendation_based_on_stats(stats)
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao gerar resumo: {e}")
        return jsonify({"error": str(e)}), 500

def get_recommendation_based_on_stats(stats: PerformanceStats) -> str:
    """Gera recomendação baseada nas estatísticas"""
    if stats.win_rate > 70:
        return "Excelente performance! Continue com a estratégia atual."
    elif stats.win_rate > 50:
        return "Performance adequada. Monitore de perto."
    elif stats.win_rate > 30:
        return "Performance abaixo do ideal. Considere ajustar estratégia."
    else:
        return "Performance crítica. Revise completamente a estratégia."

@app.route('/cleanup', methods=['POST'])
def cleanup_old_data():
    """Limpa dados antigos"""
    try:
        days_to_keep = int(request.args.get('days', 90))
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Limpar trades antigos
            cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_date,))
            trades_deleted = cursor.rowcount
            
            # Limpar análises antigas
            cursor.execute('DELETE FROM ai_analyses WHERE timestamp < ?', (cutoff_date,))
            analyses_deleted = cursor.rowcount
            
            # Limpar sinais antigos
            cursor.execute('DELETE FROM trading_signals WHERE timestamp < ?', (cutoff_date,))
            signals_deleted = cursor.rowcount
            
            # Limpar dados de mercado antigos
            cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
            market_data_deleted = cursor.rowcount
            
            conn.commit()
        
        return jsonify({
            "status": "success",
            "deleted": {
                "trades": trades_deleted,
                "analyses": analyses_deleted,
                "signals": signals_deleted,
                "market_data": market_data_deleted
            },
            "cutoff_date": cutoff_date
        })
        
    except Exception as e:
        logger.error(f"Erro na limpeza: {e}")
        return jsonify({"error": str(e)}), 500

# Inicialização da aplicação
if __name__ == '__main__':
    # Gerar alguns dados sintéticos na primeira execução
    try:
        trades = db.get_trades(limit=10)
        if len(trades) < 5:
            logger.info("Gerando dados sintéticos iniciais...")
            generate_synthetic_market_data()
    except:
        pass
    
    port = int(os.environ.get('PORT', 10001))
    app.run(host='0.0.0.0', port=port, debug=False)