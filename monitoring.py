import logging
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Configuração de logging avançado
def setup_logging():
    """Configura sistema de logging avançado"""
    # Criar diretório de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formatação
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(
        log_dir / f"ml_trading_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    
    # Configurar logger principal
    logger = logging.getLogger("MLTrading")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class MLMetrics:
    """Métricas de performance do ML"""
    timestamp: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    correct_predictions: int
    training_data_size: int
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]

@dataclass
class TradingMetrics:
    """Métricas de trading"""
    timestamp: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    ml_influenced_trades: int
    ml_accuracy_in_trades: float

class MLMonitor:
    """Sistema de monitoramento para ML Trading"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.logger = setup_logging()
        self.initialize_monitoring_db()
        
    def initialize_monitoring_db(self):
        """Inicializa banco de dados de monitoramento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de métricas ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                accuracy REAL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                training_data_size INTEGER,
                feature_importance TEXT,
                confusion_matrix TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de métricas de trading
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                ml_influenced_trades INTEGER,
                ml_accuracy_in_trades REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de alertas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                data TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de performance por símbolo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                total_trades INTEGER,
                win_rate REAL,
                avg_pnl REAL,
                ml_accuracy REAL,
                volatility_avg REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("Banco de monitoramento inicializado")
        
    def log_ml_metrics(self, metrics: MLMetrics):
        """Registra métricas de ML"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ml_metrics 
            (timestamp, model_name, accuracy, precision_val, recall_val, f1_score,
             total_predictions, correct_predictions, training_data_size, 
             feature_importance, confusion_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.model_name, metrics.accuracy,
            metrics.precision, metrics.recall, metrics.f1_score,
            metrics.total_predictions, metrics.correct_predictions,
            metrics.training_data_size, 
            json.dumps(metrics.feature_importance),
            json.dumps(metrics.confusion_matrix)
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Métricas ML registradas: {metrics.model_name} - Accuracy: {metrics.accuracy:.3f}")
        
        # Verificar alertas
        self._check_ml_alerts(metrics)
        
    def log_trading_metrics(self, metrics: TradingMetrics):
        """Registra métricas de trading"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_metrics 
            (timestamp, total_trades, winning_trades, losing_trades, win_rate,
             total_pnl, avg_win, avg_loss, max_drawdown, sharpe_ratio,
             ml_influenced_trades, ml_accuracy_in_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.total_trades, metrics.winning_trades,
            metrics.losing_trades, metrics.win_rate, metrics.total_pnl,
            metrics.avg_win, metrics.avg_loss, metrics.max_drawdown,
            metrics.sharpe_ratio, metrics.ml_influenced_trades,
            metrics.ml_accuracy_in_trades
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Métricas de trading registradas: Win Rate: {metrics.win_rate:.1f}%, PnL: {metrics.total_pnl:.2f}")
        
        # Verificar alertas de trading
        self._check_trading_alerts(metrics)
        
    def _check_ml_alerts(self, metrics: MLMetrics):
        """Verifica condições de alerta para ML"""
        alerts = []
        
        # Accuracy muito baixa
        if metrics.accuracy < 0.45:
            alerts.append({
                'type': 'low_accuracy',
                'severity': 'high',
                'message': f'Accuracy muito baixa: {metrics.accuracy:.3f} para {metrics.model_name}',
                'data': asdict(metrics)
            })
            
        # Poucos dados de treino
        if metrics.training_data_size < 100:
            alerts.append({
                'type': 'insufficient_data',
                'severity': 'medium',
                'message': f'Dados de treino insuficientes: {metrics.training_data_size} amostras',
                'data': asdict(metrics)
            })
            
        # F1 score muito baixo
        if metrics.f1_score < 0.4:
            alerts.append({
                'type': 'low_f1_score',
                'severity': 'medium',
                'message': f'F1 Score baixo: {metrics.f1_score:.3f}',
                'data': asdict(metrics)
            })
            
        # Registrar alertas
        for alert in alerts:
            self._create_alert(alert)
            
    def _check_trading_alerts(self, metrics: TradingMetrics):
        """Verifica condições de alerta para trading"""
        alerts = []
        
        # Win rate muito baixo
        if metrics.win_rate < 30:
            alerts.append({
                'type': 'low_win_rate',
                'severity': 'high',
                'message': f'Win rate muito baixo: {metrics.win_rate:.1f}%',
                'data': asdict(metrics)
            })
            
        # Drawdown alto
        if metrics.max_drawdown < -20:  # -20%
            alerts.append({
                'type': 'high_drawdown',
                'severity': 'high',
                'message': f'Drawdown alto: {metrics.max_drawdown:.1f}%',
                'data': asdict(metrics)
            })
            
        # PnL negativo significativo
        if metrics.total_pnl < -100:
            alerts.append({
                'type': 'negative_pnl',
                'severity': 'medium',
                'message': f'PnL negativo: {metrics.total_pnl:.2f}',
                'data': asdict(metrics)
            })
            
        # ML accuracy em trades baixa
        if metrics.ml_accuracy_in_trades < 40 and metrics.ml_influenced_trades > 10:
            alerts.append({
                'type': 'ml_underperforming',
                'severity': 'medium',
                'message': f'ML underperforming em trades: {metrics.ml_accuracy_in_trades:.1f}%',
                'data': asdict(metrics)
            })
            
        # Registrar alertas
        for alert in alerts:
            self._create_alert(alert)
            
    def _create_alert(self, alert: Dict):
        """Cria um alerta no sistema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, message, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            alert['type'],
            alert['severity'],
            alert['message'],
            json.dumps(alert['data'])
        ))
        
        conn.commit()
        conn.close()
        
        # Log do alerta
        self.logger.warning(f"ALERTA {alert['severity'].upper()}: {alert['message']}")
        
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Obtém alertas recentes"""
        conn = sqlite3.connect(self.db_path)
        
        since = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT * FROM alerts 
            WHERE created_at > ? 
            ORDER BY created_at DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[since.isoformat()])
        conn.close()
        
        return df.to_dict('records') if len(df) > 0 else []
        
    def get_ml_performance_summary(self, days: int = 7) -> Dict:
        """Resumo de performance do ML"""
        conn = sqlite3.connect(self.db_path)
        
        since = datetime.now() - timedelta(days=days)
        
        # Métricas ML recentes
        ml_query = '''
            SELECT * FROM ml_metrics 
            WHERE created_at > ?
            ORDER BY created_at DESC
        '''
        
        ml_df = pd.read_sql_query(ml_query, conn, params=[since.isoformat()])
        
        if len(ml_df) == 0:
            conn.close()
            return {'error': 'Nenhum dado ML encontrado'}
            
        # Calcular resumo
        summary = {
            'period_days': days,
            'total_model_updates': len(ml_df),
            'latest_accuracy': float(ml_df.iloc[0]['accuracy']) if len(ml_df) > 0 else 0,
            'avg_accuracy': float(ml_df['accuracy'].mean()),
            'best_accuracy': float(ml_df['accuracy'].max()),
            'worst_accuracy': float(ml_df['accuracy'].min()),
            'accuracy_trend': 'improving' if len(ml_df) > 1 and ml_df.iloc[0]['accuracy'] > ml_df.iloc[-1]['accuracy'] else 'declining',
            'models_performance': {}
        }
        
        # Performance por modelo
        for model in ml_df['model_name'].unique():
            model_data = ml_df[ml_df['model_name'] == model]
            summary['models_performance'][model] = {
                'updates': len(model_data),
                'latest_accuracy': float(model_data.iloc[0]['accuracy']),
                'avg_accuracy': float(model_data['accuracy'].mean())
            }
            
        conn.close()
        return summary
        
    def get_trading_performance_summary(self, days: int = 7) -> Dict:
        """Resumo de performance de trading"""
        conn = sqlite3.connect(self.db_path)
        
        since = datetime.now() - timedelta(days=days)
        
        # Métricas de trading recentes
        trading_query = '''
            SELECT * FROM trading_metrics 
            WHERE created_at > ?
            ORDER BY created_at DESC
        '''
        
        trading_df = pd.read_sql_query(trading_query, conn, params=[since.isoformat()])
        
        if len(trading_df) == 0:
            conn.close()
            return {'error': 'Nenhum dado de trading encontrado'}
            
        # Último registro
        latest = trading_df.iloc[0]
        
        summary = {
            'period_days': days,
            'total_trades': int(latest['total_trades']),
            'win_rate': float(latest['win_rate']),
            'total_pnl': float(latest['total_pnl']),
            'avg_win': float(latest['avg_win']),
            'avg_loss': float(latest['avg_loss']),
            'max_drawdown': float(latest['max_drawdown']),
            'sharpe_ratio': float(latest['sharpe_ratio']),
            'ml_influenced_trades': int(latest['ml_influenced_trades']),
            'ml_accuracy_in_trades': float(latest['ml_accuracy_in_trades']),
            'ml_influence_percentage': (int(latest['ml_influenced_trades']) / max(int(latest['total_trades']), 1)) * 100
        }
        
        conn.close()
        return summary
        
    def update_symbol_performance(self, symbol: str, trades_data: List[Dict]):
        """Atualiza performance por símbolo"""
        if not trades_data:
            return
            
        # Calcular métricas do símbolo
        total_trades = len(trades_data)
        wins = len([t for t in trades_data if t.get('outcome') == 'won'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        pnls = [t.get('pnl', 0) for t in trades_data if t.get('pnl') is not None]
        avg_pnl = np.mean(pnls) if pnls else 0
        
        volatilities = [t.get('volatility', 50) for t in trades_data if t.get('volatility') is not None]
        avg_volatility = np.mean(volatilities) if volatilities else 50
        
        # ML accuracy para este símbolo
        ml_predictions = [t for t in trades_data if t.get('ml_prediction')]
        ml_accuracy = 0
        if ml_predictions:
            correct_ml = len([t for t in ml_predictions 
                            if (t.get('ml_prediction', {}).get('prediction') == 'favor' and t.get('outcome') == 'won') or
                               (t.get('ml_prediction', {}).get('prediction') == 'avoid' and t.get('outcome') == 'lost')])
            ml_accuracy = (correct_ml / len(ml_predictions)) * 100
            
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO symbol_performance
            (symbol, timestamp, total_trades, win_rate, avg_pnl, ml_accuracy, volatility_avg)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, datetime.now().isoformat(), total_trades,
            win_rate, avg_pnl, ml_accuracy, avg_volatility
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Performance atualizada para {symbol}: {total_trades} trades, {win_rate:.1f}% win rate")
        
    def generate_daily_report(self) -> Dict:
        """Gera relatório diário completo"""
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'ml_performance': self.get_ml_performance_summary(1),  # 1 dia
            'trading_performance': self.get_trading_performance_summary(1),  # 1 dia
            'recent_alerts': self.get_recent_alerts(24),  # 24 horas
            'recommendations': []
        }
        
        # Gerar recomendações baseadas nos dados
        recommendations = []
        
        # Recomendações ML
        ml_perf = report['ml_performance']
        if not isinstance(ml_perf, dict) or 'error' in ml_perf:
            recommendations.append("🔄 Treinar modelos ML - dados insuficientes")
        elif ml_perf.get('latest_accuracy', 0) < 0.5:
            recommendations.append("📈 Melhorar accuracy do ML - considerar mais features")
        elif ml_perf.get('accuracy_trend') == 'declining':
            recommendations.append("⚠️ Accuracy do ML em declínio - investigar causas")
            
        # Recomendações Trading
        trading_perf = report['trading_performance']
        if not isinstance(trading_perf, dict) or 'error' in trading_perf:
            recommendations.append("📊 Executar mais trades para análise")
        elif trading_perf.get('win_rate', 0) < 40:
            recommendations.append("🎯 Win rate baixo - revisar estratégia")
        elif trading_perf.get('ml_influence_percentage', 0) < 30:
            recommendations.append("🧠 Aumentar uso do ML nas decisões de trade")
            
        # Recomendações baseadas em alertas
        if len(report['recent_alerts']) > 5:
            recommendations.append("🚨 Muitos alertas recentes - revisar sistema")
            
        report['recommendations'] = recommendations
        
        # Salvar relatório
        self._save_daily_report(report)
        
        return report
        
    def _save_daily_report(self, report: Dict):
        """Salva relatório diário"""
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Relatório diário salvo: {filepath}")
        
    def health_check(self) -> Dict:
        """Verificação de saúde do sistema"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {},
            'issues': []
        }
        
        try:
            # Verificar banco de dados
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar tabelas principais
            cursor.execute("SELECT COUNT(*) FROM ml_metrics")
            ml_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trading_metrics")
            trading_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE")
            unresolved_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            health['components']['database'] = {
                'status': 'healthy',
                'ml_records': ml_records,
                'trading_records': trading_records,
                'unresolved_alerts': unresolved_alerts
            }
            
            # Verificar alertas críticos
            if unresolved_alerts > 10:
                health['issues'].append("Muitos alertas não resolvidos")
                health['status'] = 'warning'
                
            # Verificar se há dados recentes
            if ml_records == 0:
                health['issues'].append("Nenhum dado ML registrado")
                health['status'] = 'warning'
                
            if trading_records == 0:
                health['issues'].append("Nenhum dado de trading registrado")
                health['status'] = 'warning'
                
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Erro no health check: {str(e)}")
            self.logger.error(f"Erro no health check: {e}")
            
        return health

# Instância global do monitor
monitor = MLMonitor()

# Funções utilitárias para integração
def log_ml_training_result(model_name: str, accuracy: float, metrics_dict: Dict):
    """Função helper para log de treino ML"""
    ml_metrics = MLMetrics(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        accuracy=accuracy,
        precision=metrics_dict.get('precision', 0),
        recall=metrics_dict.get('recall', 0),
        f1_score=metrics_dict.get('f1_score', 0),
        total_predictions=metrics_dict.get('total_predictions', 0),
        correct_predictions=metrics_dict.get('correct_predictions', 0),
        training_data_size=metrics_dict.get('training_size', 0),
        feature_importance=metrics_dict.get('feature_importance', {}),
        confusion_matrix=metrics_dict.get('confusion_matrix', [[0, 0], [0, 0]])
    )
    
    monitor.log_ml_metrics(ml_metrics)

def log_trading_session(trades: List[Dict]):
    """Função helper para log de sessão de trading"""
    if not trades:
        return
        
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.get('outcome') == 'won'])
    losing_trades = len([t for t in trades if t.get('outcome') == 'lost'])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    pnls = [t.get('pnl', 0) for t in trades if t.get('pnl') is not None]
    total_pnl = sum(pnls)
    
    wins_pnl = [t.get('pnl', 0) for t in trades if t.get('outcome') == 'won' and t.get('pnl', 0) > 0]
    losses_pnl = [t.get('pnl', 0) for t in trades if t.get('outcome') == 'lost' and t.get('pnl', 0) < 0]
    
    avg_win = np.mean(wins_pnl) if wins_pnl else 0
    avg_loss = np.mean(losses_pnl) if losses_pnl else 0
    
    # Calcular drawdown
    cumulative_pnl = np.cumsum(pnls) if pnls else [0]
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # Calcular Sharpe ratio (simplificado)
    sharpe_ratio = np.mean(pnls) / np.std(pnls) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    
    # ML metrics
    ml_influenced = len([t for t in trades if t.get('ml_prediction')])
    ml_correct = len([t for t in trades 
                     if t.get('ml_prediction') and 
                        ((t.get('ml_prediction', {}).get('prediction') == 'favor' and t.get('outcome') == 'won') or
                         (t.get('ml_prediction', {}).get('prediction') == 'avoid' and t.get('outcome') == 'lost'))])
    
    ml_accuracy_in_trades = (ml_correct / ml_influenced * 100) if ml_influenced > 0 else 0
    
    trading_metrics = TradingMetrics(
        timestamp=datetime.now().isoformat(),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        ml_influenced_trades=ml_influenced,
        ml_accuracy_in_trades=ml_accuracy_in_trades
    )
    
    monitor.log_trading_metrics(trading_metrics)