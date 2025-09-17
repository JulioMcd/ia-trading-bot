#!/usr/bin/env python3
"""
Endpoints FastAPI para Sistema Avançado de Estatísticas
Integração com o trading bot existente
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging

# Imports do sistema de estatísticas (assumindo que está no mesmo projeto)
from advanced_stats_system import (
    IntelligentTradingSystem,
    TradeResult,
    StatsDatabase,
    ErrorPatternAnalyzer,
    SuccessPatternAnalyzer,
    PerformanceTracker
)

logger = logging.getLogger(__name__)

# Modelos Pydantic para validação de dados
class TradeData(BaseModel):
    id: str
    timestamp: str
    symbol: str
    direction: str  # 'call' ou 'put'
    entry_price: float
    exit_price: float
    stake: float
    duration_planned: int = 0
    duration_actual: int = 0
    pnl: float
    pnl_percentage: float = 0.0
    status: str  # 'won', 'lost', 'cancelled'
    
    # Dados contextuais opcionais
    market_conditions: Dict[str, Any] = {}
    ai_confidence: float = 0.5
    ai_reasoning: str = ""
    entry_features: List[float] = []
    martingale_level: int = 0
    exit_reason: str = "unknown"
    error_type: Optional[str] = None
    lessons_learned: Optional[str] = None

class PerformanceRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)
    symbol: Optional[str] = None
    
class AnalysisRequest(BaseModel):
    analysis_type: str = Field(..., regex="^(errors|success|both)$")
    recent_days: int = Field(default=30, ge=1, le=90)
    min_profit: float = Field(default=5.0, ge=0)

# Instância global do sistema inteligente
trading_system = IntelligentTradingSystem()

# Criar app FastAPI
def create_enhanced_app():
    """Cria aplicação FastAPI com endpoints de estatísticas"""
    
    app = FastAPI(
        title="AI Trading Bot with Advanced Statistics",
        description="Sistema de Trading com IA e Análise Avançada de Performance",
        version="5.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ========== ENDPOINTS PRINCIPAIS ==========
    
    @app.get("/")
    async def root():
        """Status geral do sistema"""
        try:
            system_health = trading_system._assess_system_health()
            current_metrics = trading_system.performance_tracker.current_metrics
            
            return {
                "system": "AI Trading Bot with Advanced Statistics",
                "version": "5.0.0",
                "status": "operational",
                "system_health": system_health,
                "quick_stats": {
                    "total_trades": current_metrics.total_trades,
                    "win_rate": f"{current_metrics.win_rate:.1%}",
                    "total_pnl": f"${current_metrics.total_pnl:.2f}",
                    "current_drawdown": f"{current_metrics.current_drawdown:.1%}"
                },
                "endpoints": {
                    "record_trade": "/api/trade/record",
                    "performance": "/api/stats/performance",
                    "analysis": "/api/stats/analysis",
                    "recommendations": "/api/ai/recommendations",
                    "dashboard": "/dashboard"
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/api/trade/record")
    async def record_trade(trade_data: TradeData, background_tasks: BackgroundTasks):
        """
        Registra um trade completo no sistema de estatísticas
        Este é o endpoint principal que deve ser chamado após cada trade
        """
        try:
            # Converter para dict e registrar
            trade_dict = trade_data.dict()
            
            # Garantir que timestamp seja datetime
            if isinstance(trade_dict['timestamp'], str):
                trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'].replace('Z', '+00:00'))
            
            # Registrar trade no sistema inteligente
            recorded_trade = trading_system.record_trade(trade_dict)
            
            if recorded_trade:
                # Executar análises em background se necessário
                if trading_system.performance_tracker.current_metrics.total_trades % 25 == 0:
                    background_tasks.add_task(run_background_analysis)
                
                return {
                    "status": "success",
                    "trade_id": recorded_trade.id,
                    "message": "Trade registrado com sucesso",
                    "current_metrics": {
                        "total_trades": trading_system.performance_tracker.current_metrics.total_trades,
                        "win_rate": trading_system.performance_tracker.current_metrics.win_rate,
                        "total_pnl": trading_system.performance_tracker.current_metrics.total_pnl
                    }
                }
            else:
                raise HTTPException(status_code=500, detail="Falha ao registrar trade")
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    
    @app.get("/api/stats/performance")
    async def get_performance_stats(days: int = 30, symbol: Optional[str] = None):
        """
        Retorna estatísticas completas de performance
        """
        try:
            # Obter relatório de performance
            performance_report = trading_system.performance_tracker.get_performance_report(days=days)
            
            # Filtrar por símbolo se especificado
            if symbol:
                symbol_trades = trading_system.stats_db.get_trades(
                    limit=1000,
                    symbol=symbol,
                    start_date=datetime.now() - timedelta(days=days)
                )
                
                symbol_pnl = sum([t['pnl'] for t in symbol_trades])
                symbol_wins = len([t for t in symbol_trades if t['status'] == 'won'])
                symbol_win_rate = symbol_wins / len(symbol_trades) if symbol_trades else 0
                
                performance_report['symbol_specific'] = {
                    'symbol': symbol,
                    'trades': len(symbol_trades),
                    'pnl': symbol_pnl,
                    'win_rate': symbol_win_rate
                }
            
            return {
                "status": "success",
                "data": performance_report,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/stats/analysis")
    async def run_analysis(request: AnalysisRequest):
        """
        Executa análise de padrões de erro e sucesso
        """
        try:
            results = {}
            
            if request.analysis_type in ['errors', 'both']:
                # Análise de padrões de erro
                error_analysis = trading_system.error_analyzer.analyze_loss_patterns(
                    recent_days=request.recent_days
                )
                results['error_analysis'] = error_analysis
            
            if request.analysis_type in ['success', 'both']:
                # Análise de padrões de sucesso
                success_analysis = trading_system.success_analyzer.analyze_winning_patterns(
                    min_profit=request.min_profit,
                    recent_days=request.recent_days
                )
                results['success_analysis'] = success_analysis
            
            return {
                "status": "success",
                "analysis_type": request.analysis_type,
                "period_analyzed": f"{request.recent_days} days",
                "results": results,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/ai/recommendations")
    async def get_ai_recommendations():
        """
        Obtém recomendações inteligentes da IA baseadas no histórico
        """
        try:
            recommendations = trading_system.get_ai_recommendations()
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/stats/insights")
    async def get_ai_insights(days: int = 7, limit: int = 20):
        """
        Retorna insights recentes da IA
        """
        try:
            # Buscar insights do banco
            import sqlite3
            conn = sqlite3.connect(trading_system.stats_db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, insight_type, description, confidence_level, 
                       action_recommended, implementation_status
                FROM ai_insights 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY confidence_level DESC, timestamp DESC
                LIMIT ?
            '''.format(days), (limit,))
            
            insights = []
            for row in cursor.fetchall():
                insights.append({
                    'timestamp': row[0],
                    'type': row[1],
                    'description': row[2],
                    'confidence': row[3],
                    'recommended_action': row[4],
                    'status': row[5]
                })
            
            conn.close()
            
            return {
                "status": "success",
                "insights": insights,
                "period": f"{days} days",
                "total_found": len(insights)
            }
            
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/stats/summary")
    async def get_stats_summary():
        """
        Retorna resumo rápido das estatísticas mais importantes
        """
        try:
            metrics = trading_system.performance_tracker.current_metrics
            
            # Calcular algumas métricas adicionais
            today = datetime.now().date().isoformat()
            today_pnl = metrics.daily_pnl.get(today, 0)
            
            # Últimos 7 dias
            recent_days = []
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).date().isoformat()
                pnl = metrics.daily_pnl.get(date, 0)
                recent_days.append({"date": date, "pnl": pnl})
            
            return {
                "status": "success",
                "summary": {
                    "overview": {
                        "total_trades": metrics.total_trades,
                        "win_rate": f"{metrics.win_rate:.1%}",
                        "total_pnl": metrics.total_pnl,
                        "profit_factor": metrics.profit_factor,
                        "max_drawdown": f"{metrics.max_drawdown:.1%}",
                        "current_drawdown": f"{metrics.current_drawdown:.1%}"
                    },
                    "today": {
                        "pnl": today_pnl,
                        "date": today
                    },
                    "recent_performance": recent_days,
                    "streaks": {
                        "current_wins": metrics.consecutive_wins,
                        "current_losses": metrics.consecutive_losses,
                        "max_win_streak": metrics.max_consecutive_wins,
                        "max_loss_streak": metrics.max_consecutive_losses
                    },
                    "extremes": {
                        "largest_win": metrics.largest_win,
                        "largest_loss": metrics.largest_loss,
                        "average_win": metrics.average_win,
                        "average_loss": metrics.average_loss
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/stats/reset")
    async def reset_statistics(confirm: bool = False):
        """
        CUIDADO: Reseta todas as estatísticas (apenas para desenvolvimento)
        """
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Para resetar, use confirm=true. ISSO IRÁ APAGAR TODOS OS DADOS!"
            )
        
        try:
            # Backup antes de resetar
            import shutil
            backup_path = f"data/backup_before_reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy(trading_system.stats_db.db_path, backup_path)
            
            # Recriar database
            trading_system.stats_db.init_database()
            
            # Resetar métricas
            from advanced_stats_system import PerformanceMetrics
            trading_system.performance_tracker.current_metrics = PerformanceMetrics()
            trading_system.performance_tracker.performance_history.clear()
            
            return {
                "status": "success",
                "message": "Estatísticas resetadas com sucesso",
                "backup_created": backup_path,
                "warning": "Todos os dados foram removidos!"
            }
            
        except Exception as e:
            logger.error(f"Error resetting statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========== FUNÇÕES AUXILIARES ==========
    
    async def run_background_analysis():
        """Executa análises em background"""
        try:
            # Executar análise completa
            trading_system._run_periodic_analysis()
            logger.info("Background analysis completed")
        except Exception as e:
            logger.error(f"Error in background analysis: {e}")
    
    # ========== DASHBOARD MELHORADO ==========
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def enhanced_dashboard():
        """Dashboard melhorado com estatísticas avançadas"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Bot - Dashboard Avançado</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh; padding: 20px;
                }
                
                .dashboard { 
                    max-width: 1600px; margin: 0 auto; 
                    display: grid; gap: 20px;
                    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
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
                    background: rgba(255,255,255,0.1);
                    color: white;
                }
                
                .header h1 { 
                    font-size: 2.5em; margin: 0 0 10px 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
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
                .warning { background: linear-gradient(135deg, #ff9800, #f57c00); }
                
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
                
                .chart-container {
                    height: 200px; background: #f8f9fa; 
                    border-radius: 8px; margin: 15px 0;
                    display: flex; align-items: center; justify-content: center;
                    color: #666;
                }
                
                .insight-item {
                    background: #e3f2fd; padding: 12px; margin: 8px 0;
                    border-radius: 8px; border-left: 4px solid #2196F3;
                }
                
                .insight-high { border-left-color: #4CAF50; background: #e8f5e8; }
                .insight-medium { border-left-color: #ff9800; background: #fff3e0; }
                .insight-low { border-left-color: #f44336; background: #ffebee; }
                
                .analysis-result {
                    background: #f5f5f5; padding: 15px; margin: 10px 0;
                    border-radius: 8px; max-height: 300px; overflow-y: auto;
                }
                
                .status-indicator {
                    width: 12px; height: 12px; border-radius: 50%;
                    display: inline-block; margin-right: 8px;
                }
                .status-online { background: #4CAF50; }
                .status-warning { background: #ff9800; }
                .status-error { background: #f44336; }
                
                #log { 
                    height: 200px; overflow-y: auto; 
                    background: #1a1a1a; color: #00ff00; 
                    padding: 15px; border-radius: 8px; 
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                }
                
                .grid-full { grid-column: 1 / -1; }
                .grid-half { grid-column: span 2; }
                
                @media (max-width: 768px) {
                    .dashboard { grid-template-columns: 1fr; }
                    .grid-half { grid-column: span 1; }
                }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <!-- Header -->
                <div class="header">
                    <h1>🤖 AI Trading Bot - Dashboard Avançado</h1>
                    <p>Sistema Inteligente com Análise de Padrões e Aprendizado Contínuo</p>
                    <div>
                        <span class="status-indicator status-online"></span>
                        Sistema Operacional com Estatísticas Avançadas
                    </div>
                </div>

                <!-- Performance Overview -->
                <div class="card">
                    <h3>📊 Performance Geral</h3>
                    <div id="performance-overview">
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
                            <span class="metric-label">Profit Factor</span>
                            <span class="metric-value neutral" id="profit-factor">0.00</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Drawdown Atual</span>
                            <span class="metric-value neutral" id="current-drawdown">0%</span>
                        </div>
                    </div>
                </div>

                <!-- Performance Hoje -->
                <div class="card">
                    <h3>📈 Performance Hoje</h3>
                    <div id="today-performance">
                        <div class="metric">
                            <span class="metric-label">PnL Hoje</span>
                            <span class="metric-value neutral" id="today-pnl">$0.00</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Sequência Atual</span>
                            <span class="metric-value neutral" id="current-streak">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Maior Vitória</span>
                            <span class="metric-value positive" id="largest-win">$0.00</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Maior Perda</span>
                            <span class="metric-value negative" id="largest-loss">$0.00</span>
                        </div>
                    </div>
                </div>

                <!-- Sistema de Análise -->
                <div class="card">
                    <h3>🔍 Análise Inteligente</h3>
                    <div>
                        <label>Tipo de Análise:</label>
                        <select id="analysis-type" style="width: 100%; padding: 8px; margin: 10px 0;">
                            <option value="both">Erros e Sucessos</option>
                            <option value="errors">Apenas Erros</option>
                            <option value="success">Apenas Sucessos</option>
                        </select>
                        
                        <label>Período (dias):</label>
                        <input type="number" id="analysis-days" value="30" min="1" max="90" 
                               style="width: 100%; padding: 8px; margin: 10px 0;">
                        
                        <button class="btn btn-primary" onclick="runAnalysis()">
                            🔄 Executar Análise
                        </button>
                    </div>
                    
                    <div class="analysis-result" id="analysis-results" style="display: none;">
                        <div id="analysis-content"></div>
                    </div>
                </div>

                <!-- Recomendações da IA -->
                <div class="card">
                    <h3>🧠 Recomendações da IA</h3>
                    <button class="btn btn-primary" onclick="getRecommendations()">
                        🤖 Obter Recomendações
                    </button>
                    
                    <div id="recommendations-container">
                        <div id="immediate-actions"></div>
                        <div id="strategic-adjustments"></div>
                        <div id="risk-warnings"></div>
                        <div id="optimization-opportunities"></div>
                    </div>
                </div>

                <!-- Insights Recentes -->
                <div class="card">
                    <h3>💡 Insights da IA</h3>
                    <button class="btn btn-primary" onclick="getInsights()">
                        📋 Carregar Insights
                    </button>
                    <div id="insights-container"></div>
                </div>

                <!-- Gráfico de Performance (Placeholder) -->
                <div class="card grid-half">
                    <h3>📈 Tendência de Performance</h3>
                    <div class="chart-container">
                        Gráfico de performance será implementado aqui<br>
                        <small>Mostrando PnL dos últimos 7 dias</small>
                    </div>
                    <div id="recent-performance"></div>
                </div>

                <!-- Teste de Trade -->
                <div class="card">
                    <h3>🧪 Teste - Registrar Trade</h3>
                    <div style="display: grid; gap: 10px;">
                        <input type="text" id="test-symbol" placeholder="Símbolo (ex: R_50)" value="R_50">
                        <select id="test-direction">
                            <option value="call">CALL</option>
                            <option value="put">PUT</option>
                        </select>
                        <input type="number" id="test-stake" placeholder="Stake" value="5" step="0.01">
                        <input type="number" id="test-pnl" placeholder="PnL" value="0" step="0.01">
                        <select id="test-status">
                            <option value="won">Won</option>
                            <option value="lost">Lost</option>
                        </select>
                        <button class="btn btn-primary" onclick="recordTestTrade()">
                            📝 Registrar Trade Teste
                        </button>
                    </div>
                </div>

                <!-- Log do Sistema -->
                <div class="card grid-full">
                    <h3>📋 Log do Sistema</h3>
                    <div id="log"></div>
                    <button class="btn btn-primary" onclick="clearLog()" style="margin-top: 10px;">
                        🗑️ Limpar Log
                    </button>
                </div>
            </div>

            <script>
                // Estado global
                let updateInterval;

                // Função para log
                function addLog(message, type = 'info') {
                    const log = document.getElementById('log');
                    const timestamp = new Date().toLocaleTimeString();
                    const color = type === 'error' ? '#ff4444' : type === 'success' ? '#44ff44' : '#00ff00';
                    log.innerHTML += `<span style="color: ${color}">[${timestamp}] ${message}</span>\\n`;
                    log.scrollTop = log.scrollHeight;
                }

                function clearLog() {
                    document.getElementById('log').innerHTML = '';
                    addLog('Log limpo');
                }

                // Atualizar estatísticas
                async function updateStats() {
                    try {
                        const response = await fetch('/api/stats/summary');
                        const data = await response.json();

                        if (data.status === 'success') {
                            const summary = data.summary;
                            
                            // Performance geral
                            document.getElementById('total-trades').textContent = summary.overview.total_trades;
                            
                            const winRateEl = document.getElementById('win-rate');
                            winRateEl.textContent = summary.overview.win_rate;
                            winRateEl.className = 'metric-value ' + (parseFloat(summary.overview.win_rate) > 50 ? 'positive' : 'negative');
                            
                            const pnlEl = document.getElementById('total-pnl');
                            pnlEl.textContent = '$' + summary.overview.total_pnl.toFixed(2);
                            pnlEl.className = 'metric-value ' + (summary.overview.total_pnl >= 0 ? 'positive' : 'negative');
                            
                            const pfEl = document.getElementById('profit-factor');
                            pfEl.textContent = summary.overview.profit_factor.toFixed(2);
                            pfEl.className = 'metric-value ' + (summary.overview.profit_factor >= 1 ? 'positive' : 'negative');
                            
                            const ddEl = document.getElementById('current-drawdown');
                            ddEl.textContent = summary.overview.current_drawdown;
                            ddEl.className = 'metric-value ' + (parseFloat(summary.overview.current_drawdown) < 5 ? 'positive' : 'warning');
                            
                            // Performance hoje
                            const todayPnlEl = document.getElementById('today-pnl');
                            todayPnlEl.textContent = '$' + summary.today.pnl.toFixed(2);
                            todayPnlEl.className = 'metric-value ' + (summary.today.pnl >= 0 ? 'positive' : 'negative');
                            
                            // Sequências
                            const streakEl = document.getElementById('current-streak');
                            const wins = summary.streaks.current_wins;
                            const losses = summary.streaks.current_losses;
                            
                            if (wins > 0) {
                                streakEl.textContent = `${wins} vitórias`;
                                streakEl.className = 'metric-value positive';
                            } else if (losses > 0) {
                                streakEl.textContent = `${losses} perdas`;
                                streakEl.className = 'metric-value negative';
                            } else {
                                streakEl.textContent = 'Neutro';
                                streakEl.className = 'metric-value neutral';
                            }
                            
                            // Extremos
                            document.getElementById('largest-win').textContent = '$' + summary.extremes.largest_win.toFixed(2);
                            document.getElementById('largest-loss').textContent = '$' + summary.extremes.largest_loss.toFixed(2);
                            
                            // Performance recente
                            const recentPerf = document.getElementById('recent-performance');
                            recentPerf.innerHTML = '<h4>Últimos 7 dias:</h4>';
                            summary.recent_performance.forEach(day => {
                                const dayDiv = document.createElement('div');
                                dayDiv.style.cssText = 'display: flex; justify-content: space-between; padding: 4px 0;';
                                dayDiv.innerHTML = `
                                    <span>${day.date}</span>
                                    <span style="color: ${day.pnl >= 0 ? '#4CAF50' : '#f44336'}">
                                        $${day.pnl.toFixed(2)}
                                    </span>
                                `;
                                recentPerf.appendChild(dayDiv);
                            });
                        }

                    } catch (error) {
                        addLog('❌ Erro ao atualizar estatísticas: ' + error.message, 'error');
                    }
                }

                // Executar análise
                async function runAnalysis() {
                    const type = document.getElementById('analysis-type').value;
                    const days = document.getElementById('analysis-days').value;
                    
                    addLog(`🔍 Iniciando análise: ${type} (${days} dias)`);
                    
                    try {
                        const response = await fetch('/api/stats/analysis', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                analysis_type: type,
                                recent_days: parseInt(days),
                                min_profit: 5.0
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            displayAnalysisResults(data.results);
                            addLog('✅ Análise concluída com sucesso', 'success');
                        } else {
                            addLog('❌ Erro na análise: ' + data.message, 'error');
                        }
                        
                    } catch (error) {
                        addLog('❌ Erro na análise: ' + error.message, 'error');
                    }
                }

                function displayAnalysisResults(results) {
                    const container = document.getElementById('analysis-results');
                    const content = document.getElementById('analysis-content');
                    
                    content.innerHTML = '';
                    
                    if (results.error_analysis) {
                        const errorDiv = document.createElement('div');
                        errorDiv.innerHTML = `
                            <h4 style="color: #f44336;">🚨 Análise de Erros</h4>
                            <p><strong>Perdas analisadas:</strong> ${results.error_analysis.total_losses_analyzed}</p>
                            <p><strong>Padrões encontrados:</strong> ${results.error_analysis.patterns_found}</p>
                        `;
                        
                        if (results.error_analysis.overall_insights) {
                            errorDiv.innerHTML += '<h5>Insights:</h5><ul>';
                            results.error_analysis.overall_insights.forEach(insight => {
                                errorDiv.innerHTML += `<li>${insight}</li>`;
                            });
                            errorDiv.innerHTML += '</ul>';
                        }
                        
                        content.appendChild(errorDiv);
                    }
                    
                    if (results.success_analysis) {
                        const successDiv = document.createElement('div');
                        successDiv.innerHTML = `
                            <h4 style="color: #4CAF50;">🎯 Análise de Sucessos</h4>
                            <p><strong>Trades vencedores:</strong> ${results.success_analysis.winning_trades_analyzed}</p>
                            <p><strong>Padrões lucrativos:</strong> ${results.success_analysis.profitable_patterns}</p>
                        `;
                        
                        if (results.success_analysis.optimization_suggestions) {
                            successDiv.innerHTML += '<h5>Sugestões de Otimização:</h5><ul>';
                            results.success_analysis.optimization_suggestions.forEach(suggestion => {
                                successDiv.innerHTML += `<li>${suggestion}</li>`;
                            });
                            successDiv.innerHTML += '</ul>';
                        }
                        
                        content.appendChild(successDiv);
                    }
                    
                    container.style.display = 'block';
                }

                // Obter recomendações
                async function getRecommendations() {
                    addLog('🤖 Obtendo recomendações da IA...');
                    
                    try {
                        const response = await fetch('/api/ai/recommendations');
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            displayRecommendations(data.recommendations);
                            addLog('✅ Recomendações obtidas', 'success');
                        }
                        
                    } catch (error) {
                        addLog('❌ Erro ao obter recomendações: ' + error.message, 'error');
                    }
                }

                function displayRecommendations(recs) {
                    const containers = {
                        'immediate-actions': 'immediate_actions',
                        'strategic-adjustments': 'strategic_adjustments', 
                        'risk-warnings': 'risk_warnings',
                        'optimization-opportunities': 'optimization_opportunities'
                    };
                    
                    Object.keys(containers).forEach(containerId => {
                        const container = document.getElementById(containerId);
                        const recType = containers[containerId];
                        const recommendations = recs.recommendations[recType] || [];
                        
                        container.innerHTML = `<h4>${containerId.replace('-', ' ').toUpperCase()}</h4>`;
                        
                        if (recommendations.length === 0) {
                            container.innerHTML += '<p style="opacity: 0.6;">Nenhuma recomendação no momento</p>';
                        } else {
                            recommendations.forEach(rec => {
                                const recDiv = document.createElement('div');
                                recDiv.className = 'insight-item insight-' + 
                                    (rec.confidence > 0.8 ? 'high' : rec.confidence > 0.5 ? 'medium' : 'low');
                                recDiv.innerHTML = `
                                    <strong>${rec.type}</strong><br>
                                    ${rec.description}<br>
                                    <small>Ação: ${rec.action} (${(rec.confidence * 100).toFixed(0)}% confiança)</small>
                                `;
                                container.appendChild(recDiv);
                            });
                        }
                    });
                }

                // Obter insights
                async function getInsights() {
                    try {
                        const response = await fetch('/api/stats/insights?days=7&limit=10');
                        const data = await response.json();
                        
                        const container = document.getElementById('insights-container');
                        container.innerHTML = '<h4>Insights Recentes (7 dias)</h4>';
                        
                        if (data.insights && data.insights.length > 0) {
                            data.insights.forEach(insight => {
                                const insightDiv = document.createElement('div');
                                insightDiv.className = 'insight-item insight-' + 
                                    (insight.confidence > 0.8 ? 'high' : insight.confidence > 0.5 ? 'medium' : 'low');
                                insightDiv.innerHTML = `
                                    <strong>${insight.type}</strong> 
                                    <span style="float: right;">${new Date(insight.timestamp).toLocaleDateString()}</span><br>
                                    ${insight.description}<br>
                                    <small>Recomendação: ${insight.recommended_action}</small>
                                `;
                                container.appendChild(insightDiv);
                            });
                        } else {
                            container.innerHTML += '<p style="opacity: 0.6;">Nenhum insight encontrado</p>';
                        }
                        
                    } catch (error) {
                        addLog('❌ Erro ao carregar insights: ' + error.message, 'error');
                    }
                }

                // Registrar trade de teste
                async function recordTestTrade() {
                    const tradeData = {
                        id: 'test_' + Date.now(),
                        timestamp: new Date().toISOString(),
                        symbol: document.getElementById('test-symbol').value,
                        direction: document.getElementById('test-direction').value,
                        entry_price: 1000 + (Math.random() - 0.5) * 100,
                        exit_price: 1000 + (Math.random() - 0.5) * 100,
                        stake: parseFloat(document.getElementById('test-stake').value),
                        duration_planned: 120,
                        duration_actual: 118 + Math.floor(Math.random() * 10),
                        pnl: parseFloat(document.getElementById('test-pnl').value),
                        pnl_percentage: 0,
                        status: document.getElementById('test-status').value,
                        market_conditions: {
                            volatility: Math.random() * 50,
                            trend_strength: Math.random()
                        },
                        ai_confidence: 0.5 + Math.random() * 0.4,
                        ai_reasoning: 'Trade de teste simulado',
                        entry_features: Array.from({length: 10}, () => Math.random()),
                        martingale_level: 0,
                        exit_reason: 'test'
                    };
                    
                    try {
                        const response = await fetch('/api/trade/record', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(tradeData)
                        });
                        
                        const result = await response.json();
                        
                        if (result.status === 'success') {
                            addLog(`✅ Trade teste registrado: ${tradeData.status} $${tradeData.pnl}`, 'success');
                            updateStats(); // Atualizar estatísticas
                        } else {
                            addLog('❌ Erro ao registrar trade teste', 'error');
                        }
                        
                    } catch (error) {
                        addLog('❌ Erro: ' + error.message, 'error');
                    }
                }

                // Inicialização
                document.addEventListener('DOMContentLoaded', function() {
                    addLog('🚀 Dashboard Avançado iniciado');
                    addLog('📊 Sistema de estatísticas carregado');
                    addLog('🤖 IA de análise de padrões ativa');
                    
                    updateStats();
                    
                    // Atualizar a cada 30 segundos
                    updateInterval = setInterval(updateStats, 30000);
                    
                    addLog('✅ Dashboard pronto!', 'success');
                });

                // Cleanup
                window.addEventListener('beforeunload', function() {
                    if (updateInterval) clearInterval(updateInterval);
                });
            </script>
        </body>
        </html>
        """
    
    return app

# Instância da aplicação
app = create_enhanced_app()

if __name__ == "__main__":
    import uvicorn
    
    # Configurações para Render
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info("🚀 Iniciando AI Trading Bot com Sistema Avançado de Estatísticas")
    logger.info("📊 Características implementadas:")
    logger.info("   • Registro completo de trades com contexto")
    logger.info("   • Análise automática de padrões de erro")
    logger.info("   • Identificação de estratégias de sucesso")
    logger.info("   • Sistema de recomendações inteligentes")
    logger.info("   • Métricas avançadas de performance")
    logger.info("   • Dashboard interativo melhorado")
    logger.info("   • Aprendizado contínuo da IA")
    
    uvicorn.run(
        "fastapi_stats_endpoints:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
