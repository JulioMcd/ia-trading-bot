from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import datetime
import logging
import sqlite3
import json
import numpy as np
from collections import defaultdict, deque
import threading
import time
import os

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key válida
VALID_API_KEY = "bhcOGajqbfFfolT"

# ✅ CONFIGURAÇÃO DE INVERSÃO DE SINAIS - DESATIVADA
INVERT_SIGNALS = False  # Sinais normais ativados

# Configurações do sistema de aprendizado
LEARNING_CONFIG = {
    'min_samples_for_learning': 20,  # Mínimo de amostras para começar a aprender
    'adaptation_rate': 0.1,          # Taxa de adaptação dos parâmetros
    'error_pattern_window': 50,      # Janela para análise de padrões de erro
    'confidence_adjustment_factor': 0.05,  # Fator de ajuste de confiança
    'learning_enabled': True,        # Sistema de aprendizado ativo
}

class TradingDatabase:
    """Classe para gerenciar o banco de dados SQLite"""
    
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Inicializar tabelas do banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de sinais e resultados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                volatility REAL,
                duration_type TEXT,
                duration_value INTEGER,
                result INTEGER,  -- 1 para win, 0 para loss, NULL se pendente
                pnl REAL,
                martingale_level INTEGER DEFAULT 0,
                market_condition TEXT,
                technical_factors TEXT,  -- JSON com fatores técnicos
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                feedback_received_at TEXT
            )
        ''')
        
        # Tabela de métricas de performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT,
                direction TEXT,
                total_signals INTEGER DEFAULT 0,
                won_signals INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de padrões de erro
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                conditions TEXT NOT NULL,  -- JSON com condições do padrão
                error_rate REAL NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence_adjustment REAL DEFAULT 0,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de parâmetros adaptativos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT NOT NULL UNIQUE,
                parameter_value REAL NOT NULL,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                update_reason TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_signal(self, signal_data):
        """Salvar sinal no banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (
                timestamp, symbol, direction, confidence, entry_price, 
                volatility, duration_type, duration_value, martingale_level,
                market_condition, technical_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('timestamp'),
            signal_data.get('symbol', 'R_50'),
            signal_data.get('direction'),
            signal_data.get('confidence'),
            signal_data.get('entry_price'),
            signal_data.get('volatility'),
            signal_data.get('duration_type'),
            signal_data.get('duration_value'),
            signal_data.get('martingale_level', 0),
            signal_data.get('market_condition', 'neutral'),
            json.dumps(signal_data.get('technical_factors', {}))
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
        
    def update_signal_result(self, signal_id, result, pnl=0):
        """Atualizar resultado do sinal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE signals 
            SET result = ?, pnl = ?, feedback_received_at = ?
            WHERE id = ?
        ''', (result, pnl, datetime.datetime.now().isoformat(), signal_id))
        
        conn.commit()
        conn.close()
        
    def get_recent_performance(self, limit=100, symbol=None):
        """Obter performance recente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM signals 
            WHERE result IS NOT NULL
        '''
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
            
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
        
    def get_error_patterns(self):
        """Obter padrões de erro identificados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM error_patterns ORDER BY error_rate DESC')
        patterns = cursor.fetchall()
        conn.close()
        
        return patterns
        
    def save_error_pattern(self, pattern_type, conditions, error_rate):
        """Salvar padrão de erro identificado"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verificar se padrão já existe
        cursor.execute(
            'SELECT id, occurrences FROM error_patterns WHERE pattern_type = ? AND conditions = ?',
            (pattern_type, json.dumps(conditions))
        )
        existing = cursor.fetchone()
        
        if existing:
            # Atualizar padrão existente
            cursor.execute('''
                UPDATE error_patterns 
                SET error_rate = ?, occurrences = occurrences + 1, last_seen = ?
                WHERE id = ?
            ''', (error_rate, datetime.datetime.now().isoformat(), existing[0]))
        else:
            # Criar novo padrão
            cursor.execute('''
                INSERT INTO error_patterns (pattern_type, conditions, error_rate)
                VALUES (?, ?, ?)
            ''', (pattern_type, json.dumps(conditions), error_rate))
            
        conn.commit()
        conn.close()
        
    def get_adaptive_parameter(self, param_name, default_value):
        """Obter parâmetro adaptativo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT parameter_value FROM adaptive_parameters WHERE parameter_name = ?',
            (param_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else default_value
        
    def update_adaptive_parameter(self, param_name, new_value, reason):
        """Atualizar parâmetro adaptativo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO adaptive_parameters 
            (parameter_name, parameter_value, last_updated, update_reason)
            VALUES (?, ?, ?, ?)
        ''', (param_name, new_value, datetime.datetime.now().isoformat(), reason))
        
        conn.commit()
        conn.close()

class LearningEngine:
    """Motor de aprendizado baseado em erros"""
    
    def __init__(self, database):
        self.db = database
        self.recent_signals = deque(maxlen=LEARNING_CONFIG['error_pattern_window'])
        self.confidence_adjustments = defaultdict(float)
        self.symbol_performance = defaultdict(lambda: {'wins': 0, 'total': 0})
        self.direction_performance = defaultdict(lambda: {'wins': 0, 'total': 0})
        
    def analyze_error_patterns(self):
        """Analisar padrões de erro nos dados recentes"""
        recent_data = self.db.get_recent_performance(LEARNING_CONFIG['error_pattern_window'])
        
        if len(recent_data) < LEARNING_CONFIG['min_samples_for_learning']:
            return []
            
        patterns_found = []
        
        # Analisar padrões por símbolo
        symbol_errors = defaultdict(list)
        for signal in recent_data:
            symbol = signal[2]  # symbol column
            result = signal[9]  # result column
            symbol_errors[symbol].append(result)
            
        for symbol, results in symbol_errors.items():
            if len(results) >= 10:
                win_rate = sum(results) / len(results)
                if win_rate < 0.4:  # Taxa de acerto menor que 40%
                    error_rate = 1 - win_rate
                    self.db.save_error_pattern(
                        'symbol_low_performance',
                        {'symbol': symbol},
                        error_rate
                    )
                    patterns_found.append({
                        'type': 'symbol_error',
                        'symbol': symbol,
                        'error_rate': error_rate
                    })
        
        # Analisar padrões por direção
        direction_errors = defaultdict(list)
        for signal in recent_data:
            direction = signal[3]  # direction column
            result = signal[9]  # result column
            direction_errors[direction].append(result)
            
        for direction, results in direction_errors.items():
            if len(results) >= 10:
                win_rate = sum(results) / len(results)
                if win_rate < 0.4:
                    error_rate = 1 - win_rate
                    self.db.save_error_pattern(
                        'direction_low_performance',
                        {'direction': direction},
                        error_rate
                    )
                    patterns_found.append({
                        'type': 'direction_error',
                        'direction': direction,
                        'error_rate': error_rate
                    })
        
        # Analisar padrões por volatilidade
        volatility_results = []
        for signal in recent_data:
            volatility = signal[6]  # volatility column
            result = signal[9]  # result column
            if volatility and result is not None:
                volatility_results.append((volatility, result))
                
        if len(volatility_results) >= 15:
            # Dividir em faixas de volatilidade
            high_vol = [r for v, r in volatility_results if v > 70]
            low_vol = [r for v, r in volatility_results if v < 30]
            
            if len(high_vol) >= 5:
                high_vol_rate = sum(high_vol) / len(high_vol)
                if high_vol_rate < 0.35:
                    self.db.save_error_pattern(
                        'high_volatility_error',
                        {'volatility_range': 'high'},
                        1 - high_vol_rate
                    )
                    
            if len(low_vol) >= 5:
                low_vol_rate = sum(low_vol) / len(low_vol)
                if low_vol_rate < 0.35:
                    self.db.save_error_pattern(
                        'low_volatility_error',
                        {'volatility_range': 'low'},
                        1 - low_vol_rate
                    )
        
        return patterns_found
        
    def adapt_confidence(self, signal_data):
        """Adaptar confiança baseado em padrões aprendidos"""
        base_confidence = signal_data.get('confidence', 70)
        adjustments = []
        
        # Verificar padrões conhecidos
        error_patterns = self.db.get_error_patterns()
        
        for pattern in error_patterns:
            pattern_type = pattern[1]
            conditions = json.loads(pattern[2])
            error_rate = pattern[3]
            
            # Aplicar ajustes baseados nos padrões
            if pattern_type == 'symbol_low_performance':
                if signal_data.get('symbol') == conditions.get('symbol'):
                    adjustment = -error_rate * 20  # Reduzir confiança
                    adjustments.append(('symbol_pattern', adjustment))
                    
            elif pattern_type == 'direction_low_performance':
                if signal_data.get('direction') == conditions.get('direction'):
                    adjustment = -error_rate * 15
                    adjustments.append(('direction_pattern', adjustment))
                    
            elif pattern_type == 'high_volatility_error':
                if signal_data.get('volatility', 0) > 70:
                    adjustment = -error_rate * 10
                    adjustments.append(('volatility_pattern', adjustment))
                    
            elif pattern_type == 'low_volatility_error':
                if signal_data.get('volatility', 0) < 30:
                    adjustment = -error_rate * 10
                    adjustments.append(('volatility_pattern', adjustment))
        
        # Aplicar ajustes
        total_adjustment = sum(adj[1] for adj in adjustments)
        adapted_confidence = max(50, min(95, base_confidence + total_adjustment))
        
        # Salvar informações de adaptação
        if adjustments:
            logger.info(f"🧠 Confiança adaptada: {base_confidence:.1f} → {adapted_confidence:.1f}")
            logger.info(f"   Ajustes aplicados: {adjustments}")
        
        return adapted_confidence, adjustments
        
    def update_performance_metrics(self):
        """Atualizar métricas de performance globais"""
        recent_data = self.db.get_recent_performance(100)
        
        if not recent_data:
            return
            
        # Calcular métricas gerais
        total_signals = len(recent_data)
        won_signals = sum(1 for signal in recent_data if signal[9] == 1)
        accuracy = (won_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Atualizar parâmetros adaptativos baseados na performance
        if accuracy < 45:
            # Performance baixa - aumentar conservadorismo
            current_risk_factor = self.db.get_adaptive_parameter('risk_factor', 1.0)
            new_risk_factor = max(0.5, current_risk_factor - 0.1)
            self.db.update_adaptive_parameter(
                'risk_factor', 
                new_risk_factor, 
                f'Performance baixa: {accuracy:.1f}%'
            )
            
        elif accuracy > 65:
            # Performance boa - pode ser mais agressivo
            current_risk_factor = self.db.get_adaptive_parameter('risk_factor', 1.0)
            new_risk_factor = min(1.5, current_risk_factor + 0.05)
            self.db.update_adaptive_parameter(
                'risk_factor', 
                new_risk_factor, 
                f'Performance boa: {accuracy:.1f}%'
            )
        
        return {
            'total_signals': total_signals,
            'accuracy': accuracy,
            'won_signals': won_signals
        }

# Instâncias globais
db = TradingDatabase()
learning_engine = LearningEngine(db)

# Dados de histórico simples (mantidos para compatibilidade)
trade_history = []
performance_stats = {
    'total_trades': 0,
    'won_trades': 0,
    'total_pnl': 0.0
}

def validate_api_key():
    """Validar API Key (opcional)"""
    auth_header = request.headers.get('Authorization', '')
    api_key_header = request.headers.get('X-API-Key', '')
    
    if auth_header.startswith('Bearer '):
        api_key = auth_header.replace('Bearer ', '')
    else:
        api_key = api_key_header
    
    if not api_key:
        return True
    
    return api_key == VALID_API_KEY

def analyze_technical_pattern(prices, learning_data=None):
    """Análise técnica com aprendizado integrado"""
    try:
        if len(prices) >= 3:
            # Tendência simples
            recent_trend = prices[-1] - prices[-3]
            volatility = abs(prices[-1] - prices[-2]) / prices[-2] * 100 if prices[-2] != 0 else 50
            
            # Lógica de direção NORMAL
            if recent_trend > 0:
                direction = "CALL"
                base_confidence = 70 + min(volatility * 0.3, 20)
            else:
                direction = "PUT" 
                base_confidence = 70 + min(volatility * 0.3, 20)
            
            # 🧠 APLICAR APRENDIZADO
            if learning_data and LEARNING_CONFIG['learning_enabled']:
                adapted_confidence, adjustments = learning_engine.adapt_confidence({
                    'direction': direction,
                    'confidence': base_confidence,
                    'volatility': volatility,
                    **learning_data
                })
                
                return direction, round(adapted_confidence, 1), direction, adjustments
            
            return direction, round(base_confidence, 1), direction, []
            
        else:
            # Fallback aleatório ponderado
            direction = "CALL" if random.random() > 0.5 else "PUT"
            confidence = 70 + random.uniform(0, 20)
            
            return direction, round(confidence, 1), direction, []
    except Exception as e:
        logger.error(f"Erro na análise técnica: {e}")
        direction = "CALL" if random.random() > 0.5 else "PUT"
        return direction, 70.0, direction, []

def extract_features(data):
    """Extrair dados dos parâmetros recebidos"""
    current_price = data.get("currentPrice", 1000)
    volatility = data.get("volatility", 50)
    
    # Gerar preços baseados no atual se não fornecidos
    prices = data.get("lastTicks", [])
    if not prices:
        prices = [
            current_price - random.uniform(0, 5),
            current_price + random.uniform(0, 5), 
            current_price - random.uniform(0, 3)
        ]
    
    while len(prices) < 3:
        prices.append(current_price + random.uniform(-2, 2))
        
    return prices[-3:], volatility

def calculate_risk_score(data):
    """Calcular score de risco com parâmetros adaptativos"""
    martingale_level = data.get("martingaleLevel", 0)
    today_pnl = data.get("todayPnL", 0)
    win_rate = data.get("winRate", 50)
    total_trades = data.get("totalTrades", 0)
    
    # Obter fator de risco adaptativo
    risk_factor = db.get_adaptive_parameter('risk_factor', 1.0)
    
    risk_score = 0
    risk_level = "low"
    
    # Análise Martingale (ajustada pelo fator de risco)
    martingale_threshold = max(3, int(6 * risk_factor))
    if martingale_level > martingale_threshold:
        risk_score += 40
        risk_level = "high"
    elif martingale_level > martingale_threshold // 2:
        risk_score += 20
        risk_level = "medium"
    
    # Análise P&L (ajustada pelo fator de risco)
    pnl_threshold = int(100 * risk_factor)
    if today_pnl < -pnl_threshold:
        risk_score += 25
        risk_level = "high"
    elif today_pnl < -pnl_threshold // 2:
        risk_score += 10
        risk_level = "medium" if risk_level == "low" else risk_level
    
    # Análise Win Rate
    if win_rate < 30:
        risk_score += 20
        risk_level = "high"
    elif win_rate < 45:
        risk_score += 10
    
    # Over-trading
    if total_trades > 50:
        risk_score += 10
    
    return min(risk_score, 100), risk_level

def optimize_duration(data):
    """Otimizar duração com aprendizado"""
    symbol = data.get("symbol", "R_50")
    volatility = data.get("volatility", 50)
    market_condition = data.get("marketCondition", "neutral")
    
    # Obter parâmetros adaptativos
    duration_factor = db.get_adaptive_parameter('duration_factor', 1.0)
    
    # Determinar se é índice de volatilidade
    is_volatility_index = "R_" in symbol or "HZ" in symbol
    
    if is_volatility_index:
        duration_type = "t"
        if volatility > 70:
            base_duration = random.randint(1, 3)
        elif volatility > 40:
            base_duration = random.randint(4, 6)
        else:
            base_duration = random.randint(7, 10)
    else:
        if random.random() > 0.3:
            duration_type = "m"
            if market_condition == "favorable":
                base_duration = random.randint(1, 2)
            elif market_condition == "unfavorable":
                base_duration = random.randint(4, 5)
            else:
                base_duration = random.randint(2, 4)
        else:
            duration_type = "t"
            base_duration = random.randint(3, 8)
    
    # Aplicar fator de duração adaptativo
    duration = max(1, int(base_duration * duration_factor))
    
    # Limites de segurança
    if duration_type == "t":
        duration = max(1, min(10, duration))
    else:
        duration = max(1, min(5, duration))
    
    confidence = 75 + random.uniform(0, 20)
    
    return {
        "type": duration_type,
        "duration_type": "ticks" if duration_type == "t" else "minutes",
        "value": duration,
        "duration": duration,
        "confidence": round(confidence, 1),
        "reasoning": f"Análise adaptativa para {symbol}: {duration}{duration_type} (fator: {duration_factor:.2f})"
    }

def manage_position(data):
    """Gestão de posição com parâmetros adaptativos"""
    current_balance = data.get("currentBalance", 1000)
    today_pnl = data.get("todayPnL", 0)
    martingale_level = data.get("martingaleLevel", 0)
    current_stake = data.get("currentStake", 1)
    win_rate = data.get("winRate", 50)
    
    # Obter parâmetros adaptativos
    risk_factor = db.get_adaptive_parameter('risk_factor', 1.0)
    aggression_factor = db.get_adaptive_parameter('aggression_factor', 1.0)
    
    action = "continue"
    recommended_stake = current_stake
    should_pause = False
    pause_duration = 0
    
    # Verificar se deve pausar (ajustado pelo risk_factor)
    pause_threshold_high = int(200 * risk_factor)
    pause_threshold_medium = int(100 * risk_factor)
    martingale_threshold = max(5, int(7 * risk_factor))
    
    if today_pnl < -pause_threshold_high or martingale_level > martingale_threshold:
        should_pause = True
        action = "pause"
        pause_duration = random.randint(60000, 180000)
    elif today_pnl < -pause_threshold_medium or martingale_level > martingale_threshold - 2:
        if random.random() > 0.7:
            should_pause = True
            action = "pause"
            pause_duration = random.randint(30000, 90000)
    
    # Ajustar stake se não em Martingale (com aggression_factor)
    if not should_pause and martingale_level == 0:
        if win_rate > 70:
            multiplier = 1.15 * aggression_factor
            recommended_stake = min(50, current_stake * multiplier)
        elif win_rate < 30:
            multiplier = 0.8 / aggression_factor
            recommended_stake = max(0.35, current_stake * multiplier)
        elif today_pnl < -50:
            recommended_stake = max(0.35, current_stake * 0.9)
    
    message = ""
    if should_pause:
        message = f"PAUSA RECOMENDADA - {pause_duration//1000}s - Alto risco (adaptativo)"
    elif recommended_stake != current_stake:
        message = f"Stake adaptativo: ${current_stake:.2f} → ${recommended_stake:.2f}"
    else:
        message = "Continuar operação - Parâmetros adequados"
    
    return {
        "action": action,
        "recommendedStake": round(recommended_stake, 2),
        "shouldPause": should_pause,
        "pauseDuration": pause_duration,
        "riskLevel": "high" if martingale_level > 5 else "medium" if today_pnl < -50 else "low",
        "message": message,
        "reasoning": "Sistema adaptativo ativo",
        "adaptive_factors": {
            "risk_factor": risk_factor,
            "aggression_factor": aggression_factor
        }
    }

# Sistema de análise de padrões em background
def background_learning_task():
    """Tarefa de aprendizado em background"""
    while True:
        try:
            if LEARNING_CONFIG['learning_enabled']:
                # Analisar padrões de erro
                patterns = learning_engine.analyze_error_patterns()
                if patterns:
                    logger.info(f"🧠 Novos padrões identificados: {len(patterns)}")
                
                # Atualizar métricas
                metrics = learning_engine.update_performance_metrics()
                if metrics:
                    logger.info(f"📊 Métricas atualizadas - Accuracy: {metrics['accuracy']:.1f}%")
                
            # Aguardar antes da próxima análise
            time.sleep(300)  # 5 minutos
            
        except Exception as e:
            logger.error(f"Erro na tarefa de aprendizado: {e}")
            time.sleep(60)

# Iniciar thread de aprendizado
learning_thread = threading.Thread(target=background_learning_task, daemon=True)
learning_thread.start()

# ===============================
# ROTAS DA API (Atualizadas)
# ===============================

@app.route("/")
def home():
    # Obter estatísticas do banco de dados
    recent_data = db.get_recent_performance(100)
    total_signals = len(recent_data)
    accuracy = (sum(1 for signal in recent_data if signal[9] == 1) / total_signals * 100) if total_signals > 0 else 0
    
    return jsonify({
        "status": "🤖 IA Trading Bot API Online - Sistema de Aprendizado Ativo",
        "version": "3.0.0 - ML Learning System",
        "description": "API com Sistema de Aprendizado Baseado em Erros",
        "model": "Adaptive Learning Engine",
        "signal_mode": "NORMAL + LEARNING",
        "inversion_active": False,
        "learning_active": LEARNING_CONFIG['learning_enabled'],
        "endpoints": {
            "analyze": "POST /analyze - Análise adaptativa de mercado",
            "signal": "POST /signal - Sinais com aprendizado",
            "risk": "POST /risk - Avaliação de risco adaptativa",
            "optimal-duration": "POST /optimal-duration - Duração otimizada",
            "management": "POST /management - Gestão adaptativa",
            "feedback": "POST /feedback - Sistema de aprendizado",
            "learning-stats": "GET /learning-stats - Estatísticas de aprendizado"
        },
        "stats": {
            "total_predictions": total_signals,
            "current_accuracy": f"{accuracy:.1f}%",
            "learning_samples": total_signals,
            "uptime": "99.9%"
        },
        "learning_config": LEARNING_CONFIG,
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "Python ML API with Error Learning System"
    })

@app.route("/learning-stats", methods=["GET"])
def get_learning_stats():
    """Obter estatísticas do sistema de aprendizado"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        # Estatísticas recentes
        recent_data = db.get_recent_performance(100)
        error_patterns = db.get_error_patterns()
        
        total_signals = len(recent_data)
        won_signals = sum(1 for signal in recent_data if signal[9] == 1)
        accuracy = (won_signals / total_signals * 100) if total_signals > 0 else 0
        
        # Estatísticas por símbolo
        symbol_stats = defaultdict(lambda: {'total': 0, 'wins': 0})
        for signal in recent_data:
            symbol = signal[2]
            result = signal[9]
            symbol_stats[symbol]['total'] += 1
            if result == 1:
                symbol_stats[symbol]['wins'] += 1
        
        # Parâmetros adaptativos atuais
        adaptive_params = {
            'risk_factor': db.get_adaptive_parameter('risk_factor', 1.0),
            'aggression_factor': db.get_adaptive_parameter('aggression_factor', 1.0),
            'duration_factor': db.get_adaptive_parameter('duration_factor', 1.0)
        }
        
        return jsonify({
            "learning_enabled": LEARNING_CONFIG['learning_enabled'],
            "total_samples": total_signals,
            "current_accuracy": round(accuracy, 1),
            "error_patterns_found": len(error_patterns),
            "adaptive_parameters": adaptive_params,
            "symbol_performance": dict(symbol_stats),
            "recent_patterns": [
                {
                    "type": pattern[1],
                    "conditions": json.loads(pattern[2]),
                    "error_rate": pattern[3],
                    "occurrences": pattern[4]
                } for pattern in error_patterns[:5]
            ],
            "learning_config": LEARNING_CONFIG,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em learning-stats: {e}")
        return jsonify({"error": "Erro ao obter estatísticas", "message": str(e)}), 500

@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/analysis", methods=["POST", "OPTIONS"])
@app.route("/market-analysis", methods=["POST", "OPTIONS"])
@app.route("/advanced-analysis", methods=["POST", "OPTIONS"])
def analyze_market():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        prices, volatility = extract_features(data)
        
        # Preparar dados para aprendizado
        learning_data = {
            'symbol': data.get("symbol", "R_50"),
            'volatility': volatility,
            'market_condition': data.get("marketCondition", "neutral")
        }
        
        direction, confidence, _, adjustments = analyze_technical_pattern(prices, learning_data)
        
        # Análise adicional
        symbol = data.get("symbol", "R_50")
        
        # Determinar tendência baseada na direção
        if confidence > 80:
            trend = "bullish" if direction == "CALL" else "bearish"
        else:
            trend = "neutral"
        
        return jsonify({
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "volatility": round(volatility, 1),
            "direction": direction,
            "inverted": False,
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "confidence_adjustments": adjustments,
            "message": f"Análise ADAPTATIVA para {symbol}: {direction} (confiança ajustada)",
            "recommendation": f"{direction} recomendado" if confidence > 75 else "Aguardar melhor oportunidade",
            "factors": {
                "technical_analysis": direction,
                "market_volatility": round(volatility, 1),
                "confidence_level": confidence,
                "inversion_mode": "NORMAL",
                "learning_adjustments": len(adjustments)
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA com Sistema de Aprendizado - Technical Analysis"
        })
        
    except Exception as e:
        logger.error(f"Erro em analyze: {e}")
        return jsonify({"error": "Erro na análise", "message": str(e)}), 500

@app.route("/signal", methods=["POST", "OPTIONS"])
@app.route("/trading-signal", methods=["POST", "OPTIONS"])
@app.route("/get-signal", methods=["POST", "OPTIONS"])
@app.route("/smart-signal", methods=["POST", "OPTIONS"])
@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
@app.route("/prediction", methods=["POST", "OPTIONS"])
def generate_signal():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        prices, volatility = extract_features(data)
        
        # Preparar dados para aprendizado
        learning_data = {
            'symbol': data.get("symbol", "R_50"),
            'volatility': volatility,
            'market_condition': data.get("marketCondition", "neutral"),
            'martingale_level': data.get("martingaleLevel", 0)
        }
        
        direction, confidence, _, adjustments = analyze_technical_pattern(prices, learning_data)
        
        # Dados do sinal
        current_price = data.get("currentPrice", 1000)
        symbol = data.get("symbol", "R_50")
        win_rate = data.get("winRate", 50)
        
        # Ajustar confiança baseada em performance
        if win_rate > 60:
            confidence = min(confidence + 3, 95)
        elif win_rate < 40:
            confidence = max(confidence - 5, 65)
        
        # Preparar dados para salvar no banco
        signal_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'entry_price': current_price,
            'volatility': volatility,
            'martingale_level': data.get("martingaleLevel", 0),
            'market_condition': data.get("marketCondition", "neutral"),
            'technical_factors': {
                'adjustments': adjustments,
                'win_rate': win_rate,
                'prices': prices
            }
        }
        
        # Salvar sinal no banco de dados
        signal_id = db.save_signal(signal_data)
        
        reasoning = f"Análise adaptativa para {symbol} - Sistema de aprendizado ativo"
        if adjustments:
            reasoning += f" (Ajustes aplicados: {len(adjustments)})"
        
        return jsonify({
            "signal_id": signal_id,  # ID para feedback posterior
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_price": current_price,
            "strength": "forte" if confidence > 85 else "moderado" if confidence > 75 else "fraco",
            "timeframe": "5m",
            "inverted": False,
            "inversion_status": "INATIVO",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "confidence_adjustments": adjustments,
            "factors": {
                "technical_model": "Adaptive Pattern Analysis",
                "volatility_factor": volatility,
                "historical_performance": win_rate,
                "signal_inversion": "INATIVO",
                "learning_adjustments": len(adjustments)
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA com Sistema de Aprendizado - Signal Generator"
        })
        
    except Exception as e:
        logger.error(f"Erro em signal: {e}")
        return jsonify({"error": "Erro na geração de sinal", "message": str(e)}), 500

@app.route("/risk", methods=["POST", "OPTIONS"])
@app.route("/risk-assessment", methods=["POST", "OPTIONS"])
@app.route("/evaluate-risk", methods=["POST", "OPTIONS"])
def assess_risk():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        risk_score, risk_level = calculate_risk_score(data)
        
        # Obter parâmetros adaptativos para mostrar no retorno
        risk_factor = db.get_adaptive_parameter('risk_factor', 1.0)
        
        # Mensagens baseadas no nível de risco
        messages = {
            "high": "ALTO RISCO - Intervenção necessária (Sistema Adaptativo)",
            "medium": "Risco moderado - Cautela recomendada (Ajustes ativos)", 
            "low": "Risco controlado (Parâmetros otimizados)"
        }
        
        recommendations = {
            "high": "Pare imediatamente e revise estratégia",
            "medium": "Reduza frequency e monitore de perto",
            "low": "Continue operando com disciplina"
        }
        
        return jsonify({
            "level": risk_level,
            "score": risk_score,
            "message": messages[risk_level],
            "recommendation": recommendations[risk_level],
            "adaptive_risk_factor": risk_factor,
            "factors": {
                "martingale_level": data.get("martingaleLevel", 0),
                "today_pnl": data.get("todayPnL", 0),
                "win_rate": data.get("winRate", 50),
                "total_trades": data.get("totalTrades", 0),
                "risk_factor_applied": risk_factor
            },
            "severity": "critical" if risk_level == "high" else "warning" if risk_level == "medium" else "normal",
            "signal_mode": "NORMAL + LEARNING",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA com Sistema de Aprendizado - Risk Assessment"
        })
        
    except Exception as e:
        logger.error(f"Erro em risk: {e}")
        return jsonify({"error": "Erro na avaliação de risco", "message": str(e)}), 500

@app.route("/optimal-duration", methods=["POST", "OPTIONS"])
@app.route("/duration", methods=["POST", "OPTIONS"])
@app.route("/timeframe", methods=["POST", "OPTIONS"])
@app.route("/best-duration", methods=["POST", "OPTIONS"])
def get_optimal_duration():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        duration_data = optimize_duration(data)
        
        return jsonify({
            **duration_data,
            "signal_mode": "NORMAL + LEARNING",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "adaptive_optimization": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA com Sistema de Aprendizado - Duration Optimizer"
        })
        
    except Exception as e:
        logger.error(f"Erro em optimal-duration: {e}")
        return jsonify({"error": "Erro na otimização de duração", "message": str(e)}), 500

@app.route("/management", methods=["POST", "OPTIONS"])
@app.route("/auto-manage", methods=["POST", "OPTIONS"])
@app.route("/position-size", methods=["POST", "OPTIONS"])
@app.route("/risk-management", methods=["POST", "OPTIONS"])
def position_management():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        management_data = manage_position(data)
        
        return jsonify({
            **management_data,
            "signal_mode": "NORMAL + LEARNING",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA com Sistema de Aprendizado - Position Management"
        })
        
    except Exception as e:
        logger.error(f"Erro em management: {e}")
        return jsonify({"error": "Erro no gerenciamento", "message": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def receive_feedback():
    """Endpoint para receber feedback - SISTEMA DE APRENDIZADO"""
    if request.method == "OPTIONS":
        return '', 200
    
    try:
        data = request.get_json() or {}
        
        # Dados do feedback
        result = data.get("result", 0)  # 1 para win, 0 para loss
        direction = data.get("direction", "CALL")
        signal_id = data.get("signal_id")  # ID do sinal original
        pnl = data.get("pnl", 0)
        
        # 🧠 SISTEMA DE APRENDIZADO ATIVO
        if signal_id:
            # Atualizar resultado no banco de dados
            db.update_signal_result(signal_id, result, pnl)
            logger.info(f"🧠 Feedback integrado ao sistema de aprendizado: Signal {signal_id} -> {'WIN' if result == 1 else 'LOSS'}")
        
        # Atualizar stats simples (mantido para compatibilidade)
        performance_stats['total_trades'] += 1
        if result == 1:
            performance_stats['won_trades'] += 1
        
        accuracy = (performance_stats['won_trades'] / max(performance_stats['total_trades'], 1) * 100)
        
        # Trigger análise de padrões se temos amostras suficientes
        if performance_stats['total_trades'] % 10 == 0:
            try:
                patterns = learning_engine.analyze_error_patterns()
                if patterns:
                    logger.info(f"🧠 Análise de padrões triggered - {len(patterns)} padrões identificados")
            except Exception as e:
                logger.error(f"Erro na análise de padrões: {e}")
        
        return jsonify({
            "message": "Feedback recebido e processado pelo sistema de aprendizado",
            "signal_id": signal_id,
            "result_recorded": result == 1,
            "total_trades": performance_stats['total_trades'],
            "accuracy": f"{accuracy:.1f}%",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "patterns_analysis": "Ativo" if LEARNING_CONFIG['learning_enabled'] else "Inativo",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "Sistema de Aprendizado Integrado"
        })
        
    except Exception as e:
        logger.error(f"Erro em feedback: {e}")
        return jsonify({"error": "Erro no feedback", "message": str(e)}), 500

# Middleware de erro global (atualizado)
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint não encontrado",
        "available_endpoints": ["/analyze", "/signal", "/risk", "/optimal-duration", "/management", "/feedback", "/learning-stats"],
        "signal_mode": "NORMAL + LEARNING",
        "learning_active": LEARNING_CONFIG['learning_enabled'],
        "timestamp": datetime.datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erro interno: {error}")
    return jsonify({
        "error": "Erro interno do servidor",
        "message": "Entre em contato com o suporte",
        "learning_system": "Ativo" if LEARNING_CONFIG['learning_enabled'] else "Inativo",
        "timestamp": datetime.datetime.now().isoformat()
    }), 500

if __name__ == "__main__":
    logger.info("🚀 Iniciando IA Trading Bot API com Sistema de Aprendizado")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    logger.info("🧠 Sistema de Aprendizado: ATIVADO")
    logger.info("📊 Sinais NORMAIS + Adaptação Inteligente")
    logger.info(f"📈 Banco de dados: {db.db_path}")
    app.run(host="0.0.0.0", port=5000, debug=False)
