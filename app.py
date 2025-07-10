from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import datetime
import logging
import sqlite3
import json
import threading
import time
import os
from collections import defaultdict, deque
import math

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key válida
VALID_API_KEY = "bhcOGajqbfFfolT"

# ✅ SISTEMA DE INVERSÃO AUTOMÁTICA - MELHORADO!
INVERSION_SYSTEM = {
    'active': True,
    'is_inverse_mode': False,
    'consecutive_errors': 0,
    'max_errors': 3,
    'total_inversions': 0,
    'last_inversion_time': None,
    'inversion_history': [],
    'adaptive_threshold': True,  # NOVO: Threshold adaptativo
    'performance_weight': 0.7   # NOVO: Peso da performance para ajustar threshold
}

# Configuração para Render
DB_PATH = os.environ.get('DB_PATH', '/tmp/trading_data.db')

# Configurações do sistema de aprendizado AVANÇADO
LEARNING_CONFIG = {
    'min_samples_for_learning': int(os.environ.get('MIN_SAMPLES', '15')),
    'adaptation_rate': float(os.environ.get('ADAPTATION_RATE', '0.15')),
    'error_pattern_window': int(os.environ.get('PATTERN_WINDOW', '100')),
    'confidence_adjustment_factor': float(os.environ.get('CONFIDENCE_FACTOR', '0.08')),
    'learning_enabled': os.environ.get('LEARNING_ENABLED', 'true').lower() == 'true',
    'reinforcement_learning': True,  # NOVO: Aprendizado por reforço
    'temporal_learning': True,       # NOVO: Aprendizado temporal
    'sequence_learning': True,       # NOVO: Aprendizado de sequências
    'correlation_analysis': True,    # NOVO: Análise de correlação
    'dynamic_weighting': True       # NOVO: Pesos dinâmicos
}

class AdvancedTradingDatabase:
    """Classe APRIMORADA para gerenciar o banco de dados SQLite"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Inicializar tabelas do banco de dados - VERSÃO AVANÇADA"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de sinais - EXPANDIDA
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                original_direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                volatility REAL,
                duration_type TEXT,
                duration_value INTEGER,
                result INTEGER,
                pnl REAL,
                martingale_level INTEGER DEFAULT 0,
                market_condition TEXT,
                technical_factors TEXT,
                is_inverted BOOLEAN DEFAULT 0,
                consecutive_errors_before INTEGER DEFAULT 0,
                inversion_mode TEXT DEFAULT 'normal',
                hour_of_day INTEGER,           -- NOVO: Hora do dia
                day_of_week INTEGER,           -- NOVO: Dia da semana
                market_session TEXT,           -- NOVO: Sessão do mercado
                sequence_position INTEGER DEFAULT 0, -- NOVO: Posição na sequência
                confidence_source TEXT,        -- NOVO: Fonte da confiança
                learning_weight REAL DEFAULT 1.0, -- NOVO: Peso para aprendizado
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                feedback_received_at TEXT
            )
        ''')
        
        # NOVA TABELA: Q-Learning para aprendizado por reforço
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS q_learning_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_hash TEXT NOT NULL UNIQUE,
                state_description TEXT,
                action_call_value REAL DEFAULT 0.0,
                action_put_value REAL DEFAULT 0.0,
                visits_count INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                average_reward REAL DEFAULT 0.0,
                exploration_count INTEGER DEFAULT 0
            )
        ''')
        
        # NOVA TABELA: Padrões de sequência
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sequence_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_hash TEXT NOT NULL UNIQUE,
                sequence_data TEXT NOT NULL,  -- JSON com a sequência
                pattern_length INTEGER NOT NULL,
                success_rate REAL DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                total_reward REAL DEFAULT 0.0,
                confidence_multiplier REAL DEFAULT 1.0,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # NOVA TABELA: Análise temporal
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour_of_day INTEGER NOT NULL,
                day_of_week INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                total_trades INTEGER DEFAULT 0,
                won_trades INTEGER DEFAULT 0,
                average_confidence REAL DEFAULT 0.0,
                volatility_avg REAL DEFAULT 0.0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(hour_of_day, day_of_week, symbol, direction)
            )
        ''')
        
        # NOVA TABELA: Correlações entre variáveis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS correlation_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variable1 TEXT NOT NULL,
                variable2 TEXT NOT NULL,
                correlation_value REAL NOT NULL,
                significance REAL DEFAULT 0.0,
                sample_size INTEGER DEFAULT 0,
                last_calculated TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(variable1, variable2)
            )
        ''')
        
        # Tabelas existentes...
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inversion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                from_mode TEXT NOT NULL,
                to_mode TEXT NOT NULL,
                consecutive_errors INTEGER NOT NULL,
                trigger_reason TEXT,
                total_inversions_so_far INTEGER DEFAULT 0,
                performance_before REAL DEFAULT 0.0,  -- NOVO
                adaptive_threshold INTEGER DEFAULT 3,  -- NOVO
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                conditions TEXT NOT NULL,
                error_rate REAL NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence_adjustment REAL DEFAULT 0,
                severity_score REAL DEFAULT 0.0,     -- NOVO
                pattern_stability REAL DEFAULT 0.0,  -- NOVO
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT NOT NULL UNIQUE,
                parameter_value REAL NOT NULL,
                learning_rate REAL DEFAULT 0.1,      -- NOVO
                momentum REAL DEFAULT 0.0,          -- NOVO
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                update_reason TEXT,
                performance_impact REAL DEFAULT 0.0  -- NOVO
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_signal(self, signal_data):
        """Salvar sinal no banco de dados - VERSÃO EXPANDIDA"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extrair informações temporais
        timestamp = datetime.datetime.fromisoformat(signal_data.get('timestamp'))
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Determinar sessão do mercado
        market_session = self._determine_market_session(hour_of_day)
        
        cursor.execute('''
            INSERT INTO signals (
                timestamp, symbol, direction, original_direction, confidence, entry_price, 
                volatility, duration_type, duration_value, martingale_level,
                market_condition, technical_factors, is_inverted, consecutive_errors_before, 
                inversion_mode, hour_of_day, day_of_week, market_session, sequence_position,
                confidence_source, learning_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('timestamp'),
            signal_data.get('symbol', 'R_50'),
            signal_data.get('direction'),
            signal_data.get('original_direction'),
            signal_data.get('confidence'),
            signal_data.get('entry_price'),
            signal_data.get('volatility'),
            signal_data.get('duration_type'),
            signal_data.get('duration_value'),
            signal_data.get('martingale_level', 0),
            signal_data.get('market_condition', 'neutral'),
            json.dumps(signal_data.get('technical_factors', {})),
            signal_data.get('is_inverted', False),
            signal_data.get('consecutive_errors_before', 0),
            signal_data.get('inversion_mode', 'normal'),
            hour_of_day,
            day_of_week,
            market_session,
            signal_data.get('sequence_position', 0),
            signal_data.get('confidence_source', 'technical'),
            signal_data.get('learning_weight', 1.0)
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    
    def _determine_market_session(self, hour):
        """Determinar sessão do mercado baseada na hora"""
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'european'
        else:
            return 'american'
    
    def save_q_learning_state(self, state_hash, state_desc, action, reward):
        """Salvar estado Q-Learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verificar se estado existe
        cursor.execute('SELECT * FROM q_learning_states WHERE state_hash = ?', (state_hash,))
        existing = cursor.fetchone()
        
        if existing:
            # Atualizar Q-Value usando fórmula Q-Learning
            old_q_value = existing[2] if action == 'CALL' else existing[3]
            learning_rate = 0.1
            new_q_value = old_q_value + learning_rate * (reward - old_q_value)
            
            if action == 'CALL':
                cursor.execute('''
                    UPDATE q_learning_states 
                    SET action_call_value = ?, visits_count = visits_count + 1, 
                        last_updated = ?, average_reward = (average_reward * visits_count + ?) / (visits_count + 1)
                    WHERE state_hash = ?
                ''', (new_q_value, datetime.datetime.now().isoformat(), reward, state_hash))
            else:
                cursor.execute('''
                    UPDATE q_learning_states 
                    SET action_put_value = ?, visits_count = visits_count + 1,
                        last_updated = ?, average_reward = (average_reward * visits_count + ?) / (visits_count + 1)
                    WHERE state_hash = ?
                ''', (new_q_value, datetime.datetime.now().isoformat(), reward, state_hash))
        else:
            # Criar novo estado
            call_value = reward if action == 'CALL' else 0.0
            put_value = reward if action == 'PUT' else 0.0
            
            cursor.execute('''
                INSERT INTO q_learning_states 
                (state_hash, state_description, action_call_value, action_put_value, visits_count, average_reward)
                VALUES (?, ?, ?, ?, 1, ?)
            ''', (state_hash, state_desc, call_value, put_value, reward))
        
        conn.commit()
        conn.close()
    
    def get_q_learning_action(self, state_hash):
        """Obter melhor ação baseada em Q-Learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT action_call_value, action_put_value, visits_count FROM q_learning_states WHERE state_hash = ?', (state_hash,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            call_value, put_value, visits = result
            # Exploration vs Exploitation (epsilon-greedy)
            epsilon = max(0.05, 0.3 / (1 + visits * 0.1))  # Diminui exploração com experiência
            
            if random.random() < epsilon:
                return random.choice(['CALL', 'PUT']), 0.5  # Exploração
            else:
                if call_value > put_value:
                    return 'CALL', min(0.95, 0.5 + abs(call_value - put_value) * 0.1)
                else:
                    return 'PUT', min(0.95, 0.5 + abs(call_value - put_value) * 0.1)
        else:
            return random.choice(['CALL', 'PUT']), 0.5  # Primeiro encontro
    
    def save_sequence_pattern(self, sequence_data, success_rate):
        """Salvar padrão de sequência"""
        sequence_hash = hash(str(sequence_data))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, occurrences, total_reward FROM sequence_patterns WHERE sequence_hash = ?', (str(sequence_hash),))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE sequence_patterns 
                SET success_rate = ?, occurrences = occurrences + 1, 
                    total_reward = total_reward + ?, last_seen = ?
                WHERE id = ?
            ''', (success_rate, success_rate, datetime.datetime.now().isoformat(), existing[0]))
        else:
            cursor.execute('''
                INSERT INTO sequence_patterns 
                (sequence_hash, sequence_data, pattern_length, success_rate, total_reward)
                VALUES (?, ?, ?, ?, ?)
            ''', (str(sequence_hash), json.dumps(sequence_data), len(sequence_data), success_rate, success_rate))
        
        conn.commit()
        conn.close()
    
    def update_temporal_pattern(self, hour, day, symbol, direction, won):
        """Atualizar padrão temporal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO temporal_patterns 
            (hour_of_day, day_of_week, symbol, direction, success_rate, total_trades, won_trades)
            VALUES (?, ?, ?, ?, 0.0, 0, 0)
        ''', (hour, day, symbol, direction))
        
        cursor.execute('''
            UPDATE temporal_patterns 
            SET total_trades = total_trades + 1,
                won_trades = won_trades + ?,
                success_rate = CAST(won_trades + ? AS REAL) / (total_trades + 1),
                last_updated = ?
            WHERE hour_of_day = ? AND day_of_week = ? AND symbol = ? AND direction = ?
        ''', (won, won, datetime.datetime.now().isoformat(), hour, day, symbol, direction))
        
        conn.commit()
        conn.close()
    
    def get_temporal_performance(self, hour, day, symbol, direction):
        """Obter performance temporal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT success_rate, total_trades FROM temporal_patterns 
            WHERE hour_of_day = ? AND day_of_week = ? AND symbol = ? AND direction = ?
        ''', (hour, day, symbol, direction))
        
        result = cursor.fetchone()
        conn.close()
        
        return result if result and result[1] >= 5 else (0.5, 0)  # Padrão se poucos dados
    
    def calculate_correlations(self):
        """Calcular correlações entre variáveis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obter dados recentes para análise
        cursor.execute('''
            SELECT volatility, confidence, result, martingale_level, hour_of_day, day_of_week
            FROM signals WHERE result IS NOT NULL 
            ORDER BY created_at DESC LIMIT 200
        ''')
        
        data = cursor.fetchall()
        
        if len(data) < 20:
            conn.close()
            return
        
        # Calcular correlação simples entre volatilidade e sucesso
        volatilities = [row[0] for row in data if row[0] is not None]
        results = [row[2] for row in data if row[0] is not None]
        
        if len(volatilities) >= 10:
            correlation = self._calculate_correlation(volatilities, results)
            
            cursor.execute('''
                INSERT OR REPLACE INTO correlation_analysis 
                (variable1, variable2, correlation_value, sample_size)
                VALUES (?, ?, ?, ?)
            ''', ('volatility', 'success_rate', correlation, len(volatilities)))
        
        # Calcular outras correlações...
        confidences = [row[1] for row in data if row[1] is not None]
        if len(confidences) >= 10:
            correlation = self._calculate_correlation(confidences, results[:len(confidences)])
            
            cursor.execute('''
                INSERT OR REPLACE INTO correlation_analysis 
                (variable1, variable2, correlation_value, sample_size)
                VALUES (?, ?, ?, ?)
            ''', ('confidence', 'success_rate', correlation, len(confidences)))
        
        conn.commit()
        conn.close()
    
    def _calculate_correlation(self, x, y):
        """Calcular correlação de Pearson simples"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i]**2 for i in range(n))
        sum_y2 = sum(y[i]**2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    # Métodos existentes adaptados...
    def save_inversion_event(self, from_mode, to_mode, consecutive_errors, reason, performance_before=0.0, adaptive_threshold=3):
        """Salvar evento de inversão - VERSÃO MELHORADA"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO inversion_history (
                timestamp, from_mode, to_mode, consecutive_errors, trigger_reason, 
                total_inversions_so_far, performance_before, adaptive_threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            from_mode, to_mode, consecutive_errors, reason,
            INVERSION_SYSTEM['total_inversions'], performance_before, adaptive_threshold
        ))
        
        conn.commit()
        conn.close()
    
    def update_signal_result(self, signal_id, result, pnl=0):
        """Atualizar resultado do sinal - COM APRENDIZADO TEMPORAL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Atualizar sinal
        cursor.execute('''
            UPDATE signals 
            SET result = ?, pnl = ?, feedback_received_at = ?
            WHERE id = ?
        ''', (result, pnl, datetime.datetime.now().isoformat(), signal_id))
        
        # Obter dados do sinal para aprendizado
        cursor.execute('''
            SELECT symbol, direction, hour_of_day, day_of_week, volatility, confidence, technical_factors
            FROM signals WHERE id = ?
        ''', (signal_id,))
        
        signal_data = cursor.fetchone()
        if signal_data:
            symbol, direction, hour, day, volatility, confidence, technical_factors = signal_data
            
            # Atualizar padrão temporal
            self.update_temporal_pattern(hour, day, symbol, direction, result)
            
            # Q-Learning: criar estado e salvar resultado
            if LEARNING_CONFIG['reinforcement_learning']:
                state_hash = f"{symbol}_{direction}_{hour}_{volatility//10}"
                state_desc = f"Symbol:{symbol}, Direction:{direction}, Hour:{hour}, Vol:{volatility//10*10}"
                reward = 1.0 if result == 1 else -0.5
                self.save_q_learning_state(state_hash, state_desc, direction, reward)
        
        conn.commit()
        conn.close()
    
    def get_recent_performance(self, limit=100, symbol=None):
        """Obter performance recente - VERSÃO EXPANDIDA"""
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

class EnhancedInversionManager:
    """Gerenciador AVANÇADO do sistema de inversão automática"""
    
    def __init__(self, database):
        self.db = database
        
    def calculate_adaptive_threshold(self):
        """Calcular threshold adaptativo baseado na performance"""
        if not INVERSION_SYSTEM['adaptive_threshold']:
            return INVERSION_SYSTEM['max_errors']
        
        # Obter performance recente
        recent_data = self.db.get_recent_performance(50)
        if len(recent_data) < 10:
            return INVERSION_SYSTEM['max_errors']
        
        # Calcular taxa de acerto
        wins = sum(1 for signal in recent_data if signal[10] == 1)
        accuracy = wins / len(recent_data)
        
        # Ajustar threshold baseado na performance
        if accuracy > 0.65:
            # Performance boa - pode tolerar mais erros antes de inverter
            return min(5, INVERSION_SYSTEM['max_errors'] + 1)
        elif accuracy < 0.35:
            # Performance ruim - inverter mais rapidamente
            return max(2, INVERSION_SYSTEM['max_errors'] - 1)
        else:
            return INVERSION_SYSTEM['max_errors']
    
    def should_invert_mode(self):
        """Verificar se deve inverter o modo - VERSÃO ADAPTATIVA"""
        adaptive_threshold = self.calculate_adaptive_threshold()
        return INVERSION_SYSTEM['consecutive_errors'] >= adaptive_threshold
    
    def switch_inversion_mode(self, reason="Max consecutive errors reached"):
        """Alternar modo de inversão - VERSÃO MELHORADA"""
        # Calcular performance antes da inversão
        recent_data = self.db.get_recent_performance(20)
        performance_before = 0.0
        if recent_data:
            wins = sum(1 for signal in recent_data if signal[10] == 1)
            performance_before = wins / len(recent_data)
        
        old_mode = "inverse" if INVERSION_SYSTEM['is_inverse_mode'] else "normal"
        INVERSION_SYSTEM['is_inverse_mode'] = not INVERSION_SYSTEM['is_inverse_mode']
        
        adaptive_threshold = self.calculate_adaptive_threshold()
        INVERSION_SYSTEM['consecutive_errors'] = 0
        INVERSION_SYSTEM['total_inversions'] += 1
        INVERSION_SYSTEM['last_inversion_time'] = datetime.datetime.now().isoformat()
        
        new_mode = "inverse" if INVERSION_SYSTEM['is_inverse_mode'] else "normal"
        
        # Registrar no histórico
        INVERSION_SYSTEM['inversion_history'].append({
            'timestamp': INVERSION_SYSTEM['last_inversion_time'],
            'from_mode': old_mode,
            'to_mode': new_mode,
            'consecutive_errors': adaptive_threshold,
            'reason': reason,
            'performance_before': performance_before,
            'adaptive_threshold': adaptive_threshold
        })
        
        # Salvar no banco
        self.db.save_inversion_event(old_mode, new_mode, adaptive_threshold, reason, performance_before, adaptive_threshold)
        
        logger.info(f"🔄 INVERSÃO AUTOMÁTICA ADAPTATIVA: {old_mode.upper()} → {new_mode.upper()}")
        logger.info(f"   Motivo: {reason}")
        logger.info(f"   Threshold adaptativo: {adaptive_threshold}")
        logger.info(f"   Performance antes: {performance_before:.1%}")
        logger.info(f"   Total de inversões: {INVERSION_SYSTEM['total_inversions']}")
    
    def invert_signal(self, signal):
        """Inverter sinal de trading"""
        signal_map = {
            'CALL': 'PUT', 'PUT': 'CALL', 
            'BUY': 'SELL', 'SELL': 'BUY',
            'LONG': 'SHORT', 'SHORT': 'LONG',
            'COMPRA': 'VENDA', 'VENDA': 'COMPRA'
        }
        return signal_map.get(signal.upper(), signal)
    
    def handle_signal_result(self, result):
        """Processar resultado do sinal - VERSÃO MELHORADA"""
        if not INVERSION_SYSTEM['active']:
            return
            
        if result == 0:  # Loss
            INVERSION_SYSTEM['consecutive_errors'] += 1
            adaptive_threshold = self.calculate_adaptive_threshold()
            
            logger.info(f"❌ Erro #{INVERSION_SYSTEM['consecutive_errors']} de {adaptive_threshold} (Modo: {'INVERSO' if INVERSION_SYSTEM['is_inverse_mode'] else 'NORMAL'})")
            
            if self.should_invert_mode():
                self.switch_inversion_mode(f"Threshold adaptativo atingido ({adaptive_threshold})")
        else:  # Win
            if INVERSION_SYSTEM['consecutive_errors'] > 0:
                logger.info(f"✅ Win! Resetando contador de erros (era {INVERSION_SYSTEM['consecutive_errors']})")
                INVERSION_SYSTEM['consecutive_errors'] = 0
    
    def get_final_signal(self, original_signal):
        """Obter sinal final com inversão adaptativa"""
        if not INVERSION_SYSTEM['active']:
            return original_signal, False, "normal"
            
        if INVERSION_SYSTEM['is_inverse_mode']:
            inverted_signal = self.invert_signal(original_signal)
            return inverted_signal, True, "inverse"
        else:
            return original_signal, False, "normal"
    
    def get_inversion_status(self):
        """Obter status atual do sistema de inversão - VERSÃO EXPANDIDA"""
        adaptive_threshold = self.calculate_adaptive_threshold()
        
        return {
            'active': INVERSION_SYSTEM['active'],
            'current_mode': "inverse" if INVERSION_SYSTEM['is_inverse_mode'] else "normal",
            'consecutive_errors': INVERSION_SYSTEM['consecutive_errors'],
            'adaptive_threshold': adaptive_threshold,
            'original_threshold': INVERSION_SYSTEM['max_errors'],
            'total_inversions': INVERSION_SYSTEM['total_inversions'],
            'last_inversion': INVERSION_SYSTEM['last_inversion_time'],
            'errors_until_inversion': adaptive_threshold - INVERSION_SYSTEM['consecutive_errors'],
            'adaptive_mode': INVERSION_SYSTEM['adaptive_threshold']
        }

class AdvancedLearningEngine:
    """Motor de aprendizado AVANÇADO com múltiplas técnicas"""
    
    def __init__(self, database):
        self.db = database
        self.recent_signals = deque(maxlen=LEARNING_CONFIG['error_pattern_window'])
        self.confidence_adjustments = defaultdict(float)
        self.sequence_memory = deque(maxlen=10)  # Memória de sequência
        self.learning_weights = defaultdict(lambda: 1.0)
        
    def create_state_representation(self, signal_data):
        """Criar representação de estado para Q-Learning"""
        symbol = signal_data.get('symbol', 'R_50')
        volatility_bucket = int(signal_data.get('volatility', 50) // 10) * 10
        hour = datetime.datetime.now().hour
        
        state_hash = f"{symbol}_{volatility_bucket}_{hour}"
        state_description = f"Symbol:{symbol}, Vol:{volatility_bucket}, Hour:{hour}"
        
        return state_hash, state_description
    
    def get_q_learning_signal(self, signal_data):
        """Obter sinal baseado em Q-Learning"""
        if not LEARNING_CONFIG['reinforcement_learning']:
            return None, 0.0
            
        state_hash, state_desc = self.create_state_representation(signal_data)
        return self.db.get_q_learning_action(state_hash)
    
    def analyze_sequence_patterns(self):
        """Analisar padrões de sequência de trades"""
        if not LEARNING_CONFIG['sequence_learning'] or len(self.sequence_memory) < 3:
            return []
        
        patterns_found = []
        
        # Analisar últimas 3-5 operações
        for length in range(3, min(6, len(self.sequence_memory) + 1)):
            if len(self.sequence_memory) >= length:
                sequence = list(self.sequence_memory)[-length:]
                
                # Calcular taxa de sucesso da sequência
                results = [trade.get('result', 0) for trade in sequence if 'result' in trade]
                if results:
                    success_rate = sum(results) / len(results)
                    self.db.save_sequence_pattern(sequence, success_rate)
                    
                    patterns_found.append({
                        'sequence_length': length,
                        'success_rate': success_rate,
                        'pattern': sequence
                    })
        
        return patterns_found
    
    def get_temporal_adjustment(self, signal_data):
        """Obter ajuste baseado em análise temporal"""
        if not LEARNING_CONFIG['temporal_learning']:
            return 0.0, "temporal_disabled"
        
        now = datetime.datetime.now()
        hour = now.hour
        day = now.weekday()
        symbol = signal_data.get('symbol', 'R_50')
        direction = signal_data.get('direction', 'CALL')
        
        # Obter performance temporal
        success_rate, total_trades = self.db.get_temporal_performance(hour, day, symbol, direction)
        
        if total_trades >= 5:
            # Ajustar confiança baseado na performance temporal
            if success_rate > 0.65:
                adjustment = 10  # Horário favorável
            elif success_rate < 0.35:
                adjustment = -15  # Horário desfavorável
            else:
                adjustment = 0
                
            return adjustment, f"temporal_pattern_h{hour}_d{day}"
        
        return 0.0, "temporal_insufficient_data"
    
    def analyze_error_patterns(self):
        """Analisar padrões de erro - VERSÃO AVANÇADA"""
        recent_data = self.db.get_recent_performance(LEARNING_CONFIG['error_pattern_window'])
        
        if len(recent_data) < LEARNING_CONFIG['min_samples_for_learning']:
            return []
            
        patterns_found = []
        
        # 1. Análise por correlação
        if LEARNING_CONFIG['correlation_analysis']:
            self.db.calculate_correlations()
        
        # 2. Análise de volatilidade avançada
        volatility_patterns = self._analyze_volatility_patterns(recent_data)
        patterns_found.extend(volatility_patterns)
        
        # 3. Análise de sequências
        sequence_patterns = self.analyze_sequence_patterns()
        patterns_found.extend(sequence_patterns)
        
        # 4. Análise de martingale inteligente
        martingale_patterns = self._analyze_martingale_patterns(recent_data)
        patterns_found.extend(martingale_patterns)
        
        # 5. Análise de performance por sessão
        session_patterns = self._analyze_session_patterns(recent_data)
        patterns_found.extend(session_patterns)
        
        return patterns_found
    
    def _analyze_volatility_patterns(self, recent_data):
        """Análise avançada de padrões de volatilidade"""
        patterns = []
        
        # Agrupar por faixas de volatilidade mais granulares
        vol_ranges = {
            'very_low': (0, 20),
            'low': (20, 40), 
            'medium': (40, 60),
            'high': (60, 80),
            'very_high': (80, 100)
        }
        
        for range_name, (min_vol, max_vol) in vol_ranges.items():
            range_signals = [s for s in recent_data if s[7] and min_vol <= s[7] < max_vol]
            
            if len(range_signals) >= 8:
                wins = sum(1 for s in range_signals if s[10] == 1)
                win_rate = wins / len(range_signals)
                
                if win_rate < 0.35 or win_rate > 0.75:
                    self.db.save_error_pattern(
                        f'volatility_{range_name}_pattern',
                        {'volatility_range': range_name, 'min_vol': min_vol, 'max_vol': max_vol},
                        1 - win_rate if win_rate < 0.35 else 0.25 - win_rate
                    )
                    
                    patterns.append({
                        'type': f'volatility_{range_name}',
                        'win_rate': win_rate,
                        'sample_size': len(range_signals),
                        'significance': 'high' if len(range_signals) >= 15 else 'medium'
                    })
        
        return patterns
    
    def _analyze_martingale_patterns(self, recent_data):
        """Análise inteligente de padrões de martingale"""
        patterns = []
        
        # Analisar performance por nível de martingale
        martingale_levels = defaultdict(list)
        for signal in recent_data:
            level = signal[12]  # martingale_level
            result = signal[10]  # result
            if level is not None and result is not None:
                martingale_levels[level].append(result)
        
        for level, results in martingale_levels.items():
            if len(results) >= 5 and level > 0:
                win_rate = sum(results) / len(results)
                
                if win_rate < 0.4:
                    self.db.save_error_pattern(
                        f'martingale_level_{level}_poor_performance',
                        {'martingale_level': level},
                        1 - win_rate
                    )
                    
                    patterns.append({
                        'type': f'martingale_level_{level}',
                        'win_rate': win_rate,
                        'recommendation': 'avoid_high_martingale' if level >= 3 else 'caution_martingale'
                    })
        
        return patterns
    
    def _analyze_session_patterns(self, recent_data):
        """Análise de padrões por sessão de mercado"""
        patterns = []
        
        session_performance = defaultdict(list)
        for signal in recent_data:
            session = signal[18] if len(signal) > 18 else 'unknown'  # market_session
            result = signal[10]  # result
            if session and result is not None:
                session_performance[session].append(result)
        
        for session, results in session_performance.items():
            if len(results) >= 10:
                win_rate = sum(results) / len(results)
                
                if win_rate < 0.4 or win_rate > 0.7:
                    self.db.save_error_pattern(
                        f'session_{session}_pattern',
                        {'market_session': session},
                        abs(0.5 - win_rate)
                    )
                    
                    patterns.append({
                        'type': f'session_{session}',
                        'win_rate': win_rate,
                        'recommendation': 'favorable' if win_rate > 0.6 else 'unfavorable'
                    })
        
        return patterns
    
    def adapt_confidence(self, signal_data):
        """Adaptar confiança usando MÚLTIPLAS técnicas de aprendizado"""
        base_confidence = signal_data.get('confidence', 70)
        adjustments = []
        total_adjustment = 0
        
        # 1. Q-Learning adjustment
        if LEARNING_CONFIG['reinforcement_learning']:
            q_direction, q_confidence = self.get_q_learning_signal(signal_data)
            if q_direction and q_confidence > 0.6:
                q_adjustment = (q_confidence - 0.5) * 20  # Convert to confidence adjustment
                if q_direction == signal_data.get('direction'):
                    adjustments.append(('q_learning_boost', q_adjustment))
                    total_adjustment += q_adjustment
                else:
                    adjustments.append(('q_learning_conflict', -q_adjustment))
                    total_adjustment -= q_adjustment
        
        # 2. Temporal adjustment
        temporal_adj, temporal_reason = self.get_temporal_adjustment(signal_data)
        if temporal_adj != 0:
            adjustments.append((temporal_reason, temporal_adj))
            total_adjustment += temporal_adj
        
        # 3. Pattern-based adjustments (existing logic enhanced)
        error_patterns = self.db.get_error_patterns()
        for pattern in error_patterns:
            pattern_type = pattern[1]
            conditions = json.loads(pattern[2])
            error_rate = pattern[3]
            
            pattern_applies = False
            
            if pattern_type.startswith('volatility_'):
                vol = signal_data.get('volatility', 50)
                if 'min_vol' in conditions and 'max_vol' in conditions:
                    if conditions['min_vol'] <= vol < conditions['max_vol']:
                        pattern_applies = True
            
            elif pattern_type.startswith('session_'):
                current_hour = datetime.datetime.now().hour
                current_session = self._determine_session(current_hour)
                if conditions.get('market_session') == current_session:
                    pattern_applies = True
            
            elif pattern_type.startswith('martingale_'):
                if signal_data.get('martingale_level', 0) == conditions.get('martingale_level'):
                    pattern_applies = True
            
            if pattern_applies:
                pattern_adjustment = -error_rate * 25  # Stronger adjustment for learned patterns
                adjustments.append((pattern_type, pattern_adjustment))
                total_adjustment += pattern_adjustment
        
        # 4. Sequence-based adjustment
        if len(self.sequence_memory) >= 3:
            recent_results = [t.get('result', 0) for t in list(self.sequence_memory)[-3:] if 'result' in t]
            if len(recent_results) == 3:
                if sum(recent_results) == 0:  # 3 losses in a row
                    adjustments.append(('sequence_losing_streak', -10))
                    total_adjustment -= 10
                elif sum(recent_results) == 3:  # 3 wins in a row
                    adjustments.append(('sequence_winning_streak', 5))
                    total_adjustment += 5
        
        # Apply adjustments with dampening
        adapted_confidence = base_confidence + (total_adjustment * 0.7)  # Dampen aggressive adjustments
        adapted_confidence = max(50, min(95, adapted_confidence))
        
        if adjustments:
            logger.info(f"🧠 Confiança AVANÇADA: {base_confidence:.1f} → {adapted_confidence:.1f}")
            logger.info(f"   Ajustes aplicados: {adjustments}")
        
        return adapted_confidence, adjustments
    
    def _determine_session(self, hour):
        """Determinar sessão do mercado"""
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'european'
        else:
            return 'american'
    
    def update_sequence_memory(self, signal_data):
        """Atualizar memória de sequência"""
        self.sequence_memory.append({
            'direction': signal_data.get('direction'),
            'confidence': signal_data.get('confidence'),
            'volatility': signal_data.get('volatility'),
            'timestamp': signal_data.get('timestamp'),
            'result': signal_data.get('result')  # Will be updated later
        })
    
    def update_performance_metrics(self):
        """Atualizar métricas de performance - VERSÃO AVANÇADA"""
        recent_data = self.db.get_recent_performance(150)
        
        if not recent_data:
            return
            
        # Calcular métricas avançadas
        total_signals = len(recent_data)
        won_signals = sum(1 for signal in recent_data if signal[10] == 1)
        accuracy = (won_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Calcular tendência de accuracy (últimos 30 vs 30 anteriores)
        if total_signals >= 60:
            recent_30 = recent_data[:30]
            previous_30 = recent_data[30:60]
            
            recent_accuracy = sum(1 for s in recent_30 if s[10] == 1) / 30 * 100
            previous_accuracy = sum(1 for s in previous_30 if s[10] == 1) / 30 * 100
            
            accuracy_trend = recent_accuracy - previous_accuracy
        else:
            accuracy_trend = 0
        
        # Ajustar parâmetros adaptativos com mais inteligência
        risk_factor = self.db.get_adaptive_parameter('risk_factor', 1.0)
        aggression_factor = self.db.get_adaptive_parameter('aggression_factor', 1.0)
        
        # Lógica de ajuste mais sofisticada
        if accuracy < 40:
            # Performance muito baixa - modo conservador
            new_risk_factor = max(0.3, risk_factor - 0.15)
            new_aggression_factor = max(0.5, aggression_factor - 0.1)
            reason = f'Performance crítica: {accuracy:.1f}%'
        elif accuracy < 50 and accuracy_trend < -5:
            # Performance baixa e piorando
            new_risk_factor = max(0.5, risk_factor - 0.1)
            new_aggression_factor = max(0.7, aggression_factor - 0.05)
            reason = f'Performance declinante: {accuracy:.1f}% (trend: {accuracy_trend:.1f}%)'
        elif accuracy > 70 and accuracy_trend > 5:
            # Performance excelente e melhorando
            new_risk_factor = min(1.8, risk_factor + 0.1)
            new_aggression_factor = min(1.5, aggression_factor + 0.05)
            reason = f'Performance excelente: {accuracy:.1f}% (trend: +{accuracy_trend:.1f}%)'
        elif accuracy > 60:
            # Performance boa
            new_risk_factor = min(1.3, risk_factor + 0.05)
            new_aggression_factor = min(1.2, aggression_factor + 0.02)
            reason = f'Performance boa: {accuracy:.1f}%'
        else:
            # Manter parâmetros atuais
            new_risk_factor = risk_factor
            new_aggression_factor = aggression_factor
            reason = f'Performance estável: {accuracy:.1f}%'
        
        # Atualizar se houve mudança significativa
        if abs(new_risk_factor - risk_factor) > 0.02:
            self.db.update_adaptive_parameter('risk_factor', new_risk_factor, reason)
            
        if abs(new_aggression_factor - aggression_factor) > 0.02:
            self.db.update_adaptive_parameter('aggression_factor', new_aggression_factor, reason)
        
        return {
            'total_signals': total_signals,
            'accuracy': accuracy,
            'accuracy_trend': accuracy_trend,
            'won_signals': won_signals,
            'risk_factor': new_risk_factor,
            'aggression_factor': new_aggression_factor
        }

# Instâncias globais MELHORADAS
db = AdvancedTradingDatabase()
learning_engine = AdvancedLearningEngine(db)
inversion_manager = EnhancedInversionManager(db)

# Dados de histórico simples (mantidos para compatibilidade)
trade_history = []
performance_stats = {
    'total_trades': 0,
    'won_trades': 0,
    'total_pnl': 0.0
}

def validate_api_key():
    """Validar API Key"""
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
    """Análise técnica AVANÇADA com múltiplos sistemas de aprendizado"""
    try:
        if len(prices) >= 3:
            # Análise técnica base
            recent_trend = prices[-1] - prices[-3]
            volatility = abs(prices[-1] - prices[-2]) / prices[-2] * 100 if prices[-2] != 0 else 50
            
            # Direção original baseada em análise técnica
            if recent_trend > 0:
                original_direction = "CALL"
                base_confidence = 70 + min(volatility * 0.3, 20)
            else:
                original_direction = "PUT" 
                base_confidence = 70 + min(volatility * 0.3, 20)
            
            # 🧠 APLICAR APRENDIZADO AVANÇADO
            if learning_data and LEARNING_CONFIG['learning_enabled']:
                enhanced_learning_data = {
                    'direction': original_direction,
                    'confidence': base_confidence,
                    'volatility': volatility,
                    **learning_data
                }
                
                adapted_confidence, adjustments = learning_engine.adapt_confidence(enhanced_learning_data)
                
                # Q-Learning override se muito confiante
                q_direction, q_confidence = learning_engine.get_q_learning_signal(enhanced_learning_data)
                if q_direction and q_confidence > 0.8:
                    logger.info(f"🤖 Q-Learning override: {original_direction} → {q_direction} (conf: {q_confidence:.2f})")
                    original_direction = q_direction
                    adapted_confidence = min(95, adapted_confidence + 10)
                    adjustments.append(('q_learning_override', 10))
            else:
                adapted_confidence = base_confidence
                adjustments = []
            
            # 🔄 APLICAR SISTEMA DE INVERSÃO ADAPTATIVA
            final_direction, is_inverted, inversion_mode = inversion_manager.get_final_signal(original_direction)
            
            return {
                'original_direction': original_direction,
                'final_direction': final_direction,
                'confidence': round(adapted_confidence, 1),
                'is_inverted': is_inverted,
                'inversion_mode': inversion_mode,
                'adjustments': adjustments,
                'learning_applied': len(adjustments) > 0,
                'q_learning_used': any('q_learning' in adj[0] for adj in adjustments)
            }
            
        else:
            # Fallback com Q-Learning se disponível
            original_direction = "CALL" if random.random() > 0.5 else "PUT"
            confidence = 70 + random.uniform(0, 20)
            
            if learning_data and LEARNING_CONFIG['reinforcement_learning']:
                q_direction, q_confidence = learning_engine.get_q_learning_signal(learning_data)
                if q_direction:
                    original_direction = q_direction
                    confidence = max(confidence, q_confidence * 100)
            
            final_direction, is_inverted, inversion_mode = inversion_manager.get_final_signal(original_direction)
            
            return {
                'original_direction': original_direction,
                'final_direction': final_direction,
                'confidence': round(confidence, 1),
                'is_inverted': is_inverted,
                'inversion_mode': inversion_mode,
                'adjustments': [],
                'learning_applied': False,
                'q_learning_used': learning_data and LEARNING_CONFIG['reinforcement_learning']
            }
            
    except Exception as e:
        logger.error(f"Erro na análise técnica avançada: {e}")
        original_direction = "CALL" if random.random() > 0.5 else "PUT"
        final_direction, is_inverted, inversion_mode = inversion_manager.get_final_signal(original_direction)
        
        return {
            'original_direction': original_direction,
            'final_direction': final_direction,
            'confidence': 70.0,
            'is_inverted': is_inverted,
            'inversion_mode': inversion_mode,
            'adjustments': [],
            'learning_applied': False,
            'q_learning_used': False
        }

def extract_features(data):
    """Extrair dados dos parâmetros recebidos - VERSÃO EXPANDIDA"""
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

def background_learning_task():
    """Tarefa de aprendizado AVANÇADO em background"""
    while True:
        try:
            if LEARNING_CONFIG['learning_enabled']:
                # Analisar padrões de erro
                patterns = learning_engine.analyze_error_patterns()
                if patterns:
                    logger.info(f"🧠 APRENDIZADO AVANÇADO: {len(patterns)} padrões identificados")
                    for pattern in patterns[:3]:  # Log dos 3 principais
                        logger.info(f"   - {pattern.get('type', 'unknown')}: {pattern}")
                
                # Atualizar métricas avançadas
                metrics = learning_engine.update_performance_metrics()
                if metrics:
                    logger.info(f"📊 Métricas AVANÇADAS atualizadas:")
                    logger.info(f"   - Accuracy: {metrics['accuracy']:.1f}% (trend: {metrics.get('accuracy_trend', 0):+.1f}%)")
                    logger.info(f"   - Risk Factor: {metrics.get('risk_factor', 1.0):.2f}")
                    logger.info(f"   - Aggression Factor: {metrics.get('aggression_factor', 1.0):.2f}")
                
                # Calcular correlações periodicamente
                if LEARNING_CONFIG['correlation_analysis']:
                    db.calculate_correlations()
                    logger.info("🔗 Análise de correlações atualizada")
                
            # Aguardar antes da próxima análise
            time.sleep(180)  # 3 minutos para análise mais frequente
            
        except Exception as e:
            logger.error(f"Erro na tarefa de aprendizado avançado: {e}")
            time.sleep(60)

# Iniciar thread de aprendizado avançado
learning_thread = threading.Thread(target=background_learning_task, daemon=True)
learning_thread.start()

# ===============================
# ROTAS DA API - VERSÕES MELHORADAS
# ===============================

@app.route("/")
def home():
    """Home page com informações do sistema AVANÇADO"""
    recent_data = db.get_recent_performance(100)
    total_signals = len(recent_data)
    accuracy = (sum(1 for signal in recent_data if signal[10] == 1) / total_signals * 100) if total_signals > 0 else 0
    
    # Status dos sistemas
    inversion_status = inversion_manager.get_inversion_status()
    
    return jsonify({
        "status": "🚀 IA Trading Bot API - SISTEMA DE APRENDIZADO AVANÇADO + INVERSÃO ADAPTATIVA",
        "version": "5.0.0 - Advanced Learning + Adaptive Inversion + Q-Learning",
        "description": "API com Q-Learning, Análise Temporal, Sequências e Inversão Adaptativa",
        "model": "Multi-Layer Learning Engine + Adaptive Inversion System",
        "signal_mode": f"{inversion_status['current_mode'].upper()} + ADVANCED_LEARNING",
        
        "advanced_features": {
            "q_learning": LEARNING_CONFIG['reinforcement_learning'],
            "temporal_analysis": LEARNING_CONFIG['temporal_learning'],
            "sequence_patterns": LEARNING_CONFIG['sequence_learning'],
            "correlation_analysis": LEARNING_CONFIG['correlation_analysis'],
            "adaptive_inversion": inversion_status['adaptive_mode'],
            "dynamic_weighting": LEARNING_CONFIG['dynamic_weighting']
        },
        
        "inversion_system": {
            "active": inversion_status['active'],
            "current_mode": inversion_status['current_mode'],
            "consecutive_errors": inversion_status['consecutive_errors'],
            "adaptive_threshold": inversion_status['adaptive_threshold'],
            "original_threshold": inversion_status['original_threshold'],
            "errors_until_inversion": inversion_status['errors_until_inversion'],
            "total_inversions": inversion_status['total_inversions']
        },
        
        "learning_systems": {
            "reinforcement_learning": "Q-Learning para decisões baseadas em recompensa",
            "pattern_recognition": "Identificação automática de padrões de erro",
            "temporal_analysis": "Análise de performance por horário/dia",
            "sequence_learning": "Aprendizado de padrões sequenciais",
            "correlation_engine": "Análise de correlações entre variáveis",
            "adaptive_parameters": "Parâmetros que se ajustam automaticamente"
        },
        
        "endpoints": {
            "signal": "POST /signal - Sinais com aprendizado avançado + inversão adaptativa",
            "analyze": "POST /analyze - Análise de mercado com múltiplas técnicas de IA",
            "risk": "POST /risk - Avaliação de risco adaptativa",
            "feedback": "POST /feedback - Sistema de aprendizado por reforço",
            "learning-stats": "GET /learning-stats - Estatísticas de aprendizado avançado",
            "inversion-status": "GET /inversion-status - Status da inversão adaptativa",
            "q-learning-stats": "GET /q-learning-stats - Estatísticas do Q-Learning"
        },
        
        "current_stats": {
            "total_predictions": total_signals,
            "current_accuracy": f"{accuracy:.1f}%",
            "learning_samples": total_signals,
            "q_learning_states": "Ativo" if LEARNING_CONFIG['reinforcement_learning'] else "Inativo",
            "adaptive_mode": "Ativo" if inversion_status['adaptive_mode'] else "Fixo"
        },
        
        "learning_config": LEARNING_CONFIG,
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "Advanced Python AI with Q-Learning + Adaptive Inversion System"
    })

@app.route("/q-learning-stats", methods=["GET"])
def get_q_learning_stats():
    """Obter estatísticas do Q-Learning"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Obter estados Q-Learning
        cursor.execute('''
            SELECT state_hash, state_description, action_call_value, action_put_value, 
                   visits_count, average_reward
            FROM q_learning_states 
            ORDER BY visits_count DESC LIMIT 20
        ''')
        
        q_states = cursor.fetchall()
        
        # Estatísticas gerais
        cursor.execute('SELECT COUNT(*), AVG(visits_count), AVG(average_reward) FROM q_learning_states')
        general_stats = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            "q_learning_enabled": LEARNING_CONFIG['reinforcement_learning'],
            "total_states": general_stats[0] if general_stats else 0,
            "average_visits": round(general_stats[1], 2) if general_stats and general_stats[1] else 0,
            "average_reward": round(general_stats[2], 3) if general_stats and general_stats[2] else 0,
            "top_states": [
                {
                    "state": state[0],
                    "description": state[1],
                    "call_value": round(state[2], 3),
                    "put_value": round(state[3], 3),
                    "visits": state[4],
                    "avg_reward": round(state[5], 3),
                    "best_action": "CALL" if state[2] > state[3] else "PUT",
                    "confidence": abs(state[2] - state[3])
                }
                for state in q_states
            ],
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em q-learning-stats: {e}")
        return jsonify({"error": "Erro ao obter estatísticas Q-Learning", "message": str(e)}), 500

@app.route("/signal", methods=["POST", "OPTIONS"])
@app.route("/trading-signal", methods=["POST", "OPTIONS"])
@app.route("/get-signal", methods=["POST", "OPTIONS"])
@app.route("/smart-signal", methods=["POST", "OPTIONS"])
@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
@app.route("/prediction", methods=["POST", "OPTIONS"])
@app.route("/advanced-signal", methods=["POST", "OPTIONS"])
def generate_signal():
    """Gerar sinal com SISTEMA DE APRENDIZADO AVANÇADO"""
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        prices, volatility = extract_features(data)
        
        # Preparar dados EXPANDIDOS para aprendizado
        learning_data = {
            'symbol': data.get("symbol", "R_50"),
            'volatility': volatility,
            'market_condition': data.get("marketCondition", "neutral"),
            'martingale_level': data.get("martingaleLevel", 0),
            'current_price': data.get("currentPrice", 1000),
            'win_rate': data.get("winRate", 50),
            'today_pnl': data.get("todayPnL", 0)
        }
        
        # Análise técnica com APRENDIZADO AVANÇADO
        analysis_result = analyze_technical_pattern(prices, learning_data)
        
        # Dados do sinal
        current_price = data.get("currentPrice", 1000)
        symbol = data.get("symbol", "R_50")
        win_rate = data.get("winRate", 50)
        
        # Ajustar confiança baseada em performance (mantido)
        confidence = analysis_result['confidence']
        if win_rate > 60:
            confidence = min(confidence + 3, 95)
        elif win_rate < 40:
            confidence = max(confidence - 5, 65)
        
        # Preparar dados EXPANDIDOS para salvar no banco
        signal_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'symbol': symbol,
            'direction': analysis_result['final_direction'],
            'original_direction': analysis_result['original_direction'],
            'confidence': confidence,
            'entry_price': current_price,
            'volatility': volatility,
            'martingale_level': data.get("martingaleLevel", 0),
            'market_condition': data.get("marketCondition", "neutral"),
            'is_inverted': analysis_result['is_inverted'],
            'consecutive_errors_before': INVERSION_SYSTEM['consecutive_errors'],
            'inversion_mode': analysis_result['inversion_mode'],
            'sequence_position': len(learning_engine.sequence_memory),
            'confidence_source': 'advanced_learning' if analysis_result['learning_applied'] else 'technical',
            'learning_weight': 1.2 if analysis_result['q_learning_used'] else 1.0,
            'technical_factors': {
                'adjustments': analysis_result['adjustments'],
                'win_rate': win_rate,
                'prices': prices,
                'inversion_applied': analysis_result['is_inverted'],
                'q_learning_used': analysis_result['q_learning_used'],
                'learning_applied': analysis_result['learning_applied'],
                'advanced_features': {
                    'q_learning': LEARNING_CONFIG['reinforcement_learning'],
                    'temporal_analysis': LEARNING_CONFIG['temporal_learning'],
                    'sequence_patterns': LEARNING_CONFIG['sequence_learning']
                }
            }
        }
        
        # Salvar sinal no banco de dados
        signal_id = db.save_signal(signal_data)
        
        # Atualizar memória de sequência
        learning_engine.update_sequence_memory(signal_data)
        
        # Preparar reasoning AVANÇADO
        reasoning = f"Análise AVANÇADA para {symbol} - Múltiplos sistemas de IA"
        if analysis_result['is_inverted']:
            reasoning += f" - INVERSÃO ADAPTATIVA ({analysis_result['original_direction']} → {analysis_result['final_direction']})"
        if analysis_result['q_learning_used']:
            reasoning += " - Q-LEARNING APLICADO"
        if analysis_result['learning_applied']:
            reasoning += f" - {len(analysis_result['adjustments'])} AJUSTES DE APRENDIZADO"
        
        # Status detalhado para retorno
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "signal_id": signal_id,
            "direction": analysis_result['final_direction'],
            "original_direction": analysis_result['original_direction'],
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_price": current_price,
            "strength": "muito forte" if confidence > 90 else "forte" if confidence > 85 else "moderado" if confidence > 75 else "fraco",
            "timeframe": "5m",
            
            "advanced_analysis": {
                "inverted": analysis_result['is_inverted'],
                "q_learning_used": analysis_result['q_learning_used'],
                "learning_applied": analysis_result['learning_applied'],
                "confidence_adjustments": analysis_result['adjustments'],
                "sequence_position": len(learning_engine.sequence_memory)
            },
            
            "inversion_status": {
                "current_mode": analysis_result['inversion_mode'],
                "consecutive_errors": inversion_status['consecutive_errors'],
                "adaptive_threshold": inversion_status['adaptive_threshold'],
                "errors_until_inversion": inversion_status['errors_until_inversion'],
                "total_inversions": inversion_status['total_inversions'],
                "adaptive_mode": inversion_status['adaptive_mode']
            },
            
            "learning_systems": {
                "reinforcement_learning": LEARNING_CONFIG['reinforcement_learning'],
                "temporal_analysis": LEARNING_CONFIG['temporal_learning'],
                "sequence_learning": LEARNING_CONFIG['sequence_learning'],
                "correlation_analysis": LEARNING_CONFIG['correlation_analysis'],
                "adjustments_applied": len(analysis_result['adjustments'])
            },
            
            "factors": {
                "technical_model": "Advanced Multi-Layer Learning + Adaptive Inversion",
                "volatility_factor": volatility,
                "historical_performance": win_rate,
                "signal_inversion": "ATIVO" if analysis_result['is_inverted'] else "INATIVO",
                "learning_adjustments": len(analysis_result['adjustments']),
                "inversion_mode": analysis_result['inversion_mode'],
                "q_learning_influence": "ATIVO" if analysis_result['q_learning_used'] else "INATIVO",
                "advanced_confidence": confidence
            },
            
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "Advanced AI with Q-Learning + Adaptive Inversion + Multi-Layer Learning"
        })
        
    except Exception as e:
        logger.error(f"Erro em signal avançado: {e}")
        return jsonify({"error": "Erro na geração de sinal avançado", "message": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def receive_feedback():
    """Feedback para SISTEMA DE APRENDIZADO AVANÇADO + Q-LEARNING"""
    if request.method == "OPTIONS":
        return '', 200
    
    try:
        data = request.get_json() or {}
        
        # Dados do feedback
        result = data.get("result", 0)  # 1 para win, 0 para loss
        direction = data.get("direction", "CALL")
        signal_id = data.get("signal_id")
        pnl = data.get("pnl", 0)
        
        # 🧠 SISTEMA DE APRENDIZADO AVANÇADO
        if signal_id:
            # Atualizar resultado no banco (inclui Q-Learning e temporal)
            db.update_signal_result(signal_id, result, pnl)
            
            # Atualizar memória de sequência
            if learning_engine.sequence_memory:
                learning_engine.sequence_memory[-1]['result'] = result
            
            logger.info(f"🧠 Feedback AVANÇADO integrado: Signal {signal_id} -> {'WIN' if result == 1 else 'LOSS'}")
        
        # 🔄 SISTEMA DE INVERSÃO ADAPTATIVA
        inversion_manager.handle_signal_result(result)
        
        # Atualizar stats simples (compatibilidade)
        performance_stats['total_trades'] += 1
        if result == 1:
            performance_stats['won_trades'] += 1
        
        accuracy = (performance_stats['won_trades'] / max(performance_stats['total_trades'], 1) * 100)
        
        # Trigger análise AVANÇADA de padrões
        if performance_stats['total_trades'] % 8 == 0:  # Mais frequente
            try:
                patterns = learning_engine.analyze_error_patterns()
                if patterns:
                    logger.info(f"🧠 Análise AVANÇADA triggered - {len(patterns)} padrões identificados")
                    # Log tipos de padrões encontrados
                    pattern_types = set(p.get('type', 'unknown') for p in patterns)
                    logger.info(f"   Tipos: {', '.join(pattern_types)}")
            except Exception as e:
                logger.error(f"Erro na análise avançada de padrões: {e}")
        
        # Status DETALHADO do sistema
        inversion_status = inversion_manager.get_inversion_status()
        
        # Obter estatísticas Q-Learning se disponível
        q_learning_info = {}
        if LEARNING_CONFIG['reinforcement_learning']:
            try:
                conn = sqlite3.connect(db.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM q_learning_states')
                q_states_count = cursor.fetchone()[0]
                conn.close()
                q_learning_info = {
                    "states_learned": q_states_count,
                    "active": True
                }
            except:
                q_learning_info = {"states_learned": 0, "active": False}
        
        return jsonify({
            "message": "Feedback recebido - Sistema de aprendizado AVANÇADO + inversão adaptativa ativo",
            "signal_id": signal_id,
            "result_recorded": result == 1,
            "total_trades": performance_stats['total_trades'],
            "accuracy": f"{accuracy:.1f}%",
            
            "advanced_learning": {
                "reinforcement_learning": LEARNING_CONFIG['reinforcement_learning'],
                "temporal_analysis": LEARNING_CONFIG['temporal_learning'],
                "sequence_learning": LEARNING_CONFIG['sequence_learning'],
                "correlation_analysis": LEARNING_CONFIG['correlation_analysis'],
                "q_learning_info": q_learning_info
            },
            
            "inversion_system": {
                "current_mode": inversion_status['current_mode'],
                "consecutive_errors": inversion_status['consecutive_errors'],
                "adaptive_threshold": inversion_status['adaptive_threshold'],
                "errors_until_inversion": inversion_status['errors_until_inversion'],
                "total_inversions": inversion_status['total_inversions'],
                "last_inversion": inversion_status['last_inversion'],
                "adaptive_mode": inversion_status['adaptive_mode']
            },
            
            "learning_status": {
                "patterns_analysis": "Avançado" if LEARNING_CONFIG['learning_enabled'] else "Inativo",
                "sequence_memory_size": len(learning_engine.sequence_memory),
                "learning_systems": [
                    system for system, active in {
                        "Q-Learning": LEARNING_CONFIG['reinforcement_learning'],
                        "Temporal": LEARNING_CONFIG['temporal_learning'],
                        "Sequence": LEARNING_CONFIG['sequence_learning'],
                        "Correlation": LEARNING_CONFIG['correlation_analysis']
                    }.items() if active
                ]
            },
            
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "Advanced Learning System with Q-Learning + Adaptive Inversion"
        })
        
    except Exception as e:
        logger.error(f"Erro em feedback avançado: {e}")
        return jsonify({"error": "Erro no feedback avançado", "message": str(e)}), 500

# Outros endpoints atualizados para versão avançada...
@app.route("/learning-stats", methods=["GET"])
def get_learning_stats():
    """Obter estatísticas AVANÇADAS do sistema de aprendizado"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        # Estatísticas recentes
        recent_data = db.get_recent_performance(150)
        
        total_signals = len(recent_data)
        won_signals = sum(1 for signal in recent_data if signal[10] == 1)
        accuracy = (won_signals / total_signals * 100) if total_signals > 0 else 0
        
        # Análise de tendência
        if total_signals >= 60:
            recent_30 = recent_data[:30]
            previous_30 = recent_data[30:60]
            recent_accuracy = sum(1 for s in recent_30 if s[10] == 1) / 30 * 100
            previous_accuracy = sum(1 for s in previous_30 if s[10] == 1) / 30 * 100
            accuracy_trend = recent_accuracy - previous_accuracy
        else:
            accuracy_trend = 0
        
        # Estatísticas Q-Learning
        q_learning_stats = {}
        if LEARNING_CONFIG['reinforcement_learning']:
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*), AVG(visits_count), AVG(average_reward) FROM q_learning_states')
            q_stats = cursor.fetchone()
            
            cursor.execute('''
                SELECT state_hash, visits_count, average_reward 
                FROM q_learning_states 
                ORDER BY visits_count DESC LIMIT 5
            ''')
            top_states = cursor.fetchall()
            
            conn.close()
            
            q_learning_stats = {
                "total_states": q_stats[0] if q_stats else 0,
                "average_visits": round(q_stats[1], 2) if q_stats and q_stats[1] else 0,
                "average_reward": round(q_stats[2], 3) if q_stats and q_stats[2] else 0,
                "top_learned_states": [
                    {
                        "state": state[0],
                        "visits": state[1],
                        "avg_reward": round(state[2], 3)
                    } for state in top_states
                ]
            }
        
        # Estatísticas temporais
        temporal_stats = {}
        if LEARNING_CONFIG['temporal_learning']:
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT hour_of_day, AVG(success_rate), COUNT(*) 
                FROM temporal_patterns 
                GROUP BY hour_of_day 
                HAVING COUNT(*) >= 3
                ORDER BY AVG(success_rate) DESC
            ''')
            
            hourly_performance = cursor.fetchall()
            
            cursor.execute('''
                SELECT day_of_week, AVG(success_rate), COUNT(*) 
                FROM temporal_patterns 
                GROUP BY day_of_week 
                HAVING COUNT(*) >= 3
                ORDER BY AVG(success_rate) DESC
            ''')
            
            daily_performance = cursor.fetchall()
            conn.close()
            
            temporal_stats = {
                "best_hours": [
                    {
                        "hour": hour[0],
                        "success_rate": round(hour[1] * 100, 1),
                        "sample_size": hour[2]
                    } for hour in hourly_performance[:5]
                ],
                "best_days": [
                    {
                        "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day[0]],
                        "success_rate": round(day[1] * 100, 1),
                        "sample_size": day[2]
                    } for day in daily_performance[:5]
                ]
            }
        
        # Padrões de erro identificados
        error_patterns = db.get_error_patterns()
        
        # Parâmetros adaptativos
        adaptive_params = {
            'risk_factor': db.get_adaptive_parameter('risk_factor', 1.0),
            'aggression_factor': db.get_adaptive_parameter('aggression_factor', 1.0),
            'duration_factor': db.get_adaptive_parameter('duration_factor', 1.0)
        }
        
        # Status de inversão
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "advanced_learning_enabled": LEARNING_CONFIG['learning_enabled'],
            "total_samples": total_signals,
            "current_accuracy": round(accuracy, 1),
            "accuracy_trend": round(accuracy_trend, 1),
            
            "learning_systems": {
                "q_learning": {
                    "enabled": LEARNING_CONFIG['reinforcement_learning'],
                    "stats": q_learning_stats
                },
                "temporal_analysis": {
                    "enabled": LEARNING_CONFIG['temporal_learning'],
                    "stats": temporal_stats
                },
                "sequence_learning": {
                    "enabled": LEARNING_CONFIG['sequence_learning'],
                    "current_sequence_length": len(learning_engine.sequence_memory)
                },
                "correlation_analysis": {
                    "enabled": LEARNING_CONFIG['correlation_analysis']
                }
            },
            
            "error_patterns_found": len(error_patterns),
            "adaptive_parameters": adaptive_params,
            "inversion_system": inversion_status,
            
            "recent_patterns": [
                {
                    "type": pattern[1],
                    "conditions": json.loads(pattern[2]),
                    "error_rate": pattern[3],
                    "occurrences": pattern[4]
                } for pattern in error_patterns[:8]
            ],
            
            "performance_metrics": {
                "total_signals": total_signals,
                "won_signals": won_signals,
                "accuracy": accuracy,
                "accuracy_trend": accuracy_trend
            },
            
            "learning_config": LEARNING_CONFIG,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em learning-stats avançado: {e}")
        return jsonify({"error": "Erro ao obter estatísticas avançadas", "message": str(e)}), 500

if __name__ == "__main__":
    # Configuração para Render
    port = int(os.environ.get('PORT', 5000))
    logger.info("🚀 Iniciando IA Trading Bot API - SISTEMA DE APRENDIZADO AVANÇADO")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    logger.info("🧠 Sistema de Aprendizado AVANÇADO: ATIVADO")
    logger.info(f"🤖 Q-Learning: {'ATIVADO' if LEARNING_CONFIG['reinforcement_learning'] else 'DESATIVADO'}")
    logger.info(f"⏰ Análise Temporal: {'ATIVADA' if LEARNING_CONFIG['temporal_learning'] else 'DESATIVADA'}")
    logger.info(f"📈 Aprendizado de Sequências: {'ATIVADO' if LEARNING_CONFIG['sequence_learning'] else 'DESATIVADO'}")
    logger.info(f"🔗 Análise de Correlação: {'ATIVADA' if LEARNING_CONFIG['correlation_analysis'] else 'DESATIVADA'}")
    logger.info("🔄 Sistema de Inversão ADAPTATIVA: ATIVADO")
    logger.info(f"📊 Banco de dados: {DB_PATH}")
    logger.info("🐍 Pure Python - Compatível com Python 3.13")
    logger.info(f"⚙️ Threshold adaptativo de inversão: ATIVO")
    app.run(host="0.0.0.0", port=port, debug=False)
