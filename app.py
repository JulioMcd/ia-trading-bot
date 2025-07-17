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
    'adaptive_threshold': True,
    'performance_weight': 0.7
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
    'reinforcement_learning': True,
    'temporal_learning': True,
    'sequence_learning': True,
    'correlation_analysis': True,
    'dynamic_weighting': True
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
                hour_of_day INTEGER,
                day_of_week INTEGER,
                market_session TEXT,
                sequence_position INTEGER DEFAULT 0,
                confidence_source TEXT,
                learning_weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                feedback_received_at TEXT
            )
        ''')
        
        # Q-Learning para aprendizado por reforço
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
        
        # Padrões de sequência
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sequence_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_hash TEXT NOT NULL UNIQUE,
                sequence_data TEXT NOT NULL,
                pattern_length INTEGER NOT NULL,
                success_rate REAL DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                total_reward REAL DEFAULT 0.0,
                confidence_multiplier REAL DEFAULT 1.0,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Análise temporal
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
        
        # Correlações entre variáveis
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
        
        # Histórico de inversões
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inversion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                from_mode TEXT NOT NULL,
                to_mode TEXT NOT NULL,
                consecutive_errors INTEGER NOT NULL,
                trigger_reason TEXT,
                total_inversions_so_far INTEGER DEFAULT 0,
                performance_before REAL DEFAULT 0.0,
                adaptive_threshold INTEGER DEFAULT 3,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Padrões de erro
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                conditions TEXT NOT NULL,
                error_rate REAL NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence_adjustment REAL DEFAULT 0,
                severity_score REAL DEFAULT 0.0,
                pattern_stability REAL DEFAULT 0.0,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Parâmetros adaptativos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT NOT NULL UNIQUE,
                parameter_value REAL NOT NULL,
                learning_rate REAL DEFAULT 0.1,
                momentum REAL DEFAULT 0.0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                update_reason TEXT,
                performance_impact REAL DEFAULT 0.0
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
            epsilon = max(0.05, 0.3 / (1 + visits * 0.1))
            
            if random.random() < epsilon:
                return random.choice(['CALL', 'PUT']), 0.5
            else:
                if call_value > put_value:
                    return 'CALL', min(0.95, 0.5 + abs(call_value - put_value) * 0.1)
                else:
                    return 'PUT', min(0.95, 0.5 + abs(call_value - put_value) * 0.1)
        else:
            return random.choice(['CALL', 'PUT']), 0.5
    
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
        
        return result if result and result[1] >= 5 else (0.5, 0)
    
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
    
    def save_inversion_event(self, from_mode, to_mode, consecutive_errors, reason, performance_before=0.0, adaptive_threshold=3):
        """Salvar evento de inversão"""
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
                state_hash = f"{symbol}_{direction}_{hour}_{volatility//10 if volatility else 5}"
                state_desc = f"Symbol:{symbol}, Direction:{direction}, Hour:{hour}, Vol:{volatility//10*10 if volatility else 50}"
                reward = 1.0 if result == 1 else -0.5
                self.save_q_learning_state(state_hash, state_desc, direction, reward)
        
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
        
    def get_inversion_history(self, limit=20):
        """Obter histórico de inversões"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM inversion_history 
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        
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
            return min(5, INVERSION_SYSTEM['max_errors'] + 1)
        elif accuracy < 0.35:
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
        self.sequence_memory = deque(maxlen=10)
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
        """Analisar padrões de erro nos dados recentes"""
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
        won_signals = sum(1 for signal in recent_data if signal[10] == 1)  # ajustado coluna
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
    
    def update_sequence_memory(self, signal_data):
        """Atualizar memória de sequência"""
        self.sequence_memory.append({
            'direction': signal_data.get('direction'),
            'confidence': signal_data.get('confidence'),
            'volatility': signal_data.get('volatility'),
            'timestamp': signal_data.get('timestamp'),
            'result': signal_data.get('result')  # Will be updated later
        })

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
                    logger.info(f"   - Accuracy: {metrics['accuracy']:.1f}%")
                
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
        
        "endpoints": {
            "signal": "POST /signal - Sinais com aprendizado avançado + inversão adaptativa",
            "advanced-signal": "POST /advanced-signal - Sinais com todas as funcionalidades",
            "analyze": "POST /analyze - Análise de mercado com múltiplas técnicas de IA",
            "risk": "POST /risk - Avaliação de risco adaptativa",
            "optimal-duration": "POST /optimal-duration - Duração otimizada pela IA",
            "management": "POST /management - Gerenciamento automático",
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

@app.route("/signal", methods=["POST", "OPTIONS"])
@app.route("/advanced-signal", methods=["POST", "OPTIONS"])
@app.route("/trading-signal", methods=["POST", "OPTIONS"])
@app.route("/get-signal", methods=["POST", "OPTIONS"])
@app.route("/smart-signal", methods=["POST", "OPTIONS"])
@app.route("/evolutionary-signal", methods=["POST", "OPTIONS"])
@app.route("/prediction", methods=["POST", "OPTIONS"])
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

@app.route("/analyze", methods=["POST", "OPTIONS"])
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
        
        # Análise técnica com inversão
        analysis_result = analyze_technical_pattern(prices, learning_data)
        
        # Análise adicional
        symbol = data.get("symbol", "R_50")
        confidence = analysis_result['confidence']
        
        # Determinar tendência baseada na direção final
        if confidence > 80:
            trend = "bullish" if analysis_result['final_direction'] == "CALL" else "bearish"
        else:
            trend = "neutral"
        
        # Status de inversão
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "volatility": round(volatility, 1),
            "direction": analysis_result['final_direction'],
            "original_direction": analysis_result['original_direction'],
            "inverted": analysis_result['is_inverted'],
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "confidence_adjustments": analysis_result['adjustments'],
            "inversion_status": {
                "current_mode": inversion_status['current_mode'],
                "consecutive_errors": inversion_status['consecutive_errors'],
                "errors_until_inversion": inversion_status['errors_until_inversion']
            },
            "message": f"Análise ADAPTATIVA para {symbol}: {analysis_result['final_direction']}" + (" (INVERTIDO)" if analysis_result['is_inverted'] else ""),
            "recommendation": f"{analysis_result['final_direction']} recomendado" if confidence > 75 else "Aguardar melhor oportunidade",
            "factors": {
                "technical_analysis": analysis_result['final_direction'],
                "market_volatility": round(volatility, 1),
                "confidence_level": confidence,
                "inversion_mode": analysis_result['inversion_mode'],
                "learning_adjustments": len(analysis_result['adjustments']),
                "signal_inverted": analysis_result['is_inverted']
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Pure Python com Sistema de Inversão Automática + Aprendizado"
        })
        
    except Exception as e:
        logger.error(f"Erro em analyze: {e}")
        return jsonify({"error": "Erro na análise", "message": str(e)}), 500

@app.route("/risk", methods=["POST", "OPTIONS"])
def assess_risk():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        
        # Calcular risco básico
        martingale_level = data.get("martingaleLevel", 0)
        today_pnl = data.get("todayPnL", 0)
        win_rate = data.get("winRate", 50)
        total_trades = data.get("totalTrades", 0)
        
        risk_score = 0
        risk_level = "low"
        
        # Análise Martingale
        if martingale_level > 5:
            risk_score += 40
            risk_level = "high"
        elif martingale_level > 2:
            risk_score += 20
            risk_level = "medium"
        
        # Análise P&L
        if today_pnl < -100:
            risk_score += 25
            risk_level = "high"
        elif today_pnl < -50:
            risk_score += 10
            risk_level = "medium" if risk_level == "low" else risk_level
        
        # Análise Win Rate
        if win_rate < 30:
            risk_score += 20
            risk_level = "high"
        elif win_rate < 45:
            risk_score += 10
        
        # Obter parâmetros adaptativos para mostrar no retorno
        risk_factor = db.get_adaptive_parameter('risk_factor', 1.0)
        inversion_status = inversion_manager.get_inversion_status()
        
        # Mensagens baseadas no nível de risco
        messages = {
            "high": "ALTO RISCO - Intervenção necessária (Sistema de Inversão ativo)",
            "medium": "Risco moderado - Cautela recomendada (Monitoramento de inversão)", 
            "low": "Risco controlado (Sistema adaptativo + inversão funcionando)"
        }
        
        recommendations = {
            "high": "Pare imediatamente e revise estratégia - verifique sistema de inversão",
            "medium": "Reduza frequency e monitore inversões de perto",
            "low": "Continue operando com disciplina - sistema de inversão ativo"
        }
        
        return jsonify({
            "level": risk_level,
            "score": min(risk_score, 100),
            "message": messages[risk_level],
            "recommendation": recommendations[risk_level],
            "adaptive_risk_factor": risk_factor,
            "inversion_system": inversion_status,
            "factors": {
                "martingale_level": martingale_level,
                "today_pnl": today_pnl,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "risk_factor_applied": risk_factor,
                "consecutive_errors": inversion_status['consecutive_errors'],
                "inversion_mode": inversion_status['current_mode']
            },
            "severity": "critical" if risk_level == "high" else "warning" if risk_level == "medium" else "normal",
            "signal_mode": f"{inversion_status['current_mode'].upper()} + LEARNING",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Pure Python com Sistema de Inversão Automática + Aprendizado"
        })
        
    except Exception as e:
        logger.error(f"Erro em risk: {e}")
        return jsonify({"error": "Erro na avaliação de risco", "message": str(e)}), 500

@app.route("/optimal-duration", methods=["POST", "OPTIONS"])
def get_optimal_duration():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
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
        
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "type": duration_type,
            "duration_type": "ticks" if duration_type == "t" else "minutes",
            "value": duration,
            "duration": duration,
            "confidence": round(confidence, 1),
            "reasoning": f"Análise adaptativa para {symbol}: {duration}{duration_type} (fator: {duration_factor:.2f})",
            "signal_mode": f"{inversion_status['current_mode'].upper()} + LEARNING",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "inversion_system": inversion_status,
            "adaptive_optimization": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Pure Python com Sistema de Inversão Automática + Aprendizado"
        })
        
    except Exception as e:
        logger.error(f"Erro em optimal-duration: {e}")
        return jsonify({"error": "Erro na otimização de duração", "message": str(e)}), 500

@app.route("/management", methods=["POST", "OPTIONS"])
def position_management():
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        
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
        
        # Pausar se muitas inversões recentes
        if INVERSION_SYSTEM['consecutive_errors'] >= INVERSION_SYSTEM['max_errors'] - 1:
            should_pause = True
            action = "pause"
            pause_duration = random.randint(30000, 60000)
        elif today_pnl < -pause_threshold_high or martingale_level > martingale_threshold:
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
            message = f"PAUSA RECOMENDADA - {pause_duration//1000}s - Alto risco (Sistema de Inversão ativo)"
        elif recommended_stake != current_stake:
            message = f"Stake adaptativo: ${current_stake:.2f} → ${recommended_stake:.2f}"
        else:
            message = "Continuar operação - Parâmetros adequados"
        
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "action": action,
            "recommendedStake": round(recommended_stake, 2),
            "shouldPause": should_pause,
            "pauseDuration": pause_duration,
            "riskLevel": "high" if martingale_level > 5 else "medium" if today_pnl < -50 else "low",
            "message": message,
            "reasoning": "Sistema adaptativo + inversão ativo",
            "adaptive_factors": {
                "risk_factor": risk_factor,
                "aggression_factor": aggression_factor
            },
            "inversion_status": inversion_status,
            "signal_mode": f"{inversion_status['current_mode'].upper()} + LEARNING",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Pure Python com Sistema de Inversão Automática + Aprendizado"
        })
        
    except Exception as e:
        logger.error(f"Erro em management: {e}")
        return jsonify({"error": "Erro no gerenciamento", "message": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def receive_feedback():
    """Endpoint para receber feedback - SISTEMA DE APRENDIZADO + INVERSÃO"""
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
            logger.info(f"🧠 Feedback integrado: Signal {signal_id} -> {'WIN' if result == 1 else 'LOSS'}")
        
        # 🔄 SISTEMA DE INVERSÃO AUTOMÁTICA
        inversion_manager.handle_signal_result(result)
        
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
        
        # Status atual do sistema de inversão
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "message": "Feedback recebido - Sistema de inversão automática + aprendizado ativo",
            "signal_id": signal_id,
            "result_recorded": result == 1,
            "total_trades": performance_stats['total_trades'],
            "accuracy": f"{accuracy:.1f}%",
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "inversion_system": {
                "current_mode": inversion_status['current_mode'],
                "consecutive_errors": inversion_status['consecutive_errors'],
                "errors_until_inversion": inversion_status['errors_until_inversion'],
                "total_inversions": inversion_status['total_inversions'],
                "last_inversion": inversion_status['last_inversion']
            },
            "patterns_analysis": "Ativo" if LEARNING_CONFIG['learning_enabled'] else "Inativo",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "Sistema de Inversão Automática + Aprendizado Pure Python"
        })
        
    except Exception as e:
        logger.error(f"Erro em feedback: {e}")
        return jsonify({"error": "Erro no feedback", "message": str(e)}), 500

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
            
            "learning_systems": {
                "q_learning": {
                    "enabled": LEARNING_CONFIG['reinforcement_learning'],
                    "stats": q_learning_stats
                },
                "temporal_analysis": {
                    "enabled": LEARNING_CONFIG['temporal_learning']
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
                "accuracy": accuracy
            },
            
            "learning_config": LEARNING_CONFIG,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em learning-stats avançado: {e}")
        return jsonify({"error": "Erro ao obter estatísticas avançadas", "message": str(e)}), 500

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

@app.route("/inversion-status", methods=["GET"])
def get_inversion_status():
    """Obter status detalhado do sistema de inversão"""
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        inversion_status = inversion_manager.get_inversion_status()
        inversion_history = db.get_inversion_history(10)
        
        return jsonify({
            "inversion_system": inversion_status,
            "recent_inversions": [
                {
                    "timestamp": inv[1],
                    "from_mode": inv[2],
                    "to_mode": inv[3],
                    "consecutive_errors": inv[4],
                    "reason": inv[5],
                    "total_inversions_so_far": inv[6]
                } for inv in inversion_history
            ],
            "inversion_rules": {
                "max_errors_before_inversion": INVERSION_SYSTEM['max_errors'],
                "auto_reset_on_win": True,
                "alternates_between_modes": True,
                "adaptive_threshold": True
            },
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em inversion-status: {e}")
        return jsonify({"error": "Erro ao obter status de inversão", "message": str(e)}), 500

# Middleware de erro global
@app.errorhandler(404)
def not_found(error):
    inversion_status = inversion_manager.get_inversion_status()
    return jsonify({
        "error": "Endpoint não encontrado",
        "available_endpoints": ["/", "/signal", "/advanced-signal", "/analyze", "/risk", "/optimal-duration", "/management", "/feedback", "/learning-stats", "/inversion-status", "/q-learning-stats"],
        "signal_mode": f"{inversion_status['current_mode'].upper()} + LEARNING",
        "learning_active": LEARNING_CONFIG['learning_enabled'],
        "inversion_system": inversion_status,
        "timestamp": datetime.datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erro interno: {error}")
    return jsonify({
        "error": "Erro interno do servidor",
        "message": "Entre em contato com o suporte",
        "learning_system": "Ativo" if LEARNING_CONFIG['learning_enabled'] else "Inativo",
        "inversion_system": "Ativo" if INVERSION_SYSTEM['active'] else "Inativo",
        "timestamp": datetime.datetime.now().isoformat()
    }), 500

if __name__ == "__main__":
    # Configuração para Render
    port = int(os.environ.get('PORT', 5000))
    logger.info("🚀 Iniciando IA Trading Bot API - SISTEMA DE APRENDIZADO AVANÇADO")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    logger.info("🧠 Sistema de Aprendizado AVANÇADO: ATIVADO")
    logger.info(f"🤖 Q-Learning: {'ATIVADO' if LEARNING_CONFIG['reinforcement_learning'] else 'INATIVO'}")
    logger.info(f"⏰ Análise Temporal: {'ATIVADA' if LEARNING_CONFIG['temporal_learning'] else 'INATIVA'}")
    logger.info(f"📈 Aprendizado de Sequências: {'ATIVADO' if LEARNING_CONFIG['sequence_learning'] else 'INATIVO'}")
    logger.info(f"🔗 Análise de Correlação: {'ATIVADA' if LEARNING_CONFIG['correlation_analysis'] else 'INATIVA'}")
    logger.info(f"🔄 Sistema de Inversão ADAPTATIVA: {'ATIVADO' if INVERSION_SYSTEM['active'] else 'INATIVO'}")
    logger.info(f"📊 Banco de dados: {DB_PATH}")
    logger.info("🐍 Pure Python - Compatível com Python 3.13")
    logger.info(f"⚙️ Threshold adaptativo de inversão: {'ATIVO' if INVERSION_SYSTEM['adaptive_threshold'] else 'FIXO'}")
    
    # Verificar conectividade do banco de dados
    try:
        test_data = db.get_recent_performance(1)
        logger.info("✅ Conexão com banco de dados OK")
    except Exception as e:
        logger.warning(f"⚠️ Aviso no banco de dados: {e}")
    
    # Logs de configuração final
    logger.info("=" * 60)
    logger.info("🎯 RECURSOS AVANÇADOS CARREGADOS:")
    logger.info(f"   - Q-Learning com estados adaptativos")
    logger.info(f"   - Análise temporal por horário/dia")
    logger.info(f"   - Padrões de sequência inteligentes")
    logger.info(f"   - Correlações automáticas")
    logger.info(f"   - Inversão adaptativa com threshold dinâmico")
    logger.info(f"   - Parâmetros adaptativos de risco")
    logger.info(f"   - Sistema de aprendizado por reforço")
    logger.info("=" * 60)
    
    # Inicializar Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
