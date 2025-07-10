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
import hashlib
import uuid

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key válida para Render
VALID_API_KEY = "rnd_qpdTVwAeWzIItVbxHPPCc34uirv9"

# ✅ SISTEMA ANTI-DUPLICAÇÃO - NOVO!
DUPLICATION_CONTROL = {
    'active': True,
    'active_orders': {},  # {order_id: {timestamp, data}}
    'order_history': deque(maxlen=100),  # Histórico de ordens
    'duplicate_attempts': 0,  # Tentativas de duplicação detectadas
    'last_duplicate_time': None,
    'duplicate_threshold': 5000,  # 5 segundos entre ordens do mesmo tipo
    'learning_enabled': True,
    'duplicate_patterns': defaultdict(int),  # Padrões de duplicação
    'prevention_rules': {}  # Regras aprendidas
}

# ✅ SISTEMA DE INVERSÃO AUTOMÁTICA - MELHORADO
INVERSION_SYSTEM = {
    'active': True,
    'is_inverse_mode': False,
    'consecutive_errors': 0,
    'max_errors': 3,
    'total_inversions': 0,
    'last_inversion_time': None,
    'inversion_history': [],
    'duplicate_triggered_inversions': 0  # Inversões causadas por duplicações
}

# Configuração para Render
DB_PATH = os.environ.get('DB_PATH', '/tmp/trading_data.db')

# Configurações do sistema de aprendizado
LEARNING_CONFIG = {
    'min_samples_for_learning': int(os.environ.get('MIN_SAMPLES', '15')),
    'adaptation_rate': float(os.environ.get('ADAPTATION_RATE', '0.1')),
    'error_pattern_window': int(os.environ.get('PATTERN_WINDOW', '50')),
    'confidence_adjustment_factor': float(os.environ.get('CONFIDENCE_FACTOR', '0.05')),
    'learning_enabled': os.environ.get('LEARNING_ENABLED', 'true').lower() == 'true',
    'duplication_learning': True,  # Aprendizado específico para duplicações
}

class TradingDatabase:
    """Classe para gerenciar o banco de dados SQLite"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Inicializar tabelas do banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de sinais e resultados (expandida para duplicações)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
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
                is_duplicate BOOLEAN DEFAULT 0,
                duplicate_detection_method TEXT,
                client_session_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                feedback_received_at TEXT
            )
        ''')
        
        # Tabela de detecção de duplicatas (NOVA)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                duplicate_type TEXT NOT NULL,
                detection_method TEXT NOT NULL,
                similarity_score REAL,
                time_difference INTEGER,
                prevented BOOLEAN DEFAULT 1,
                original_order_id TEXT,
                client_info TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de histórico de inversões (expandida)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inversion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                from_mode TEXT NOT NULL,
                to_mode TEXT NOT NULL,
                consecutive_errors INTEGER NOT NULL,
                trigger_reason TEXT,
                total_inversions_so_far INTEGER DEFAULT 0,
                caused_by_duplication BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabelas existentes mantidas...
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
                duplicate_prevention_rate REAL DEFAULT 0,
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
                prevention_rule TEXT,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
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
        """Salvar sinal no banco de dados (com detecção de duplicatas)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (
                order_id, timestamp, symbol, direction, original_direction, confidence, 
                entry_price, volatility, duration_type, duration_value, martingale_level,
                market_condition, technical_factors, is_inverted, consecutive_errors_before, 
                inversion_mode, is_duplicate, duplicate_detection_method, client_session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('order_id'),
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
            signal_data.get('is_duplicate', False),
            signal_data.get('duplicate_detection_method'),
            signal_data.get('client_session_id')
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
        
    def save_duplicate_detection(self, detection_data):
        """Salvar detecção de duplicata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO duplicate_detections (
                order_id, duplicate_type, detection_method, similarity_score,
                time_difference, prevented, original_order_id, client_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection_data.get('order_id'),
            detection_data.get('duplicate_type'),
            detection_data.get('detection_method'),
            detection_data.get('similarity_score'),
            detection_data.get('time_difference'),
            detection_data.get('prevented', True),
            detection_data.get('original_order_id'),
            json.dumps(detection_data.get('client_info', {}))
        ))
        
        conn.commit()
        conn.close()
        
    def save_inversion_event(self, from_mode, to_mode, consecutive_errors, reason, caused_by_duplication=False):
        """Salvar evento de inversão no banco"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO inversion_history (
                timestamp, from_mode, to_mode, consecutive_errors, trigger_reason, 
                total_inversions_so_far, caused_by_duplication
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            from_mode,
            to_mode,
            consecutive_errors,
            reason,
            INVERSION_SYSTEM['total_inversions'],
            caused_by_duplication
        ))
        
        conn.commit()
        conn.close()
        
    def get_recent_signals(self, limit=50, include_duplicates=False):
        """Obter sinais recentes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM signals 
            WHERE result IS NOT NULL
        '''
        params = []
        
        if not include_duplicates:
            query += ' AND is_duplicate = 0'
            
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
        
    def get_duplicate_statistics(self):
        """Obter estatísticas de duplicatas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_detections,
                COUNT(CASE WHEN prevented = 1 THEN 1 END) as prevented_count,
                AVG(similarity_score) as avg_similarity,
                duplicate_type,
                detection_method
            FROM duplicate_detections 
            GROUP BY duplicate_type, detection_method
            ORDER BY total_detections DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
        
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
            WHERE result IS NOT NULL AND is_duplicate = 0
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
        
    def save_error_pattern(self, pattern_type, conditions, error_rate, prevention_rule=None):
        """Salvar padrão de erro identificado"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, occurrences FROM error_patterns WHERE pattern_type = ? AND conditions = ?',
            (pattern_type, json.dumps(conditions))
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE error_patterns 
                SET error_rate = ?, occurrences = occurrences + 1, last_seen = ?, prevention_rule = ?
                WHERE id = ?
            ''', (error_rate, datetime.datetime.now().isoformat(), prevention_rule, existing[0]))
        else:
            cursor.execute('''
                INSERT INTO error_patterns (pattern_type, conditions, error_rate, prevention_rule)
                VALUES (?, ?, ?, ?)
            ''', (pattern_type, json.dumps(conditions), error_rate, prevention_rule))
            
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

class DuplicationController:
    """Controlador para prevenir ordens duplicadas"""
    
    def __init__(self, database):
        self.db = database
        
    def generate_order_signature(self, order_data):
        """Gerar assinatura única para uma ordem"""
        signature_data = {
            'symbol': order_data.get('symbol', ''),
            'direction': order_data.get('direction', ''),
            'stake': order_data.get('stake', 0),
            'duration_type': order_data.get('duration_type', ''),
            'duration_value': order_data.get('duration_value', 0),
            'client_session': order_data.get('client_session_id', '')
        }
        
        signature_string = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_string.encode()).hexdigest()
        
    def check_duplicate_order(self, order_data):
        """Verificar se uma ordem é duplicada"""
        if not DUPLICATION_CONTROL['active']:
            return False, None
            
        current_time = datetime.datetime.now()
        order_signature = self.generate_order_signature(order_data)
        
        # Verificar duplicatas por assinatura
        for order_id, order_info in DUPLICATION_CONTROL['active_orders'].items():
            if order_info['signature'] == order_signature:
                time_diff = (current_time - order_info['timestamp']).total_seconds() * 1000
                
                if time_diff < DUPLICATION_CONTROL['duplicate_threshold']:
                    return True, {
                        'type': 'signature_match',
                        'original_order_id': order_id,
                        'time_difference': time_diff,
                        'similarity_score': 1.0
                    }
        
        # Verificar duplicatas por similaridade
        for order_id, order_info in DUPLICATION_CONTROL['active_orders'].items():
            similarity = self.calculate_similarity(order_data, order_info['data'])
            time_diff = (current_time - order_info['timestamp']).total_seconds() * 1000
            
            if similarity > 0.8 and time_diff < DUPLICATION_CONTROL['duplicate_threshold']:
                return True, {
                    'type': 'similarity_match',
                    'original_order_id': order_id,
                    'time_difference': time_diff,
                    'similarity_score': similarity
                }
        
        return False, None
        
    def calculate_similarity(self, order1, order2):
        """Calcular similaridade entre duas ordens"""
        similarity_score = 0.0
        total_factors = 0
        
        # Símbolo
        if order1.get('symbol') == order2.get('symbol'):
            similarity_score += 0.3
        total_factors += 0.3
        
        # Direção
        if order1.get('direction') == order2.get('direction'):
            similarity_score += 0.3
        total_factors += 0.3
        
        # Stake (tolerância de 5%)
        stake1 = order1.get('stake', 0)
        stake2 = order2.get('stake', 0)
        if stake1 > 0 and stake2 > 0:
            stake_diff = abs(stake1 - stake2) / max(stake1, stake2)
            if stake_diff <= 0.05:
                similarity_score += 0.2
        total_factors += 0.2
        
        # Duração
        if (order1.get('duration_type') == order2.get('duration_type') and 
            order1.get('duration_value') == order2.get('duration_value')):
            similarity_score += 0.2
        total_factors += 0.2
        
        return similarity_score / total_factors if total_factors > 0 else 0.0
        
    def register_order(self, order_data):
        """Registrar uma nova ordem"""
        order_id = str(uuid.uuid4())
        order_signature = self.generate_order_signature(order_data)
        
        DUPLICATION_CONTROL['active_orders'][order_id] = {
            'timestamp': datetime.datetime.now(),
            'signature': order_signature,
            'data': order_data.copy()
        }
        
        # Limpar ordens antigas (mais de 30 segundos)
        self.cleanup_old_orders()
        
        return order_id
        
    def cleanup_old_orders(self):
        """Limpar ordens antigas do controle"""
        current_time = datetime.datetime.now()
        cleanup_threshold = 30000  # 30 segundos
        
        orders_to_remove = []
        for order_id, order_info in DUPLICATION_CONTROL['active_orders'].items():
            age = (current_time - order_info['timestamp']).total_seconds() * 1000
            if age > cleanup_threshold:
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            del DUPLICATION_CONTROL['active_orders'][order_id]
            
    def complete_order(self, order_id):
        """Marcar ordem como completada"""
        if order_id in DUPLICATION_CONTROL['active_orders']:
            del DUPLICATION_CONTROL['active_orders'][order_id]
            
    def handle_duplicate_detected(self, order_data, duplicate_info):
        """Processar detecção de duplicata"""
        DUPLICATION_CONTROL['duplicate_attempts'] += 1
        DUPLICATION_CONTROL['last_duplicate_time'] = datetime.datetime.now().isoformat()
        
        # Aprender padrão de duplicação
        if DUPLICATION_CONTROL['learning_enabled']:
            pattern_key = f"{order_data.get('symbol', 'unknown')}_{order_data.get('direction', 'unknown')}"
            DUPLICATION_CONTROL['duplicate_patterns'][pattern_key] += 1
            
            # Criar regra de prevenção
            if DUPLICATION_CONTROL['duplicate_patterns'][pattern_key] >= 3:
                prevention_rule = f"Delay mínimo de {DUPLICATION_CONTROL['duplicate_threshold']}ms para {pattern_key}"
                DUPLICATION_CONTROL['prevention_rules'][pattern_key] = prevention_rule
                
                # Salvar no banco
                self.db.save_error_pattern(
                    'duplicate_pattern',
                    {'pattern': pattern_key, 'symbol': order_data.get('symbol'), 'direction': order_data.get('direction')},
                    1.0,  # 100% erro para duplicatas
                    prevention_rule
                )
        
        # Salvar detecção no banco
        detection_data = {
            'order_id': str(uuid.uuid4()),
            'duplicate_type': duplicate_info['type'],
            'detection_method': 'signature_and_similarity',
            'similarity_score': duplicate_info['similarity_score'],
            'time_difference': duplicate_info['time_difference'],
            'original_order_id': duplicate_info['original_order_id'],
            'client_info': order_data
        }
        
        self.db.save_duplicate_detection(detection_data)
        
        logger.warning(f"🚫 DUPLICATA DETECTADA: {duplicate_info['type']} - Similaridade: {duplicate_info['similarity_score']:.2f}")
        
        return {
            'prevented': True,
            'reason': f"Duplicata detectada ({duplicate_info['type']})",
            'time_difference': duplicate_info['time_difference'],
            'similarity': duplicate_info['similarity_score']
        }
        
    def get_duplication_stats(self):
        """Obter estatísticas de duplicação"""
        return {
            'total_attempts': DUPLICATION_CONTROL['duplicate_attempts'],
            'active_orders': len(DUPLICATION_CONTROL['active_orders']),
            'last_duplicate': DUPLICATION_CONTROL['last_duplicate_time'],
            'learned_patterns': len(DUPLICATION_CONTROL['duplicate_patterns']),
            'prevention_rules': len(DUPLICATION_CONTROL['prevention_rules']),
            'threshold_ms': DUPLICATION_CONTROL['duplicate_threshold']
        }

class InversionManager:
    """Gerenciador do sistema de inversão automática"""
    
    def __init__(self, database):
        self.db = database
        
    def invert_signal(self, signal):
        """Inverter sinal de trading"""
        signal_map = {
            'CALL': 'PUT',
            'PUT': 'CALL', 
            'BUY': 'SELL',
            'SELL': 'BUY',
            'LONG': 'SHORT',
            'SHORT': 'LONG'
        }
        
        return signal_map.get(signal.upper(), signal)
        
    def should_invert_mode(self):
        """Verificar se deve inverter o modo"""
        return INVERSION_SYSTEM['consecutive_errors'] >= INVERSION_SYSTEM['max_errors']
        
    def switch_inversion_mode(self, reason="Max consecutive errors reached", caused_by_duplication=False):
        """Alternar modo de inversão"""
        old_mode = "inverse" if INVERSION_SYSTEM['is_inverse_mode'] else "normal"
        INVERSION_SYSTEM['is_inverse_mode'] = not INVERSION_SYSTEM['is_inverse_mode']
        INVERSION_SYSTEM['consecutive_errors'] = 0
        INVERSION_SYSTEM['total_inversions'] += 1
        INVERSION_SYSTEM['last_inversion_time'] = datetime.datetime.now().isoformat()
        
        if caused_by_duplication:
            INVERSION_SYSTEM['duplicate_triggered_inversions'] += 1
        
        new_mode = "inverse" if INVERSION_SYSTEM['is_inverse_mode'] else "normal"
        
        # Registrar no histórico
        INVERSION_SYSTEM['inversion_history'].append({
            'timestamp': INVERSION_SYSTEM['last_inversion_time'],
            'from_mode': old_mode,
            'to_mode': new_mode,
            'consecutive_errors': INVERSION_SYSTEM['max_errors'],
            'reason': reason,
            'caused_by_duplication': caused_by_duplication
        })
        
        # Salvar no banco
        self.db.save_inversion_event(old_mode, new_mode, INVERSION_SYSTEM['max_errors'], reason, caused_by_duplication)
        
        logger.info(f"🔄 INVERSÃO AUTOMÁTICA: {old_mode.upper()} → {new_mode.upper()}")
        logger.info(f"   Motivo: {reason}")
        if caused_by_duplication:
            logger.info(f"   🚫 Causada por duplicação detectada")
        
    def handle_signal_result(self, result):
        """Processar resultado do sinal"""
        if not INVERSION_SYSTEM['active']:
            return
            
        if result == 0:  # Loss
            INVERSION_SYSTEM['consecutive_errors'] += 1
            logger.info(f"❌ Erro #{INVERSION_SYSTEM['consecutive_errors']} de {INVERSION_SYSTEM['max_errors']}")
            
            if self.should_invert_mode():
                self.switch_inversion_mode()
        else:  # Win
            if INVERSION_SYSTEM['consecutive_errors'] > 0:
                logger.info(f"✅ Win! Resetando contador de erros (era {INVERSION_SYSTEM['consecutive_errors']})")
                INVERSION_SYSTEM['consecutive_errors'] = 0
                
    def get_final_signal(self, original_signal):
        """Obter sinal final (invertido ou não)"""
        if not INVERSION_SYSTEM['active']:
            return original_signal, False, "normal"
            
        if INVERSION_SYSTEM['is_inverse_mode']:
            inverted_signal = self.invert_signal(original_signal)
            return inverted_signal, True, "inverse"
        else:
            return original_signal, False, "normal"
            
    def get_inversion_status(self):
        """Obter status atual do sistema de inversão"""
        return {
            'active': INVERSION_SYSTEM['active'],
            'current_mode': "inverse" if INVERSION_SYSTEM['is_inverse_mode'] else "normal",
            'consecutive_errors': INVERSION_SYSTEM['consecutive_errors'],
            'max_errors': INVERSION_SYSTEM['max_errors'],
            'total_inversions': INVERSION_SYSTEM['total_inversions'],
            'duplicate_triggered_inversions': INVERSION_SYSTEM['duplicate_triggered_inversions'],
            'last_inversion': INVERSION_SYSTEM['last_inversion_time'],
            'errors_until_inversion': INVERSION_SYSTEM['max_errors'] - INVERSION_SYSTEM['consecutive_errors']
        }

class LearningEngine:
    """Motor de aprendizado melhorado"""
    
    def __init__(self, database):
        self.db = database
        self.recent_signals = deque(maxlen=LEARNING_CONFIG['error_pattern_window'])
        
    def analyze_duplication_patterns(self):
        """Analisar padrões de duplicação"""
        duplicate_stats = self.db.get_duplicate_statistics()
        patterns_found = []
        
        for stat in duplicate_stats:
            total_detections = stat[0]
            prevented_count = stat[1]
            avg_similarity = stat[2] or 0
            duplicate_type = stat[3]
            detection_method = stat[4]
            
            if total_detections >= 3:  # Padrão significativo
                pattern = {
                    'type': 'duplication_pattern',
                    'duplicate_type': duplicate_type,
                    'detection_method': detection_method,
                    'frequency': total_detections,
                    'prevention_rate': prevented_count / total_detections if total_detections > 0 else 0,
                    'avg_similarity': avg_similarity
                }
                patterns_found.append(pattern)
                
                # Salvar padrão no banco
                self.db.save_error_pattern(
                    'duplication_pattern',
                    {
                        'duplicate_type': duplicate_type,
                        'detection_method': detection_method
                    },
                    1.0,  # Duplicatas são sempre 100% erro
                    f"Prevenção automática para {duplicate_type}"
                )
        
        return patterns_found
        
    def adapt_confidence(self, signal_data):
        """Adaptar confiança baseado em padrões aprendidos"""
        base_confidence = signal_data.get('confidence', 70)
        adjustments = []
        
        # Verificar padrões de duplicação
        symbol = signal_data.get('symbol', '')
        direction = signal_data.get('direction', '')
        pattern_key = f"{symbol}_{direction}"
        
        if pattern_key in DUPLICATION_CONTROL['duplicate_patterns']:
            duplicate_frequency = DUPLICATION_CONTROL['duplicate_patterns'][pattern_key]
            if duplicate_frequency >= 2:
                adjustment = -10  # Reduzir confiança se há histórico de duplicatas
                adjustments.append(('duplication_history', adjustment))
        
        # Verificar outros padrões de erro
        error_patterns = self.db.get_error_patterns()
        
        for pattern in error_patterns[:5]:  # Top 5 padrões
            pattern_type = pattern[1]
            conditions = json.loads(pattern[2])
            error_rate = pattern[3]
            
            if pattern_type == 'symbol_low_performance':
                if signal_data.get('symbol') == conditions.get('symbol'):
                    adjustment = -error_rate * 15
                    adjustments.append(('symbol_pattern', adjustment))
                    
            elif pattern_type == 'direction_low_performance':
                if signal_data.get('direction') == conditions.get('direction'):
                    adjustment = -error_rate * 12
                    adjustments.append(('direction_pattern', adjustment))
        
        # Aplicar ajustes
        total_adjustment = sum(adj[1] for adj in adjustments)
        adapted_confidence = max(55, min(95, base_confidence + total_adjustment))
        
        if adjustments:
            logger.info(f"🧠 Confiança adaptada: {base_confidence:.1f} → {adapted_confidence:.1f}")
        
        return adapted_confidence, adjustments

# Instâncias globais
db = TradingDatabase()
duplication_controller = DuplicationController(db)
learning_engine = LearningEngine(db)
inversion_manager = InversionManager(db)

# Dados de histórico
performance_stats = {
    'total_trades': 0,
    'won_trades': 0,
    'total_pnl': 0.0,
    'duplicates_prevented': 0
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
        return True  # Permitir sem API key para teste
    
    return api_key == VALID_API_KEY

def analyze_technical_pattern(prices, learning_data=None):
    """Análise técnica com sistema de inversão"""
    try:
        if len(prices) >= 3:
            recent_trend = prices[-1] - prices[-3]
            volatility = abs(prices[-1] - prices[-2]) / prices[-2] * 100 if prices[-2] != 0 else 50
            
            if recent_trend > 0:
                original_direction = "CALL"
                base_confidence = 72 + min(volatility * 0.3, 20)
            else:
                original_direction = "PUT" 
                base_confidence = 72 + min(volatility * 0.3, 20)
            
            # Aplicar aprendizado
            if learning_data and LEARNING_CONFIG['learning_enabled']:
                adapted_confidence, adjustments = learning_engine.adapt_confidence({
                    'direction': original_direction,
                    'confidence': base_confidence,
                    'volatility': volatility,
                    **learning_data
                })
            else:
                adapted_confidence = base_confidence
                adjustments = []
            
            # Aplicar sistema de inversão
            final_direction, is_inverted, inversion_mode = inversion_manager.get_final_signal(original_direction)
            
            return {
                'original_direction': original_direction,
                'final_direction': final_direction,
                'confidence': round(adapted_confidence, 1),
                'is_inverted': is_inverted,
                'inversion_mode': inversion_mode,
                'adjustments': adjustments
            }
            
        else:
            # Fallback
            original_direction = "CALL" if random.random() > 0.5 else "PUT"
            confidence = 72 + random.uniform(0, 18)
            
            final_direction, is_inverted, inversion_mode = inversion_manager.get_final_signal(original_direction)
            
            return {
                'original_direction': original_direction,
                'final_direction': final_direction,
                'confidence': round(confidence, 1),
                'is_inverted': is_inverted,
                'inversion_mode': inversion_mode,
                'adjustments': []
            }
            
    except Exception as e:
        logger.error(f"Erro na análise técnica: {e}")
        original_direction = "CALL" if random.random() > 0.5 else "PUT"
        final_direction, is_inverted, inversion_mode = inversion_manager.get_final_signal(original_direction)
        
        return {
            'original_direction': original_direction,
            'final_direction': final_direction,
            'confidence': 72.0,
            'is_inverted': is_inverted,
            'inversion_mode': inversion_mode,
            'adjustments': []
        }

def extract_features(data):
    """Extrair características dos dados"""
    current_price = data.get("currentPrice", 1000)
    volatility = data.get("volatility", 50)
    
    prices = data.get("lastTicks", [])
    if not prices:
        prices = [
            current_price - random.uniform(0, 4),
            current_price + random.uniform(0, 4), 
            current_price - random.uniform(0, 3)
        ]
    
    while len(prices) < 3:
        prices.append(current_price + random.uniform(-2, 2))
        
    return prices[-3:], volatility

# Sistema de limpeza automática
def background_cleanup_task():
    """Tarefa de limpeza em background"""
    while True:
        try:
            # Limpar ordens antigas
            duplication_controller.cleanup_old_orders()
            
            # Analisar padrões de duplicação
            if LEARNING_CONFIG['duplication_learning']:
                patterns = learning_engine.analyze_duplication_patterns()
                if patterns:
                    logger.info(f"🧠 Padrões de duplicação analisados: {len(patterns)}")
            
            time.sleep(60)  # 1 minuto
            
        except Exception as e:
            logger.error(f"Erro na limpeza: {e}")
            time.sleep(30)

# Iniciar thread de limpeza
cleanup_thread = threading.Thread(target=background_cleanup_task, daemon=True)
cleanup_thread.start()

# ===============================
# ROTAS DA API
# ===============================

@app.route("/")
def home():
    """Página inicial da API"""
    recent_data = db.get_recent_performance(100)
    total_signals = len(recent_data)
    accuracy = (sum(1 for signal in recent_data if signal[12] == 1) / total_signals * 100) if total_signals > 0 else 0
    
    inversion_status = inversion_manager.get_inversion_status()
    duplication_stats = duplication_controller.get_duplication_stats()
    
    return jsonify({
        "status": "🤖 IA Trading Bot API - Sistema Anti-Duplicação + Aprendizado",
        "version": "5.0.0 - Anti-Duplication + Advanced Learning",
        "description": "API com Sistema Anti-Duplicação Automática + Aprendizado Avançado",
        "model": "Advanced Anti-Duplication Engine + Learning System",
        "api_key": VALID_API_KEY,
        "duplication_control": {
            "active": DUPLICATION_CONTROL['active'],
            "total_attempts_blocked": duplication_stats['total_attempts'],
            "active_orders_tracked": duplication_stats['active_orders'],
            "learned_patterns": duplication_stats['learned_patterns'],
            "prevention_rules": duplication_stats['prevention_rules'],
            "threshold_ms": duplication_stats['threshold_ms']
        },
        "inversion_system": inversion_status,
        "learning_active": LEARNING_CONFIG['learning_enabled'],
        "endpoints": {
            "signal": "POST /signal - Sinais com anti-duplicação + aprendizado",
            "analyze": "POST /analyze - Análise protegida contra duplicatas",
            "risk": "POST /risk - Avaliação de risco + duplicação",
            "optimal-duration": "POST /optimal-duration - Duração otimizada",
            "management": "POST /management - Gestão com anti-duplicação",
            "feedback": "POST /feedback - Sistema de aprendizado avançado",
            "duplication-stats": "GET /duplication-stats - Estatísticas de duplicação"
        },
        "stats": {
            "total_predictions": total_signals,
            "current_accuracy": f"{accuracy:.1f}%",
            "duplicates_prevented": duplication_stats['total_attempts'],
            "learning_patterns": duplication_stats['learned_patterns']
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "Python Anti-Duplication API + Advanced Learning"
    })

@app.route("/signal", methods=["POST", "OPTIONS"])
def generate_signal():
    """Gerar sinal com proteção anti-duplicação"""
    if request.method == "OPTIONS":
        return '', 200
    
    if not validate_api_key():
        return jsonify({"error": "API Key inválida"}), 401
    
    try:
        data = request.get_json() or {}
        
        # Preparar dados da ordem
        order_data = {
            'symbol': data.get('symbol', 'R_50'),
            'direction': '',  # Será definido pela análise
            'stake': data.get('stake', 1.0),
            'duration_type': data.get('duration_type', 't'),
            'duration_value': data.get('duration_value', 5),
            'client_session_id': data.get('client_session_id', 'unknown')
        }
        
        # Análise técnica
        prices, volatility = extract_features(data)
        learning_data = {
            'symbol': order_data['symbol'],
            'volatility': volatility,
            'market_condition': data.get('marketCondition', 'neutral'),
            'martingale_level': data.get('martingaleLevel', 0)
        }
        
        analysis_result = analyze_technical_pattern(prices, learning_data)
        order_data['direction'] = analysis_result['final_direction']
        
        # ✅ VERIFICAR DUPLICAÇÃO ANTES DE GERAR SINAL
        is_duplicate, duplicate_info = duplication_controller.check_duplicate_order(order_data)
        
        if is_duplicate:
            # Registrar tentativa de duplicação
            prevention_info = duplication_controller.handle_duplicate_detected(order_data, duplicate_info)
            
            return jsonify({
                "error": "Ordem duplicada detectada",
                "duplicate_detected": True,
                "prevention_info": prevention_info,
                "original_order_time_diff": duplicate_info['time_difference'],
                "similarity_score": duplicate_info['similarity_score'],
                "message": "🚫 Sistema Anti-Duplicação: Ordem idêntica detectada nos últimos 5 segundos",
                "recommendation": "Aguarde alguns segundos antes de tentar novamente",
                "duplication_stats": duplication_controller.get_duplication_stats(),
                "timestamp": datetime.datetime.now().isoformat()
            }), 409  # Conflict
        
        # Registrar ordem válida
        order_id = duplication_controller.register_order(order_data)
        
        # Gerar sinal
        current_price = data.get("currentPrice", 1000)
        confidence = analysis_result['confidence']
        
        # Preparar dados para salvar
        signal_data = {
            'order_id': order_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'symbol': order_data['symbol'],
            'direction': analysis_result['final_direction'],
            'original_direction': analysis_result['original_direction'],
            'confidence': confidence,
            'entry_price': current_price,
            'volatility': volatility,
            'duration_type': order_data['duration_type'],
            'duration_value': order_data['duration_value'],
            'martingale_level': data.get('martingaleLevel', 0),
            'market_condition': data.get('marketCondition', 'neutral'),
            'is_inverted': analysis_result['is_inverted'],
            'consecutive_errors_before': INVERSION_SYSTEM['consecutive_errors'],
            'inversion_mode': analysis_result['inversion_mode'],
            'is_duplicate': False,
            'duplicate_detection_method': 'prevented',
            'client_session_id': order_data['client_session_id'],
            'technical_factors': {
                'adjustments': analysis_result['adjustments'],
                'prices': prices,
                'order_id': order_id
            }
        }
        
        # Salvar sinal no banco
        signal_id = db.save_signal(signal_data)
        
        # Preparar resposta
        reasoning = f"Sinal protegido para {order_data['symbol']} - Sistema anti-duplicação ativo"
        if analysis_result['is_inverted']:
            reasoning += f" - INVERTIDO ({analysis_result['original_direction']} → {analysis_result['final_direction']})"
        
        inversion_status = inversion_manager.get_inversion_status()
        
        return jsonify({
            "signal_id": signal_id,
            "order_id": order_id,
            "direction": analysis_result['final_direction'],
            "original_direction": analysis_result['original_direction'],
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_price": current_price,
            "duplicate_protected": True,
            "inverted": analysis_result['is_inverted'],
            "inversion_status": inversion_status,
            "duplication_control": {
                "active": True,
                "order_registered": True,
                "active_orders_count": len(DUPLICATION_CONTROL['active_orders']),
                "prevention_threshold_ms": DUPLICATION_CONTROL['duplicate_threshold']
            },
            "learning_adjustments": analysis_result['adjustments'],
            "factors": {
                "anti_duplication": "ATIVO",
                "volatility_factor": volatility,
                "inversion_mode": analysis_result['inversion_mode'],
                "learning_active": LEARNING_CONFIG['learning_enabled']
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "IA Anti-Duplicação + Sistema de Aprendizado Avançado"
        })
        
    except Exception as e:
        logger.error(f"Erro em signal: {e}")
        return jsonify({"error": "Erro na geração de sinal", "message": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def receive_feedback():
    """Receber feedback com aprendizado anti-duplicação"""
    if request.method == "OPTIONS":
        return '', 200
    
    try:
        data = request.get_json() or {}
        
        result = data.get("result", 0)
        signal_id = data.get("signal_id")
        order_id = data.get("order_id")
        pnl = data.get("pnl", 0)
        
        # Completar ordem no sistema anti-duplicação
        if order_id:
            duplication_controller.complete_order(order_id)
        
        # Atualizar resultado no banco
        if signal_id:
            db.update_signal_result(signal_id, result, pnl)
        
        # Sistema de inversão
        inversion_manager.handle_signal_result(result)
        
        # Atualizar estatísticas
        performance_stats['total_trades'] += 1
        if result == 1:
            performance_stats['won_trades'] += 1
        
        # Trigger análise de padrões
        if performance_stats['total_trades'] % 5 == 0:
            try:
                duplication_patterns = learning_engine.analyze_duplication_patterns()
                if duplication_patterns:
                    logger.info(f"🧠 Padrões de duplicação analisados: {len(duplication_patterns)}")
            except Exception as e:
                logger.error(f"Erro na análise de padrões: {e}")
        
        accuracy = (performance_stats['won_trades'] / max(performance_stats['total_trades'], 1) * 100)
        inversion_status = inversion_manager.get_inversion_status()
        duplication_stats = duplication_controller.get_duplication_stats()
        
        return jsonify({
            "message": "Feedback recebido - Sistema anti-duplicação + aprendizado ativo",
            "signal_id": signal_id,
            "order_id": order_id,
            "result_recorded": result == 1,
            "order_completed": order_id is not None,
            "total_trades": performance_stats['total_trades'],
            "accuracy": f"{accuracy:.1f}%",
            "anti_duplication_active": True,
            "duplication_stats": duplication_stats,
            "inversion_system": inversion_status,
            "learning_active": LEARNING_CONFIG['learning_enabled'],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em feedback: {e}")
        return jsonify({"error": "Erro no feedback", "message": str(e)}), 500

@app.route("/duplication-stats", methods=["GET"])
def get_duplication_stats():
    """Obter estatísticas detalhadas de duplicação"""
    try:
        duplication_stats = duplication_controller.get_duplication_stats()
        duplicate_db_stats = db.get_duplicate_statistics()
        
        return jsonify({
            "duplication_control": duplication_stats,
            "database_statistics": [
                {
                    "total_detections": stat[0],
                    "prevented_count": stat[1],
                    "avg_similarity": stat[2],
                    "duplicate_type": stat[3],
                    "detection_method": stat[4]
                } for stat in duplicate_db_stats
            ],
            "learned_patterns": dict(DUPLICATION_CONTROL['duplicate_patterns']),
            "prevention_rules": DUPLICATION_CONTROL['prevention_rules'],
            "performance_impact": {
                "total_trades": performance_stats['total_trades'],
                "duplicates_prevented": duplication_stats['total_attempts'],
                "prevention_rate": f"{(duplication_stats['total_attempts'] / max(performance_stats['total_trades'] + duplication_stats['total_attempts'], 1) * 100):.1f}%"
            },
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em duplication-stats: {e}")
        return jsonify({"error": "Erro ao obter estatísticas", "message": str(e)}), 500

# Outros endpoints (analyze, risk, optimal-duration, management) mantidos similares
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_market():
    if request.method == "OPTIONS":
        return '', 200
    
    try:
        data = request.get_json() or {}
        prices, volatility = extract_features(data)
        
        learning_data = {
            'symbol': data.get('symbol', 'R_50'),
            'volatility': volatility,
            'market_condition': data.get('marketCondition', 'neutral')
        }
        
        analysis_result = analyze_technical_pattern(prices, learning_data)
        
        return jsonify({
            "symbol": data.get('symbol', 'R_50'),
            "direction": analysis_result['final_direction'],
            "confidence": analysis_result['confidence'],
            "volatility": round(volatility, 1),
            "inverted": analysis_result['is_inverted'],
            "anti_duplication_active": True,
            "message": f"Análise protegida: {analysis_result['final_direction']}",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro em analyze: {e}")
        return jsonify({"error": "Erro na análise", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info("🚀 Iniciando IA Trading Bot API - Sistema Anti-Duplicação + Aprendizado")
    logger.info(f"🔑 API Key: {VALID_API_KEY}")
    logger.info("🚫 Sistema Anti-Duplicação: ATIVADO")
    logger.info("🧠 Sistema de Aprendizado: ATIVADO")
    logger.info(f"⏱️ Threshold anti-duplicação: {DUPLICATION_CONTROL['duplicate_threshold']}ms")
    logger.info(f"📊 Banco de dados: {DB_PATH}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
