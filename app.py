import sqlite3
import time
import threading
from contextlib import contextmanager
from functools import wraps

# Adicione esta classe para gerenciar conexões de forma thread-safe
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.local = threading.local()
        self._lock = threading.Lock()
    
    def get_connection(self):
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # Timeout de 30 segundos
                isolation_level='IMMEDIATE',
                check_same_thread=False
            )
            # Configurações otimizadas para evitar locks
            self.local.connection.execute('PRAGMA journal_mode=WAL')
            self.local.connection.execute('PRAGMA synchronous=NORMAL')
            self.local.connection.execute('PRAGMA busy_timeout=30000')
            self.local.connection.execute('PRAGMA temp_store=MEMORY')
        return self.local.connection
    
    @contextmanager
    def get_cursor(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()

# Inicialize o gerenciador de banco
db_manager = DatabaseManager('/tmp/trading_data.db')

# Decorator para retry automático em caso de database lock
def retry_on_lock(max_retries=3, delay=0.1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        print(f"Database locked, tentativa {attempt + 1}/{max_retries}")
                        time.sleep(delay * (2 ** attempt))  # Backoff exponencial
                        continue
                    raise e
            return None
        return wrapper
    return decorator

# Exemplo de como usar nas suas funções existentes
@retry_on_lock(max_retries=5, delay=0.2)
def save_feedback_safe(symbol, result, success):
    """Função segura para salvar feedback"""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO feedback (symbol, result, success, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (symbol, result, success, time.time()))
            print(f"✅ Feedback salvo: {symbol} - {result} - {'Sucesso' if success else 'Falha'}")
    except Exception as e:
        print(f"❌ Erro ao salvar feedback: {e}")
        raise

@retry_on_lock(max_retries=3, delay=0.1)
def get_learning_stats_safe():
    """Função segura para obter estatísticas"""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as accuracy
                FROM feedback
            ''')
            result = cursor.fetchone()
            return {
                'total': result[0] if result else 0,
                'successful': result[1] if result else 0,
                'accuracy': result[2] if result else 0.0
            }
    except Exception as e:
        print(f"❌ Erro ao obter estatísticas: {e}")
        return {'total': 0, 'successful': 0, 'accuracy': 0.0}

# Substitua suas rotas existentes por versões thread-safe
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        result = data.get('result')
        success = data.get('success', False)
        
        # Use a função segura
        save_feedback_safe(symbol, result, success)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback salvo com sucesso'
        })
    except Exception as e:
        print(f"ERROR: Erro em feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/learning-stats', methods=['GET'])
def learning_stats():
    try:
        stats = get_learning_stats_safe()
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        print(f"ERROR: Erro ao obter estatísticas: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Função para inicializar o banco de forma segura
@retry_on_lock(max_retries=5, delay=0.5)
def init_database():
    """Inicializa o banco de dados de forma segura"""
    try:
        with db_manager.get_cursor() as cursor:
            # Tabela de feedback
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    result TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            # Tabela de Q-Learning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS q_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reward REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_symbol ON feedback(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qlearning_state ON q_learning(state)')
            
            print("✅ Banco de dados inicializado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao inicializar banco: {e}")
        raise

# Adicione esta função de limpeza no final do arquivo
def cleanup_database():
    """Limpa conexões antigas do banco"""
    try:
        if hasattr(db_manager.local, 'connection'):
            db_manager.local.connection.close()
    except:
        pass

import atexit
atexit.register(cleanup_database)

# No início do seu app, substitua a inicialização do banco por:
if __name__ == '__main__':
    print("🚀 Iniciando IA Trading Bot API - SISTEMA DE APRENDIZADO AVANÇADO")
    print(f"🔑 API Key: {API_KEY}")
    print("🧠 Sistema de Aprendizado AVANÇADO: ATIVADO")
    
    # Inicializar banco de forma segura
    init_database()
    
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)
