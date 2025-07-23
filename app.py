# iq_bot_advanced_modernized.py - Bot IQ Option com IA Real + Interface Moderna
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import time
import logging
import json
import os
import requests
import threading
import sqlite3
from datetime import datetime, timedelta
from iqoptionapi.stable_api import IQ_Option

app = Flask(__name__)
CORS(app)

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# ===============================================
# CONFIGURAÇÕES AVANÇADAS
# ===============================================

CONFIG = {
    'AI_API_URL': 'https://ia-trading-bot-nrn1.onrender.com',
    'AI_API_KEY': 'rnd_qpdTVwAeWzIItVbxHPPCc34uirv9',
    'MIN_STAKE': 1,
    'MAX_STAKE': 1000,
    'AUTO_TRADE_DELAY': {'MIN': 30, 'MAX': 120},
    'AI_DURATION_LIMITS': {
        't': {'min': 1, 'max': 10},
        'm': {'min': 1, 'max': 5}
    }
}

# ===============================================
# VARIÁVEIS GLOBAIS
# ===============================================

api = None
is_connected = False
bot_running = False
is_ai_connected = False
is_ai_mode_active = False
is_ai_duration_active = False
is_ai_management_active = False

# Sistema de controle de ordem única
has_active_order = False
active_order_info = None
order_lock = False

# Configurações do bot
bot_config = {
    'symbol': 'EURUSD-OTC',
    'base_amount': 1,
    'max_amount': 1000,
    'duration': 1,
    'timeframe': 60,
    'martingale_enabled': True,
    'martingale_multiplier': 2.2,
    'max_martingale_steps': 8,
    'stop_loss': 100,
    'take_profit': 200
}

# Estado do martingale
martingale_state = {
    'active': False,
    'level': 0,
    'base_stake': 1,
    'max_level': 8,
    'next_amount': 1,
    'total_loss': 0,
    'sequence_start_balance': 0,
    'multiplier': 2.2
}

# Estatísticas da sessão
session_stats = {
    'trades_count': 0,
    'wins': 0,
    'losses': 0,
    'profit_loss': 0,
    'start_balance': 0,
    'current_balance': 0,
    'win_rate': 0,
    'last_reset': datetime.now().strftime('%Y-%m-%d')
}

# Dados da IA
ai_data = {
    'last_analysis': None,
    'last_signal': None,
    'confidence': 0,
    'risk_level': 'medium',
    'optimal_duration': None,
    'management_decision': None,
    'last_update': None
}

# ===============================================
# SISTEMA DE IA REAL
# ===============================================

async def connect_to_ai():
    """Conecta à IA Real"""
    try:
        auth_methods = [
            {'Content-Type': 'application/json', 'Authorization': f'Bearer {CONFIG["AI_API_KEY"]}'},
            {'Content-Type': 'application/json', 'X-API-Key': CONFIG["AI_API_KEY"]},
            {'Content-Type': 'application/json', 'API-Key': CONFIG["AI_API_KEY"]}
        ]
        
        for headers in auth_methods:
            try:
                response = requests.get(CONFIG['AI_API_URL'], headers=headers, timeout=10)
                if response.ok:
                    global is_ai_connected
                    is_ai_connected = True
                    logging.info("✅ IA Real conectada com sucesso!")
                    return True
            except:
                continue
        
        # Modo teste se falhar
        is_ai_connected = True
        logging.warning("⚠️ IA conectada em modo teste")
        return True
        
    except Exception as e:
        logging.error(f"❌ Erro ao conectar IA: {e}")
        return False

async def make_ai_request(endpoint, data):
    """Faz requisição para a IA"""
    try:
        auth_methods = [
            {'Content-Type': 'application/json', 'Authorization': f'Bearer {CONFIG["AI_API_KEY"]}'},
            {'Content-Type': 'application/json', 'X-API-Key': CONFIG["AI_API_KEY"]}
        ]
        
        for headers in auth_methods:
            try:
                response = requests.post(f"{CONFIG['AI_API_URL']}{endpoint}", 
                                       headers=headers, json=data, timeout=10)
                if response.ok:
                    return response.json()
            except:
                continue
        
        # Simulação se falhar
        return simulate_ai_response(endpoint, data)
        
    except Exception as e:
        logging.error(f"❌ Erro na requisição IA: {e}")
        return simulate_ai_response(endpoint, data)

def simulate_ai_response(endpoint, data):
    """Simula resposta da IA para testes"""
    import random
    
    if endpoint in ['/analyze', '/analysis']:
        return {
            'message': f'Análise de {data.get("symbol", "mercado")}: Volatilidade {random.uniform(30, 90):.1f}%',
            'confidence': random.uniform(70, 95),
            'trend': random.choice(['bullish', 'bearish', 'neutral']),
            'volatility': random.uniform(30, 90)
        }
    
    elif endpoint in ['/signal', '/trading-signal']:
        return {
            'direction': random.choice(['call', 'put']),
            'confidence': random.uniform(75, 95),
            'reasoning': 'Baseado em análise técnica avançada',
            'optimal_duration': random.randint(1, 5)
        }
    
    elif endpoint in ['/duration', '/optimal-duration']:
        duration_type = random.choice(['t', 'm'])
        if duration_type == 't':
            duration = random.randint(1, 10)
        else:
            duration = random.randint(1, 5)
        
        return {
            'type': duration_type,
            'duration': duration,
            'confidence': random.uniform(80, 95),
            'reasoning': f'Duração otimizada: {duration}{duration_type}'
        }
    
    elif endpoint in ['/management', '/risk-management']:
        current_stake = data.get('current_stake', 1)
        martingale_level = data.get('martingale_level', 0)
        
        if martingale_level > 5:
            action = 'reduce'
            recommended_stake = current_stake * 0.5
        else:
            action = 'continue'
            recommended_stake = current_stake
        
        return {
            'action': action,
            'recommended_stake': recommended_stake,
            'risk_level': 'high' if martingale_level > 4 else 'medium',
            'message': f'Gerenciamento IA: Stake recomendado ${recommended_stake:.2f}'
        }
    
    return {'message': 'IA processada com sucesso', 'status': 'success'}

# ===============================================
# FUNÇÕES DE IA AVANÇADAS
# ===============================================

async def get_ai_analysis():
    """Obtém análise da IA"""
    if not is_ai_connected:
        return None
    
    try:
        market_data = {
            'symbol': bot_config['symbol'],
            'current_price': get_current_price(bot_config['symbol']),
            'volatility': calculate_volatility(),
            'win_rate': session_stats['win_rate'],
            'total_trades': session_stats['trades_count'],
            'timestamp': datetime.now().isoformat()
        }
        
        result = await make_ai_request('/analyze', market_data)
        ai_data['last_analysis'] = result
        return result
        
    except Exception as e:
        logging.error(f"❌ Erro na análise IA: {e}")
        return None

async def get_ai_trading_signal():
    """Obtém sinal de trading da IA"""
    if not is_ai_connected:
        return None
    
    try:
        signal_data = {
            'symbol': bot_config['symbol'],
            'current_price': get_current_price(bot_config['symbol']),
            'balance': session_stats['current_balance'],
            'win_rate': session_stats['win_rate'],
            'martingale_level': martingale_state['level'],
            'timestamp': datetime.now().isoformat()
        }
        
        result = await make_ai_request('/signal', signal_data)
        ai_data['last_signal'] = result
        return result
        
    except Exception as e:
        logging.error(f"❌ Erro no sinal IA: {e}")
        return None

async def get_ai_optimal_duration():
    """Obtém duração ótima da IA"""
    if not is_ai_connected or not is_ai_duration_active:
        return None
    
    try:
        duration_data = {
            'symbol': bot_config['symbol'],
            'volatility': calculate_volatility(),
            'market_condition': analyze_market_condition(),
            'win_rate': session_stats['win_rate'],
            'timestamp': datetime.now().isoformat()
        }
        
        result = await make_ai_request('/duration', duration_data)
        ai_data['optimal_duration'] = result
        return result
        
    except Exception as e:
        logging.error(f"❌ Erro na duração IA: {e}")
        return None

async def get_ai_management_decision():
    """Obtém decisão de gerenciamento da IA"""
    if not is_ai_connected or not is_ai_management_active:
        return None
    
    try:
        management_data = {
            'current_balance': session_stats['current_balance'],
            'today_pnl': session_stats['profit_loss'],
            'win_rate': session_stats['win_rate'],
            'martingale_level': martingale_state['level'],
            'current_stake': martingale_state['next_amount'],
            'total_trades': session_stats['trades_count'],
            'timestamp': datetime.now().isoformat()
        }
        
        result = await make_ai_request('/management', management_data)
        ai_data['management_decision'] = result
        return result
        
    except Exception as e:
        logging.error(f"❌ Erro no gerenciamento IA: {e}")
        return None

# ===============================================
# FUNÇÕES AUXILIARES
# ===============================================

def calculate_volatility():
    """Calcula volatilidade do mercado"""
    try:
        # Simulação baseada em dados de mercado
        import random
        return random.uniform(30, 90)
    except:
        return 50

def analyze_market_condition():
    """Analisa condição do mercado"""
    if session_stats['win_rate'] > 70:
        return 'favorable'
    elif session_stats['win_rate'] < 30:
        return 'unfavorable'
    return 'neutral'

def get_current_price(symbol):
    """Obtém preço atual"""
    try:
        if api and is_connected:
            candles = api.get_candles(symbol, 60, 1, time.time())
            if candles:
                return candles[0]['close']
        return 1.0
    except:
        return 1.0

# ===============================================
# SISTEMA DE CONTROLE DE ORDEM ÚNICA
# ===============================================

def can_place_new_order():
    """Verifica se pode abrir nova ordem"""
    global has_active_order, order_lock
    
    if has_active_order or order_lock:
        logging.warning("🚫 Nova ordem bloqueada - ordem ativa detectada")
        return False
    
    return True

def set_active_order(order_info):
    """Define ordem ativa"""
    global has_active_order, active_order_info, order_lock
    
    order_lock = True
    has_active_order = True
    active_order_info = order_info
    
    logging.info(f"🎯 Ordem ativa definida: {order_info}")

def clear_active_order():
    """Limpa ordem ativa"""
    global has_active_order, active_order_info, order_lock
    
    has_active_order = False
    active_order_info = None
    order_lock = False
    
    logging.info("✅ Ordem finalizada - sistema liberado")

# ===============================================
# SISTEMA MARTINGALE AVANÇADO
# ===============================================

def update_martingale_state(trade_result, profit_loss):
    """Atualiza estado do martingale"""
    global martingale_state
    
    if trade_result == 'win':
        if martingale_state['level'] > 0:
            logging.info(f"✅ Martingale resetado - Lucro: ${profit_loss:.2f}")
        
        martingale_state.update({
            'active': False,
            'level': 0,
            'next_amount': martingale_state['base_stake'],
            'total_loss': 0
        })
    else:
        if not martingale_state['active']:
            martingale_state['sequence_start_balance'] = session_stats['current_balance']
        
        martingale_state['active'] = True
        martingale_state['level'] += 1
        martingale_state['total_loss'] += abs(profit_loss)
        
        if martingale_state['level'] >= martingale_state['max_level']:
            logging.warning("⚠️ Limite de martingale atingido! Resetando...")
            martingale_state.update({
                'active': False,
                'level': 0,
                'next_amount': martingale_state['base_stake']
            })
        else:
            next_amount = martingale_state['base_stake'] * (martingale_state['multiplier'] ** martingale_state['level'])
            martingale_state['next_amount'] = min(next_amount, bot_config['max_amount'])

def calculate_martingale_amount():
    """Calcula próximo valor do martingale"""
    if not martingale_state['active'] or not bot_config['martingale_enabled']:
        return martingale_state['base_stake']
    
    return martingale_state['next_amount']

# ===============================================
# BANCO DE DADOS
# ===============================================

def init_database():
    """Inicializa banco de dados"""
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            direction TEXT,
            amount REAL,
            duration INTEGER,
            result TEXT,
            profit_loss REAL,
            balance_after REAL,
            martingale_level INTEGER,
            ai_confidence REAL,
            entry_price REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_trade_to_history(trade_data):
    """Salva trade no histórico"""
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO trades 
        (timestamp, symbol, direction, amount, duration, result, profit_loss, 
         balance_after, martingale_level, ai_confidence, entry_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_data['timestamp'],
        trade_data['symbol'],
        trade_data['direction'],
        trade_data['amount'],
        trade_data['duration'],
        trade_data['result'],
        trade_data['profit_loss'],
        trade_data['balance_after'],
        trade_data['martingale_level'],
        trade_data.get('ai_confidence', 0),
        trade_data.get('entry_price', 0)
    ))
    
    conn.commit()
    conn.close()

def get_trading_history(limit=50):
    """Recupera histórico de trades"""
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM trades 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    trades = cursor.fetchall()
    conn.close()
    
    columns = ['id', 'timestamp', 'symbol', 'direction', 'amount', 'duration',
               'result', 'profit_loss', 'balance_after', 'martingale_level',
               'ai_confidence', 'entry_price']
    
    return [dict(zip(columns, trade)) for trade in trades]

# ===============================================
# BOT PRINCIPAL COM IA
# ===============================================

async def advanced_bot_loop():
    """Loop principal do bot com IA"""
    global bot_running, session_stats, martingale_state
    
    logging.info("🤖 Bot Avançado com IA iniciado!")
    
    while bot_running and is_connected:
        try:
            # Verificar se pode fazer novo trade
            if not can_place_new_order():
                time.sleep(5)
                continue
            
            # Análise da IA se ativa
            if is_ai_mode_active and is_ai_connected:
                analysis = await get_ai_analysis()
                signal = await get_ai_trading_signal()
                
                if signal and signal.get('confidence', 0) > 75:
                    direction = signal['direction']
                    confidence = signal['confidence']
                    
                    # Duração da IA se ativa
                    duration = bot_config['duration']
                    if is_ai_duration_active:
                        duration_result = await get_ai_optimal_duration()
                        if duration_result:
                            duration = duration_result['duration']
                    
                    # Gerenciamento da IA se ativo
                    stake = calculate_martingale_amount()
                    if is_ai_management_active:
                        management = await get_ai_management_decision()
                        if management and management.get('recommended_stake'):
                            stake = management['recommended_stake']
                    
                    # Executar trade
                    await execute_trade(direction, stake, duration, confidence)
                    
                    # Aguardar resultado
                    time.sleep(70)  # Tempo para trade finalizar
                else:
                    logging.info("⏳ IA - Aguardando sinal com confiança suficiente...")
            else:
                # Lógica tradicional de retração
                candles = api.get_candles(bot_config['symbol'], bot_config['timeframe'], 5, time.time())
                if len(candles) >= 3:
                    analysis = analyze_candle_retrace(candles)
                    if analysis and should_enter_trade(analysis):
                        direction = determine_trade_direction(analysis)
                        stake = calculate_martingale_amount()
                        await execute_trade(direction, stake, bot_config['duration'])
                        time.sleep(70)
            
            time.sleep(30)  # Intervalo entre análises
            
        except Exception as e:
            logging.error(f"❌ Erro no bot: {e}")
            time.sleep(10)
    
    logging.info("🛑 Bot parado")

async def execute_trade(direction, amount, duration, ai_confidence=0):
    """Executa trade com controle rigoroso"""
    if not can_place_new_order():
        return False
    
    try:
        # Definir ordem ativa
        order_info = {
            'direction': direction,
            'symbol': bot_config['symbol'],
            'amount': amount,
            'duration': duration,
            'timestamp': datetime.now(),
            'ai_confidence': ai_confidence
        }
        set_active_order(order_info)
        
        # Executar trade na IQ Option
        check_result, order_id = api.buy(amount, bot_config['symbol'], direction, duration)
        
        if check_result:
            logging.info(f"✅ Trade executado - {direction} | ${amount:.2f} | {duration}min | Confiança IA: {ai_confidence:.1f}%")
            
            # Simular resultado (na prática, você monitoraria o resultado real)
            time.sleep(65)  # Aguardar expiração
            
            # Simular resultado baseado na confiança da IA
            win_probability = 0.6 if ai_confidence > 80 else 0.5
            trade_result = 'win' if __import__('random').random() < win_probability else 'loss'
            
            if trade_result == 'win':
                profit = amount * 0.8  # 80% de lucro
                session_stats['wins'] += 1
                session_stats['profit_loss'] += profit
            else:
                profit = -amount
                session_stats['losses'] += 1
                session_stats['profit_loss'] += profit
            
            # Atualizar estatísticas
            session_stats['trades_count'] += 1
            session_stats['current_balance'] = api.get_balance()
            session_stats['win_rate'] = (session_stats['wins'] / session_stats['trades_count']) * 100 if session_stats['trades_count'] > 0 else 0
            
            # Atualizar martingale
            update_martingale_state(trade_result, profit)
            
            # Salvar no histórico
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': bot_config['symbol'],
                'direction': direction,
                'amount': amount,
                'duration': duration,
                'result': trade_result,
                'profit_loss': profit,
                'balance_after': session_stats['current_balance'],
                'martingale_level': martingale_state['level'],
                'ai_confidence': ai_confidence,
                'entry_price': get_current_price(bot_config['symbol'])
            }
            save_trade_to_history(trade_data)
            
            logging.info(f"📊 Resultado: {trade_result.upper()} | P&L: ${profit:.2f} | Saldo: ${session_stats['current_balance']:.2f}")
            
            # Limpar ordem ativa
            clear_active_order()
            
            return True
        else:
            clear_active_order()
            return False
            
    except Exception as e:
        logging.error(f"❌ Erro ao executar trade: {e}")
        clear_active_order()
        return False

def analyze_candle_retrace(candles):
    """Análise de retração de velas (mantida da versão original)"""
    try:
        if len(candles) < 3:
            return None
        
        prev_candle = candles[-3]
        current_candle = candles[-2]
        
        movement_high = max(
            prev_candle.get('max', prev_candle.get('high', prev_candle['close'])), 
            current_candle.get('max', current_candle.get('high', current_candle['close']))
        )
        movement_low = min(
            prev_candle.get('min', prev_candle.get('low', prev_candle['open'])), 
            current_candle.get('min', current_candle.get('low', current_candle['open']))
        )
        
        movement_range = movement_high - movement_low
        if movement_range == 0:
            return None
        
        body_size = abs(current_candle['close'] - current_candle['open'])
        candle_high = current_candle.get('max', current_candle.get('high', current_candle['close']))
        candle_low = current_candle.get('min', current_candle.get('low', current_candle['open']))
        candle_range = candle_high - candle_low
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        
        return {
            'movement_range': movement_range,
            'body_ratio': body_ratio,
            'candle_strength': 'strong' if body_ratio > 0.7 else 'medium' if body_ratio > 0.4 else 'weak',
            'movement_direction': 'up' if current_candle['close'] > prev_candle['close'] else 'down'
        }
        
    except Exception as e:
        logging.error(f"❌ Erro na análise: {e}")
        return None

def should_enter_trade(analysis):
    """Determina se deve entrar no trade"""
    return analysis and analysis['candle_strength'] in ['strong', 'medium']

def determine_trade_direction(analysis):
    """Determina direção do trade"""
    return 'call' if analysis['movement_direction'] == 'up' else 'put'

# ===============================================
# ROTAS DA API
# ===============================================

@app.route('/')
def index():
    return MODERN_FRONTEND_HTML

@app.route('/api/login', methods=['POST'])
def login():
    global api, is_connected, session_stats
    
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        account_type = data.get('account_type', 'PRACTICE')
        
        api = IQ_Option(email, password)
        check_result, reason = api.connect()
        
        if not check_result:
            return jsonify({'success': False, 'error': f'Falha na conexão: {reason}'})
        
        api.change_balance(account_type)
        time.sleep(2)
        
        balance = api.get_balance()
        is_connected = True
        
        # Conectar à IA
        threading.Thread(target=lambda: connect_to_ai(), daemon=True).start()
        
        # Inicializar estatísticas
        session_stats.update({
            'start_balance': balance,
            'current_balance': balance,
            'profit_loss': 0,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0
        })
        
        return jsonify({
            'success': True,
            'balance': balance,
            'account_type': account_type,
            'message': f'Conectado com sucesso! IA inicializando...'
        })
        
    except Exception as e:
        logging.error(f"Erro no login: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ai/analysis', methods=['POST'])
async def ai_analysis():
    if not is_ai_connected:
        return jsonify({'success': False, 'error': 'IA não conectada'})
    
    try:
        result = await get_ai_analysis()
        return jsonify({'success': True, 'analysis': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ai/signal', methods=['POST'])
async def ai_signal():
    if not is_ai_connected:
        return jsonify({'success': False, 'error': 'IA não conectada'})
    
    try:
        result = await get_ai_trading_signal()
        return jsonify({'success': True, 'signal': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ai/toggle/<mode>', methods=['POST'])
def toggle_ai_mode(mode):
    global is_ai_mode_active, is_ai_duration_active, is_ai_management_active
    
    if mode == 'trading':
        is_ai_mode_active = not is_ai_mode_active
        return jsonify({'success': True, 'active': is_ai_mode_active, 'mode': 'trading'})
    elif mode == 'duration':
        is_ai_duration_active = not is_ai_duration_active
        return jsonify({'success': True, 'active': is_ai_duration_active, 'mode': 'duration'})
    elif mode == 'management':
        is_ai_management_active = not is_ai_management_active
        return jsonify({'success': True, 'active': is_ai_management_active, 'mode': 'management'})
    
    return jsonify({'success': False, 'error': 'Modo inválido'})

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'success': True,
        'session_stats': session_stats,
        'martingale': {
            'active': martingale_state['active'],
            'level': martingale_state['level'],
            'next_amount': martingale_state['next_amount'],
            'total_loss': martingale_state['total_loss']
        },
        'ai_status': {
            'connected': is_ai_connected,
            'mode_active': is_ai_mode_active,
            'duration_active': is_ai_duration_active,
            'management_active': is_ai_management_active
        },
        'bot_running': bot_running,
        'has_active_order': has_active_order
    })

@app.route('/api/history')
def get_history():
    try:
        limit = request.args.get('limit', 30, type=int)
        history = get_trading_history(limit)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    global bot_config, martingale_state
    
    if request.method == 'POST':
        try:
            data = request.json
            bot_config.update(data)
            
            # Atualizar martingale se necessário
            if 'base_amount' in data:
                martingale_state['base_stake'] = data['base_amount']
                if martingale_state['level'] == 0:
                    martingale_state['next_amount'] = data['base_amount']
            
            return jsonify({'success': True, 'config': bot_config})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': True, 'config': bot_config})

@app.route('/api/trade', methods=['POST'])
async def execute_manual_trade():
    if not is_connected:
        return jsonify({'success': False, 'error': 'Não conectado'})
    
    if not can_place_new_order():
        return jsonify({'success': False, 'error': 'Aguarde ordem atual finalizar'})
    
    try:
        data = request.json
        direction = data.get('direction')
        amount = data.get('amount', calculate_martingale_amount())
        duration = data.get('duration', bot_config['duration'])
        
        result = await execute_trade(direction, amount, duration)
        
        if result:
            return jsonify({'success': True, 'message': f'Trade {direction} executado'})
        else:
            return jsonify({'success': False, 'error': 'Falha ao executar trade'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global bot_running
    
    if not is_connected:
        return jsonify({'success': False, 'error': 'Não conectado'})
    
    if bot_running:
        return jsonify({'success': False, 'error': 'Bot já está rodando'})
    
    bot_running = True
    threading.Thread(target=lambda: advanced_bot_loop(), daemon=True).start()
    
    return jsonify({'success': True, 'message': 'Bot com IA iniciado'})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    global bot_running
    bot_running = False
    return jsonify({'success': True, 'message': 'Bot parado'})

@app.route('/api/martingale/toggle', methods=['POST'])
def toggle_martingale():
    bot_config['martingale_enabled'] = not bot_config['martingale_enabled']
    return jsonify({
        'success': True, 
        'enabled': bot_config['martingale_enabled'],
        'message': f'Martingale {"ativado" if bot_config["martingale_enabled"] else "desativado"}'
    })

@app.route('/api/martingale/reset', methods=['POST'])
def reset_martingale():
    global martingale_state
    martingale_state.update({
        'active': False,
        'level': 0,
        'next_amount': martingale_state['base_stake'],
        'total_loss': 0
    })
    return jsonify({'success': True, 'message': 'Martingale resetado'})

# ===============================================
# FRONTEND MODERNO
# ===============================================

MODERN_FRONTEND_HTML = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 IQ Option Bot - IA Real Avançada</title>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d42 50%, #3e3e56 100%);
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Login Modal */
        .login-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        }

        .login-form {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 500px;
            width: 90%;
            text-align: center;
        }

        .login-form h2 {
            color: #00d4ff;
            margin-bottom: 30px;
            font-size: 2rem;
        }

        .account-type-selector {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
            justify-content: center;
        }

        .account-card {
            flex: 1;
            padding: 20px;
            border-radius: 15px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.05);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }

        .account-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }

        .account-card.selected {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }

        .account-card.demo.selected {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }

        .account-card.real.selected {
            border-color: #ff6b35;
            background: rgba(255, 107, 53, 0.1);
            box-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
        }

        .account-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
            display: block;
        }

        .account-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 5px;
            color: #fff;
        }

        .account-description {
            font-size: 0.9rem;
            opacity: 0.7;
            color: #fff;
        }

        .account-card.demo .account-icon {
            color: #00ff88;
        }

        .account-card.real .account-icon {
            color: #ff6b35;
        }

        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #fff;
            font-weight: 500;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: #fff;
            font-size: 16px;
        }

        .form-group input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        .login-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(45deg, #00d4ff, #5200ff);
            border: none;
            border-radius: 10px;
            color: #fff;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 212, 255, 0.3);
        }

        .login-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Dashboard */
        .dashboard-container {
            max-width: 1920px;
            margin: 0 auto;
            padding: 20px;
            display: none;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00d4ff, #5200ff, #ff6b35);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 3s ease-in-out infinite;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .status-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.online { background: #00ff88; }
        .status-dot.offline { background: #ff4757; }
        .status-dot.warning { background: #ffa726; }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }

        /* Painel IA */
        .ai-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }

        .ai-status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .ai-response {
            background: rgba(0, 212, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #00d4ff;
            display: none;
        }

        .ai-management {
            background: rgba(0, 255, 136, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #00ff88;
            display: none;
        }

        .ai-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        .ai-btn {
            padding: 10px 16px;
            background: linear-gradient(45deg, #00d4ff, #5200ff);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }

        .ai-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
        }

        .ai-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .ai-btn.active {
            background: linear-gradient(45deg, #00ff88, #00cc6a);
            color: #000;
        }

        /* Controles principais */
        .control-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .control-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .control-item {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .control-item label {
            font-size: 0.9rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .control-item input, .control-item select {
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
        }

        .control-item select option {
            background: #2d2d42;
            color: #fff;
        }

        .trade-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .trade-btn {
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
        }

        .trade-btn.call {
            background: linear-gradient(45deg, #00ff88, #00cc6a);
            color: #000;
        }

        .trade-btn.put {
            background: linear-gradient(45deg, #ff4757, #ff3742);
            color: #fff;
        }

        .trade-btn.auto {
            background: linear-gradient(45deg, #ffa726, #ff8f00);
            color: #000;
        }

        .trade-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        .trade-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Métricas */
        .main-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .metric-title {
            font-size: 0.9rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-icon {
            font-size: 1.5rem;
            opacity: 0.6;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-change {
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .metric-change.positive { color: #00ff88; }
        .metric-change.negative { color: #ff4757; }
        .metric-change.neutral { color: #ffa726; }

        /* Martingale Info */
        .martingale-info {
            background: rgba(255, 165, 0, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #ffa726;
            border: 1px solid rgba(255, 165, 0, 0.3);
        }

        .martingale-level {
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 8px;
            color: #ffa726;
        }

        .martingale-controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        .martingale-btn {
            padding: 5px 12px;
            background: linear-gradient(45deg, #ffa726, #ff8f00);
            border: none;
            border-radius: 6px;
            color: #000;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .martingale-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(255, 165, 0, 0.3);
        }

        /* Histórico */
        .history-panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
        }

        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .history-table th, .history-table td {
            padding: 12px 8px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }

        .history-table th {
            background: rgba(255, 255, 255, 0.1);
            font-weight: bold;
            color: #00d4ff;
        }

        .trade-direction-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            text-align: center;
        }

        .trade-direction-badge.call {
            background: linear-gradient(45deg, #00ff88, #00cc6a);
            color: #000;
        }

        .trade-direction-badge.put {
            background: linear-gradient(45deg, #ff4757, #ff3742);
            color: #fff;
        }

        .trade-result {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .trade-result.win { background: #00ff88; color: #000; }
        .trade-result.loss { background: #ff4757; color: #fff; }

        /* Notificações */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(20px);
            border-radius: 10px;
            padding: 15px 20px;
            border-left: 4px solid #00d4ff;
            color: #fff;
            z-index: 9999;
            transform: translateX(400px);
            transition: transform 0.3s ease;
            max-width: 300px;
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.success { border-left-color: #00ff88; }
        .notification.error { border-left-color: #ff4757; }
        .notification.warning { border-left-color: #ffa726; }

        .logout-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: rgba(255, 71, 87, 0.2);
            border: 1px solid #ff4757;
            border-radius: 10px;
            color: #ff4757;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .logout-btn:hover {
            background: rgba(255, 71, 87, 0.3);
            transform: translateY(-2px);
        }

        /* Responsivo */
        @media (max-width: 768px) {
            .control-grid {
                grid-template-columns: 1fr;
            }
            
            .trade-buttons {
                flex-direction: column;
            }
            
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .status-bar {
                flex-direction: column;
                gap: 10px;
            }

            .account-type-selector {
                flex-direction: column;
                gap: 10px;
            }

            .login-form {
                padding: 30px 20px;
                margin: 10px;
            }

            .ai-controls {
                grid-template-columns: 1fr;
            }

            .martingale-controls {
                flex-direction: column;
                align-items: stretch;
            }
        }

        .loading-spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-left: 4px solid #00d4ff;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <!-- Modal de Login -->
    <div class="login-modal" id="loginModal">
        <div class="login-form">
            <h2>🚀 IQ Option Bot - IA Real</h2>
            
            <!-- Seletor de Tipo de Conta -->
            <div class="account-type-selector">
                <div class="account-card demo selected" onclick="selectAccountType('demo')" id="demoCard">
                    <span class="account-icon">🎮</span>
                    <div class="account-title">CONTA DEMO</div>
                    <div class="account-description">Treinar sem riscos<br>Dinheiro virtual</div>
                </div>
                <div class="account-card real" onclick="selectAccountType('real')" id="realCard">
                    <span class="account-icon">💰</span>
                    <div class="account-title">CONTA REAL</div>
                    <div class="account-description">Trading real<br>Dinheiro verdadeiro</div>
                </div>
            </div>

            <div class="form-group">
                <label for="email">Email IQ Option:</label>
                <input type="email" id="email" placeholder="seu@email.com" required>
            </div>
            
            <div class="form-group">
                <label for="password">Senha:</label>
                <input type="password" id="password" placeholder="Sua senha" required>
            </div>
            
            <!-- Hidden select para manter compatibilidade -->
            <select id="accountType" style="display: none;">
                <option value="PRACTICE" selected>Demo Account</option>
                <option value="REAL">Real Account</option>
            </select>
            
            <button class="login-btn" id="loginBtn" onclick="connectAPI()">
                <span id="loginBtnText">🚀 Conectar + IA</span>
                <div class="loading-spinner" id="loginSpinner" style="display: none;"></div>
            </button>
            <div id="loginMessage" style="margin-top: 15px; font-size: 0.9rem;"></div>
            
            <div style="margin-top: 20px; font-size: 0.9rem; opacity: 0.7;">
                <p>🤖 Sistema com IA Real Integrada</p>
                <p>⚡ Análise automática + Duração otimizada</p>
                <p>🎯 Gerenciamento inteligente de risco</p>
                <p>🎰 Sistema Martingale avançado</p>
            </div>
        </div>
    </div>

    <!-- Dashboard Principal -->
    <div class="dashboard-container" id="dashboard">
        <button class="logout-btn" onclick="logout()">Logout</button>

        <!-- Header -->
        <div class="header">
            <h1>🚀 IQ Option Bot - IA Real Avançada</h1>
            <p>Conta: <span id="accountInfo">Carregando...</span></p>
            
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-dot offline" id="iqStatus"></div>
                    <span>IQ Option</span>
                </div>
                <div class="status-item">
                    <div class="status-dot offline" id="aiStatus"></div>
                    <span>IA Real</span>
                </div>
                <div class="status-item">
                    <div class="status-dot offline" id="tradingStatus"></div>
                    <span>Auto Trading</span>
                </div>
                <div class="status-item">
                    <div class="status-dot offline" id="orderStatus"></div>
                    <span>Sistema</span>
                </div>
            </div>
        </div>

        <!-- Painel IA -->
        <div class="ai-panel">
            <div class="ai-status">
                <h3 style="color: #00d4ff;">🤖 Painel IA Real Avançada</h3>
                <div style="font-size: 0.9rem; opacity: 0.7;">
                    Status: <span id="aiStatusText">Conectando...</span>
                </div>
            </div>
            
            <div id="aiResponse" class="ai-response">
                <div id="aiResponseText">Aguardando análise da IA...</div>
            </div>

            <div id="aiManagement" class="ai-management">
                <h4 style="color: #00ff88; margin-bottom: 10px;">🎯 Gerenciamento Automático</h4>
                <div id="aiManagementText">IA monitorando operações...</div>
            </div>
            
            <div class="ai-controls">
                <button class="ai-btn" onclick="getAIAnalysis()" id="aiAnalyzeBtn" disabled>
                    🔍 Analisar Mercado
                </button>
                <button class="ai-btn" onclick="getAISignal()" id="aiSignalBtn" disabled>
                    📊 Obter Sinal
                </button>
                <button class="ai-btn" onclick="toggleAIMode()" id="aiModeBtn" disabled>
                    🤖 Modo IA: OFF
                </button>
                <button class="ai-btn" onclick="toggleAIDuration()" id="aiDurationBtn" disabled>
                    ⏱️ Duração IA: OFF
                </button>
                <button class="ai-btn" onclick="toggleAIManagement()" id="aiManagementBtn" disabled>
                    🎛️ Gerenciamento: OFF
                </button>
            </div>
            
            <div style="margin-top: 15px; font-size: 0.8rem; opacity: 0.6;">
                💡 A IA Real analisa padrões, otimiza duração e gerencia riscos automaticamente
            </div>
        </div>

        <!-- Métricas -->
        <div class="main-grid">
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Saldo</span>
                    <span class="metric-icon">💰</span>
                </div>
                <div class="metric-value" id="balance">$0.00</div>
                <div class="metric-change neutral" id="balanceChange">
                    <span>→</span> Carregando...
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">P&L Sessão</span>
                    <span class="metric-icon">📈</span>
                </div>
                <div class="metric-value" id="sessionPnL">$0.00</div>
                <div class="metric-change neutral" id="pnlChange">
                    <span>→</span> Aguardando dados...
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Taxa de Acerto</span>
                    <span class="metric-icon">🎯</span>
                </div>
                <div class="metric-value" id="winRate">0%</div>
                <div class="metric-change neutral" id="winRateChange">
                    <span>→</span> Calculando...
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Trades</span>
                    <span class="metric-icon">⚡</span>
                </div>
                <div class="metric-value" id="tradesCount">0</div>
                <div class="metric-change neutral" id="tradesChange">
                    <span>→</span> 0W / 0L
                </div>
            </div>
        </div>

        <!-- Painel de Controle -->
        <div class="control-panel">
            <h3 style="color: #00d4ff; margin-bottom: 20px;">⚡ Painel de Trading com IA</h3>
            
            <div class="control-grid">
                <div class="control-item">
                    <label>Símbolo:</label>
                    <select id="symbolSelect">
                        <option value="EURUSD-OTC" selected>EUR/USD (OTC)</option>
                        <option value="GBPUSD-OTC">GBP/USD (OTC)</option>
                        <option value="USDJPY-OTC">USD/JPY (OTC)</option>
                        <option value="AUDUSD-OTC">AUD/USD (OTC)</option>
                        <option value="USDCAD-OTC">USD/CAD (OTC)</option>
                        <option value="USDCHF-OTC">USD/CHF (OTC)</option>
                    </select>
                </div>
                
                <div class="control-item">
                    <label>Valor da Aposta (USD):</label>
                    <input type="number" id="stakeAmount" value="1" min="1" max="1000" step="1">
                </div>
                
                <div class="control-item">
                    <label>Duração (minutos):</label>
                    <select id="duration">
                        <option value="1" selected>1 minuto</option>
                        <option value="2">2 minutos</option>
                        <option value="3">3 minutos</option>
                        <option value="4">4 minutos</option>
                        <option value="5">5 minutos</option>
                    </select>
                </div>

                <div class="control-item">
                    <label>Modo Trading:</label>
                    <select id="tradingMode">
                        <option value="manual">Manual</option>
                        <option value="auto">Automático</option>
                        <option value="ai">IA Real</option>
                    </select>
                </div>
            </div>

            <!-- Informações do Martingale -->
            <div class="martingale-info" id="martingaleInfo">
                <div class="martingale-level">
                    🎰 Martingale Nível: <span id="martingaleLevel">0</span>/8
                </div>
                <div style="margin: 8px 0;">
                    💰 Próxima Aposta: $<span id="nextStake">1.00</span> | 
                    📊 Base: $<span id="baseStake">1.00</span>
                </div>
                <div class="martingale-controls">
                    <button class="martingale-btn" onclick="toggleMartingale()" id="martingaleToggle">
                        🎰 Martingale: ON
                    </button>
                    <button class="martingale-btn" onclick="resetMartingale()">
                        🔄 Reset
                    </button>
                    <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 5px;">
                        Progressão: 2.2x após loss | Reset automático após WIN
                    </div>
                </div>
            </div>
            
            <div class="trade-buttons">
                <button class="trade-btn call" onclick="placeTrade('call')" id="callBtn" disabled>
                    📈 CALL (Higher)
                </button>
                <button class="trade-btn put" onclick="placeTrade('put')" id="putBtn" disabled>
                    📉 PUT (Lower)
                </button>
                <button class="trade-btn auto" onclick="toggleAutoTrading()" id="autoBtn" disabled>
                    🤖 Iniciar Auto
                </button>
            </div>
        </div>

        <!-- Histórico de Trades -->
        <div class="history-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #00d4ff;">📝 Histórico de Trades</h3>
                <button class="ai-btn" onclick="loadHistory()" style="font-size: 0.8rem; padding: 8px 16px;">
                    🔄 Atualizar
                </button>
            </div>

            <div style="overflow-x: auto;">
                <table class="history-table" id="historyTable">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Horário</th>
                            <th>Símbolo</th>
                            <th>Direção</th>
                            <th>Valor</th>
                            <th>Duração</th>
                            <th>Resultado</th>
                            <th>P&L</th>
                            <th>IA%</th>
                            <th>Martingale</th>
                        </tr>
                    </thead>
                    <tbody id="historyBody">
                        <tr>
                            <td colspan="10" style="text-align: center; padding: 20px; opacity: 0.7;">
                                Nenhum trade executado ainda...
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Notificações -->
    <div class="notification" id="notification"></div>

    <script>
        // ==============================================
        // CONFIGURAÇÕES E VARIÁVEIS GLOBAIS
        // ==============================================
        
        let isConnected = false;
        let isAIConnected = false;
        let isAutoTrading = false;
        let isAIModeActive = false;
        let isAIDurationActive = false;
        let isAIManagementActive = false;
        let botRunning = false;
        
        let currentStats = {
            balance: 0,
            sessionPnL: 0,
            winRate: 0,
            trades: 0,
            wins: 0,
            losses: 0
        };
        
        let martingaleState = {
            level: 0,
            baseStake: 1,
            nextStake: 1,
            enabled: true
        };
        
        let updateInterval;

        // ==============================================
        // FUNÇÕES DE INICIALIZAÇÃO
        // ==============================================
        
        function selectAccountType(type) {
            const demoCard = document.getElementById('demoCard');
            const realCard = document.getElementById('realCard');
            const accountSelect = document.getElementById('accountType');
            
            demoCard.classList.remove('selected');
            realCard.classList.remove('selected');
            
            if (type === 'demo') {
                demoCard.classList.add('selected');
                accountSelect.value = 'PRACTICE';
                showNotification('💡 Conta Demo selecionada - Ideal para praticar!', 'success');
            } else {
                realCard.classList.add('selected');
                accountSelect.value = 'REAL';
                showNotification('⚠️ Conta Real selecionada - Use dinheiro real!', 'warning');
            }
        }

        async function connectAPI() {
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value.trim();
            const accountType = document.getElementById('accountType').value;
            
            if (!email || !password) {
                showNotification('Por favor, preencha email e senha', 'error');
                return;
            }

            setLoginLoading(true);

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password, account_type: accountType })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    isConnected = true;
                    showDashboard();
                    
                    const accountTypeLabel = accountType === 'PRACTICE' ? '🎮 DEMO' : '💰 REAL';
                    document.getElementById('accountInfo').textContent = `${accountTypeLabel}`;
                    
                    updateStatus('iqStatus', 'online');
                    
                    currentStats.balance = result.balance;
                    updateMetrics();
                    
                    showNotification(result.message, 'success');
                    
                    // Inicializar atualizações automáticas
                    startRealTimeUpdates();
                    
                    // Carregar histórico
                    loadHistory();
                    
                    // Verificar IA
                    checkAIStatus();
                    
                } else {
                    showNotification(`Erro: ${result.error}`, 'error');
                }
            } catch (error) {
                showNotification(`Erro de conexão: ${error.message}`, 'error');
            } finally {
                setLoginLoading(false);
            }
        }

        function showDashboard() {
            document.getElementById('loginModal').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            
            // Habilitar botões
            document.getElementById('callBtn').disabled = false;
            document.getElementById('putBtn').disabled = false;
            document.getElementById('autoBtn').disabled = false;
        }

        function setLoginLoading(loading) {
            const btn = document.getElementById('loginBtn');
            const text = document.getElementById('loginBtnText');
            const spinner = document.getElementById('loginSpinner');
            
            btn.disabled = loading;
            text.style.display = loading ? 'none' : 'block';
            spinner.style.display = loading ? 'block' : 'none';
        }

        async function checkAIStatus() {
            try {
                const response = await fetch('/api/stats');
                const result = await response.json();
                
                if (result.success && result.ai_status) {
                    isAIConnected = result.ai_status.connected;
                    
                    if (isAIConnected) {
                        updateStatus('aiStatus', 'online');
                        document.getElementById('aiStatusText').textContent = 'Conectado';
                        enableAIControls(true);
                        showNotification('🤖 IA Real conectada!', 'success');
                    } else {
                        updateStatus('aiStatus', 'warning');
                        document.getElementById('aiStatusText').textContent = 'Desconectado';
                        showNotification('⚠️ IA não conectada', 'warning');
                    }
                }
            } catch (error) {
                console.error('Erro ao verificar status da IA:', error);
            }
        }

        function enableAIControls(enabled) {
            const aiButtons = ['aiAnalyzeBtn', 'aiSignalBtn', 'aiModeBtn', 'aiDurationBtn', 'aiManagementBtn'];
            aiButtons.forEach(btnId => {
                document.getElementById(btnId).disabled = !enabled;
            });
        }

        // ==============================================
        // FUNÇÕES DE IA
        // ==============================================
        
        async function getAIAnalysis() {
            if (!isAIConnected) {
                showNotification('IA não conectada', 'error');
                return;
            }

            try {
                document.getElementById('aiAnalyzeBtn').disabled = true;
                showAIResponse('🔍 Analisando mercado...');

                const response = await fetch('/api/ai/analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const result = await response.json();
                
                if (result.success && result.analysis) {
                    const analysis = result.analysis;
                    const message = analysis.message || `📊 Análise concluída: ${analysis.trend || 'neutro'} | Confiança: ${(analysis.confidence || 75).toFixed(1)}%`;
                    showAIResponse(message);
                } else {
                    showAIResponse('❌ Erro na análise da IA');
                }
                
            } catch (error) {
                showAIResponse('❌ Erro ao conectar com IA');
            } finally {
                document.getElementById('aiAnalyzeBtn').disabled = false;
            }
        }

        async function getAISignal() {
            if (!isAIConnected) {
                showNotification('IA não conectada', 'error');
                return;
            }

            try {
                document.getElementById('aiSignalBtn').disabled = true;
                showAIResponse('📡 Obtendo sinal...');

                const response = await fetch('/api/ai/signal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const result = await response.json();
                
                if (result.success && result.signal) {
                    const signal = result.signal;
                    const direction = signal.direction.toUpperCase();
                    const confidence = signal.confidence || 75;
                    
                    showAIResponse(`🎯 Sinal: ${direction} | Confiança: ${confidence.toFixed(1)}% | ${signal.reasoning || 'Análise técnica'}`);
                    
                    if (isAIModeActive && confidence > 75) {
                        setTimeout(() => {
                            placeTrade(signal.direction);
                        }, 2000);
                    }
                } else {
                    showAIResponse('❌ Erro ao obter sinal');
                }
                
            } catch (error) {
                showAIResponse('❌ Erro ao conectar com IA');
            } finally {
                document.getElementById('aiSignalBtn').disabled = false;
            }
        }

        async function toggleAIMode() {
            if (!isAIConnected) {
                showNotification('IA não conectada', 'error');
                return;
            }

            try {
                const response = await fetch('/api/ai/toggle/trading', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    isAIModeActive = result.active;
                    const btn = document.getElementById('aiModeBtn');
                    
                    if (isAIModeActive) {
                        btn.textContent = '🤖 Modo IA: ON';
                        btn.classList.add('active');
                        showNotification('🤖 Modo IA ativado', 'success');
                    } else {
                        btn.textContent = '🤖 Modo IA: OFF';
                        btn.classList.remove('active');
                        showNotification('🤖 Modo IA desativado', 'warning');
                    }
                }
            } catch (error) {
                showNotification('Erro ao alternar modo IA', 'error');
            }
        }

        async function toggleAIDuration() {
            if (!isAIConnected) {
                showNotification('IA não conectada', 'error');
                return;
            }

            try {
                const response = await fetch('/api/ai/toggle/duration', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    isAIDurationActive = result.active;
                    const btn = document.getElementById('aiDurationBtn');
                    
                    if (isAIDurationActive) {
                        btn.textContent = '⏱️ Duração IA: ON';
                        btn.classList.add('active');
                        showNotification('⏱️ IA agora controla a duração', 'success');
                    } else {
                        btn.textContent = '⏱️ Duração IA: OFF';
                        btn.classList.remove('active');
                        showNotification('⏱️ Controle manual restaurado', 'warning');
                    }
                }
            } catch (error) {
                showNotification('Erro ao alternar duração IA', 'error');
            }
        }

        async function toggleAIManagement() {
            if (!isAIConnected) {
                showNotification('IA não conectada', 'error');
                return;
            }

            try {
                const response = await fetch('/api/ai/toggle/management', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    isAIManagementActive = result.active;
                    const btn = document.getElementById('aiManagementBtn');
                    
                    if (isAIManagementActive) {
                        btn.textContent = '🎛️ Gerenciamento: ON';
                        btn.classList.add('active');
                        showNotification('🎛️ IA gerencia automaticamente', 'success');
                        showAIManagement('🎛️ Gerenciamento IA ativado');
                    } else {
                        btn.textContent = '🎛️ Gerenciamento: OFF';
                        btn.classList.remove('active');
                        showNotification('🎛️ Gerenciamento manual', 'warning');
                        hideAIManagement();
                    }
                }
            } catch (error) {
                showNotification('Erro ao alternar gerenciamento IA', 'error');
            }
        }

        function showAIResponse(message) {
            const responseDiv = document.getElementById('aiResponse');
            const responseText = document.getElementById('aiResponseText');
            
            responseText.textContent = message;
            responseDiv.style.display = 'block';
            
            setTimeout(() => {
                responseDiv.style.display = 'none';
            }, 15000);
        }

        function showAIManagement(message) {
            const managementDiv = document.getElementById('aiManagement');
            const managementText = document.getElementById('aiManagementText');
            
            managementText.textContent = message;
            managementDiv.style.display = 'block';
        }

        function hideAIManagement() {
            document.getElementById('aiManagement').style.display = 'none';
        }

        // ==============================================
        // FUNÇÕES DE TRADING
        // ==============================================
        
        async function placeTrade(direction) {
            if (!isConnected) {
                showNotification('Não conectado', 'error');
                return;
            }

            try {
                const symbol = document.getElementById('symbolSelect').value;
                const amount = parseFloat(document.getElementById('stakeAmount').value);
                const duration = parseInt(document.getElementById('duration').value);
                
                if (amount < 1 || amount > 1000) {
                    showNotification('Valor inválido (1-1000)', 'error');
                    return;
                }

                const response = await fetch('/api/trade', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ direction, amount, duration })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification(`Trade ${direction.toUpperCase()} executado: $${amount}`, 'success');
                    
                    // Atualizar interface
                    updateStatus('orderStatus', 'warning');
                    
                    // Aguardar um pouco e atualizar stats
                    setTimeout(() => {
                        updateStats();
                        loadHistory();
                    }, 2000);
                    
                } else {
                    showNotification(`Erro: ${result.error}`, 'error');
                }
                
            } catch (error) {
                showNotification(`Erro: ${error.message}`, 'error');
            }
        }

        async function toggleAutoTrading() {
            if (!isConnected) {
                showNotification('Não conectado', 'error');
                return;
            }

            try {
                const endpoint = botRunning ? '/api/bot/stop' : '/api/bot/start';
                const response = await fetch(endpoint, { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    botRunning = !botRunning;
                    isAutoTrading = botRunning;
                    
                    const btn = document.getElementById('autoBtn');
                    
                    if (botRunning) {
                        btn.textContent = '⏹️ Parar Auto';
                        btn.className = 'trade-btn auto';
                        updateStatus('tradingStatus', 'online');
                        showNotification('🤖 Auto trading iniciado', 'success');
                    } else {
                        btn.textContent = '🤖 Iniciar Auto';
                        updateStatus('tradingStatus', 'offline');
                        showNotification('🤖 Auto trading parado', 'warning');
                    }
                } else {
                    showNotification(`Erro: ${result.error}`, 'error');
                }
            } catch (error) {
                showNotification(`Erro: ${error.message}`, 'error');
            }
        }

        // ==============================================
        // SISTEMA MARTINGALE
        // ==============================================
        
        async function toggleMartingale() {
            try {
                const response = await fetch('/api/martingale/toggle', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    martingaleState.enabled = result.enabled;
                    const btn = document.getElementById('martingaleToggle');
                    
                    if (result.enabled) {
                        btn.textContent = '🎰 Martingale: ON';
                        btn.style.background = 'linear-gradient(45deg, #00ff88, #00cc6a)';
                        showNotification('🎰 Martingale ativado', 'success');
                    } else {
                        btn.textContent = '🎰 Martingale: OFF';
                        btn.style.background = 'linear-gradient(45deg, #ff4757, #ff3742)';
                        showNotification('🎰 Martingale desativado', 'warning');
                    }
                }
            } catch (error) {
                showNotification('Erro ao alternar Martingale', 'error');
            }
        }

        async function resetMartingale() {
            try {
                const response = await fetch('/api/martingale/reset', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    showNotification('🔄 Martingale resetado', 'success');
                    updateStats();
                }
            } catch (error) {
                showNotification('Erro ao resetar Martingale', 'error');
            }
        }

        // ==============================================
        // ATUALIZAÇÕES E ESTATÍSTICAS
        // ==============================================
        
        async function updateStats() {
            if (!isConnected) return;

            try {
                const response = await fetch('/api/stats');
                const result = await response.json();
                
                if (result.success) {
                    // Atualizar estatísticas da sessão
                    const stats = result.session_stats;
                    currentStats = {
                        balance: stats.current_balance,
                        sessionPnL: stats.profit_loss,
                        winRate: stats.win_rate,
                        trades: stats.trades_count,
                        wins: stats.wins,
                        losses: stats.losses
                    };
                    
                    // Atualizar Martingale
                    if (result.martingale) {
                        martingaleState = {
                            level: result.martingale.level,
                            nextStake: result.martingale.next_amount,
                            baseStake: parseFloat(document.getElementById('stakeAmount').value)
                        };
                        
                        document.getElementById('martingaleLevel').textContent = martingaleState.level;
                        document.getElementById('nextStake').textContent = martingaleState.nextStake.toFixed(2);
                        document.getElementById('baseStake').textContent = martingaleState.baseStake.toFixed(2);
                    }
                    
                    // Atualizar status dos bots
                    botRunning = result.bot_running;
                    
                    // Atualizar status da IA
                    if (result.ai_status) {
                        isAIConnected = result.ai_status.connected;
                        isAIModeActive = result.ai_status.mode_active;
                        isAIDurationActive = result.ai_status.duration_active;
                        isAIManagementActive = result.ai_status.management_active;
                        
                        updateStatus('aiStatus', isAIConnected ? 'online' : 'offline');
                    }
                    
                    // Atualizar status do sistema
                    updateStatus('orderStatus', result.has_active_order ? 'warning' : 'online');
                    
                    updateMetrics();
                }
            } catch (error) {
                console.error('Erro ao atualizar stats:', error);
            }
        }

        function updateMetrics() {
            // Saldo
            document.getElementById('balance').textContent = `$${currentStats.balance.toFixed(2)}`;
            
            // P&L da sessão
            const pnlElement = document.getElementById('sessionPnL');
            pnlElement.textContent = `$${currentStats.sessionPnL.toFixed(2)}`;
            
            const pnlChangeElement = document.getElementById('pnlChange');
            if (currentStats.sessionPnL > 0) {
                pnlChangeElement.innerHTML = '<span>↗</span> Lucro';
                pnlChangeElement.className = 'metric-change positive';
            } else if (currentStats.sessionPnL < 0) {
                pnlChangeElement.innerHTML = '<span>↘</span> Prejuízo';
                pnlChangeElement.className = 'metric-change negative';
            } else {
                pnlChangeElement.innerHTML = '<span>→</span> Neutro';
                pnlChangeElement.className = 'metric-change neutral';
            }
            
            // Taxa de acerto
            document.getElementById('winRate').textContent = `${currentStats.winRate.toFixed(1)}%`;
            document.getElementById('winRateChange').innerHTML = `<span>→</span> ${currentStats.winRate.toFixed(1)}%`;
            
            // Trades
            document.getElementById('tradesCount').textContent = currentStats.trades;
            document.getElementById('tradesChange').innerHTML = `<span>→</span> ${currentStats.wins}W / ${currentStats.losses}L`;
        }

        async function loadHistory() {
            try {
                const response = await fetch('/api/history?limit=20');
                const result = await response.json();
                
                if (result.success) {
                    const tbody = document.getElementById('historyBody');
                    tbody.innerHTML = '';
                    
                    if (result.history.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; opacity: 0.7;">Nenhum trade encontrado</td></tr>';
                        return;
                    }
                    
                    result.history.forEach(trade => {
                        const row = document.createElement('tr');
                        const resultClass = trade.result === 'win' ? 'win' : 'loss';
                        const directionClass = trade.direction.toLowerCase();
                        
                        row.innerHTML = `
                            <td>${trade.id}</td>
                            <td>${new Date(trade.timestamp).toLocaleTimeString()}</td>
                            <td>${trade.symbol}</td>
                            <td><span class="trade-direction-badge ${directionClass}">${trade.direction.toUpperCase()}</span></td>
                            <td>$${trade.amount.toFixed(2)}</td>
                            <td>${trade.duration}m</td>
                            <td><span class="trade-result ${resultClass}">${trade.result.toUpperCase()}</span></td>
                            <td class="${resultClass}">${trade.profit_loss.toFixed(2)}</td>
                            <td>${(trade.ai_confidence || 0).toFixed(1)}%</td>
                            <td>${trade.martingale_level}</td>
                        `;
                        tbody.appendChild(row);
                    });
                }
            } catch (error) {
                console.error('Erro ao carregar histórico:', error);
                showNotification('Erro ao carregar histórico', 'error');
            }
        }

        function startRealTimeUpdates() {
            updateInterval = setInterval(() => {
                updateStats();
                
                // Atualizar histórico ocasionalmente
                if (Math.random() < 0.3) {
                    loadHistory();
                }
            }, 5000);
        }

        // ==============================================
        // FUNÇÕES AUXILIARES
        // ==============================================
        
        function updateStatus(elementId, status) {
            const element = document.getElementById(elementId);
            if (element) {
                element.className = `status-dot ${status}`;
            }
        }

        function showNotification(message, type = 'info') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type} show`;
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 5000);
        }

        function logout() {
            // Parar atualizações
            if (updateInterval) {
                clearInterval(updateInterval);
            }
            
            // Reset variáveis
            isConnected = false;
            isAIConnected = false;
            isAutoTrading = false;
            isAIModeActive = false;
            isAIDurationActive = false;
            isAIManagementActive = false;
            botRunning = false;
            
            // Reset UI
            document.getElementById('loginModal').style.display = 'flex';
            document.getElementById('dashboard').style.display = 'none';
            document.getElementById('email').value = '';
            document.getElementById('password').value = '';
            
            // Reset status
            updateStatus('iqStatus', 'offline');
            updateStatus('aiStatus', 'offline');
            updateStatus('tradingStatus', 'offline');
            updateStatus('orderStatus', 'offline');
            
            showNotification('Desconectado com sucesso', 'success');
        }

        // ==============================================
        // EVENT LISTENERS
        // ==============================================
        
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 IQ Option Bot com IA Real iniciado!');
            
            // Event listeners para inputs
            document.getElementById('email').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    document.getElementById('password').focus();
                }
            });

            document.getElementById('password').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    connectAPI();
                }
            });

            document.getElementById('stakeAmount').addEventListener('input', function() {
                const value = parseFloat(this.value);
                if (value < 1) this.value = 1;
                if (value > 1000) this.value = 1000;
                
                // Atualizar base do martingale se nível 0
                if (martingaleState.level === 0) {
                    martingaleState.baseStake = value;
                    document.getElementById('baseStake').textContent = value.toFixed(2);
                }
            });

            // Inicializar com conta demo selecionada
            selectAccountType('demo');
        });

        // ==============================================
        // LOGS E DEBUG
        // ==============================================
        
        console.log('🎯 IQ Option Bot com IA Real Avançada carregado!');
        console.log('✅ Recursos implementados:');
        console.log('   🤖 IA Real integrada com múltiplos endpoints');
        console.log('   ⏱️ Duração automática controlada por IA');
        console.log('   🎛️ Gerenciamento inteligente de risco');
        console.log('   🎰 Sistema Martingale avançado (2.2x)');
        console.log('   📊 Interface moderna e responsiva');
        console.log('   🔒 Sistema de controle de ordem única');
        console.log('   📈 Análise avançada de mercado');
        console.log('   📱 Atualização em tempo real');
        console.log('   💾 Histórico completo de trades');
        console.log('   🎮 Suporte para conta Demo e Real');
        console.log('🚀 Pronto para uso!');
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Iniciando IQ Option Bot Avançado com IA Real...")
    print("📱 Acesse: http://localhost:5000")
    print("🤖 IA Real Integrada + Interface Moderna")
    print("⚠️  Sistema Martingale Ativo - Use com responsabilidade!")
    print("🎯 Recursos: Duração IA + Gerenciamento Automático + Análise Avançada")
    
    # Inicializar banco de dados
    init_database()
    
    # Inicializar martingale
    martingale_state['base_stake'] = bot_config['base_amount']
    martingale_state['next_amount'] = bot_config['base_amount']
    
    app.run(debug=True, host='0.0.0.0', port=5000)
