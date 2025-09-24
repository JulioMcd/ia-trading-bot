from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import sqlite3
import pandas as pd
from trading_manager import TradingManager

# Configurações
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'trading_stats_online.db')
PORT = int(os.getenv('PORT', 5000))

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização das instâncias na ordem correta
trading_manager = TradingManager()
app = Flask(__name__)
CORS(app)

# Importar e inicializar EnhancedTradingAI após criar TradingManager
from TRADING_AI import EnhancedTradingAI
trading_ai = EnhancedTradingAI(trading_manager=trading_manager)

@app.route('/')
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Enhanced Deriv Trading AI',
        'version': '4.0.0',
        'timestamp': datetime.now().isoformat(),
        'supported_symbols': list(trading_ai.deriv_manager.symbol_configs.keys()),
        'supported_timeframes': {
            'ticks': list(trading_ai.deriv_manager.timeframe_mapping['t'].keys()),
            'minutes': list(trading_ai.deriv_manager.timeframe_mapping['m'].keys())
        },
        'full_ai_mode': trading_ai.full_ai_mode
    })

@app.route('/analyze', methods=['POST'])
def analyze_symbol():
    """Análise específica para símbolo e timeframe do usuário"""
    try:
        data = request.get_json()
        
        # Parâmetros obrigatórios
        symbol = data.get('symbol', 'R_50')
        duration_type = data.get('duration_type', 't')
        duration_value = int(data.get('duration', 5))
        
        # Parâmetros opcionais
        stake = data.get('stake')
        ai_mode = data.get('ai_mode', False)
        
        # Validar entrada
        if not symbol or not duration_type or not duration_value:
            return jsonify({
                'error': 'Parâmetros obrigatórios: symbol, duration_type, duration',
                'example': {
                    'symbol': 'R_50',
                    'duration_type': 't',
                    'duration': 5,
                    'stake': 1.0
                }
            }), 400
        
        # Fazer análise
        result = trading_ai.predict_with_user_config(
            symbol, duration_type, duration_value, stake, ai_mode
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in analyze_market: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process-trade', methods=['POST'])
def process_trade():
    try:
        data = request.get_json()
        result = data.get('result')
        stake = data.get('stake')
        pnl = data.get('pnl')

        # Processa resultado do trade
        trade_stats = trading_manager.process_trade_result(result, stake, pnl)

        return jsonify({
            'status': 'success',
            'trade_processed': True,
            'stats': trade_stats
        })
    except Exception as e:
        logger.error(f"Error in process_trade: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/full-ai-analysis', methods=['POST'])
def full_ai_analysis():
    """Análise completa automatizada da IA"""
    try:
        data = request.get_json() or {}
        preferred_symbols = data.get('preferred_symbols')
        
        result = trading_ai.full_ai_analysis(preferred_symbols)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na análise completa: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/symbol-info/<symbol>', methods=['GET'])
def get_symbol_info(symbol):
    """Informações detalhadas sobre um símbolo"""
    try:
        config = trading_ai.deriv_manager.get_symbol_config(symbol)
        market_data = trading_ai.get_market_data_for_symbol(symbol)
        
        return jsonify({
            'symbol': symbol,
            'config': config,
            'current_market_data': market_data,
            'symbol_type': trading_ai.get_symbol_type(symbol),
            'recommended_timeframes': trading_ai.get_recommended_timeframes(symbol, config.get('volatility', 1.0)),
            'optimal_timeframe': trading_ai.deriv_manager.get_optimal_timeframe(symbol, 75, market_data)
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter info do símbolo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/market-overview', methods=['GET'])
def market_overview():
    """Overview geral do mercado para todos os símbolos"""
    try:
        symbols = list(trading_ai.deriv_manager.symbol_configs.keys())[:10]  # Limitar a 10
        analyses = []
        
        for symbol in symbols:
            try:
                market_data = trading_ai.get_market_data_for_symbol(symbol)
                optimal_tf = trading_ai.deriv_manager.get_optimal_timeframe(symbol, 70, market_data)
                
                quick_analysis = {
                    'symbol': symbol,
                    'price': market_data['price'],
                    'trend': market_data['trend'],
                    'volatility': market_data['volatility'],
                    'rsi': market_data['rsi'],
                    'optimal_timeframe': f"{optimal_tf['duration']}{optimal_tf['type']}"
                }
                analyses.append(quick_analysis)
                
            except Exception as e:
                logger.error(f"Erro ao analisar {symbol}: {e}")
                continue
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(analyses),
            'market_data': analyses,
            'overall_sentiment': trading_ai.generate_market_overview([{'market_data': a, 'direction': 'CALL'} for a in analyses])
        })
        
    except Exception as e:
        logger.error(f"Erro no overview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Estatísticas do sistema"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        
        # Estatísticas básicas
        trades_df = pd.read_sql_query('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100', conn)
        
        if len(trades_df) == 0:
            return jsonify({
                'total_trades': 0,
                'message': 'Nenhum trade registrado ainda'
            })
        
        wins = len(trades_df[trades_df['result'] == 'win'])
        losses = len(trades_df[trades_df['result'] == 'loss'])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Estatísticas por símbolo
        symbol_stats = {}
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            symbol_wins = len(symbol_trades[symbol_trades['result'] == 'win'])
            symbol_stats[symbol] = {
                'total': len(symbol_trades),
                'wins': symbol_wins,
                'win_rate': symbol_wins / len(symbol_trades) if len(symbol_trades) > 0 else 0
            }
        
        # Estatísticas por timeframe
        timeframe_stats = {}
        for tf in trades_df['timeframe_used'].unique():
            if tf:
                tf_trades = trades_df[trades_df['timeframe_used'] == tf]
                tf_wins = len(tf_trades[tf_trades['result'] == 'win'])
                timeframe_stats[tf] = {
                    'total': len(tf_trades),
                    'wins': tf_wins,
                    'win_rate': tf_wins / len(tf_trades) if len(tf_trades) > 0 else 0
                }
        
        conn.close()
        
        return jsonify({
            'total_trades': len(trades_df),
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate * 100, 2),
            'symbol_performance': symbol_stats,
            'timeframe_performance': timeframe_stats,
            'recent_trades': trades_df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Erro nas estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/report-trade', methods=['POST'])
def report_trade():
    """Reporta resultado de trade"""
    try:
        data = request.get_json()
        
        # Validação
        required = ['symbol', 'direction', 'result', 'entry_price', 'stake']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Campo obrigatório: {field}'}), 400
        
        # Processar resultado no gerenciador de trading
        pnl = data.get('pnl', 0)
        trading_stats = trading_ai.trading_manager.process_trade_result(
            data['result'],
            data['stake'],
            pnl
        )
        
        # Salvar no banco
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, direction, stake, duration_type, duration_value,
             entry_price, exit_price, result, pnl, confidence, timeframe_used, ai_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            data['symbol'],
            data['direction'],
            data['stake'],
            data.get('duration_type', 't'),
            data.get('duration_value', 5),
            data['entry_price'],
            data.get('exit_price', data['entry_price']),
            data['result'],
            pnl,
            data.get('confidence', 0),
            data.get('timeframe_used', '5t'),
            data.get('ai_mode', False)
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Trade reportado com sucesso',
            'trade_id': cursor.lastrowid,
            'trading_stats': trading_stats,
            'next_trade': {
                'can_trade': trading_stats['can_trade'],
                'martingale_level': trading_stats['martingale_level'],
                'recommended_stake': trading_ai.trading_manager.calculate_position_size(
                    data.get('confidence', 70),
                    data.get('volatility', 1.0)
                ) if trading_stats['can_trade'] else 0.0
            }
        })
        
    except Exception as e:
        logger.error(f"Erro ao reportar trade: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)