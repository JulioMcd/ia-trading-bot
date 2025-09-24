from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from trading_manager import TradingManager

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização da aplicação Flask
app = Flask(__name__)
CORS(app)

# Inicialização do TradingManager
trading_manager = TradingManager()

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Trading API is running',
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_market():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        if not symbol:
            return jsonify({
                'error': 'Symbol é obrigatório',
                'can_trade': False
            }), 400
            
        confidence = data.get('confidence', 75)
        volatility = data.get('volatility', 1.0)
        timeframe = data.get('timeframe', 0)  # Timeframe em ticks

        # Verifica se timeframe é adequado
        if timeframe < trading_manager.min_timeframe:
            return jsonify({
                'error': f'Timeframe muito curto. Mínimo requerido: {trading_manager.min_timeframe} ticks',
                'can_trade': False
            }), 400

        # Calcula tamanho da posição
        position_size = trading_manager.calculate_position_size(confidence, volatility)

        # Obtem dados adicionais para análise
        signals = data.get('signals', [])
        price_data = data.get('price_data', [])
        candles = data.get('candles', [])
        
        # Verifica condições de mercado
        is_good_entry, signal_strength, message = trading_manager.analyze_market_conditions(
            signals, 
            timeframe=timeframe,
            last_trade_time=trading_manager.last_trade_time,
            price_data=price_data,
            candles=candles)

        # Calcula tempo desde última operação
        time_since_last_trade = None
        if trading_manager.last_trade_time:
            time_since_last_trade = (datetime.now() - trading_manager.last_trade_time).total_seconds()

        return jsonify({
            'symbol': symbol,
            'recommended_stake': position_size,
            'can_trade': trading_manager.can_trade() and is_good_entry,
            'is_good_entry': is_good_entry,
            'signal_strength': signal_strength,
            'message': message,
            'time_since_last_trade': time_since_last_trade,
            'min_time_between_trades': trading_manager.min_time_between_trades,
            'trading_stats': trading_manager.get_trading_stats()
        })
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

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = trading_manager.get_trading_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/risk-assessment', methods=['POST'])
def assess_risk():
    try:
        data = request.get_json()
        confidence = data.get('confidence', 75)
        volatility = data.get('volatility', 1.0)
        
        position_size = trading_manager.calculate_position_size(confidence, volatility)
        can_trade = trading_manager.can_trade()
        stats = trading_manager.get_trading_stats()
        
        risk_level = 'high' if stats['losses'] > 3 else 'medium' if stats['losses'] > 1 else 'low'
        
        return jsonify({
            'risk_level': risk_level,
            'recommended_stake': position_size,
            'can_trade': can_trade,
            'martingale_level': stats['martingale_level'],
            'message': f"Risk assessment completed. Current martingale level: {stats['martingale_level']}"
        })
    except Exception as e:
        logger.error(f"Error in assess_risk: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
