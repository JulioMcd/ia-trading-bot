from flask import Flask, request, jsonify
from ml_engine import ml_engine
import json
from datetime import datetime
import threading
import time

class MLTradingAPI:
    """
    🤖 API de Machine Learning para Trading Bot
    - Endpoints para predições ML
    - Aprendizado contínuo
    - Otimização automática
    - Integração com trading bot
    """
    
    def __init__(self, app):
        self.app = app
        self.ml_engine = ml_engine
        self.setup_routes()
        
        # Background learning thread
        self.learning_thread = None
        self.learning_active = True
        self.start_background_learning()
        
        print("🤖 ML Trading API inicializada")
    
    def setup_routes(self):
        """🛣️ Configura rotas da API ML"""
        
        @self.app.route('/api/ml/predict', methods=['POST'])
        def ml_predict():
            """🎯 Predição ML de direção"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Dados de mercado necessários'}), 400
                
                # Fazer predição
                prediction = self.ml_engine.predict_direction(data)
                
                # Log da predição
                print(f"🎯 ML Predição: {prediction['direction']} ({prediction['confidence']:.1f}%)")
                
                return jsonify(prediction)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro na predição ML',
                    'details': str(e),
                    'fallback': {
                        'direction': 'CALL',
                        'confidence': 65.0,
                        'reasoning': 'Predição de fallback devido a erro'
                    }
                }), 500
        
        @self.app.route('/api/ml/risk', methods=['POST'])
        def ml_risk_analysis():
            """⚠️ Análise de risco ML"""
            try:
                data = request.get_json()
                
                market_data = data.get('market_data', {})
                trade_history = data.get('trade_history', [])
                
                # Análise de risco
                risk_analysis = self.ml_engine.analyze_risk(market_data, trade_history)
                
                print(f"⚠️ ML Risco: {risk_analysis['level']} (Score: {risk_analysis['score']:.1f})")
                
                return jsonify(risk_analysis)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro na análise de risco ML',
                    'details': str(e),
                    'fallback': {
                        'level': 'medium',
                        'score': 50.0,
                        'message': 'Análise de fallback',
                        'recommendation': 'Operar com cautela'
                    }
                }), 500
        
        @self.app.route('/api/ml/martingale', methods=['POST'])
        def ml_martingale_optimization():
            """🎰 Otimização Martingale ML"""
            try:
                data = request.get_json()
                
                current_level = data.get('current_level', 0)
                recent_performance = data.get('recent_performance', {})
                
                # Otimização Martingale
                optimization = self.ml_engine.optimize_martingale(current_level, recent_performance)
                
                print(f"🎰 ML Martingale: {optimization['action']} (Nível {optimization['recommended_level']})")
                
                return jsonify(optimization)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro na otimização Martingale ML',
                    'details': str(e),
                    'fallback': {
                        'action': 'continue' if current_level < 4 else 'pause',
                        'recommended_level': min(current_level + 1, 8),
                        'reasoning': 'Recomendação de fallback'
                    }
                }), 500
        
        @self.app.route('/api/ml/analysis', methods=['POST'])
        def ml_market_analysis():
            """📈 Análise completa de mercado ML"""
            try:
                data = request.get_json()
                
                symbol = data.get('symbol', 'R_50')
                timeframe_data = data.get('timeframe_data', [])
                
                # Análise completa
                analysis = self.ml_engine.get_market_analysis(symbol, timeframe_data)
                
                print(f"📈 ML Análise: {symbol} - {analysis['trend']}")
                
                return jsonify(analysis)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro na análise de mercado ML',
                    'details': str(e),
                    'fallback': {
                        'symbol': symbol,
                        'trend': 'neutral',
                        'volatility': 50.0,
                        'recommendation': 'hold'
                    }
                }), 500
        
        @self.app.route('/api/ml/learn', methods=['POST'])
        def ml_learn_from_trade():
            """📊 Aprendizado a partir de trade"""
            try:
                trade_data = request.get_json()
                
                if not trade_data:
                    return jsonify({'error': 'Dados de trade necessários'}), 400
                
                # Adicionar trade para aprendizado
                success = self.ml_engine.add_trade_data(trade_data)
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': 'Trade adicionado para aprendizado',
                        'total_trades': len(self.ml_engine.historical_data),
                        'model_accuracy': self.ml_engine.metrics['accuracy']
                    })
                else:
                    return jsonify({'error': 'Falha ao adicionar trade'}), 500
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro no aprendizado ML',
                    'details': str(e)
                }), 500
        
        @self.app.route('/api/ml/performance', methods=['POST'])
        def ml_update_performance():
            """📊 Atualizar performance do modelo"""
            try:
                data = request.get_json()
                
                prediction_id = data.get('prediction_id')
                actual_result = data.get('actual_result')
                
                if not prediction_id or actual_result is None:
                    return jsonify({'error': 'prediction_id e actual_result necessários'}), 400
                
                # Atualizar performance
                success = self.ml_engine.update_performance(prediction_id, actual_result)
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': 'Performance atualizada',
                        'current_accuracy': self.ml_engine.metrics['accuracy'],
                        'total_predictions': self.ml_engine.metrics['total_predictions']
                    })
                else:
                    return jsonify({'error': 'Falha ao atualizar performance'}), 500
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro ao atualizar performance',
                    'details': str(e)
                }), 500
        
        @self.app.route('/api/ml/stats', methods=['GET'])
        def ml_statistics():
            """📊 Estatísticas do ML"""
            try:
                stats = {
                    'model_metrics': self.ml_engine.metrics,
                    'training_data_size': len(self.ml_engine.historical_data),
                    'models_status': {
                        name: 'trained' if model is not None else 'not_trained'
                        for name, model in self.ml_engine.models.items()
                    },
                    'last_predictions': self.ml_engine.performance_history[-10:] if self.ml_engine.performance_history else [],
                    'config': self.ml_engine.config,
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(stats)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro ao obter estatísticas ML',
                    'details': str(e)
                }), 500
        
        @self.app.route('/api/ml/retrain', methods=['POST'])
        def ml_force_retrain():
            """🔄 Forçar retreinamento dos modelos"""
            try:
                data = request.get_json() or {}
                force = data.get('force', False)
                
                if not force and len(self.ml_engine.historical_data) < self.ml_engine.config['min_samples_for_training']:
                    return jsonify({
                        'error': 'Dados insuficientes para retreinamento',
                        'current_samples': len(self.ml_engine.historical_data),
                        'required_samples': self.ml_engine.config['min_samples_for_training']
                    }), 400
                
                # Executar retreinamento em thread separada
                def retrain_models():
                    self.ml_engine._retrain_models()
                
                retrain_thread = threading.Thread(target=retrain_models)
                retrain_thread.start()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Retreinamento iniciado',
                    'training_data_size': len(self.ml_engine.historical_data)
                })
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro ao iniciar retreinamento',
                    'details': str(e)
                }), 500
        
        @self.app.route('/api/ml/config', methods=['GET', 'POST'])
        def ml_configuration():
            """⚙️ Configuração do ML"""
            try:
                if request.method == 'GET':
                    return jsonify(self.ml_engine.config)
                
                elif request.method == 'POST':
                    new_config = request.get_json()
                    
                    if not new_config:
                        return jsonify({'error': 'Configuração necessária'}), 400
                    
                    # Atualizar configuração
                    for key, value in new_config.items():
                        if key in self.ml_engine.config:
                            self.ml_engine.config[key] = value
                    
                    return jsonify({
                        'status': 'success',
                        'message': 'Configuração atualizada',
                        'config': self.ml_engine.config
                    })
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro na configuração ML',
                    'details': str(e)
                }), 500
        
        @self.app.route('/api/ml/strategy', methods=['POST'])
        def ml_strategy_recommendation():
            """🎯 Recomendação de estratégia ML"""
            try:
                data = request.get_json()
                
                market_data = data.get('market_data', {})
                current_performance = data.get('current_performance', {})
                risk_tolerance = data.get('risk_tolerance', 'medium')
                
                # Gerar recomendação de estratégia
                strategy = self._generate_strategy_recommendation(
                    market_data, current_performance, risk_tolerance
                )
                
                return jsonify(strategy)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro na recomendação de estratégia',
                    'details': str(e)
                }), 500
        
        @self.app.route('/api/ml/backtest', methods=['POST'])
        def ml_backtest():
            """📈 Backtest de estratégia ML"""
            try:
                data = request.get_json()
                
                historical_data = data.get('historical_data', [])
                strategy_params = data.get('strategy_params', {})
                
                if len(historical_data) < 10:
                    return jsonify({'error': 'Dados históricos insuficientes'}), 400
                
                # Executar backtest
                backtest_results = self._run_backtest(historical_data, strategy_params)
                
                return jsonify(backtest_results)
                
            except Exception as e:
                return jsonify({
                    'error': 'Erro no backtest',
                    'details': str(e)
                }), 500
    
    def start_background_learning(self):
        """🔄 Inicia aprendizado em background"""
        def learning_loop():
            while self.learning_active:
                try:
                    # Verificar se precisa salvar modelos
                    if len(self.ml_engine.historical_data) % 50 == 0 and len(self.ml_engine.historical_data) > 0:
                        self.ml_engine.save_models()
                    
                    # Verificar se precisa retreinar
                    if len(self.ml_engine.historical_data) % self.ml_engine.config['retrain_frequency'] == 0:
                        print("🔄 Iniciando retreinamento automático...")
                        self.ml_engine._retrain_models()
                    
                    time.sleep(60)  # Verificar a cada minuto
                    
                except Exception as e:
                    print(f"❌ Erro no learning loop: {e}")
                    time.sleep(60)
        
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
        print("🔄 Background learning iniciado")
    
    def stop_background_learning(self):
        """⏹️ Para aprendizado em background"""
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        print("⏹️ Background learning parado")
    
    def _generate_strategy_recommendation(self, market_data, performance, risk_tolerance):
        """🎯 Gera recomendação de estratégia"""
        try:
            # Obter predições do ML
            direction_pred = self.ml_engine.predict_direction(market_data)
            risk_analysis = self.ml_engine.analyze_risk(market_data, [])
            
            # Determinar estratégia baseada em ML + risco
            strategy = {
                'recommended_action': direction_pred['direction'],
                'confidence': direction_pred['confidence'],
                'risk_level': risk_analysis['level'],
                'suggested_stake': self._calculate_optimal_stake(market_data, risk_tolerance),
                'suggested_duration': self._calculate_optimal_duration(market_data),
                'martingale_recommendation': 'enabled' if risk_analysis['level'] != 'high' else 'disabled',
                'timing': 'immediate' if direction_pred['confidence'] > 80 else 'wait_for_better_signal',
                'reasoning': f"ML predição {direction_pred['direction']} com {direction_pred['confidence']:.1f}% confiança. Risco {risk_analysis['level']}.",
                'expected_win_rate': min(85, direction_pred['confidence']),
                'stop_conditions': self._generate_stop_conditions(risk_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            return strategy
            
        except Exception as e:
            return {
                'error': 'Erro ao gerar estratégia',
                'details': str(e),
                'fallback_strategy': {
                    'recommended_action': 'CALL',
                    'confidence': 60.0,
                    'risk_level': 'medium'
                }
            }
    
    def _calculate_optimal_stake(self, market_data, risk_tolerance):
        """💰 Calcula stake ótimo"""
        base_stake = 1.0
        
        # Ajustar baseado no risco
        risk_multipliers = {'low': 1.2, 'medium': 1.0, 'high': 0.7}
        risk_level = market_data.get('riskLevel', 'medium')
        
        # Ajustar baseado no Martingale
        martingale_level = market_data.get('martingaleLevel', 0)
        if martingale_level > 0:
            base_stake *= (2 ** martingale_level)
        
        # Aplicar multiplicador de risco
        optimal_stake = base_stake * risk_multipliers.get(risk_level, 1.0)
        
        # Limitar entre min e max
        return max(0.35, min(optimal_stake, 50.0))
    
    def _calculate_optimal_duration(self, market_data):
        """⏱️ Calcula duração ótima"""
        volatility = market_data.get('volatility', 50)
        
        # Alta volatilidade = menor duração
        if volatility > 70:
            return {'type': 't', 'value': 3, 'text': '3 ticks'}
        elif volatility > 50:
            return {'type': 't', 'value': 5, 'text': '5 ticks'}
        else:
            return {'type': 'm', 'value': 1, 'text': '1 minuto'}
    
    def _generate_stop_conditions(self, risk_analysis):
        """🛑 Gera condições de parada"""
        conditions = []
        
        if risk_analysis['level'] == 'high':
            conditions.append('Parar se Martingale > nível 3')
            conditions.append('Parar se perda > 10% do saldo')
        
        conditions.append('Parar se taxa de acerto < 30%')
        conditions.append('Parar se 5 perdas consecutivas')
        
        return conditions
    
    def _run_backtest(self, historical_data, strategy_params):
        """📈 Executa backtest"""
        try:
            results = {
                'total_trades': len(historical_data),
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
                'trades_detail': []
            }
            
            # Simular trades baseado na estratégia
            balance = 1000.0
            peak_balance = balance
            
            for i, trade_data in enumerate(historical_data):
                # Simular predição ML para este ponto histórico
                prediction = self.ml_engine.predict_direction(trade_data)
                
                # Simular resultado baseado na predição vs realidade
                actual_direction = trade_data.get('actual_direction', 'CALL')
                predicted_direction = prediction['direction']
                
                is_win = (predicted_direction == actual_direction)
                stake = strategy_params.get('stake', 1.0)
                
                if is_win:
                    pnl = stake * 0.8  # 80% payout
                    results['winning_trades'] += 1
                else:
                    pnl = -stake
                    results['losing_trades'] += 1
                
                balance += pnl
                results['total_pnl'] += pnl
                
                # Atualizar drawdown
                if balance > peak_balance:
                    peak_balance = balance
                
                current_drawdown = (peak_balance - balance) / peak_balance * 100
                results['max_drawdown'] = max(results['max_drawdown'], current_drawdown)
                
                # Adicionar detalhes do trade
                results['trades_detail'].append({
                    'trade_id': i,
                    'prediction': predicted_direction,
                    'actual': actual_direction,
                    'result': 'win' if is_win else 'loss',
                    'pnl': pnl,
                    'balance': balance
                })
            
            # Calcular métricas finais
            total_trades = results['winning_trades'] + results['losing_trades']
            results['win_rate'] = (results['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
            
            # Limitar detalhes para não sobrecarregar resposta
            results['trades_detail'] = results['trades_detail'][-50:]  # Últimos 50 trades
            
            results['final_balance'] = balance
            results['roi'] = ((balance - 1000) / 1000 * 100)
            
            return results
            
        except Exception as e:
            return {
                'error': 'Erro no backtest',
                'details': str(e)
            }

# Função para integrar com app Flask existente
def setup_ml_api(app):
    """🔧 Configura API ML no app Flask"""
    ml_api = MLTradingAPI(app)
    return ml_api

if __name__ == "__main__":
    # Teste standalone
    from flask import Flask
    
    app = Flask(__name__)
    ml_api = setup_ml_api(app)
    
    print("🤖 ML API Test Server")
    app.run(debug=True, port=5001)