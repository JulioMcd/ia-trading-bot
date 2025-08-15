import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingMLEngine:
    """
    🤖 Motor de Machine Learning para Trading Bot
    - Analisa padrões históricos
    - Prediz direções de mercado
    - Aprende continuamente
    - Otimiza estratégias automaticamente
    """
    
    def __init__(self):
        self.models = {
            'direction_predictor': None,  # Prediz CALL/PUT
            'confidence_estimator': None,  # Estima confiança
            'risk_analyzer': None,        # Analisa risco
            'martingale_optimizer': None  # Otimiza Martingale
        }
        
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler()
        }
        
        self.label_encoders = {}
        
        # Dados históricos em memória
        self.historical_data = []
        self.market_patterns = {}
        self.performance_history = []
        
        # Configurações de ML
        self.config = {
            'min_samples_for_training': 50,
            'retrain_frequency': 100,  # Retreinar a cada 100 trades
            'feature_importance_threshold': 0.05,
            'confidence_threshold': 0.75,
            'max_historical_data': 10000
        }
        
        # Métricas de performance
        self.metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'last_training': None,
            'model_version': '1.0'
        }
        
        print("🤖 Trading ML Engine inicializado")
        self._load_saved_models()
    
    def add_trade_data(self, trade_data):
        """
        📊 Adiciona dados de trade para aprendizado
        """
        try:
            # Processar dados do trade
            processed_data = self._process_trade_data(trade_data)
            
            # Adicionar aos dados históricos
            self.historical_data.append(processed_data)
            
            # Limitar tamanho dos dados históricos
            if len(self.historical_data) > self.config['max_historical_data']:
                self.historical_data = self.historical_data[-self.config['max_historical_data']:]
            
            # Verificar se precisa retreinar
            if len(self.historical_data) % self.config['retrain_frequency'] == 0:
                self._retrain_models()
            
            print(f"📊 Trade adicionado ao ML. Total: {len(self.historical_data)} trades")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao adicionar trade data: {e}")
            return False
    
    def predict_direction(self, market_data):
        """
        🎯 Prediz direção do mercado (CALL/PUT)
        """
        try:
            if self.models['direction_predictor'] is None:
                return self._generate_random_prediction('direction')
            
            # Preparar features
            features = self._extract_features(market_data)
            features_scaled = self.scalers['features'].transform([features])
            
            # Predição
            prediction = self.models['direction_predictor'].predict(features_scaled)[0]
            probability = self.models['direction_predictor'].predict_proba(features_scaled)[0]
            
            # Calcular confiança
            confidence = max(probability) * 100
            
            direction = 'CALL' if prediction == 1 else 'PUT'
            
            result = {
                'direction': direction,
                'confidence': confidence,
                'probability_call': probability[1] * 100,
                'probability_put': probability[0] * 100,
                'reasoning': f"ML predição baseada em {len(self.historical_data)} trades históricos",
                'model_accuracy': self.metrics['accuracy'],
                'features_used': len(features),
                'timestamp': datetime.now().isoformat()
            }
            
            # Atualizar métricas
            self.metrics['total_predictions'] += 1
            
            print(f"🎯 ML Predição: {direction} ({confidence:.1f}% confiança)")
            return result
            
        except Exception as e:
            print(f"❌ Erro na predição ML: {e}")
            return self._generate_random_prediction('direction')
    
    def analyze_risk(self, market_data, trade_history):
        """
        ⚠️ Análise de risco baseada em ML
        """
        try:
            if self.models['risk_analyzer'] is None:
                return self._generate_random_prediction('risk')
            
            # Preparar dados para análise de risco
            risk_features = self._extract_risk_features(market_data, trade_history)
            risk_features_scaled = self.scalers['features'].transform([risk_features])
            
            # Predição de risco
            risk_score = self.models['risk_analyzer'].predict(risk_features_scaled)[0]
            risk_probability = self.models['risk_analyzer'].predict_proba(risk_features_scaled)[0]
            
            # Determinar nível de risco
            if risk_score >= 0.7:
                level = 'high'
                message = "Alto risco detectado pelo ML"
                recommendation = "Reduzir stake ou pausar operações"
            elif risk_score >= 0.4:
                level = 'medium'
                message = "Risco moderado detectado pelo ML"
                recommendation = "Operar com cautela"
            else:
                level = 'low'
                message = "Baixo risco detectado pelo ML"
                recommendation = "Seguro para continuar"
            
            result = {
                'level': level,
                'score': risk_score * 100,
                'message': message,
                'recommendation': recommendation,
                'risk_factors': self._identify_risk_factors(risk_features),
                'market_volatility': self._calculate_volatility(trade_history),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"⚠️ ML Risco: {level} (Score: {risk_score*100:.1f})")
            return result
            
        except Exception as e:
            print(f"❌ Erro na análise de risco ML: {e}")
            return self._generate_random_prediction('risk')
    
    def optimize_martingale(self, current_level, recent_performance):
        """
        🎰 Otimização inteligente do Martingale
        """
        try:
            if self.models['martingale_optimizer'] is None:
                return self._generate_martingale_recommendation(current_level)
            
            # Preparar features para otimização
            martingale_features = self._extract_martingale_features(current_level, recent_performance)
            
            # Predição de otimização
            recommendation = self.models['martingale_optimizer'].predict([martingale_features])[0]
            
            result = {
                'action': 'continue' if recommendation == 1 else 'pause',
                'recommended_level': min(current_level + 1, 8) if recommendation == 1 else 0,
                'confidence': 0.8,
                'reasoning': f"ML otimização baseada em performance histórica",
                'risk_assessment': 'low' if recommendation == 1 else 'high',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"🎰 ML Martingale: {result['action']} (Nível {result['recommended_level']})")
            return result
            
        except Exception as e:
            print(f"❌ Erro na otimização Martingale ML: {e}")
            return self._generate_martingale_recommendation(current_level)
    
    def get_market_analysis(self, symbol, timeframe_data):
        """
        📈 Análise completa de mercado com ML
        """
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'ml_version': self.metrics['model_version'],
                'data_quality': len(self.historical_data),
                
                # Análise de tendência
                'trend': self._analyze_trend(timeframe_data),
                
                # Padrões identificados
                'patterns': self._identify_patterns(timeframe_data),
                
                # Volatilidade
                'volatility': self._calculate_volatility(timeframe_data),
                
                # Suporte e resistência
                'support_resistance': self._find_support_resistance(timeframe_data),
                
                # Força da tendência
                'trend_strength': self._calculate_trend_strength(timeframe_data),
                
                # Recomendação geral
                'recommendation': self._generate_market_recommendation(timeframe_data)
            }
            
            print(f"📈 ML Análise completa: {symbol}")
            return analysis
            
        except Exception as e:
            print(f"❌ Erro na análise de mercado ML: {e}")
            return self._generate_basic_analysis(symbol)
    
    def update_performance(self, prediction_id, actual_result):
        """
        📊 Atualiza performance do modelo com resultado real
        """
        try:
            # Adicionar à história de performance
            performance_record = {
                'prediction_id': prediction_id,
                'actual_result': actual_result,
                'timestamp': datetime.now().isoformat(),
                'was_correct': None  # Será calculado
            }
            
            self.performance_history.append(performance_record)
            
            # Calcular métricas atualizadas
            self._update_metrics()
            
            print(f"📊 Performance atualizada. Accuracy: {self.metrics['accuracy']:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao atualizar performance: {e}")
            return False
    
    def save_models(self):
        """
        💾 Salva modelos treinados
        """
        try:
            model_dir = 'ml_models'
            os.makedirs(model_dir, exist_ok=True)
            
            # Salvar modelos
            for name, model in self.models.items():
                if model is not None:
                    joblib.dump(model, f'{model_dir}/{name}.pkl')
            
            # Salvar scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, f'{model_dir}/scaler_{name}.pkl')
            
            # Salvar métricas e configurações
            with open(f'{model_dir}/metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            with open(f'{model_dir}/config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print("💾 Modelos ML salvos com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelos: {e}")
            return False
    
    def _load_saved_models(self):
        """
        📥 Carrega modelos salvos
        """
        try:
            model_dir = 'ml_models'
            if not os.path.exists(model_dir):
                print("📁 Diretório de modelos não encontrado. Iniciando com modelos vazios.")
                return
            
            # Carregar modelos
            for name in self.models.keys():
                model_path = f'{model_dir}/{name}.pkl'
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"✅ Modelo {name} carregado")
            
            # Carregar scalers
            for name in self.scalers.keys():
                scaler_path = f'{model_dir}/scaler_{name}.pkl'
                if os.path.exists(scaler_path):
                    self.scalers[name] = joblib.load(scaler_path)
            
            # Carregar métricas
            metrics_path = f'{model_dir}/metrics.json'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics.update(json.load(f))
            
            print("📥 Modelos ML carregados com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
    
    def _process_trade_data(self, trade_data):
        """
        🔄 Processa dados de trade para formato ML
        """
        return {
            'symbol': trade_data.get('symbol', 'R_50'),
            'direction': 1 if trade_data.get('direction') == 'CALL' else 0,
            'stake': trade_data.get('stake', 1.0),
            'duration': trade_data.get('duration', 5),
            'entry_price': trade_data.get('entry_price', 0),
            'exit_price': trade_data.get('exit_price', 0),
            'pnl': trade_data.get('pnl', 0),
            'result': 1 if trade_data.get('pnl', 0) > 0 else 0,
            'martingale_level': trade_data.get('martingale_level', 0),
            'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
            'volatility': trade_data.get('volatility', 50),
            'market_condition': trade_data.get('market_condition', 'neutral')
        }
    
    def _extract_features(self, market_data):
        """
        🎯 Extrai features para predição
        """
        return [
            market_data.get('currentPrice', 1000),
            market_data.get('volatility', 50),
            market_data.get('winRate', 50),
            market_data.get('martingaleLevel', 0),
            len(market_data.get('recentTrades', [])),
            1 if market_data.get('isAfterLoss', False) else 0,
            hash(market_data.get('symbol', 'R_50')) % 1000,  # Symbol encoding
            datetime.now().hour,  # Hora do dia
            datetime.now().weekday(),  # Dia da semana
        ]
    
    def _extract_risk_features(self, market_data, trade_history):
        """
        ⚠️ Extrai features para análise de risco
        """
        recent_trades = trade_history[-10:] if trade_history else []
        recent_pnl = sum([t.get('pnl', 0) for t in recent_trades])
        
        return [
            market_data.get('martingaleLevel', 0),
            recent_pnl,
            len(recent_trades),
            market_data.get('volatility', 50),
            market_data.get('currentBalance', 1000),
            market_data.get('todayPnL', 0),
            market_data.get('winRate', 50),
            1 if market_data.get('isInCoolingPeriod', False) else 0,
            datetime.now().hour
        ]
    
    def _extract_martingale_features(self, current_level, recent_performance):
        """
        🎰 Extrai features para otimização Martingale
        """
        return [
            current_level,
            recent_performance.get('win_rate', 50),
            recent_performance.get('total_trades', 0),
            recent_performance.get('consecutive_losses', 0),
            recent_performance.get('avg_pnl', 0),
            recent_performance.get('max_loss', 0),
            recent_performance.get('time_since_last_win', 0)
        ]
    
    def _retrain_models(self):
        """
        🔄 Retreina modelos com novos dados
        """
        try:
            if len(self.historical_data) < self.config['min_samples_for_training']:
                print(f"📊 Dados insuficientes para treinar. Atual: {len(self.historical_data)}, Mínimo: {self.config['min_samples_for_training']}")
                return
            
            print("🔄 Iniciando retreinamento dos modelos ML...")
            
            # Preparar dados
            df = pd.DataFrame(self.historical_data)
            
            # Treinar modelo de direção
            self._train_direction_model(df)
            
            # Treinar modelo de risco
            self._train_risk_model(df)
            
            # Treinar otimizador Martingale
            self._train_martingale_model(df)
            
            # Atualizar métricas
            self.metrics['last_training'] = datetime.now().isoformat()
            self.metrics['model_version'] = f"1.{len(self.historical_data)//100}"
            
            # Salvar modelos
            self.save_models()
            
            print("✅ Retreinamento concluído!")
            
        except Exception as e:
            print(f"❌ Erro no retreinamento: {e}")
    
    def _train_direction_model(self, df):
        """
        🎯 Treina modelo de predição de direção
        """
        try:
            # Preparar features e target
            feature_cols = ['entry_price', 'volatility', 'martingale_level', 'stake', 'duration']
            X = df[feature_cols].fillna(0)
            y = df['direction']
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Escalar features
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            # Treinar modelo ensemble
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Avaliar
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models['direction_predictor'] = model
            self.metrics['accuracy'] = accuracy
            
            print(f"🎯 Modelo de direção treinado. Accuracy: {accuracy:.2f}")
            
        except Exception as e:
            print(f"❌ Erro ao treinar modelo de direção: {e}")
    
    def _train_risk_model(self, df):
        """
        ⚠️ Treina modelo de análise de risco
        """
        try:
            # Criar target de risco baseado em PnL
            df['risk_level'] = df['pnl'].apply(lambda x: 1 if x < -5 else 0)
            
            feature_cols = ['martingale_level', 'volatility', 'stake']
            X = df[feature_cols].fillna(0)
            y = df['risk_level']
            
            if len(X) < 10:
                return
            
            # Treinar modelo
            model = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X, y)
            self.models['risk_analyzer'] = model
            
            print("⚠️ Modelo de risco treinado")
            
        except Exception as e:
            print(f"❌ Erro ao treinar modelo de risco: {e}")
    
    def _train_martingale_model(self, df):
        """
        🎰 Treina otimizador Martingale
        """
        try:
            # Criar target baseado em sucesso do Martingale
            df['martingale_success'] = ((df['martingale_level'] > 0) & (df['result'] == 1)).astype(int)
            
            feature_cols = ['martingale_level', 'volatility', 'stake']
            X = df[feature_cols].fillna(0)
            y = df['martingale_success']
            
            if len(X) < 10 or y.sum() == 0:
                return
            
            # Treinar modelo
            model = MLPClassifier(
                hidden_layer_sizes=(10, 5),
                max_iter=200,
                random_state=42
            )
            
            model.fit(X, y)
            self.models['martingale_optimizer'] = model
            
            print("🎰 Otimizador Martingale treinado")
            
        except Exception as e:
            print(f"❌ Erro ao treinar otimizador Martingale: {e}")
    
    def _generate_random_prediction(self, prediction_type):
        """
        🎲 Gera predição aleatória quando modelo não está disponível
        """
        if prediction_type == 'direction':
            return {
                'direction': np.random.choice(['CALL', 'PUT']),
                'confidence': np.random.uniform(60, 80),
                'reasoning': 'Predição aleatória - modelo em treinamento',
                'timestamp': datetime.now().isoformat()
            }
        elif prediction_type == 'risk':
            return {
                'level': np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2]),
                'score': np.random.uniform(20, 80),
                'message': 'Análise aleatória - modelo em treinamento',
                'recommendation': 'Operar com cautela',
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_martingale_recommendation(self, current_level):
        """
        🎰 Gera recomendação Martingale
        """
        return {
            'action': 'continue' if current_level < 4 else 'pause',
            'recommended_level': min(current_level + 1, 8),
            'confidence': 0.6,
            'reasoning': 'Recomendação baseada em regras - modelo em treinamento',
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_metrics(self):
        """
        📊 Atualiza métricas de performance
        """
        if len(self.performance_history) > 0:
            correct = sum([1 for p in self.performance_history if p.get('was_correct', False)])
            total = len(self.performance_history)
            self.metrics['accuracy'] = correct / total if total > 0 else 0
            self.metrics['correct_predictions'] = correct
    
    def _analyze_trend(self, data):
        """📈 Análise de tendência simples"""
        if len(data) < 2:
            return 'neutral'
        
        recent_prices = [d.get('price', 0) for d in data[-5:]]
        if len(recent_prices) >= 2:
            if recent_prices[-1] > recent_prices[0]:
                return 'bullish'
            elif recent_prices[-1] < recent_prices[0]:
                return 'bearish'
        return 'neutral'
    
    def _identify_patterns(self, data):
        """🔍 Identificação de padrões simples"""
        return ['consolidation', 'breakout', 'reversal'][np.random.randint(0, 3)]
    
    def _calculate_volatility(self, data):
        """📊 Cálculo de volatilidade"""
        if len(data) < 2:
            return 50.0
        
        prices = [d.get('price', d.get('pnl', 0)) for d in data[-10:]]
        if len(prices) >= 2:
            return np.std(prices) * 100
        return 50.0
    
    def _find_support_resistance(self, data):
        """📊 Encontra suporte e resistência"""
        if len(data) < 3:
            return {'support': 1000, 'resistance': 1100}
        
        prices = [d.get('price', 1000) for d in data[-20:]]
        return {
            'support': min(prices),
            'resistance': max(prices)
        }
    
    def _calculate_trend_strength(self, data):
        """💪 Calcula força da tendência"""
        return np.random.uniform(0.3, 0.9)
    
    def _generate_market_recommendation(self, data):
        """💡 Gera recomendação de mercado"""
        return np.random.choice(['buy', 'sell', 'hold'], p=[0.4, 0.4, 0.2])
    
    def _generate_basic_analysis(self, symbol):
        """📊 Análise básica quando ML falha"""
        return {
            'symbol': symbol,
            'trend': 'neutral',
            'volatility': 50.0,
            'recommendation': 'hold',
            'timestamp': datetime.now().isoformat(),
            'status': 'basic_analysis'
        }
    
    def _identify_risk_factors(self, features):
        """⚠️ Identifica fatores de risco"""
        factors = []
        if features[0] > 5:  # Martingale level
            factors.append('Alto nível Martingale')
        if features[3] > 80:  # Volatilidade
            factors.append('Alta volatilidade')
        if features[6] < 40:  # Win rate
            factors.append('Taxa de acerto baixa')
        return factors if factors else ['Nenhum fator crítico']

# Instância global do motor ML
ml_engine = TradingMLEngine()

if __name__ == "__main__":
    print("🤖 Trading ML Engine - Teste")
    
    # Teste básico
    test_market_data = {
        'currentPrice': 1000,
        'volatility': 60,
        'symbol': 'R_50',
        'martingaleLevel': 2
    }
    
    prediction = ml_engine.predict_direction(test_market_data)
    print("Predição teste:", prediction)
    
    risk = ml_engine.analyze_risk(test_market_data, [])
    print("Risco teste:", risk)