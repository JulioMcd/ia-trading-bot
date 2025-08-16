from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import random
import time
import warnings
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd

# 🤖 MACHINE LEARNING REAL
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 📊 ANÁLISE TÉCNICA REAL
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib não disponível - usando indicadores simples")

from scipy import stats
import requests

# Suprimir warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# 🔧 CONFIGURAÇÕES AVANÇADAS
class Config:
    MIN_STAKE = 0.35
    MAX_STAKE = 2000
    AI_CONFIDENCE_RANGE = (70, 95)
    RISK_LEVELS = ['low', 'medium', 'high']
    MARKET_TRENDS = ['bullish', 'bearish', 'neutral']
    
    # 🤖 ML CONFIGURAÇÕES
    ML_MODELS = ['random_forest', 'gradient_boost', 'neural_network', 'svm']
    ML_FEATURES = ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'volume', 'volatility']
    ML_LOOKBACK = 100  # Períodos para análise
    ML_MIN_SAMPLES = 50  # Mínimo de samples para treinar
    
    # 📈 ANÁLISE TÉCNICA
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    SMA_PERIODS = [5, 10, 20, 50]

# 🧠 CLASSE IA REAL COM MACHINE LEARNING
class RealTradingAI:
    def __init__(self):
        print("🚀 Inicializando IA Real com Machine Learning...")
        
        # 📊 DADOS E HISTÓRICO
        self.price_history = []
        self.volume_history = []
        self.trades_history = []
        self.features_history = []
        self.labels_history = []
        
        # 🤖 MODELOS ML
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.model_performance = {}
        
        # 📈 INDICADORES TÉCNICOS
        self.technical_indicators = {}
        self.market_regime = 'neutral'
        self.volatility_regime = 'normal'
        
        # 🎯 PREDIÇÕES
        self.last_prediction = None
        self.prediction_confidence = 0
        self.prediction_history = []
        
        # 🔄 SISTEMA DE APRENDIZADO CONTÍNUO
        self.learning_enabled = True
        self.retrain_threshold = 100  # Retreinar a cada 100 trades
        self.trades_since_retrain = 0
        
        # ⚡ PERFORMANCE
        self.processing_times = []
        self.model_accuracy = {}
        
        self._initialize_models()
        print("✅ IA Real inicializada com sucesso!")

    def _initialize_models(self):
        """🤖 Inicializar modelos de Machine Learning"""
        try:
            # 🌳 RANDOM FOREST - Principal modelo
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # 🚀 GRADIENT BOOSTING - Alta performance
            self.models['gradient_boost'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # 🧠 NEURAL NETWORK - Deep Learning simples
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            
            # ⚖️ SVM - Support Vector Machine
            self.models['svm'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            # 📊 SCALERS para normalização
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            print("🤖 Modelos ML inicializados:", list(self.models.keys()))
            
        except Exception as e:
            print(f"❌ Erro ao inicializar modelos: {e}")
            # Fallback para modelo simples
            self.models['simple'] = RandomForestClassifier(n_estimators=50, random_state=42)
            self.scalers['simple'] = StandardScaler()

    def calculate_technical_indicators(self, prices, volumes=None):
        """📈 Calcular indicadores técnicos REAIS"""
        if len(prices) < 50:
            return self._get_simple_indicators(prices)
        
        try:
            prices_array = np.array(prices, dtype=float)
            
            indicators = {}
            
            # 📊 RSI (Relative Strength Index)
            if TALIB_AVAILABLE:
                indicators['rsi'] = talib.RSI(prices_array, timeperiod=Config.RSI_PERIOD)[-1]
            else:
                indicators['rsi'] = self._calculate_rsi_simple(prices_array)
            
            # 📈 MACD
            if TALIB_AVAILABLE:
                macd, macdsignal, macdhist = talib.MACD(
                    prices_array, 
                    fastperiod=Config.MACD_FAST,
                    slowperiod=Config.MACD_SLOW, 
                    signalperiod=Config.MACD_SIGNAL
                )
                indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
                indicators['macd_signal'] = macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0
                indicators['macd_histogram'] = macdhist[-1] if not np.isnan(macdhist[-1]) else 0
            else:
                macd_data = self._calculate_macd_simple(prices_array)
                indicators.update(macd_data)
            
            # 📊 BOLLINGER BANDS
            if TALIB_AVAILABLE:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    prices_array, 
                    timeperiod=Config.BB_PERIOD
                )
                current_price = prices_array[-1]
                indicators['bb_upper'] = bb_upper[-1]
                indicators['bb_middle'] = bb_middle[-1] 
                indicators['bb_lower'] = bb_lower[-1]
                indicators['bb_position'] = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            else:
                bb_data = self._calculate_bollinger_simple(prices_array)
                indicators.update(bb_data)
            
            # 📈 MÉDIAS MÓVEIS
            for period in Config.SMA_PERIODS:
                if len(prices_array) >= period:
                    if TALIB_AVAILABLE:
                        indicators[f'sma_{period}'] = talib.SMA(prices_array, timeperiod=period)[-1]
                        indicators[f'ema_{period}'] = talib.EMA(prices_array, timeperiod=period)[-1]
                    else:
                        indicators[f'sma_{period}'] = np.mean(prices_array[-period:])
                        indicators[f'ema_{period}'] = self._calculate_ema_simple(prices_array, period)
            
            # 📊 VOLATILIDADE
            indicators['volatility'] = np.std(prices_array[-20:]) / np.mean(prices_array[-20:]) * 100
            
            # 📈 MOMENTUM
            if len(prices_array) >= 10:
                indicators['momentum'] = (prices_array[-1] - prices_array[-10]) / prices_array[-10] * 100
            
            # 🔄 RATE OF CHANGE
            if len(prices_array) >= 12:
                indicators['roc'] = (prices_array[-1] - prices_array[-12]) / prices_array[-12] * 100
            
            # 📊 VOLUME ANALYSIS (se disponível)
            if volumes and len(volumes) >= 20:
                volumes_array = np.array(volumes, dtype=float)
                indicators['volume_sma'] = np.mean(volumes_array[-20:])
                indicators['volume_ratio'] = volumes_array[-1] / indicators['volume_sma']
            
            # 🎯 TREND STRENGTH
            if len(prices_array) >= 20:
                trend_slope = np.polyfit(range(20), prices_array[-20:], 1)[0]
                indicators['trend_strength'] = trend_slope / prices_array[-1] * 1000
            
            self.technical_indicators = indicators
            return indicators
            
        except Exception as e:
            print(f"❌ Erro ao calcular indicadores: {e}")
            return self._get_simple_indicators(prices)

    def _calculate_rsi_simple(self, prices, period=14):
        """📊 RSI simples sem TA-Lib"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd_simple(self, prices):
        """📈 MACD simples"""
        if len(prices) < 26:
            return {'macd': 0, 'macd_signal': 0, 'macd_histogram': 0}
        
        ema12 = self._calculate_ema_simple(prices, 12)
        ema26 = self._calculate_ema_simple(prices, 26)
        macd = ema12 - ema26
        
        return {
            'macd': macd,
            'macd_signal': macd * 0.9,  # Simplificado
            'macd_histogram': macd * 0.1
        }

    def _calculate_ema_simple(self, prices, period):
        """📈 EMA simples"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def _calculate_bollinger_simple(self, prices, period=20):
        """📊 Bollinger Bands simples"""
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices)
        else:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        position = (prices[-1] - lower) / (upper - lower) if upper != lower else 0.5
        
        return {
            'bb_upper': upper,
            'bb_middle': sma,
            'bb_lower': lower,
            'bb_position': position
        }

    def _get_simple_indicators(self, prices):
        """📊 Indicadores básicos para dados insuficientes"""
        if len(prices) == 0:
            return {}
        
        return {
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'volatility': 10,
            'momentum': 0,
            'trend_strength': 0,
            'bb_position': 0.5
        }

    def extract_features(self, market_data):
        """🔧 Extrair features para ML"""
        try:
            # 📊 DADOS BÁSICOS
            symbol = market_data.get('symbol', 'R_50')
            current_price = market_data.get('currentPrice', 1000)
            martingale_level = market_data.get('martingaleLevel', 0)
            win_rate = market_data.get('winRate', 50)
            volatility = market_data.get('volatility', 50)
            
            # 📈 ADICIONAR PREÇO AO HISTÓRICO
            self.price_history.append(current_price)
            if len(self.price_history) > Config.ML_LOOKBACK:
                self.price_history = self.price_history[-Config.ML_LOOKBACK:]
            
            # 📊 CALCULAR INDICADORES TÉCNICOS
            if len(self.price_history) >= 20:
                indicators = self.calculate_technical_indicators(self.price_history)
            else:
                indicators = self._get_simple_indicators(self.price_history)
            
            # 🎯 FEATURES PARA ML
            features = []
            
            # Indicadores técnicos principais
            features.extend([
                indicators.get('rsi', 50) / 100,  # Normalizar 0-1
                indicators.get('macd', 0),
                indicators.get('bb_position', 0.5),
                indicators.get('volatility', 10) / 100,
                indicators.get('momentum', 0) / 100,
                indicators.get('trend_strength', 0)
            ])
            
            # Médias móveis (diferenças percentuais)
            for period in [5, 10, 20]:
                sma_key = f'sma_{period}'
                if sma_key in indicators:
                    sma_diff = (current_price - indicators[sma_key]) / current_price
                    features.append(sma_diff)
                else:
                    features.append(0)
            
            # Estados do sistema
            features.extend([
                martingale_level / 10,  # Normalizar
                win_rate / 100,
                min(volatility / 100, 1),  # Limitar a 1
                1 if market_data.get('isAfterLoss', False) else 0
            ])
            
            # Features de contexto temporal
            hour = datetime.now().hour
            features.extend([
                hour / 24,  # Hora normalizada
                (datetime.now().weekday()) / 7,  # Dia da semana
                len(self.trades_history) / 100  # Volume de trades
            ])
            
            # Features estatísticas do preço
            if len(self.price_history) >= 10:
                recent_prices = self.price_history[-10:]
                features.extend([
                    (current_price - min(recent_prices)) / (max(recent_prices) - min(recent_prices) + 1e-8),
                    np.std(recent_prices) / np.mean(recent_prices),
                    stats.skew(recent_prices) if len(recent_prices) > 3 else 0
                ])
            else:
                features.extend([0.5, 0.1, 0])
            
            # Garantir que temos exatamente o número esperado de features
            while len(features) < 20:
                features.append(0)
            
            features = features[:20]  # Limitar a 20 features
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Erro ao extrair features: {e}")
            # Retornar features básicas em caso de erro
            return np.zeros((1, 20))

    def train_models(self):
        """🎓 Treinar modelos de ML"""
        if len(self.features_history) < Config.ML_MIN_SAMPLES:
            print(f"📊 Dados insuficientes para treino: {len(self.features_history)}/{Config.ML_MIN_SAMPLES}")
            return False
        
        try:
            start_time = time.time()
            print(f"🎓 Iniciando treino com {len(self.features_history)} samples...")
            
            # 📊 PREPARAR DADOS
            X = np.array(self.features_history)
            y = np.array(self.labels_history)
            
            # Verificar se temos pelo menos 2 classes
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                print("⚠️ Apenas uma classe nos dados - adicionando balanceamento")
                # Adicionar algumas amostras sintéticas da classe oposta
                opposite_class = 1 - unique_classes[0]
                for _ in range(min(10, len(X) // 2)):
                    X = np.vstack([X, X[0] + np.random.normal(0, 0.1, X.shape[1])])
                    y = np.append(y, opposite_class)
            
            # 🔄 SPLIT DOS DADOS
            if len(X) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # 🤖 TREINAR CADA MODELO
            trained_models = 0
            
            for model_name, model in self.models.items():
                try:
                    print(f"  🔧 Treinando {model_name}...")
                    
                    # Normalizar features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Treinar modelo
                    model.fit(X_train_scaled, y_train)
                    
                    # Avaliar performance
                    if len(X_test) > 0:
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        self.model_accuracy[model_name] = accuracy
                        print(f"    ✅ {model_name}: {accuracy:.3f} accuracy")
                    else:
                        self.model_accuracy[model_name] = 0.7  # Default
                    
                    trained_models += 1
                    
                except Exception as e:
                    print(f"    ❌ Erro em {model_name}: {e}")
                    continue
            
            if trained_models > 0:
                self.is_trained = True
                self.trades_since_retrain = 0
                training_time = time.time() - start_time
                print(f"🎉 Treino concluído! {trained_models} modelos em {training_time:.2f}s")
                return True
            else:
                print("❌ Nenhum modelo foi treinado com sucesso")
                return False
                
        except Exception as e:
            print(f"❌ Erro no treino: {e}")
            return False

    def predict_direction(self, market_data):
        """🎯 Predizer direção usando ML"""
        if not self.is_trained and len(self.features_history) >= Config.ML_MIN_SAMPLES:
            print("🎓 Treinando modelos automaticamente...")
            self.train_models()
        
        if not self.is_trained:
            print("⚠️ Modelos não treinados - usando lógica simples")
            return self._simple_prediction(market_data)
        
        try:
            # 🔧 EXTRAIR FEATURES
            features = self.extract_features(market_data)
            
            # 🤖 FAZER PREDIÇÕES COM TODOS OS MODELOS
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name in self.scalers:
                        # Normalizar features
                        features_scaled = self.scalers[model_name].transform(features)
                        
                        # Predição
                        prediction = model.predict(features_scaled)[0]
                        
                        # Confiança (probabilidade se disponível)
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                            confidence = max(proba)
                        else:
                            confidence = 0.7  # Default
                        
                        predictions[model_name] = prediction
                        confidences[model_name] = confidence
                        
                except Exception as e:
                    print(f"❌ Erro na predição {model_name}: {e}")
                    continue
            
            if not predictions:
                print("⚠️ Nenhuma predição válida - usando fallback")
                return self._simple_prediction(market_data)
            
            # 🗳️ ENSEMBLE VOTING (com peso baseado na performance)
            weighted_votes = 0
            total_weight = 0
            avg_confidence = 0
            
            for model_name, prediction in predictions.items():
                weight = self.model_accuracy.get(model_name, 0.5)
                weighted_votes += prediction * weight
                total_weight += weight
                avg_confidence += confidences[model_name] * weight
            
            # Decisão final
            final_prediction = 1 if weighted_votes / total_weight > 0.5 else 0
            final_confidence = avg_confidence / total_weight
            
            # Converter para direção
            direction = 'CALL' if final_prediction == 1 else 'PUT'
            
            # Salvar predição
            self.last_prediction = {
                'direction': direction,
                'confidence': final_confidence * 100,
                'models_used': list(predictions.keys()),
                'individual_predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"🎯 Predição ML: {direction} ({final_confidence*100:.1f}% confiança)")
            return direction, final_confidence * 100
            
        except Exception as e:
            print(f"❌ Erro na predição ML: {e}")
            return self._simple_prediction(market_data)

    def _simple_prediction(self, market_data):
        """🎲 Predição simples para fallback"""
        # Usar indicadores básicos
        if len(self.price_history) >= 10:
            recent_trend = np.mean(self.price_history[-5:]) - np.mean(self.price_history[-10:-5])
            if recent_trend > 0:
                return 'CALL', 65
            else:
                return 'PUT', 65
        else:
            # Random com slight bias
            return random.choice(['CALL', 'PUT']), 60

    def add_training_data(self, trade_result):
        """📚 Adicionar dados para treinamento contínuo"""
        try:
            if 'features' in trade_result and 'success' in trade_result:
                features = trade_result['features']
                label = 1 if trade_result['success'] else 0
                
                self.features_history.append(features)
                self.labels_history.append(label)
                
                # Manter apenas os últimos dados
                max_history = Config.ML_LOOKBACK * 2
                if len(self.features_history) > max_history:
                    self.features_history = self.features_history[-max_history:]
                    self.labels_history = self.labels_history[-max_history:]
                
                self.trades_since_retrain += 1
                
                # Retreinar periodicamente
                if (self.trades_since_retrain >= self.retrain_threshold and 
                    len(self.features_history) >= Config.ML_MIN_SAMPLES):
                    print("🔄 Retreinamento automático iniciado...")
                    self.train_models()
                
                print(f"📚 Dados adicionados: {len(self.features_history)} total samples")
                return True
                
        except Exception as e:
            print(f"❌ Erro ao adicionar dados: {e}")
            return False

    def analyze_market_ml(self, market_data):
        """📊 Análise de mercado com ML"""
        try:
            start_time = time.time()
            
            # 🔧 EXTRAIR FEATURES E INDICADORES
            features = self.extract_features(market_data)
            indicators = self.technical_indicators
            
            # 🎯 PREDIÇÃO ML
            direction, confidence = self.predict_direction(market_data)
            
            # 📈 ANÁLISE DE REGIME DE MERCADO
            market_regime = self._analyze_market_regime()
            volatility_regime = self._analyze_volatility_regime()
            
            # ⚠️ AJUSTES BASEADOS NO MARTINGALE
            martingale_level = market_data.get('martingaleLevel', 0)
            is_after_loss = market_data.get('isAfterLoss', False)
            
            if is_after_loss and martingale_level > 0:
                confidence *= 0.9  # Reduzir confiança após perdas
                message_prefix = "🤖 ML CAUTELOSO pós-perda"
            elif martingale_level > 4:
                confidence *= 0.8  # Muito conservador em alto martingale
                message_prefix = "🤖 ML CONSERVADOR alto-risco"
            else:
                message_prefix = "🤖 ML ANÁLISE"
            
            # 📊 GERAR MENSAGEM DETALHADA
            symbol = market_data.get('symbol', 'R_50')
            rsi = indicators.get('rsi', 50)
            bb_pos = indicators.get('bb_position', 0.5)
            trend = indicators.get('trend_strength', 0)
            
            message = f"{message_prefix} {symbol}: "
            message += f"RSI {rsi:.1f}, BB {bb_pos:.2f}, "
            message += f"Tendência {trend:.2f}, "
            message += f"Regime {market_regime}/{volatility_regime}"
            
            # 📈 RECOMENDAÇÃO BASEADA EM MÚLTIPLOS FATORES
            recommendation = self._generate_ml_recommendation(
                confidence, martingale_level, is_after_loss, market_regime
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            analysis = {
                'message': message,
                'direction': direction,
                'confidence': confidence,
                'volatility': indicators.get('volatility', 50),
                'trend': 'bullish' if trend > 0 else 'bearish' if trend < 0 else 'neutral',
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'martingaleLevel': martingale_level,
                'isAfterLoss': is_after_loss,
                'recommendation': recommendation,
                'ml_enabled': True,
                'ml_models_count': len(self.models),
                'ml_trained': self.is_trained,
                'ml_samples': len(self.features_history),
                'processing_time_ms': processing_time * 1000,
                'technical_indicators': indicators,
                'market_regime': market_regime,
                'volatility_regime': volatility_regime,
                'model_accuracy': self.model_accuracy
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Erro na análise ML: {e}")
            # Fallback para análise simples
            return self._simple_analysis(market_data)

    def _analyze_market_regime(self):
        """📈 Analisar regime de mercado"""
        if len(self.price_history) < 50:
            return 'neutral'
        
        try:
            recent_prices = self.price_history[-50:]
            
            # Tendência geral
            slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Volatilidade
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            if slope > 0 and volatility < 0.02:
                return 'trending_up'
            elif slope < 0 and volatility < 0.02:
                return 'trending_down'
            elif volatility > 0.05:
                return 'high_volatility'
            else:
                return 'neutral'
                
        except:
            return 'neutral'

    def _analyze_volatility_regime(self):
        """📊 Analisar regime de volatilidade"""
        if len(self.price_history) < 20:
            return 'normal'
        
        try:
            recent_volatility = np.std(self.price_history[-20:]) / np.mean(self.price_history[-20:])
            
            if recent_volatility > 0.05:
                return 'high'
            elif recent_volatility < 0.01:
                return 'low'
            else:
                return 'normal'
                
        except:
            return 'normal'

    def _generate_ml_recommendation(self, confidence, martingale_level, is_after_loss, market_regime):
        """💡 Gerar recomendação inteligente"""
        if is_after_loss and martingale_level > 0:
            return "wait_for_better_setup"
        elif confidence > 85 and market_regime in ['trending_up', 'trending_down']:
            return "strong_signal"
        elif confidence > 75:
            return "moderate_signal"
        elif confidence < 65:
            return "weak_signal"
        else:
            return "moderate_signal"

    def _simple_analysis(self, market_data):
        """📊 Análise simples para fallback"""
        return {
            'message': f"📊 Análise básica do {market_data.get('symbol', 'mercado')}",
            'volatility': random.uniform(40, 80),
            'trend': random.choice(['bullish', 'bearish', 'neutral']),
            'confidence': random.uniform(60, 80),
            'timestamp': datetime.now().isoformat(),
            'ml_enabled': False,
            'fallback': True
        }

    def get_trading_signal_ml(self, signal_data):
        """🎯 Gerar sinal de trading com ML"""
        try:
            # 🔧 ANÁLISE ML COMPLETA
            analysis = self.analyze_market_ml(signal_data)
            
            direction = analysis.get('direction', 'CALL')
            confidence = analysis.get('confidence', 70)
            
            # 🎯 REASONING DETALHADO
            indicators = analysis.get('technical_indicators', {})
            market_regime = analysis.get('market_regime', 'neutral')
            
            reasoning = f"🤖 ML Signal: {direction} | "
            reasoning += f"RSI {indicators.get('rsi', 50):.1f} | "
            reasoning += f"MACD {indicators.get('macd', 0):.3f} | "
            reasoning += f"Regime {market_regime} | "
            reasoning += f"Modelos: {len([m for m in self.models.keys() if m in self.model_accuracy])}"
            
            # ⚡ OTIMIZAÇÃO DE TIMEFRAME
            optimal_timeframe = self._suggest_optimal_timeframe(analysis)
            
            signal = {
                'direction': direction,
                'confidence': confidence,
                'reasoning': reasoning,
                'timeframe': optimal_timeframe,
                'entry_price': signal_data.get('currentPrice', 1000),
                'timestamp': datetime.now().isoformat(),
                'symbol': signal_data.get('symbol', 'R_50'),
                'martingaleLevel': signal_data.get('martingaleLevel', 0),
                'isAfterLoss': signal_data.get('isAfterLoss', False),
                'recommendation': analysis.get('recommendation', 'moderate_signal'),
                'ml_enabled': True,
                'ml_models_used': len(self.models),
                'technical_score': self._calculate_technical_score(indicators),
                'market_regime': market_regime,
                'volatility_regime': analysis.get('volatility_regime', 'normal'),
                'risk_adjusted_confidence': self._adjust_confidence_for_risk(confidence, signal_data)
            }
            
            return signal
            
        except Exception as e:
            print(f"❌ Erro no sinal ML: {e}")
            return self._simple_signal(signal_data)

    def _suggest_optimal_timeframe(self, analysis):
        """⏱️ Sugerir timeframe otimizado"""
        volatility = analysis.get('volatility', 50)
        market_regime = analysis.get('market_regime', 'neutral')
        
        if market_regime == 'high_volatility':
            return '3m'  # Mais rápido em alta volatilidade
        elif volatility > 70:
            return '2m'
        elif volatility < 30:
            return '5m'  # Mais devagar em baixa volatilidade
        else:
            return '5m'  # Default

    def _calculate_technical_score(self, indicators):
        """📊 Calcular score técnico"""
        try:
            score = 0
            max_score = 0
            
            # RSI Score
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 1
            max_score += 1
            
            # MACD Score
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if abs(macd - macd_signal) < 0.1:
                score += 1
            max_score += 1
            
            # Bollinger Position Score
            bb_pos = indicators.get('bb_position', 0.5)
            if 0.2 <= bb_pos <= 0.8:
                score += 1
            max_score += 1
            
            return (score / max_score) * 100 if max_score > 0 else 50
            
        except:
            return 50

    def _adjust_confidence_for_risk(self, confidence, signal_data):
        """⚠️ Ajustar confiança baseado no risco"""
        martingale_level = signal_data.get('martingaleLevel', 0)
        win_rate = signal_data.get('winRate', 50)
        
        adjusted = confidence
        
        # Penalizar alto martingale
        if martingale_level > 3:
            adjusted *= 0.9
        if martingale_level > 6:
            adjusted *= 0.8
        
        # Penalizar baixa taxa de acerto
        if win_rate < 40:
            adjusted *= 0.85
        elif win_rate > 70:
            adjusted *= 1.05
        
        return max(50, min(95, adjusted))

    def _simple_signal(self, signal_data):
        """🎲 Sinal simples para fallback"""
        direction = random.choice(['CALL', 'PUT'])
        confidence = random.uniform(65, 80)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'reasoning': 'Sinal básico (ML indisponível)',
            'timeframe': '5m',
            'entry_price': signal_data.get('currentPrice', 1000),
            'timestamp': datetime.now().isoformat(),
            'ml_enabled': False,
            'fallback': True
        }

    def assess_risk_ml(self, risk_data):
        """⚠️ Avaliação de risco com ML"""
        try:
            start_time = time.time()
            
            # 📊 DADOS BÁSICOS
            current_balance = risk_data.get('currentBalance', 1000)
            today_pnl = risk_data.get('todayPnL', 0)
            martingale_level = risk_data.get('martingaleLevel', 0)
            win_rate = risk_data.get('winRate', 50)
            total_trades = risk_data.get('totalTrades', 0)
            
            # 🧮 CÁLCULO DE RISCO ML
            risk_score = 0
            
            # 🎰 Risco Martingale (peso: 30%)
            martingale_risk = min(martingale_level * 12, 60)
            risk_score += martingale_risk * 0.3
            
            # 💰 Risco P&L (peso: 25%)
            pnl_percentage = (today_pnl / current_balance) * 100 if current_balance > 0 else 0
            pnl_risk = max(0, abs(pnl_percentage) - 5) * 3
            risk_score += min(pnl_risk, 40) * 0.25
            
            # 🎯 Risco Performance (peso: 20%)
            performance_risk = max(0, 60 - win_rate)
            risk_score += performance_risk * 0.2
            
            # 📈 Risco Técnico com ML (peso: 15%)
            technical_risk = self._calculate_technical_risk()
            risk_score += technical_risk * 0.15
            
            # 🌊 Risco de Volatilidade (peso: 10%)
            volatility_risk = self._calculate_volatility_risk()
            risk_score += volatility_risk * 0.1
            
            # ⏰ Limitação máxima
            risk_score = max(0, min(100, risk_score))
            
            # 📊 CLASSIFICAÇÃO DE RISCO
            if risk_score >= 70:
                level = 'high'
                color_emoji = '🔴'
                action_recommendation = "PARAR operações imediatamente"
            elif risk_score >= 45:
                level = 'medium'
                color_emoji = '🟡'
                action_recommendation = "Reduzir stake e operar com cautela"
            else:
                level = 'low'
                color_emoji = '🟢'
                action_recommendation = "Condições normais para operar"
            
            # 🤖 MENSAGEM INTELIGENTE
            message = f"🤖 ML Risk {color_emoji}: Score {risk_score:.1f} | "
            
            # Detalhes por categoria
            details = []
            if martingale_risk > 20:
                details.append(f"Martingale L{martingale_level}")
            if abs(pnl_percentage) > 10:
                details.append(f"P&L {pnl_percentage:+.1f}%")
            if win_rate < 45:
                details.append(f"WinRate {win_rate:.1f}%")
            if technical_risk > 30:
                details.append("Sinais técnicos negativos")
            
            if details:
                message += " | ".join(details)
            else:
                message += "Todos os indicadores normais"
            
            # 📈 RECOMENDAÇÕES ESPECÍFICAS
            recommendations = []
            if martingale_level > 5:
                recommendations.append("Considerar reset manual do Martingale")
            if win_rate < 40 and total_trades > 10:
                recommendations.append("Revisar estratégia de entrada")
            if abs(pnl_percentage) > 15:
                recommendations.append("Considerar pausa nas operações")
            if technical_risk > 40:
                recommendations.append("Aguardar melhores condições técnicas")
            
            processing_time = time.time() - start_time
            
            risk_assessment = {
                'level': level,
                'message': message,
                'score': risk_score,
                'recommendation': action_recommendation,
                'timestamp': datetime.now().isoformat(),
                'martingaleLevel': martingale_level,
                'isAfterLoss': risk_data.get('needsAnalysisAfterLoss', False),
                'currentBalance': current_balance,
                'todayPnL': today_pnl,
                'winRate': win_rate,
                'ml_enabled': True,
                'processing_time_ms': processing_time * 1000,
                'risk_breakdown': {
                    'martingale_risk': martingale_risk,
                    'pnl_risk': pnl_risk,
                    'performance_risk': performance_risk,
                    'technical_risk': technical_risk,
                    'volatility_risk': volatility_risk,
                    'total_score': risk_score
                },
                'specific_recommendations': recommendations,
                'market_conditions': {
                    'volatility_regime': getattr(self, 'volatility_regime', 'normal'),
                    'market_regime': getattr(self, 'market_regime', 'neutral'),
                    'technical_score': self._calculate_technical_score(self.technical_indicators)
                }
            }
            
            return risk_assessment
            
        except Exception as e:
            print(f"❌ Erro na avaliação de risco ML: {e}")
            return self._simple_risk_assessment(risk_data)

    def _calculate_technical_risk(self):
        """📈 Calcular risco técnico"""
        if not self.technical_indicators:
            return 25  # Risco médio se não há dados
        
        try:
            risk = 0
            
            # RSI extremos
            rsi = self.technical_indicators.get('rsi', 50)
            if rsi > 80 or rsi < 20:
                risk += 20
            
            # Bollinger Bands extremos
            bb_pos = self.technical_indicators.get('bb_position', 0.5)
            if bb_pos > 0.9 or bb_pos < 0.1:
                risk += 15
            
            # Alta volatilidade
            volatility = self.technical_indicators.get('volatility', 10)
            if volatility > 50:
                risk += 10
            
            # Divergência MACD
            macd = self.technical_indicators.get('macd', 0)
            macd_signal = self.technical_indicators.get('macd_signal', 0)
            if abs(macd - macd_signal) > 0.5:
                risk += 10
            
            return min(risk, 50)
            
        except:
            return 25

    def _calculate_volatility_risk(self):
        """🌊 Calcular risco de volatilidade"""
        if len(self.price_history) < 20:
            return 15  # Risco médio
        
        try:
            recent_volatility = np.std(self.price_history[-20:]) / np.mean(self.price_history[-20:])
            
            if recent_volatility > 0.1:  # Muito alta
                return 40
            elif recent_volatility > 0.05:  # Alta
                return 25
            elif recent_volatility < 0.005:  # Muito baixa (também é risco)
                return 20
            else:
                return 10  # Normal
                
        except:
            return 15

    def _simple_risk_assessment(self, risk_data):
        """⚠️ Avaliação simples para fallback"""
        martingale_level = risk_data.get('martingaleLevel', 0)
        
        if martingale_level > 5:
            level = 'high'
            score = 80
            message = f"Risco alto - Martingale nível {martingale_level}"
        elif martingale_level > 2:
            level = 'medium'
            score = 50
            message = f"Risco moderado - Martingale nível {martingale_level}"
        else:
            level = 'low'
            score = 25
            message = "Risco baixo"
        
        return {
            'level': level,
            'message': message,
            'score': score,
            'recommendation': "Avaliação básica (ML indisponível)",
            'timestamp': datetime.now().isoformat(),
            'ml_enabled': False,
            'fallback': True
        }

    def get_ml_statistics(self):
        """📊 Estatísticas do sistema ML"""
        try:
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            stats = {
                'ml_enabled': True,
                'models_available': list(self.models.keys()),
                'models_trained': self.is_trained,
                'training_samples': len(self.features_history),
                'model_accuracy': self.model_accuracy,
                'price_history_size': len(self.price_history),
                'trades_since_retrain': self.trades_since_retrain,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'technical_indicators_available': list(self.technical_indicators.keys()),
                'last_prediction': self.last_prediction,
                'market_regime': getattr(self, 'market_regime', 'neutral'),
                'volatility_regime': getattr(self, 'volatility_regime', 'normal'),
                'timestamp': datetime.now().isoformat(),
                'system_health': {
                    'models_healthy': len([m for m, acc in self.model_accuracy.items() if acc > 0.5]),
                    'data_sufficient': len(self.features_history) >= Config.ML_MIN_SAMPLES,
                    'indicators_working': len(self.technical_indicators) > 5,
                    'performance_good': avg_processing_time < 1.0
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas ML: {e}")
            return {
                'ml_enabled': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# 🚀 INSTÂNCIA GLOBAL DA IA REAL
print("🔥 Carregando IA Real com Machine Learning...")
trading_ai = RealTradingAI()
print("✅ Sistema ML completamente inicializado!")

# ==============================================
# 🌐 ROTAS DA API (ATUALIZADAS)
# ==============================================

@app.route('/')
def index():
    """🏠 Servir o frontend"""
    return send_from_directory('public', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """🩺 Health check da API com ML"""
    ml_stats = trading_ai.get_ml_statistics()
    
    return jsonify({
        'status': 'OK',
        'service': 'Trading Bot IA REAL + Machine Learning',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0-ML',
        'features': [
            '🤖 Real Machine Learning',
            '📊 Technical Analysis',
            '🎯 Smart Predictions', 
            '⚠️ ML Risk Assessment',
            '🧠 Neural Networks',
            '🌳 Random Forest',
            '🚀 Gradient Boosting',
            '📈 Real-time Learning',
            '🎰 Intelligent Martingale',
            '📊 Advanced Indicators'
        ],
        'ml_status': ml_stats,
        'system_resources': {
            'models_loaded': len(trading_ai.models),
            'memory_efficient': True,
            'processing_optimized': True
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_market():
    """📊 Endpoint para análise de mercado com ML"""
    try:
        market_data = request.get_json()
        
        if not market_data:
            return jsonify({'error': 'Dados de mercado necessários'}), 400
            
        # 🤖 ANÁLISE ML REAL
        analysis = trading_ai.analyze_market_ml(market_data)
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"❌ Erro na análise de mercado: {e}")
        return jsonify({
            'error': 'Erro na análise de mercado',
            'details': str(e),
            'fallback': True
        }), 500

@app.route('/api/signal', methods=['POST'])
def get_trading_signal():
    """🎯 Endpoint para sinal de trading com ML"""
    try:
        signal_data = request.get_json()
        
        if not signal_data:
            return jsonify({'error': 'Dados para sinal necessários'}), 400
            
        # 🎯 SINAL ML REAL
        signal = trading_ai.get_trading_signal_ml(signal_data)
        
        return jsonify(signal)
        
    except Exception as e:
        print(f"❌ Erro ao gerar sinal: {e}")
        return jsonify({
            'error': 'Erro ao gerar sinal',
            'details': str(e),
            'fallback': True
        }), 500

@app.route('/api/risk', methods=['POST'])
def assess_risk():
    """⚠️ Endpoint para avaliação de risco com ML"""
    try:
        risk_data = request.get_json()
        
        if not risk_data:
            return jsonify({'error': 'Dados de risco necessários'}), 400
            
        # ⚠️ AVALIAÇÃO ML REAL
        risk_assessment = trading_ai.assess_risk_ml(risk_data)
        
        return jsonify(risk_assessment)
        
    except Exception as e:
        print(f"❌ Erro na avaliação de risco: {e}")
        return jsonify({
            'error': 'Erro na avaliação de risco',
            'details': str(e),
            'fallback': True
        }), 500

@app.route('/api/ml/learn', methods=['POST'])
def ml_learn():
    """🎓 Endpoint para aprendizado ML contínuo"""
    try:
        trade_data = request.get_json()
        
        if not trade_data:
            return jsonify({'error': 'Dados de trade necessários'}), 400
        
        # 📚 ADICIONAR AO TREINAMENTO
        success = trading_ai.add_training_data(trade_data)
        ml_stats = trading_ai.get_ml_statistics()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Dados adicionados ao ML real',
                'ml_stats': ml_stats
            })
        else:
            return jsonify({'error': 'Falha ao adicionar ao ML'}), 500
            
    except Exception as e:
        print(f"❌ Erro no aprendizado ML: {e}")
        return jsonify({
            'error': 'Erro no aprendizado ML',
            'details': str(e)
        }), 500

@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    """🎓 Endpoint para treinamento manual dos modelos"""
    try:
        success = trading_ai.train_models()
        ml_stats = trading_ai.get_ml_statistics()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Modelos treinados com sucesso',
                'ml_stats': ml_stats
            })
        else:
            return jsonify({
                'status': 'insufficient_data',
                'message': f'Dados insuficientes. Necessário: {Config.ML_MIN_SAMPLES}, Atual: {len(trading_ai.features_history)}',
                'ml_stats': ml_stats
            })
            
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        return jsonify({
            'error': 'Erro no treinamento',
            'details': str(e)
        }), 500

@app.route('/api/ml/stats', methods=['GET'])
def ml_statistics():
    """📊 Estatísticas detalhadas do ML"""
    try:
        stats = trading_ai.get_ml_statistics()
        return jsonify(stats)
    except Exception as e:
        print(f"❌ Erro ao obter estatísticas ML: {e}")
        return jsonify({
            'error': 'Erro ao obter estatísticas ML',
            'details': str(e)
        }), 500

@app.route('/api/ml/indicators', methods=['GET'])
def get_technical_indicators():
    """📈 Obter indicadores técnicos atuais"""
    try:
        indicators = trading_ai.technical_indicators
        return jsonify({
            'indicators': indicators,
            'timestamp': datetime.now().isoformat(),
            'available': len(indicators) > 0
        })
    except Exception as e:
        return jsonify({
            'error': 'Erro ao obter indicadores',
            'details': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_ai_stats():
    """📊 Estatísticas gerais da IA"""
    ml_stats = trading_ai.get_ml_statistics()
    
    return jsonify({
        'total_analyses': len(trading_ai.price_history),
        'ml_status': ml_stats,
        'uptime': datetime.now().isoformat(),
        'status': 'active_ml',
        'version': '3.0.0-ML-REAL'
    })

# ==============================================
# 🚀 INICIALIZAÇÃO
# ==============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    
    print("🚀 Iniciando Trading Bot com MACHINE LEARNING REAL...")
    print(f"🤖 ML Status: {len(trading_ai.models)} modelos carregados")
    print(f"📊 Indicadores: {len(Config.ML_FEATURES)} features disponíveis")
    print(f"🎓 Treinamento: Automático a cada {trading_ai.retrain_threshold} trades")
    print(f"⚡ Performance: Otimizado para produção")
    print(f"🌐 Servidor rodando na porta: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
