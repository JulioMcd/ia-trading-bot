from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import ta
import joblib
import logging
from typing import Dict, List, Any
import json
import asyncio
import aiohttp
import threading
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# URL da API de dados (segunda API que criaremos)
DATA_API_URL = os.environ.get('DATA_API_URL', 'https://sua-api-dados.onrender.com')
API_KEY = os.environ.get('API_KEY', 'rnd_qpdTVwAeWzIItVbxHPPCc34uirv9')

class TradingMLEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.performance_history = []
        self.model_path = 'trading_model.pkl'
        self.scaler_path = 'scaler.pkl'
        self.load_model()
        
    def load_model(self):
        """Carrega modelo treinado se existir"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info("Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.create_default_model()
    
    def create_default_model(self):
        """Cria modelo padrão"""
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        logger.info("Modelo padrão criado")
    
    def save_model(self):
        """Salva o modelo treinado"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Modelo salvo com sucesso")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
    
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Obtém dados de mercado em tempo real"""
        try:
            # Para índices sintéticos da Deriv, simulamos dados baseados em padrões reais
            if symbol.startswith('R_') or symbol.startswith('1HZ'):
                return self.simulate_synthetic_data(symbol)
            else:
                # Para outros símbolos, usar yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                return data
        except Exception as e:
            logger.error(f"Erro ao obter dados de mercado: {e}")
            return self.simulate_synthetic_data(symbol)
    
    def simulate_synthetic_data(self, symbol: str) -> pd.DataFrame:
        """Simula dados para índices sintéticos"""
        try:
            # Obter dados históricos da API de dados
            response = requests.get(f"{DATA_API_URL}/market-data/{symbol}")
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['prices'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
        except:
            pass
        
        # Fallback: gerar dados sintéticos
        periods = 100
        base_price = 1000
        volatility = 0.02 if 'R_10' in symbol else 0.05
        
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(periods, 0, -1)]
        prices = []
        
        current_price = base_price
        for _ in range(periods):
            change = np.random.normal(0, volatility) * current_price
            current_price += change
            prices.append(current_price)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, periods)
        }, index=timestamps)
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos"""
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            # Momentum
            df['momentum'] = df['Close'].pct_change(5)
            
            # Price relative to moving averages
            df['price_vs_sma'] = (df['Close'] - df['sma_20']) / df['sma_20']
            df['price_vs_ema'] = (df['Close'] - df['ema_12']) / df['ema_12']
            
            return df.fillna(method='bfill').fillna(0)
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return df
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extrai features para o modelo"""
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_low', 'bb_mid',
            'sma_20', 'ema_12', 'ema_26',
            'stoch_k', 'stoch_d', 'atr', 'momentum',
            'price_vs_sma', 'price_vs_ema', 'volume_sma'
        ]
        
        # Adicionar features de tempo
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['minute'] = df.index.minute
        
        feature_columns.extend(['hour', 'day_of_week', 'minute'])
        
        # Filtrar apenas colunas que existem
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("Nenhuma feature disponível, usando preços básicos")
            return df[['Close']].values
        
        self.feature_columns = available_columns
        return df[available_columns].values
    
    async def get_historical_performance(self) -> Dict:
        """Obtém performance histórica da API de dados"""
        try:
            response = requests.get(f"{DATA_API_URL}/performance/history")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Erro ao obter performance histórica: {e}")
        
        return {"trades": [], "win_rate": 0, "total_pnl": 0}
    
    def train_with_historical_data(self, performance_data: Dict):
        """Treina modelo com dados históricos"""
        try:
            trades = performance_data.get('trades', [])
            if len(trades) < 10:
                logger.warning("Dados insuficientes para treinamento")
                return
            
            # Preparar dados de treinamento
            X_data = []
            y_data = []
            
            for trade in trades:
                if 'features' in trade and 'result' in trade:
                    X_data.append(trade['features'])
                    y_data.append(1 if trade['result'] == 'win' else 0)
            
            if len(X_data) < 10:
                logger.warning("Features insuficientes para treinamento")
                return
            
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar modelo
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliar
            y_pred = self.model.predict(X_test_scaled)
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            
            self.is_trained = True
            self.save_model()
            
            logger.info(f"Modelo treinado com acurácia: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
    
    async def analyze_market(self, symbol: str, context: Dict) -> Dict:
        """Análise principal do mercado"""
        try:
            # Obter dados de mercado
            df = await self.get_market_data(symbol)
            
            if df.empty:
                return self.get_fallback_analysis(symbol, context)
            
            # Calcular indicadores
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Análise de tendência
            trend_analysis = self.analyze_trend(df_with_indicators)
            
            # Análise de volatilidade
            volatility_analysis = self.analyze_volatility(df_with_indicators)
            
            # Análise de volume
            volume_analysis = self.analyze_volume(df_with_indicators)
            
            # Detecção de padrões
            pattern_analysis = self.detect_patterns(df_with_indicators)
            
            # Considerar context do Martingale
            martingale_adjustment = self.analyze_martingale_context(context)
            
            # Análise de horário de mercado
            time_analysis = self.analyze_market_timing()
            
            confidence = self.calculate_confidence([
                trend_analysis, volatility_analysis, volume_analysis,
                pattern_analysis, martingale_adjustment, time_analysis
            ])
            
            return {
                "message": f"Análise do {symbol}: Tendência {trend_analysis['direction']}, Volatilidade {volatility_analysis['level']}%",
                "trend": trend_analysis,
                "volatility": volatility_analysis,
                "volume": volume_analysis,
                "patterns": pattern_analysis,
                "martingale_context": martingale_adjustment,
                "market_timing": time_analysis,
                "confidence": confidence,
                "recommendation": self.get_recommendation(confidence, context),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return self.get_fallback_analysis(symbol, context)
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analisa tendência do mercado"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-10] if len(df) >= 10 else df.iloc[0]
            
            price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
            
            # Análise de médias móveis
            sma_trend = "bullish" if latest['Close'] > latest['sma_20'] else "bearish"
            ema_trend = "bullish" if latest['ema_12'] > latest['ema_26'] else "bearish"
            
            # MACD
            macd_trend = "bullish" if latest['macd'] > latest['macd_signal'] else "bearish"
            
            trends = [sma_trend, ema_trend, macd_trend]
            bullish_count = trends.count("bullish")
            
            if bullish_count >= 2:
                direction = "bullish"
                strength = bullish_count / len(trends)
            else:
                direction = "bearish"
                strength = (len(trends) - bullish_count) / len(trends)
            
            return {
                "direction": direction,
                "strength": strength,
                "price_change": price_change,
                "sma_trend": sma_trend,
                "ema_trend": ema_trend,
                "macd_trend": macd_trend
            }
        except Exception as e:
            logger.error(f"Erro na análise de tendência: {e}")
            return {"direction": "neutral", "strength": 0.5, "price_change": 0}
    
    def analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analisa volatilidade do mercado"""
        try:
            latest = df.iloc[-1]
            
            # ATR normalizado
            atr_pct = (latest['atr'] / latest['Close']) * 100
            
            # Bollinger Bands width
            bb_width = ((latest['bb_high'] - latest['bb_low']) / latest['Close']) * 100
            
            # Volatilidade histórica
            returns = df['Close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252) * 100  # Anualizada
            
            # Classificar volatilidade
            if atr_pct > 2:
                level = "high"
            elif atr_pct > 1:
                level = "medium"
            else:
                level = "low"
            
            return {
                "level": level,
                "atr_percentage": atr_pct,
                "bb_width": bb_width,
                "historical_volatility": hist_vol,
                "is_expanding": bb_width > hist_vol
            }
        except Exception as e:
            logger.error(f"Erro na análise de volatilidade: {e}")
            return {"level": "medium", "atr_percentage": 1.5}
    
    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analisa volume de negociação"""
        try:
            latest_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
            
            vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1
            
            if vol_ratio > 1.5:
                analysis = "high"
            elif vol_ratio > 0.8:
                analysis = "normal"
            else:
                analysis = "low"
            
            return {
                "analysis": analysis,
                "current_volume": latest_vol,
                "average_volume": avg_vol,
                "volume_ratio": vol_ratio
            }
        except Exception as e:
            logger.error(f"Erro na análise de volume: {e}")
            return {"analysis": "normal", "volume_ratio": 1.0}
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detecta padrões técnicos"""
        try:
            latest = df.iloc[-1]
            patterns = []
            
            # RSI Overbought/Oversold
            if latest['rsi'] > 70:
                patterns.append({"type": "rsi_overbought", "strength": (latest['rsi'] - 70) / 30})
            elif latest['rsi'] < 30:
                patterns.append({"type": "rsi_oversold", "strength": (30 - latest['rsi']) / 30})
            
            # Bollinger Bands
            if latest['Close'] > latest['bb_high']:
                patterns.append({"type": "bb_upper_break", "strength": 0.8})
            elif latest['Close'] < latest['bb_low']:
                patterns.append({"type": "bb_lower_break", "strength": 0.8})
            
            # MACD Divergência
            if latest['macd_diff'] > 0 and df['macd_diff'].iloc[-2] <= 0:
                patterns.append({"type": "macd_bullish_cross", "strength": 0.7})
            elif latest['macd_diff'] < 0 and df['macd_diff'].iloc[-2] >= 0:
                patterns.append({"type": "macd_bearish_cross", "strength": 0.7})
            
            return {
                "detected_patterns": patterns,
                "pattern_count": len(patterns),
                "strongest_pattern": max(patterns, key=lambda x: x['strength']) if patterns else None
            }
        except Exception as e:
            logger.error(f"Erro na detecção de padrões: {e}")
            return {"detected_patterns": [], "pattern_count": 0}
    
    def analyze_martingale_context(self, context: Dict) -> Dict:
        """Analisa contexto do Martingale"""
        try:
            martingale_level = context.get('martingaleLevel', 0)
            is_after_loss = context.get('isAfterLoss', False)
            win_rate = context.get('winRate', 50)
            
            risk_multiplier = 1.0
            recommendation = "normal"
            
            if martingale_level > 0:
                risk_multiplier = 1 + (martingale_level * 0.2)
                
                if martingale_level > 3:
                    recommendation = "high_caution"
                elif martingale_level > 1:
                    recommendation = "moderate_caution"
            
            if is_after_loss:
                recommendation = "wait_for_better_signal"
                risk_multiplier += 0.3
            
            if win_rate < 40:
                recommendation = "consider_pause"
                risk_multiplier += 0.5
            
            return {
                "martingale_level": martingale_level,
                "is_after_loss": is_after_loss,
                "risk_multiplier": risk_multiplier,
                "recommendation": recommendation,
                "should_be_conservative": martingale_level > 2 or is_after_loss or win_rate < 40
            }
        except Exception as e:
            logger.error(f"Erro na análise do Martingale: {e}")
            return {"recommendation": "normal", "risk_multiplier": 1.0}
    
    def analyze_market_timing(self) -> Dict:
        """Analisa timing do mercado"""
        try:
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            
            # Horários de maior volatilidade (UTC)
            high_volatility_hours = [8, 9, 10, 13, 14, 15, 16, 17, 20, 21]
            
            is_high_volatility_time = hour in high_volatility_hours
            is_weekend = day_of_week >= 5
            
            # Análise de sessões de mercado
            if 7 <= hour <= 16:
                session = "european"
                activity_level = "high"
            elif 13 <= hour <= 22:
                session = "american"
                activity_level = "high"
            elif 22 <= hour <= 7:
                session = "asian"
                activity_level = "medium"
            else:
                session = "overlap"
                activity_level = "very_high"
            
            return {
                "session": session,
                "activity_level": activity_level,
                "is_high_volatility_time": is_high_volatility_time,
                "is_weekend": is_weekend,
                "hour": hour,
                "optimal_for_trading": is_high_volatility_time and not is_weekend
            }
        except Exception as e:
            logger.error(f"Erro na análise de timing: {e}")
            return {"session": "unknown", "activity_level": "medium"}
    
    def calculate_confidence(self, analyses: List[Dict]) -> float:
        """Calcula confiança geral da análise"""
        try:
            confidence_factors = []
            
            for analysis in analyses:
                if isinstance(analysis, dict):
                    # Trend confidence
                    if 'strength' in analysis:
                        confidence_factors.append(analysis['strength'])
                    
                    # Volatility confidence
                    if 'level' in analysis and analysis['level'] != 'high':
                        confidence_factors.append(0.8)
                    elif 'level' in analysis:
                        confidence_factors.append(0.6)  # High volatility reduces confidence
                    
                    # Pattern confidence
                    if 'pattern_count' in analysis:
                        pattern_confidence = min(0.9, analysis['pattern_count'] * 0.3)
                        confidence_factors.append(pattern_confidence)
                    
                    # Market timing confidence
                    if 'optimal_for_trading' in analysis:
                        timing_confidence = 0.9 if analysis['optimal_for_trading'] else 0.6
                        confidence_factors.append(timing_confidence)
                    
                    # Martingale adjustment
                    if 'should_be_conservative' in analysis:
                        martingale_confidence = 0.5 if analysis['should_be_conservative'] else 0.8
                        confidence_factors.append(martingale_confidence)
            
            if not confidence_factors:
                return 0.6  # Default moderate confidence
            
            # Weighted average
            base_confidence = sum(confidence_factors) / len(confidence_factors)
            
            # Normalize to 60-95% range
            normalized_confidence = 0.6 + (base_confidence * 0.35)
            
            return min(0.95, max(0.6, normalized_confidence))
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança: {e}")
            return 0.7
    
    def get_recommendation(self, confidence: float, context: Dict) -> str:
        """Gera recomendação baseada na análise"""
        try:
            martingale_level = context.get('martingaleLevel', 0)
            is_after_loss = context.get('isAfterLoss', False)
            
            if confidence < 0.65:
                return "wait_for_better_setup"
            
            if martingale_level > 4:
                return "high_risk_pause"
            
            if is_after_loss and confidence < 0.8:
                return "conservative_entry"
            
            if confidence > 0.85:
                return "strong_signal"
            
            return "moderate_entry"
            
        except Exception as e:
            logger.error(f"Erro na recomendação: {e}")
            return "moderate_entry"
    
    def get_fallback_analysis(self, symbol: str, context: Dict) -> Dict:
        """Análise de fallback quando dados não estão disponíveis"""
        return {
            "message": f"Análise básica do {symbol}: Dados limitados, usando análise conservadora",
            "trend": {"direction": "neutral", "strength": 0.5},
            "volatility": {"level": "medium", "atr_percentage": 1.5},
            "confidence": 0.6,
            "recommendation": "conservative_entry",
            "fallback": True,
            "timestamp": datetime.now().isoformat()
        }

# Instância global do engine
ml_engine = TradingMLEngine()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "Trading ML API",
        "version": "1.0.0",
        "model_trained": ml_engine.is_trained,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
@app.route('/analysis', methods=['POST'])
@app.route('/market-analysis', methods=['POST'])
async def analyze_market():
    """Endpoint principal de análise de mercado"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', 'R_50')
        context = {
            'martingaleLevel': data.get('martingaleLevel', 0),
            'isAfterLoss': data.get('isAfterLoss', False),
            'winRate': data.get('winRate', 50),
            'currentPrice': data.get('currentPrice', 1000),
            'balance': data.get('balance', 1000),
            'volatility': data.get('volatility', 50)
        }
        
        # Realizar análise
        analysis = await ml_engine.analyze_market(symbol, context)
        
        # Salvar análise na API de dados
        try:
            analysis_data = {
                'symbol': symbol,
                'analysis': analysis,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            requests.post(f"{DATA_API_URL}/save-analysis", json=analysis_data)
        except:
            pass  # Não falhar se não conseguir salvar
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({
            "error": "Erro na análise",
            "message": "Usando análise básica",
            "confidence": 0.6,
            "trend": "neutral",
            "fallback": True
        }), 200

@app.route('/signal', methods=['POST'])
@app.route('/trading-signal', methods=['POST'])
@app.route('/get-signal', methods=['POST'])
async def get_trading_signal():
    """Endpoint para sinais de trading"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', 'R_50')
        context = {
            'martingaleLevel': data.get('martingaleLevel', 0),
            'isAfterLoss': data.get('isAfterLoss', False),
            'winRate': data.get('winRate', 50),
            'recentTrades': data.get('recentTrades', [])
        }
        
        # Obter análise do mercado
        analysis = await ml_engine.analyze_market(symbol, context)
        
        # Determinar direção baseada na análise
        direction = "CALL"
        confidence = analysis.get('confidence', 0.7)
        
        # Lógica de direção baseada na tendência
        trend = analysis.get('trend', {})
        if trend.get('direction') == 'bearish' and trend.get('strength', 0) > 0.6:
            direction = "PUT"
        elif trend.get('direction') == 'bullish' and trend.get('strength', 0) > 0.6:
            direction = "CALL"
        else:
            # Use RSI se tendência não for clara
            patterns = analysis.get('patterns', {}).get('detected_patterns', [])
            for pattern in patterns:
                if pattern['type'] == 'rsi_overbought':
                    direction = "PUT"
                elif pattern['type'] == 'rsi_oversold':
                    direction = "CALL"
        
        # Ajustar confiança baseada no contexto Martingale
        martingale_context = analysis.get('martingale_context', {})
        if martingale_context.get('should_be_conservative'):
            confidence *= 0.85
        
        # Gerar reasoning
        reasoning = f"Baseado em análise técnica: tendência {trend.get('direction', 'neutral')}"
        if context['isAfterLoss']:
            reasoning += " (análise conservadora pós-perda)"
        
        signal_data = {
            'direction': direction,
            'confidence': confidence * 100,
            'reasoning': reasoning,
            'timeframe': '5m',
            'entry_price': data.get('currentPrice', 1000),
            'analysis': analysis,
            'martingaleLevel': context['martingaleLevel'],
            'isAfterLoss': context['isAfterLoss']
        }
        
        # Salvar sinal na API de dados
        try:
            requests.post(f"{DATA_API_URL}/save-signal", json={
                'symbol': symbol,
                'signal': signal_data,
                'timestamp': datetime.now().isoformat()
            })
        except:
            pass
        
        return jsonify(signal_data)
        
    except Exception as e:
        logger.error(f"Erro no sinal: {e}")
        return jsonify({
            "direction": "CALL",
            "confidence": 65,
            "reasoning": "Sinal básico - análise limitada",
            "fallback": True
        }), 200

@app.route('/risk', methods=['POST'])
@app.route('/risk-assessment', methods=['POST'])
async def assess_risk():
    """Endpoint para avaliação de risco"""
    try:
        data = request.get_json()
        
        martingale_level = data.get('martingaleLevel', 0)
        balance = data.get('currentBalance', 1000)
        today_pnl = data.get('todayPnL', 0)
        win_rate = data.get('winRate', 50)
        recent_trades = data.get('recentTrades', [])
        
        # Calcular nível de risco
        risk_score = 0
        risk_factors = []
        
        # Risco do Martingale
        if martingale_level > 4:
            risk_score += 40
            risk_factors.append(f"Martingale alto (nível {martingale_level})")
        elif martingale_level > 2:
            risk_score += 20
            risk_factors.append(f"Martingale moderado (nível {martingale_level})")
        
        # Risco do P&L
        if today_pnl < -balance * 0.1:
            risk_score += 30
            risk_factors.append("Perda diária significativa")
        elif today_pnl < 0:
            risk_score += 10
            risk_factors.append("P&L negativo hoje")
        
        # Risco da taxa de acerto
        if win_rate < 30:
            risk_score += 25
            risk_factors.append("Taxa de acerto muito baixa")
        elif win_rate < 45:
            risk_score += 10
            risk_factors.append("Taxa de acerto baixa")
        
        # Risco de trades recentes
        if len(recent_trades) >= 3:
            recent_losses = sum(1 for t in recent_trades[-3:] if t.get('status') == 'lost')
            if recent_losses >= 2:
                risk_score += 15
                risk_factors.append("Perdas recentes consecutivas")
        
        # Determinar nível
        if risk_score >= 60:
            level = "high"
            recommendation = "Pausar operações e revisar estratégia"
        elif risk_score >= 30:
            level = "medium"
            recommendation = "Operar com extrema cautela"
        else:
            level = "low"
            recommendation = "Continuar operando normalmente"
        
        risk_assessment = {
            'level': level,
            'score': risk_score,
            'message': f"Risco {level} detectado (score: {risk_score})",
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'martingaleLevel': martingale_level,
            'suggestions': self.get_risk_suggestions(level, risk_factors)
        }
        
        # Salvar avaliação na API de dados
        try:
            requests.post(f"{DATA_API_URL}/save-risk-assessment", json={
                'assessment': risk_assessment,
                'timestamp': datetime.now().isoformat()
            })
        except:
            pass
        
        return jsonify(risk_assessment)
        
    except Exception as e:
        logger.error(f"Erro na avaliação de risco: {e}")
        return jsonify({
            "level": "medium",
            "score": 50,
            "message": "Avaliação básica de risco",
            "recommendation": "Operar com cautela",
            "fallback": True
        }), 200

def get_risk_suggestions(level: str, risk_factors: List[str]) -> List[str]:
    """Gera sugestões baseadas no nível de risco"""
    suggestions = []
    
    if level == "high":
        suggestions.extend([
            "Pare todas as operações por pelo menos 30 minutos",
            "Revise sua estratégia de trading",
            "Considere reduzir o valor base das apostas",
            "Analise os últimos trades para identificar padrões de erro"
        ])
    elif level == "medium":
        suggestions.extend([
            "Reduza o valor das próximas apostas",
            "Aguarde sinais mais fortes antes de operar",
            "Monitore de perto o nível do Martingale",
            "Considere fazer uma pausa se o risco aumentar"
        ])
    else:
        suggestions.extend([
            "Continue operando com sua estratégia atual",
            "Mantenha disciplina no gerenciamento de risco",
            "Monitore regularmente as métricas de performance"
        ])
    
    return suggestions

@app.route('/train', methods=['POST'])
async def train_model():
    """Endpoint para treinar o modelo com novos dados"""
    try:
        # Obter dados históricos
        performance_data = await ml_engine.get_historical_performance()
        
        # Treinar modelo
        ml_engine.train_with_historical_data(performance_data)
        
        return jsonify({
            "status": "success",
            "message": "Modelo treinado com sucesso",
            "trades_used": len(performance_data.get('trades', [])),
            "model_trained": ml_engine.is_trained
        })
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        return jsonify({
            "status": "error",
            "message": "Erro no treinamento do modelo"
        }), 500

@app.route('/model-status', methods=['GET'])
def model_status():
    """Status do modelo"""
    return jsonify({
        "is_trained": ml_engine.is_trained,
        "feature_count": len(ml_engine.feature_columns),
        "model_type": type(ml_engine.model).__name__,
        "last_updated": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Treinar modelo na inicialização se houver dados
    try:
        asyncio.run(ml_engine.get_historical_performance())
    except:
        pass
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
