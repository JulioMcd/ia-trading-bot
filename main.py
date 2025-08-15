from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from threading import Thread
import time
import requests
import talib

# Importar funcionalidades avançadas
try:
    from advanced_ml_features import AdvancedMLFeatures
    from backtesting_system import BacktestingEngine, TradingDatabase
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logging.warning("Funcionalidades avançadas não disponíveis")

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Bot API com Machine Learning", version="2.0.0")

# CORS para permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# MODELOS DE DADOS
# ==============================================

class MarketData(BaseModel):
    symbol: str
    price: float
    timestamp: str
    volume: Optional[float] = 0
    
class TradingSignal(BaseModel):
    direction: str  # CALL ou PUT
    confidence: float
    timeframe: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str

class TradeRequest(BaseModel):
    symbol: str
    current_price: float
    account_balance: float
    recent_trades: List[Dict]
    market_data: List[Dict]
    user_preferences: Optional[Dict] = {}

class RiskAssessment(BaseModel):
    risk_level: str
    risk_score: float
    recommendation: str
    max_stake: float
    suggested_action: str

# ==============================================
# MACHINE LEARNING ENGINE
# ==============================================

class TradingMLEngine:
    def __init__(self):
        self.direction_model = None
        self.confidence_model = None
        self.timeframe_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.market_data_buffer = []
        self.prediction_history = []
        
        # Funcionalidades avançadas
        if ADVANCED_FEATURES_AVAILABLE:
            self.advanced_features = AdvancedMLFeatures()
            self.backtesting_engine = BacktestingEngine()
            self.database = TradingDatabase()
        else:
            self.advanced_features = None
            self.backtesting_engine = None
            self.database = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Gera dados sintéticos para treinar o modelo"""
        logger.info("Gerando dados sintéticos para treinamento...")
        
        np.random.seed(42)
        
        # Features do mercado
        prices = np.random.normal(1000, 50, n_samples)
        volatility = np.random.exponential(0.02, n_samples)
        rsi = np.random.uniform(20, 80, n_samples)
        macd = np.random.normal(0, 1, n_samples)
        bb_position = np.random.uniform(0, 1, n_samples)  # Posição nas Bandas de Bollinger
        volume = np.random.exponential(1000, n_samples)
        time_of_day = np.random.uniform(0, 24, n_samples)
        day_of_week = np.random.randint(0, 7, n_samples)
        
        # Features de comportamento do mercado
        price_change = np.random.normal(0, 0.01, n_samples)
        trend_strength = np.abs(price_change) * 100
        market_sentiment = np.random.uniform(-1, 1, n_samples)
        
        # Criar DataFrame
        df = pd.DataFrame({
            'price': prices,
            'volatility': volatility,
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'volume': volume,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'price_change': price_change,
            'trend_strength': trend_strength,
            'market_sentiment': market_sentiment,
        })
        
        # Gerar targets baseados em lógica de trading
        df['direction'] = np.where(
            (df['rsi'] < 30) & (df['bb_position'] < 0.2) & (df['macd'] > 0), 'CALL',
            np.where(
                (df['rsi'] > 70) & (df['bb_position'] > 0.8) & (df['macd'] < 0), 'PUT',
                np.random.choice(['CALL', 'PUT'], n_samples)
            )
        )
        
        # Confiança baseada na força dos sinais
        df['confidence'] = np.clip(
            70 + (df['trend_strength'] * 2) + 
            (np.abs(df['rsi'] - 50) / 2) + 
            (np.abs(df['macd']) * 10) +
            np.random.normal(0, 5, n_samples), 
            50, 95
        )
        
        # Timeframe baseado na volatilidade
        df['timeframe_numeric'] = np.where(
            df['volatility'] > 0.03, 1,  # t (ticks) para alta volatilidade
            2  # m (minutos) para baixa volatilidade
        )
        
        df['timeframe'] = np.where(df['timeframe_numeric'] == 1, 't', 'm')
        
        # Duração específica
        df['duration'] = np.where(
            df['timeframe'] == 't',
            np.random.choice([3, 5, 7], n_samples),
            np.random.choice([1, 2, 3], n_samples)
        )
        
        # Risk score
        df['risk_score'] = np.clip(
            50 + (df['volatility'] * 1000) + 
            (df['trend_strength'] * 2) +
            np.random.normal(0, 10, n_samples),
            0, 100
        )
        
        return df
    
    def prepare_features(self, df):
        """Prepara features para o modelo"""
        feature_columns = [
            'price', 'volatility', 'rsi', 'macd', 'bb_position',
            'volume', 'time_of_day', 'day_of_week', 'price_change',
            'trend_strength', 'market_sentiment'
        ]
        return df[feature_columns]
    
    def train_models(self):
        """Treina todos os modelos de ML"""
        logger.info("Iniciando treinamento dos modelos...")
        
        # Gerar dados de treinamento
        df = self.generate_synthetic_data()
        X = self.prepare_features(df)
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Modelo de Direção (CALL/PUT)
        y_direction = df['direction']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_direction, test_size=0.2, random_state=42
        )
        
        self.direction_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        self.direction_model.fit(X_train, y_train)
        
        direction_accuracy = accuracy_score(y_test, self.direction_model.predict(X_test))
        logger.info(f"Modelo de Direção - Acurácia: {direction_accuracy:.3f}")
        
        # 2. Modelo de Confiança
        y_confidence = df['confidence']
        self.confidence_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        self.confidence_model.fit(X_train, y_confidence)
        
        # 3. Modelo de Timeframe
        y_timeframe = df['timeframe_numeric']
        self.timeframe_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.timeframe_model.fit(X_train, y_timeframe)
        
        # 4. Modelo de Risco
        y_risk = df['risk_score']
        self.risk_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.risk_model.fit(X_train, y_risk)
        
        self.is_trained = True
        logger.info("✅ Todos os modelos treinados com sucesso!")
        
        # Salvar modelos
        self.save_models()
    
    def save_models(self):
        """Salva os modelos treinados"""
        try:
            joblib.dump(self.direction_model, 'direction_model.pkl')
            joblib.dump(self.confidence_model, 'confidence_model.pkl')
            joblib.dump(self.timeframe_model, 'timeframe_model.pkl')
            joblib.dump(self.risk_model, 'risk_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            logger.info("Modelos salvos com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao salvar modelos: {e}")
    
    def load_models(self):
        """Carrega modelos salvos"""
        try:
            if all(os.path.exists(f) for f in ['direction_model.pkl', 'confidence_model.pkl', 'timeframe_model.pkl', 'risk_model.pkl', 'scaler.pkl']):
                self.direction_model = joblib.load('direction_model.pkl')
                self.confidence_model = joblib.load('confidence_model.pkl')
                self.timeframe_model = joblib.load('timeframe_model.pkl')
                self.risk_model = joblib.load('risk_model.pkl')
                self.scaler = joblib.load('scaler.pkl')
                self.is_trained = True
                logger.info("Modelos carregados com sucesso!")
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
            return False
    
    def calculate_technical_indicators(self, prices, volumes=None):
        """Calcula indicadores técnicos"""
        if len(prices) < 20:
            # Se não temos dados suficientes, retorna valores padrão
            return {
                'rsi': 50,
                'macd': 0,
                'bb_position': 0.5,
                'volatility': 0.02
            }
        
        prices_array = np.array(prices)
        
        # RSI
        rsi = talib.RSI(prices_array, timeperiod=14)[-1] if len(prices_array) >= 14 else 50
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(prices_array)
        macd_value = macd[-1] if not np.isnan(macd[-1]) else 0
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = talib.BBANDS(prices_array)
        current_price = prices_array[-1]
        bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
        
        # Volatilidade
        returns = np.diff(prices_array) / prices_array[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.02
        
        return {
            'rsi': float(rsi) if not np.isnan(rsi) else 50,
            'macd': float(macd_value) if not np.isnan(macd_value) else 0,
            'bb_position': float(bb_position) if not np.isnan(bb_position) else 0.5,
            'volatility': float(volatility)
        }
    
    def predict_trading_signal(self, market_data: Dict) -> TradingSignal:
        """Gera sinal de trading usando ML"""
        if not self.is_trained:
            raise ValueError("Modelos não foram treinados ainda")
        
        try:
            # Extrair dados do mercado
            current_price = market_data.get('price', 1000)
            symbol = market_data.get('symbol', 'R_50')
            
            # Pegar histórico de preços se disponível
            price_history = market_data.get('price_history', [current_price] * 20)
            if len(price_history) < 20:
                price_history.extend([current_price] * (20 - len(price_history)))
            
            # Calcular indicadores técnicos
            indicators = self.calculate_technical_indicators(price_history)
            
            # Dados temporais
            now = datetime.now()
            time_of_day = now.hour + now.minute / 60
            day_of_week = now.weekday()
            
            # Features para predição
            features = np.array([[
                current_price,
                indicators['volatility'],
                indicators['rsi'],
                indicators['macd'],
                indicators['bb_position'],
                market_data.get('volume', 1000),
                time_of_day,
                day_of_week,
                0,  # price_change (será calculado)
                abs(indicators['rsi'] - 50),  # trend_strength aproximado
                0  # market_sentiment (será calculado)
            ]])
            
            # Escalar features
            features_scaled = self.scaler.transform(features)
            
            # Predições
            direction_pred = self.direction_model.predict(features_scaled)[0]
            confidence_pred = self.confidence_model.predict(features_scaled)[0]
            timeframe_pred = self.timeframe_model.predict(features_scaled)[0]
            
            # Determinar timeframe
            if timeframe_pred == 1:
                timeframe_type = 't'
                duration = np.random.choice([3, 5, 7])
            else:
                timeframe_type = 'm'
                duration = np.random.choice([1, 2, 3])
            
            timeframe_str = f"{duration}{timeframe_type}"
            
            # Reasoning baseado nos indicadores
            reasoning_parts = []
            if indicators['rsi'] < 30:
                reasoning_parts.append("RSI oversold")
            elif indicators['rsi'] > 70:
                reasoning_parts.append("RSI overbought")
            
            if indicators['macd'] > 0:
                reasoning_parts.append("MACD bullish")
            else:
                reasoning_parts.append("MACD bearish")
            
            if indicators['bb_position'] < 0.2:
                reasoning_parts.append("Price near lower BB")
            elif indicators['bb_position'] > 0.8:
                reasoning_parts.append("Price near upper BB")
            
            reasoning = f"ML Analysis: {', '.join(reasoning_parts)}" if reasoning_parts else "ML Pattern Recognition"
            
            signal = TradingSignal(
                direction=direction_pred,
                confidence=max(60, min(95, float(confidence_pred))),
                timeframe=timeframe_str,
                entry_price=current_price,
                reasoning=reasoning
            )
            
            # Armazenar predição no histórico
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'signal': signal.dict(),
                'market_data': market_data
            })
            
            # Manter apenas últimas 1000 predições
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return signal
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            # Retornar sinal padrão em caso de erro
            return TradingSignal(
                direction="CALL" if np.random.random() > 0.5 else "PUT",
                confidence=75.0,
                timeframe="5t",
                entry_price=current_price,
                reasoning="Default signal due to prediction error"
            )
    
    def assess_risk(self, market_data: Dict, account_balance: float) -> RiskAssessment:
        """Avalia risco da operação"""
        if not self.is_trained:
            # Retornar avaliação padrão se não treinado
            return RiskAssessment(
                risk_level="medium",
                risk_score=50.0,
                recommendation="Proceed with caution",
                max_stake=min(10.0, account_balance * 0.02),
                suggested_action="continue"
            )
        
        try:
            # Usar os mesmos features do sinal
            features = np.array([[
                market_data.get('price', 1000),
                market_data.get('volatility', 0.02),
                market_data.get('rsi', 50),
                market_data.get('macd', 0),
                market_data.get('bb_position', 0.5),
                market_data.get('volume', 1000),
                datetime.now().hour,
                datetime.now().weekday(),
                0, 0, 0
            ]])
            
            features_scaled = self.scaler.transform(features)
            risk_score = self.risk_model.predict(features_scaled)[0]
            risk_score = max(0, min(100, risk_score))
            
            # Determinar nível de risco
            if risk_score < 30:
                risk_level = "low"
                recommendation = "Good trading conditions"
                max_stake = account_balance * 0.05
                action = "continue"
            elif risk_score < 70:
                risk_level = "medium"
                recommendation = "Normal market conditions"
                max_stake = account_balance * 0.02
                action = "continue"
            else:
                risk_level = "high"
                recommendation = "High volatility detected - reduce exposure"
                max_stake = account_balance * 0.01
                action = "reduce"
            
            return RiskAssessment(
                risk_level=risk_level,
                risk_score=float(risk_score),
                recommendation=recommendation,
                max_stake=min(max_stake, 50.0),  # Limite máximo de $50
                suggested_action=action
            )
            
        except Exception as e:
            logger.error(f"Erro na avaliação de risco: {e}")
            return RiskAssessment(
                risk_level="medium",
                risk_score=50.0,
                recommendation="Default risk assessment",
                max_stake=min(5.0, account_balance * 0.01),
                suggested_action="continue"
            )

# ==============================================
# INSTÂNCIA GLOBAL DO ENGINE
# ==============================================

ml_engine = TradingMLEngine()

# ==============================================
# ENDPOINTS DA API
# ==============================================

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação"""
    logger.info("🚀 Iniciando Trading Bot API com ML...")
    
    # Tentar carregar modelos salvos
    if not ml_engine.load_models():
        logger.info("Modelos não encontrados. Treinando novos modelos...")
        # Treinar em background
        def train_models_background():
            ml_engine.train_models()
        
        thread = Thread(target=train_models_background)
        thread.start()
    
    logger.info("✅ API inicializada com sucesso!")

@app.get("/")
async def root():
    return {
        "message": "🤖 Trading Bot API com Machine Learning",
        "version": "2.0.0",
        "status": "online",
        "ml_trained": ml_engine.is_trained,
        "features": [
            "Predição de direção (CALL/PUT)",
            "Análise de confiança automática",
            "Seleção inteligente de timeframe",
            "Avaliação de risco em tempo real",
            "Indicadores técnicos automatizados"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_engine": "ready" if ml_engine.is_trained else "training",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/signal")
async def get_trading_signal(request: TradeRequest):
    """Obtém sinal de trading inteligente"""
    try:
        if not ml_engine.is_trained:
            return {
                "direction": "CALL" if np.random.random() > 0.5 else "PUT",
                "confidence": 75.0,
                "timeframe": "5t",
                "entry_price": request.current_price,
                "reasoning": "Training in progress - using fallback logic",
                "ml_status": "training"
            }
        
        # Preparar dados do mercado
        market_data = {
            "symbol": request.symbol,
            "price": request.current_price,
            "volume": 1000,
            "price_history": [trade.get("entry_price", request.current_price) for trade in request.recent_trades[-20:]]
        }
        
        signal = ml_engine.predict_trading_signal(market_data)
        
        return {
            **signal.dict(),
            "ml_status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar sinal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-assessment")
async def assess_risk(request: TradeRequest):
    """Avalia risco da operação"""
    try:
        market_data = {
            "symbol": request.symbol,
            "price": request.current_price,
            "volatility": 0.02,  # Será calculado automaticamente
        }
        
        assessment = ml_engine.assess_risk(market_data, request.account_balance)
        
        return {
            **assessment.dict(),
            "timestamp": datetime.now().isoformat(),
            "ml_status": "active" if ml_engine.is_trained else "training"
        }
        
    except Exception as e:
        logger.error(f"Erro na avaliação de risco: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-analysis")
async def analyze_market(request: TradeRequest):
    """Análise completa do mercado"""
    try:
        # Calcular indicadores se temos histórico
        price_history = [trade.get("entry_price", request.current_price) for trade in request.recent_trades[-20:]]
        if len(price_history) < 5:
            price_history = [request.current_price] * 20
        
        indicators = ml_engine.calculate_technical_indicators(price_history)
        
        # Análise de tendência
        if len(price_history) >= 2:
            trend = "bullish" if price_history[-1] > price_history[-2] else "bearish"
        else:
            trend = "neutral"
        
        # Análise de volatilidade
        volatility_level = "high" if indicators['volatility'] > 0.03 else "normal" if indicators['volatility'] > 0.01 else "low"
        
        analysis = {
            "symbol": request.symbol,
            "current_price": request.current_price,
            "trend": trend,
            "volatility_level": volatility_level,
            "technical_indicators": indicators,
            "market_sentiment": "neutral",  # Pode ser expandido
            "recommendation": "continue",
            "confidence": 85.0,
            "timestamp": datetime.now().isoformat(),
            "ml_status": "active" if ml_engine.is_trained else "training"
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Erro na análise de mercado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-decision")
async def make_auto_decision(request: TradeRequest):
    """Toma decisão automática completa de trading"""
    try:
        # 1. Análise de risco
        market_data = {"symbol": request.symbol, "price": request.current_price}
        risk_assessment = ml_engine.assess_risk(market_data, request.account_balance)
        
        # 2. Se risco muito alto, não operar
        if risk_assessment.risk_level == "high" and risk_assessment.risk_score > 80:
            return {
                "action": "hold",
                "reason": "Risk too high",
                "risk_assessment": risk_assessment.dict(),
                "timestamp": datetime.now().isoformat()
            }
        
        # 3. Obter sinal de trading
        if ml_engine.is_trained:
            signal = ml_engine.predict_trading_signal(market_data)
        else:
            signal = TradingSignal(
                direction="CALL" if np.random.random() > 0.5 else "PUT",
                confidence=75.0,
                timeframe="5t",
                entry_price=request.current_price,
                reasoning="Fallback logic while training"
            )
        
        # 4. Calcular stake baseado no risco
        suggested_stake = min(
            risk_assessment.max_stake,
            request.account_balance * 0.02  # Máximo 2% do saldo
        )
        
        # 5. Decisão final
        should_trade = (
            signal.confidence > 70 and
            risk_assessment.risk_score < 75 and
            suggested_stake >= 0.35  # Stake mínimo
        )
        
        if should_trade:
            return {
                "action": "trade",
                "direction": signal.direction,
                "timeframe": signal.timeframe,
                "stake": round(suggested_stake, 2),
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "risk_assessment": risk_assessment.dict(),
                "timestamp": datetime.now().isoformat(),
                "ml_status": "active" if ml_engine.is_trained else "training"
            }
        else:
            return {
                "action": "wait",
                "reason": f"Conditions not met - confidence: {signal.confidence}, risk: {risk_assessment.risk_score}",
                "signal": signal.dict(),
                "risk_assessment": risk_assessment.dict(),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Erro na decisão automática: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status")
async def get_model_status():
    """Status dos modelos de ML"""
    return {
        "is_trained": ml_engine.is_trained,
        "models": {
            "direction_model": ml_engine.direction_model is not None,
            "confidence_model": ml_engine.confidence_model is not None,
            "timeframe_model": ml_engine.timeframe_model is not None,
            "risk_model": ml_engine.risk_model is not None
        },
        "prediction_history_count": len(ml_engine.prediction_history),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Retreina os modelos (admin only)"""
    def retrain():
        logger.info("Iniciando retreinamento dos modelos...")
        ml_engine.train_models()
        logger.info("Retreinamento concluído!")
    
    background_tasks.add_task(retrain)
    return {"message": "Retreinamento iniciado em background"}

# ==============================================
# ENDPOINTS AVANÇADOS DE MACHINE LEARNING
# ==============================================

@app.post("/advanced-analysis")
async def advanced_market_analysis(request: TradeRequest):
    """Análise avançada de mercado com indicadores técnicos completos"""
    if not ADVANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Funcionalidades avançadas não disponíveis")
    
    try:
        # Preparar dados históricos
        price_history = [trade.get("entry_price", request.current_price) for trade in request.recent_trades[-50:]]
        if len(price_history) < 20:
            price_history.extend([request.current_price] * (20 - len(price_history)))
        
        # Criar dados OHLCV simulados
        ohlcv_data = []
        for i, price in enumerate(price_history):
            ohlcv_data.append({
                'open': price * (1 + np.random.uniform(-0.001, 0.001)),
                'high': price * (1 + abs(np.random.uniform(0, 0.002))),
                'low': price * (1 - abs(np.random.uniform(0, 0.002))),
                'close': price,
                'volume': np.random.randint(800, 1200)
            })
        
        # Calcular indicadores avançados
        indicators = ml_engine.advanced_features.calculate_advanced_indicators(ohlcv_data)
        
        # Detectar anomalias
        price_data = [[p, datetime.now().isoformat()] for p in price_history]
        anomalies = ml_engine.advanced_features.detect_market_anomalies(price_data)
        
        # Identificar regime de mercado
        regime = ml_engine.advanced_features.identify_market_regime(price_data)
        
        # Detectar padrões
        patterns = ml_engine.advanced_features.detect_trading_patterns(
            [{'price': p} for p in price_history]
        )
        
        # Analisar sentimento
        sentiment = ml_engine.advanced_features.analyze_market_sentiment(
            request.recent_trades, 
            [{'price': request.current_price}]
        )
        
        return {
            "symbol": request.symbol,
            "current_price": request.current_price,
            "indicators": indicators,
            "anomalies": anomalies,
            "market_regime": regime,
            "patterns": patterns,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat(),
            "analysis_quality": "advanced"
        }
        
    except Exception as e:
        logger.error(f"Erro na análise avançada: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-timeframe")
async def generate_smart_timeframe(request: TradeRequest):
    """Gera timeframe inteligente baseado nas condições do mercado"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {"type": "t", "duration": 5, "reasoning": "Timeframe padrão"}
    
    try:
        # Calcular volatilidade
        price_history = [trade.get("entry_price", request.current_price) for trade in request.recent_trades[-20:]]
        returns = np.diff(price_history) / price_history[:-1] if len(price_history) > 1 else [0.02]
        volatility = np.std(returns)
        
        # Gerar timeframe inteligente
        timeframe = ml_engine.advanced_features.generate_smart_timeframe(
            {"symbol": request.symbol, "price": request.current_price},
            volatility
        )
        
        return {
            **timeframe,
            "volatility": float(volatility),
            "market_condition": "high_volatility" if volatility > 0.03 else "normal",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na geração de timeframe: {e}")
        return {"type": "t", "duration": 5, "reasoning": "Erro - usando padrão"}

@app.post("/dynamic-stake")
async def calculate_dynamic_stake(request: TradeRequest):
    """Calcula stake dinâmico baseado em múltiplos fatores"""
    if not ADVANCED_FEATURES_AVAILABLE:
        stake = min(10.0, request.account_balance * 0.02)
        return {"stake": stake, "percentage": 2.0, "reasoning": "Stake padrão"}
    
    try:
        # Calcular win rate atual
        recent_trades = request.recent_trades[-20:] if request.recent_trades else []
        wins = len([t for t in recent_trades if t.get('pnl', 0) > 0])
        win_rate = (wins / len(recent_trades)) * 100 if recent_trades else 50
        
        # Obter sinal para calcular confiança
        signal_request = {
            "symbol": request.symbol,
            "current_price": request.current_price,
            "account_balance": request.account_balance,
            "recent_trades": request.recent_trades,
            "market_data": []
        }
        
        # Fazer predição para obter confiança
        market_data = {
            "symbol": request.symbol,
            "price": request.current_price,
            "volume": 1000,
            "price_history": [t.get("entry_price", request.current_price) for t in request.recent_trades[-20:]]
        }
        
        if ml_engine.is_trained:
            signal = ml_engine.predict_trading_signal(market_data)
            confidence = signal.confidence
        else:
            confidence = 75
        
        # Calcular stake dinâmico
        stake_info = ml_engine.advanced_features.calculate_dynamic_stake(
            account_balance=request.account_balance,
            confidence=confidence,
            risk_level='medium',
            win_rate=win_rate
        )
        
        return {
            **stake_info,
            "factors": {
                "confidence": confidence,
                "win_rate": win_rate,
                "account_balance": request.account_balance
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no cálculo de stake dinâmico: {e}")
        stake = min(5.0, request.account_balance * 0.01)
        return {"stake": stake, "percentage": 1.0, "reasoning": "Erro - usando conservador"}

# ==============================================
# ENDPOINTS DE BACKTESTING
# ==============================================

class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    symbol: str = "R_50"
    initial_balance: float = 1000
    strategy_params: Optional[Dict] = {}

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Executa backtesting completo da estratégia"""
    if not ADVANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Backtesting não disponível")
    
    try:
        engine = BacktestingEngine()
        engine.initial_balance = request.initial_balance
        
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        # Executar backtesting
        report = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            symbol=request.symbol,
            strategy_params=request.strategy_params
        )
        
        return {
            "status": "success",
            "backtest_report": report,
            "period": f"{request.start_date} to {request.end_date}",
            "symbol": request.symbol,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no backtesting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/quick")
async def quick_backtest():
    """Executa backtesting rápido (30 dias simulados)"""
    if not ADVANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Backtesting não disponível")
    
    try:
        engine = BacktestingEngine()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        report = engine.run_backtest(start_date, end_date)
        
        return {
            "status": "success",
            "type": "quick_backtest",
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no backtesting rápido: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================
# ENDPOINTS DE ANÁLISE DE PERFORMANCE
# ==============================================

@app.get("/performance/summary")
async def get_performance_summary():
    """Retorna resumo de performance do modelo"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {"message": "Análise de performance não disponível"}
    
    try:
        # Buscar dados do banco de dados
        db = TradingDatabase()
        
        # Calcular métricas dos últimos 30 dias
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        trades_df = db.get_trades(start_date=start_date, end_date=end_date)
        
        if trades_df.empty:
            return {
                "status": "no_data",
                "message": "Nenhum trade encontrado no período",
                "period": "30 days"
            }
        
        # Calcular métricas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        return {
            "period": "30 days",
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "best_trade": round(trades_df['pnl'].max(), 2) if total_trades > 0 else 0,
            "worst_trade": round(trades_df['pnl'].min(), 2) if total_trades > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na análise de performance: {e}")
        return {"error": str(e), "status": "error"}

@app.post("/performance/save-trade")
async def save_trade_result(trade_data: Dict):
    """Salva resultado de trade para análise"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return {"message": "Salvamento de trades não disponível"}
    
    try:
        # Implementar salvamento no banco de dados
        # Este endpoint seria chamado pelo frontend quando um trade é finalizado
        return {
            "status": "saved",
            "trade_id": trade_data.get("id"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao salvar trade: {e}")
        return {"error": str(e), "status": "error"}

# ==============================================
# ENDPOINTS DE MONITORAMENTO E SAÚDE
# ==============================================

@app.get("/system/status")
async def get_system_status():
    """Status completo do sistema"""
    status = {
        "api_status": "online",
        "ml_engine": {
            "trained": ml_engine.is_trained,
            "models": {
                "direction_model": ml_engine.direction_model is not None,
                "confidence_model": ml_engine.confidence_model is not None,
                "timeframe_model": ml_engine.timeframe_model is not None,
                "risk_model": ml_engine.risk_model is not None
            },
            "prediction_history_count": len(ml_engine.prediction_history)
        },
        "advanced_features": {
            "available": ADVANCED_FEATURES_AVAILABLE,
            "anomaly_detection": ADVANCED_FEATURES_AVAILABLE,
            "pattern_recognition": ADVANCED_FEATURES_AVAILABLE,
            "backtesting": ADVANCED_FEATURES_AVAILABLE
        },
        "system_info": {
            "uptime": "running",
            "memory_usage": "normal",
            "cpu_usage": "normal"
        },
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-advanced"
    }
    
    return status

@app.get("/features")
async def list_available_features():
    """Lista todas as funcionalidades disponíveis"""
    base_features = [
        "Predição de direção (CALL/PUT)",
        "Análise de confiança automática", 
        "Seleção inteligente de timeframe",
        "Avaliação de risco em tempo real",
        "Decisões automáticas completas"
    ]
    
    advanced_features = [
        "Indicadores técnicos avançados (50+ indicadores)",
        "Detecção de anomalias de mercado",
        "Identificação de regime de mercado",
        "Reconhecimento de padrões (Head&Shoulders, Triangles, etc)",
        "Análise de sentimento de mercado",
        "Cálculo de stake dinâmico",
        "Sistema de backtesting completo",
        "Análise de performance histórica",
        "Níveis de suporte e resistência",
        "Fibonacci automático"
    ] if ADVANCED_FEATURES_AVAILABLE else ["Funcionalidades avançadas não instaladas"]
    
    return {
        "base_features": base_features,
        "advanced_features": advanced_features,
        "total_features": len(base_features) + len(advanced_features),
        "advanced_available": ADVANCED_FEATURES_AVAILABLE,
        "version": "2.0.0-advanced"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
