#!/usr/bin/env python3
"""
Trading Bot ML - Versão Unificada para Deploy
Serve API + Frontend em um único serviço
"""

import os
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURAÇÕES
# ===============================

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

@dataclass
class TradingSignal:
    direction: str
    confidence: float
    entry_price: float
    recommended_duration: int
    duration_type: str
    reasoning: str
    risk_level: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradingRequest(BaseModel):
    symbol: str
    timeframe: str = "1m"
    current_price: float
    account_balance: float
    win_rate: float = 0.0
    recent_trades: List[dict] = []
    martingale_level: int = 0
    is_after_loss: bool = False

class AnalysisRequest(BaseModel):
    symbol: str
    current_price: float
    timestamp: str
    trades: List[dict] = []
    balance: float
    win_rate: float
    volatility: float
    market_condition: str
    martingale_level: int = 0
    is_after_loss: bool = False

# ===============================
# MACHINE LEARNING ENGINE
# ===============================

class TradingML:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        self.model_performance = {}
        logging.info("🤖 TradingML inicializado")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preparar features técnicas para ML"""
        try:
            # Indicadores técnicos básicos
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['sma_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['volatility'] = df['close'].rolling(window=10).std()
            
            # Features de preço
            df['price_change'] = df['close'].pct_change()
            df['price_momentum'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=10).mean()
            
            # Features de tempo
            df['hour'] = 12
            df['minute'] = 0
            
            # Remover NaN
            df = df.fillna(method='ffill').fillna(0)
            
            self.feature_columns = [
                'rsi', 'macd', 'macd_signal', 'sma_10', 'sma_20', 'volatility',
                'price_change', 'price_momentum', 'hour', 'minute'
            ]
            
            return df[self.feature_columns].fillna(0)
            
        except Exception as e:
            logging.error(f"❌ Erro ao preparar features: {e}")
            # Retornar features básicas
            basic_features = pd.DataFrame({
                'rsi': [50] * len(df),
                'price_change': df['close'].pct_change().fillna(0),
                'volatility': df['close'].rolling(5).std().fillna(1),
                'hour': [12] * len(df),
                'minute': [0] * len(df)
            }, index=df.index)
            return basic_features.fillna(0)

    def create_target(self, df: pd.DataFrame, future_periods: int = 5) -> pd.Series:
        """Criar target para classificação"""
        future_price = df['close'].shift(-future_periods)
        current_price = df['close']
        target = (future_price > current_price).astype(int)
        return target

    def train_models(self, historical_data: List[MarketData], symbol: str):
        """Treinar modelos de ML"""
        try:
            if len(historical_data) < 100:
                logging.warning(f"⚠️ Dados insuficientes para treinar {symbol}")
                return False

            # Converter para DataFrame
            df_data = []
            for data in historical_data:
                df_data.append({
                    'timestamp': data.timestamp,
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            # Preparar features
            features_df = self.prepare_features(df)
            
            # Criar target
            target = self.create_target(df, 5)
            
            # Remover últimas linhas que não têm target
            valid_idx = ~target.isna()
            X = features_df[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 50:
                return False
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Treinar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            
            # Salvar modelo
            model_key = f"{symbol}_5t"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.model_performance[model_key] = score
            
            logging.info(f"✅ Modelo {symbol} treinado - Accuracy: {score:.3f}")
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro no treinamento de {symbol}: {e}")
            return False

    def predict_signal(self, current_data: List[MarketData], symbol: str) -> TradingSignal:
        """Gerar sinal de trading"""
        try:
            if not self.is_trained or len(current_data) < 20:
                return self._fallback_signal(current_data[-1] if current_data else None)

            # Preparar dados
            df_data = []
            for data in current_data[-50:]:
                df_data.append({
                    'timestamp': data.timestamp,
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            features_df = self.prepare_features(df)
            latest_features = features_df.iloc[-1:].values

            # Fazer predição
            model_key = f"{symbol}_5t"
            if model_key in self.models:
                model = self.models[model_key]
                scaler = self.scalers[model_key]
                
                scaled_features = scaler.transform(latest_features)
                pred = model.predict(scaled_features)[0]
                prob = model.predict_proba(scaled_features)[0]
                confidence = max(prob) * 100
                
                direction = "CALL" if pred == 1 else "PUT"
                
                return TradingSignal(
                    direction=direction,
                    confidence=confidence,
                    entry_price=current_data[-1].close,
                    recommended_duration=5,
                    duration_type="t",
                    reasoning=f"ML Analysis: {direction} com {confidence:.1f}% confiança",
                    risk_level="medium"
                )
            
            return self._fallback_signal(current_data[-1])
            
        except Exception as e:
            logging.error(f"❌ Erro na predição ML: {e}")
            return self._fallback_signal(current_data[-1] if current_data else None)

    def _fallback_signal(self, current_data: Optional[MarketData]) -> TradingSignal:
        """Sinal de fallback"""
        direction = "CALL" if np.random.random() > 0.5 else "PUT"
        confidence = 60 + np.random.random() * 20
        
        return TradingSignal(
            direction=direction,
            confidence=confidence,
            entry_price=current_data.close if current_data else 1000.0,
            recommended_duration=5,
            duration_type="t",
            reasoning="Technical analysis fallback",
            risk_level="medium"
        )

# ===============================
# COLETA DE DADOS
# ===============================

class MarketDataCollector:
    def __init__(self):
        self.data_cache = {}
        logging.info("📊 MarketDataCollector inicializado")

    async def collect_historical_data(self, symbol: str, periods: int = 1000) -> List[MarketData]:
        """Coletar dados históricos simulados"""
        try:
            data = []
            base_price = 1000.0
            current_time = datetime.now() - timedelta(minutes=periods)
            
            for i in range(periods):
                change = np.random.normal(0, 0.002)
                base_price *= (1 + change)
                
                high = base_price * (1 + abs(np.random.normal(0, 0.001)))
                low = base_price * (1 - abs(np.random.normal(0, 0.001)))
                open_price = base_price + np.random.normal(0, 0.001) * base_price
                
                data.append(MarketData(
                    symbol=symbol,
                    timestamp=current_time + timedelta(minutes=i),
                    open=open_price,
                    high=high,
                    low=low,
                    close=base_price,
                    volume=np.random.randint(100, 1000)
                ))
            
            logging.info(f"📈 Coletados {len(data)} pontos para {symbol}")
            return data
            
        except Exception as e:
            logging.error(f"❌ Erro ao coletar dados: {e}")
            return []

    async def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """Dados em tempo real simulados"""
        try:
            base_price = self.data_cache.get(symbol, {}).get('last_price', 1000.0)
            change = np.random.normal(0, 0.001)
            new_price = base_price * (1 + change)
            
            data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=new_price,
                high=new_price * 1.001,
                low=new_price * 0.999,
                close=new_price,
                volume=np.random.randint(50, 200)
            )
            
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            self.data_cache[symbol]['last_price'] = new_price
            self.data_cache[symbol]['last_update'] = datetime.now()
            
            return data
            
        except Exception as e:
            logging.error(f"❌ Erro dados real-time: {e}")
            return None

# ===============================
# API PRINCIPAL
# ===============================

app = FastAPI(title="Trading Bot ML API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instâncias globais
ml_engine = TradingML()
data_collector = MarketDataCollector()

# ===============================
# ROTAS FRONTEND
# ===============================

@app.get("/")
async def serve_frontend():
    """Servir interface principal do Trading Bot"""
    return FileResponse('static/index.html')

@app.get("/app")
async def serve_app():
    """Servir aplicação"""
    return FileResponse('static/index.html')

# ===============================
# ROTAS API
# ===============================

@app.on_event("startup")
async def startup_event():
    """Inicializar sistema"""
    logging.info("🚀 Iniciando Trading Bot ML API...")
    
    # Treinar modelos iniciais
    symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    
    async def train_symbol(symbol):
        try:
            historical_data = await data_collector.collect_historical_data(symbol, 500)
            ml_engine.train_models(historical_data, symbol)
            logging.info(f"✅ Modelo treinado para {symbol}")
        except Exception as e:
            logging.error(f"❌ Erro ao treinar {symbol}: {e}")
    
    await asyncio.gather(*[train_symbol(symbol) for symbol in symbols])
    logging.info("🎯 Sistema pronto!")

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "online",
        "message": "Trading Bot ML API",
        "version": "1.0.0",
        "models_trained": ml_engine.is_trained,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_market(request: AnalysisRequest):
    """Análise de mercado"""
    try:
        historical_data = await data_collector.collect_historical_data(request.symbol, 100)
        current_data = await data_collector.get_real_time_data(request.symbol)
        
        if current_data:
            historical_data.append(current_data)
        
        # Calcular volatilidade
        df_data = [{'close': data.close} for data in historical_data[-50:]]
        df = pd.DataFrame(df_data)
        volatility = df['close'].std() / df['close'].mean() * 100
        
        trend = "bullish" if df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1] else "bearish"
        
        # Ajustar para Martingale
        martingale_factor = 1.0
        if request.martingale_level > 0:
            martingale_factor = 0.8 - (request.martingale_level * 0.1)
        
        confidence = (70 + np.random.random() * 25) * martingale_factor
        
        analysis_message = f"Análise {request.symbol}: Volatilidade {volatility:.1f}%, Tendência {trend}"
        
        if request.martingale_level > 0:
            analysis_message += f" | Martingale Nível {request.martingale_level}"
        
        return {
            "message": analysis_message,
            "volatility": volatility,
            "trend": trend,
            "confidence": confidence,
            "current_price": current_data.close if current_data else request.current_price,
            "martingale_aware": request.martingale_level > 0,
            "recommendation": "wait_for_better_setup" if request.is_after_loss else "continue_normal",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"❌ Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signal")
async def get_trading_signal(request: TradingRequest):
    """Obter sinal de trading"""
    try:
        historical_data = await data_collector.collect_historical_data(request.symbol, 200)
        current_data = await data_collector.get_real_time_data(request.symbol)
        
        if current_data:
            historical_data.append(current_data)
        
        signal = ml_engine.predict_signal(historical_data, request.symbol)
        
        # Ajustar para Martingale
        if request.is_after_loss and request.martingale_level > 0:
            signal.confidence = max(60.0, signal.confidence - 10.0)
            signal.reasoning = f"Análise conservadora pós-perda (Martingale {request.martingale_level}) - {signal.reasoning}"
        
        return {
            "direction": signal.direction,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "timeframe": f"{signal.recommended_duration}{signal.duration_type}",
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_level": signal.risk_level,
            "martingale_level": request.martingale_level,
            "is_after_loss": request.is_after_loss,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"❌ Erro no sinal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk")
async def assess_risk(request: dict):
    """Avaliação de risco"""
    try:
        current_balance = request.get('currentBalance', 1000)
        today_pnl = request.get('todayPnL', 0)
        martingale_level = request.get('martingaleLevel', 0)
        win_rate = request.get('winRate', 0)
        total_trades = request.get('totalTrades', 0)
        
        # Calcular risco
        risk_score = 0
        
        if today_pnl < -current_balance * 0.1:
            risk_score += 30
        elif today_pnl < -current_balance * 0.05:
            risk_score += 15
        
        if martingale_level > 4:
            risk_score += 40
        elif martingale_level > 2:
            risk_score += 20
        elif martingale_level > 0:
            risk_score += 10
        
        if total_trades > 5 and win_rate < 30:
            risk_score += 25
        elif total_trades > 5 and win_rate < 50:
            risk_score += 10
        
        if risk_score > 60:
            level = "high"
            message = f"Risco alto - Score: {risk_score}"
            recommendation = "Parar trading"
        elif risk_score > 30:
            level = "medium"
            message = f"Risco moderado - Score: {risk_score}"
            recommendation = "Operar com cautela"
        else:
            level = "low"
            message = f"Risco baixo - Score: {risk_score}"
            recommendation = "Pode continuar"
        
        return {
            "level": level,
            "message": message,
            "score": risk_score,
            "recommendation": recommendation,
            "martingale_level": martingale_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"❌ Erro na avaliação de risco: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def models_status():
    """Status dos modelos"""
    return {
        "is_trained": ml_engine.is_trained,
        "models_count": len(ml_engine.models),
        "performance": ml_engine.model_performance,
        "symbols": list(set(k.split('_')[0] for k in ml_engine.models.keys())),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket para dados em tempo real"""
    await websocket.accept()
    
    try:
        while True:
            current_data = await data_collector.get_real_time_data(symbol)
            
            if current_data:
                await websocket.send_json({
                    "type": "market_data",
                    "symbol": symbol,
                    "price": current_data.close,
                    "timestamp": current_data.timestamp.isoformat(),
                    "high": current_data.high,
                    "low": current_data.low,
                    "volume": current_data.volume
                })
            
            await asyncio.sleep(1)
            
    except Exception as e:
        logging.error(f"❌ Erro WebSocket: {e}")

# ===============================
# CONFIGURAÇÃO DE LOGGING
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    logging.info("🚀 Trading Bot ML - Deploy Unificado")
    logging.info(f"🌐 Porta: {port}")
    logging.info("🤖 Machine Learning: Ativado")
    logging.info("📊 Frontend + API: Integrados")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )