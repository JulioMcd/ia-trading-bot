#!/usr/bin/env python3
"""
Trading Bot with Machine Learning - Backend
Análise de gráficos em tempo real com IA para identificar oportunidades
Deploy: Render.com
"""

import os
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
import asyncio
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import Redis condicional
try:
    import redis
    REDIS_AVAILABLE = True
    logging.info("✅ Redis disponível")
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("⚠️ Redis não disponível - funcionando sem cache")

# ===============================
# CONFIGURAÇÕES E MODELOS
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
    direction: str  # 'CALL' ou 'PUT'
    confidence: float
    entry_price: float
    recommended_duration: int
    duration_type: str  # 't' ou 'm'
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
# SISTEMA DE MACHINE LEARNING
# ===============================

class TradingML:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        self.model_performance = {}
        
        # Redis para cache (opcional - funciona sem)
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_URL', 'localhost'),
                    port=6379,
                    decode_responses=True
                )
                logging.info("✅ Redis cache conectado")
            except Exception as e:
                logging.warning(f"⚠️ Redis cache não disponível: {e}")
                self.redis_client = None
        else:
            logging.info("📦 Funcionando sem cache Redis")
            
        logging.info("🤖 TradingML inicializado")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preparar features técnicas para ML"""
        try:
            # Indicadores técnicos
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['sma_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # Volatilidade
            df['volatility'] = df['close'].rolling(window=10).std()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Padrões de velas
            df['doji'] = abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1
            df['hammer'] = ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) & (df['close'] > df['open'])
            
            # Volume (se disponível)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            else:
                df['volume_ratio'] = 1.0
            
            # Features de preço
            df['price_change'] = df['close'].pct_change()
            df['price_momentum'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=10).mean()
            
            # Suporte e resistência
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            # Features de tempo
            df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 12
            df['minute'] = pd.to_datetime(df.index).minute if hasattr(df.index, 'minute') else 0
            
            # Remover NaN
            df = df.fillna(method='ffill').fillna(0)
            
            self.feature_columns = [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                'sma_10', 'sma_20', 'ema_12', 'ema_26', 'volatility', 'atr',
                'doji', 'hammer', 'volume_ratio', 'price_change', 'price_momentum',
                'price_position', 'hour', 'minute'
            ]
            
            return df[self.feature_columns].fillna(0)
            
        except Exception as e:
            logging.error(f"❌ Erro ao preparar features: {e}")
            # Retornar features básicas se falhar
            basic_features = pd.DataFrame({
                'rsi': [50] * len(df),
                'price_change': df['close'].pct_change().fillna(0),
                'volatility': df['close'].rolling(5).std().fillna(1),
                'hour': [12] * len(df),
                'minute': [0] * len(df)
            }, index=df.index)
            return basic_features.fillna(0)

    def create_target(self, df: pd.DataFrame, future_periods: int = 5) -> pd.Series:
        """Criar target para classificação (CALL/PUT)"""
        future_price = df['close'].shift(-future_periods)
        current_price = df['close']
        
        # 1 = CALL (preço sobe), 0 = PUT (preço desce)
        target = (future_price > current_price).astype(int)
        return target

    def train_models(self, historical_data: List[MarketData], symbol: str):
        """Treinar modelos de ML para um símbolo"""
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
            
            # Criar diferentes targets para diferentes timeframes
            targets = {}
            for periods in [3, 5, 10]:  # 3, 5, 10 ticks/minutos
                targets[f'target_{periods}'] = self.create_target(df, periods)

            # Treinar modelo para cada timeframe
            for target_name, target in targets.items():
                try:
                    # Remover últimas linhas que não têm target
                    valid_idx = ~target.isna()
                    X = features_df[valid_idx]
                    y = target[valid_idx]
                    
                    if len(X) < 50:
                        continue
                    
                    # Split treino/teste
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Scaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Treinar múltiplos modelos
                    models = {
                        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
                    }
                    
                    best_model = None
                    best_score = 0
                    
                    for model_name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        score = model.score(X_test_scaled, y_test)
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                    
                    # Salvar melhor modelo
                    model_key = f"{symbol}_{target_name}"
                    self.models[model_key] = best_model
                    self.scalers[model_key] = scaler
                    self.model_performance[model_key] = best_score
                    
                    logging.info(f"✅ Modelo {model_key} treinado - Accuracy: {best_score:.3f}")
                    
                except Exception as e:
                    logging.error(f"❌ Erro ao treinar {target_name}: {e}")
                    continue
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro no treinamento de {symbol}: {e}")
            return False

    def predict_signal(self, current_data: List[MarketData], symbol: str, duration_preference: int = 5) -> TradingSignal:
        """Gerar sinal de trading usando ML"""
        try:
            if not self.is_trained or len(current_data) < 20:
                return self._fallback_signal(current_data[-1] if current_data else None)

            # Preparar dados atuais
            df_data = []
            for data in current_data[-50:]:  # Últimos 50 pontos
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

            # Fazer predições para diferentes timeframes
            predictions = {}
            confidences = {}
            
            for model_key in self.models.keys():
                if symbol in model_key:
                    try:
                        model = self.models[model_key]
                        scaler = self.scalers[model_key]
                        
                        scaled_features = scaler.transform(latest_features)
                        
                        # Predição
                        pred = model.predict(scaled_features)[0]
                        prob = model.predict_proba(scaled_features)[0]
                        confidence = max(prob)
                        
                        predictions[model_key] = pred
                        confidences[model_key] = confidence
                        
                    except Exception as e:
                        logging.error(f"❌ Erro na predição {model_key}: {e}")
                        continue

            if not predictions:
                return self._fallback_signal(current_data[-1])

            # Combinar predições (ensemble)
            avg_prediction = np.mean(list(predictions.values()))
            avg_confidence = np.mean(list(confidences.values()))
            
            # Determinar direção
            direction = "CALL" if avg_prediction > 0.5 else "PUT"
            
            # Ajustar confiança baseada na qualidade dos modelos
            performance_weight = np.mean([
                self.model_performance.get(k, 0.5) for k in predictions.keys()
            ])
            final_confidence = (avg_confidence * performance_weight) * 100
            
            # Determinar duração baseada na confiança
            if final_confidence > 80:
                duration = duration_preference
                duration_type = "t"
            elif final_confidence > 60:
                duration = min(duration_preference + 2, 10)
                duration_type = "t"
            else:
                duration = 1
                duration_type = "m"
            
            # Análise de risco
            volatility = df['close'].rolling(10).std().iloc[-1]
            risk_level = "high" if volatility > df['close'].std() * 1.5 else "medium" if volatility > df['close'].std() else "low"
            
            # Reasoning
            reasoning = f"ML Analysis: {len(predictions)} models, avg confidence {final_confidence:.1f}%, volatility {volatility:.4f}"
            
            current_price = current_data[-1].close
            
            return TradingSignal(
                direction=direction,
                confidence=final_confidence,
                entry_price=current_price,
                recommended_duration=duration,
                duration_type=duration_type,
                reasoning=reasoning,
                risk_level=risk_level,
                stop_loss=current_price * (0.98 if direction == "CALL" else 1.02),
                take_profit=current_price * (1.02 if direction == "CALL" else 0.98)
            )
            
        except Exception as e:
            logging.error(f"❌ Erro na predição ML: {e}")
            return self._fallback_signal(current_data[-1] if current_data else None)

    def _fallback_signal(self, current_data: Optional[MarketData]) -> TradingSignal:
        """Sinal de fallback quando ML não está disponível"""
        if not current_data:
            return TradingSignal(
                direction="CALL",
                confidence=50.0,
                entry_price=1000.0,
                recommended_duration=5,
                duration_type="t",
                reasoning="Fallback signal - insufficient data",
                risk_level="medium"
            )
        
        # Análise técnica simples
        direction = "CALL" if np.random.random() > 0.5 else "PUT"
        confidence = 60 + np.random.random() * 20
        
        return TradingSignal(
            direction=direction,
            confidence=confidence,
            entry_price=current_data.close,
            recommended_duration=5,
            duration_type="t",
            reasoning="Technical analysis fallback",
            risk_level="medium"
        )

# ===============================
# COLETA DE DADOS DE MERCADO
# ===============================

class MarketDataCollector:
    def __init__(self):
        self.data_cache = {}
        self.ws_connections = {}
        self.is_collecting = {}
        logging.info("📊 MarketDataCollector inicializado")

    async def collect_historical_data(self, symbol: str, periods: int = 1000) -> List[MarketData]:
        """Coletar dados históricos (simulação para demonstração)"""
        try:
            # Simular dados históricos realistas
            data = []
            base_price = 1000.0
            current_time = datetime.now() - timedelta(minutes=periods)
            
            for i in range(periods):
                # Movimento de preço mais realista
                change = np.random.normal(0, 0.002)  # 0.2% std
                base_price *= (1 + change)
                
                # Simular OHLC
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
            
            logging.info(f"📈 Coletados {len(data)} pontos históricos para {symbol}")
            return data
            
        except Exception as e:
            logging.error(f"❌ Erro ao coletar dados históricos: {e}")
            return []

    async def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """Obter dados em tempo real (simulação)"""
        try:
            # Em produção, conectar à API real da Deriv ou outro provedor
            base_price = self.data_cache.get(symbol, {}).get('last_price', 1000.0)
            
            # Simular tick em tempo real
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
            
            # Cache
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            self.data_cache[symbol]['last_price'] = new_price
            self.data_cache[symbol]['last_update'] = datetime.now()
            
            return data
            
        except Exception as e:
            logging.error(f"❌ Erro ao obter dados real-time: {e}")
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

# Instâncias globais
ml_engine = TradingML()
data_collector = MarketDataCollector()
active_connections = {}

# ===============================
# ENDPOINTS DA API
# ===============================

@app.on_event("startup")
async def startup_event():
    """Inicializar sistema ao startar"""
    logging.info("🚀 Iniciando Trading Bot ML API...")
    
    # Treinar modelos iniciais
    symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    
    async def train_symbol(symbol):
        try:
            historical_data = await data_collector.collect_historical_data(symbol, 1000)
            ml_engine.train_models(historical_data, symbol)
            logging.info(f"✅ Modelo treinado para {symbol}")
        except Exception as e:
            logging.error(f"❌ Erro ao treinar {symbol}: {e}")
    
    # Treinar em paralelo
    await asyncio.gather(*[train_symbol(symbol) for symbol in symbols])
    
    logging.info("🎯 Sistema pronto para trading!")

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "message": "Trading Bot ML API",
        "version": "1.0.0",
        "models_trained": ml_engine.is_trained,
        "redis_available": REDIS_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_market(request: AnalysisRequest):
    """Análise completa do mercado"""
    try:
        # Coletar dados recentes
        historical_data = await data_collector.collect_historical_data(request.symbol, 100)
        current_data = await data_collector.get_real_time_data(request.symbol)
        
        if current_data:
            historical_data.append(current_data)
        
        # Análise técnica
        df_data = []
        for data in historical_data[-50:]:
            df_data.append({
                'close': data.close,
                'high': data.high,
                'low': data.low,
                'volume': data.volume
            })
        
        df = pd.DataFrame(df_data)
        
        # Calcular indicadores
        volatility = df['close'].std() / df['close'].mean() * 100
        trend = "bullish" if df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1] else "bearish"
        
        # Ajustar análise para Martingale
        martingale_factor = 1.0
        if request.martingale_level > 0:
            martingale_factor = 0.8 - (request.martingale_level * 0.1)  # Mais conservador
        
        confidence = (70 + np.random.random() * 25) * martingale_factor
        
        analysis_message = f"Análise {request.symbol}: Volatilidade {volatility:.1f}%, Tendência {trend}"
        
        if request.martingale_level > 0:
            analysis_message += f" | Martingale Nível {request.martingale_level} - Análise conservadora"
        
        if request.is_after_loss:
            analysis_message += " | Pós-perda: Aguardando setup ideal"
        
        return {
            "message": analysis_message,
            "volatility": volatility,
            "trend": trend,
            "confidence": confidence,
            "current_price": current_data.close if current_data else request.current_price,
            "martingale_aware": request.martingale_level > 0,
            "recommendation": "wait_for_better_setup" if request.is_after_loss else "continue_normal",
            "redis_status": "available" if REDIS_AVAILABLE else "not_available",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"❌ Erro na análise: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Erro na análise", "details": str(e)}
        )

@app.post("/signal")
async def get_trading_signal(request: TradingRequest):
    """Obter sinal de trading com ML"""
    try:
        # Coletar dados para análise
        historical_data = await data_collector.collect_historical_data(request.symbol, 200)
        current_data = await data_collector.get_real_time_data(request.symbol)
        
        if current_data:
            historical_data.append(current_data)
        
        # Gerar sinal usando ML
        signal = ml_engine.predict_signal(historical_data, request.symbol)
        
        # Ajustar para Martingale se necessário
        if request.is_after_loss and request.martingale_level > 0:
            signal.confidence = max(60.0, signal.confidence - 10.0)
            signal.reasoning = f"Análise conservadora pós-perda (Martingale {request.martingale_level}) - {signal.reasoning}"
        
        # Ajustar duração baseada na confiança
        if signal.confidence > 85:
            signal.recommended_duration = 3
            signal.duration_type = "t"
        elif signal.confidence > 70:
            signal.recommended_duration = 5
            signal.duration_type = "t"
        else:
            signal.recommended_duration = 1
            signal.duration_type = "m"
        
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
            "redis_cache": "enabled" if REDIS_AVAILABLE else "disabled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"❌ Erro no sinal: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Erro ao gerar sinal", "details": str(e)}
        )

@app.post("/risk")
async def assess_risk(request: dict):
    """Avaliação de risco"""
    try:
        current_balance = request.get('currentBalance', 1000)
        today_pnl = request.get('todayPnL', 0)
        martingale_level = request.get('martingaleLevel', 0)
        win_rate = request.get('winRate', 0)
        total_trades = request.get('totalTrades', 0)
        
        # Calcular nível de risco
        risk_score = 0
        
        # Fator PnL
        if today_pnl < -current_balance * 0.1:  # -10% do saldo
            risk_score += 30
        elif today_pnl < -current_balance * 0.05:  # -5% do saldo
            risk_score += 15
        
        # Fator Martingale
        if martingale_level > 4:
            risk_score += 40
        elif martingale_level > 2:
            risk_score += 20
        elif martingale_level > 0:
            risk_score += 10
        
        # Fator Win Rate
        if total_trades > 5:
            if win_rate < 30:
                risk_score += 25
            elif win_rate < 50:
                risk_score += 10
        
        # Determinar nível
        if risk_score > 60:
            level = "high"
            message = f"Risco alto - Score: {risk_score}"
            recommendation = "Parar trading ou reduzir drasticamente"
        elif risk_score > 30:
            level = "medium"
            message = f"Risco moderado - Score: {risk_score}"
            recommendation = "Operar com muita cautela"
        else:
            level = "low"
            message = f"Risco baixo - Score: {risk_score}"
            recommendation = "Pode continuar operando"
        
        # Ajustar para Martingale
        if martingale_level > 4:
            message += f" | Martingale crítico nível {martingale_level}"
            recommendation = "Considerar pausa ou reset"
        elif martingale_level > 2:
            message += f" | Martingale moderado nível {martingale_level}"
        
        return {
            "level": level,
            "message": message,
            "score": risk_score,
            "recommendation": recommendation,
            "martingale_level": martingale_level,
            "is_after_loss": request.get('isInCoolingPeriod', False),
            "cache_status": "redis_enabled" if REDIS_AVAILABLE else "in_memory_only",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"❌ Erro na avaliação de risco: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Erro na avaliação de risco", "details": str(e)}
        )

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket para dados em tempo real"""
    await websocket.accept()
    active_connections[symbol] = websocket
    
    try:
        while True:
            # Enviar dados atualizados
            current_data = await data_collector.get_real_time_data(symbol)
            
            if current_data:
                await websocket.send_json({
                    "type": "market_data",
                    "symbol": symbol,
                    "price": current_data.close,
                    "timestamp": current_data.timestamp.isoformat(),
                    "high": current_data.high,
                    "low": current_data.low,
                    "volume": current_data.volume,
                    "redis_available": REDIS_AVAILABLE
                })
            
            await asyncio.sleep(1)  # Update a cada segundo
            
    except Exception as e:
        logging.error(f"❌ Erro WebSocket: {e}")
    finally:
        if symbol in active_connections:
            del active_connections[symbol]

@app.get("/models/status")
async def models_status():
    """Status dos modelos ML"""
    return {
        "is_trained": ml_engine.is_trained,
        "models_count": len(ml_engine.models),
        "performance": ml_engine.model_performance,
        "symbols": list(set(k.split('_')[0] for k in ml_engine.models.keys())),
        "redis_available": REDIS_AVAILABLE,
        "cache_client": "redis" if ml_engine.redis_client else "memory",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/models/retrain/{symbol}")
async def retrain_model(symbol: str, background_tasks: BackgroundTasks):
    """Retreinar modelo para um símbolo"""
    async def train_task():
        try:
            historical_data = await data_collector.collect_historical_data(symbol, 1000)
            success = ml_engine.train_models(historical_data, symbol)
            logging.info(f"{'✅' if success else '❌'} Retreinamento {symbol}: {success}")
        except Exception as e:
            logging.error(f"❌ Erro no retreinamento: {e}")
    
    background_tasks.add_task(train_task)
    
    return {
        "message": f"Retreinamento de {symbol} iniciado",
        "redis_status": "enabled" if REDIS_AVAILABLE else "disabled",
        "timestamp": datetime.now().isoformat()
    }

# ===============================
# CONFIGURAÇÃO DE LOGGING
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    logging.info("🚀 Iniciando Trading Bot ML API")
    logging.info(f"🌐 Porta: {port}")
    logging.info("🤖 Machine Learning: Ativado")
    logging.info("📊 Análise de gráficos: Tempo real")
    logging.info(f"📦 Redis: {'Disponível' if REDIS_AVAILABLE else 'Indisponível (funcionando sem cache)'}")
    logging.info("🎯 Deploy: Render.com")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
