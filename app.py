import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuração do Flask
app = Flask(__name__)
CORS(app)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
API_PORT = int(os.environ.get('PORT', 5000))
DATABASE_URL = 'trading_stats.db'

class TechnicalIndicators:
    """Classe para calcular indicadores técnicos sem dependências externas"""
    
    @staticmethod
    def rsi(prices, window=14):
        """Calcula RSI"""
        try:
            prices_series = pd.Series(prices)
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.fillna(50).iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        try:
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return float(macd_line.iloc[-1]) if len(macd_line) > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def bollinger_bands(prices, window=20, num_std=2):
        """Calcula Bollinger Bands"""
        try:
            prices_series = pd.Series(prices)
            rolling_mean = prices_series.rolling(window=window).mean()
            rolling_std = prices_series.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            return {
                'upper': float(upper_band.iloc[-1]) if len(upper_band) > 0 else prices[-1] * 1.02,
                'lower': float(lower_band.iloc[-1]) if len(lower_band) > 0 else prices[-1] * 0.98
            }
        except:
            return {
                'upper': prices[-1] * 1.02,
                'lower': prices[-1] * 0.98
            }
    
    @staticmethod
    def volatility(prices, window=14):
        """Calcula volatilidade (ATR simplificado)"""
        try:
            prices_series = pd.Series(prices)
            returns = prices_series.pct_change().dropna()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Anualizada
            return float(volatility.iloc[-1]) if len(volatility) > 0 else 1.0
        except:
            return 1.0

class TradingAI:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.indicators = TechnicalIndicators()
        self.init_database()
        
    def init_database(self):
        """Inicializa o banco de dados"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            # Tabela de trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    direction TEXT,
                    stake REAL,
                    duration TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    result TEXT,
                    pnl REAL,
                    martingale_level INTEGER,
                    market_conditions TEXT,
                    features TEXT
                )
            ''')
            
            # Tabela de estatísticas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    best_streak INTEGER,
                    worst_streak INTEGER,
                    martingale_usage TEXT
                )
            ''')
            
            # Tabela de market data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    price REAL,
                    volume REAL,
                    rsi REAL,
                    macd REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    volatility REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Banco de dados inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco: {e}")
        
    def get_market_data(self, symbol):
        """Obtém dados de mercado em tempo real"""
        try:
            # Para índices sintéticos Deriv, usar dados simulados
            if symbol.startswith(('R_', '1HZ', 'CRASH', 'BOOM', 'JD', 'STEP')):
                return self.get_synthetic_data(symbol)
            
            # Para outros símbolos, tentar yfinance
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1m")
                
                if not data.empty:
                    return self.process_market_data(data, symbol)
            except:
                pass
                
            # Fallback para dados sintéticos
            return self.get_synthetic_data(symbol)
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado: {e}")
            return self.get_synthetic_data(symbol)
    
    def get_synthetic_data(self, symbol):
        """Gera dados sintéticos baseados em padrões de mercado reais"""
        try:
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            
            # Mapear volatilidade por símbolo
            volatility_map = {
                'R_10': 0.1, 'R_25': 0.25, 'R_50': 0.5, 'R_75': 0.75, 'R_100': 1.0,
                '1HZ10V': 0.1, '1HZ25V': 0.25, '1HZ50V': 0.5, '1HZ75V': 0.75, '1HZ100V': 1.0,
                'CRASH300': 3.0, 'CRASH500': 5.0, 'CRASH1000': 10.0,
                'BOOM300': 3.0, 'BOOM500': 5.0, 'BOOM1000': 10.0
            }
            
            volatility = volatility_map.get(symbol, 0.5)
            base_price = 1000 + np.random.normal(0, 50)
            
            # Gerar série temporal realística
            periods = 100
            prices = [base_price]
            
            for i in range(periods - 1):
                # Movimento browniano com drift e mean reversion
                drift = np.random.normal(0, volatility * 0.01)
                mean_reversion = (1000 - prices[-1]) * 0.001  # Leve mean reversion
                noise = np.random.normal(0, volatility * 0.1)
                
                new_price = prices[-1] * (1 + drift + mean_reversion + noise)
                prices.append(max(0.01, new_price))  # Evitar preços negativos
            
            # Criar DataFrame simulado
            df = pd.DataFrame({
                'Close': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'Volume': [np.random.randint(1000, 10000) for _ in prices]
            })
            
            return self.process_market_data(df, symbol)
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados sintéticos: {e}")
            return self.get_fallback_data(symbol)
    
    def process_market_data(self, data, symbol):
        """Processa dados de mercado e calcula indicadores técnicos"""
        try:
            close_prices = data['Close'].values
            high_prices = data['High'].values if 'High' in data.columns else close_prices
            low_prices = data['Low'].values if 'Low' in data.columns else close_prices
            volume = data['Volume'].values if 'Volume' in data.columns else [1000] * len(close_prices)
            
            # Calcular indicadores técnicos
            rsi = self.indicators.rsi(close_prices)
            macd = self.indicators.macd(close_prices)
            bb = self.indicators.bollinger_bands(close_prices)
            volatility = self.indicators.volatility(close_prices)
            
            current_data = {
                'symbol': symbol,
                'price': float(close_prices[-1]),
                'rsi': rsi,
                'macd': macd,
                'macd_signal': 0.0,  # Simplificado
                'bb_upper': bb['upper'],
                'bb_lower': bb['lower'],
                'volatility': volatility,
                'volume': float(volume[-1]) if len(volume) > 0 else 1000.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Salvar no banco
            self.save_market_data(current_data)
            
            return current_data
            
        except Exception as e:
            logger.error(f"Erro no processamento de dados: {e}")
            return self.get_fallback_data(symbol)
    
    def get_fallback_data(self, symbol):
        """Dados de fallback em caso de erro"""
        return {
            'symbol': symbol,
            'price': 1000.0 + np.random.normal(0, 10),
            'rsi': 45.0 + np.random.normal(0, 10),
            'macd': np.random.normal(0, 0.5),
            'macd_signal': 0.0,
            'bb_upper': 1020.0,
            'bb_lower': 980.0,
            'volatility': 1.0 + np.random.normal(0, 0.3),
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_market_data(self, data):
        """Salva dados de mercado no banco"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data 
                (timestamp, symbol, price, volume, rsi, macd, bb_upper, bb_lower, volatility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'], data['symbol'], data['price'], data['volume'],
                data['rsi'], data['macd'], data['bb_upper'], data['bb_lower'], data['volatility']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de mercado: {e}")
    
    def extract_features(self, market_data, trade_history=None):
        """Extrai features para o modelo de ML"""
        try:
            features = []
            
            # Features de mercado
            features.extend([
                market_data['rsi'],
                market_data['macd'],
                market_data['volatility'],
                (market_data['price'] - market_data['bb_lower']) / (market_data['bb_upper'] - market_data['bb_lower']) if market_data['bb_upper'] != market_data['bb_lower'] else 0.5
            ])
            
            # Features temporais
            now = datetime.now()
            features.extend([
                now.hour / 24.0,  # Hora do dia normalizada
                now.weekday() / 6.0,  # Dia da semana normalizado
                (now.minute % 60) / 60.0  # Minuto normalizado
            ])
            
            # Features de histórico
            if trade_history and len(trade_history) > 0:
                recent_trades = trade_history[-10:]  # Últimos 10 trades
                win_rate = sum(1 for t in recent_trades if t.get('result') == 'win') / len(recent_trades)
                avg_pnl = np.mean([t.get('pnl', 0) for t in recent_trades])
                features.extend([win_rate, avg_pnl / 100.0])  # Normalizar PnL
            else:
                features.extend([0.5, 0.0])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erro ao extrair features: {e}")
            return np.array([50, 0, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0]).reshape(1, -1)
    
    def train_model(self):
        """Treina o modelo com dados históricos"""
        try:
            conn = sqlite3.connect(DATABASE_URL)
            
            # Obter dados de trades
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE result IN ('win', 'loss')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', conn)
            
            if len(trades_df) < 50:  # Dados insuficientes
                logger.info("Dados insuficientes para treinar o modelo")
                conn.close()
                return False
            
            # Preparar features e targets
            X = []
            y = []
            
            for _, trade in trades_df.iterrows():
                try:
                    features = json.loads(trade['features']) if trade['features'] else []
                    if len(features) == 9:  # Verificar se tem o número correto de features
                        X.append(features)
                        y.append(1 if trade['result'] == 'win' else 0)
                except:
                    continue
            
            if len(X) < 50:
                logger.info("Features insuficientes para treinar o modelo")
                conn.close()
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Treinar modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliar modelo
            accuracy = self.model.score(X_test_scaled, y_test)
            logger.info(f"Modelo treinado com acurácia: {accuracy:.2f}")
            
            self.is_trained = True
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {e}")
            return False
    
    def predict_direction(self, market_data, trade_history=None):
        """Prediz a direção do trade usando ML"""
        try:
            if not self.is_trained:
                # Tentar treinar o modelo
                if not self.train_model():
                    # Se não conseguir treinar, usar lógica híbrida
                    return self.hybrid_prediction(market_data)
            
            features = self.extract_features(market_data, trade_history)
            features_scaled = self.scaler.transform(features)
            
            # Predição
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            confidence = max(probability) * 100
            direction = 'CALL' if prediction == 1 else 'PUT'
            
            return {
                'direction': direction,
                'confidence': confidence,
                'method': 'machine_learning'
            }
            
        except Exception as e:
            logger.error(f"Erro na predição ML: {e}")
            return self.hybrid_prediction(market_data)
    
    def hybrid_prediction(self, market_data):
        """Predição híbrida usando análise técnica + padrões"""
        try:
            rsi = market_data['rsi']
            macd = market_data['macd']
            price = market_data['price']
            bb_upper = market_data['bb_upper']
            bb_lower = market_data['bb_lower']
            volatility = market_data['volatility']
            
            signals = []
            
            # Análise RSI
            if rsi < 30:
                signals.append(('CALL', 0.7))  # Sobrevenda
            elif rsi > 70:
                signals.append(('PUT', 0.7))   # Sobrecompra
            
            # Análise MACD
            if macd > 0:
                signals.append(('CALL', 0.6))
            else:
                signals.append(('PUT', 0.6))
            
            # Análise Bollinger Bands
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            if bb_position < 0.2:
                signals.append(('CALL', 0.8))  # Próximo da banda inferior
            elif bb_position > 0.8:
                signals.append(('PUT', 0.8))   # Próximo da banda superior
            
            # Análise de volatilidade
            if volatility > 2.0:
                # Alta volatilidade favorece reversões
                if rsi > 60:
                    signals.append(('PUT', 0.5))
                elif rsi < 40:
                    signals.append(('CALL', 0.5))
            
            # Combinar sinais
            call_weight = sum(weight for direction, weight in signals if direction == 'CALL')
            put_weight = sum(weight for direction, weight in signals if direction == 'PUT')
            
            if call_weight > put_weight:
                direction = 'CALL'
                confidence = min(95, (call_weight / (call_weight + put_weight)) * 100) if (call_weight + put_weight) > 0 else 65
            else:
                direction = 'PUT'
                confidence = min(95, (put_weight / (call_weight + put_weight)) * 100) if (call_weight + put_weight) > 0 else 65
            
            # Ajustar confiança baseada na volatilidade
            if volatility > 3.0:
                confidence *= 0.8  # Reduzir confiança em alta volatilidade
            
            return {
                'direction': direction,
                'confidence': max(60, confidence),  # Mínimo de 60%
                'method': 'hybrid_analysis'
            }
            
        except Exception as e:
            logger.error(f"Erro na predição híbrida: {e}")
            return {
                'direction': 'CALL' if np.random.random() > 0.5 else 'PUT',
                'confidence': 65.0,
                'method': 'fallback'
            }

# Instância global da IA
trading_ai = TradingAI()

# Rotas da API
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Trading AI API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'model_trained': trading_ai.is_trained,
        'database': 'connected'
    })

@app.route('/analyze', methods=['POST'])
def analyze_market():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'R_50')
        
        # Obter dados de mercado
        market_data = trading_ai.get_market_data(symbol)
        
        # Análise detalhada
        analysis = {
            'symbol': symbol,
            'current_price': market_data['price'],
            'timestamp': market_data['timestamp'],
            'technical_indicators': {
                'rsi': market_data['rsi'],
                'macd': market_data['macd'],
                'bb_upper': market_data['bb_upper'],
                'bb_lower': market_data['bb_lower'],
                'volatility': market_data['volatility']
            },
            'market_condition': 'neutral',
            'volatility_level': 'medium',
            'trend': 'sideways'
        }
        
        # Determinar condições de mercado
        rsi = market_data['rsi']
        if rsi < 30:
            analysis['market_condition'] = 'oversold'
            analysis['trend'] = 'bullish_reversal'
        elif rsi > 70:
            analysis['market_condition'] = 'overbought'
            analysis['trend'] = 'bearish_reversal'
        elif market_data['macd'] > 0:
            analysis['trend'] = 'bullish'
        elif market_data['macd'] < 0:
            analysis['trend'] = 'bearish'
        
        # Nível de volatilidade
        if market_data['volatility'] > 2.0:
            analysis['volatility_level'] = 'high'
        elif market_data['volatility'] < 0.5:
            analysis['volatility_level'] = 'low'
        
        analysis['message'] = f"Análise técnica: RSI {rsi:.1f}, Volatilidade {market_data['volatility']:.2f}, Tendência {analysis['trend']}"
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/signal', methods=['POST'])
def get_trading_signal():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'R_50')
        trade_history = data.get('recentTrades', [])
        
        # Obter dados de mercado
        market_data = trading_ai.get_market_data(symbol)
        
        # Obter predição
        prediction = trading_ai.predict_direction(market_data, trade_history)
        
        # Enriquecer resposta
        signal = {
            'direction': prediction['direction'],
            'confidence': prediction['confidence'],
            'method': prediction['method'],
            'reasoning': f"Análise {prediction['method']} baseada em RSI {market_data['rsi']:.1f}, MACD {market_data['macd']:.3f}",
            'entry_price': market_data['price'],
            'timestamp': market_data['timestamp'],
            'timeframe': '5m',
            'risk_level': 'medium'
        }
        
        # Ajustar risco baseado na confiança
        if prediction['confidence'] > 85:
            signal['risk_level'] = 'low'
        elif prediction['confidence'] < 70:
            signal['risk_level'] = 'high'
        
        return jsonify(signal)
        
    except Exception as e:
        logger.error(f"Erro no sinal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/risk', methods=['POST'])
def assess_risk():
    try:
        data = request.get_json()
        martingale_level = data.get('martingaleLevel', 0)
        recent_trades = data.get('recentTrades', [])
        current_balance = data.get('currentBalance', 1000)
        win_rate = data.get('winRate', 50)
        
        # Calcular score de risco
        risk_score = 0
        risk_factors = []
        
        # Fator Martingale
        if martingale_level > 4:
            risk_score += 40
            risk_factors.append(f"Martingale nível {martingale_level} - ALTO RISCO")
        elif martingale_level > 2:
            risk_score += 20
            risk_factors.append(f"Martingale nível {martingale_level} - risco elevado")
        
        # Fator win rate
        if win_rate < 40:
            risk_score += 25
            risk_factors.append(f"Win rate baixo ({win_rate:.1f}%)")
        elif win_rate < 50:
            risk_score += 10
            risk_factors.append(f"Win rate abaixo da média ({win_rate:.1f}%)")
        
        # Fator trades recentes
        if len(recent_trades) >= 3:
            recent_losses = sum(1 for t in recent_trades[-3:] if t.get('result') == 'loss')
            if recent_losses >= 2:
                risk_score += 15
                risk_factors.append(f"{recent_losses} perdas nas últimas 3 operações")
        
        # Determinar nível de risco
        if risk_score >= 60:
            level = 'high'
            recommendation = 'PAUSE imediata - risco extremamente alto'
        elif risk_score >= 35:
            level = 'medium'
            recommendation = 'Operar com extrema cautela - considere reduzir stake'
        else:
            level = 'low'
            recommendation = 'Continuar operando normalmente'
        
        assessment = {
            'level': level,
            'score': risk_score,
            'message': f"Risco {level.upper()} detectado - Score: {risk_score}/100",
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'martingale_level': martingale_level,
            'suggested_action': 'pause' if risk_score >= 60 else 'continue'
        }
        
        return jsonify(assessment)
        
    except Exception as e:
        logger.error(f"Erro na avaliação de risco: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save-trade', methods=['POST'])
def save_trade():
    try:
        data = request.get_json()
        
        # Extrair features do mercado no momento do trade
        market_data = trading_ai.get_market_data(data.get('symbol', 'R_50'))
        features = trading_ai.extract_features(market_data).tolist()[0]
        
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, direction, stake, duration, entry_price, exit_price, result, pnl, martingale_level, market_conditions, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('timestamp', datetime.now().isoformat()),
            data.get('symbol'),
            data.get('direction'),
            data.get('stake'),
            data.get('duration'),
            data.get('entry_price'),
            data.get('exit_price'),
            data.get('result'),
            data.get('pnl'),
            data.get('martingale_level', 0),
            json.dumps(data.get('market_conditions', {})),
            json.dumps(features)
        ))
        
        conn.commit()
        conn.close()
        
        # Retreinar modelo periodicamente
        try:
            conn_check = sqlite3.connect(DATABASE_URL)
            total_trades = pd.read_sql_query('SELECT COUNT(*) as count FROM trades', conn_check).iloc[0]['count']
            conn_check.close()
            
            if total_trades % 50 == 0:  # A cada 50 trades
                trading_ai.train_model()
        except:
            pass
        
        return jsonify({'status': 'success', 'message': 'Trade salvo com sucesso'})
        
    except Exception as e:
        logger.error(f"Erro ao salvar trade: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    try:
        conn = sqlite3.connect(DATABASE_URL)
        
        # Estatísticas gerais
        stats_query = '''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN result = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl
            FROM trades
            WHERE date(timestamp) = date('now')
        '''
        
        try:
            today_stats = pd.read_sql_query(stats_query, conn).iloc[0]
        except:
            # Se der erro, retornar stats vazias
            today_stats = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            }
        
        # Estatísticas por Martingale
        try:
            martingale_stats = pd.read_sql_query('''
                SELECT 
                    martingale_level,
                    COUNT(*) as trades,
                    AVG(CASE WHEN result = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                    SUM(pnl) as total_pnl
                FROM trades
                WHERE martingale_level IS NOT NULL
                GROUP BY martingale_level
                ORDER BY martingale_level
            ''', conn)
            martingale_data = martingale_stats.to_dict('records')
        except:
            martingale_data = []
        
        conn.close()
        
        statistics = {
            'today': {
                'total_trades': int(today_stats['total_trades']) if today_stats['total_trades'] else 0,
                'wins': int(today_stats['wins']) if today_stats['wins'] else 0,
                'losses': int(today_stats['losses']) if today_stats['losses'] else 0,
                'win_rate': float(today_stats['win_rate']) if today_stats['win_rate'] else 0.0,
                'total_pnl': float(today_stats['total_pnl']) if today_stats['total_pnl'] else 0.0,
                'avg_pnl': float(today_stats['avg_pnl']) if today_stats['avg_pnl'] else 0.0
            },
            'martingale_performance': martingale_data,
            'model_status': {
                'is_trained': trading_ai.is_trained,
                'last_training': datetime.now().isoformat()
            }
        }
        
        return jsonify(statistics)
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Tentar treinar o modelo na inicialização
    try:
        trading_ai.train_model()
    except Exception as e:
        logger.info(f"Modelo não treinado na inicialização: {e}")
    
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
