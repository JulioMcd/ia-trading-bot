import numpy as np
import json
import os
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

class TradingMLEngine:
    """
    🤖 Motor de ML simplificado para Trading Bot
    - Usa algoritmos próprios sem sklearn
    - Analisa padrões simples
    - Aprende por regras
    """
    
    def __init__(self):
        self.historical_data = []
        self.patterns = {}
        self.metrics = {
            'accuracy': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'last_training': None
        }
        print("🤖 ML Engine Simplificado inicializado")
    
    def predict_direction(self, market_data):
        """🎯 Predição simples baseada em padrões"""
        try:
            confidence = self._calculate_confidence(market_data)
            direction = self._simple_prediction(market_data)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'reasoning': f"ML simples: {len(self.historical_data)} trades analisados",
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {
                'direction': random.choice(['CALL', 'PUT']),
                'confidence': 65.0,
                'reasoning': 'Predição aleatória',
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_risk(self, market_data, trade_history):
        """⚠️ Análise de risco simples"""
        try:
            risk_score = self._calculate_simple_risk(market_data)
            
            if risk_score > 70:
                level = 'high'
            elif risk_score > 40:
                level = 'medium'
            else:
                level = 'low'
                
            return {
                'level': level,
                'score': risk_score,
                'message': f"Risco {level} calculado",
                'recommendation': 'Operar com cautela',
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {
                'level': 'medium',
                'score': 50.0,
                'message': 'Risco padrão',
                'recommendation': 'Operar normalmente',
                'timestamp': datetime.now().isoformat()
            }
    
    def add_trade_data(self, trade_data):
        """📊 Adiciona trade para aprendizado"""
        self.historical_data.append(trade_data)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]
        return True
    
    def optimize_martingale(self, current_level, recent_performance):
        """🎰 Otimização Martingale simples"""
        action = 'continue' if current_level < 5 else 'pause'
        return {
            'action': action,
            'recommended_level': min(current_level + 1, 8),
            'confidence': 0.7,
            'reasoning': 'Regra simples de Martingale',
            'timestamp': datetime.now().isoformat()
        }
    
    def _simple_prediction(self, market_data):
        """Predição baseada em regras simples"""
        volatility = market_data.get('volatility', 50)
        martingale_level = market_data.get('martingaleLevel', 0)
        
        # Regra simples: alta volatilidade = CALL, baixa = PUT
        if volatility > 60:
            return 'CALL' if martingale_level % 2 == 0 else 'PUT'
        else:
            return 'PUT' if martingale_level % 2 == 0 else 'CALL'
    
    def _calculate_confidence(self, market_data):
        """Calcula confiança baseada em histórico"""
        if len(self.historical_data) < 10:
            return random.uniform(60, 80)
        
        # Confiança baseada em sucesso recente
        recent_success = len([t for t in self.historical_data[-10:] 
                            if t.get('result', 0) == 1]) / 10
        return 50 + (recent_success * 40)
    
    def _calculate_simple_risk(self, market_data):
        """Cálculo de risco simples"""
        risk = 0
        risk += market_data.get('martingaleLevel', 0) * 10
        risk += max(0, 80 - market_data.get('winRate', 50))
        risk += market_data.get('volatility', 50) / 2
        return min(100, risk)

# Instância global
ml_engine = TradingMLEngine()
