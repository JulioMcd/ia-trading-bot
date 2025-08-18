import requests
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List

class MLTradingAPIClient:
    """Cliente para interagir com a API ML de Trading"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        
    def health_check(self) -> Dict:
        """Verifica se a API está funcionando"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def save_trade(self, trade_data: Dict) -> Dict:
        """Salva dados de um trade"""
        try:
            response = self.session.post(
                f"{self.api_url}/trade/save",
                json=trade_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_trade(self, prediction_request: Dict) -> Dict:
        """Obtém predição ML para um trade"""
        try:
            response = self.session.post(
                f"{self.api_url}/ml/predict",
                json=prediction_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_market(self, analysis_request: Dict) -> Dict:
        """Análise de mercado com ML"""
        try:
            response = self.session.post(
                f"{self.api_url}/ml/analyze",
                json=analysis_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_trading_signal(self, signal_request: Dict) -> Dict:
        """Obtém sinal de trading"""
        try:
            response = self.session.post(
                f"{self.api_url}/ml/signal",
                json=signal_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def assess_risk(self, risk_request: Dict) -> Dict:
        """Avaliação de risco"""
        try:
            response = self.session.post(
                f"{self.api_url}/ml/risk",
                json=risk_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def force_training(self) -> Dict:
        """Força retreinamento dos modelos"""
        try:
            response = self.session.post(f"{self.api_url}/ml/train")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict:
        """Obtém estatísticas do ML"""
        try:
            response = self.session.get(f"{self.api_url}/ml/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_patterns(self) -> Dict:
        """Obtém padrões identificados"""
        try:
            response = self.session.get(f"{self.api_url}/ml/patterns")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def example_usage():
    """Exemplo de uso da API ML"""
    
    # Configurar cliente
    api_url = "https://your-ml-api.onrender.com"  # Substitua pela sua URL
    client = MLTradingAPIClient(api_url)
    
    print("🧠 Exemplo de uso da API ML Trading")
    print("=" * 50)
    
    # 1. Verificar saúde da API
    print("\n1. 🔍 Verificando saúde da API...")
    health = client.health_check()
    print(f"Status: {health.get('status', 'unknown')}")
    print(f"Modelos carregados: {health.get('models_loaded', 0)}")
    
    if health.get('status') != 'healthy':
        print("❌ API não está saudável. Verifique a conexão.")
        return
    
    # 2. Exemplo de dados de trade
    print("\n2. 📊 Salvando exemplo de trade...")
    
    sample_trade = {
        "id": f"trade_{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "symbol": "R_50",
        "direction": "CALL",
        "stake": 1.0,
        "duration": "5t",
        "entry_price": 1234.56,
        "exit_price": 1235.78,
        "outcome": "won",
        "market_context": {
            "current_price": 1234.56,
            "volatility": 45.2,
            "recent_results": ["won", "lost", "won"]
        },
        "martingale_level": 0,
        "volatility": 45.2,
        "trend": "bullish"
    }
    
    save_result = client.save_trade(sample_trade)
    if "error" not in save_result:
        print("✅ Trade salvo com sucesso!")
    else:
        print(f"❌ Erro ao salvar trade: {save_result['error']}")
    
    # 3. Obter predição ML
    print("\n3. 🎯 Obtendo predição ML...")
    
    prediction_request = {
        "symbol": "R_50",
        "current_price": 1234.56,
        "direction": "CALL",
        "stake": 1.0,
        "duration": 5,
        "trend": "bullish",
        "volatility": 45.2,
        "martingale_level": 0,
        "recent_wins": 2,
        "recent_losses": 1,
        "recent_win_rate": 0.67
    }
    
    prediction = client.predict_trade(prediction_request)
    if "error" not in prediction:
        print(f"🎯 Predição: {prediction.get('prediction', 'N/A')}")
        print(f"🎯 Confiança: {prediction.get('confidence', 0):.1%}")
        print(f"🎯 Modelo: {prediction.get('model_used', 'N/A')}")
    else:
        print(f"❌ Erro na predição: {prediction['error']}")
    
    # 4. Análise de mercado
    print("\n4. 📈 Análise de mercado...")
    
    analysis_request = {
        "symbol": "R_50",
        "current_price": 1234.56,
        "timestamp": datetime.now().isoformat(),
        "trades": [sample_trade],
        "balance": 1000.0,
        "win_rate": 67.0,
        "volatility": 45.2,
        "market_condition": "normal",
        "martingale_level": 0,
        "is_after_loss": False,
        "ml_patterns": 5,
        "ml_accuracy": 0.65
    }
    
    analysis = client.analyze_market(analysis_request)
    if "error" not in analysis:
        print(f"📈 Análise: {analysis.get('message', 'N/A')}")
        print(f"📈 Confiança: {analysis.get('confidence', 0):.1f}%")
        print(f"📈 Recomendação: {analysis.get('ml_recommendation', 'N/A')}")
    else:
        print(f"❌ Erro na análise: {analysis['error']}")
    
    # 5. Sinal de trading
    print("\n5. 📡 Obtendo sinal de trading...")
    
    signal_request = {
        "symbol": "R_50",
        "current_price": 1234.56,
        "account_balance": 1000.0,
        "win_rate": 67.0,
        "recent_trades": [sample_trade],
        "timestamp": datetime.now().isoformat(),
        "volatility": 45.2,
        "market_condition": "normal",
        "martingale_level": 0,
        "is_after_loss": False,
        "ml_data": {
            "patterns": 5,
            "accuracy": 0.65,
            "experience": 100
        }
    }
    
    signal = client.get_trading_signal(signal_request)
    if "error" not in signal:
        print(f"📡 Direção: {signal.get('direction', 'N/A')}")
        print(f"📡 Confiança: {signal.get('confidence', 0):.1f}%")
        print(f"📡 Razão: {signal.get('reasoning', 'N/A')}")
    else:
        print(f"❌ Erro no sinal: {signal['error']}")
    
    # 6. Avaliação de risco
    print("\n6. ⚠️ Avaliação de risco...")
    
    risk_request = {
        "current_balance": 1000.0,
        "today_pnl": 25.5,
        "martingale_level": 0,
        "recent_trades": [sample_trade],
        "win_rate": 67.0,
        "total_trades": 15,
        "timestamp": datetime.now().isoformat(),
        "is_in_cooling_period": False,
        "needs_analysis_after_loss": False,
        "ml_risk": {
            "error_patterns": 2,
            "adaptations_active": 3
        }
    }
    
    risk = client.assess_risk(risk_request)
    if "error" not in risk:
        print(f"⚠️ Nível: {risk.get('level', 'N/A').upper()}")
        print(f"⚠️ Score: {risk.get('score', 0)}")
        print(f"⚠️ Recomendação: {risk.get('recommendation', 'N/A')}")
    else:
        print(f"❌ Erro na avaliação: {risk['error']}")
    
    # 7. Estatísticas ML
    print("\n7. 📊 Estatísticas do ML...")
    
    stats = client.get_statistics()
    if "error" not in stats:
        ml_stats = stats.get('ml_stats', {})
        print(f"📊 Total de trades: {ml_stats.get('total_trades', 0)}")
        print(f"📊 Taxa de acerto: {ml_stats.get('overall_win_rate', 0):.1%}")
        print(f"📊 Modelos carregados: {ml_stats.get('models_loaded', 0)}")
        
        if ml_stats.get('last_training'):
            last_training = ml_stats['last_training']
            print(f"📊 Último treino: {last_training.get('model_type', 'N/A')} "
                  f"({last_training.get('accuracy', 0):.1%})")
    else:
        print(f"❌ Erro nas estatísticas: {stats['error']}")
    
    # 8. Padrões identificados
    print("\n8. 🔍 Padrões identificados...")
    
    patterns = client.get_patterns()
    if "error" not in patterns:
        pattern_list = patterns.get('patterns', [])
        print(f"🔍 Total de padrões: {len(pattern_list)}")
        
        for i, pattern in enumerate(pattern_list[:3], 1):  # Mostrar apenas os 3 primeiros
            print(f"   {i}. {pattern.get('description', 'N/A')} "
                  f"(Confiança: {pattern.get('confidence', 0):.1%})")
    else:
        print(f"❌ Erro nos padrões: {patterns['error']}")
    
    # 9. Forçar treinamento (opcional)
    print("\n9. 🎓 Forçando retreinamento (opcional)...")
    
    user_input = input("Deseja forçar o retreinamento dos modelos? (y/N): ")
    if user_input.lower() == 'y':
        training = client.force_training()
        if "error" not in training:
            print("🎓 Treinamento iniciado em background!")
        else:
            print(f"❌ Erro no treinamento: {training['error']}")
    else:
        print("⏭️ Treinamento pulado.")
    
    print("\n" + "=" * 50)
    print("✅ Exemplo de uso concluído!")
    print("\n💡 Dicas:")
    print("   - Use a API em produção conectando com Deriv WebSocket")
    print("   - Monitore as estatísticas regularmente")
    print("   - Deixe o sistema coletar dados por alguns dias antes de usar ML intensivamente")
    print("   - Sempre teste em conta demo primeiro")

def stress_test():
    """Teste de stress da API"""
    
    api_url = "https://your-ml-api.onrender.com"  # Substitua pela sua URL
    client = MLTradingAPIClient(api_url)
    
    print("🧪 Teste de stress da API ML")
    print("=" * 30)
    
    # Verificar se API está funcionando
    health = client.health_check()
    if health.get('status') != 'healthy':
        print("❌ API não está saudável. Cancelando teste.")
        return
    
    # Simular múltiplos trades
    print("\n📊 Simulando 100 trades...")
    
    symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    directions = ["CALL", "PUT"]
    outcomes = ["won", "lost"]
    
    start_time = time.time()
    
    for i in range(100):
        # Dados aleatórios para simulação
        trade_data = {
            "id": f"stress_test_{i}_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbols[i % len(symbols)],
            "direction": directions[i % len(directions)],
            "stake": 1.0 + (i % 5),
            "duration": f"{5 + (i % 5)}t",
            "entry_price": 1000 + (i * 0.1),
            "exit_price": 1000 + (i * 0.1) + (-1 if i % 2 else 1),
            "outcome": outcomes[i % len(outcomes)],
            "market_context": {
                "current_price": 1000 + (i * 0.1),
                "volatility": 30 + (i % 40),
                "recent_results": ["won", "lost"][:(i % 3) + 1]
            },
            "martingale_level": i % 4,
            "volatility": 30 + (i % 40),
            "trend": ["bullish", "bearish", "neutral"][i % 3]
        }
        
        # Salvar trade
        result = client.save_trade(trade_data)
        if "error" in result:
            print(f"❌ Erro no trade {i}: {result['error']}")
            break
        
        # A cada 10 trades, fazer uma predição
        if i % 10 == 0:
            prediction_request = {
                "symbol": trade_data["symbol"],
                "current_price": trade_data["entry_price"],
                "direction": trade_data["direction"],
                "stake": trade_data["stake"],
                "duration": 5,
                "trend": trade_data["trend"],
                "volatility": trade_data["volatility"],
                "martingale_level": trade_data["martingale_level"],
                "recent_wins": i // 2,
                "recent_losses": i // 2,
                "recent_win_rate": 0.5
            }
            
            prediction = client.predict_trade(prediction_request)
            if "error" not in prediction:
                print(f"✅ Trade {i}: Predição {prediction.get('prediction', 'N/A')}")
            
        # Pequeno delay para não sobrecarregar
        time.sleep(0.1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Teste concluído em {total_time:.2f} segundos")
    print(f"📊 Taxa: {100/total_time:.1f} trades/segundo")
    
    # Verificar estatísticas finais
    stats = client.get_statistics()
    if "error" not in stats:
        ml_stats = stats.get('ml_stats', {})
        print(f"📈 Total de trades no sistema: {ml_stats.get('total_trades', 0)}")

if __name__ == "__main__":
    print("🚀 API ML Trading - Exemplos de Uso")
    print("\nEscolha uma opção:")
    print("1. Exemplo básico de uso")
    print("2. Teste de stress")
    print("3. Sair")
    
    choice = input("\nDigite sua escolha (1-3): ")
    
    if choice == "1":
        example_usage()
    elif choice == "2":
        stress_test()
    else:
        print("👋 Até logo!")
