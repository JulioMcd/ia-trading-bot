import requests
import json
import time
from datetime import datetime

# Configurações
API_URL = "http://localhost:5000"
API_KEY = "bhcOGajqbfFfolT"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

class TradingLearningDemo:
    """Demonstração prática do sistema de aprendizado"""
    
    def __init__(self):
        self.api_url = API_URL
        self.headers = headers
        
    def get_learning_stats(self):
        """Obter estatísticas atuais do aprendizado"""
        try:
            response = requests.get(f"{self.api_url}/learning-stats", headers=self.headers)
            if response.status_code == 200:
                stats = response.json()
                print("📊 ESTATÍSTICAS DE APRENDIZADO")
                print("=" * 50)
                print(f"🎯 Accuracy Atual: {stats['current_accuracy']}%")
                print(f"🔢 Total de Amostras: {stats['total_samples']}")
                print(f"🧠 Padrões Identificados: {stats['error_patterns_found']}")
                print(f"⚙️ Aprendizado Ativo: {stats['learning_enabled']}")
                
                print("\n📈 PARÂMETROS ADAPTATIVOS:")
                for param, value in stats['adaptive_parameters'].items():
                    print(f"   {param}: {value:.3f}")
                
                if stats['recent_patterns']:
                    print("\n🔍 PADRÕES RECENTES:")
                    for pattern in stats['recent_patterns']:
                        print(f"   • {pattern['type']}: {pattern['error_rate']:.2f} erro")
                        print(f"     Condições: {pattern['conditions']}")
                
                return stats
            else:
                print(f"❌ Erro ao obter stats: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return None
    
    def generate_signal_with_learning(self, symbol="R_50", volatility=50):
        """Gerar sinal com sistema de aprendizado"""
        signal_data = {
            "symbol": symbol,
            "currentPrice": 1000 + (volatility - 50) * 2,  # Simular preço baseado na volatilidade
            "volatility": volatility,
            "winRate": 55,  # Simular win rate
            "martingaleLevel": 0,
            "marketCondition": "neutral",
            "lastTicks": [999, 1001, 1000]
        }
        
        try:
            response = requests.post(f"{self.api_url}/signal", 
                                   headers=self.headers, 
                                   json=signal_data)
            
            if response.status_code == 200:
                signal = response.json()
                print(f"\n🎯 SINAL GERADO PARA {symbol}")
                print("=" * 40)
                print(f"📍 Direção: {signal['direction']}")
                print(f"🎲 Confiança: {signal['confidence']}%")
                print(f"🆔 Signal ID: {signal['signal_id']}")
                print(f"💡 Reasoning: {signal['reasoning']}")
                print(f"🧠 Learning Ativo: {signal['learning_active']}")
                
                if signal.get('confidence_adjustments'):
                    print(f"⚙️ Ajustes Aplicados: {len(signal['confidence_adjustments'])}")
                    for adjustment in signal['confidence_adjustments']:
                        print(f"   • {adjustment}")
                
                return signal
            else:
                print(f"❌ Erro ao gerar sinal: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return None
    
    def send_feedback(self, signal_id, result, pnl=0):
        """Enviar feedback de resultado"""
        feedback_data = {
            "signal_id": signal_id,
            "result": result,  # 1 para win, 0 para loss
            "pnl": pnl,
            "direction": "CALL"  # Pode pegar do sinal original
        }
        
        try:
            response = requests.post(f"{self.api_url}/feedback", 
                                   headers=self.headers, 
                                   json=feedback_data)
            
            if response.status_code == 200:
                feedback = response.json()
                result_text = "WIN ✅" if result == 1 else "LOSS ❌"
                print(f"\n🔄 FEEDBACK ENVIADO: {result_text}")
                print("=" * 30)
                print(f"🆔 Signal ID: {signal_id}")
                print(f"📊 Total Trades: {feedback['total_trades']}")
                print(f"🎯 Accuracy: {feedback['accuracy']}")
                print(f"🧠 Learning: {feedback['learning_active']}")
                print(f"📈 Análise: {feedback['patterns_analysis']}")
                
                return feedback
            else:
                print(f"❌ Erro ao enviar feedback: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return None
    
    def test_risk_assessment(self, martingale_level=0, today_pnl=0):
        """Testar avaliação de risco adaptativa"""
        risk_data = {
            "currentBalance": 1000,
            "todayPnL": today_pnl,
            "martingaleLevel": martingale_level,
            "currentStake": 10,
            "winRate": 45,
            "totalTrades": 50
        }
        
        try:
            response = requests.post(f"{self.api_url}/risk", 
                                   headers=self.headers, 
                                   json=risk_data)
            
            if response.status_code == 200:
                risk = response.json()
                print(f"\n⚠️ AVALIAÇÃO DE RISCO")
                print("=" * 30)
                print(f"🔴 Nível: {risk['level'].upper()}")
                print(f"📊 Score: {risk['score']}/100")
                print(f"💬 Mensagem: {risk['message']}")
                print(f"💡 Recomendação: {risk['recommendation']}")
                print(f"⚙️ Risk Factor: {risk['adaptive_risk_factor']:.3f}")
                
                return risk
            else:
                print(f"❌ Erro na avaliação: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return None
    
    def simulate_learning_cycle(self, num_signals=10):
        """Simular um ciclo completo de aprendizado"""
        print("🚀 INICIANDO SIMULAÇÃO DE APRENDIZADO")
        print("=" * 50)
        
        # 1. Ver stats iniciais
        initial_stats = self.get_learning_stats()
        
        signals_generated = []
        
        # 2. Gerar vários sinais
        symbols = ["R_50", "R_75", "R_100"]
        volatilities = [30, 50, 70, 85]
        
        for i in range(num_signals):
            symbol = symbols[i % len(symbols)]
            volatility = volatilities[i % len(volatilities)]
            
            print(f"\n--- SINAL {i+1}/{num_signals} ---")
            signal = self.generate_signal_with_learning(symbol, volatility)
            
            if signal:
                signals_generated.append(signal)
                
                # Simular resultado (70% win rate para demonstração)
                import random
                result = 1 if random.random() < 0.7 else 0
                pnl = 0.85 if result == 1 else -1.0
                
                # Enviar feedback
                feedback = self.send_feedback(signal['signal_id'], result, pnl)
                
                # Pequena pausa para simular tempo real
                time.sleep(0.5)
        
        # 3. Ver stats finais
        print(f"\n\n📈 ESTATÍSTICAS FINAIS APÓS {num_signals} SINAIS")
        print("=" * 50)
        final_stats = self.get_learning_stats()
        
        # 4. Comparar evolução
        if initial_stats and final_stats:
            print(f"\n🔄 EVOLUÇÃO DO SISTEMA:")
            print(f"   Amostras: {initial_stats['total_samples']} → {final_stats['total_samples']}")
            print(f"   Accuracy: {initial_stats['current_accuracy']}% → {final_stats['current_accuracy']}%")
            print(f"   Padrões: {initial_stats['error_patterns_found']} → {final_stats['error_patterns_found']}")
        
        return signals_generated
    
    def demonstrate_pattern_detection(self):
        """Demonstrar detecção de padrões enviando sinais problemáticos"""
        print("🔍 DEMONSTRAÇÃO DE DETECÇÃO DE PADRÕES")
        print("=" * 50)
        
        # Simular sinais ruins para um símbolo específico
        problem_symbol = "R_75"
        
        print(f"📉 Gerando sinais com baixa performance para {problem_symbol}...")
        
        for i in range(15):
            signal = self.generate_signal_with_learning(problem_symbol, 60)
            if signal:
                # Simular 80% de loss para criar padrão
                result = 0 if i < 12 else 1  # 80% loss
                pnl = -1.0 if result == 0 else 0.85
                
                self.send_feedback(signal['signal_id'], result, pnl)
                print(f"   Sinal {i+1}: {'LOSS' if result == 0 else 'WIN'}")
        
        print(f"\n🧠 Aguardando detecção de padrão...")
        time.sleep(2)
        
        # Verificar se padrão foi detectado
        stats = self.get_learning_stats()
        if stats and stats['recent_patterns']:
            print(f"✅ Padrão detectado para {problem_symbol}!")
        
        # Gerar novo sinal para ver adaptação
        print(f"\n🎯 Gerando novo sinal para {problem_symbol} (deve ter confiança reduzida):")
        adapted_signal = self.generate_signal_with_learning(problem_symbol, 60)
        
        return adapted_signal

def main():
    """Função principal de demonstração"""
    demo = TradingLearningDemo()
    
    print("🤖 DEMO DO SISTEMA DE APRENDIZADO TRADING API")
    print("=" * 60)
    
    while True:
        print("\n📋 ESCOLHA UMA OPÇÃO:")
        print("1. Ver estatísticas de aprendizado")
        print("2. Gerar sinal com aprendizado")
        print("3. Testar avaliação de risco")
        print("4. Simular ciclo de aprendizado")
        print("5. Demonstrar detecção de padrões")
        print("6. Sair")
        
        choice = input("\n👉 Digite sua escolha (1-6): ").strip()
        
        if choice == "1":
            demo.get_learning_stats()
            
        elif choice == "2":
            symbol = input("Digite o símbolo (R_50): ").strip() or "R_50"
            try:
                volatility = float(input("Digite a volatilidade (50): ") or "50")
            except:
                volatility = 50
            demo.generate_signal_with_learning(symbol, volatility)
            
        elif choice == "3":
            try:
                martingale = int(input("Nível Martingale (0): ") or "0")
                pnl = float(input("P&L hoje (0): ") or "0")
            except:
                martingale, pnl = 0, 0
            demo.test_risk_assessment(martingale, pnl)
            
        elif choice == "4":
            try:
                num_signals = int(input("Quantos sinais simular (10): ") or "10")
            except:
                num_signals = 10
            demo.simulate_learning_cycle(num_signals)
            
        elif choice == "5":
            demo.demonstrate_pattern_detection()
            
        elif choice == "6":
            print("👋 Saindo...")
            break
            
        else:
            print("❌ Opção inválida!")
        
        input("\n⏸️ Pressione Enter para continuar...")

if __name__ == "__main__":
    main()

# Exemplo de uso direto (sem menu interativo)
def quick_test():
    """Teste rápido do sistema"""
    demo = TradingLearningDemo()
    
    # 1. Ver stats
    demo.get_learning_stats()
    
    # 2. Gerar sinal
    signal = demo.generate_signal_with_learning("R_50", 45)
    
    # 3. Simular resultado e feedback
    if signal:
        result = 1  # WIN
        demo.send_feedback(signal['signal_id'], result, 0.85)
    
    # 4. Ver stats novamente
    demo.get_learning_stats()

# Descomente para teste rápido:
# quick_test()
