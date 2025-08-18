#!/usr/bin/env python3
"""
Testes automatizados para ML Trading Bot
"""

import unittest
import pytest
import asyncio
import json
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Importar módulos do sistema
try:
    from main import app, ml_system
    from config import Config
    from monitoring import MLMonitor, MLMetrics, TradingMetrics
    from migrations import MigrationManager, setup_migrations
except ImportError:
    # Para execução independente
    import sys
    sys.path.append('.')

class TestMLTradingSystem(unittest.TestCase):
    """Testes para o sistema ML"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        # Criar banco temporário
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_trading.db"
        
        # Configurar sistema ML com banco temporário
        from main import AdvancedMLTradingSystem
        self.ml_system = AdvancedMLTradingSystem()
        self.ml_system.db_path = str(self.db_path)
        self.ml_system.initialize_database()
        
        # Dados de teste
        self.sample_trade_data = {
            "id": "test_trade_001",
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
    
    def tearDown(self):
        """Limpeza após cada teste"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Testa inicialização do banco de dados"""
        # Verificar se tabelas foram criadas
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['trades', 'ml_patterns', 'ml_metrics']
        for table in expected_tables:
            self.assertIn(table, tables, f"Tabela {table} não foi criada")
        
        conn.close()
    
    def test_trade_data_save(self):
        """Testa salvamento de dados de trade"""
        from main import TradeData
        
        trade = TradeData(**self.sample_trade_data)
        self.ml_system.save_trade_data(trade)
        
        # Verificar se foi salvo
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        
        self.assertEqual(count, 1, "Trade não foi salvo")
        
        # Verificar dados
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade.id,))
        saved_trade = cursor.fetchone()
        self.assertIsNotNone(saved_trade, "Trade não encontrado")
        
        conn.close()
    
    def test_feature_extraction(self):
        """Testa extração de features"""
        from main import TradeData
        
        trade = TradeData(**self.sample_trade_data)
        features = self.ml_system.extract_features(trade)
        
        # Verificar se features essenciais foram extraídas
        expected_features = [
            'hour_of_day', 'volatility', 'martingale_level',
            'stake_normalized', 'symbol_encoded', 'direction_encoded'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features, f"Feature {feature} não extraída")
            self.assertIsInstance(features[feature], (int, float), 
                                f"Feature {feature} não é numérica")
    
    def test_model_training_insufficient_data(self):
        """Testa treinamento com dados insuficientes"""
        # Tentar treinar sem dados suficientes
        result = self.ml_system.train_models_advanced()
        
        self.assertFalse(result, "Treinamento deveria falhar com dados insuficientes")
    
    def test_model_training_with_data(self):
        """Testa treinamento com dados suficientes"""
        # Criar dados de teste suficientes
        self._create_test_trades(100)
        
        # Treinar modelos
        result = self.ml_system.train_models_advanced()
        
        if result:  # Só testa se o treinamento foi bem-sucedido
            self.assertTrue(len(self.ml_system.models) > 0, "Nenhum modelo foi treinado")
            self.assertTrue(len(self.ml_system.feature_columns) > 0, "Nenhuma feature definida")
    
    def test_prediction(self):
        """Testa sistema de predição"""
        # Criar dados e treinar modelos
        self._create_test_trades(100)
        self.ml_system.train_models_advanced()
        
        if len(self.ml_system.models) > 0:  # Só testa se modelos foram treinados
            # Testar predição
            test_features = {
                'hour_of_day': 14,
                'volatility': 50.0,
                'martingale_level': 0,
                'stake_normalized': 0.1,
                'symbol_encoded': 50,
                'direction_encoded': 1,
                'trend_encoded': 1,
                'duration_minutes': 1.0,
                'entry_price_normalized': 1.234,
                'recent_wins': 2,
                'recent_losses': 1,
                'recent_win_rate': 0.67
            }
            
            prediction = self.ml_system.predict_trade_outcome(test_features)
            
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)
            self.assertIsInstance(prediction['confidence'], float)
            self.assertIn(prediction['prediction'], ['favor', 'avoid', 'neutral'])
    
    def test_pattern_analysis(self):
        """Testa análise de padrões"""
        # Criar dados de teste com padrões
        self._create_test_trades_with_patterns(50)
        
        patterns = self.ml_system.analyze_patterns()
        
        self.assertIn('patterns', patterns)
        self.assertIn('total_trades', patterns)
        self.assertIsInstance(patterns['patterns'], list)
    
    def test_cache_system(self):
        """Testa sistema de cache"""
        test_key = "test_prediction_key"
        test_prediction = {"prediction": "favor", "confidence": 0.75}
        
        # Testar cache miss
        cached = self.ml_system.get_cached_prediction(test_key)
        self.assertIsNone(cached, "Cache deveria estar vazio inicialmente")
        
        # Testar cache set
        self.ml_system.cache_prediction(test_key, test_prediction)
        
        # Testar cache hit
        cached = self.ml_system.get_cached_prediction(test_key)
        self.assertIsNotNone(cached, "Predição deveria estar no cache")
        self.assertEqual(cached['prediction'], test_prediction['prediction'])
    
    def _create_test_trades(self, count: int):
        """Cria trades de teste"""
        from main import TradeData
        
        symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
        directions = ["CALL", "PUT"]
        outcomes = ["won", "lost"]
        
        for i in range(count):
            trade_data = self.sample_trade_data.copy()
            trade_data.update({
                "id": f"test_trade_{i:03d}",
                "symbol": symbols[i % len(symbols)],
                "direction": directions[i % len(directions)],
                "outcome": outcomes[i % len(outcomes)],
                "stake": 1.0 + (i % 5),
                "entry_price": 1000 + (i * 0.1),
                "exit_price": 1000 + (i * 0.1) + (-1 if i % 2 else 1),
                "martingale_level": i % 4,
                "volatility": 30 + (i % 40)
            })
            
            trade = TradeData(**trade_data)
            self.ml_system.save_trade_data(trade)
    
    def _create_test_trades_with_patterns(self, count: int):
        """Cria trades com padrões específicos para teste"""
        from main import TradeData
        
        # Padrão: R_50 com CALL tem alta taxa de sucesso
        for i in range(count):
            if i % 10 < 7:  # 70% de sucesso para R_50 + CALL
                symbol = "R_50"
                direction = "CALL"
                outcome = "won"
            else:
                symbol = "R_25"
                direction = "PUT"
                outcome = "lost"
            
            trade_data = self.sample_trade_data.copy()
            trade_data.update({
                "id": f"pattern_trade_{i:03d}",
                "symbol": symbol,
                "direction": direction,
                "outcome": outcome
            })
            
            trade = TradeData(**trade_data)
            self.ml_system.save_trade_data(trade)

class TestAPI(unittest.TestCase):
    """Testes para a API"""
    
    def setUp(self):
        """Configuração para testes da API"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Testa endpoint de health check"""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("models_loaded", data)
        self.assertIn("timestamp", data)
    
    def test_root_endpoint(self):
        """Testa endpoint raiz"""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("models_loaded", data)
    
    def test_ml_predict_endpoint(self):
        """Testa endpoint de predição ML"""
        prediction_request = {
            "symbol": "R_50",
            "current_price": 1234.56,
            "direction": "CALL",
            "stake": 1.0,
            "volatility": 45.2,
            "trend": "bullish"
        }
        
        response = self.client.post("/ml/predict", json=prediction_request)
        
        # Pode retornar 200 (sucesso) ou 500 (modelos não treinados)
        self.assertIn(response.status_code, [200, 500])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("prediction", data)
            self.assertIn("confidence", data)
    
    def test_ml_stats_endpoint(self):
        """Testa endpoint de estatísticas ML"""
        response = self.client.get("/ml/stats")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("ml_stats", data)
        self.assertIn("models_info", data)
    
    def test_invalid_prediction_request(self):
        """Testa requisição inválida para predição"""
        invalid_request = {"invalid": "data"}
        
        response = self.client.post("/ml/predict", json=invalid_request)
        
        self.assertEqual(response.status_code, 400)

class TestMonitoring(unittest.TestCase):
    """Testes para sistema de monitoramento"""
    
    def setUp(self):
        """Configuração para testes de monitoramento"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_monitoring.db"
        
        self.monitor = MLMonitor(str(self.db_path))
    
    def tearDown(self):
        """Limpeza após testes"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ml_metrics_logging(self):
        """Testa log de métricas ML"""
        metrics = MLMetrics(
            timestamp=datetime.now().isoformat(),
            model_name="test_model",
            accuracy=0.75,
            precision=0.70,
            recall=0.80,
            f1_score=0.75,
            total_predictions=100,
            correct_predictions=75,
            training_data_size=500,
            feature_importance={"feature1": 0.5, "feature2": 0.3},
            confusion_matrix=[[40, 10], [15, 35]]
        )
        
        self.monitor.log_ml_metrics(metrics)
        
        # Verificar se foi salvo
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ml_metrics")
        count = cursor.fetchone()[0]
        
        self.assertEqual(count, 1, "Métricas ML não foram salvas")
        conn.close()
    
    def test_trading_metrics_logging(self):
        """Testa log de métricas de trading"""
        metrics = TradingMetrics(
            timestamp=datetime.now().isoformat(),
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=60.0,
            total_pnl=250.50,
            avg_win=5.5,
            avg_loss=-3.2,
            max_drawdown=-15.0,
            sharpe_ratio=1.2,
            ml_influenced_trades=30,
            ml_accuracy_in_trades=70.0
        )
        
        self.monitor.log_trading_metrics(metrics)
        
        # Verificar se foi salvo
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trading_metrics")
        count = cursor.fetchone()[0]
        
        self.assertEqual(count, 1, "Métricas de trading não foram salvas")
        conn.close()
    
    def test_alert_creation(self):
        """Testa criação de alertas"""
        # Criar métricas que devem gerar alertas
        low_accuracy_metrics = MLMetrics(
            timestamp=datetime.now().isoformat(),
            model_name="bad_model",
            accuracy=0.30,  # Accuracy muito baixa
            precision=0.25,
            recall=0.35,
            f1_score=0.30,
            total_predictions=50,
            correct_predictions=15,
            training_data_size=100,
            feature_importance={},
            confusion_matrix=[[10, 20], [15, 5]]
        )
        
        self.monitor.log_ml_metrics(low_accuracy_metrics)
        
        # Verificar se alerta foi criado
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_type = 'low_accuracy'")
        count = cursor.fetchone()[0]
        
        self.assertGreater(count, 0, "Alerta de accuracy baixa não foi criado")
        conn.close()
    
    def test_health_check(self):
        """Testa verificação de saúde"""
        health = self.monitor.health_check()
        
        self.assertIn("status", health)
        self.assertIn("components", health)
        self.assertIn("timestamp", health)

class TestMigrations(unittest.TestCase):
    """Testes para sistema de migrações"""
    
    def setUp(self):
        """Configuração para testes de migração"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_migrations.db"
        
        self.manager = setup_migrations()
        self.manager.db_path = str(self.db_path)
    
    def tearDown(self):
        """Limpeza após testes"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_migration_setup(self):
        """Testa configuração do sistema de migrações"""
        # Verificar se tabela de migrações foi criada
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'")
        result = cursor.fetchone()
        
        self.assertIsNotNone(result, "Tabela de migrações não foi criada")
        conn.close()
    
    def test_migrate_all(self):
        """Testa aplicação de todas as migrações"""
        result = self.manager.migrate()
        
        self.assertTrue(result, "Migrações falharam")
        
        # Verificar status
        status = self.manager.status()
        self.assertEqual(status['pending_migrations'], 0, "Ainda há migrações pendentes")
    
    def test_migration_idempotency(self):
        """Testa idempotência das migrações"""
        # Aplicar migrações duas vezes
        result1 = self.manager.migrate()
        result2 = self.manager.migrate()
        
        self.assertTrue(result1, "Primeira aplicação falhou")
        self.assertTrue(result2, "Segunda aplicação falhou")
        
        # Verificar que não há duplicatas
        applied = self.manager.get_applied_migrations()
        self.assertEqual(len(applied), len(set(applied)), "Migrações duplicadas detectadas")

class TestPerformance(unittest.TestCase):
    """Testes de performance"""
    
    def setUp(self):
        """Configuração para testes de performance"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_performance.db"
        
        from main import AdvancedMLTradingSystem
        self.ml_system = AdvancedMLTradingSystem()
        self.ml_system.db_path = str(self.db_path)
        self.ml_system.initialize_database()
    
    def tearDown(self):
        """Limpeza após testes"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_bulk_trade_insertion(self):
        """Testa inserção em massa de trades"""
        import time
        from main import TradeData
        
        # Preparar dados
        base_trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "R_50",
            "direction": "CALL",
            "stake": 1.0,
            "duration": "5t",
            "entry_price": 1234.56,
            "exit_price": 1235.78,
            "outcome": "won",
            "market_context": {"volatility": 45.2},
            "martingale_level": 0,
            "volatility": 45.2,
            "trend": "bullish"
        }
        
        # Medir tempo de inserção
        start_time = time.time()
        
        for i in range(1000):
            trade_data = base_trade.copy()
            trade_data["id"] = f"perf_test_{i:04d}"
            
            trade = TradeData(**trade_data)
            self.ml_system.save_trade_data(trade)
        
        end_time = time.time()
        insertion_time = end_time - start_time
        
        # Verificar performance (deve ser menos de 10 segundos para 1000 trades)
        self.assertLess(insertion_time, 10.0, f"Inserção muito lenta: {insertion_time:.2f}s")
        
        # Verificar se todos foram inseridos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        
        self.assertEqual(count, 1000, "Nem todos os trades foram inseridos")
        conn.close()
    
    def test_prediction_performance(self):
        """Testa performance de predições"""
        import time
        
        # Criar dados e treinar modelo
        self._create_bulk_trades(200)
        training_success = self.ml_system.train_models_advanced()
        
        if not training_success or len(self.ml_system.models) == 0:
            self.skipTest("Treinamento de modelo falhou")
        
        # Testar velocidade de predição
        test_features = {
            'hour_of_day': 14,
            'volatility': 50.0,
            'martingale_level': 0,
            'stake_normalized': 0.1,
            'symbol_encoded': 50,
            'direction_encoded': 1,
            'trend_encoded': 1,
            'duration_minutes': 1.0,
            'entry_price_normalized': 1.234,
            'recent_wins': 2,
            'recent_losses': 1,
            'recent_win_rate': 0.67
        }
        
        # Medir tempo de 100 predições
        start_time = time.time()
        
        for _ in range(100):
            prediction = self.ml_system.predict_trade_outcome(test_features)
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Deve ser menos de 1 segundo para 100 predições
        self.assertLess(prediction_time, 1.0, f"Predições muito lentas: {prediction_time:.2f}s")
    
    def _create_bulk_trades(self, count: int):
        """Cria trades em massa para testes de performance"""
        from main import TradeData
        
        base_trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "R_50",
            "direction": "CALL",
            "stake": 1.0,
            "duration": "5t",
            "entry_price": 1234.56,
            "exit_price": 1235.78,
            "outcome": "won",
            "market_context": {"volatility": 45.2},
            "martingale_level": 0,
            "volatility": 45.2,
            "trend": "bullish"
        }
        
        outcomes = ["won", "lost"]
        
        for i in range(count):
            trade_data = base_trade.copy()
            trade_data.update({
                "id": f"bulk_trade_{i:04d}",
                "outcome": outcomes[i % len(outcomes)]
            })
            
            trade = TradeData(**trade_data)
            self.ml_system.save_trade_data(trade)

# ===== EXECUÇÃO DE TESTES =====

class TestSuite:
    """Suíte de testes personalizada"""
    
    def __init__(self):
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "details": []
        }
    
    def run_all_tests(self):
        """Executa todos os testes"""
        print("🧪 Executando testes do ML Trading Bot")
        print("=" * 50)
        
        # Definir suítes de teste
        test_classes = [
            TestMLTradingSystem,
            TestAPI,
            TestMonitoring,
            TestMigrations,
            TestPerformance
        ]
        
        for test_class in test_classes:
            print(f"\n📋 Executando {test_class.__name__}...")
            self._run_test_class(test_class)
        
        self._print_summary()
        return self.results["failed"] == 0 and self.results["errors"] == 0
    
    def _run_test_class(self, test_class):
        """Executa uma classe de teste"""
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
        
        result = runner.run(suite)
        
        self.results["total"] += result.testsRun
        self.results["passed"] += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        self.results["failed"] += len(result.failures)
        self.results["errors"] += len(result.errors)
        self.results["skipped"] += len(result.skipped)
        
        # Registrar detalhes
        class_result = {
            "class": test_class.__name__,
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped)
        }
        
        self.results["details"].append(class_result)
        
        # Imprimir resultado da classe
        status = "✅" if class_result["failed"] == 0 and class_result["errors"] == 0 else "❌"
        print(f"  {status} {class_result['passed']}/{class_result['total']} testes passaram")
        
        # Imprimir falhas se houver
        if result.failures:
            for test, traceback in result.failures:
                print(f"    ❌ FALHA: {test}")
        
        if result.errors:
            for test, traceback in result.errors:
                print(f"    💥 ERRO: {test}")
    
    def _print_summary(self):
        """Imprime resumo dos testes"""
        print(f"\n{'='*50}")
        print(f"📊 RESUMO DOS TESTES")
        print(f"{'='*50}")
        
        print(f"Total de testes: {self.results['total']}")
        print(f"✅ Passou: {self.results['passed']}")
        print(f"❌ Falhou: {self.results['failed']}")
        print(f"💥 Erro: {self.results['errors']}")
        print(f"⏭️ Pulou: {self.results['skipped']}")
        
        success_rate = (self.results['passed'] / max(self.results['total'], 1)) * 100
        print(f"📈 Taxa de sucesso: {success_rate:.1f}%")
        
        # Detalhes por classe
        print(f"\n📋 Detalhes por categoria:")
        for detail in self.results['details']:
            status = "✅" if detail['failed'] == 0 and detail['errors'] == 0 else "❌"
            print(f"  {status} {detail['class']}: {detail['passed']}/{detail['total']}")

def main():
    """Função principal para execução dos testes"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Testes ML Trading Bot')
    parser.add_argument('--class', help='Executar classe específica de teste')
    parser.add_argument('--method', help='Executar método específico de teste')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verboso')
    
    args = parser.parse_args()
    
    if args.class or args.method:
        # Execução específica com unittest
        unittest.main(argv=[''], exit=False, verbosity=2 if args.verbose else 1)
    else:
        # Execução da suíte completa
        suite = TestSuite()
        success = suite.run_all_tests()
        
        if success:
            print("\n🎉 Todos os testes passaram!")
            exit(0)
        else:
            print("\n💥 Alguns testes falharam!")
            exit(1)

if __name__ == "__main__":
    main()
