#!/usr/bin/env python3
"""
Guia de Deployment Seguro para Produção
Sistema Trading Bot com todas as melhorias implementadas
"""

import os
import sys
import json
import logging
import subprocess
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn
from contextlib import asynccontextmanager

# Configurar logging para produção
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/production.log', mode='a')
    ]
)
logger = logging.getLogger('SecureProductionDeploy')

class ProductionDeployment:
    """Sistema de deployment seguro para produção"""
    
    def __init__(self):
        self.is_production = os.getenv('ENVIRONMENT', 'development') == 'production'
        self.port = int(os.getenv('PORT', 8000))
        self.host = '0.0.0.0'
        
        # Configurações de produção
        self.production_config = {
            'workers': 1,  # Single worker para free tier
            'max_memory_mb': 512,
            'timeout': 30,
            'keepalive': 5,
            'ssl_enabled': False,  # Configurar conforme necessário
            'api_key_required': True,
            'rate_limiting': True,
            'monitoring': True
        }
        
        self.security_checklist = []
        self.deployment_status = {}
        
    def pre_deployment_security_check(self) -> bool:
        """Checklist de segurança pré-deployment"""
        logger.info("🔐 Executando checklist de segurança...")
        
        checks = [
            ("API Key configurada", self._check_api_key),
            ("Rate Limiting ativo", self._check_rate_limiting),
            ("CORS configurado", self._check_cors),
            ("Risk Management ativo", self._check_risk_management),
            ("Circuit Breakers ativos", self._check_circuit_breakers),
            ("Kelly Criterion ativo", self._check_kelly),
            ("ML Temporal Validation", self._check_ml_validation),
            ("Backup automático", self._check_backup),
            ("Logs de segurança", self._check_logging),
            ("Headers de segurança", self._check_security_headers)
        ]
        
        passed = 0
        total = len(checks)
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"  {status} {check_name}")
                
                if result:
                    passed += 1
                else:
                    self.security_checklist.append(f"FALHOU: {check_name}")
                    
            except Exception as e:
                logger.error(f"  ❌ ERROR {check_name}: {e}")
                self.security_checklist.append(f"ERRO: {check_name}")
        
        score = (passed / total) * 100
        logger.info(f"🔐 Score de Segurança: {score:.1f}% ({passed}/{total})")
        
        if score < 80:
            logger.error("❌ Score de segurança insuficiente para produção!")
            self._print_security_recommendations()
            return False
        
        logger.info("✅ Checklist de segurança aprovado")
        return True
    
    def _check_api_key(self) -> bool:
        """Verifica se API key está configurada"""
        return os.getenv('API_KEY') is not None and len(os.getenv('API_KEY', '')) >= 16
    
    def _check_rate_limiting(self) -> bool:
        """Verifica rate limiting"""
        return os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    
    def _check_cors(self) -> bool:
        """Verifica CORS"""
        cors_origins = os.getenv('CORS_ORIGINS', '')
        return cors_origins and '*' not in cors_origins  # Não permitir wildcard em prod
    
    def _check_risk_management(self) -> bool:
        """Verifica risk management"""
        max_loss = float(os.getenv('MAX_DAILY_LOSS_PCT', '5.0'))
        return 1.0 <= max_loss <= 10.0  # Entre 1% e 10%
    
    def _check_circuit_breakers(self) -> bool:
        """Verifica circuit breakers"""
        return os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
    
    def _check_kelly(self) -> bool:
        """Verifica Kelly Criterion"""
        return os.getenv('KELLY_ENABLED', 'true').lower() == 'true'
    
    def _check_ml_validation(self) -> bool:
        """Verifica ML temporal validation"""
        return os.getenv('TEMPORAL_VALIDATION', 'true').lower() == 'true'
    
    def _check_backup(self) -> bool:
        """Verifica backup automático"""
        return os.getenv('AUTO_BACKUP_ENABLED', 'true').lower() == 'true'
    
    def _check_logging(self) -> bool:
        """Verifica logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        return log_level in ['INFO', 'WARNING', 'ERROR']
    
    def _check_security_headers(self) -> bool:
        """Verifica headers de segurança"""
        return os.getenv('SECURITY_HEADERS_ENABLED', 'true').lower() == 'true'
    
    def _print_security_recommendations(self):
        """Imprime recomendações de segurança"""
        logger.error("🚨 RECOMENDAÇÕES DE SEGURANÇA:")
        
        recommendations = [
            "🔑 Configure API_KEY com pelo menos 16 caracteres",
            "🚦 Ative RATE_LIMIT_ENABLED=true",
            "🌐 Configure CORS_ORIGINS sem wildcards",
            "🛡️ Configure MAX_DAILY_LOSS_PCT entre 1-10",
            "🚨 Ative CIRCUIT_BREAKER_ENABLED=true",
            "💰 Ative KELLY_ENABLED=true",
            "📊 Ative TEMPORAL_VALIDATION=true",
            "💾 Ative AUTO_BACKUP_ENABLED=true",
            "📝 Configure LOG_LEVEL adequado",
            "🔒 Ative SECURITY_HEADERS_ENABLED=true"
        ]
        
        for rec in recommendations:
            logger.error(f"  {rec}")
    
    def setup_production_environment(self):
        """Configura ambiente de produção"""
        logger.info("🚀 Configurando ambiente de produção...")
        
        # Criar diretórios de produção
        self._create_production_directories()
        
        # Configurar variáveis de ambiente de produção
        self._setup_production_env_vars()
        
        # Configurar logging de produção
        self._setup_production_logging()
        
        # Inicializar monitoramento
        self._setup_monitoring()
        
        # Verificar recursos do sistema
        self._check_system_resources()
        
        logger.info("✅ Ambiente de produção configurado")
    
    def _create_production_directories(self):
        """Cria diretórios para produção"""
        directories = [
            'data',
            'logs', 
            'models',
            'backups',
            'reports',
            'temp',
            'config',
            'ssl'  # Para certificados SSL se necessário
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            
            # Permissões seguras em produção
            if os.name != 'nt' and self.is_production:
                if directory in ['ssl', 'config']:
                    os.chmod(dir_path, 0o700)  # Apenas proprietário
                else:
                    os.chmod(dir_path, 0o755)  # Leitura para grupo
        
        logger.info("📁 Diretórios de produção criados")
    
    def _setup_production_env_vars(self):
        """Configura variáveis de ambiente para produção"""
        production_defaults = {
            'ENVIRONMENT': 'production',
            'LOG_LEVEL': 'INFO',
            'API_DEBUG': 'false',
            
            # Security
            'API_KEY_REQUIRED': 'true',
            'RATE_LIMIT_ENABLED': 'true',
            'SECURITY_HEADERS_ENABLED': 'true',
            
            # Risk Management
            'MAX_DAILY_LOSS_PCT': '3.0',  # Mais conservador em produção
            'MAX_DRAWDOWN_PCT': '10.0',
            'MAX_POSITION_SIZE_PCT': '1.5',
            'CIRCUIT_BREAKER_ENABLED': 'true',
            'KELLY_ENABLED': 'true',
            
            # ML
            'ENSEMBLE_ENABLED': 'true',
            'TEMPORAL_VALIDATION': 'true',
            'MIN_CONFIDENCE_THRESHOLD': '0.70',  # Mais conservador
            'MIN_TRADES_FOR_TRAINING': '50',
            
            # Database
            'AUTO_BACKUP_ENABLED': 'true',
            'BACKUP_INTERVAL_HOURS': '4',  # Backup mais frequente
            
            # Monitoring
            'HEALTH_CHECK_ENABLED': 'true',
            'ALERTS_ENABLED': 'true',
            'DAILY_REPORTS': 'true',
            
            # Performance
            'MAX_MEMORY_MB': '450',  # Deixar margem
            'CACHE_ENABLED': 'true'
        }
        
        for key, value in production_defaults.items():
            if not os.getenv(key):
                os.environ[key] = value
        
        logger.info("⚙️ Variáveis de ambiente de produção configuradas")
    
    def _setup_production_logging(self):
        """Configura logging para produção"""
        # Configurar handler de arquivo rotativo
        from logging.handlers import RotatingFileHandler
        
        log_handler = RotatingFileHandler(
            'logs/production.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        
        log_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        
        logger.info("📝 Logging de produção configurado")
    
    def _setup_monitoring(self):
        """Configura monitoramento de produção"""
        try:
            # Importar sistema de monitoramento
            from monitoring import monitor
            
            if monitor:
                # Configurar alertas de produção
                monitor.setup_production_alerts()
                logger.info("📊 Monitoramento de produção ativo")
            else:
                logger.warning("⚠️ Sistema de monitoramento não disponível")
                
        except ImportError:
            logger.warning("⚠️ Módulo de monitoramento não encontrado")
    
    def _check_system_resources(self):
        """Verifica recursos do sistema"""
        try:
            import psutil
            
            # Verificar memória
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < 0.8:  # Menos de 800MB
                logger.warning(f"⚠️ Pouca memória: {memory_gb:.1f}GB")
                self._apply_low_memory_optimizations()
            
            # Verificar CPU
            cpu_count = psutil.cpu_count()
            logger.info(f"🖥️ CPUs: {cpu_count}, Memória: {memory_gb:.1f}GB")
            
            # Verificar espaço em disco
            disk = psutil.disk_usage('.')
            disk_free_gb = disk.free / (1024**3)
            
            if disk_free_gb < 1.0:  # Menos de 1GB livre
                logger.warning(f"⚠️ Pouco espaço em disco: {disk_free_gb:.1f}GB")
            
        except ImportError:
            logger.warning("⚠️ psutil não disponível - pular verificação de recursos")
        except Exception as e:
            logger.error(f"❌ Erro ao verificar recursos: {e}")
    
    def _apply_low_memory_optimizations(self):
        """Aplica otimizações para baixa memória"""
        optimizations = {
            'SKLEARN_CACHE_SIZE': '20',
            'ML_BATCH_SIZE': '5',
            'MAX_TRAINING_SAMPLES': '300',
            'BACKUP_COMPRESS': 'true',
            'CACHE_SIZE': '50'
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
        
        logger.info("🔧 Otimizações de baixa memória aplicadas")
    
    def run_migrations(self) -> bool:
        """Executa migrações de banco"""
        try:
            logger.info("🔄 Executando migrações...")
            
            from migrations import run_migrations
            success = run_migrations()
            
            if success:
                logger.info("✅ Migrações executadas com sucesso")
                return True
            else:
                logger.error("❌ Falha nas migrações")
                return False
                
        except ImportError:
            logger.error("❌ Módulo de migrações não encontrado")
            return False
        except Exception as e:
            logger.error(f"❌ Erro nas migrações: {e}")
            return False
    
    def initialize_ml_system(self) -> bool:
        """Inicializa sistema ML"""
        try:
            logger.info("🧠 Inicializando sistema ML...")
            
            from main import ml_system
            
            if ml_system:
                # Verificar se precisa treinar
                stats = ml_system.get_stats()
                total_trades = stats.get('ml_stats', {}).get('total_trades', 0)
                
                if total_trades < 30:
                    logger.info("🎓 Treinando modelos com dados iniciais...")
                    ml_system.train_models()
                
                logger.info("✅ Sistema ML inicializado")
                return True
            else:
                logger.error("❌ Sistema ML não disponível")
                return False
                
        except ImportError:
            logger.error("❌ Módulo ML principal não encontrado")
            return False
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar ML: {e}")
            return False
    
    def setup_signal_handlers(self):
        """Configura handlers para sinais do sistema"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Sinal recebido: {signum}")
            self.graceful_shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)
        
        logger.info("📡 Signal handlers configurados")
    
    def graceful_shutdown(self):
        """Shutdown gracioso"""
        logger.info("🛑 Iniciando shutdown gracioso...")
        
        try:
            # Parar trading automático
            logger.info("⏹️ Parando trading automático...")
            
            # Fechar conexões de banco
            logger.info("🗃️ Fechando conexões de banco...")
            
            # Salvar estado atual
            logger.info("💾 Salvando estado...")
            
            # Fazer backup final
            logger.info("📦 Backup final...")
            self._final_backup()
            
            logger.info("✅ Shutdown gracioso concluído")
            
        except Exception as e:
            logger.error(f"❌ Erro no shutdown: {e}")
        
        sys.exit(0)
    
    def _final_backup(self):
        """Backup final antes do shutdown"""
        try:
            from datetime import datetime
            import shutil
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Backup do banco
            if Path('data/trading_data.db').exists():
                backup_path = f"backups/final_backup_{timestamp}.db"
                shutil.copy2('data/trading_data.db', backup_path)
                logger.info(f"💾 Backup final salvo: {backup_path}")
                
        except Exception as e:
            logger.error(f"❌ Erro no backup final: {e}")
    
    def create_production_app(self):
        """Cria aplicação otimizada para produção"""
        from main import app
        from secure_config import secure_config
        
        # Aplicar middleware de segurança
        self._apply_security_middleware(app)
        
        # Configurar CORS para produção
        self._setup_production_cors(app)
        
        # Adicionar health check otimizado
        self._setup_health_check(app)
        
        return app
    
    def _apply_security_middleware(self, app):
        """Aplica middleware de segurança"""
        from fastapi import Request, HTTPException
        from fastapi.responses import JSONResponse
        import time
        
        # Rate limiting simples
        request_counts = {}
        
        @app.middleware("http")
        async def security_middleware(request: Request, call_next):
            # Rate limiting por IP
            client_ip = request.client.host
            current_time = time.time()
            
            if client_ip in request_counts:
                if current_time - request_counts[client_ip]['time'] < 60:  # 1 minuto
                    request_counts[client_ip]['count'] += 1
                    if request_counts[client_ip]['count'] > 60:  # 60 req/min
                        raise HTTPException(429, "Rate limit exceeded")
                else:
                    request_counts[client_ip] = {'time': current_time, 'count': 1}
            else:
                request_counts[client_ip] = {'time': current_time, 'count': 1}
            
            # Headers de segurança
            response = await call_next(request)
            
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Referrer-Policy': 'strict-origin-when-cross-origin'
            }
            
            if self.is_production:
                security_headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            
            for header, value in security_headers.items():
                response.headers[header] = value
            
            return response
        
        logger.info("🔒 Middleware de segurança aplicado")
    
    def _setup_production_cors(self, app):
        """Configura CORS para produção"""
        from fastapi.middleware.cors import CORSMiddleware
        
        # CORS restrito para produção
        allowed_origins = os.getenv('CORS_ORIGINS', '').split(',')
        allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
        
        if not allowed_origins:
            allowed_origins = ["http://localhost:3000"]  # Fallback seguro
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        
        logger.info(f"🌐 CORS configurado para: {allowed_origins}")
    
    def _setup_health_check(self, app):
        """Configura health check de produção"""
        @app.get("/health")
        async def production_health_check():
            try:
                from monitoring import monitor
                
                health = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'environment': 'production',
                    'uptime': self._get_uptime(),
                    'version': '2.0.0-secure'
                }
                
                # Verificações adicionais
                if monitor:
                    monitor_health = monitor.health_check()
                    health['monitoring'] = monitor_health.get('status', 'unknown')
                
                return health
                
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info("❤️ Health check de produção configurado")
    
    def _get_uptime(self) -> float:
        """Calcula uptime da aplicação"""
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        
        return time.time() - self.start_time
    
    def run_production_server(self):
        """Executa servidor de produção"""
        logger.info("🚀 Iniciando servidor de produção...")
        
        # 1. Checklist de segurança
        if not self.pre_deployment_security_check():
            logger.error("❌ Falha no checklist de segurança - DEPLOYMENT ABORTADO")
            sys.exit(1)
        
        # 2. Setup do ambiente
        self.setup_production_environment()
        
        # 3. Migrações
        if not self.run_migrations():
            logger.error("❌ Falha nas migrações - DEPLOYMENT ABORTADO")
            sys.exit(1)
        
        # 4. Inicializar ML
        if not self.initialize_ml_system():
            logger.warning("⚠️ Sistema ML não inicializado - continuando sem ML")
        
        # 5. Signal handlers
        self.setup_signal_handlers()
        
        # 6. Criar aplicação
        app = self.create_production_app()
        
        # 7. Configurações do servidor
        server_config = {
            'host': self.host,
            'port': self.port,
            'workers': self.production_config['workers'],
            'timeout_keep_alive': self.production_config['keepalive'],
            'timeout_graceful_shutdown': self.production_config['timeout'],
            'log_level': 'info',
            'access_log': True,
            'server_header': False,
            'date_header': True,
            'proxy_headers': True,  # Para Render/Heroku
            'forwarded_allow_ips': '*'  # Para reverse proxy
        }
        
        logger.info(f"🌍 Servidor iniciando em {self.host}:{self.port}")
        logger.info(f"🔧 Workers: {server_config['workers']}")
        logger.info(f"⏱️ Timeout: {server_config['timeout_graceful_shutdown']}s")
        logger.info("🛡️ SISTEMA SEGURO ATIVO - TODAS AS PROTEÇÕES HABILITADAS")
        
        try:
            uvicorn.run(
                app,
                **server_config
            )
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar servidor: {e}")
            sys.exit(1)

def main():
    """Função principal para deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Seguro ML Trading Bot')
    parser.add_argument('--check-only', action='store_true',
                       help='Apenas executar checklist de segurança')
    parser.add_argument('--force', action='store_true',
                       help='Forçar deployment mesmo com avisos')
    
    args = parser.parse_args()
    
    deployment = ProductionDeployment()
    
    if args.check_only:
        # Apenas verificar segurança
        success = deployment.pre_deployment_security_check()
        print(f"\n{'✅ APROVADO' if success else '❌ REPROVADO'} - Checklist de Segurança")
        sys.exit(0 if success else 1)
    
    # Deployment completo
    print("🚀 ML Trading Bot - Sistema Seguro")
    print("=" * 50)
    print("🛡️ TODAS AS MELHORIAS IMPLEMENTADAS:")
    print("   ❌ MARTINGALE REMOVIDO (Substituído por Kelly)")
    print("   ✅ KELLY CRITERION para position sizing")
    print("   ✅ RISK MANAGEMENT com Circuit Breakers")
    print("   ✅ ML AVANÇADO com 35+ features")
    print("   ✅ ENSEMBLE de modelos com validação temporal")
    print("   ✅ API SECURITY com rate limiting")
    print("   ✅ CORS seguro para produção")
    print("   ✅ MONITORING completo")
    print("   ✅ BACKUP automático")
    print("   ✅ LOGGING de produção")
    print("=" * 50)
    
    try:
        deployment.run_production_server()
    except KeyboardInterrupt:
        logger.info("👋 Deployment interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro crítico no deployment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
