#!/usr/bin/env python3
"""
Script de Deploy Otimizado para Render - ML Trading Bot
Configura e otimiza a aplicação para produção na Render
"""

import os
import sys
import json
import logging
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
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
logger = logging.getLogger('MLTradingDeploy')

class RenderOptimizer:
    """Otimizador para deploy na Render"""
    
    def __init__(self):
        self.is_render = os.getenv('RENDER') == 'true'
        self.port = int(os.getenv('PORT', 8000))
        self.host = '0.0.0.0'
        self.environment = os.getenv('ENVIRONMENT', 'production')
        
        # Configurações específicas da Render
        self.render_config = {
            'max_memory_mb': 512,  # Limite da Render free tier
            'max_workers': 1,      # Single worker para free tier
            'timeout': 30,
            'keepalive': 5
        }
        
    def setup_production_environment(self):
        """Configura ambiente de produção"""
        logger.info("🚀 Configurando ambiente de produção para Render...")
        
        # Criar diretórios necessários
        self._create_directories()
        
        # Configurar variáveis de ambiente
        self._setup_environment_variables()
        
        # Otimizar configurações para Render
        self._optimize_for_render()
        
        # Verificar recursos disponíveis
        self._check_system_resources()
        
        logger.info("✅ Ambiente de produção configurado")
    
    def _create_directories(self):
        """Cria diretórios necessários"""
        directories = ['data', 'logs', 'models', 'reports', 'backups']
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        logger.info("📁 Diretórios criados")
    
    def _setup_environment_variables(self):
        """Configura variáveis de ambiente para produção"""
        env_config = {
            'API_HOST': '0.0.0.0',
            'API_PORT': str(self.port),
            'API_DEBUG': 'false',
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production',
            'DATABASE_PATH': 'data/trading_data.db',
            'MIN_TRADES_FOR_TRAINING': '30',  # Reduzido para produção
            'AUTO_RETRAIN_INTERVAL': '25',
            'PATTERN_CONFIDENCE_THRESHOLD': '0.65',
            'MAX_MEMORY_MB': str(self.render_config['max_memory_mb']),
            'CACHE_ENABLED': 'true',
            'MONITORING_ENABLED': 'true',
            'HEALTH_CHECK_INTERVAL': '60'
        }
        
        # Aplicar configurações se não existirem
        for key, value in env_config.items():
            if not os.getenv(key):
                os.environ[key] = value
        
        logger.info("⚙️ Variáveis de ambiente configuradas")
    
    def _optimize_for_render(self):
        """Otimizações específicas para Render"""
        if not self.is_render:
            return
            
        # Otimizações de memória
        import gc
        gc.set_threshold(700, 10, 10)  # Mais agressivo
        
        # Configurar sklearn para usar menos memória
        os.environ['SKLEARN_ASSUME_FINITE'] = 'true'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        logger.info("🎯 Otimizações para Render aplicadas")
    
    def _check_system_resources(self):
        """Verifica recursos do sistema"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            logger.info(f"💾 Memória: {memory.total / (1024**3):.1f}GB disponível")
            logger.info(f"🖥️ CPUs: {cpu_count}")
            
            if memory.total < 1024**3:  # Menos de 1GB
                logger.warning("⚠️ Baixa memória detectada - aplicando otimizações")
                self._apply_low_memory_optimizations()
                
        except Exception as e:
            logger.warning(f"Não foi possível verificar recursos: {e}")
    
    def _apply_low_memory_optimizations(self):
        """Aplica otimizações para baixa memória"""
        os.environ['SKLEARN_CACHE_SIZE'] = '50'  # MB
        os.environ['ML_BATCH_SIZE'] = '10'
        os.environ['MAX_TRAINING_SAMPLES'] = '500'
        
        logger.info("🔧 Otimizações de baixa memória aplicadas")

class HealthMonitor:
    """Monitor de saúde para produção"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
    def get_health_status(self):
        """Status de saúde da aplicação"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            health = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime,
                'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'cpu_percent': process.cpu_percent(),
                'request_count': self.request_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.request_count, 1),
                'environment': os.getenv('ENVIRONMENT', 'unknown')
            }
            
            # Determinar status baseado em métricas
            if health['memory_usage_mb'] > 400:  # 80% de 512MB
                health['status'] = 'warning'
                health['warnings'] = ['High memory usage']
            
            if health['error_rate'] > 0.05:  # 5% de erro
                health['status'] = 'warning'
                health['warnings'] = health.get('warnings', []) + ['High error rate']
            
            return health
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Instância global
optimizer = RenderOptimizer()
health_monitor = HealthMonitor()

@asynccontextmanager
async def lifespan(app):
    """Gerenciamento do ciclo de vida da aplicação"""
    # Startup
    logger.info("🚀 Iniciando ML Trading Bot em produção...")
    optimizer.setup_production_environment()
    
    # Verificar se é necessário executar migrações
    try:
        from migrations import run_migrations
        run_migrations()
    except Exception as e:
        logger.error(f"Erro nas migrações: {e}")
    
    # Inicializar sistema ML
    try:
        from main import ml_system
        logger.info(f"📊 Sistema ML inicializado: {len(ml_system.models)} modelos")
    except Exception as e:
        logger.error(f"Erro ao inicializar ML: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Finalizando aplicação...")

def create_production_app():
    """Cria aplicação otimizada para produção"""
    from main import app
    
    # Adicionar health check otimizado
    @app.get("/health")
    async def health_check():
        health_monitor.request_count += 1
        return health_monitor.get_health_status()
    
    # Middleware para monitoramento
    @app.middleware("http")
    async def monitoring_middleware(request, call_next):
        health_monitor.request_count += 1
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            health_monitor.error_count += 1
            raise e
    
    # Configurar CORS para produção
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Em produção, especificar domínios específicos
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    return app

def run_production_server():
    """Executa servidor de produção"""
    logger.info("🚀 Iniciando servidor de produção na Render...")
    
    # Configurar aplicação
    app = create_production_app()
    app.router.lifespan_context = lifespan
    
    # Configurações do servidor para Render
    server_config = {
        'host': optimizer.host,
        'port': optimizer.port,
        'workers': optimizer.render_config['max_workers'],
        'timeout-keep-alive': optimizer.render_config['keepalive'],
        'timeout-graceful-shutdown': optimizer.render_config['timeout'],
        'log_level': 'info',
        'access_log': True,
        'use_colors': False,  # Melhor para logs de produção
        'server_header': False,  # Segurança
        'date_header': True
    }
    
    logger.info(f"🌐 Servidor iniciando em {optimizer.host}:{optimizer.port}")
    logger.info(f"⚙️ Workers: {server_config['workers']}")
    logger.info(f"🔧 Timeout: {server_config['timeout-graceful-shutdown']}s")
    
    try:
        uvicorn.run(
            "deploy:create_production_app",
            factory=True,
            **server_config
        )
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar servidor: {e}")
        sys.exit(1)

def generate_render_yaml():
    """Gera arquivo render.yaml para deploy automático"""
    render_config = {
        'services': [
            {
                'type': 'web',
                'name': 'ml-trading-bot',
                'env': 'python',
                'buildCommand': 'pip install -r requirements.txt && python init_script.py --no-tests',
                'startCommand': 'python deploy.py',
                'plan': 'free',
                'healthCheckPath': '/health',
                'envVars': [
                    {
                        'key': 'ENVIRONMENT',
                        'value': 'production'
                    },
                    {
                        'key': 'LOG_LEVEL',
                        'value': 'INFO'
                    },
                    {
                        'key': 'MIN_TRADES_FOR_TRAINING',
                        'value': '30'
                    }
                ]
            }
        ]
    }
    
    with open('render.yaml', 'w') as f:
        import yaml
        yaml.dump(render_config, f, default_flow_style=False)
    
    logger.info("📄 Arquivo render.yaml gerado")

def setup_logging_for_render():
    """Configura logging otimizado para Render"""
    # Render captura stdout, então configuramos para escrever lá
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Handler para stdout (capturado pela Render)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    root_logger.addHandler(stdout_handler)
    
    # Reduzir verbosidade de bibliotecas externas
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy ML Trading Bot na Render')
    parser.add_argument('--generate-yaml', action='store_true',
                       help='Gerar arquivo render.yaml')
    parser.add_argument('--check-env', action='store_true',
                       help='Verificar ambiente')
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging_for_render()
    
    if args.generate_yaml:
        generate_render_yaml()
        return
    
    if args.check_env:
        optimizer.setup_production_environment()
        health = health_monitor.get_health_status()
        print(json.dumps(health, indent=2))
        return
    
    # Executar servidor
    run_production_server()

if __name__ == "__main__":
    main()
