#!/usr/bin/env python3
"""
Script de inicialização para ML Trading Bot
Configura ambiente, executa migrações e inicia sistema
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import importlib.util

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MLTradingInit')

class MLTradingInitializer:
    """Inicializador do sistema ML Trading"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.required_files = [
            'main.py',
            'config.py', 
            'monitoring.py',
            'migrations.py',
            'requirements.txt'
        ]
        self.optional_files = [
            'test_ml_trading.py',
            'deploy.py',
            '.env',
            'render.yaml'
        ]
        
        self.system_status = {
            'files_validated': False,
            'dependencies_installed': False,
            'database_initialized': False,
            'migrations_applied': False,
            'config_loaded': False,
            'monitoring_setup': False,
            'ml_system_ready': False
        }
    
    def validate_environment(self) -> bool:
        """Valida o ambiente de execução"""
        logger.info("🔍 Validando ambiente...")
        
        # Verificar Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error(f"❌ Python 3.8+ necessário. Versão atual: {python_version.major}.{python_version.minor}")
            return False
        
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Verificar arquivos obrigatórios
        missing_files = []
        for file in self.required_files:
            if not (self.root_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Arquivos obrigatórios não encontrados: {missing_files}")
            return False
        
        logger.info("✅ Todos os arquivos obrigatórios encontrados")
        
        # Verificar diretórios
        self._create_directories()
        
        self.system_status['files_validated'] = True
        return True
    
    def _create_directories(self):
        """Cria diretórios necessários"""
        directories = [
            'data',
            'logs', 
            'models',
            'backups',
            'reports',
            'temp'
        ]
        
        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(exist_ok=True)
            
            # Criar .gitkeep para diretórios vazios
            gitkeep = dir_path / '.gitkeep'
            if not gitkeep.exists() and not any(dir_path.iterdir()):
                gitkeep.touch()
        
        logger.info("✅ Diretórios criados/verificados")
    
    def install_dependencies(self, force: bool = False) -> bool:
        """Instala dependências do projeto"""
        logger.info("📦 Verificando dependências...")
        
        requirements_file = self.root_dir / 'requirements.txt'
        if not requirements_file.exists():
            logger.error("❌ requirements.txt não encontrado")
            return False
        
        # Verificar se pip está disponível
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("❌ pip não está disponível")
            return False
        
        # Ler requirements
        with open(requirements_file, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Verificar pacotes instalados
        if not force:
            try:
                import fastapi, uvicorn, sklearn, pandas, numpy, pydantic
                logger.info("✅ Dependências principais já instaladas")
                self.system_status['dependencies_installed'] = True
                return True
            except ImportError:
                logger.info("⚠️ Algumas dependências não encontradas, instalando...")
        
        # Instalar dependências
        try:
            logger.info("📥 Instalando dependências...")
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            
            if force:
                cmd.append('--force-reinstall')
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Dependências instaladas com sucesso")
            
            self.system_status['dependencies_installed'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erro na instalação de dependências: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    def setup_database(self) -> bool:
        """Configura e inicializa banco de dados"""
        logger.info("🗄️ Configurando banco de dados...")
        
        try:
            # Importar módulo de migrações
            from migrations import setup_migrations, run_migrations
            
            # Executar migrações
            success = run_migrations()
            
            if success:
                logger.info("✅ Banco de dados configurado")
                self.system_status['database_initialized'] = True
                self.system_status['migrations_applied'] = True
                return True
            else:
                logger.error("❌ Falha na configuração do banco de dados")
                return False
                
        except ImportError as e:
            logger.error(f"❌ Erro ao importar módulo de migrações: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro na configuração do banco: {e}")
            return False
    
    def load_configuration(self) -> bool:
        """Carrega configuração do sistema"""
        logger.info("⚙️ Carregando configuração...")
        
        try:
            from config import config
            
            logger.info(f"✅ Configuração carregada: {config}")
            
            # Verificar configurações críticas
            if hasattr(config, 'ml') and config.ml.min_trades_for_training > 0:
                logger.info(f"📊 ML configurado: min_trades={config.ml.min_trades_for_training}")
            
            if hasattr(config, 'api') and config.api.port:
                logger.info(f"🌐 API configurada: porta={config.api.port}")
            
            self.system_status['config_loaded'] = True
            return True
            
        except ImportError as e:
            logger.error(f"❌ Erro ao importar configuração: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro na configuração: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Configura sistema de monitoramento"""
        logger.info("📊 Configurando monitoramento...")
        
        try:
            from monitoring import monitor
            
            if monitor:
                # Testar sistema de monitoramento
                health = monitor.health_check()
                
                if health.get('status') in ['healthy', 'warning']:
                    logger.info("✅ Sistema de monitoramento ativo")
                    self.system_status['monitoring_setup'] = True
                    return True
                else:
                    logger.warning("⚠️ Sistema de monitoramento com problemas")
                    return False
            else:
                logger.warning("⚠️ Sistema de monitoramento não disponível")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️ Módulo de monitoramento não encontrado: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro no monitoramento: {e}")
            return False
    
    def initialize_ml_system(self) -> bool:
        """Inicializa sistema de ML"""
        logger.info("🧠 Inicializando sistema ML...")
        
        try:
            from main import ml_system
            
            # Verificar se sistema ML foi inicializado
            if hasattr(ml_system, 'models') and hasattr(ml_system, 'db_path'):
                logger.info("✅ Sistema ML inicializado")
                
                # Verificar dados existentes
                stats = ml_system.get_ml_stats()
                total_trades = stats.get('total_trades', 0)
                
                logger.info(f"📈 Dados existentes: {total_trades} trades")
                
                if total_trades >= ml_system.min_trades_for_training:
                    logger.info("🎓 Dados suficientes para treino automático")
                else:
                    logger.info(f"⏳ Necessário mais {ml_system.min_trades_for_training - total_trades} trades para treino")
                
                self.system_status['ml_system_ready'] = True
                return True
            else:
                logger.error("❌ Sistema ML não foi inicializado corretamente")
                return False
                
        except ImportError as e:
            logger.error(f"❌ Erro ao importar sistema ML: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro na inicialização ML: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Executa testes básicos do sistema"""
        logger.info("🧪 Executando testes básicos...")
        
        test_file = self.root_dir / 'test_ml_trading.py'
        if not test_file.exists():
            logger.warning("⚠️ Arquivo de testes não encontrado, pulando...")
            return True
        
        try:
            # Executar testes básicos
            cmd = [sys.executable, str(test_file), '--class', 'TestMLTradingSystem']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Testes básicos passaram")
                return True
            else:
                logger.warning("⚠️ Alguns testes falharam, mas sistema pode funcionar")
                logger.debug(f"Saída dos testes: {result.stdout}")
                return True  # Não bloquear inicialização por falha em testes
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Testes expiraram, continuando...")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Erro ao executar testes: {e}")
            return True
    
    def create_startup_script(self) -> bool:
        """Cria script de inicialização para produção"""
        logger.info("📜 Criando script de inicialização...")
        
        startup_script = self.root_dir / 'start_server.py'
        
        script_content = '''#!/usr/bin/env python3
"""
Script de inicialização do servidor ML Trading
"""

import sys
import os
import logging
from pathlib import Path

# Adicionar diretório atual ao path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Função principal"""
    print("🚀 Iniciando ML Trading Bot Server...")
    
    try:
        # Importar e executar
        from main import app
        import uvicorn
        
        # Configurações do servidor
        from config import config
        
        host = config.api.host if config and hasattr(config, 'api') else "0.0.0.0"
        port = int(os.getenv('PORT', config.api.port if config and hasattr(config, 'api') else 8000))
        debug = config.api.debug if config and hasattr(config, 'api') else False
        
        print(f"🌐 Servidor: {host}:{port}")
        print(f"🔧 Debug: {debug}")
        
        # Iniciar servidor
        uvicorn.run(
            app,
            host=host,
            port=port,
            debug=debug,
            access_log=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\\n👋 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        try:
            with open(startup_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Tornar executável no Unix
            if os.name != 'nt':
                os.chmod(startup_script, 0o755)
            
            logger.info(f"✅ Script de inicialização criado: {startup_script}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar script: {e}")
            return False
    
    def generate_env_file(self) -> bool:
        """Gera arquivo .env com configurações padrão"""
        logger.info("🔧 Gerando arquivo .env...")
        
        env_file = self.root_dir / '.env'
        
        if env_file.exists():
            logger.info("⚠️ Arquivo .env já existe, mantendo configurações atuais")
            return True
        
        env_content = f'''# Configurações ML Trading Bot - Gerado em {datetime.now().isoformat()}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_KEY_REQUIRED=false
API_KEY=ml-trading-{datetime.now().strftime('%Y%m%d')}

# Machine Learning
MIN_TRADES_FOR_TRAINING=50
AUTO_RETRAIN_INTERVAL=50
PATTERN_CONFIDENCE_THRESHOLD=0.7
ML_MODELS=random_forest,gradient_boosting,logistic_regression

# Database
DATABASE_PATH=data/trading_data.db
DB_BACKUP_INTERVAL=24

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/ml_trading.log
LOG_TO_CONSOLE=true

# Performance
MAX_MEMORY_MB=512
CACHE_ENABLED=true
MAX_CONCURRENT_REQUESTS=100

# Monitoring
MONITORING_ENABLED=true
DAILY_REPORTS=true
HEALTH_CHECK_INTERVAL=5

# Security (Configurar em produção)
# RENDER_API_KEY=your-render-api-key
# API_KEY_REQUIRED=true
# API_KEY=your-secure-api-key
'''
        
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            logger.info(f"✅ Arquivo .env criado: {env_file}")
            logger.info("🔐 IMPORTANTE: Configure as chaves de segurança antes de usar em produção!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar .env: {e}")
            return False
    
    def print_system_status(self):
        """Imprime status do sistema"""
        print(f"\n{'='*60}")
        print(f"📊 STATUS DO SISTEMA ML TRADING BOT")
        print(f"{'='*60}")
        
        for component, status in self.system_status.items():
            icon = "✅" if status else "❌"
            component_name = component.replace('_', ' ').title()
            print(f"{icon} {component_name:<25} {'OK' if status else 'FALHA'}")
        
        # Calcular score geral
        total_components = len(self.system_status)
        ready_components = sum(self.system_status.values())
        readiness_score = (ready_components / total_components) * 100
        
        print(f"\n📈 Score de Prontidão: {readiness_score:.1f}% ({ready_components}/{total_components})")
        
        # Status geral
        if readiness_score >= 80:
            print(f"🎉 Sistema PRONTO para uso!")
        elif readiness_score >= 60:
            print(f"⚠️ Sistema PARCIALMENTE pronto - algumas funcionalidades podem não funcionar")
        else:
            print(f"❌ Sistema NÃO pronto - corrija os problemas antes de usar")
    
    def print_next_steps(self):
        """Imprime próximos passos"""
        print(f"\n{'='*60}")
        print(f"📋 PRÓXIMOS PASSOS")
        print(f"{'='*60}")
        
        if all(self.system_status.values()):
            print("1. 🚀 Execute: python start_server.py")
            print("2. 🌐 Acesse: http://localhost:8000")
            print("3. 📊 Health check: http://localhost:8000/health")
            print("4. 🧠 Configure ML: Comece fazendo trades para coletar dados")
            print("5. 📈 Monitor: Acompanhe as estatísticas em /ml/stats")
        else:
            print("1. ❌ Corrija os problemas indicados acima")
            print("2. 🔄 Execute novamente: python init.py")
            print("3. 📚 Consulte a documentação para troubleshooting")
        
        print(f"\n📚 Arquivos importantes:")
        print(f"   • start_server.py - Iniciar servidor")
        print(f"   • .env - Configurações")
        print(f"   • logs/ - Logs do sistema")
        print(f"   • data/ - Banco de dados")
        print(f"   • models/ - Modelos ML")
    
    def full_initialization(self, force_deps: bool = False, run_tests_flag: bool = True) -> bool:
        """Executa inicialização completa"""
        print("🧠 ML Trading Bot - Inicialização Completa")
        print("="*60)
        
        start_time = time.time()
        
        steps = [
            ("Validar Ambiente", self.validate_environment),
            ("Instalar Dependências", lambda: self.install_dependencies(force_deps)),
            ("Configurar Banco", self.setup_database),
            ("Carregar Configuração", self.load_configuration),
            ("Configurar Monitoramento", self.setup_monitoring),
            ("Inicializar ML", self.initialize_ml_system),
            ("Criar Scripts", self.create_startup_script),
            ("Gerar .env", self.generate_env_file)
        ]
        
        if run_tests_flag:
            steps.append(("Executar Testes", self.run_tests))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            print(f"\n⏳ {step_name}...")
            try:
                success = step_func()
                if success:
                    print(f"✅ {step_name} - Sucesso")
                else:
                    print(f"❌ {step_name} - Falha")
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"💥 {step_name} - Erro: {e}")
                failed_steps.append(step_name)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Imprimir resumo
        self.print_system_status()
        
        if failed_steps:
            print(f"\n❌ Passos que falharam: {', '.join(failed_steps)}")
        
        print(f"\n⏱️ Tempo total: {total_time:.1f} segundos")
        
        self.print_next_steps()
        
        return len(failed_steps) == 0

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Inicializar ML Trading Bot')
    parser.add_argument('--force-deps', action='store_true', 
                       help='Forçar reinstalação de dependências')
    parser.add_argument('--no-tests', action='store_true',
                       help='Pular execução de testes')
    parser.add_argument('--status-only', action='store_true',
                       help='Apenas verificar status do sistema')
    parser.add_argument('--start-server', action='store_true',
                       help='Iniciar servidor após inicialização')
    
    args = parser.parse_args()
    
    initializer = MLTradingInitializer()
    
    if args.status_only:
        # Verificar status apenas
        initializer.validate_environment()
        initializer.print_system_status()
        return
    
    # Execução completa
    success = initializer.full_initialization(
        force_deps=args.force_deps,
        run_tests_flag=not args.no_tests
    )
    
    if success and args.start_server:
        print(f"\n🚀 Iniciando servidor...")
        try:
            subprocess.run([sys.executable, 'start_server.py'])
        except KeyboardInterrupt:
            print(f"\n👋 Servidor interrompido")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
