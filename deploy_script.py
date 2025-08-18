#!/usr/bin/env python3
"""
Script de deploy automatizado para ML Trading Bot na Render
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RenderDeployer:
    """Classe para automatizar deploy na Render"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RENDER_API_KEY')
        self.base_url = "https://api.render.com/v1"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Configurações do projeto
        self.project_config = {
            'name': 'ml-trading-bot',
            'type': 'web_service',
            'repo': None,  # Será definido pelo usuário
            'branch': 'main',
            'build_command': 'pip install -r requirements.txt',
            'start_command': 'python main.py',
            'env_vars': {
                'PYTHON_VERSION': '3.11.0',
                'PORT': '8000',
                'LOG_LEVEL': 'INFO',
                'MIN_TRADES_FOR_TRAINING': '50',
                'AUTO_RETRAIN_INTERVAL': '50'
            }
        }
    
    def validate_environment(self) -> bool:
        """Valida se o ambiente está pronto para deploy"""
        logger.info("🔍 Validando ambiente de deploy...")
        
        required_files = [
            'main.py',
            'requirements.txt',
            'config.py',
            'monitoring.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Arquivos obrigatórios não encontrados: {missing_files}")
            return False
        
        # Verificar requirements.txt
        try:
            with open('requirements.txt', 'r') as f:
                requirements = f.read()
                
            required_packages = [
                'fastapi', 'uvicorn', 'scikit-learn', 
                'pandas', 'numpy', 'pydantic'
            ]
            
            missing_packages = []
            for package in required_packages:
                if package not in requirements:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.warning(f"⚠️ Pacotes importantes não encontrados em requirements.txt: {missing_packages}")
        
        except Exception as e:
            logger.error(f"❌ Erro ao verificar requirements.txt: {e}")
            return False
        
        logger.info("✅ Ambiente validado com sucesso")
        return True
    
    def create_render_yaml(self) -> bool:
        """Cria arquivo render.yaml se não existir"""
        logger.info("📝 Criando arquivo render.yaml...")
        
        render_config = {
            'services': [{
                'type': 'web',
                'name': self.project_config['name'],
                'env': 'python',
                'plan': 'free',
                'buildCommand': self.project_config['build_command'],
                'startCommand': self.project_config['start_command'],
                'healthCheckPath': '/health',
                'envVars': []
            }]
        }
        
        # Adicionar variáveis de ambiente
        for key, value in self.project_config['env_vars'].items():
            render_config['services'][0]['envVars'].append({
                'key': key,
                'value': str(value)
            })
        
        try:
            import yaml
            with open('render.yaml', 'w') as f:
                yaml.dump(render_config, f, default_flow_style=False)
            
            logger.info("✅ Arquivo render.yaml criado")
            return True
            
        except ImportError:
            # Fallback para JSON se PyYAML não estiver disponível
            with open('render.yaml', 'w') as f:
                f.write(f"""services:
  - type: web
    name: {self.project_config['name']}
    env: python
    plan: free
    buildCommand: {self.project_config['build_command']}
    startCommand: {self.project_config['start_command']}
    healthCheckPath: /health
    envVars:
""")
                for key, value in self.project_config['env_vars'].items():
                    f.write(f"      - key: {key}\n        value: {value}\n")
            
            logger.info("✅ Arquivo render.yaml criado (formato YAML manual)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar render.yaml: {e}")
            return False
    
    def get_services(self) -> List[Dict]:
        """Lista serviços existentes"""
        if not self.api_key:
            logger.warning("⚠️ API key não configurada - modo manual")
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/services",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Erro ao listar serviços: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erro na API Render: {e}")
            return []
    
    def create_service(self, repo_url: str) -> Optional[str]:
        """Cria novo serviço na Render"""
        if not self.api_key:
            logger.warning("⚠️ API key não configurada - configure manualmente na Render")
            return None
        
        logger.info("🚀 Criando serviço na Render...")
        
        service_data = {
            'type': 'web_service',
            'name': self.project_config['name'],
            'repo': repo_url,
            'branch': self.project_config['branch'],
            'buildCommand': self.project_config['build_command'],
            'startCommand': self.project_config['start_command'],
            'envVars': [
                {'key': key, 'value': str(value)} 
                for key, value in self.project_config['env_vars'].items()
            ],
            'plan': 'free',
            'region': 'oregon',
            'healthCheckPath': '/health'
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/services",
                headers=self.headers,
                json=service_data
            )
            
            if response.status_code == 201:
                service = response.json()
                service_id = service.get('id')
                logger.info(f"✅ Serviço criado: {service_id}")
                return service_id
            else:
                logger.error(f"❌ Erro ao criar serviço: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro na criação do serviço: {e}")
            return None
    
    def wait_for_deployment(self, service_id: str, timeout: int = 600) -> bool:
        """Aguarda conclusão do deploy"""
        if not self.api_key:
            return True
        
        logger.info("⏳ Aguardando conclusão do deploy...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/services/{service_id}",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    service = response.json()
                    status = service.get('serviceDetails', {}).get('deployStatus')
                    
                    if status == 'live':
                        logger.info("✅ Deploy concluído com sucesso!")
                        return True
                    elif status == 'build_failed':
                        logger.error("❌ Deploy falhou!")
                        return False
                    else:
                        logger.info(f"🔄 Status: {status}")
                        time.sleep(30)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao verificar status: {e}")
                time.sleep(30)
        
        logger.error("⏰ Timeout no deploy")
        return False
    
    def test_deployment(self, service_url: str) -> bool:
        """Testa se o deploy está funcionando"""
        logger.info("🧪 Testando deployment...")
        
        test_endpoints = [
            ('/', 'Endpoint raiz'),
            ('/health', 'Health check'),
            ('/ml/stats', 'Estatísticas ML (sem auth)')
        ]
        
        for endpoint, description in test_endpoints:
            try:
                url = f"{service_url.rstrip('/')}{endpoint}"
                response = requests.get(url, timeout=30)
                
                if response.status_code in [200, 401, 503]:  # 401 = auth required, 503 = service starting
                    logger.info(f"✅ {description}: OK")
                else:
                    logger.warning(f"⚠️ {description}: Status {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ {description}: Erro - {e}")
        
        logger.info("🎉 Teste de deployment concluído!")
        return True
    
    def deploy(self, repo_url: str) -> bool:
        """Executa deploy completo"""
        logger.info("🚀 Iniciando deploy do ML Trading Bot")
        
        # 1. Validar ambiente
        if not self.validate_environment():
            return False
        
        # 2. Criar render.yaml
        if not self.create_render_yaml():
            return False
        
        # 3. Verificar se serviço já existe
        services = self.get_services()
        existing_service = None
        
        for service in services:
            if service.get('name') == self.project_config['name']:
                existing_service = service
                break
        
        if existing_service:
            logger.info(f"📦 Serviço existente encontrado: {existing_service.get('id')}")
            service_id = existing_service.get('id')
            service_url = existing_service.get('serviceDetails', {}).get('url')
        else:
            # 4. Criar novo serviço
            service_id = self.create_service(repo_url)
            if not service_id:
                logger.error("❌ Falha na criação do serviço")
                return False
            
            # 5. Aguardar deploy
            if not self.wait_for_deployment(service_id):
                return False
            
            # 6. Obter URL do serviço
            service_url = f"https://{self.project_config['name']}.onrender.com"
        
        # 7. Testar deployment
        if service_url:
            time.sleep(10)  # Aguardar serviço inicializar
            self.test_deployment(service_url)
        
        logger.info("🎉 Deploy concluído com sucesso!")
        logger.info(f"🌐 URL do serviço: {service_url}")
        
        return True

def create_project_structure():
    """Cria estrutura básica do projeto"""
    logger.info("📁 Criando estrutura do projeto...")
    
    directories = [
        'models',
        'data', 
        'logs',
        'backups',
        'reports',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # Criar .gitkeep para manter diretórios vazios no Git
        gitkeep_file = Path(directory) / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    logger.info("✅ Estrutura do projeto criada")

def generate_env_example():
    """Gera arquivo .env.example"""
    logger.info("🔧 Gerando .env.example...")
    
    env_content = """# Configurações da API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_KEY_REQUIRED=false
API_KEY=your-secret-api-key

# Configurações do ML
MIN_TRADES_FOR_TRAINING=50
AUTO_RETRAIN_INTERVAL=50
PATTERN_CONFIDENCE_THRESHOLD=0.7

# Configurações do Banco
DATABASE_PATH=data/trading_data.db

# Configurações de Log
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/ml_trading.log

# Configurações de Performance
MAX_MEMORY_MB=512
CACHE_ENABLED=true

# Configurações de Monitoramento
MONITORING_ENABLED=true
DAILY_REPORTS=true

# Render API (opcional)
RENDER_API_KEY=your-render-api-key
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    logger.info("✅ Arquivo .env.example criado")

def setup_git_hooks():
    """Configura Git hooks para deploy automático"""
    logger.info("🔗 Configurando Git hooks...")
    
    try:
        # Verificar se é um repositório Git
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        
        # Criar hook pre-commit
        hooks_dir = Path('.git/hooks')
        if hooks_dir.exists():
            pre_commit_hook = hooks_dir / 'pre-commit'
            
            hook_content = """#!/bin/bash
# Pre-commit hook para validar código
echo "🔍 Validando código antes do commit..."

# Verificar arquivos Python
python -m py_compile main.py config.py monitoring.py
if [ $? -ne 0 ]; then
    echo "❌ Erro de sintaxe Python encontrado!"
    exit 1
fi

# Verificar requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt não encontrado!"
    exit 1
fi

echo "✅ Validação concluída"
"""
            
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)
            
            # Tornar executável
            os.chmod(pre_commit_hook, 0o755)
            
            logger.info("✅ Git hooks configurados")
        
    except subprocess.CalledProcessError:
        logger.warning("⚠️ Não é um repositório Git - hooks não configurados")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao configurar Git hooks: {e}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Deploy ML Trading Bot na Render')
    parser.add_argument('--repo', type=str, help='URL do repositório GitHub')
    parser.add_argument('--api-key', type=str, help='Render API Key')
    parser.add_argument('--setup-only', action='store_true', help='Apenas configurar projeto')
    parser.add_argument('--test-url', type=str, help='Testar URL específica')
    
    args = parser.parse_args()
    
    print("🧠 ML Trading Bot - Deploy Automatizado")
    print("=" * 50)
    
    # Setup do projeto
    if args.setup_only or not args.repo:
        create_project_structure()
        generate_env_example()
        setup_git_hooks()
        
        if not args.repo:
            print("\n📋 Próximos passos:")
            print("1. Faça upload dos arquivos para um repositório GitHub")
            print("2. Execute: python deploy.py --repo https://github.com/seu-usuario/seu-repo.git")
            print("3. Ou configure manualmente na Render usando os arquivos gerados")
            return
    
    # Deploy
    if args.repo:
        deployer = RenderDeployer(args.api_key)
        
        if deployer.deploy(args.repo):
            print("\n🎉 Deploy realizado com sucesso!")
            print("🌐 Acesse o painel da Render para monitorar o serviço")
        else:
            print("\n❌ Deploy falhou. Verifique os logs acima.")
            sys.exit(1)
    
    # Teste de URL específica
    if args.test_url:
        deployer = RenderDeployer()
        deployer.test_deployment(args.test_url)

if __name__ == "__main__":
    main()
