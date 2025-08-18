import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    path: str = "data/trading_data.db"
    backup_interval_hours: int = 24
    max_backup_files: int = 7
    connection_timeout: int = 30
    
@dataclass
class MLConfig:
    """Configurações de Machine Learning"""
    # Modelos
    models_to_train: List[str] = None
    model_save_path: str = "models"
    
    # Treinamento
    min_trades_for_training: int = 50
    auto_retrain_interval: int = 50  # A cada X trades
    retrain_on_startup: bool = True
    
    # Features
    feature_engineering: bool = True
    feature_selection: bool = True
    max_features: int = 20
    
    # Validação
    test_size: float = 0.2
    cross_validation_folds: int = 5
    stratify: bool = True
    
    # Performance
    pattern_confidence_threshold: float = 0.7
    min_pattern_occurrences: int = 5
    max_patterns_stored: int = 100
    
    # Predições
    prediction_confidence_threshold: float = 0.6
    ensemble_voting: str = "soft"  # soft, hard
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = [
                "random_forest",
                "gradient_boosting", 
                "logistic_regression",
                "svm",
                "neural_network"
            ]

@dataclass
class APIConfig:
    """Configurações da API"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    
    # CORS
    allow_origins: List[str] = None
    allow_credentials: bool = True
    allow_methods: List[str] = None
    allow_headers: List[str] = None
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_calls: int = 100
    rate_limit_period: int = 60  # seconds
    
    # Timeouts
    request_timeout: int = 30
    keepalive_timeout: int = 5
    
    def __post_init__(self):
        if self.allow_origins is None:
            self.allow_origins = ["*"]
        if self.allow_methods is None:
            self.allow_methods = ["*"]
        if self.allow_headers is None:
            self.allow_headers = ["*"]

@dataclass
class LoggingConfig:
    """Configurações de logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Arquivos
    log_to_file: bool = True
    log_file_path: str = "logs/ml_trading.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # Console
    log_to_console: bool = True
    console_level: str = "WARNING"
    
    # Módulos específicos
    module_levels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.module_levels is None:
            self.module_levels = {
                "uvicorn": "WARNING",
                "fastapi": "INFO",
                "sklearn": "WARNING",
                "pandas": "WARNING"
            }

@dataclass
class MonitoringConfig:
    """Configurações de monitoramento"""
    enabled: bool = True
    
    # Métricas
    collect_ml_metrics: bool = True
    collect_trading_metrics: bool = True
    collect_performance_metrics: bool = True
    
    # Relatórios
    daily_reports: bool = True
    report_time_hour: int = 23  # 23:00
    
    # Alertas
    alerts_enabled: bool = True
    alert_thresholds: Dict[str, float] = None
    
    # Health checks
    health_check_interval_minutes: int = 5
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "low_accuracy": 0.45,
                "low_win_rate": 30.0,
                "high_drawdown": -20.0,
                "low_f1_score": 0.4
            }

@dataclass
class SecurityConfig:
    """Configurações de segurança"""
    api_key_required: bool = False
    api_key: Optional[str] = None
    
    # Rate limiting por IP
    ip_rate_limiting: bool = True
    max_requests_per_ip: int = 1000
    ip_ban_duration_minutes: int = 60
    
    # Headers de segurança
    security_headers: bool = True
    
    # HTTPS
    force_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class PerformanceConfig:
    """Configurações de performance"""
    # Cache
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    
    # Processamento
    max_concurrent_requests: int = 100
    background_tasks_enabled: bool = True
    
    # Memória
    max_memory_usage_mb: int = 512
    gc_threshold: int = 1000
    
    # Banco de dados
    connection_pool_size: int = 10
    max_overflow: int = 20

class Config:
    """Configuração principal do sistema"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Carregar configurações padrão
        self.database = DatabaseConfig()
        self.ml = MLConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Carregar de arquivo se especificado
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Sobrescrever com variáveis de ambiente
        self.load_from_env()
        
        # Validar configurações
        self.validate()
        
        # Criar diretórios necessários
        self.create_directories()
    
    def load_from_file(self, config_file: str):
        """Carrega configurações de arquivo JSON"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Atualizar configurações
            for section, values in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                            
        except Exception as e:
            print(f"Erro ao carregar configurações do arquivo: {e}")
    
    def load_from_env(self):
        """Carrega configurações de variáveis de ambiente"""
        env_mappings = {
            # Database
            'DATABASE_PATH': ('database', 'path'),
            'DB_BACKUP_INTERVAL': ('database', 'backup_interval_hours'),
            
            # ML
            'MIN_TRADES_FOR_TRAINING': ('ml', 'min_trades_for_training'),
            'AUTO_RETRAIN_INTERVAL': ('ml', 'auto_retrain_interval'),
            'PATTERN_CONFIDENCE_THRESHOLD': ('ml', 'pattern_confidence_threshold'),
            
            # API
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'API_DEBUG': ('api', 'debug'),
            'API_WORKERS': ('api', 'workers'),
            
            # Logging
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_FILE_PATH': ('logging', 'log_file_path'),
            
            # Monitoring
            'MONITORING_ENABLED': ('monitoring', 'enabled'),
            'DAILY_REPORTS': ('monitoring', 'daily_reports'),
            
            # Security
            'API_KEY': ('security', 'api_key'),
            'API_KEY_REQUIRED': ('security', 'api_key_required'),
            
            # Performance
            'MAX_MEMORY_MB': ('performance', 'max_memory_usage_mb'),
            'CACHE_ENABLED': ('performance', 'enable_cache'),
        }
        
        for env_var, (section, attr) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, section)
                
                # Converter tipo se necessário
                current_value = getattr(config_obj, attr)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(config_obj, attr, value)
    
    def validate(self):
        """Valida as configurações"""
        errors = []
        
        # Validar ML
        if self.ml.min_trades_for_training < 10:
            errors.append("min_trades_for_training deve ser >= 10")
        
        if not 0 < self.ml.pattern_confidence_threshold <= 1:
            errors.append("pattern_confidence_threshold deve estar entre 0 e 1")
        
        if not 0 < self.ml.test_size < 1:
            errors.append("test_size deve estar entre 0 e 1")
        
        # Validar API
        if not 1000 <= self.api.port <= 65535:
            errors.append("port deve estar entre 1000 e 65535")
        
        # Validar Performance
        if self.performance.max_memory_usage_mb < 128:
            errors.append("max_memory_usage_mb deve ser >= 128")
        
        if errors:
            raise ValueError(f"Erros de configuração: {'; '.join(errors)}")
    
    def create_directories(self):
        """Cria diretórios necessários"""
        directories = [
            Path(self.database.path).parent,
            Path(self.ml.model_save_path),
            Path(self.logging.log_file_path).parent,
            Path("data"),
            Path("reports"),
            Path("backups")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_to_file(self, config_file: str):
        """Salva configurações para arquivo"""
        config_dict = {}
        
        for attr_name in ['database', 'ml', 'api', 'logging', 'monitoring', 'security', 'performance']:
            config_obj = getattr(self, attr_name)
            config_dict[attr_name] = {}
            
            for field_name in config_obj.__dataclass_fields__:
                value = getattr(config_obj, field_name)
                config_dict[attr_name][field_name] = value
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def get_environment_summary(self) -> Dict:
        """Retorna resumo do ambiente"""
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "python_version": os.getenv("PYTHON_VERSION", "unknown"),
            "port": self.api.port,
            "debug": self.api.debug,
            "ml_models": len(self.ml.models_to_train),
            "min_training_data": self.ml.min_trades_for_training,
            "monitoring_enabled": self.monitoring.enabled,
            "security_enabled": self.security.api_key_required,
            "cache_enabled": self.performance.enable_cache,
            "max_memory_mb": self.performance.max_memory_usage_mb
        }
    
    def __str__(self):
        """Representação string da configuração"""
        summary = self.get_environment_summary()
        return f"MLTradingConfig({', '.join(f'{k}={v}' for k, v in summary.items())})"

# Instância global de configuração
config = Config()

# Função para recarregar configurações
def reload_config(config_file: Optional[str] = None):
    """Recarrega as configurações"""
    global config
    config = Config(config_file)
    return config

# Função para obter configuração por seção
def get_config(section: str):
    """Obtém configuração de uma seção específica"""
    return getattr(config, section, None)

# Função para atualizar configuração
def update_config(section: str, **kwargs):
    """Atualiza configuração de uma seção"""
    if hasattr(config, section):
        config_obj = getattr(config, section)
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
        return True
    return False

# Decorator para verificar configuração
def requires_config(section: str, attribute: str):
    """Decorator que verifica se uma configuração está definida"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config_obj = getattr(config, section, None)
            if not config_obj or not getattr(config_obj, attribute, None):
                raise ValueError(f"Configuração {section}.{attribute} não definida")
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Exemplo de uso
    print("🔧 Configurações do ML Trading Bot")
    print("=" * 40)
    
    print(f"Configuração carregada: {config}")
    print(f"\nResumo do ambiente:")
    for key, value in config.get_environment_summary().items():
        print(f"  {key}: {value}")
    
    # Salvar configuração exemplo
    config.save_to_file("config_example.json")
    print(f"\n💾 Configuração salva em: config_example.json")
