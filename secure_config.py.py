#!/usr/bin/env python3
"""
Configuração Segura e Atualizada para Trading Bot
Implementa todas as melhorias de segurança e risk management
"""

import os
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Configurações de segurança avançadas"""
    # API Security
    api_key_required: bool = True
    api_key_header: str = "X-API-Key"
    api_key_hash: Optional[str] = None
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    rate_limit_by_ip: bool = True
    
    # Request Security
    max_request_size_mb: int = 10
    allowed_content_types: List[str] = field(default_factory=lambda: [
        "application/json", 
        "multipart/form-data"
    ])
    
    # CORS Security
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://your-domain.com"  # Substituir em produção
    ])
    cors_credentials: bool = True
    cors_max_age: int = 3600
    
    # Headers de Segurança
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    })
    
    # SSL/TLS
    force_https: bool = False  # Ativar em produção
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class RiskManagementConfig:
    """Configurações avançadas de Risk Management"""
    # Limites Básicos
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 10.0
    max_monthly_loss_pct: float = 20.0
    max_drawdown_pct: float = 15.0
    
    # Position Sizing
    max_position_size_pct: float = 2.0  # Kelly será limitado a este valor
    min_position_size: float = 0.35
    max_position_size: float = 100.0
    
    # Trading Limits
    max_concurrent_trades: int = 3
    max_trades_per_hour: int = 20
    max_trades_per_day: int = 100
    
    # Performance Thresholds
    min_win_rate_threshold: float = 30.0
    max_consecutive_losses: int = 5
    min_profit_factor: float = 1.0
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_loss_threshold: float = 3.0  # % em 1 hora
    circuit_breaker_cooldown_minutes: int = 60
    
    # Kelly Criterion
    kelly_enabled: bool = True
    kelly_fraction_cap: float = 0.25  # Máximo 25% (Quarter Kelly)
    kelly_confidence_factor: float = 0.8  # Fator de desconto
    kelly_min_trades_required: int = 10
    
    # Stop Loss Dinâmico
    dynamic_stop_loss: bool = True
    stop_loss_pct: float = 2.0  # Por trade
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 1.0

@dataclass
class MLConfig:
    """Configurações ML Avançadas"""
    # Modelos
    ensemble_enabled: bool = True
    models_to_train: List[str] = field(default_factory=lambda: [
        "random_forest",
        "gradient_boosting", 
        "logistic_regression",
        "svm",
        "neural_network",
        "xgboost"  # Adicionado XGBoost
    ])
    
    # Feature Engineering
    advanced_features_enabled: bool = True
    total_features: int = 35
    feature_selection_enabled: bool = True
    max_features_selected: int = 20
    
    # Validation
    temporal_validation: bool = True  # Nunca misturar dados futuros
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_method: str = "time_series_split"
    
    # Training
    min_trades_for_training: int = 100  # Aumentado
    auto_retrain_interval: int = 30  # A cada 30 trades
    retrain_on_startup: bool = True
    
    # Prediction Thresholds
    min_confidence_threshold: float = 0.65
    ensemble_voting: str = "soft"  # probabilidades
    prediction_timeout_seconds: int = 15
    
    # Performance Tracking
    track_feature_importance: bool = True
    track_model_drift: bool = True
    performance_window_trades: int = 50

@dataclass
class DatabaseConfig:
    """Configurações de Banco de Dados Seguras"""
    # Paths
    db_path: str = "data/trading_data_secure.db"
    backup_path: str = "backups/"
    
    # Security
    enable_encryption: bool = False  # Implementar se necessário
    encryption_key_path: Optional[str] = None
    
    # Backup
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 6  # Mais frequente
    max_backup_files: int = 24  # 1 semana de backups
    compress_backups: bool = True
    
    # Connection
    connection_timeout: int = 30
    connection_pool_size: int = 5
    max_overflow: int = 10
    
    # Maintenance
    auto_vacuum: bool = True
    vacuum_interval_hours: int = 24
    
    # Data Retention
    keep_trades_days: int = 365  # 1 ano
    keep_metrics_days: int = 90   # 3 meses
    keep_logs_days: int = 30      # 1 mês

@dataclass
class MonitoringConfig:
    """Configurações de Monitoramento Avançado"""
    # Health Checks
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    
    # Metrics Collection
    collect_system_metrics: bool = True
    collect_trading_metrics: bool = True
    collect_ml_metrics: bool = True
    collect_risk_metrics: bool = True
    
    # Alerts
    alerts_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["console", "file"])
    
    # Alert Thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_cpu_usage": 80.0,
        "high_memory_usage": 85.0,
        "low_ml_accuracy": 45.0,
        "low_win_rate": 30.0,
        "high_drawdown": 10.0,
        "high_risk_score": 75.0,
        "circuit_breaker_triggered": 1.0
    })
    
    # Reporting
    daily_reports_enabled: bool = True
    weekly_reports_enabled: bool = True
    report_email: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/secure_trading.log"
    log_rotation_size_mb: int = 50
    log_retention_days: int = 30

@dataclass
class TradingConfig:
    """Configurações de Trading Seguras"""
    # Symbols
    allowed_symbols: List[str] = field(default_factory=lambda: [
        # Volatility Indices
        "R_10", "R_25", "R_50", "R_75", "R_100",
        # Volatility Indices (1s)
        "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
        # Jump Indices  
        "JD10", "JD25", "JD50", "JD75", "JD100",
        # Crash/Boom (limitado para segurança)
        "CRASH500", "BOOM500"
    ])
    
    # Duration Limits
    min_duration_ticks: int = 5
    max_duration_ticks: int = 10
    min_duration_minutes: int = 1
    max_duration_minutes: int = 15  # Limitado para reduzir exposição
    
    # Market Hours (UTC)
    trading_hours_enabled: bool = True
    trading_start_hour: int = 6   # 06:00 UTC
    trading_end_hour: int = 22    # 22:00 UTC
    weekend_trading: bool = False
    
    # Auto Trading
    auto_trading_enabled: bool = True
    auto_trading_max_duration_hours: int = 8  # Máximo 8h contínuas
    auto_trading_break_minutes: int = 30      # Pausa obrigatória
    
    # API Limits (Deriv)
    api_requests_per_second: int = 5
    api_requests_per_minute: int = 100
    api_timeout_seconds: int = 30
    
    # WebSocket
    ws_reconnect_enabled: bool = True
    ws_max_reconnect_attempts: int = 5
    ws_reconnect_delay_seconds: int = 5

class SecureConfigManager:
    """Gerenciador de Configuração Segura"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "secure_config.json"
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Configurações
        self.security = SecurityConfig()
        self.risk_management = RiskManagementConfig()
        self.ml = MLConfig()
        self.database = DatabaseConfig()
        self.monitoring = MonitoringConfig()
        self.trading = TradingConfig()
        
        # Carregamento
        self.load_from_file()
        self.load_from_environment()
        self.validate_config()
        self.setup_security()
        
        logger.info("🔐 Configuração segura carregada")
    
    def setup_security(self):
        """Configura segurança inicial"""
        # Gerar API key se não existir
        if self.security.api_key_required and not self.security.api_key_hash:
            api_key = self.generate_api_key()
            self.security.api_key_hash = self.hash_api_key(api_key)
            logger.warning(f"🔑 Nova API Key gerada: {api_key}")
            logger.warning("⚠️ SALVE ESTA CHAVE EM LOCAL SEGURO!")
        
        # Criar diretórios seguros
        self.create_secure_directories()
    
    def generate_api_key(self) -> str:
        """Gera API key segura"""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash da API key para armazenamento seguro"""
        salt = secrets.token_hex(16)
        key_hash = hashlib.pbkdf2_hmac('sha256', 
                                       api_key.encode(), 
                                       salt.encode(), 
                                       100000)
        return f"{salt}:{key_hash.hex()}"
    
    def verify_api_key(self, provided_key: str) -> bool:
        """Verifica API key"""
        if not self.security.api_key_hash:
            return True  # Sem autenticação configurada
        
        try:
            salt, stored_hash = self.security.api_key_hash.split(':')
            key_hash = hashlib.pbkdf2_hmac('sha256',
                                           provided_key.encode(),
                                           salt.encode(),
                                           100000)
            return key_hash.hex() == stored_hash
        except Exception:
            return False
    
    def create_secure_directories(self):
        """Cria diretórios com permissões seguras"""
        directories = [
            Path(self.database.db_path).parent,
            Path(self.database.backup_path),
            Path(self.monitoring.log_file_path).parent,
            Path("data"),
            Path("models"),
            Path("reports"),
            Path("temp")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Definir permissões seguras no Unix
            if os.name != 'nt':  # Não Windows
                os.chmod(directory, 0o755)
    
    def load_from_file(self):
        """Carrega configuração do arquivo"""
        config_path = self.config_dir / self.config_file
        
        if not config_path.exists():
            logger.info("📄 Arquivo de configuração não encontrado, usando padrões")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Aplicar configurações carregadas
            for section, data in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
            
            logger.info(f"📄 Configuração carregada de {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar configuração: {e}")
    
    def load_from_environment(self):
        """Carrega configurações de variáveis de ambiente"""
        env_mappings = {
            # Security
            'API_KEY_REQUIRED': ('security', 'api_key_required', bool),
            'FORCE_HTTPS': ('security', 'force_https', bool),
            'CORS_ORIGINS': ('security', 'cors_origins', 'list'),
            
            # Risk Management
            'MAX_DAILY_LOSS_PCT': ('risk_management', 'max_daily_loss_pct', float),
            'MAX_DRAWDOWN_PCT': ('risk_management', 'max_drawdown_pct', float),
            'MAX_POSITION_SIZE_PCT': ('risk_management', 'max_position_size_pct', float),
            'CIRCUIT_BREAKER_ENABLED': ('risk_management', 'circuit_breaker_enabled', bool),
            'KELLY_ENABLED': ('risk_management', 'kelly_enabled', bool),
            
            # ML
            'ENSEMBLE_ENABLED': ('ml', 'ensemble_enabled', bool),
            'MIN_TRADES_FOR_TRAINING': ('ml', 'min_trades_for_training', int),
            'MIN_CONFIDENCE_THRESHOLD': ('ml', 'min_confidence_threshold', float),
            'TEMPORAL_VALIDATION': ('ml', 'temporal_validation', bool),
            
            # Database
            'DB_PATH': ('database', 'db_path', str),
            'AUTO_BACKUP_ENABLED': ('database', 'auto_backup_enabled', bool),
            'BACKUP_INTERVAL_HOURS': ('database', 'backup_interval_hours', int),
            
            # Monitoring
            'HEALTH_CHECK_ENABLED': ('monitoring', 'health_check_enabled', bool),
            'ALERTS_ENABLED': ('monitoring', 'alerts_enabled', bool),
            'LOG_LEVEL': ('monitoring', 'log_level', str),
            
            # Trading
            'WEEKEND_TRADING': ('trading', 'weekend_trading', bool),
            'AUTO_TRADING_ENABLED': ('trading', 'auto_trading_enabled', bool),
        }
        
        for env_var, (section, attr, type_hint) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, section)
                
                # Converter tipo
                if type_hint == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif type_hint == int:
                    value = int(value)
                elif type_hint == float:
                    value = float(value)
                elif type_hint == 'list':
                    value = [item.strip() for item in value.split(',')]
                
                setattr(config_obj, attr, value)
                logger.info(f"🌍 Env override: {section}.{attr} = {value}")
    
    def validate_config(self):
        """Valida configurações"""
        errors = []
        
        # Validar Risk Management
        if not 0 < self.risk_management.max_daily_loss_pct <= 50:
            errors.append("max_daily_loss_pct deve estar entre 0 e 50")
        
        if not 0 < self.risk_management.max_position_size_pct <= 10:
            errors.append("max_position_size_pct deve estar entre 0 e 10")
        
        if self.risk_management.max_concurrent_trades < 1:
            errors.append("max_concurrent_trades deve ser >= 1")
        
        # Validar ML
        if self.ml.min_trades_for_training < 20:
            errors.append("min_trades_for_training deve ser >= 20")
        
        if not 0 < self.ml.min_confidence_threshold <= 1:
            errors.append("min_confidence_threshold deve estar entre 0 e 1")
        
        # Validar Trading
        if self.trading.min_duration_ticks < 1:
            errors.append("min_duration_ticks deve ser >= 1")
        
        if not self.trading.allowed_symbols:
            errors.append("allowed_symbols não pode estar vazio")
        
        if errors:
            raise ValueError(f"Erros de configuração: {'; '.join(errors)}")
        
        logger.info("✅ Configuração validada com sucesso")
    
    def save_to_file(self):
        """Salva configuração no arquivo"""
        config_path = self.config_dir / self.config_file
        
        config_dict = {
            'security': self._dataclass_to_dict(self.security),
            'risk_management': self._dataclass_to_dict(self.risk_management),
            'ml': self._dataclass_to_dict(self.ml),
            'database': self._dataclass_to_dict(self.database),
            'monitoring': self._dataclass_to_dict(self.monitoring),
            'trading': self._dataclass_to_dict(self.trading),
            'metadata': {
                'version': '2.0.0',
                'created_at': datetime.now().isoformat(),
                'description': 'Configuração Segura do Trading Bot'
            }
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # Definir permissões seguras
            if os.name != 'nt':
                os.chmod(config_path, 0o600)  # Apenas proprietário
            
            logger.info(f"💾 Configuração salva em {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar configuração: {e}")
    
    def _dataclass_to_dict(self, obj) -> Dict:
        """Converte dataclass para dict"""
        result = {}
        for field_name, field_obj in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            
            # Não salvar senhas/hashes em plain text
            if 'password' in field_name.lower() or 'hash' in field_name.lower():
                if value:
                    result[field_name] = '[PROTECTED]'
                continue
            
            result[field_name] = value
        
        return result
    
    def get_risk_limits_dict(self) -> Dict[str, float]:
        """Retorna limites de risco para JavaScript"""
        return {
            'MAX_DAILY_LOSS_PCT': self.risk_management.max_daily_loss_pct,
            'MAX_WEEKLY_LOSS_PCT': self.risk_management.max_weekly_loss_pct,
            'MAX_MONTHLY_LOSS_PCT': self.risk_management.max_monthly_loss_pct,
            'MAX_DRAWDOWN_PCT': self.risk_management.max_drawdown_pct,
            'MAX_POSITION_SIZE_PCT': self.risk_management.max_position_size_pct,
            'MAX_CONCURRENT_TRADES': self.risk_management.max_concurrent_trades,
            'MIN_WIN_RATE': self.risk_management.min_win_rate_threshold,
            'MAX_CONSECUTIVE_LOSSES': self.risk_management.max_consecutive_losses,
            'CIRCUIT_BREAKER_THRESHOLD': self.risk_management.circuit_breaker_loss_threshold
        }
    
    def get_ml_config_dict(self) -> Dict[str, Any]:
        """Retorna configuração ML para JavaScript"""
        return {
            'ENSEMBLE_ENABLED': self.ml.ensemble_enabled,
            'MIN_CONFIDENCE': self.ml.min_confidence_threshold,
            'LEARNING_ENABLED': True,
            'AUTO_FEEDBACK': True,
            'RETRAIN_INTERVAL': self.ml.auto_retrain_interval,
            'PREDICTION_TIMEOUT': self.ml.prediction_timeout_seconds * 1000,
            'TOTAL_FEATURES': self.ml.total_features
        }
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Retorna resumo de segurança"""
        return {
            'api_key_required': self.security.api_key_required,
            'rate_limiting': self.security.rate_limit_enabled,
            'cors_configured': len(self.security.cors_origins) > 0,
            'https_enforced': self.security.force_https,
            'risk_management': self.risk_management.circuit_breaker_enabled,
            'kelly_criterion': self.risk_management.kelly_enabled,
            'ml_validation': self.ml.temporal_validation,
            'monitoring': self.monitoring.alerts_enabled
        }
    
    def __str__(self) -> str:
        summary = self.get_security_summary()
        return f"SecureConfig(security_features={len([k for k, v in summary.items() if v])}/8)"

# Instância global
secure_config = SecureConfigManager()

# Função para recarregar configuração
def reload_secure_config(config_file: Optional[str] = None) -> SecureConfigManager:
    """Recarrega configuração segura"""
    global secure_config
    secure_config = SecureConfigManager(config_file)
    return secure_config

# Decorador para verificar configuração de segurança
def requires_security_feature(feature: str):
    """Decorator que verifica se feature de segurança está habilitada"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_features = {
                'api_key': secure_config.security.api_key_required,
                'rate_limiting': secure_config.security.rate_limit_enabled,
                'circuit_breaker': secure_config.risk_management.circuit_breaker_enabled,
                'kelly': secure_config.risk_management.kelly_enabled,
                'temporal_validation': secure_config.ml.temporal_validation
            }
            
            if not security_features.get(feature, False):
                raise ValueError(f"Feature de segurança '{feature}' não está habilitada")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Demonstração da configuração segura
    print("🔐 Configuração Segura do Trading Bot")
    print("=" * 50)
    
    config = SecureConfigManager()
    
    print(f"Configuração: {config}")
    print(f"\nResumo de Segurança:")
    for feature, enabled in config.get_security_summary().items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    print(f"\nLimites de Risco:")
    for limit, value in config.get_risk_limits_dict().items():
        print(f"  {limit}: {value}")
    
    # Salvar configuração de exemplo
    config.save_to_file()
    print(f"\n💾 Configuração salva em: config/secure_config.json")
    
    print(f"\n🔑 Features de Segurança Implementadas:")
    print(f"  🛡️ Risk Management com Circuit Breakers")
    print(f"  💰 Kelly Criterion para Position Sizing")
    print(f"  🔐 API Key Authentication")
    print(f"  🚦 Rate Limiting")
    print(f"  🌐 CORS Seguro")
    print(f"  📊 Validação Temporal ML")
    print(f"  📈 Monitoramento Completo")
    print(f"  🔄 Backup Automático")
