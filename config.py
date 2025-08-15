#!/usr/bin/env python3
"""
Trading Bot ML - Configurações Centralizadas
Gerenciamento de configurações para diferentes ambientes
"""

import os
from typing import List, Dict, Any
from pydantic import BaseSettings, validator
import logging


class Settings(BaseSettings):
    """Configurações centralizadas do sistema"""
    
    # ===============================
    # CONFIGURAÇÕES BÁSICAS
    # ===============================
    
    # Ambiente
    environment: str = "development"
    debug: bool = False
    
    # Aplicação
    app_name: str = "Trading Bot ML"
    app_version: str = "1.0.0"
    port: int = 8000
    host: str = "0.0.0.0"
    workers: int = 1
    
    # ===============================
    # CONFIGURAÇÕES DE API
    # ===============================
    
    # URLs
    deriv_ws_url: str = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    deriv_app_id: int = 1089
    
    # CORS
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # ===============================
    # CONFIGURAÇÕES DE ML
    # ===============================
    
    # Modelos
    ml_models_path: str = "./models"
    ml_history_size: int = 1000
    ml_retrain_interval_hours: int = 24
    ml_min_confidence: float = 60.0
    ml_workers: int = 2
    
    # Features
    ml_feature_window: int = 50
    ml_target_periods: List[int] = [3, 5, 10]
    ml_test_size: float = 0.2
    
    # Algoritmos
    ml_algorithms: Dict[str, Dict[str, Any]] = {
        "random_forest": {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 5
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "random_state": 42,
            "learning_rate": 0.1,
            "max_depth": 6
        }
    }
    
    # ===============================
    # CONFIGURAÇÕES DE TRADING
    # ===============================
    
    # Símbolos suportados
    supported_symbols: List[str] = [
        "R_10", "R_25", "R_50", "R_75", "R_100",
        "1HZ10V", "1HZ25V", "1HZ50V", "1HZ100V",
        "JD10", "JD25", "JD50", "JD75", "JD100"
    ]
    
    # Limites de trading
    min_stake: float = 0.35
    max_stake: float = 2000.0
    default_stake: float = 1.0
    
    # Timeframes
    supported_timeframes: List[str] = ["1m", "5m", "15m", "1h"]
    default_timeframe: str = "1m"
    
    # Durações (ticks)
    tick_durations: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Durações (minutos)
    minute_durations: List[int] = [1, 2, 3, 4, 5, 10, 15, 30, 60]
    
    # ===============================
    # CONFIGURAÇÕES DE DADOS
    # ===============================
    
    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 300  # 5 minutos
    
    # Histórico
    max_price_history: int = 1000
    data_collection_interval: int = 1  # segundos
    
    # ===============================
    # CONFIGURAÇÕES DE SEGURANÇA
    # ===============================
    
    # Chaves
    secret_key: str = "your-secret-key-change-this"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Headers de segurança
    security_headers: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
    
    # ===============================
    # CONFIGURAÇÕES DE LOG
    # ===============================
    
    # Níveis
    log_level: str = "INFO"
    log_file: str = "./logs/trading_bot.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "midnight"
    log_retention: int = 7  # dias
    
    # ===============================
    # CONFIGURAÇÕES DE MONITORAMENTO
    # ===============================
    
    # Métricas
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Health checks
    health_check_interval: int = 30  # segundos
    
    # Alertas
    alert_webhook_url: str = ""
    alert_email: str = ""
    
    # ===============================
    # CONFIGURAÇÕES EXTERNAS
    # ===============================
    
    # APIs externas (opcional)
    alpha_vantage_key: str = ""
    twelve_data_key: str = ""
    
    # Notificações
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Backup
    s3_bucket_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    
    # Monitoramento
    sentry_dsn: str = ""
    new_relic_license_key: str = ""
    
    # ===============================
    # VALIDADORES
    # ===============================
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @validator('ml_min_confidence')
    def validate_confidence(cls, v):
        if not 50.0 <= v <= 100.0:
            raise ValueError('ML confidence must be between 50.0 and 100.0')
        return v
    
    @validator('min_stake', 'max_stake')
    def validate_stakes(cls, v):
        if v <= 0:
            raise ValueError('Stake values must be positive')
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # ===============================
    # MÉTODOS UTILITÁRIOS
    # ===============================
    
    def is_development(self) -> bool:
        """Verifica se está em desenvolvimento"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Verifica se está em produção"""
        return self.environment == "production"
    
    def get_database_url(self) -> str:
        """Retorna URL do banco de dados"""
        return self.redis_url
    
    def get_log_config(self) -> Dict[str, Any]:
        """Retorna configuração de logging"""
        return {
            'level': getattr(logging, self.log_level),
            'format': self.log_format,
            'filename': self.log_file,
            'filemode': 'a'
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Retorna configuração CORS"""
        return {
            'allow_origins': self.cors_origins,
            'allow_methods': self.cors_methods,
            'allow_headers': self.cors_headers,
            'allow_credentials': True
        }
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Retorna configuração ML"""
        return {
            'history_size': self.ml_history_size,
            'min_confidence': self.ml_min_confidence,
            'workers': self.ml_workers,
            'algorithms': self.ml_algorithms,
            'target_periods': self.ml_target_periods
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Retorna configuração de trading"""
        return {
            'symbols': self.supported_symbols,
            'min_stake': self.min_stake,
            'max_stake': self.max_stake,
            'default_stake': self.default_stake,
            'timeframes': self.supported_timeframes,
            'tick_durations': self.tick_durations,
            'minute_durations': self.minute_durations
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# ===============================
# CONFIGURAÇÕES POR AMBIENTE
# ===============================

class DevelopmentSettings(Settings):
    """Configurações para desenvolvimento"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    workers: int = 1
    ml_history_size: int = 500  # Menor para desenvolvimento


class ProductionSettings(Settings):
    """Configurações para produção"""
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    workers: int = 2
    ml_history_size: int = 2000  # Maior para produção
    
    # Segurança reforçada
    cors_origins: List[str] = []  # Definir URLs específicas
    rate_limit_per_minute: int = 30  # Mais restritivo


# ===============================
# FACTORY DE CONFIGURAÇÕES
# ===============================

def get_settings() -> Settings:
    """Factory para obter configurações baseadas no ambiente"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "staging":
        # Pode criar StagingSettings se necessário
        return Settings(environment="staging")
    else:
        return DevelopmentSettings()


# Instância global das configurações
settings = get_settings()


# ===============================
# CONFIGURAÇÕES DE INDICADORES TÉCNICOS
# ===============================

TECHNICAL_INDICATORS = {
    'momentum': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'stoch_k': 14,
        'stoch_d': 3
    },
    'trend': {
        'sma_periods': [10, 20, 50],
        'ema_periods': [12, 26, 50],
        'bb_period': 20,
        'bb_std': 2
    },
    'volatility': {
        'atr_period': 14,
        'bb_period': 20,
        'keltner_period': 20
    },
    'volume': {
        'volume_sma': 10,
        'volume_ema': 21
    }
}


# ===============================
# CONFIGURAÇÕES DE PADRÕES DE VELAS
# ===============================

CANDLESTICK_PATTERNS = {
    'doji_threshold': 0.1,
    'hammer_ratio': 3.0,
    'engulfing_threshold': 0.05,
    'star_gap_threshold': 0.1
}


# ===============================
# CONFIGURAÇÕES DE RISK MANAGEMENT
# ===============================

RISK_MANAGEMENT = {
    'max_daily_loss_pct': 10.0,  # 10% do saldo
    'max_consecutive_losses': 5,
    'martingale_max_level': 8,
    'cooling_period_seconds': 15,
    'analysis_wait_seconds': 10
}


# ===============================
# LOGGING CONFIGURADO
# ===============================

def setup_logging():
    """Configura logging do sistema"""
    log_config = settings.get_log_config()
    
    # Criar diretório de logs se não existir
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    
    logging.basicConfig(**log_config)
    
    # Logger específico para trading
    trading_logger = logging.getLogger('trading')
    trading_logger.setLevel(log_config['level'])
    
    # Logger específico para ML
    ml_logger = logging.getLogger('ml')
    ml_logger.setLevel(log_config['level'])
    
    return logging.getLogger(__name__)


# Configurar logging na importação
logger = setup_logging()
logger.info(f"🚀 Trading Bot ML configurado para ambiente: {settings.environment}")
logger.info(f"🤖 Configurações ML: {len(settings.supported_symbols)} símbolos suportados")
logger.info(f"📊 Confidence mínima: {settings.ml_min_confidence}%")