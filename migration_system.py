#!/usr/bin/env python3
"""
Sistema de migração de banco de dados para ML Trading Bot
"""

import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Optional
import hashlib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Migration:
    """Classe base para migrações"""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.timestamp = datetime.now().isoformat()
    
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Aplica a migração"""
        raise NotImplementedError("Método up() deve ser implementado")
    
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Reverte a migração"""
        raise NotImplementedError("Método down() deve ser implementado")
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        """Valida se a migração foi aplicada corretamente"""
        return True

class MigrationManager:
    """Gerenciador de migrações"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.migrations = []
        self.setup_migrations_table()
    
    def setup_migrations_table(self):
        """Cria tabela de controle de migrações"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT,
                execution_time_ms INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_migration(self, migration: Migration):
        """Registra uma migração"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    def get_applied_migrations(self) -> List[str]:
        """Retorna lista de migrações já aplicadas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version FROM migrations ORDER BY version")
        applied = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return applied
    
    def calculate_checksum(self, migration: Migration) -> str:
        """Calcula checksum da migração"""
        content = f"{migration.version}:{migration.description}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def apply_migration(self, migration: Migration) -> bool:
        """Aplica uma migração específica"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            logger.info(f"Aplicando migração {migration.version}: {migration.description}")
            
            start_time = datetime.now()
            
            # Aplicar migração
            migration.up(cursor)
            
            # Validar migração
            if not migration.validate(cursor):
                raise Exception("Validação da migração falhou")
            
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Registrar migração aplicada
            checksum = self.calculate_checksum(migration)
            cursor.execute('''
                INSERT INTO migrations (version, description, checksum, execution_time_ms)
                VALUES (?, ?, ?, ?)
            ''', (migration.version, migration.description, checksum, execution_time))
            
            conn.commit()
            logger.info(f"✅ Migração {migration.version} aplicada em {execution_time}ms")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro ao aplicar migração {migration.version}: {e}")
            return False
        finally:
            conn.close()
    
    def migrate(self) -> bool:
        """Aplica todas as migrações pendentes"""
        applied_migrations = self.get_applied_migrations()
        pending_migrations = [
            m for m in self.migrations 
            if m.version not in applied_migrations
        ]
        
        if not pending_migrations:
            logger.info("✅ Nenhuma migração pendente")
            return True
        
        logger.info(f"📋 {len(pending_migrations)} migrações pendentes")
        
        success_count = 0
        for migration in pending_migrations:
            if self.apply_migration(migration):
                success_count += 1
            else:
                logger.error(f"❌ Migração falhou: {migration.version}")
                break
        
        logger.info(f"✅ {success_count}/{len(pending_migrations)} migrações aplicadas")
        return success_count == len(pending_migrations)
    
    def rollback(self, target_version: str) -> bool:
        """Reverte migrações até uma versão específica"""
        applied_migrations = self.get_applied_migrations()
        
        # Encontrar migrações para reverter
        migrations_to_rollback = []
        for version in reversed(applied_migrations):
            if version == target_version:
                break
            
            # Encontrar migração correspondente
            migration = next((m for m in self.migrations if m.version == version), None)
            if migration:
                migrations_to_rollback.append(migration)
        
        if not migrations_to_rollback:
            logger.info("✅ Nenhuma migração para reverter")
            return True
        
        logger.info(f"📋 Revertendo {len(migrations_to_rollback)} migrações")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for migration in migrations_to_rollback:
                logger.info(f"Revertendo migração {migration.version}")
                
                # Reverter migração
                migration.down(cursor)
                
                # Remover registro da migração
                cursor.execute("DELETE FROM migrations WHERE version = ?", (migration.version,))
            
            conn.commit()
            logger.info("✅ Rollback concluído")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro no rollback: {e}")
            return False
        finally:
            conn.close()
    
    def status(self) -> Dict:
        """Retorna status das migrações"""
        applied_migrations = self.get_applied_migrations()
        
        return {
            "total_migrations": len(self.migrations),
            "applied_migrations": len(applied_migrations),
            "pending_migrations": len(self.migrations) - len(applied_migrations),
            "last_applied": applied_migrations[-1] if applied_migrations else None,
            "migrations": [
                {
                    "version": m.version,
                    "description": m.description,
                    "applied": m.version in applied_migrations
                }
                for m in self.migrations
            ]
        }

# ===== MIGRAÇÕES ESPECÍFICAS =====

class Migration001InitialSchema(Migration):
    """Migração inicial - cria schema básico"""
    
    def __init__(self):
        super().__init__("001", "Criar schema inicial do banco de dados")
    
    def up(self, cursor: sqlite3.Cursor):
        # Tabela de trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                stake REAL NOT NULL,
                duration TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                outcome TEXT,
                market_context TEXT,
                martingale_level INTEGER DEFAULT 0,
                volatility REAL,
                trend TEXT,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome)')
    
    def down(self, cursor: sqlite3.Cursor):
        cursor.execute('DROP TABLE IF EXISTS trades')
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        return cursor.fetchone() is not None

class Migration002MLMetrics(Migration):
    """Adiciona tabelas de métricas ML"""
    
    def __init__(self):
        super().__init__("002", "Adicionar tabelas de métricas ML")
    
    def up(self, cursor: sqlite3.Cursor):
        # Tabela de métricas ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                training_data_size INTEGER,
                feature_importance TEXT,
                confusion_matrix TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de padrões ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence REAL NOT NULL,
                occurrences INTEGER DEFAULT 1,
                success_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_metrics_model ON ml_metrics(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_patterns_type ON ml_patterns(pattern_type)')
    
    def down(self, cursor: sqlite3.Cursor):
        cursor.execute('DROP TABLE IF EXISTS ml_metrics')
        cursor.execute('DROP TABLE IF EXISTS ml_patterns')

class Migration003TradingMetrics(Migration):
    """Adiciona tabelas de métricas de trading"""
    
    def __init__(self):
        super().__init__("003", "Adicionar tabelas de métricas de trading")
    
    def up(self, cursor: sqlite3.Cursor):
        # Tabela de métricas de trading
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                ml_influenced_trades INTEGER DEFAULT 0,
                ml_accuracy_in_trades REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de performance por símbolo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                ml_accuracy REAL DEFAULT 0,
                volatility_avg REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_metrics_timestamp ON trading_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_performance_symbol ON symbol_performance(symbol)')
    
    def down(self, cursor: sqlite3.Cursor):
        cursor.execute('DROP TABLE IF EXISTS trading_metrics')
        cursor.execute('DROP TABLE IF EXISTS symbol_performance')

class Migration004Alerts(Migration):
    """Sistema de alertas"""
    
    def __init__(self):
        super().__init__("004", "Adicionar sistema de alertas")
    
    def up(self, cursor: sqlite3.Cursor):
        # Tabela de alertas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved)')
    
    def down(self, cursor: sqlite3.Cursor):
        cursor.execute('DROP TABLE IF EXISTS alerts')

class Migration005EnhancedTrades(Migration):
    """Melhorias na tabela de trades"""
    
    def __init__(self):
        super().__init__("005", "Adicionar campos avançados à tabela trades")
    
    def up(self, cursor: sqlite3.Cursor):
        # Verificar se colunas já existem
        cursor.execute("PRAGMA table_info(trades)")
        columns = [column[1] for column in cursor.fetchall()]
        
        new_columns = [
            ('pnl', 'REAL DEFAULT 0'),
            ('confidence_score', 'REAL'),
            ('ml_prediction', 'TEXT'),
            ('risk_score', 'REAL'),
            ('session_id', 'TEXT'),
            ('trade_source', 'TEXT DEFAULT "manual"')
        ]
        
        for column_name, column_def in new_columns:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE trades ADD COLUMN {column_name} {column_def}')
        
        # Novos índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_source ON trades(trade_source)')
    
    def down(self, cursor: sqlite3.Cursor):
        # SQLite não suporta DROP COLUMN, então criamos nova tabela
        cursor.execute('''
            CREATE TABLE trades_backup AS 
            SELECT id, timestamp, symbol, direction, stake, duration, 
                   entry_price, exit_price, outcome, market_context, 
                   martingale_level, volatility, trend, features, created_at
            FROM trades
        ''')
        
        cursor.execute('DROP TABLE trades')
        cursor.execute('ALTER TABLE trades_backup RENAME TO trades')

class Migration006BackupSystem(Migration):
    """Sistema de backup automático"""
    
    def __init__(self):
        super().__init__("006", "Adicionar sistema de backup automático")
    
    def up(self, cursor: sqlite3.Cursor):
        # Tabela de controle de backups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                checksum TEXT,
                status TEXT DEFAULT 'success',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Configurações do sistema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                data_type TEXT DEFAULT 'string',
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Inserir configurações padrão
        default_configs = [
            ('backup_enabled', 'true', 'boolean', 'Backup automático habilitado'),
            ('backup_interval_hours', '24', 'integer', 'Intervalo entre backups em horas'),
            ('max_backup_files', '7', 'integer', 'Número máximo de arquivos de backup'),
            ('auto_cleanup_days', '30', 'integer', 'Dias para limpeza automática'),
            ('ml_retrain_threshold', '50', 'integer', 'Número de trades para retreino automático')
        ]
        
        for key, value, data_type, description in default_configs:
            cursor.execute('''
                INSERT OR IGNORE INTO system_config (key, value, data_type, description)
                VALUES (?, ?, ?, ?)
            ''', (key, value, data_type, description))
    
    def down(self, cursor: sqlite3.Cursor):
        cursor.execute('DROP TABLE IF EXISTS backup_log')
        cursor.execute('DROP TABLE IF EXISTS system_config')

# ===== FUNÇÃO PRINCIPAL =====

def setup_migrations() -> MigrationManager:
    """Configura o gerenciador de migrações com todas as migrações"""
    manager = MigrationManager()
    
    # Registrar todas as migrações
    manager.register_migration(Migration001InitialSchema())
    manager.register_migration(Migration002MLMetrics())
    manager.register_migration(Migration003TradingMetrics())
    manager.register_migration(Migration004Alerts())
    manager.register_migration(Migration005EnhancedTrades())
    manager.register_migration(Migration006BackupSystem())
    
    return manager

def run_migrations(db_path: str = "trading_data.db") -> bool:
    """Executa todas as migrações pendentes"""
    logger.info("🔄 Iniciando migrações do banco de dados...")
    
    manager = setup_migrations()
    manager.db_path = db_path
    
    # Verificar status atual
    status = manager.status()
    logger.info(f"📊 Status: {status['applied_migrations']}/{status['total_migrations']} migrações aplicadas")
    
    if status['pending_migrations'] > 0:
        logger.info(f"⏳ Aplicando {status['pending_migrations']} migrações pendentes...")
        
        if manager.migrate():
            logger.info("✅ Todas as migrações aplicadas com sucesso!")
            return True
        else:
            logger.error("❌ Falha na aplicação das migrações")
            return False
    else:
        logger.info("✅ Banco de dados já está atualizado")
        return True

def main():
    """Função principal para execução direta"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Migração ML Trading Bot')
    parser.add_argument('--db', default='trading_data.db', help='Caminho do banco de dados')
    parser.add_argument('--action', choices=['migrate', 'status', 'rollback'], default='migrate', 
                       help='Ação a executar')
    parser.add_argument('--target', help='Versão alvo para rollback')
    
    args = parser.parse_args()
    
    manager = setup_migrations()
    manager.db_path = args.db
    
    if args.action == 'migrate':
        success = manager.migrate()
        exit(0 if success else 1)
    
    elif args.action == 'status':
        status = manager.status()
        print(f"📊 Status das Migrações:")
        print(f"  Total: {status['total_migrations']}")
        print(f"  Aplicadas: {status['applied_migrations']}")
        print(f"  Pendentes: {status['pending_migrations']}")
        
        if status['last_applied']:
            print(f"  Última aplicada: {status['last_applied']}")
        
        print(f"\n📋 Lista de Migrações:")
        for migration in status['migrations']:
            status_icon = "✅" if migration['applied'] else "⏳"
            print(f"  {status_icon} {migration['version']}: {migration['description']}")
    
    elif args.action == 'rollback':
        if not args.target:
            logger.error("❌ Versão alvo é obrigatória para rollback")
            exit(1)
        
        success = manager.rollback(args.target)
        exit(0 if success else 1)

if __name__ == "__main__":
    main()
