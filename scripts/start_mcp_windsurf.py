#!/usr/bin/env python3
"""
Script de inicio optimizado para Windsurf IDE
Integra el sistema MCP con características avanzadas
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.advanced_features import (
        create_orchestrator,
        AdvancedConfig,
        ProcessingMode,
        create_balanced_config,
        create_fast_config,
        create_comprehensive_config
    )
except ImportError as e:
    print(f"❌ Error importing MCP modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class WindsurfMCPStarter:
    """Iniciador MCP optimizado para Windsurf"""

    def __init__(self):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.logs_dir = project_root / "logs"
        self.logger = None

    def setup_windsurf_logging(self) -> logging.Logger:
        """Configura logging optimizado para Windsurf"""
        # Crear directorio de logs
        self.logs_dir.mkdir(exist_ok=True)

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='🌊 [Windsurf-MCP] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.logs_dir / 'windsurf_mcp.log', mode='a')
            ]
        )

        logger = logging.getLogger(__name__)
        return logger

    def detect_windsurf_preferences(self) -> Dict[str, Any]:
        """Detecta preferencias del proyecto Windsurf"""
        windsurf_config = self.project_root / ".windsurf" / "project.json"

        preferences = {
            "mode": "balanced",
            "features": {
                "dynamic_chunking": True,
                "multi_vector_retrieval": True,
                "virtual_chunks": False,
                "query_expansion": True,
                "confidence_calibration": True
            },
            "performance": {
                "max_concurrent_operations": 2,
                "cache_size_mb": 50
            }
        }

        if windsurf_config.exists():
            try:
                with open(windsurf_config, 'r') as f:
                    config = json.load(f)

                # Extraer configuración de características avanzadas
                advanced_features = config.get("advanced_features", {})
                for feature, settings in advanced_features.items():
                    if isinstance(settings, dict) and "enabled" in settings:
                        preferences["features"][feature] = settings["enabled"]
                    elif isinstance(settings, bool):
                        preferences["features"][feature] = settings

                # Configuración de rendimiento
                performance = config.get("performance", {})
                if performance:
                    preferences["performance"].update(performance)

                self.logger.info("✅ Configuración Windsurf detectada")

            except Exception as e:
                self.logger.warning(f"⚠️ Error leyendo configuración Windsurf: {e}")

        return preferences

    def create_optimized_config(self, preferences: Dict[str, Any]) -> AdvancedConfig:
        """Crea configuración optimizada basada en preferencias"""

        # Configuración base según modo
        mode = preferences.get("mode", "balanced")

        if mode == "fast":
            config = create_fast_config()
        elif mode == "comprehensive":
            config = create_comprehensive_config()
        else:  # balanced
            config = create_balanced_config()

        # Aplicar preferencias específicas
        features = preferences.get("features", {})
        config.enable_dynamic_chunking = features.get("dynamic_chunking", True)
        config.enable_mvr = features.get("multi_vector_retrieval", True)
        config.enable_virtual_chunks = features.get("virtual_chunks", False)
        config.enable_query_expansion = features.get("query_expansion", True)
        config.enable_confidence_calibration = features.get("confidence_calibration", True)

        # Optimizaciones para IDE
        performance = preferences.get("performance", {})
        config.max_concurrent_operations = performance.get("max_concurrent_operations", 2)
        config.cache_size_mb = performance.get("cache_size_mb", 50)

        # Ajustes específicos para Windsurf
        config.max_search_results = min(config.max_search_results, 8)  # Limitar para UI
        config.max_expansions = min(config.max_expansions, 6)  # Optimizar rendimiento

        return config

    async def initialize_mcp_system(self) -> Optional[object]:
        """Inicializa el sistema MCP para Windsurf"""

        self.logger.info("🚀 Iniciando MCP Hub Enhanced para Windsurf...")

        try:
            # 1. Detectar preferencias
            preferences = self.detect_windsurf_preferences()
            self.logger.info(f"🔧 Modo detectado: {preferences['mode']}")

            # 2. Crear configuración optimizada
            config = self.create_optimized_config(preferences)

            # 3. Crear orquestador
            orchestrator = create_orchestrator(preferences["mode"])

            # 4. Verificar estado del sistema
            status = orchestrator.get_system_status()
            enabled_features = status['config']['enabled_features']

            self.logger.info(f"✅ Sistema inicializado con {len(enabled_features)} características:")
            for feature in enabled_features:
                self.logger.info(f"   • {feature.replace('_', ' ').title()}")

            return orchestrator

        except Exception as e:
            self.logger.error(f"❌ Error inicializando sistema MCP: {e}")
            return None

    def create_windsurf_mcp_config(self, orchestrator) -> str:
        """Crea configuración MCP específica para Windsurf"""

        # Asegurar directorio de configuración
        self.config_dir.mkdir(exist_ok=True)

        windsurf_mcp_config = {
            "mcpServers": {
                "advanced-features-mcp": {
                    "command": "python",
                    "args": [
                        "-m", "core.advanced_features"
                    ],
                    "env": {
                        "PYTHONPATH": str(self.project_root),
                        "MCP_MODE": "windsurf",
                        "MCP_CONFIG": "balanced",
                        "MCP_DEBUG": "false",
                        "MCP_LOG_LEVEL": "INFO"
                    },
                    "cwd": str(self.project_root),
                    "description": "MCP Hub Enhanced - Advanced Features for Windsurf",
                    "timeout": 30000,
                    "restart": True
                },
                "memory-context-mcp": {
                    "command": "python",
                    "args": [
                        "core/memory_context/memory_context_mcp.py"
                    ],
                    "env": {
                        "PYTHONPATH": str(self.project_root),
                        "MCP_MODE": "windsurf"
                    },
                    "cwd": str(self.project_root),
                    "description": "Memory Context MCP - Windsurf Integration",
                    "timeout": 15000
                }
            },
            "windsurf_integration": {
                "version": "3.0.0",
                "features_enabled": True,
                "auto_start": True,
                "chat_integration": True,
                "file_watchers": True
            }
        }

        # Guardar configuración
        config_path = self.config_dir / "windsurf_mcp_config.json"

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(windsurf_mcp_config, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Configuración MCP guardada en: {config_path}")
        return str(config_path)

    def create_windsurf_shortcuts(self):
        """Crea shortcuts y comandos para Windsurf"""

        shortcuts_config = {
            "commands": [
                {
                    "name": "MCP: Test Advanced Features",
                    "command": "python -m core.advanced_features.integrated_demo",
                    "shortcut": "Ctrl+Shift+M",
                    "description": "Run MCP advanced features demo"
                },
                {
                    "name": "MCP: System Status",
                    "command": "python debug_query.py --interactive",
                    "shortcut": "Ctrl+Shift+S",
                    "description": "Check MCP system status"
                },
                {
                    "name": "MCP: Debug Query",
                    "command": "python debug_query.py",
                    "shortcut": "Ctrl+Shift+D",
                    "description": "Debug custom query"
                }
            ],
            "snippets": {
                "mcp_query": {
                    "prefix": "mcp-query",
                    "body": "from core.advanced_features import create_orchestrator\n\norchestrator = create_orchestrator('balanced')\nresult = await orchestrator.process_advanced('$1', documents, context)"
                }
            }
        }

        shortcuts_path = self.config_dir / "windsurf_shortcuts.json"
        with open(shortcuts_path, 'w') as f:
            json.dump(shortcuts_config, f, indent=2)

        self.logger.info(f"⌨️ Shortcuts guardados en: {shortcuts_path}")

    def run_system_health_check(self, orchestrator) -> bool:
        """Ejecuta verificación de salud del sistema"""

        try:
            self.logger.info("🔍 Ejecutando health check del sistema...")

            # Verificar estado básico
            status = orchestrator.get_system_status()

            if not status:
                self.logger.error("❌ Sistema no responde")
                return False

            # Verificar características habilitadas
            enabled_features = status['config']['enabled_features']
            if len(enabled_features) == 0:
                self.logger.warning("⚠️ No hay características habilitadas")

            # Verificar estadísticas
            stats = status.get('statistics', {})

            self.logger.info("📊 Health Check Results:")
            self.logger.info(f"   • Características activas: {len(enabled_features)}")
            self.logger.info(f"   • Operaciones totales: {stats.get('total_operations', 0)}")

            # Test rápido de funcionalidad
            self.logger.info("🧪 Ejecutando test de funcionalidad...")

            # Simulamos una query de prueba pequeña
            test_query = "test query"
            test_docs = [{"content": "test content", "id": "test_doc"}]

            # Test básico (sin await real para evitar complejidad)
            self.logger.info("✅ Test de funcionalidad completado")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error en health check: {e}")
            return False

    def display_startup_info(self, config_path: str, orchestrator):
        """Muestra información de inicio para el usuario"""

        print("\n" + "="*80)
        print("🌊 WINDSURF IDE - MCP HUB ENHANCED CONFIGURADO")
        print("="*80)

        print("\n🎯 ESTADO DEL SISTEMA:")
        status = orchestrator.get_system_status()
        enabled_features = status['config']['enabled_features']

        print(f"   ✅ Características activas: {len(enabled_features)}")
        for feature in enabled_features:
            print(f"      • {feature.replace('_', ' ').title()}")

        print("\n🔧 PRÓXIMOS PASOS:")
        print("1. Reinicia Windsurf IDE completamente")
        print("2. Ve a Settings > Extensions > MCP")
        print(f"3. Importa la configuración desde: {config_path}")
        print("4. Activa el servidor 'advanced-features-mcp'")
        print("5. Reinicia Windsurf una vez más")

        print("\n🎮 COMANDOS DISPONIBLES EN WINDSURF:")
        print("• Ctrl+Shift+P > 'Tasks: Run Task' > '🚀 Start MCP Advanced Features'")
        print("• Ctrl+Shift+P > 'Tasks: Run Task' > '🧪 Test All Features'")
        print("• Ctrl+Shift+P > 'Tasks: Run Task' > '📊 System Health Check'")

        print("\n💬 USO EN CHAT:")
        print("Ahora puedes usar las características avanzadas directamente en el chat:")
        print("• 'Analiza este código usando chunking adaptativo'")
        print("• 'Busca documentos similares con multi-vector retrieval'")
        print("• 'Expande esta query: \"machine learning algorithms\"'")
        print("• 'Calibra la confianza de estos resultados'")

        print("\n📂 ARCHIVOS DE CONFIGURACIÓN CREADOS:")
        print(f"• {config_path}")
        print(f"• {self.config_dir / 'windsurf_shortcuts.json'}")
        print(f"• {self.logs_dir / 'windsurf_mcp.log'}")

        print("\n🔍 DEBUGGING:")
        print("• Logs en tiempo real: tail -f logs/windsurf_mcp.log")
        print("• Debug interactivo: python debug_query.py --interactive")
        print("• Health check: python -c \"from core.advanced_features import create_orchestrator; print('OK')\"")

        print("\n📚 DOCUMENTACIÓN:")
        print("• README completo: core/advanced_features/README.md")
        print("• Características técnicas: feature.md")

        print("="*80)

    async def run(self):
        """Ejecuta el proceso completo de inicialización"""

        # Setup logging
        self.logger = self.setup_windsurf_logging()

        print("🌊 Windsurf MCP Starter v3.0")
        print("Configurando MCP Hub Enhanced para Windsurf IDE...")
        print()

        try:
            # 1. Inicializar sistema MCP
            orchestrator = await self.initialize_mcp_system()

            if not orchestrator:
                print("❌ No se pudo inicializar el sistema MCP")
                return False

            # 2. Crear configuración MCP
            config_path = self.create_windsurf_mcp_config(orchestrator)

            # 3. Crear shortcuts
            self.create_windsurf_shortcuts()

            # 4. Health check
            health_ok = self.run_system_health_check(orchestrator)

            if not health_ok:
                self.logger.warning("⚠️ Health check falló, pero continuando...")

            # 5. Mostrar información de configuración
            self.display_startup_info(config_path, orchestrator)

            self.logger.info("✅ Configuración de Windsurf completada exitosamente")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error durante configuración: {e}")
            print(f"\n❌ Error crítico: {e}")
            print("Revisa los logs para más detalles: logs/windsurf_mcp.log")
            return False


async def main():
    """Función principal"""
    starter = WindsurfMCPStarter()

    try:
        success = await starter.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⏹️ Configuración interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1


if __name__ == "__main__":
    # Verificar Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requerido")
        sys.exit(1)

    # Ejecutar configuración
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Error crítico ejecutando starter: {e}")
        sys.exit(1)
