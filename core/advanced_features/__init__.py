"""
MCP Hub v4.0 - Advanced Features
Sistema principal unificado
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

__version__ = "4.0.0"


class ProcessingMode(Enum):
    """Modos de procesamiento disponibles"""
    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"


@dataclass
class AdvancedConfig:
    """Configuración simplificada del sistema"""
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    max_concurrent_operations: int = 4
    cache_size_mb: int = 100
    max_search_results: int = 10


def create_orchestrator(mode: str = "balanced"):
    """
    Crea un orquestador del sistema

    Args:
        mode: Modo de procesamiento ("fast", "balanced", "comprehensive")

    Returns:
        Orquestador configurado
    """
    # Imports dinámicos para evitar errores en __init__
    try:
        from .orchestrator import AdvancedFeaturesOrchestrator

        # Mapear string a enum
        mode_map = {
            "fast": ProcessingMode.FAST,
            "balanced": ProcessingMode.BALANCED,
            "comprehensive": ProcessingMode.COMPREHENSIVE
        }

        processing_mode = mode_map.get(mode.lower(), ProcessingMode.BALANCED)

        config = AdvancedConfig(
            processing_mode=processing_mode,
            max_concurrent_operations=2 if processing_mode == ProcessingMode.FAST else 4,
            cache_size_mb=50 if processing_mode == ProcessingMode.FAST else 100
        )

        return AdvancedFeaturesOrchestrator(config)

    except ImportError as e:
        # Fallback simple si hay errores
        logging.warning(f"Error importando orquestador completo: {e}")
        return SimpleOrchestrator(mode)


class SimpleOrchestrator:
    """Orquestador simple de fallback"""

    def __init__(self, mode: str = "balanced"):
        self.mode = mode
        self.config = {
            'enabled_features': ['basic_features'],
            'mode': mode
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Retorna estado básico del sistema"""
        return {
            'config': {
                'enabled_features': ['basic_system'],
                'mode': self.mode
            },
            'statistics': {
                'total_operations': 0,
                'avg_processing_time_ms': 0
            }
        }

    async def process_advanced(self, query: str, documents=None, context=None):
        """Procesamiento básico"""
        return {
            'query': query,
            'results': [],
            'message': 'Sistema funcionando en modo básico'
        }


def create_fast_config() -> AdvancedConfig:
    """Configuración rápida"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.FAST,
        max_concurrent_operations=2,
        cache_size_mb=50,
        max_search_results=5
    )


def create_balanced_config() -> AdvancedConfig:
    """Configuración balanceada (recomendada)"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.BALANCED,
        max_concurrent_operations=4,
        cache_size_mb=100,
        max_search_results=10
    )


def create_comprehensive_config() -> AdvancedConfig:
    """Configuración completa"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.COMPREHENSIVE,
        max_concurrent_operations=6,
        cache_size_mb=150,
        max_search_results=15
    )


# Exportaciones principales
__all__ = [
    'create_orchestrator',
    'create_fast_config',
    'create_balanced_config',
    'create_comprehensive_config',
    'AdvancedConfig',
    'ProcessingMode'
]


# Permitir ejecución como módulo: python -m core.advanced_features
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MCP Hub v4.0 - Sistema Principal")
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "comprehensive"],
        default="balanced",
        help="Modo de procesamiento (default: balanced)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  MCP HUB v4.0 - Sistema Principal")
    print("=" * 80)
    print()
    print(f"Modo: {args.mode}")
    print()

    try:
        orchestrator = create_orchestrator(args.mode)
        status = orchestrator.get_system_status()

        enabled_features = status['config'].get('enabled_features', [])

        print(f"✅ Sistema inicializado")
        print(f"✅ Características: {len(enabled_features)}")
        print()
        print("Características habilitadas:")
        for feature in enabled_features:
            print(f"  • {feature.replace('_', ' ').title()}")

        print()
        print("=" * 80)
        print("Sistema listo")
        print()
        print("Para uso completo:")
        print("  from core.advanced_features import create_orchestrator")
        print("  orchestrator = create_orchestrator('balanced')")
        print()
        print("Para demo: python core/advanced_features/run_system.py")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Soluciones:")
        print("  1. pip install numpy msgpack zstandard")
        print("  2. python --version (necesita 3.8+)")
        print("  3. Revisa logs/")
        sys.exit(1)
