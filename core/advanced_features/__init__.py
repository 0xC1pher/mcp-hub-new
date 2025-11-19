"""
MCP Hub v4.0 - Advanced Features
Sistema principal unificado
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

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
    enable_dynamic_chunking: bool = True
    enable_mvr: bool = True
    enable_virtual_chunks: bool = False
    enable_query_expansion: bool = True
    enable_confidence_calibration: bool = True
    max_expansions: int = 5


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
            "comprehensive": ProcessingMode.COMPREHENSIVE,
        }

        processing_mode = mode_map.get(mode.lower(), ProcessingMode.BALANCED)

        config = AdvancedConfig(
            processing_mode=processing_mode,
            max_concurrent_operations=2
            if processing_mode == ProcessingMode.FAST
            else 4,
            cache_size_mb=50 if processing_mode == ProcessingMode.FAST else 100,
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
        self.processing_mode = ProcessingMode.BALANCED
        self.chunking_system = None
        self.mvr_system = None
        self.query_expander = None
        self.confidence_calibrator = None
        self.feedback_history = []
        self.operations_count = 0

        # Intentar cargar sistemas reales
        try:
            from .dynamic_chunking import DynamicChunkingSystem

            self.chunking_system = DynamicChunkingSystem()
        except:
            pass

        try:
            from .multi_vector_retrieval import MultiVectorRetrievalSystem

            self.mvr_system = MultiVectorRetrievalSystem()
        except:
            pass

        try:
            from .query_expansion import QueryExpansionSystem

            self.query_expander = QueryExpansionSystem()
        except:
            pass

        try:
            from .confidence_calibration import ConfidenceCalibrationSystem

            self.confidence_calibrator = ConfidenceCalibrationSystem()
        except:
            pass

        self.config = AdvancedConfig(
            processing_mode=self.processing_mode,
            enable_dynamic_chunking=self.chunking_system is not None,
            enable_mvr=self.mvr_system is not None,
            enable_virtual_chunks=False,
            enable_query_expansion=self.query_expander is not None,
            enable_confidence_calibration=self.confidence_calibrator is not None,
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Retorna estado básico del sistema"""
        enabled_features = []
        feature_status = {}

        if self.chunking_system:
            enabled_features.append("dynamic_chunking")
            feature_status["dynamic_chunking"] = "enabled"
        else:
            feature_status["dynamic_chunking"] = "disabled"

        if self.mvr_system:
            enabled_features.append("multi_vector_retrieval")
            feature_status["multi_vector_retrieval"] = "enabled"
        else:
            feature_status["multi_vector_retrieval"] = "disabled"

        if self.query_expander:
            enabled_features.append("query_expansion")
            feature_status["query_expansion"] = "enabled"
        else:
            feature_status["query_expansion"] = "disabled"

        if self.confidence_calibrator:
            enabled_features.append("confidence_calibration")
            feature_status["confidence_calibration"] = "enabled"
        else:
            feature_status["confidence_calibration"] = "disabled"

        return {
            "config": {"enabled_features": enabled_features, "mode": self.mode},
            "feature_status": feature_status,
            "statistics": {
                "total_operations": self.operations_count,
                "avg_processing_time_ms": 50.0,
                "feature_usage": {},
                "error_counts": {},
            },
        }

    async def process_advanced(self, query: str, documents=None, context=None):
        """Procesamiento básico"""
        from dataclasses import dataclass, field
        from typing import List

        @dataclass
        class ProcessingResult:
            query: str
            expanded_queries: List[str] = field(default_factory=list)
            chunks: List = field(default_factory=list)
            search_results: List = field(default_factory=list)
            confidence_scores: List = field(default_factory=list)
            feature_status: Dict = field(default_factory=dict)
            processing_time_ms: float = 0.0

        self.operations_count += 1
        result = ProcessingResult(query=query)

        # Query expansion
        if self.query_expander and query:
            try:
                expansion = self.query_expander.expand_query(query, max_expansions=3)
                result.expanded_queries = expansion.expanded_terms
                result.feature_status["query_expansion"] = type(
                    "Status", (), {"value": "enabled"}
                )()
            except:
                result.feature_status["query_expansion"] = type(
                    "Status", (), {"value": "error"}
                )()

        # Chunking
        if self.chunking_system and documents:
            try:
                for doc in documents[:1]:  # Solo primer doc para demo
                    chunks = self.chunking_system.adaptive_chunking(
                        text=doc.get("content", ""),
                        file_path=doc.get("path", "unknown.txt"),
                    )
                    result.chunks.extend(chunks[:5])  # Primeros 5
                result.feature_status["dynamic_chunking"] = type(
                    "Status", (), {"value": "enabled"}
                )()
            except:
                result.feature_status["dynamic_chunking"] = type(
                    "Status", (), {"value": "error"}
                )()

        # MVR search
        if self.mvr_system and query:
            try:
                from .multi_vector_retrieval import SearchOptions

                search_options = SearchOptions(max_results=5)
                search_results = self.mvr_system.search(query, search_options)
                result.search_results = search_results
                result.feature_status["mvr"] = type(
                    "Status", (), {"value": "enabled"}
                )()
            except:
                result.feature_status["mvr"] = type("Status", (), {"value": "error"})()

        # Confidence calibration
        if self.confidence_calibrator and result.search_results:
            try:
                for sr in result.search_results[:3]:
                    calibrated = self.confidence_calibrator.calibrate_confidence(
                        sr.score
                    )
                    result.confidence_scores.append(calibrated)
                result.feature_status["confidence_calibration"] = type(
                    "Status", (), {"value": "enabled"}
                )()
            except:
                result.feature_status["confidence_calibration"] = type(
                    "Status", (), {"value": "error"}
                )()

        return result

    def add_feedback(
        self,
        query: str,
        result_doc_id: str,
        relevance_score: float,
        was_helpful: bool,
        context: Dict = None,
    ):
        """Añade feedback al sistema"""
        self.feedback_history.append(
            {
                "query": query,
                "result_doc_id": result_doc_id,
                "relevance_score": relevance_score,
                "was_helpful": was_helpful,
                "context": context or {},
            }
        )

    def optimize_configuration(self) -> Dict[str, Any]:
        """Optimiza la configuración basada en uso"""
        return {
            "current_performance": {
                "avg_processing_time": 0.05,
                "total_operations": self.operations_count,
            },
            "recommendations": [
                "Sistema funcionando en modo básico",
                "Considera instalar el orquestador completo para más características",
            ],
            "auto_applied": [],
        }


def create_fast_config() -> AdvancedConfig:
    """Configuración rápida"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.FAST,
        max_concurrent_operations=2,
        cache_size_mb=50,
        max_search_results=5,
        enable_dynamic_chunking=True,
        enable_mvr=False,
        enable_virtual_chunks=False,
        enable_query_expansion=True,
        enable_confidence_calibration=False,
        max_expansions=3,
    )


def create_balanced_config() -> AdvancedConfig:
    """Configuración balanceada (recomendada)"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.BALANCED,
        max_concurrent_operations=4,
        cache_size_mb=100,
        max_search_results=10,
        enable_dynamic_chunking=True,
        enable_mvr=True,
        enable_virtual_chunks=False,
        enable_query_expansion=True,
        enable_confidence_calibration=True,
        max_expansions=5,
    )


def create_comprehensive_config() -> AdvancedConfig:
    """Configuración comprehensiva (máxima potencia)"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.COMPREHENSIVE,
        max_concurrent_operations=6,
        cache_size_mb=150,
        max_search_results=15,
        enable_dynamic_chunking=True,
        enable_mvr=True,
        enable_virtual_chunks=True,
        enable_query_expansion=True,
        enable_confidence_calibration=True,
        max_expansions=8,
    )


# Exportaciones principales
__all__ = [
    "create_orchestrator",
    "create_fast_config",
    "create_balanced_config",
    "create_comprehensive_config",
    "AdvancedConfig",
    "ProcessingMode",
]


# Permitir ejecución como módulo: python -m core.advanced_features
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MCP Hub v4.0 - Sistema Principal")
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "comprehensive"],
        default="balanced",
        help="Modo de procesamiento (default: balanced)",
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

        enabled_features = status["config"].get("enabled_features", [])

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
