"""
Demo Integrado - Demostración completa del sistema de características avanzadas
Este script muestra todas las características funcionando en conjunto de manera coordinada
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Importar el orquestador y configuraciones
try:
    from . import (
        AdvancedConfig,
        ProcessingMode,
        create_comprehensive_config,
        create_orchestrator,
    )
    from .confidence_calibration import CalibrationMethod
    from .dynamic_chunking import ContentType
    from .multi_vector_retrieval import VectorType
    from .project_knowledge import ProjectKnowledgeManager
    from .query_expansion import QueryType
except ImportError:
    # Para ejecución directa
    import sys

    sys.path.append(os.path.dirname(__file__))
    from __init__ import (
        AdvancedConfig,
        ProcessingMode,
        create_comprehensive_config,
        create_orchestrator,
    )
    from confidence_calibration import CalibrationMethod
    from dynamic_chunking import ContentType
    from multi_vector_retrieval import VectorType
    from project_knowledge import ProjectKnowledgeManager
    from query_expansion import QueryType


class IntegratedDemo:
    """Clase principal para la demostración integrada"""

    def __init__(self):
        self.orchestrator = None
        self.knowledge_manager = ProjectKnowledgeManager(
            base_dir=Path(__file__).resolve().parents[2]
        )
        self.demo_data = self._create_demo_data()

    def _create_demo_data(self) -> Dict[str, Any]:
        """Prepara documentos y contexto interactivo para la demostración."""

        documents = self.knowledge_manager.load_documents()
        if not documents:
            raise FileNotFoundError(
                "No se pudieron cargar documentos de conocimiento. Verifica feature.md."
            )

        project_context = self.knowledge_manager.gather_project_context()
        queries = self.knowledge_manager.collect_user_queries(self._default_queries())
        context_payload = self.knowledge_manager.build_context_payload(project_context)

        return {
            "documents": documents,
            "queries": queries,
            "context": context_payload,
        }

    def _default_queries(self) -> List[str]:
        return [
            "Resumen ejecutivo del feature.md",
            "Listado de reglas críticas del proyecto",
            "Pasos obligatorios para habilitar el orquestador avanzado",
        ]

    async def run_comprehensive_demo(self):
        """Ejecuta la demostración completa del sistema integrado"""

        print("🚀 DEMO INTEGRADO - MCP HUB ENHANCED")
        print("=" * 80)
        print(
            "Demostración de todas las características avanzadas trabajando en conjunto"
        )
        print()

        # 1. Inicialización del sistema
        print("📋 FASE 1: Inicialización del Sistema")
        print("-" * 40)

        print("Configurando sistema con modo COMPREHENSIVE...")
        config = create_comprehensive_config()
        self.orchestrator = create_orchestrator("comprehensive")

        # Mostrar configuración
        print(f"   ✅ Modo de procesamiento: {config.processing_mode.value}")
        enabled_count = sum(
            1
            for v in [
                config.enable_dynamic_chunking,
                config.enable_mvr,
                config.enable_virtual_chunks,
                config.enable_query_expansion,
                config.enable_confidence_calibration,
            ]
            if v
        )
        print(f"   ✅ Características habilitadas: {enabled_count}/5")
        print(f"   ✅ Operaciones concurrentes: {config.max_concurrent_operations}")
        print(f"   ✅ Resultados máximos: {config.max_search_results}")

        # Estado inicial del sistema
        initial_status = self.orchestrator.get_system_status()
        print("\n📊 Estado inicial de características:")
        for feature, status in initial_status["feature_status"].items():
            emoji = "✅" if status == "enabled" else "❌" if status == "error" else "⏳"
            print(f"   {emoji} {feature.replace('_', ' ').title()}: {status}")

        # 2. Preparación de datos
        print(f"\n📚 FASE 2: Preparación de Datos")
        print("-" * 40)

        print("Cargando documentos del proyecto...")
        documents = self.demo_data["documents"]

        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc['id']}")
            print(f"      Tipo: {doc['type']} | Dominio: {doc['domain']}")
            print(
                f"      Tamaño: {len(doc['content'])} chars | Complejidad: {doc['complexity']}"
            )
            print(f"      Fuente: {doc['path']}")

        project_context = self.demo_data.get("context", {}).get("project_context", {})
        if project_context:
            print("\n🧭 Contexto recopilado del usuario:")
            for key, value in project_context.items():
                if key == "summary":
                    continue
                print(f"   • {key}: {value}")

        # 3. Añadir documentos al sistema MVR (si está habilitado)
        if self.orchestrator.mvr_system:
            print(f"\n🔧 Indexando documentos en sistema MVR...")
            for doc in documents:
                success = self.orchestrator.mvr_system.add_document(
                    doc_id=doc["id"],
                    content=doc["content"],
                    metadata={
                        "type": doc["type"],
                        "domain": doc["domain"],
                        "path": doc["path"],
                        "complexity": doc["complexity"],
                    },
                )
                emoji = "✅" if success else "❌"
                print(f"   {emoji} {doc['id']}")

        # 4. Procesamiento de queries
        print(f"\n🔍 FASE 3: Procesamiento de Queries")
        print("-" * 40)

        queries = self.demo_data["queries"][:3]  # Primeras 3 queries para la demo

        print("\nℹ️  Consultas seleccionadas:")
        for idx, query in enumerate(queries, 1):
            print(f"   {idx}. {query}")

        for i, query in enumerate(queries, 1):
            print(f"\n>>> Query {i}: '{query}'")
            print("   " + "─" * 50)

            start_time = time.time()

            # Procesamiento avanzado
            base_context = dict(self.demo_data.get("context", {}))
            base_context.update({"demo_query": i, "timestamp": time.time()})

            result = await self.orchestrator.process_advanced(
                query=query,
                documents=documents,
                context=base_context,
            )

            processing_time = time.time() - start_time

            # Mostrar resultados
            print(f"   ⏱️  Tiempo de procesamiento: {processing_time:.3f}s")
            print(
                f"   🔧 Características usadas: {len([s for s in result.feature_status.values() if s.value == 'enabled'])}"
            )

            # Query Expansion
            if result.expanded_queries:
                print(f"   🔄 Queries expandidas ({len(result.expanded_queries)}):")
                for j, exp_query in enumerate(result.expanded_queries[:3], 1):
                    print(f"      {j}. {exp_query}")

            # Dynamic Chunking
            if result.chunks:
                print(f"   📄 Chunks generados: {len(result.chunks)}")
                chunk_types = {}
                for chunk in result.chunks:
                    if hasattr(chunk.metadata, "chunk_type"):
                        chunk_type = chunk.metadata.chunk_type.value
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                for chunk_type, count in chunk_types.items():
                    print(f"      - {chunk_type}: {count} chunks")

            # Search Results
            if result.search_results:
                print(f"   🎯 Resultados de búsqueda ({len(result.search_results)}):")
                for j, search_result in enumerate(result.search_results[:3], 1):
                    print(
                        f"      {j}. {search_result.doc_id} (Score: {search_result.score:.4f})"
                    )
                    if hasattr(search_result, "vector_scores"):
                        vector_info = ", ".join(
                            [
                                f"{k.value}: {v:.3f}"
                                for k, v in list(search_result.vector_scores.items())[
                                    :2
                                ]
                            ]
                        )
                        print(f"         Vectores: {vector_info}")

            # Confidence Calibration
            if result.confidence_scores:
                print(f"   🎯 Calibración de confianza:")
                for j, conf_score in enumerate(result.confidence_scores[:3], 1):
                    print(
                        f"      {j}. Raw: {conf_score.raw_score:.3f} → Calibrated: {conf_score.calibrated_score:.3f}"
                    )
                    print(
                        f"         Nivel: {conf_score.confidence_level.value} | Incertidumbre: {conf_score.uncertainty_estimate:.3f}"
                    )

        # 5. Simulación de feedback
        print(f"\n🔄 FASE 4: Simulación de Feedback")
        print("-" * 40)

        print("Añadiendo feedback simulado para mejorar el sistema...")

        # Generar feedback sintético
        np.random.seed(42)
        feedback_data: List[Dict[str, Any]] = []

        for i, query in enumerate(queries):
            # Simular múltiples interacciones por query
            for j in range(5):
                relevance_score = np.random.beta(2, 1)  # Sesgado hacia scores altos
                was_helpful = relevance_score > 0.6  # Threshold para utilidad

                feedback_data.append(
                    {
                        "query": query,
                        "result_doc_id": f"doc_{i}_{j}",
                        "relevance_score": relevance_score,
                        "was_helpful": was_helpful,
                    }
                )

                # Añadir feedback al sistema
                feedback_context = {
                    **self.demo_data.get("context", {}),
                    "simulation": True,
                    "query_idx": i,
                }
                self.orchestrator.add_feedback(
                    query=query,
                    result_doc_id=f"doc_{i}_{j}",
                    relevance_score=relevance_score,
                    was_helpful=was_helpful,
                    context=feedback_context,
                )

        print(f"   ✅ Añadido feedback para {len(feedback_data)} interacciones")
        helpful_ratio = np.mean([f["was_helpful"] for f in feedback_data]) if feedback_data else 0.0
        print(f"   📊 Tasa de utilidad promedio: {helpful_ratio:.1%}")

        # 6. Análisis de rendimiento
        print(f"\n📈 FASE 5: Análisis de Rendimiento")
        print("-" * 40)

        final_status = self.orchestrator.get_system_status()

        print(f"\n📊 Estadísticas de operación:")
        stats = final_status.get("statistics", {})
        total_operations = stats.get("total_operations", stats.get("operations", 0))
        avg_time_ms = stats.get("avg_processing_time_ms", 0.0)

        print(f"   • Total de operaciones: {total_operations}")
        print(f"   • Tiempo promedio: {avg_time_ms:.1f}ms")

        feature_usage = stats.get("feature_usage", {})
        if feature_usage:
            print(f"   • Uso por característica:")
            for feature, count in feature_usage.items():
                print(f"     - {feature.replace('_', ' ').title()}: {count} veces")

        error_counts = stats.get("error_counts", {})
        if error_counts:
            print(f"   • Errores detectados:")
            for feature, errors in error_counts.items():
                print(f"     - {feature}: {errors} errores")

        # 7. Métricas de calibración (si está disponible)
        cc_status = final_status.get("confidence_calibration_system")
        if self.orchestrator.confidence_calibrator and cc_status:
            print(f"\n🎯 Métricas de calibración:")

            metrics = cc_status.get("recent_metrics", {})
            if metrics:
                print(f"   • Expected Calibration Error: {metrics.get('ece', 0):.4f}")
                print(f"   • Brier Score: {metrics.get('brier_score', 0):.4f}")
                print(f"   • Reliability Score: {metrics.get('reliability', 0):.4f}")

            print(f"   • Muestras de feedback: {cc_status.get('feedback_samples', 0)}")
            print(f"   • Método actual: {cc_status.get('current_best_method', 'N/A')}")

        # 8. Optimización automática
        print(f"\n⚡ FASE 6: Optimización Automática")
        print("-" * 40)

        optimization_report = self.orchestrator.optimize_configuration()

        print("📊 Análisis de rendimiento actual:")
        perf = optimization_report.get("current_performance", {})
        avg_processing_time = perf.get(
            "avg_processing_time",
            perf.get("avg_processing_time_ms", 0.0) / 1000.0,
        )
        perf_total_operations = perf.get(
            "total_operations",
            perf.get("operations", total_operations),
        )

        print(f"   • Tiempo promedio: {avg_processing_time:.3f}s")
        print(f"   • Operaciones totales: {perf_total_operations}")

        recommendations = optimization_report.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recomendaciones de optimización:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        auto_applied = optimization_report.get("auto_applied", [])
        if auto_applied:
            print(f"\n🔄 Optimizaciones aplicadas automáticamente:")
            for i, opt in enumerate(auto_applied, 1):
                print(f"   {i}. {opt}")

        # 9. Demostración de características específicas
        print(f"\n🔬 FASE 7: Demostración de Características Específicas")
        print("-" * 40)

        await self._demonstrate_specific_features()

        # 10. Resumen final
        print(f"\n🎉 RESUMEN FINAL")
        print("-" * 40)

        final_stats = self.orchestrator.get_system_status()

        print("✅ Demostración completada exitosamente!")
        print(f"\n📋 Características demostradas:")

        demos_completed = [
            "✅ Dynamic Chunking Adaptativo",
            "✅ Multi-Vector Retrieval (MVR)",
            "✅ Query Expansion Automática",
            "✅ Confidence Calibration Dinámica",
            "✅ Sistema Integrado de Orquestación",
            "✅ Feedback Loop y Optimización",
            "✅ Procesamiento Paralelo",
            "✅ Métricas y Monitoreo en Tiempo Real",
        ]

        for demo in demos_completed:
            print(f"   {demo}")

        print(f"\n📊 Estadísticas finales del sistema:")
        print(f"   • Queries procesadas: {len(queries)}")
        print(f"   • Documentos indexados: {len(documents)}")
        print(f"   • Feedback recibido: {len(feedback_data)} interacciones")
        print(
            f"   • Características activas: {len(final_stats['config']['enabled_features'])}"
        )
        print(f"   • Tiempo total de demo: {time.time() - start_time:.1f}s")

        print(f"\n💡 Próximos pasos sugeridos:")
        print("   1. Integrar con fuentes de datos reales")
        print("   2. Ajustar configuración según casos de uso específicos")
        print("   3. Implementar monitoreo continuo en producción")
        print("   4. Configurar pipelines de reentrenamiento automático")

    async def _demonstrate_specific_features(self):
        """Demuestra características específicas en detalle"""

        print("🔧 Demostraciones específicas:")

        # 1. Dynamic Chunking con diferentes tipos de contenido
        if self.orchestrator.chunking_system:
            print("\n   📄 Dynamic Chunking:")

            test_content = """# Título de prueba

Este es un párrafo de ejemplo con contenido variado.

## Subsección con código

```python
def example_function():
    return "Hello, World!"
```

Y más texto después del código."""

            chunks = self.orchestrator.chunking_system.adaptive_chunking(
                text=test_content, file_path="test.md"
            )

            print(f"      ✅ {len(chunks)} chunks generados")
            for i, chunk in enumerate(chunks, 1):
                print(
                    f"         {i}. Tipo: {chunk.metadata.chunk_type.value}, Tamaño: {chunk.metadata.size}"
                )

        # 2. Query Expansion con diferentes tipos
        if self.orchestrator.query_expander:
            print("\n   🔄 Query Expansion:")

            test_queries = [
                "¿Cómo funciona el algoritmo?",
                "Mejores prácticas de programación",
                "Diferencias entre modelos",
            ]

            for query in test_queries:
                expansion = self.orchestrator.query_expander.expand_query(
                    query, max_expansions=3
                )
                print(f"      '{query}' →")
                print(f"         Tipo: {expansion.query_type.value}")
                print(f"         Expansiones: {len(expansion.expanded_terms)}")

        # 3. Confidence Calibration en acción
        if self.orchestrator.confidence_calibrator:
            print("\n   🎯 Confidence Calibration:")

            test_scores = [0.3, 0.6, 0.9]
            for score in test_scores:
                calibrated = (
                    self.orchestrator.confidence_calibrator.calibrate_confidence(score)
                )
                print(
                    f"      {score:.1f} → {calibrated.calibrated_score:.3f} ({calibrated.confidence_level.value})"
                )

        print("      ✅ Demostraciones específicas completadas")


def create_demo_config() -> AdvancedConfig:
    """Crea configuración optimizada para la demo"""
    return AdvancedConfig(
        processing_mode=ProcessingMode.COMPREHENSIVE,
        max_concurrent_operations=4,
        cache_size_mb=50,
        enable_dynamic_chunking=True,
        enable_mvr=True,
        enable_virtual_chunks=False,  # Deshabilitado para simplicidad de demo
        enable_query_expansion=True,
        enable_confidence_calibration=True,
        max_search_results=8,
        max_expansions=6,
    )


async def run_demo():
    """Función principal para ejecutar la demo"""
    demo = IntegratedDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    """
    Ejecutar la demo integrada completa

    Este script demuestra todas las características avanzadas del MCP Hub Enhanced:

    1. Dynamic Chunking Adaptativo
    2. Multi-Vector Retrieval (MVR)
    3. Query Expansion Automática
    4. Confidence Calibration Dinámica
    5. Sistema Integrado de Orquestación

    Uso:
        python integrated_demo.py

    O desde el directorio padre:
        python -m core.advanced_features.integrated_demo
    """

    print("🚀 Iniciando Demo Integrado del MCP Hub Enhanced...")
    print("   Preparando sistema avanzado con todas las características...")
    print()

    try:
        # Configurar logging
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Ejecutar demo
        asyncio.run(run_demo())

        print("\n🎉 Demo completado exitosamente!")
        print("   Todas las características avanzadas han sido demostradas.")
        print("   El sistema está listo para integración en producción.")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrumpido por el usuario")

    except Exception as e:
        print(f"\n❌ Error durante la demo: {e}")
        print("   Revisa los logs para más detalles.")
        raise

    finally:
        print("\n📋 Demo finalizado")
        print("=" * 80)
