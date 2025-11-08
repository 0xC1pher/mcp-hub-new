#!/usr/bin/env python3
"""
Debug Query Tool - Herramienta de debug personalizada para MCP Hub Enhanced
Permite testing rápido de queries específicas con configuración personalizada
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

# Configurar Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from core.advanced_features import (
        create_orchestrator,
        AdvancedConfig,
        ProcessingMode,
        AdvancedFeaturesOrchestrator
    )
    from core.advanced_features.dynamic_chunking import ChunkType
    from core.advanced_features.multi_vector_retrieval import VectorType, FusionStrategy
    from core.advanced_features.query_expansion import QueryType, ExpansionStrategy
    from core.advanced_features.confidence_calibration import CalibrationMethod
except ImportError as e:
    print(f"❌ Error importing MCP modules: {e}")
    print("Ensure you're running from the project root directory")
    sys.exit(1)


class DebugQueryTool:
    """Herramienta de debug interactiva para queries MCP"""

    def __init__(self, mode: str = "balanced", verbose: bool = False):
        self.mode = mode
        self.verbose = verbose
        self.orchestrator = None
        self.debug_history = []

    def setup_orchestrator(self) -> bool:
        """Configura el orquestador con el modo especificado"""
        try:
            print(f"🔧 Configurando orquestador en modo: {self.mode}")
            self.orchestrator = create_orchestrator(self.mode)

            # Verificar estado
            status = self.orchestrator.get_system_status()
            enabled_features = status['config']['enabled_features']

            print(f"✅ Sistema configurado con {len(enabled_features)} características:")
            for feature in enabled_features:
                print(f"   • {feature.replace('_', ' ').title()}")

            return True

        except Exception as e:
            print(f"❌ Error configurando sistema: {e}")
            return False

    async def debug_query(self,
                         query: str,
                         documents: List[Dict[str, Any]] = None,
                         context: Dict[str, Any] = None,
                         breakdown: bool = True) -> Dict[str, Any]:
        """Ejecuta debug detallado de una query"""

        print(f"\n🔍 DEBUGGING QUERY: '{query}'")
        print("=" * 80)

        start_time = time.time()

        # Configurar contexto por defecto
        if context is None:
            context = {
                "debug_mode": True,
                "timestamp": start_time,
                "source": "debug_tool"
            }

        # Documentos de ejemplo si no se proporcionan
        if documents is None:
            documents = self._get_sample_documents()

        debug_result = {
            "query": query,
            "mode": self.mode,
            "start_time": start_time,
            "documents_count": len(documents),
            "context": context,
            "results": {},
            "timing": {},
            "errors": []
        }

        try:
            # Procesar query con características avanzadas
            if breakdown:
                print("📊 Ejecutando con breakdown detallado...")
                result = await self._process_with_breakdown(query, documents, context)
            else:
                print("⚡ Ejecutando procesamiento directo...")
                result = await self.orchestrator.process_advanced(query, documents, context)

            # Almacenar resultados
            debug_result["results"] = {
                "expanded_queries": result.expanded_queries,
                "chunks_count": len(result.chunks),
                "search_results_count": len(result.search_results),
                "confidence_scores_count": len(result.confidence_scores),
                "processing_time": result.processing_time,
                "feature_status": {k: v.value for k, v in result.feature_status.items()}
            }

            # Mostrar resultados
            self._display_results(result, debug_result)

        except Exception as e:
            error_msg = f"Error procesando query: {e}"
            print(f"❌ {error_msg}")
            debug_result["errors"].append(error_msg)

        # Timing final
        debug_result["total_time"] = time.time() - start_time

        # Guardar en historial
        self.debug_history.append(debug_result)

        return debug_result

    async def _process_with_breakdown(self, query, documents, context):
        """Procesa query con breakdown paso a paso"""

        # 1. Query Expansion
        if self.orchestrator.query_expander:
            print("\n🔄 PASO 1: Query Expansion")
            expansion_start = time.time()

            expansion = self.orchestrator.query_expander.expand_query(query, max_expansions=5)
            expansion_time = time.time() - expansion_start

            print(f"   ⏱️  Tiempo: {expansion_time:.3f}s")
            print(f"   📝 Tipo detectado: {expansion.query_type.value}")
            print(f"   🔍 Queries expandidas: {len(expansion.expanded_queries)}")

            for i, exp_query in enumerate(expansion.expanded_queries[:3], 1):
                print(f"      {i}. {exp_query}")

        # 2. Dynamic Chunking
        if self.orchestrator.chunking_system and documents:
            print("\n📄 PASO 2: Dynamic Chunking")
            chunking_start = time.time()

            all_chunks = []
            for doc in documents[:3]:  # Limitar para debug
                content = doc.get('content', '')
                if content:
                    chunks = self.orchestrator.chunking_system.adaptive_chunking(
                        text=content[:2000],  # Limitar contenido para debug
                        file_path=doc.get('path', 'debug.txt')
                    )
                    all_chunks.extend(chunks)

            chunking_time = time.time() - chunking_start
            print(f"   ⏱️  Tiempo: {chunking_time:.3f}s")
            print(f"   📊 Chunks generados: {len(all_chunks)}")

            # Mostrar estadísticas de chunks
            if all_chunks:
                chunk_types = {}
                for chunk in all_chunks:
                    ct = chunk.metadata.chunk_type.value
                    chunk_types[ct] = chunk_types.get(ct, 0) + 1

                for chunk_type, count in chunk_types.items():
                    print(f"      • {chunk_type}: {count}")

        # 3. Procesamiento completo
        print("\n🚀 PASO 3: Procesamiento Integrado")
        integration_start = time.time()

        result = await self.orchestrator.process_advanced(query, documents, context)

        integration_time = time.time() - integration_start
        print(f"   ⏱️  Tiempo total: {integration_time:.3f}s")

        return result

    def _display_results(self, result, debug_result):
        """Muestra resultados detallados"""

        print(f"\n📊 RESULTADOS DEL DEBUG")
        print("-" * 50)

        # Timing
        print(f"⏱️  Tiempo de procesamiento: {result.processing_time:.3f}s")
        print(f"⏱️  Tiempo total de debug: {debug_result['total_time']:.3f}s")

        # Features utilizadas
        active_features = [k for k, v in result.feature_status.items() if v.value == 'enabled']
        print(f"🔧 Características activas: {len(active_features)}")

        # Query Expansion
        if result.expanded_queries:
            print(f"\n🔄 Query Expansion ({len(result.expanded_queries)} queries):")
            for i, exp_query in enumerate(result.expanded_queries[:5], 1):
                print(f"   {i}. {exp_query}")

        # Chunking
        if result.chunks:
            print(f"\n📄 Dynamic Chunking ({len(result.chunks)} chunks):")
            sizes = [chunk.metadata.size for chunk in result.chunks]
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            print(f"   📏 Tamaño promedio: {avg_size:.0f} chars")
            print(f"   📏 Rango: {min(sizes) if sizes else 0} - {max(sizes) if sizes else 0}")

        # Search Results
        if result.search_results:
            print(f"\n🎯 Search Results ({len(result.search_results)} resultados):")
            for i, search_result in enumerate(result.search_results[:5], 1):
                print(f"   {i}. {search_result.doc_id} (Score: {search_result.score:.4f})")

        # Confidence Scores
        if result.confidence_scores:
            print(f"\n🎲 Confidence Calibration ({len(result.confidence_scores)} scores):")
            for i, conf_score in enumerate(result.confidence_scores[:3], 1):
                print(f"   {i}. Raw: {conf_score.raw_score:.3f} → "
                      f"Calibrated: {conf_score.calibrated_score:.3f} "
                      f"({conf_score.confidence_level.value})")

        if self.verbose:
            self._display_verbose_info(result)

    def _display_verbose_info(self, result):
        """Muestra información verbose adicional"""
        print(f"\n🔬 INFORMACIÓN VERBOSE")
        print("-" * 30)

        # System status
        system_status = self.orchestrator.get_system_status()
        stats = system_status.get('statistics', {})

        if stats:
            print("📈 Estadísticas del sistema:")
            print(f"   • Operaciones totales: {stats.get('total_operations', 0)}")
            print(f"   • Tiempo promedio: {stats.get('avg_processing_time_ms', 0):.1f}ms")

            if stats.get('feature_usage'):
                print("   • Uso por característica:")
                for feature, count in stats['feature_usage'].items():
                    print(f"     - {feature}: {count}")

    def _get_sample_documents(self) -> List[Dict[str, Any]]:
        """Obtiene documentos de ejemplo para testing"""
        return [
            {
                "id": "sample_ml",
                "content": """Machine Learning es una rama de la inteligencia artificial que permite a las computadoras aprender de datos sin ser programadas explícitamente. Los algoritmos de ML pueden encontrar patrones en datos y hacer predicciones sobre nuevos datos. Existen tres tipos principales: supervisado, no supervisado y por refuerzo.""",
                "path": "ml_intro.md",
                "type": "educational"
            },
            {
                "id": "sample_python",
                "content": """Python es un lenguaje de programación versátil y poderoso. Es especialmente popular en ciencia de datos y machine learning debido a bibliotecas como NumPy, Pandas y Scikit-learn. Su sintaxis clara lo hace ideal para principiantes y expertos.""",
                "path": "python_guide.md",
                "type": "tutorial"
            },
            {
                "id": "sample_code",
                "content": """def train_model(X, y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Ejemplo de uso
X_train, y_train = load_data()
trained_model = train_model(X_train, y_train)""",
                "path": "example.py",
                "type": "code"
            }
        ]

    def save_debug_session(self, filename: str = None) -> str:
        """Guarda la sesión de debug"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"debug_session_{timestamp}.json"

        session_data = {
            "mode": self.mode,
            "timestamp": time.time(),
            "total_queries": len(self.debug_history),
            "history": self.debug_history
        }

        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        print(f"💾 Sesión de debug guardada en: {filename}")
        return filename

    def interactive_mode(self):
        """Modo interactivo para debugging"""
        print("🎮 MODO INTERACTIVO DE DEBUG")
        print("Comandos disponibles:")
        print("  - query <texto>: Procesar query")
        print("  - mode <fast|balanced|comprehensive>: Cambiar modo")
        print("  - status: Ver estado del sistema")
        print("  - history: Ver historial")
        print("  - save: Guardar sesión")
        print("  - exit: Salir")
        print()

        while True:
            try:
                command = input("🔍 Debug> ").strip()

                if command.startswith("query "):
                    query = command[6:].strip()
                    if query:
                        asyncio.run(self.debug_query(query))
                    else:
                        print("❌ Query vacía")

                elif command.startswith("mode "):
                    new_mode = command[5:].strip()
                    if new_mode in ["fast", "balanced", "comprehensive"]:
                        self.mode = new_mode
                        print(f"🔄 Modo cambiado a: {new_mode}")
                        if not self.setup_orchestrator():
                            print("❌ Error reconfigurando sistema")
                    else:
                        print("❌ Modo inválido. Use: fast, balanced, comprehensive")

                elif command == "status":
                    if self.orchestrator:
                        status = self.orchestrator.get_system_status()
                        print("📊 Estado del sistema:")
                        print(json.dumps(status, indent=2))
                    else:
                        print("❌ Sistema no inicializado")

                elif command == "history":
                    print(f"📚 Historial ({len(self.debug_history)} queries):")
                    for i, entry in enumerate(self.debug_history, 1):
                        print(f"  {i}. '{entry['query']}' ({entry.get('total_time', 0):.3f}s)")

                elif command == "save":
                    self.save_debug_session()

                elif command in ["exit", "quit"]:
                    print("👋 Saliendo del modo debug...")
                    break

                elif command == "help":
                    print("📚 Ayuda del modo interactivo:")
                    print("  query <texto> - Procesar una query")
                    print("  mode <modo> - Cambiar modo de procesamiento")
                    print("  status - Ver estado del sistema")
                    print("  history - Ver historial de queries")
                    print("  save - Guardar sesión actual")
                    print("  exit - Salir")

                elif command:
                    print(f"❓ Comando desconocido: '{command}'. Use 'help' para ver comandos disponibles.")

            except KeyboardInterrupt:
                print("\n👋 Saliendo...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="MCP Debug Query Tool")
    parser.add_argument("query", nargs="?", help="Query to debug")
    parser.add_argument("--mode", choices=["fast", "balanced", "comprehensive"],
                       default="balanced", help="Processing mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--breakdown", "-b", action="store_true", default=True,
                       help="Show step-by-step breakdown")
    parser.add_argument("--save", "-s", help="Save results to file")

    args = parser.parse_args()

    # Crear herramienta de debug
    debug_tool = DebugQueryTool(mode=args.mode, verbose=args.verbose)

    # Configurar sistema
    print("🚀 Iniciando MCP Debug Tool...")
    if not debug_tool.setup_orchestrator():
        print("❌ No se pudo inicializar el sistema")
        return 1

    # Modo interactivo
    if args.interactive or not args.query:
        debug_tool.interactive_mode()
        return 0

    # Procesar query específica
    try:
        result = await debug_tool.debug_query(
            query=args.query,
            breakdown=args.breakdown
        )

        # Guardar si se especifica
        if args.save:
            debug_tool.save_debug_session(args.save)

        print("\n✅ Debug completado exitosamente")
        return 0

    except Exception as e:
        print(f"❌ Error durante debug: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Proceso interrumpido")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        sys.exit(1)
