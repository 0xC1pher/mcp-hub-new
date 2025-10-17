#!/usr/bin/env python3
"""
Script de prueba para validar todas las optimizaciones implementadas
Ejecuta pruebas exhaustivas de cada componente optimizado
"""

import json
import time
import requests
import threading
import subprocess
import sys
from pathlib import Path

def log(message, status="INFO"):
    """Logging con colores"""
    colors = {
        "INFO": "\033[34m",  # Azul
        "SUCCESS": "\033[32m",  # Verde
        "WARNING": "\033[33m",  # Amarillo
        "ERROR": "\033[31m",   # Rojo
        "RESET": "\033[0m"
    }
    print(f"{colors.get(status, colors['RESET'])}[{status}] {message}{colors['RESET']}")

def test_server_startup():
    """Prueba el inicio del servidor optimizado"""
    log("🚀 Probando inicio del servidor optimizado...")

    try:
        # Iniciar servidor en background
        server_process = subprocess.Popen([
            sys.executable,
            "servers/context-query/server.py",
            "8082"  # Puerto diferente para pruebas
        ], cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Esperar que inicie
        time.sleep(3)

        # Verificar que está corriendo
        if server_process.poll() is None:
            log("✅ Servidor iniciado correctamente", "SUCCESS")

            # Probar health endpoint
            try:
                response = requests.get("http://localhost:8082/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    log("✅ Health check exitoso", "SUCCESS")
                    log(f"   Versión: {health_data.get('version', 'N/A')}")
                    log(f"   Optimizaciones activas: {len(health_data.get('optimizations', {}))} módulos")
                else:
                    log(f"❌ Health check falló: {response.status_code}", "ERROR")
            except Exception as e:
                log(f"❌ Error en health check: {e}", "ERROR")

            # Detener servidor
            server_process.terminate()
            server_process.wait(timeout=5)
            log("✅ Servidor detenido correctamente", "SUCCESS")

            return True
        else:
            stdout, stderr = server_process.communicate()
            log(f"❌ Servidor falló al iniciar: {stderr.decode()}", "ERROR")
            return False

    except Exception as e:
        log(f"❌ Error al probar servidor: {e}", "ERROR")
        return False

def test_optimizations():
    """Prueba todas las optimizaciones implementadas"""
    log("🧪 Probando optimizaciones implementadas...")

    try:
        # Importar módulos de optimización
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'servers', 'context-query'))

        from optimizations import (
            token_budget, semantic_chunker, cache, query_optimizer,
            rate_limiter, resource_monitor, fuzzy_search, relevance_scorer
        )

        from spec_driven import SpecParser, SpecIndexer
        from document_loader import TrainingManager
        from reflector import Reflector
        from curator import Curator

        tests_passed = 0
        total_tests = 0

        # 1. Token Budgeting
        total_tests += 1
        try:
            sections = [
                {"content": "Esto es un test", "relevance": 0.8, "tokens": 10},
                {"content": "Otro contenido más largo para testing", "relevance": 0.6, "tokens": 15}
            ]
            allocated = token_budget.allocate_tokens(sections)
            if len(allocated) > 0:
                log("✅ Token Budgeting funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Token Budgeting falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Token Budgeting: {e}", "ERROR")

        # 2. Semantic Chunking
        total_tests += 1
        try:
            text = "Este es un párrafo de prueba. Contiene varias oraciones. Cada una con diferente contenido semántico. Para probar el chunking avanzado."
            chunks = semantic_chunker.semantic_chunk(text)
            if len(chunks) > 0 and 'content' in chunks[0]:
                log("✅ Semantic Chunking funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Semantic Chunking falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Semantic Chunking: {e}", "ERROR")

        # 3. Multi-level Cache
        total_tests += 1
        try:
            test_key = "test_cache_key"
            test_value = {"data": "test_value"}
            cache.set(test_key, test_value, ttl=60)
            retrieved = cache.get(test_key)
            if retrieved and retrieved['data'] == test_value['data']:
                log("✅ Multi-level Cache funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Multi-level Cache falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Multi-level Cache: {e}", "ERROR")

        # 4. Query Optimization
        total_tests += 1
        try:
            result = query_optimizer.optimize_query("¿Cómo funciona el código?")
            expected_keys = {"original_query", "normalized_query", "expanded_terms", "filtered_terms"}
            if expected_keys.issubset(result.keys()) and result['normalized_query']:
                log("✅ Query Optimization funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Query Optimization falló: estructura inesperada", "ERROR")
        except Exception as e:
            log(f"❌ Error en Query Optimization: {e}", "ERROR")

        # 5. Rate Limiting
        total_tests += 1
        try:
            # Test básico de rate limiting
            allowed1 = rate_limiter.check_limit("127.0.0.1")
            allowed2 = rate_limiter.check_limit("127.0.0.1")
            if isinstance(allowed1, bool) and isinstance(allowed2, bool):
                log("✅ Rate Limiting funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Rate Limiting falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Rate Limiting: {e}", "ERROR")

        # 6. Fuzzy Search
        total_tests += 1
        try:
            # Crear índice de prueba
            test_docs = {
                "doc1": {"content": "python django model", "section_id": "coding", "section_title": "Coding"},
                "doc2": {"content": "seguridad autenticacion", "section_id": "security", "section_title": "Security"}
            }
            fuzzy_search.build_index(test_docs)

            if not fuzzy_search.has_index():
                raise AssertionError("Índice fuzzy no fue construido")

            results = fuzzy_search.search("python modelo")
            top_doc = results[0][0] if results else None

            if results and top_doc == "doc1":
                # Confirmar que podemos recuperar el documento original
                retrieved_doc = fuzzy_search.get_document(top_doc)
                if retrieved_doc and retrieved_doc.get("section_title") == "Coding":
                    log("✅ Fuzzy Search funciona correctamente", "SUCCESS")
                    tests_passed += 1
                else:
                    log("❌ Fuzzy Search falló al recuperar documento", "ERROR")
            else:
                log("❌ Fuzzy Search falló en ranking", "ERROR")
        except Exception as e:
            log(f"❌ Error en Fuzzy Search: {e}", "ERROR")

        # 8. Spec-Driven Development (SpecParser)
        total_tests += 1
        try:
            parser = SpecParser()
            test_content = """
## User Stories
As a user, I want to login so that I can access my account.

## API Specifications
POST /api/login
Content-Type: application/json
{
  "username": "string",
  "password": "string"
}
"""
            specs = parser.parse_document(test_content, "test_doc.md")
            if 'user_stories' in specs and 'api_specifications' in specs:
                if len(specs['user_stories']) > 0 and len(specs['api_specifications']) > 0:
                    log("✅ Spec-Driven Parser funciona correctamente", "SUCCESS")
                    tests_passed += 1
                else:
                    log("❌ Spec-Driven Parser no extrajo specs", "ERROR")
            else:
                log("❌ Spec-Driven Parser falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Spec-Driven Parser: {e}", "ERROR")

        # 9. Spec-Driven Development (SpecIndexer)
        total_tests += 1
        try:
            indexer = SpecIndexer()
            # Simular specs para indexar
            test_specs = {
                "test_doc.md": {
                    "user_stories": [{"content": "As a user I want to login", "confidence": 0.9}],
                    "api_specifications": [{"content": "POST /api/login", "confidence": 0.8}]
                }
            }
            indexer.index_specs(test_specs)
            results = indexer.search_specs("login")
            if results and len(results) > 0:
                log("✅ Spec-Driven Indexer funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Spec-Driven Indexer falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Spec-Driven Indexer: {e}", "ERROR")

        # 10. Training Manager
        total_tests += 1
        try:
            import tempfile
            import os

            # Crear directorio temporal con archivo de prueba
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = os.path.join(temp_dir, "test.md")
                with open(test_file, "w") as f:
                    f.write("# Test Document\n\n## User Stories\nAs a user, I want to test.")

                manager = TrainingManager(temp_dir)
                result = manager.train_system()
                if result['status'] == 'trained':
                    log("✅ Training Manager funciona correctamente", "SUCCESS")
                    tests_passed += 1
                else:
                    log("❌ Training Manager falló", "ERROR")
        except Exception as e:
            log(f"❌ Error en Training Manager: {e}", "ERROR")

        # 11. Reflector (ACE)
        total_tests += 1
        try:
            import tempfile

            # Crear archivo feedback temporal
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump([
                    {"query": "test", "response": "test response", "helpful": True},
                    {"query": "test2", "response": "test response 2", "helpful": False}
                ], f)
                feedback_file = f.name

            reflector_instance = Reflector(feedback_file)
            analysis = reflector_instance.analyze_feedback()
            if 'insights' in analysis:
                log("✅ Reflector (ACE) funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Reflector (ACE) falló", "ERROR")

            os.unlink(feedback_file)
        except Exception as e:
            log(f"❌ Error en Reflector (ACE): {e}", "ERROR")

        # 12. Curator (ACE)
        total_tests += 1
        try:
            import tempfile

            # Crear archivos temporales
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as idx_f:
                json.dump({"test": "index"}, idx_f)
                index_file = idx_f.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as guide_f:
                json.dump({"test": "guidelines"}, guide_f)
                guidelines_file = guide_f.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as fb_f:
                json.dump([], fb_f)
                feedback_file = fb_f.name

            curator_instance = Curator(index_file, guidelines_file, feedback_file)
            insights = [{"type": "missing_keyword", "keyword": "test"}]
            updates = curator_instance.apply_insights(insights)
            if isinstance(updates, list):
                log("✅ Curator (ACE) funciona correctamente", "SUCCESS")
                tests_passed += 1
            else:
                log("❌ Curator (ACE) falló", "ERROR")

            for f in [index_file, guidelines_file, feedback_file]:
                os.unlink(f)
        except Exception as e:
            log(f"❌ Error en Curator (ACE): {e}", "ERROR")
        success_rate = (tests_passed / total_tests) * 100
        log(f"📊 Tests completados: {tests_passed}/{total_tests} ({success_rate:.1f}%)", "SUCCESS" if success_rate >= 80 else "WARNING")

        return success_rate >= 80

    except Exception as e:
        log(f"❌ Error general en pruebas de optimización: {e}", "ERROR")
        return False

def main():
    """Función principal de pruebas"""
    log("🎯 Iniciando suite de pruebas del MCP Hub Optimizado")
    log("=" * 60)

    tests_passed = 0
    total_tests = 3  # servidor + optimizaciones básicas + nuevas técnicas

    # 1. Prueba de servidor
    if test_server_startup():
        tests_passed += 1
        log("✅ Test de servidor: PASÓ", "SUCCESS")
    else:
        log("❌ Test de servidor: FALLÓ", "ERROR")

    log("-" * 40)

    # 2. Prueba de optimizaciones
    if test_optimizations():
        tests_passed += 1
        log("✅ Tests de optimización: PASÓ", "SUCCESS")
    else:
        log("❌ Tests de optimización: FALLÓ", "ERROR")

    # Resultado final
    log("=" * 60)
    final_success = (tests_passed == total_tests)
    if final_success:
        log("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE", "SUCCESS")
        log("🚀 El MCP Hub Optimizado 2.0 está listo para producción!")
    else:
        log(f"⚠️  {tests_passed}/{total_tests} pruebas pasaron", "WARNING")
        log("🔧 Revisar errores antes del despliegue")

    return final_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
