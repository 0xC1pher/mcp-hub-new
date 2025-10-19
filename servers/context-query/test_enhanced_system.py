#!/usr/bin/env python3
"""
Test del Sistema Enhanced MCP con Cache Inteligente
Demuestra el flujo: Cache Local → Modelo → Feedback System
"""

import json
import time
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_mcp_system():
    """Prueba el sistema completo Enhanced MCP"""
    
    print("🚀 Iniciando prueba del Enhanced MCP System")
    print("=" * 60)
    
    try:
        # Importar el servidor mejorado
        from enhanced_mcp_server import EnhancedMCPServer
        
        # Crear instancia del servidor
        server = EnhancedMCPServer()
        
        print(f"✅ Servidor inicializado")
        print(f"   - Feedback System: {'✅' if server.feedback_system else '❌'}")
        print(f"   - Cache Inteligente: {'✅' if server.intelligent_cache else '❌'}")
        
        # Simular consultas para probar el flujo
        test_queries = [
            "¿Cómo funciona el sistema de cache?",
            "¿Cuáles son las optimizaciones implementadas?",
            "¿Cómo se previenen las alucinaciones?",
            "¿Qué es el Context Feedback System?",
            "¿Cómo funciona el chunking semántico?"
        ]
        
        print(f"\n🔍 Probando {len(test_queries)} consultas...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Consulta {i}: {query}")
            print("-" * 50)
            
            # Simular request MCP
            request_params = {
                "name": "context_query",
                "arguments": {"query": query}
            }
            
            start_time = time.time()
            
            # Procesar consulta
            response = server.handle_tools_call(request_params)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            # Mostrar resultado
            if "content" in response:
                content = response["content"][0].get("text", "")
                content_preview = content[:200] + "..." if len(content) > 200 else content
                
                print(f"⏱️  Tiempo de respuesta: {response_time:.2f}ms")
                print(f"📄 Respuesta: {content_preview}")
                
                # Mostrar metadata si existe
                if "cache_metadata" in response:
                    metadata = response["cache_metadata"]
                    print(f"💾 Cache Hit: {metadata.get('cache_hit', False)}")
                    print(f"📊 Resultados: {metadata.get('results_count', 0)}")
                
                if "context_metadata" in response:
                    metadata = response["context_metadata"]
                    print(f"🔄 Query Count: {metadata.get('query_count', 0)}")
                    print(f"✅ Feature Check: {metadata.get('feature_requirements_checked', False)}")
            
            # Pequeña pausa entre consultas
            time.sleep(1)
        
        # Probar métricas del cache
        print(f"\n📊 Probando métricas del cache...")
        print("-" * 50)
        
        metrics_request = {
            "name": "cache_metrics",
            "arguments": {}
        }
        
        metrics_response = server.handle_tools_call(metrics_request)
        if "content" in metrics_response:
            metrics_text = metrics_response["content"][0].get("text", "")
            print(metrics_text[:500] + "..." if len(metrics_text) > 500 else metrics_text)
        
        # Probar búsqueda directa en cache
        print(f"\n🔍 Probando búsqueda directa en cache...")
        print("-" * 50)
        
        cache_search_request = {
            "name": "cache_search", 
            "arguments": {"query": "optimizaciones", "max_results": 3}
        }
        
        cache_response = server.handle_tools_call(cache_search_request)
        if "content" in cache_response:
            cache_text = cache_response["content"][0].get("text", "")
            print(cache_text[:400] + "..." if len(cache_text) > 400 else cache_text)
        
        print(f"\n✅ Prueba completada exitosamente")
        
        # Cerrar servidor limpiamente
        if hasattr(server, 'shutdown'):
            server.shutdown()
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("   Asegúrate de que todos los módulos estén disponibles")
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

def test_cache_performance():
    """Prueba específica de rendimiento del cache"""
    
    print(f"\n🏃 Prueba de Rendimiento del Cache")
    print("=" * 60)
    
    try:
        from intelligent_cache_system import IntelligentCacheSystem
        
        # Crear directorio temporal con archivos de prueba
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Crear archivos de prueba
            test_files = [
                ("test1.py", "def hello_world():\n    return 'Hello, World!'"),
                ("test2.md", "# Test Document\nThis is a test markdown file."),
                ("test3.json", '{"name": "test", "value": 42}'),
                ("test4.txt", "Simple text file for testing purposes."),
            ]
            
            for filename, content in test_files:
                (temp_dir / filename).write_text(content)
            
            # Inicializar cache
            cache = IntelligentCacheSystem(
                source_directory=str(temp_dir),
                l1_size=10,
                l2_size=50, 
                disk_size=100
            )
            
            # Esperar a que se cacheen los archivos
            time.sleep(3)
            
            # Probar búsquedas
            test_searches = [
                "hello world",
                "test document", 
                "json value",
                "simple text"
            ]
            
            total_time = 0
            total_searches = 0
            
            for search in test_searches:
                start_time = time.time()
                results = cache.search(search, max_results=3)
                end_time = time.time()
                
                search_time = (end_time - start_time) * 1000
                total_time += search_time
                total_searches += 1
                
                print(f"🔍 '{search}': {len(results)} resultados en {search_time:.2f}ms")
            
            # Mostrar métricas
            metrics = cache.get_metrics()
            print(f"\n📊 Métricas del Cache:")
            print(f"   Hit Rate: {metrics['hit_rate']:.2%}")
            print(f"   L1 Hits: {metrics['l1_hits']}")
            print(f"   L2 Hits: {metrics['l2_hits']}")
            print(f"   Disk Hits: {metrics['disk_hits']}")
            print(f"   Misses: {metrics['misses']}")
            print(f"   Tiempo promedio: {total_time/total_searches:.2f}ms")
            
            cache.shutdown()
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"❌ Error en prueba de cache: {e}")

def main():
    """Función principal"""
    
    print("🧪 Enhanced MCP System - Suite de Pruebas")
    print("=" * 60)
    
    # Prueba 1: Sistema completo
    test_enhanced_mcp_system()
    
    # Prueba 2: Rendimiento del cache
    test_cache_performance()
    
    print(f"\n🎉 Todas las pruebas completadas")

if __name__ == "__main__":
    main()
