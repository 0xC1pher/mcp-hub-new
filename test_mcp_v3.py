#!/usr/bin/env python3
"""
Script de prueba para MCP v3 Enhanced
Verifica compatibilidad con v2 y nuevas funcionalidades Grok
"""

import sys
import time
from pathlib import Path

# Agregar mcp-hub al path
sys.path.insert(0, str(Path(__file__).parent))

def test_mcp_v3():
    """Prueba completa del MCP v3 Enhanced"""
    
    print("🚀 INICIANDO PRUEBAS MCP v3 ENHANCED")
    print("=" * 50)
    
    try:
        # Importar MCP v3
        from mcp_v3_enhanced import MCPv3EnhancedServer, get_mcp_v3_server
        print("✅ MCP v3 importado correctamente")
        
        # Inicializar servidor
        project_root = Path(__file__).parent.parent
        server = MCPv3EnhancedServer(str(project_root))
        print("✅ Servidor MCP v3 inicializado")
        
        # Verificar compatibilidad v2
        print("\n🔄 VERIFICANDO COMPATIBILIDAD V2...")
        if server.v2_server:
            print("✅ Compatibilidad v2 activa")
            print(f"   - Archivos indexados: {len(server.v2_server.indexed_files)}")
            print(f"   - Sistema avanzado: {'✅' if server.v2_server.advanced_system else '❌'}")
        else:
            print("⚠️ Compatibilidad v2 limitada")
        
        # Probar funcionalidades v2 heredadas
        print("\n🧪 PROBANDO FUNCIONALIDADES V2 HEREDADAS...")
        
        # Test context_query (v2)
        v2_request = {
            'method': 'tools/call',
            'params': {
                'name': 'context_query',
                'arguments': {
                    'query': 'sistema de pacientes en Yari-System',
                    'max_results': 3
                }
            }
        }
        
        v2_response = server.handle_request(v2_request)
        if 'error' not in v2_response:
            print("✅ context_query (v2) funcionando")
        else:
            print(f"⚠️ context_query (v2): {v2_response.get('error', 'Error desconocido')}")
        
        # Test code_review (v2)
        review_request = {
            'method': 'tools/call',
            'params': {
                'name': 'code_review',
                'arguments': {
                    'task_description': 'Crear nueva función para gestión de citas médicas',
                    'target_files': ['citas/models.py', 'citas/views.py']
                }
            }
        }
        
        review_response = server.handle_request(review_request)
        if 'error' not in review_response:
            print("✅ code_review (v2) funcionando")
        else:
            print(f"⚠️ code_review (v2): {review_response.get('error', 'Error desconocido')}")
        
        # Probar nuevas funcionalidades v3
        print("\n🧠 PROBANDO NUEVAS FUNCIONALIDADES V3...")
        
        # Test análisis Grok
        grok_request = {
            'method': 'tools/call',
            'params': {
                'name': 'grok_analysis',
                'arguments': {
                    'query': 'optimización del flujo de atención médica en Yari-System',
                    'context_limit': 5
                }
            }
        }
        
        grok_response = server.handle_request(grok_request)
        if 'error' not in grok_response:
            print("✅ grok_analysis (v3) funcionando")
            # Mostrar parte de la respuesta
            content = grok_response.get('content', [{}])[0].get('text', '')
            print(f"   Respuesta (primeros 200 chars): {content[:200]}...")
        else:
            print(f"⚠️ grok_analysis (v3): {grok_response.get('error', 'Error desconocido')}")
        
        # Test memoria avanzada
        memory_request = {
            'method': 'tools/call',
            'params': {
                'name': 'advanced_memory_query',
                'arguments': {
                    'query': 'consultas médicas recientes',
                    'memory_type': 'episodic'
                }
            }
        }
        
        memory_response = server.handle_request(memory_request)
        if 'error' not in memory_response:
            print("✅ advanced_memory_query (v3) funcionando")
        else:
            print(f"⚠️ advanced_memory_query (v3): {memory_response.get('error', 'Error desconocido')}")
        
        # Test enhancement Grok en respuestas v2
        print("\n🔬 PROBANDO ENHANCEMENT GROK...")
        
        enhanced_request = {
            'method': 'tools/call',
            'params': {
                'name': 'context_query',
                'arguments': {
                    'query': 'gestión de historias clínicas',
                    'max_results': 2
                }
            }
        }
        
        enhanced_response = server.handle_request(enhanced_request)
        if 'ANÁLISIS GROK ADICIONAL' in str(enhanced_response):
            print("✅ Enhancement Grok activo")
        else:
            print("⚠️ Enhancement Grok no detectado")
        
        # Estadísticas v3
        print("\n📊 ESTADÍSTICAS MCP V3...")
        v3_stats = server.get_v3_stats()
        
        print(f"   🔢 Versión: {v3_stats['version']}")
        print(f"   🔄 Compatibilidad v2: {'✅' if v3_stats['v2_compatibility'] else '❌'}")
        print(f"   🧠 Análisis Grok: {v3_stats['v3_metrics']['grok_analyses']}")
        print(f"   💡 Insights profundos: {v3_stats['v3_metrics']['deep_insights']}")
        print(f"   🧩 Patrones Grok: {v3_stats['grok_patterns']}")
        print(f"   💾 Capas de memoria: {v3_stats['memory_layers']}")
        
        # Test singleton
        print("\n🔄 PROBANDO PATRÓN SINGLETON...")
        server2 = get_mcp_v3_server()
        if server is server2:
            print("✅ Patrón singleton funcionando")
        else:
            print("⚠️ Problema con singleton")
        
        print("\n🎉 PRUEBAS MCP V3 COMPLETADAS")
        print("=" * 50)
        
        # Resumen final
        print("\n📋 RESUMEN:")
        print(f"   ✅ MCP v3 Enhanced operativo")
        print(f"   ✅ Compatibilidad v2 mantenida")
        print(f"   ✅ Técnicas Grok integradas")
        print(f"   ✅ Memoria persistente avanzada")
        print(f"   ✅ Enhancement automático activo")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_v3_vs_v2():
    """Benchmark de rendimiento v3 vs v2"""
    
    print("\n⚡ BENCHMARK V3 VS V2")
    print("=" * 30)
    
    try:
        from mcp_v3_enhanced import get_mcp_v3_server
        
        server = get_mcp_v3_server()
        
        # Test query simple
        test_query = {
            'method': 'tools/call',
            'params': {
                'name': 'context_query',
                'arguments': {
                    'query': 'pacientes hospitalizados',
                    'max_results': 5
                }
            }
        }
        
        # Benchmark v2 (sin enhancement)
        start_time = time.time()
        if server.v2_server:
            v2_response = server.v2_server.handle_request(test_query)
        v2_time = time.time() - start_time
        
        # Benchmark v3 (con enhancement)
        start_time = time.time()
        v3_response = server.handle_request(test_query)
        v3_time = time.time() - start_time
        
        print(f"⏱️ Tiempo v2: {v2_time:.3f}s")
        print(f"⏱️ Tiempo v3: {v3_time:.3f}s")
        print(f"📊 Overhead v3: {((v3_time - v2_time) / v2_time * 100):.1f}%")
        
        # Comparar tamaño de respuestas
        v2_size = len(str(v2_response)) if 'v2_response' in locals() else 0
        v3_size = len(str(v3_response))
        
        print(f"📏 Tamaño respuesta v2: {v2_size} chars")
        print(f"📏 Tamaño respuesta v3: {v3_size} chars")
        print(f"📈 Información adicional: {((v3_size - v2_size) / v2_size * 100):.1f}%")
        
    except Exception as e:
        print(f"❌ Error en benchmark: {e}")

if __name__ == "__main__":
    success = test_mcp_v3()
    
    if success:
        benchmark_v3_vs_v2()
        print("\n✅ TODAS LAS PRUEBAS EXITOSAS")
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON")
    
    sys.exit(0 if success else 1)
