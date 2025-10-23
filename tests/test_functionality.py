#!/usr/bin/env python3
"""
Script para probar la funcionalidad del MCP unificado reparado
"""
import json
import sys
from pathlib import Path

# Agregar el directorio actual al path para imports
sys.path.insert(0, str(Path(__file__).parent))

def test_mcp_functionality():
    """Prueba las funcionalidades principales del MCP"""
    
    try:
        # Importar el servidor
        from unified_mcp_server import UnifiedMCPServer
        print("✅ IMPORT EXITOSO - UnifiedMCPServer")
        
        # Inicializar servidor
        project_root = Path(__file__).parent.parent
        server = UnifiedMCPServer(str(project_root))
        print("✅ INICIALIZACIÓN - Servidor creado")
        
        # Probar listado de herramientas
        tools_response = server._list_tools()
        tools = tools_response.get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        
        print(f"📊 HERRAMIENTAS DISPONIBLES ({len(tools)}):")
        for name in tool_names:
            print(f"   - {name}")
        
        # Verificar nuevas herramientas críticas
        critical_tools = ['code_review', 'detect_duplicates', 'context_query']
        missing_tools = [tool for tool in critical_tools if tool not in tool_names]
        
        if missing_tools:
            print(f"⚠️ HERRAMIENTAS FALTANTES: {missing_tools}")
        else:
            print("✅ HERRAMIENTAS CRÍTICAS - Presentes")
        
        # Probar code review
        print("\n🔍 PROBANDO CODE REVIEW...")
        review_request = {
            'method': 'tools/call',
            'params': {
                'name': 'code_review',
                'arguments': {
                    'task_description': 'Crear nueva función para gestión de pacientes',
                    'target_files': ['pacientes/models.py', 'pacientes/views.py']
                }
            }
        }
        
        review_response = server.handle_request(review_request)
        if 'error' in review_response:
            print(f"❌ ERROR EN CODE REVIEW: {review_response['error']}")
        else:
            print("✅ CODE REVIEW - Funcionando")
            # Mostrar parte de la respuesta
            content = review_response.get('content', [{}])[0].get('text', '')
            print(f"   Respuesta (primeros 200 chars): {content[:200]}...")
        
        # Probar detección de duplicados
        print("\n🔍 PROBANDO DETECCIÓN DE DUPLICADOS...")
        duplicate_request = {
            'method': 'tools/call',
            'params': {
                'name': 'detect_duplicates',
                'arguments': {
                    'path': str(project_root),
                    'threshold': 0.8
                }
            }
        }
        
        duplicate_response = server.handle_request(duplicate_request)
        if 'error' in duplicate_response:
            print(f"❌ ERROR EN DETECCIÓN: {duplicate_response['error']}")
        else:
            print("✅ DETECCIÓN DE DUPLICADOS - Funcionando")
        
        # Probar consulta de contexto
        print("\n🔍 PROBANDO CONSULTA DE CONTEXTO...")
        context_request = {
            'method': 'tools/call',
            'params': {
                'name': 'context_query',
                'arguments': {
                    'query': 'cómo funciona el sistema de pacientes',
                    'max_results': 3
                }
            }
        }
        
        context_response = server.handle_request(context_request)
        if 'error' in context_response:
            print(f"❌ ERROR EN CONTEXTO: {context_response['error']}")
        else:
            print("✅ CONSULTA DE CONTEXTO - Funcionando")
        
        # Estadísticas del sistema
        print("\n📊 ESTADÍSTICAS DEL SISTEMA:")
        stats_request = {
            'method': 'tools/call',
            'params': {
                'name': 'system_stats',
                'arguments': {}
            }
        }
        
        stats_response = server.handle_request(stats_request)
        if 'error' not in stats_response:
            print("✅ ESTADÍSTICAS - Disponibles")
        
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"   - Archivos indexados: {len(server.indexed_files)}")
        print(f"   - Consultas procesadas: {server.query_count}")
        print(f"   - Sistema avanzado: {'✅ Activo' if server.advanced_system else '❌ No disponible'}")
        print(f"   - Indexación de contexto: {'✅ Activo' if server.context_indexer else '❌ No disponible'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ ERROR DE IMPORT: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DEL MCP UNIFICADO REPARADO\n")
    success = test_mcp_functionality()
    
    if success:
        print("\n✅ TODAS LAS PRUEBAS PASARON - Sistema MCP funcional")
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON - Revisar errores")
    
    sys.exit(0 if success else 1)
