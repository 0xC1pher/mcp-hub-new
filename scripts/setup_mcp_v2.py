"""
Script de instalación y configuración del Sistema MCP Optimizado v2.0
Ejecutar después de instalar requirements-mcp.txt
"""
import os
import sys
from pathlib import Path

def print_header(text):
    """Imprime encabezado con estilo"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"✓ {text}")

def print_error(text):
    """Imprime mensaje de error"""
    print(f"✗ {text}")

def print_info(text):
    """Imprime mensaje informativo"""
    print(f"ℹ {text}")

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print_header("Verificando Dependencias")
    
    required_packages = [
        'chromadb',
        'sentence_transformers',
        'transformers',
        'torch',
        'numpy',
        'pydantic'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} instalado")
        except ImportError:
            print_error(f"{package} NO instalado")
            missing.append(package)
    
    if missing:
        print_error(f"\nFaltan dependencias: {', '.join(missing)}")
        print_info("Ejecuta: pip install -r requirements-mcp.txt")
        return False
    
    print_success("\nTodas las dependencias están instaladas")
    return True

def create_directories():
    """Crea directorios necesarios"""
    print_header("Creando Directorios")
    
    directories = [
        'chroma_db',
        'cache',
        'mcp_core'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Directorio creado: {directory}/")
        else:
            print_info(f"Directorio ya existe: {directory}/")
    
    print_success("\nDirectorios configurados")

def test_mcp_import():
    """Prueba importar el módulo MCP"""
    print_header("Probando Importación de MCP Core")
    
    try:
        from mcp_core import (
            VectorStoreManager,
            IntelligentIndexer,
            SmartCache,
            OptimizedMCPService,
            get_mcp_service
        )
        print_success("VectorStoreManager importado")
        print_success("IntelligentIndexer importado")
        print_success("SmartCache importado")
        print_success("OptimizedMCPService importado")
        print_success("get_mcp_service importado")
        
        print_success("\nMódulo MCP Core importado correctamente")
        return True
    except Exception as e:
        print_error(f"Error importando MCP Core: {e}")
        return False

def initialize_mcp():
    """Inicializa el servicio MCP"""
    print_header("Inicializando Servicio MCP")
    
    try:
        from mcp_core import get_mcp_service
        
        project_root = Path(__file__).parent
        print_info(f"Raíz del proyecto: {project_root}")
        
        mcp = get_mcp_service(project_root=str(project_root))
        print_success("Servicio MCP inicializado")
        
        # Health check
        health = mcp.health_check()
        
        if health['status'] == 'healthy':
            print_success(f"Estado del sistema: {health['status']}")
        else:
            print_error(f"Estado del sistema: {health['status']}")
        
        for component, info in health['components'].items():
            if info['status'] == 'ok':
                print_success(f"  {component}: {info['status']}")
            else:
                print_error(f"  {component}: {info['status']}")
        
        return True
    except Exception as e:
        print_error(f"Error inicializando MCP: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_example_query():
    """Ejecuta una consulta de ejemplo"""
    print_header("Ejecutando Consulta de Ejemplo")
    
    try:
        from mcp_core import get_mcp_service
        
        project_root = Path(__file__).parent
        mcp = get_mcp_service(project_root=str(project_root))
        
        # Indexar primero (solo archivos nuevos)
        print_info("Indexando proyecto (esto puede tomar unos minutos)...")
        stats = mcp.initialize_index(force_reindex=False)
        
        print_success(f"Archivos escaneados: {stats['scanned']}")
        print_success(f"Archivos nuevos: {stats['new']}")
        print_success(f"Archivos modificados: {stats['modified']}")
        
        if stats['scanned'] > 0:
            # Consulta de ejemplo
            print_info("\nEjecutando consulta de ejemplo...")
            
            response = mcp.query(
                query_text="modelo de paciente",
                n_results=3,
                search_mode='hybrid'
            )
            
            print_success(f"Consulta completada en {response['response_time_ms']}ms")
            print_success(f"Resultados encontrados: {response['total_results']}")
            print_success(f"Fuente: {response['source']}")
            
            if response['results']:
                print_info("\nPrimer resultado:")
                result = response['results'][0]
                print(f"  Archivo: {result['metadata'].get('source', 'N/A')}")
                print(f"  Similitud: {result.get('similarity', 0):.4f}")
                print(f"  Contenido: {result['content'][:150]}...")
        else:
            print_info("No hay archivos para indexar aún")
        
        return True
    except Exception as e:
        print_error(f"Error en consulta de ejemplo: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """Muestra los siguientes pasos"""
    print_header("Siguientes Pasos")
    
    print("1. Indexar el proyecto completo:")
    print("   python manage.py mcp_index index")
    print()
    print("2. Consultar el contexto:")
    print("   python manage.py mcp_index query --query \"tu consulta aquí\"")
    print()
    print("3. Ver estadísticas:")
    print("   python manage.py mcp_index stats")
    print()
    print("4. Optimizar el sistema:")
    print("   python manage.py mcp_index optimize")
    print()
    print("5. Verificar salud:")
    print("   python manage.py mcp_index health")
    print()
    print("📖 Documentación completa: docs/MCP_OPTIMIZADO_V2.md")
    print()

def main():
    """Función principal"""
    print_header("🚀 Setup MCP Optimizado v2.0")
    print("Sistema de Base de Datos Vectorizada + Cache Multinivel")
    
    # Paso 1: Verificar dependencias
    if not check_dependencies():
        print_error("\nInstalación fallida: Faltan dependencias")
        sys.exit(1)
    
    # Paso 2: Crear directorios
    create_directories()
    
    # Paso 3: Probar importación
    if not test_mcp_import():
        print_error("\nInstalación fallida: Error importando módulos")
        sys.exit(1)
    
    # Paso 4: Inicializar MCP
    if not initialize_mcp():
        print_error("\nInstalación fallida: Error inicializando MCP")
        sys.exit(1)
    
    # Paso 5: Ejecutar ejemplo (opcional)
    print_info("\n¿Deseas ejecutar una consulta de ejemplo? (s/N): ")
    try:
        response = input().strip().lower()
        if response == 's':
            run_example_query()
    except:
        pass
    
    # Mostrar siguientes pasos
    show_next_steps()
    
    print_header("✅ Instalación Completada")
    print_success("Sistema MCP Optimizado v2.0 listo para usar")
    print()

if __name__ == '__main__':
    main()
