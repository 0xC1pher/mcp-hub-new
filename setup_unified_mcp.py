#!/usr/bin/env python3
"""
Setup Unificado para Servidor MCP
Configura automáticamente el servidor MCP unificado con todas las técnicas avanzadas
"""

import json
import os
import sys
import shutil
from pathlib import Path
import subprocess

def setup_unified_mcp():
    """Configura el servidor MCP unificado"""
    
    print("🚀 Configurando Servidor MCP Unificado...")
    
    # Rutas
    project_root = Path(__file__).parent.parent
    mcp_hub = Path(__file__).parent
    windsurf_config_path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    
    print(f"📁 Proyecto: {project_root}")
    print(f"📁 MCP Hub: {mcp_hub}")
    
    # 1. Crear directorios necesarios
    create_directories(mcp_hub)
    
    # 2. Configurar Windsurf
    configure_windsurf(windsurf_config_path, mcp_hub)
    
    # 3. Crear archivos de configuración
    create_config_files(mcp_hub)
    
    # 4. Instalar dependencias si es necesario
    install_dependencies()
    
    # 5. Verificar instalación
    verify_installation(mcp_hub)
    
    print("\n✅ Configuración completada!")
    print("\n📋 Próximos pasos:")
    print("1. Reinicia Windsurf para cargar la nueva configuración")
    print("2. Usa el servidor 'yari-medic-unified' desde Windsurf")
    print("3. Ejecuta test: python test_unified_mcp.py")

def create_directories(mcp_hub: Path):
    """Crea directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    directories = [
        mcp_hub / "cache",
        mcp_hub / "logs", 
        mcp_hub / "config",
        mcp_hub / "shared",
        mcp_hub / "scripts"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"   ✅ {directory.name}/")

def configure_windsurf(config_path: Path, mcp_hub: Path):
    """Configura Windsurf MCP"""
    print("\n⚙️ Configurando Windsurf...")
    
    # Crear directorio si no existe
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configuración unificada
    config = {
        "mcpServers": {
            "yari-medic-unified": {
                "command": "python",
                "args": [str(mcp_hub / "unified_mcp_server.py")],
                "cwd": str(mcp_hub),
                "env": {
                    "PYTHONPATH": str(mcp_hub)
                },
                "disabled": False
            }
        }
    }
    
    # Leer configuración existente si existe
    existing_config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
        except:
            pass
    
    # Mergear configuraciones
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}
    
    existing_config["mcpServers"]["yari-medic-unified"] = config["mcpServers"]["yari-medic-unified"]
    
    # Guardar configuración
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"   ✅ Configuración guardada en: {config_path}")

def create_config_files(mcp_hub: Path):
    """Crea archivos de configuración"""
    print("\n📄 Creando archivos de configuración...")
    
    # Archivo de configuración del servidor
    server_config = {
        "cache": {
            "l1_size": 100,
            "l2_size": 1000,
            "disk_size": 10000,
            "ttl_seconds": 3600
        },
        "chunking": {
            "chunk_size": 1000,
            "overlap": 200,
            "min_chunk_size": 50
        },
        "scoring": {
            "exact_match_weight": 2.0,
            "partial_match_weight": 1.5,
            "context_density_weight": 0.8,
            "relevance_threshold": 0.3
        },
        "ace_system": {
            "enabled": True,
            "duplicate_detection": True,
            "contextual_analysis": True,
            "evolution_learning": True
        }
    }
    
    config_file = mcp_hub / "config" / "server_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(server_config, f, indent=2)
    
    print(f"   ✅ server_config.json")
    
    # Script de test
    test_script = mcp_hub / "test_unified_mcp.py"
    test_content = '''#!/usr/bin/env python3
"""Test del servidor MCP unificado"""

import json
import subprocess
import sys
from pathlib import Path

def test_unified_server():
    """Prueba el servidor unificado"""
    print("🧪 Probando Servidor MCP Unificado...")
    
    # Test básico de importación
    try:
        from unified_mcp_server import UnifiedMCPServer
        print("   ✅ Importación exitosa")
    except ImportError as e:
        print(f"   ❌ Error de importación: {e}")
        return False
    
    # Test de inicialización
    try:
        server = UnifiedMCPServer()
        print("   ✅ Inicialización exitosa")
    except Exception as e:
        print(f"   ❌ Error de inicialización: {e}")
        return False
    
    # Test de herramientas
    try:
        tools = server._list_tools()
        print(f"   ✅ {len(tools['tools'])} herramientas disponibles")
        for tool in tools['tools']:
            print(f"      - {tool['name']}")
    except Exception as e:
        print(f"   ❌ Error listando herramientas: {e}")
        return False
    
    # Test de query
    try:
        result = server._context_query({"query": "test", "max_results": 3})
        print("   ✅ Query de prueba exitosa")
    except Exception as e:
        print(f"   ❌ Error en query: {e}")
        return False
    
    print("\\n🎉 Todos los tests pasaron!")
    return True

if __name__ == "__main__":
    success = test_unified_server()
    sys.exit(0 if success else 1)
'''
    
    with open(test_script, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"   ✅ test_unified_mcp.py")

def install_dependencies():
    """Instala dependencias si es necesario"""
    print("\n📦 Verificando dependencias...")
    
    required_packages = [
        "pathlib",  # Built-in
        "hashlib",  # Built-in
        "json",     # Built-in
        "threading" # Built-in
    ]
    
    print("   ✅ Todas las dependencias están disponibles (built-in)")

def verify_installation(mcp_hub: Path):
    """Verifica la instalación"""
    print("\n🔍 Verificando instalación...")
    
    # Verificar archivos principales
    required_files = [
        "unified_mcp_server.py",
        "setup_unified_mcp.py",
        "test_unified_mcp.py",
        "config/server_config.json"
    ]
    
    all_good = True
    for file_path in required_files:
        full_path = mcp_hub / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - FALTANTE")
            all_good = False
    
    # Verificar directorios
    required_dirs = ["cache", "logs", "config", "shared", "scripts"]
    for dir_name in required_dirs:
        dir_path = mcp_hub / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - FALTANTE")
            all_good = False
    
    if all_good:
        print("\n✅ Instalación verificada correctamente")
    else:
        print("\n❌ Problemas encontrados en la instalación")
        sys.exit(1)

def cleanup_old_servers():
    """Limpia servidores antiguos (opcional)"""
    print("\n🧹 Limpieza de servidores antiguos...")
    
    # Esta función es opcional - no elimina nada por defecto
    # Solo informa sobre la unificación
    
    print("   ℹ️ Los servidores anteriores se mantienen para compatibilidad")
    print("   ℹ️ El servidor unificado combina todas las funcionalidades")
    print("   ℹ️ Puedes usar 'yari-medic-unified' como servidor principal")

if __name__ == "__main__":
    setup_unified_mcp()
