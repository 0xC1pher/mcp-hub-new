#!/usr/bin/env python3
"""
Instalador de dependencias MCP Hub v4.0
Simple, directo, sin complicaciones
"""

import subprocess
import sys
import os

def print_header(text):
    """Imprime encabezado"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_step(step, text):
    """Imprime paso"""
    print(f"\n[{step}] {text}")

def check_python():
    """Verifica versión de Python"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Advertencia: Python 3.8+ recomendado")
        print(f"   Tu versión: {version.major}.{version.minor}")
        return False
    return True

def install_package(package_name, version=""):
    """Instala un paquete con pip"""
    package_spec = f"{package_name}{version}" if version else package_name

    try:
        print(f"  Instalando {package_name}...", end=" ")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✅")
            return True
        else:
            print("❌")
            if "--verbose" in sys.argv:
                print(f"    Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ (timeout)")
        return False
    except Exception as e:
        print(f"❌ ({str(e)[:50]})")
        return False

def verify_import(module_name):
    """Verifica que un módulo se pueda importar"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    """Función principal"""
    print_header("Instalador de Dependencias - MCP Hub v4.0")

    # Verificar Python
    print_step("1/4", "Verificando Python...")
    check_python()

    # Actualizar pip
    print_step("2/4", "Actualizando pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True
    )
    print("  pip actualizado ✅")

    # Dependencias principales
    print_step("3/4", "Instalando dependencias principales...")

    dependencies = [
        ("numpy", ">=1.21.0"),
        ("msgpack", ">=1.0.5"),
        ("zstandard", ">=0.19.0"),
    ]

    installed = 0
    failed = 0

    for package, version in dependencies:
        if install_package(package, version):
            installed += 1
        else:
            failed += 1

    # Verificación
    print_step("4/4", "Verificando instalación...")

    modules_to_verify = {
        "numpy": "NumPy",
        "msgpack": "MessagePack",
        "zstandard": "Zstandard"
    }

    verified = 0
    for module, name in modules_to_verify.items():
        if verify_import(module):
            print(f"  {name}: ✅")
            verified += 1
        else:
            print(f"  {name}: ❌")

    # Resumen
    print_header("Resumen de Instalación")
    print(f"  Paquetes instalados: {installed}/{len(dependencies)}")
    print(f"  Paquetes fallidos:   {failed}/{len(dependencies)}")
    print(f"  Módulos verificados: {verified}/{len(modules_to_verify)}")

    # Verificar sistema v4.0
    print("\n" + "=" * 80)
    print("  Verificando Sistema v4.0...")
    print("=" * 80)

    try:
        # Añadir directorio actual al path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from core.advanced_features import create_orchestrator
        orchestrator = create_orchestrator("fast")
        status = orchestrator.get_system_status()
        enabled = len(status['config']['enabled_features'])

        print(f"\n  ✅ Sistema v4.0 FUNCIONAL")
        print(f"  ✅ {enabled} características habilitadas")

    except Exception as e:
        print(f"\n  ⚠️  Sistema v4.0 no disponible")
        print(f"     Error: {str(e)[:100]}")
        print(f"\n  El sistema funcionará en modo básico")

    # Conclusión
    print("\n" + "=" * 80)

    if installed == len(dependencies) and verified == len(modules_to_verify):
        print("  ✅ INSTALACIÓN COMPLETA Y EXITOSA")
        print("\n  Siguiente paso:")
        print("    Ejecuta: START.bat")
        print("    Selecciona: [1] Iniciar Sistema")
    elif installed > 0:
        print("  ⚠️  INSTALACIÓN PARCIAL")
        print(f"\n  {installed} de {len(dependencies)} paquetes instalados")
        print("  El sistema funcionará con funcionalidad limitada")
    else:
        print("  ❌ INSTALACIÓN FALLÓ")
        print("\n  Soluciones:")
        print("    1. Verifica conexión a internet")
        print("    2. Ejecuta como administrador")
        print("    3. Actualiza Python: python.org")

    print("=" * 80)

    input("\nPresiona Enter para salir...")

    return 0 if installed == len(dependencies) else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Instalación interrumpida")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        sys.exit(1)
