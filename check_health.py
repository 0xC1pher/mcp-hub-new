import sys
import os
import importlib.util
from pathlib import Path

def check_import(module_name, friendly_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {friendly_name}: INSTALADO")
        return True
    except ImportError as e:
        print(f"❌ {friendly_name}: ERROR ({e})")
        return False

def check_file(path, description):
    if os.path.exists(path):
        print(f"✅ {description}: ENCONTRADO")
        return True
    else:
        print(f"⚠️ {description}: NO ENCONTRADO (Se creará al iniciar)")
        return False

print("\n=== MCP v6 HEALTH CHECK ===\n")

# 1. Verificar Dependencias Críticas
deps_ok = True
deps_ok &= check_import("torch", "PyTorch Core")
deps_ok &= check_import("transformers", "HuggingFace Transformers")
deps_ok &= check_import("sentence_transformers", "Sentence Transformers")
deps_ok &= check_import("pymp4", "PyMP4 (Storage)")
deps_ok &= check_import("hnswlib", "HNSWLib (Vector Index)")

print("-" * 30)

# 2. Verificar Módulos Internos v6
internal_ok = True
sys.path.append(os.getcwd())
try:
    from core.shared.toon_serializer import TOONSerializer
    print(f"✅ TOON Serializer: ACTIVO")
except ImportError as e:
    print(f"❌ TOON Serializer: FALLÓ ({e})")
    internal_ok = False

try:
    from core.storage.mp4_storage import MP4Storage
    print(f"✅ MP4 Storage Engine: ACTIVO")
except ImportError as e:
    print(f"❌ MP4 Storage Engine: FALLÓ ({e})")
    internal_ok = False

print("-" * 30)

# 3. Verificar Datos
check_file("config/v5_config.json", "Configuración")
check_file("data/context_vectors.mp4", "Base de Datos Vectorial")

print("\n=== RESULTADO ===")
if deps_ok and internal_ok:
    print("🚀 TODO LISTO: El sistema está 100% operativo.")
else:
    print("⚠️ ATENCIÓN: Hay problemas que necesitan revisión.")
