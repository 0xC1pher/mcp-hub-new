#!/usr/bin/env python3
"""
Script para validar la sintaxis del unified_mcp_server.py
"""
import ast
import sys

def test_syntax():
    try:
        with open('unified_mcp_server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parsear el código
        ast.parse(content)
        print("✅ SINTAXIS CORRECTA - unified_mcp_server.py")
        
        # Verificar imports
        try:
            import json, sys, logging, time, hashlib, os
            from pathlib import Path
            from typing import Dict, Any, List, Optional, Tuple
            from collections import defaultdict
            import threading
            from datetime import datetime, timedelta
            print("✅ IMPORTS BÁSICOS - Disponibles")
        except ImportError as e:
            print(f"⚠️ IMPORT FALTANTE: {e}")
        
        # Verificar estructura básica
        tree = ast.parse(content)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        print(f"📊 ESTADÍSTICAS:")
        print(f"   - Clases encontradas: {len(classes)}")
        print(f"   - Funciones encontradas: {len(functions)}")
        
        # Verificar clases críticas
        required_classes = ['UnifiedMCPServer', 'UnifiedCacheSystem', 'SemanticChunker']
        missing_classes = [cls for cls in required_classes if cls not in classes]
        
        if missing_classes:
            print(f"⚠️ CLASES FALTANTES: {missing_classes}")
        else:
            print("✅ CLASES CRÍTICAS - Presentes")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ ERROR DE SINTAXIS:")
        print(f"   Línea {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        return False

if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)
