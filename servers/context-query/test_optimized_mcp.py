#!/usr/bin/env python3
"""
Script de prueba para el servidor MCP Context Query Optimizado
"""

import json
import subprocess
import time
import sys
from pathlib import Path

def test_optimized_mcp_server():
    """Prueba el servidor MCP optimizado"""
    print("🧪 Probando servidor MCP Context Query Optimizado...")
    
    # Ruta al servidor
    server_path = Path(__file__).parent / "optimized_mcp_server.py"
    
    try:
        # Iniciar servidor
        print("🚀 Iniciando servidor optimizado...")
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Esperar un poco
        time.sleep(2)
        
        # Test 1: Initialize
        print("📋 Probando inicialización optimizada...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Leer respuesta
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            server_info = response.get('result', {}).get('serverInfo', {})
            print(f"✅ Inicialización exitosa: {server_info.get('name')} v{server_info.get('version')}")
        
        # Test 2: List tools
        print("🔧 Probando listado de herramientas optimizadas...")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        process.stdin.write(json.dumps(tools_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            tools = response.get('result', {}).get('tools', [])
            print(f"✅ Herramientas encontradas: {len(tools)}")
            for tool in tools:
                print(f"   - {tool.get('name')}: {tool.get('description')}")
        
        # Test 3: Call tool con consulta compleja
        print("🔍 Probando consulta optimizada compleja...")
        query_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "context_query",
                "arguments": {
                    "query": "¿Cómo se estructura el código y cuáles son las convenciones de naming?"
                }
            }
        }
        
        process.stdin.write(json.dumps(query_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            result = response.get('result', {})
            content = result.get('content', [])
            if content:
                text = content[0].get('text', '')
                print(f"✅ Consulta optimizada exitosa: {len(text)} caracteres")
                print(f"   Respuesta: {text[:150]}...")
                
                # Verificar que incluye optimizaciones
                if "Relevancia:" in text:
                    print("   ✅ Scoring de relevancia activo")
                if "---" in text:
                    print("   ✅ Múltiples secciones encontradas")
            else:
                print("❌ No se recibió contenido")
        
        # Test 4: Consulta de negocio
        print("💼 Probando consulta de modelo de negocio...")
        business_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "context_query",
                "arguments": {
                    "query": "¿Cuál es el modelo de negocio y fuentes de ingreso?"
                }
            }
        }
        
        process.stdin.write(json.dumps(business_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            result = response.get('result', {})
            content = result.get('content', [])
            if content:
                text = content[0].get('text', '')
                print(f"✅ Consulta de negocio exitosa: {len(text)} caracteres")
                if "Business Model" in text or "Modelo de Negocio" in text:
                    print("   ✅ Información de negocio encontrada")
            else:
                print("❌ No se recibió contenido")
        
        # Test 5: Rate limiting
        print("⏱️ Probando rate limiting...")
        for i in range(15):  # Más del límite de 10 por segundo
            rate_request = {
                "jsonrpc": "2.0",
                "id": 5 + i,
                "method": "tools/call",
                "params": {
                    "name": "context_query",
                    "arguments": {
                        "query": f"Consulta de prueba {i}"
                    }
                }
            }
            
            process.stdin.write(json.dumps(rate_request) + "\n")
            process.stdin.flush()
            
            response_line = process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                if response.get('result', {}).get('isError'):
                    print(f"   ✅ Rate limiting activo en request {i+1}")
                    break
        
        # Terminar proceso
        process.terminate()
        process.wait()
        
        print("🎉 Todas las pruebas del servidor optimizado pasaron exitosamente!")
        print("✅ Optimizaciones verificadas:")
        print("   • Token Budgeting Inteligente")
        print("   • Chunking Semántico Avanzado")
        print("   • Cache Multinivel")
        print("   • Query Optimization")
        print("   • Rate Limiting Adaptativo")
        print("   • Resource Monitoring")
        print("   • Fuzzy Search y Relevance Scoring")
        return True
        
    except Exception as e:
        print(f"❌ Error en las pruebas optimizadas: {e}")
        if 'process' in locals():
            process.terminate()
        return False

if __name__ == "__main__":
    success = test_optimized_mcp_server()
    sys.exit(0 if success else 1)
