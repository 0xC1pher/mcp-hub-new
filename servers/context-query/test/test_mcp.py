#!/usr/bin/env python3
"""
Script de prueba para el servidor HTTP MCP Context Query
Prueba todos los endpoints incluyendo las nuevas funcionalidades Spec-Driven
"""

import json
import subprocess
import time
import sys
import requests
from pathlib import Path

def test_http_server():
    """Prueba el servidor HTTP MCP con todos los endpoints"""
    print("🧪 Probando servidor HTTP MCP Context Query...")
    print("=" * 60)

    # Ruta al servidor HTTP
    server_path = Path(__file__).parent / "server.py"

    try:
        # Iniciar servidor en puerto de prueba
        print("🚀 Iniciando servidor HTTP en puerto 8083...")
        process = subprocess.Popen(
            [sys.executable, str(server_path), "8083"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Esperar que inicie
        time.sleep(3)

        base_url = "http://localhost:8083"

        tests_passed = 0
        total_tests = 6

        # Test 1: Health Check
        print("\n📊 Test 1: Health Check")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health check exitoso")
                print(f"   Status: {health_data.get('status')}")
                print(f"   Optimizaciones: {len(health_data.get('optimizations', {}))}")
                print(f"   Spec-Driven: {health_data.get('spec_driven', {}).get('training_status', 'N/A')}")
                tests_passed += 1
            else:
                print(f"❌ Health check falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en health check: {e}")

        # Test 2: Manifest
        print("\n📋 Test 2: Manifest")
        try:
            response = requests.get(f"{base_url}/manifest", timeout=5)
            if response.status_code == 200:
                manifest = response.json()
                print("✅ Manifest recuperado exitosamente")
                print(f"   Nombre: {manifest.get('name', 'N/A')}")
                print(f"   Versión: {manifest.get('version', 'N/A')}")
                tests_passed += 1
            else:
                print(f"❌ Manifest falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en manifest: {e}")

        # Test 3: Context Query (básico)
        print("\n🔍 Test 3: Context Query Básico")
        try:
            query_data = {"query": "¿Cómo se estructura el código?"}
            response = requests.post(
                f"{base_url}/tools/context_query",
                json=query_data,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                content = result.get('result', '')
                print("✅ Context query exitoso")
                print(f"   Respuesta: {len(content)} caracteres")
                print(f"   Preview: {content[:100]}..." if content else "   Sin contenido")
                tests_passed += 1
            else:
                print(f"❌ Context query falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en context query: {e}")

        # Test 4: Training System (Spec-Driven)
        print("\n🎓 Test 4: Training System (Spec-Driven)")
        try:
            response = requests.post(f"{base_url}/tools/train_system", timeout=15)
            if response.status_code == 200:
                training_result = response.json()
                print("✅ Training system exitoso")
                print(f"   Status: {training_result.get('training', {}).get('status')}")
                specs_summary = training_result.get('specs_summary', {})
                print(f"   Specs indexadas: {specs_summary.get('total_specs', 0)}")
                tests_passed += 1
            else:
                print(f"❌ Training system falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en training system: {e}")

        # Test 5: Context Query con Specs (después del training)
        print("\n🔍 Test 5: Context Query con Specs")
        try:
            query_data = {"query": "user story login"}
            response = requests.post(
                f"{base_url}/tools/context_query",
                json=query_data,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                content = result.get('result', '')
                print("✅ Context query con specs exitoso")
                print(f"   Respuesta: {len(content)} caracteres")
                # Verificar si contiene info de specs
                if 'User Stories' in content or 'API' in content:
                    print("   ✅ Respuesta incluye contenido de especificaciones")
                else:
                    print("   ℹ️  Respuesta fallback a búsqueda tradicional")
                tests_passed += 1
            else:
                print(f"❌ Context query con specs falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en context query con specs: {e}")

        # Test 6: Feedback (ACE)
        print("\n💬 Test 6: Feedback (ACE)")
        try:
            feedback_data = {
                "query": "test query",
                "response": "test response",
                "helpful": True,
                "suggestion": "Great response"
            }
            response = requests.post(
                f"{base_url}/tools/feedback",
                json=feedback_data,
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                print("✅ Feedback enviado exitosamente")
                print(f"   Status: {result.get('status')}")
                tests_passed += 1
            else:
                print(f"❌ Feedback falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en feedback: {e}")

        # Test 7: Analyze Feedback (ACE)
        print("\n🧠 Test 7: Analyze Feedback (ACE)")
        try:
            response = requests.post(f"{base_url}/tools/analyze_feedback", timeout=10)
            if response.status_code == 200:
                analysis_result = response.json()
                print("✅ Feedback analysis exitoso")
                print(f"   Insights generados: {len(analysis_result.get('analysis', {}).get('insights', []))}")
                print(f"   Updates aplicados: {len(analysis_result.get('updates_applied', []))}")
                tests_passed += 1
            else:
                print(f"❌ Feedback analysis falló: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en feedback analysis: {e}")

        # Resultado final
        print("\n" + "=" * 60)
        success_rate = (tests_passed / 7) * 100  # Incluyendo el test 7 que agregué
        if success_rate >= 80:
            print("🎉 SUITE DE PRUEBAS COMPLETADA EXITOSAMENTE!")
            print(f"Éxito: {success_rate:.1f}%")
        else:
            print(f"⚠️  Tests parcialmente exitosos: {success_rate:.1f}%")
        print(f"Tests pasados: {tests_passed}/7")
        print("=" * 60)

        # Terminar proceso
        process.terminate()
        process.wait(timeout=5)

        return success_rate >= 80

    except Exception as e:
        print(f"❌ Error general en las pruebas: {e}")
        if 'process' in locals():
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                pass
        return False

if __name__ == "__main__":
    success = test_http_server()
    sys.exit(0 if success else 1)
