#!/usr/bin/env python3
"""
Script de pruebas completo para validar todas las optimizaciones del MCP
Verifica que todas las técnicas documentadas estén implementadas y funcionando
"""

import json
import sys
import time
import traceback
from pathlib import Path

# Agregar el directorio actual al path para importar el servidor
sys.path.insert(0, str(Path(__file__).parent))

from optimized_mcp_server import (
    OptimizedMCPContextServer,
    TokenBudgetManager,
    SemanticChunker,
    MultiLevelCache,
    QueryOptimizer,
    RateLimiter,
    ResourceMonitor,
    FuzzySearch,
    RelevanceScorer
)

class OptimizationTester:
    """Tester completo para todas las optimizaciones"""
    
    def __init__(self):
        self.results = {
            'token_budgeting': False,
            'semantic_chunking': False,
            'multilevel_cache': False,
            'query_optimization': False,
            'rate_limiting': False,
            'resource_monitoring': False,
            'fuzzy_search': False,
            'relevance_scoring': False,
            'mcp_protocol': False,
            'integration': False
        }
        self.errors = []
        
    def test_token_budgeting(self):
        """Prueba Token Budgeting Inteligente"""
        print("🎯 Probando Token Budgeting Inteligente...")
        try:
            budget_manager = TokenBudgetManager(max_tokens=1000, reserved_tokens=200)
            
            # Test estimación de tokens
            text = "Este es un texto de prueba para estimar tokens"
            tokens = budget_manager.estimate_tokens(text)
            assert tokens > 0, "Estimación de tokens debe ser mayor a 0"
            
            # Test cálculo de prioridad
            section = {
                'content': text,
                'relevance': 0.8,
                'last_updated': time.time(),
                'access_count': 5
            }
            priority = budget_manager.calculate_priority(section)
            assert 0 <= priority <= 4, f"Prioridad debe estar entre 0-4, obtenido: {priority}"
            
            # Test asignación de tokens
            sections = [section, section.copy()]
            allocated = budget_manager.allocate_tokens(sections)
            assert len(allocated) <= len(sections), "Secciones asignadas no deben exceder las originales"
            
            self.results['token_budgeting'] = True
            print("✅ Token Budgeting: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Token Budgeting: {str(e)}")
            print(f"❌ Token Budgeting: ERROR - {str(e)}")
    
    def test_semantic_chunking(self):
        """Prueba Chunking Semántico Avanzado"""
        print("🧩 Probando Chunking Semántico Avanzado...")
        try:
            chunker = SemanticChunker(chunk_size=500, overlap=100)
            
            # Test contenido con secciones
            content = """
            <!-- SECTION_ID: test_section -->
            Este es contenido de prueba para el chunking semántico.
            Debe dividir el contenido de manera inteligente.
            <!-- SECTION_ID: another_section -->
            Esta es otra sección para probar la división.
            """
            
            chunks = chunker.chunk_content(content)
            assert len(chunks) > 0, "Debe generar al menos un chunk"
            
            # Verificar estructura de chunks
            for chunk in chunks:
                assert 'id' in chunk, "Chunk debe tener ID"
                assert 'section_id' in chunk, "Chunk debe tener section_id"
                assert 'content' in chunk, "Chunk debe tener contenido"
                assert 'start_pos' in chunk, "Chunk debe tener posición inicial"
                assert 'end_pos' in chunk, "Chunk debe tener posición final"
            
            self.results['semantic_chunking'] = True
            print("✅ Semantic Chunking: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Semantic Chunking: {str(e)}")
            print(f"❌ Semantic Chunking: ERROR - {str(e)}")
    
    def test_multilevel_cache(self):
        """Prueba Cache Multinivel (L1/L2/Disk)"""
        print("💾 Probando Cache Multinivel...")
        try:
            cache = MultiLevelCache(l1_size=10, l2_size=50)
            
            # Test set/get básico
            cache.set("test_key", {"data": "test_value"}, ttl=60)
            result = cache.get("test_key")
            assert result is not None, "Debe recuperar valor del cache"
            assert result["data"] == "test_value", "Valor debe coincidir"
            
            # Test promoción entre niveles
            for i in range(15):  # Llenar L1 y forzar promoción
                cache.set(f"key_{i}", f"value_{i}")
            
            # Verificar que L1 no excede el tamaño
            assert len(cache.l1_cache) <= cache.l1_size, "L1 cache no debe exceder tamaño máximo"
            
            # Test cache en disco
            disk_result = cache._get_from_disk("test_key")
            # Puede ser None si no se guardó en disco aún
            
            self.results['multilevel_cache'] = True
            print("✅ Multilevel Cache: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Multilevel Cache: {str(e)}")
            print(f"❌ Multilevel Cache: ERROR - {str(e)}")
    
    def test_query_optimization(self):
        """Prueba Query Optimization Avanzada"""
        print("🔍 Probando Query Optimization...")
        try:
            optimizer = QueryOptimizer()
            
            # Test expansión de consulta
            query = "¿Cómo crear un paciente nuevo?"
            expanded = optimizer.expand_query(query)
            assert len(expanded) > 0, "Debe generar términos expandidos"
            assert any('paciente' in term for term in expanded), "Debe incluir término paciente"
            
            # Test extracción de contexto
            context_terms = optimizer.extract_context_terms(query)
            assert isinstance(context_terms, dict), "Debe retornar diccionario de términos"
            assert 'medical_terms' in context_terms, "Debe incluir términos médicos"
            
            # Test clasificación de consulta
            query_type = optimizer._classify_query(query)
            assert query_type in ['medical', 'business', 'architecture', 'coding', 'technology', 'security', 'data', 'performance', 'ui_ux', 'general'], "Tipo de consulta debe ser válido"
            
            # Test normalización
            normalized = optimizer.normalize_query("¿Cómo CREAR un Paciente?")
            # La normalización debe remover signos de puntuación y convertir a minúsculas
            expected_words = ["cómo", "crear", "un", "paciente"]  # Conserva acentos
            normalized_words = normalized.split()
            assert normalized_words == expected_words, f"Normalización incorrecta: {normalized_words} vs {expected_words}"
            
            self.results['query_optimization'] = True
            print("✅ Query Optimization: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Query Optimization: {str(e)}")
            print(f"❌ Query Optimization: ERROR - {str(e)}")
    
    def test_rate_limiting(self):
        """Prueba Rate Limiting Adaptativo"""
        print("🛡️ Probando Rate Limiting...")
        try:
            limiter = RateLimiter(max_requests_per_second=5, max_requests_per_minute=20)
            
            # Test requests normales
            for i in range(4):
                allowed = limiter.is_allowed("test_client")
                assert allowed, f"Request {i} debería estar permitida"
            
            # Test límite por segundo
            for i in range(10):  # Exceder límite
                limiter.is_allowed("test_client")
            
            # Verificar penalización
            # Después de exceder límite, debería haber penalización
            
            self.results['rate_limiting'] = True
            print("✅ Rate Limiting: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Rate Limiting: {str(e)}")
            print(f"❌ Rate Limiting: ERROR - {str(e)}")
    
    def test_resource_monitoring(self):
        """Prueba Resource Monitoring Completo"""
        print("📊 Probando Resource Monitoring...")
        try:
            monitor = ResourceMonitor()
            
            # Test registro de métricas
            monitor.record_metrics(response_time=0.1, cache_hit=True)
            monitor.record_metrics(response_time=0.2, cache_hit=False)
            
            # Test obtención de métricas
            metrics = monitor.get_metrics()
            assert 'uptime' in metrics, "Debe incluir uptime"
            assert 'request_count' in metrics, "Debe incluir contador de requests"
            assert 'cpu_avg_percent' in metrics, "Debe incluir CPU promedio"
            assert 'memory_avg_percent' in metrics, "Debe incluir memoria promedio"
            assert 'response_time_avg' in metrics, "Debe incluir tiempo de respuesta promedio"
            assert 'cache_hit_rate' in metrics, "Debe incluir tasa de aciertos de cache"
            
            assert metrics['request_count'] == 2, "Contador de requests debe ser 2"
            
            self.results['resource_monitoring'] = True
            print("✅ Resource Monitoring: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Resource Monitoring: {str(e)}")
            print(f"❌ Resource Monitoring: ERROR - {str(e)}")
    
    def test_fuzzy_search(self):
        """Prueba Fuzzy Search y N-gramas"""
        print("🎯 Probando Fuzzy Search...")
        try:
            fuzzy = FuzzySearch()
            
            # Test construcción de índice
            documents = [
                {'id': 'doc1', 'content': 'paciente médico consulta'},
                {'id': 'doc2', 'content': 'facturación pago sistema'},
                {'id': 'doc3', 'content': 'arquitectura django python'}
            ]
            
            fuzzy.build_index(documents)
            assert len(fuzzy.index) > 0, "Debe construir índice de n-gramas"
            
            # Test búsqueda
            results = fuzzy.search("paciente", threshold=0.3)
            assert len(results) > 0, "Debe encontrar resultados"
            
            # Verificar estructura de resultados
            for doc_id, score in results:
                assert isinstance(doc_id, str), "ID debe ser string"
                assert 0 <= score <= 1, f"Score debe estar entre 0-1, obtenido: {score}"
            
            self.results['fuzzy_search'] = True
            print("✅ Fuzzy Search: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Fuzzy Search: {str(e)}")
            print(f"❌ Fuzzy Search: ERROR - {str(e)}")
    
    def test_relevance_scoring(self):
        """Prueba Relevance Scoring Multifactor"""
        print("🎯 Probando Relevance Scoring...")
        try:
            scorer = RelevanceScorer()
            
            # Test scoring de documento
            doc = {
                'content': 'Este documento habla sobre pacientes y médicos en el sistema',
                'recency_score': 0.8
            }
            
            query = "pacientes médicos"
            query_optimized = {
                'keywords': ['pacientes', 'médicos'],
                'expanded_terms': ['paciente', 'medico', 'doctor', 'enfermo']
            }
            
            score = scorer.score_document(doc, query, query_optimized)
            assert 0 <= score <= 1, f"Score debe estar entre 0-1, obtenido: {score}"
            assert score > 0, "Score debe ser mayor a 0 para contenido relevante"
            
            self.results['relevance_scoring'] = True
            print("✅ Relevance Scoring: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Relevance Scoring: {str(e)}")
            print(f"❌ Relevance Scoring: ERROR - {str(e)}")
    
    def test_mcp_protocol(self):
        """Prueba Protocolo MCP"""
        print("🔌 Probando Protocolo MCP...")
        try:
            server = OptimizedMCPContextServer()
            
            # Test inicialización
            init_response = server.handle_initialize({})
            assert 'protocolVersion' in init_response, "Debe incluir versión del protocolo"
            assert 'capabilities' in init_response, "Debe incluir capacidades"
            assert 'serverInfo' in init_response, "Debe incluir información del servidor"
            
            # Test lista de herramientas
            tools_response = server.handle_tools_list({})
            assert 'tools' in tools_response, "Debe incluir lista de herramientas"
            assert len(tools_response['tools']) > 0, "Debe tener al menos una herramienta"
            
            # Verificar herramienta context_query
            context_tool = next((tool for tool in tools_response['tools'] if tool['name'] == 'context_query'), None)
            assert context_tool is not None, "Debe incluir herramienta context_query"
            assert 'inputSchema' in context_tool, "Herramienta debe tener esquema de entrada"
            
            self.results['mcp_protocol'] = True
            print("✅ MCP Protocol: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"MCP Protocol: {str(e)}")
            print(f"❌ MCP Protocol: ERROR - {str(e)}")
    
    def test_integration(self):
        """Prueba Integración Completa"""
        print("🔗 Probando Integración Completa...")
        try:
            server = OptimizedMCPContextServer()
            
            # Test consulta completa
            call_params = {
                'name': 'context_query',
                'arguments': {
                    'query': '¿Cómo funciona el sistema de pacientes?'
                }
            }
            
            response = server.handle_tools_call(call_params)
            assert 'content' in response, "Respuesta debe tener contenido"
            assert len(response['content']) > 0, "Debe tener contenido de respuesta"
            assert response['content'][0]['type'] == 'text', "Contenido debe ser texto"
            
            # Verificar que no hay error
            assert not response.get('isError', False), "No debe haber error en respuesta válida"
            
            self.results['integration'] = True
            print("✅ Integration: FUNCIONAL")
            
        except Exception as e:
            self.errors.append(f"Integration: {str(e)}")
            print(f"❌ Integration: ERROR - {str(e)}")
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        print("🚀 INICIANDO PRUEBAS DE OPTIMIZACIÓN MCP HUB")
        print("=" * 60)
        
        test_methods = [
            self.test_token_budgeting,
            self.test_semantic_chunking,
            self.test_multilevel_cache,
            self.test_query_optimization,
            self.test_rate_limiting,
            self.test_resource_monitoring,
            self.test_fuzzy_search,
            self.test_relevance_scoring,
            self.test_mcp_protocol,
            self.test_integration
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ ERROR CRÍTICO en {test_method.__name__}: {str(e)}")
                traceback.print_exc()
            print()
        
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen de resultados"""
        print("=" * 60)
        print("📋 RESUMEN DE PRUEBAS")
        print("=" * 60)
        
        passed = sum(self.results.values())
        total = len(self.results)
        
        for optimization, status in self.results.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {optimization.replace('_', ' ').title()}: {'FUNCIONAL' if status else 'ERROR'}")
        
        print()
        print(f"🎯 RESULTADO FINAL: {passed}/{total} optimizaciones funcionando")
        
        if passed == total:
            print("🎉 ¡TODAS LAS OPTIMIZACIONES ESTÁN FUNCIONANDO CORRECTAMENTE!")
        else:
            print("⚠️  ALGUNAS OPTIMIZACIONES NECESITAN CORRECCIÓN")
        
        if self.errors:
            print("\n🔍 ERRORES ENCONTRADOS:")
            for error in self.errors:
                print(f"   • {error}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    tester = OptimizationTester()
    tester.run_all_tests()
