#!/usr/bin/env python3
"""
Servidor MCP Context Query para Yari Medic - Versión Optimizada 2.0
Implementa todas las estrategias avanzadas de OPTIMIZATION-STRATEGIES.md:
- Token Budgeting Inteligente
- Chunking Semántico Avanzado
- Cache Multinivel (L1/L2/Disk)
- Query Optimization con expansión semántica
- Rate Limiting Adaptativo
- Resource Monitoring y Performance Metrics
- Fuzzy Search y Relevance Scoring
- Arquitectura modular y escalable
"""

import json
import re
import time
import threading
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import logging

# Importar optimizaciones avanzadas
from optimizations import (
    token_budget, semantic_chunker, cache, query_optimizer,
    rate_limiter, resource_monitor, fuzzy_search, relevance_scorer
)

# Importar componentes ACE
from reflector import Reflector
from curator import Curator

# Importar componentes Spec-Driven
from spec_driven import SpecParser, SpecIndexer
from document_loader import TrainingManager

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/context-query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContextQueryHandler(BaseHTTPRequestHandler):
    """Handler HTTP para el servidor MCP Context Query"""

    def __init__(self, *args, **kwargs):
        # Cargar archivos al inicializar
        self.base_path = Path(__file__).resolve().parent
        self.guidelines_file = self.base_path / "context" / "project-guidelines.md"
        self.index_file = self.base_path / "index" / "keyword-to-sections.json"
        self.manifest_file = self.base_path / "manifest.json"
        self.feedback_file = self.base_path / "feedback.json"

        # Instancias ACE
        self.reflector = Reflector(str(self.feedback_file))
        self.curator = Curator(str(self.index_file), str(self.guidelines_file), str(self.feedback_file))

        # Instancias Spec-Driven
        self.spec_parser = SpecParser()
        self.spec_indexer = SpecIndexer()
        self.training_manager = TrainingManager(str(self.base_path.parent))  # Directorio padre

        # Cache para archivos (recarga cada 10 segundos)
        self.cache = {}
        self.cache_timestamp = 0
        self.cache_ttl = 10

        super().__init__(*args, **kwargs)

    def _load_files(self):
        """Carga archivos con optimizaciones de cache multinivel"""
        # Verificar cache primero
        cached_data = cache.get('project_files')
        if cached_data:
            documents = cached_data.get('documents')
            if documents and not fuzzy_search.has_index():
                fuzzy_search.build_index(documents)
            return cached_data

        try:
            # Cargar archivos del sistema de archivos
            guidelines_content = self._load_guidelines()
            index_data = self._load_index()
            manifest_data = self._load_manifest()

            # Procesar con chunking semántico
            processed_guidelines = self._process_guidelines_with_chunking(guidelines_content)

            # Construir índice de búsqueda fuzzy y documentos normalizados
            documents = self._build_search_index(processed_guidelines)

            # Guardar bullets para ACE
            self._save_bullets_to_file(processed_guidelines)

            # Entrenar sistema con Spec-Driven Development
            self._train_spec_driven_system()

            data = {
                'guidelines': guidelines_content,
                'processed_guidelines': processed_guidelines,
                'index': index_data,
                'manifest': manifest_data,
                'documents': documents
            }

            # Cachear datos procesados
            cache.set('project_files', data, ttl=600)  # 10 minutos
            cache.save_to_disk('project_files', data, ttl=3600)  # 1 hora en disco

            return data

        except Exception as e:
            logger.error(f"Error cargando archivos: {e}")
            return {'guidelines': '', 'processed_guidelines': [], 'index': {}, 'manifest': {}}

    def _load_guidelines(self):
        """Carga guidelines con optimizaciones"""
        if self.guidelines_file.exists():
            with open(self.guidelines_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def _load_index(self):
        """Carga índice con cache"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_manifest(self):
        """Carga manifest con cache"""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _process_guidelines_with_chunking(self, guidelines_content):
        """Procesa guidelines con chunking semántico"""
        sections = self._extract_sections(guidelines_content)
        processed_sections = []

        for section_id, content in sections.items():
            section_title = self._derive_section_title(section_id, content)

            # Aplicar chunking semántico
            chunks = semantic_chunker.semantic_chunk(
                content,
                {
                    'section_id': section_id,
                    'section_title': section_title
                }
            )

            for chunk_idx, chunk in enumerate(chunks):
                bullet_id = f"{section_id}_{chunk_idx}_{hash(chunk['content'][:50])}"
                processed_sections.append({
                    'section_id': section_id,
                    'section_title': section_title,
                    'content': chunk['content'],
                    'tokens': chunk['tokens'],
                    'metadata': chunk['metadata'],
                    'last_updated': time.time(),
                    # Bullet structure ACE
                    'bullet_id': bullet_id,
                    'helpful_count': 0,
                    'harmful_count': 0,
                    'theme': section_id,
                    'source': 'guidelines_chunk'
                })

        return processed_sections

    def _derive_section_title(self, section_id, content):
        """Obtiene un título legible para la sección"""
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            return stripped.lstrip('#').strip()
        return section_id.replace('_', ' ').title()

    def _extract_sections(self, content):
        """Extrae secciones de guidelines"""
        sections = {}
        pattern = r"<!-- SECTION_ID: ([a-z0-9_]+) -->(.*?)(?=<!-- SECTION_ID:|$)"
        matches = re.findall(pattern, content, re.DOTALL)

        for section_id, section_content in matches:
            sections[section_id] = section_content.strip()

        return sections

    def _build_search_index(self, processed_sections):
        """Construye índice de búsqueda fuzzy y devuelve documentos indexados"""
        documents = {}
        for section in processed_sections:
            doc_id = section['bullet_id']  # Usar bullet_id como doc_id
            documents[doc_id] = {
                'content': section['content'],
                'section_id': section['section_id'],
                'section_title': section.get('section_title'),
                'tokens': section.get('tokens'),
                'metadata': section.get('metadata', {}),
                'last_updated': section.get('last_updated', time.time()),
                # Metadata ACE
                'bullet_id': section.get('bullet_id'),
                'helpful_count': section.get('helpful_count', 0),
                'harmful_count': section.get('harmful_count', 0),
                'theme': section.get('theme'),
                'source': section.get('source')
            }

        fuzzy_search.build_index(documents)
        return documents

    def do_GET(self):
        """Maneja requests GET"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/manifest':
            self._handle_manifest()
        elif parsed_path.path == '/health':
            self._handle_health()
        else:
            self._send_error(404, "Endpoint no encontrado")

    def do_POST(self):
        """Maneja requests POST"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/tools/context_query':
            self._handle_context_query()
        elif parsed_path.path == '/tools/feedback':
            self._handle_feedback()
        elif parsed_path.path == '/tools/analyze_feedback':
            self._handle_analyze_feedback()
        elif parsed_path.path == '/tools/train_system':
            self._handle_train_system()
        else:
            self._send_error(404, "Endpoint no encontrado")

    def _handle_manifest(self):
        """Devuelve el manifest.json"""
        try:
            cache = self._load_files()
            manifest = cache.get('manifest', {})

            self._send_json_response(200, manifest)
            logger.info("Manifest solicitado")

        except Exception as e:
            logger.error(f"Error en manifest: {e}")
            self._send_error(500, "Error interno del servidor")

    def _handle_health(self):
        """Endpoint de health check con métricas optimizadas"""
        try:
            # Obtener métricas de recursos
            resource_metrics = resource_monitor.get_metrics_summary()
            cache_stats = cache.get_stats()

            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "2.0.0-optimized",
                "optimizations": {
                    "cache": {
                        "hit_rate": cache_stats.get('hit_rate', 0),
                        "l1_size": cache_stats.get('l1_size', 0),
                        "l2_size": cache_stats.get('l2_size', 0),
                        "disk_size": cache_stats.get('disk_size', 0)
                    },
                    "resources": {
                        "memory_avg_percent": resource_metrics.get('memory_avg_percent', 0),
                        "cpu_avg_percent": resource_metrics.get('cpu_avg_percent', 0),
                        "response_time_avg": resource_metrics.get('response_time_avg', 0)
                    },
                    "token_budget": {
                        "max_tokens": token_budget.max_tokens,
                        "available_tokens": token_budget.available_tokens,
                        "reserved_tokens": token_budget.reserved_tokens
                    },
                    "spec_driven": {
                        "training_status": self.training_manager.get_training_status(),
                        "specs_summary": self.spec_indexer.get_spec_summary()
                    }
                },
                "files_loaded": all([
                    self.guidelines_file.exists(),
                    self.index_file.exists(),
                    self.manifest_file.exists(),
                    self.feedback_file.exists()
                ])
            }

            self._send_json_response(200, health_status)
            logger.info("Health check optimizado solicitado")

        except Exception as e:
            logger.error(f"Error en health check: {e}")
            self._send_json_response(500, {"status": "error", "message": str(e)})

    def _handle_context_query(self):
        """Maneja la consulta de contexto con optimizaciones avanzadas"""
        start_time = time.time()

        try:
            # Verificar rate limiting
            client_ip = self.client_address[0]
            if not rate_limiter.check_limit(client_ip):
                self._send_json_response(429, {"error": "Rate limit exceeded"})
                return

            # Leer y parsear request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            query = request_data.get('query', '').strip()
            if not query:
                self._send_json_response(400, {"error": "Query requerida"})
                return

            # Optimizar consulta
            optimized_query = query_optimizer.optimize_query(query)

            # Procesar consulta con todas las optimizaciones
            result = self._process_optimized_query(optimized_query)

            # Registrar métricas
            response_time = time.time() - start_time
            resource_monitor.record_response_time(response_time)

            # Actualizar hit rate de cache
            cache_stats = cache.get_stats()
            resource_monitor.update_cache_hit_rate(cache_stats['hit_rate'])

            self._send_json_response(200, {"result": result})
            logger.info(f"Consulta optimizada procesada: '{query}' -> {len(result)} chars, {response_time:.2f}s")

        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "JSON inválido"})
        except Exception as e:
            logger.error(f"Error en context query optimizado: {e}")

    def _handle_feedback(self):
        """Maneja el feedback de consultas"""
        try:
            # Leer y parsear request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            query = request_data.get('query', '').strip()
            response = request_data.get('response', '').strip()
            helpful = request_data.get('helpful', False)
            suggestion = request_data.get('suggestion', '').strip()

            if not query or not response:
                self._send_json_response(400, {"error": "Query y response requeridas"})
                return

            # Cargar feedback existente
            feedback_list = self._load_feedback()

            # Agregar nueva entrada
            feedback_entry = {
                "query": query,
                "response": response,
                "helpful": helpful,
                "suggestion": suggestion,
                "timestamp": time.time()
            }
            feedback_list.append(feedback_entry)

            # Guardar
            self._save_feedback(feedback_list)

            self._send_json_response(200, {"status": "Feedback guardado"})
            logger.info(f"Feedback recibido para query: '{query}' - Helpful: {helpful}")

        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "JSON inválido"})
        except Exception as e:
            logger.error(f"Error en feedback: {e}")
            self._send_error(500, "Error interno del servidor")

    def _handle_analyze_feedback(self):
        """Analiza feedback y aplica mejoras con ACE"""
        try:
            # Ejecutar análisis del Reflector
            analysis_result = self.reflector.analyze_feedback()

            # Aplicar insights con el Curator
            insights = analysis_result.get('insights', [])
            updates = self.curator.apply_insights(insights)

            # Refinar bullets
            refined_bullets = self.curator.refine_bullets()

            result = {
                "analysis": analysis_result,
                "updates_applied": updates,
                "bullets_refined": len(refined_bullets),
                "timestamp": time.time()
            }

            self._send_json_response(200, result)
            logger.info(f"Análisis ACE completado: {len(insights)} insights, {len(updates)} updates")

        except Exception as e:
            logger.error(f"Error en análisis ACE: {e}")
            self._send_json_response(500, {"error": f"Error en análisis: {str(e)}"})

    def _handle_train_system(self):
        """Entrena el sistema manualmente"""
        try:
            force_retrain = self.headers.get('X-Force-Retrain', 'false').lower() == 'true'
            training_result = self.training_manager.train_system(force_retrain)

            # Si se entrenó, re-entrenar specs
            if training_result['status'] == 'trained':
                self._train_spec_driven_system()

            # Obtener resumen de specs
            spec_summary = self.spec_indexer.get_spec_summary()

            result = {
                'training': training_result,
                'specs_summary': spec_summary,
                'timestamp': time.time()
            }

            self._send_json_response(200, result)
            logger.info(f"Entrenamiento completado: {training_result}")

        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            self._send_json_response(500, {"error": f"Error en entrenamiento: {str(e)}"})

    def _load_feedback(self):
        """Carga feedback desde archivo"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_feedback(self, feedback_list):
        """Guarda feedback a archivo"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=2)

    def _save_bullets_to_file(self, bullets):
        """Guarda bullets a archivo para ACE"""
        bullets_file = self.base_path / "context_bullets.json"
        with open(bullets_file, 'w', encoding='utf-8') as f:
            json.dump(bullets, f, ensure_ascii=False, indent=2)

    def _train_spec_driven_system(self):
        """Entrena el sistema con Spec-Driven Development"""
        try:
            # Entrenar con documentos disponibles
            training_result = self.training_manager.train_system()

            if training_result['status'] == 'trained':
                # Parsear specs de todos los documentos
                all_specs = {}
                documents = self.training_manager.training_data.get('documents', {})

                for file_path, content in documents.items():
                    filename = Path(file_path).name
                    specs = self.spec_parser.parse_document(content, filename)
                    if specs:
                        all_specs[file_path] = specs

                # Indexar specs
                self.spec_indexer.index_specs(all_specs)

                logger.info(f"Sistema entrenado con {len(documents)} documentos, {len(all_specs)} specs indexadas")

        except Exception as e:
            logger.error(f"Error en entrenamiento Spec-Driven: {e}")

    def _process_optimized_query(self, optimized_query):
        """Procesa consulta usando Spec-Driven Development + optimizaciones"""
        query = optimized_query.get('original_query', '')

        # Intentar búsqueda Spec-Driven primero
        spec_results = self.spec_indexer.search_specs(query, max_results=3)

        if spec_results:
            # Formatear resultados de specs
            return self._format_spec_results(spec_results)

        # Fallback a búsqueda fuzzy tradicional
        # Cargar documentos para acceso directo
        data = self._load_files()
        documents = data.get('documents', {})

        # Usar búsqueda fuzzy optimizada
        search_results = fuzzy_search.search(query, threshold=0.7)

        if not search_results:
            return "No se encontró contexto relevante para tu consulta."

        # Aplicar puntuación de relevancia avanzada
        scored_results = []
        for doc_id, fuzzy_score in search_results[:10]:  # Top 10 resultados
            # Obtener datos del documento
            doc_data = self._get_document_data(doc_id, documents)
            if doc_data:
                relevance_score = relevance_scorer.calculate_relevance(query, doc_data)
                # Boost por feedback histórico ACE
                historical_boost = self._calculate_historical_boost(doc_data)
                combined_score = (fuzzy_score + relevance_score + historical_boost) / 3
                scored_results.append((doc_id, combined_score, doc_data))

        # Ordenar por score combinado
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Aplicar token budgeting
        selected_sections = []
        for doc_id, score, doc_data in scored_results[:3]:  # Top 3
            selected_sections.append({
                'content': doc_data['content'],
                'relevance': score,
                'section_id': doc_data.get('section_id', ''),
                'section_title': doc_data.get('section_title'),
                'tokens': doc_data.get('tokens') or token_budget.estimate_tokens(doc_data['content'])
            })

        # Asignar tokens disponibles
        allocated_sections = token_budget.allocate_tokens(selected_sections)

        # Formatear resultado final
        return self._format_optimized_result(allocated_sections)

    def _format_spec_results(self, spec_results):
        """Formatea resultados de specs para respuesta"""
        if not spec_results:
            return "No se encontraron especificaciones relevantes."

        results = []
        for spec in spec_results:
            spec_type_display = spec['spec_type'].replace('_', ' ').title()
            content = spec['content'][:500] + "..." if len(spec['content']) > 500 else spec['content']

            result = f"**{spec_type_display}** (de {spec['filename']}):\n\n{content}\n\n*Confianza: {spec['confidence']:.1%} | Relevancia: {spec.get('relevance_score', 0):.2f}*"
            results.append(result)

        return '\n\n---\n\n'.join(results)

    def _get_document_data(self, doc_id, documents=None):
        """Obtiene datos de documento del índice fuzzy"""
        if documents is None:
            data = self._load_files()
            documents = data.get('documents', {})

        doc = documents.get(doc_id)
        if doc:
            return doc

        # Último recurso: solicitarlo al índice global
        return fuzzy_search.get_document(doc_id)

    def _format_optimized_result(self, allocated_sections):
        """Formatea resultado optimizado"""
        if not allocated_sections:
            return "No se encontró contexto relevante para tu consulta."

        results = []
        for section in allocated_sections:
            raw_title = section.get('section_title') or section.get('section_id', '').replace('_', ' ').title()
            content = section.get('content', '')

            if section.get('content_truncated', False):
                content += "\n\n*(Contenido truncado por límite de tokens)*"

            results.append(f"**{raw_title}:**\n\n{content}")

        return '\n\n---\n\n'.join(results)

    def _send_json_response(self, status_code, data):
        """Envía respuesta JSON"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(json_data.encode('utf-8'))

    def _send_error(self, status_code, message):
        """Envía respuesta de error"""
        self._send_json_response(status_code, {"error": message})

    def log_message(self, format, *args):
        """Override para usar nuestro logger"""
        logger.info(f"HTTP {self.address_string()} - {format % args}")


def run_server(port=8081):
    """Ejecuta el servidor con optimizaciones"""
    # Iniciar monitoreo de recursos
    resource_monitor.start_monitoring()

    server_address = ('', port)
    httpd = HTTPServer(server_address, ContextQueryHandler)

    logger.info(f"Servidor MCP Context Query Optimizado iniciando en puerto {port}")
    logger.info("Endpoints disponibles:")
    logger.info("   GET  /manifest         - Manifest MCP")
    logger.info("   GET  /health           - Health check con métricas")
    logger.info("   POST /tools/context_query - Consulta de contexto optimizada")
    logger.info("   POST /tools/feedback     - Feedback de consultas para ACE")
    logger.info("   POST /tools/analyze_feedback - Análisis y mejora ACE")
    logger.info("   POST /tools/train_system - Entrenamiento Spec-Driven")
    logger.info("")
    logger.info("Optimizaciones activas:")
    logger.info("   • Cache multinivel (L1/L2/Disk)")
    logger.info("   • Chunking semántico avanzado")
    logger.info("   • Búsqueda fuzzy optimizada")
    logger.info("   • Token budgeting inteligente")
    logger.info("   • Monitoreo de recursos")
    logger.info("   • Puntuación de relevancia avanzada")
    logger.info("")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")
        resource_monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Error en servidor: {e}")
        resource_monitor.stop_monitoring()
    finally:
        httpd.shutdown()


if __name__ == '__main__':

    port = 8081
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error("Puerto debe ser un número")
            sys.exit(1)

    run_server(port)
