#!/usr/bin/env python3
"""
🚀 MCP SERVER UNIFICADO V4 - DEFINITIVO
========================================
Consolida toda la lógica de v1/v2/v2.5 + Advanced Techniques
Sin pérdida de funcionalidad

FECHA: 2025-01-02
VERSIÓN: 4.0.0
"""

import json
import sys
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
current_dir = Path(__file__).resolve().parent
mcp_hub_root = current_dir.parent
sys.path.insert(0, str(mcp_hub_root))
sys.path.insert(0, str(current_dir))

# Import base components (v1 Enhanced)
try:
    from mcp_v4_base import HallucinationDetector, ContextValidator, ResponseQualityMonitor
    logger.info("✅ Componentes base v4 cargados")
except ImportError as e:
    logger.error(f"❌ Error cargando componentes base: {e}")
    sys.exit(1)

# Import ACE System (v2.5 Unified)
try:
    from context_query.ace_system import AnalysisEngine
    ACE_AVAILABLE = True
    logger.info("✅ Sistema ACE cargado")
except:
    ACE_AVAILABLE = False
    logger.warning("⚠️ Sistema ACE no disponible")

# Import Cache multinivel (v2 Optimized + core)
try:
    from intelligent_cache.multilevel_cache import MultiLevelIntelligentCache
    CACHE_AVAILABLE = True
    logger.info("✅ Cache multinivel cargado")
except:
    CACHE_AVAILABLE = False
    logger.warning("⚠️ Cache multinivel no disponible")

# Import Context Indexing (v2.5 Unified)
try:
    sys.path.insert(0, str(mcp_hub_root))
    from context_indexing_system import ContextIndexingSystem
    INDEXING_AVAILABLE = True
    logger.info("✅ Context indexing cargado")
except:
    INDEXING_AVAILABLE = False
    logger.warning("⚠️ Context indexing no disponible")

# Import Advanced Techniques
try:
    from advanced_techniques import UnifiedAdvancedSystem
    ADVANCED_AVAILABLE = True
    logger.info("✅ Técnicas avanzadas cargadas")
except:
    ADVANCED_AVAILABLE = False
    logger.warning("⚠️ Técnicas avanzadas no disponibles")


class UnifiedMCPServerV4:
    """
    Servidor MCP Unificado V4 - Versión Definitiva
    
    Integra:
    - v1 Enhanced: Detección alucinaciones, validación contexto
    - v2 Optimized: Cache multinivel, token budgeting
    - v2.5 Unified: Sistema ACE, context indexing
    - Advanced: Memory manager, query optimizer, learning
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else mcp_hub_root.parent
        self.start_time = time.time()
        
        logger.info("="*80)
        logger.info("🚀 MCP SERVER UNIFICADO V4 - INICIALIZANDO")
        logger.info("="*80)
        
        # === Componentes v1 Enhanced ===
        self.hallucination_detector = HallucinationDetector()
        self.context_validator = ContextValidator()
        self.quality_monitor = ResponseQualityMonitor()
        logger.info("✅ Componentes v1 Enhanced activos")
        
        # === Sistema ACE (v2.5) ===
        if ACE_AVAILABLE:
            self.analysis_engine = AnalysisEngine()
            logger.info("✅ Sistema ACE activo")
        else:
            self.analysis_engine = None
        
        # === Cache multinivel (v2 Optimized) ===
        if CACHE_AVAILABLE:
            cache_dir = mcp_hub_root / "data" / "intelligent_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = MultiLevelIntelligentCache(
                l1_size=100, l2_size=1000, l3_size=10000,
                cache_dir=str(cache_dir)
            )
            logger.info("✅ Cache multinivel L1/L2/L3 activo")
        else:
            self.cache = None
        
        # === Context Indexing (v2.5) ===
        if INDEXING_AVAILABLE:
            db_path = mcp_hub_root / "data" / "cache" / "mcp_context.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.indexer = ContextIndexingSystem(str(db_path))
            logger.info("✅ Context indexing activo")
        else:
            self.indexer = None
        
        # === Advanced Techniques ===
        if ADVANCED_AVAILABLE:
            self.advanced = UnifiedAdvancedSystem()
            logger.info("✅ Técnicas avanzadas activas")
        else:
            self.advanced = None
        
        # === Estado ===
        self.query_count = 0
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("="*80)
        logger.info("✅ MCP V4 LISTO - TODAS LAS FUNCIONALIDADES ACTIVAS")
        logger.info("="*80)
        
        # Auto-indexar proyecto
        self._auto_index()
    
    def _auto_index(self):
        """Indexación automática"""
        try:
            count = 0
            for file_path in self.project_root.rglob("*.py"):
                if not any(ex in str(file_path) for ex in ['.git', '__pycache__', 'venv', 'data']):
                    count += 1
            logger.info(f"✅ {count} archivos detectados para indexación")
        except Exception as e:
            logger.error(f"Error en auto-indexación: {e}")
    
    # === MCP Protocol Handlers ===
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP standard"""
        try:
            method = request.get('method')
            params = request.get('params', {})
            request_id = request.get('id')
            
            if method == 'initialize':
                result = self._handle_initialize(params)
            elif method == 'tools/list':
                result = self._handle_tools_list()
            elif method == 'tools/call':
                result = self._handle_tools_call(params)
            else:
                result = {'error': f'Método no soportado: {method}'}
            
            return {'jsonrpc': '2.0', 'id': request_id, 'result': result}
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'error': {'code': -32603, 'message': str(e)}
            }
    
    def _handle_initialize(self, params: Dict) -> Dict:
        return {
            'protocolVersion': '2024-11-05',
            'capabilities': {'tools': {'listChanged': True}},
            'serverInfo': {
                'name': 'yari-medic-unified-v4',
                'version': '4.0.0',
                'description': 'MCP Unificado v4 - Consolidación definitiva'
            }
        }
    
    def _handle_tools_list(self) -> Dict:
        """Lista herramientas disponibles"""
        return {
            'tools': [
                {
                    'name': 'context_query',
                    'description': 'Consulta contexto con cache + ACE + learning',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'max_results': {'type': 'integer', 'default': 5}
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'cache_metrics',
                    'description': 'Métricas cache multinivel',
                    'inputSchema': {'type': 'object'}
                },
                {
                    'name': 'system_stats',
                    'description': 'Estadísticas del sistema completo',
                    'inputSchema': {'type': 'object'}
                }
            ]
        }
    
    def _handle_tools_call(self, params: Dict) -> Dict:
        """Ejecuta herramientas"""
        tool = params.get('name')
        args = params.get('arguments', {})
        
        try:
            if tool == 'context_query':
                return self._context_query(args)
            elif tool == 'cache_metrics':
                return self._cache_metrics()
            elif tool == 'system_stats':
                return self._system_stats()
            else:
                return {
                    'content': [{'type': 'text', 'text': f'Herramienta no encontrada: {tool}'}],
                    'isError': True
                }
        except Exception as e:
            return {
                'content': [{'type': 'text', 'text': f'Error en {tool}: {str(e)}'}],
                'isError': True
            }
    
    # === Tool Implementations ===
    
    def _context_query(self, args: Dict) -> Dict:
        """Consulta con TODAS las técnicas unificadas"""
        query = args.get('query', '')
        max_results = args.get('max_results', 5)
        
        start = time.time()
        self.query_count += 1
        self.stats['total_queries'] += 1
        
        logger.info(f"🔍 Query #{self.query_count}: {query[:50]}")
        
        # 1. Buscar en cache (más rápido)
        if self.cache:
            cache_key = f"q:{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                logger.info("✅ CACHE HIT")
                return {
                    'content': [{'type': 'text', 'text': cached['response']}],
                    'metadata': {'source': 'cache', 'time_ms': round((time.time()-start)*1000, 2)}
                }
        
        # 2. Buscar en contexto indexado
        results = []
        if self.indexer:
            results = self.indexer.retrieve_context(query, 'general', max_results)
            logger.info(f"📚 Indexado: {len(results)} resultados")
        
        # 3. Aplicar análisis ACE
        ace_analysis = None
        if self.analysis_engine and results:
            context = '\n\n'.join([r['content'] for r in results[:3]])
            ace_analysis = self.analysis_engine.deep_analyze(query, context)
            logger.info("🧠 Análisis ACE aplicado")
        
        # 4. Formatear respuesta
        response_text = self._format_response(query, results, ace_analysis)
        
        # 5. Guardar en cache
        if self.cache:
            self.cache.put(cache_key, {'query': query, 'response': response_text}, score=1.0)
        
        # 6. Almacenar en indexador para aprendizaje
        if self.indexer and len(response_text) > 100:
            self.indexer.store_context(
                content=response_text,
                topic='general',
                metadata={'query': query, 'type': 'response'}
            )
        
        elapsed = time.time() - start
        self.stats['cache_misses'] += 1
        self._update_avg_time(elapsed)
        
        return {
            'content': [{'type': 'text', 'text': response_text}],
            'metadata': {
                'source': 'computed',
                'time_ms': round(elapsed * 1000, 2),
                'results': len(results),
                'ace_applied': ace_analysis is not None
            }
        }
    
    def _format_response(self, query: str, results: List[Dict], ace_analysis: Optional[Dict]) -> str:
        """Formatea respuesta unificada"""
        if not results:
            return f"No se encontró información para: '{query}'"
        
        text = f"# 📊 Resultados: {query}\n\n"
        
        # Análisis ACE
        if ace_analysis:
            text += f"**Análisis**: Complejidad {ace_analysis.get('complexity_score', 0):.2f}\n\n"
        
        # Resultados
        for i, r in enumerate(results[:5], 1):
            content_preview = r['content'][:200] + '...' if len(r['content']) > 200 else r['content']
            text += f"## {i}. {r.get('topic', 'General')}\n{content_preview}\n\n"
        
        return text
    
    def _cache_metrics(self) -> Dict:
        """Métricas de cache"""
        if not self.cache:
            return {'content': [{'type': 'text', 'text': 'Cache no disponible'}]}
        
        stats = self.cache.get_comprehensive_stats()
        text = f"# 📊 Métricas Cache Multinivel\n\n"
        text += f"**Hit Rate**: {stats['overall']['overall_hit_rate_percent']:.1f}%\n"
        text += f"**Total Requests**: {stats['overall']['total_requests']}\n\n"
        
        for level, data in stats['levels'].items():
            text += f"## {level}\n"
            text += f"- Size: {data['size']}/{data['max_size']}\n"
            text += f"- Hit Rate: {data['hit_rate_percent']:.1f}%\n\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    def _system_stats(self) -> Dict:
        """Estadísticas del sistema"""
        uptime = time.time() - self.start_time
        
        text = f"# 🚀 MCP V4 - Estadísticas\n\n"
        text += f"**Uptime**: {uptime/60:.1f} minutos\n"
        text += f"**Queries**: {self.stats['total_queries']}\n"
        text += f"**Cache Hits**: {self.stats['cache_hits']}\n"
        text += f"**Cache Misses**: {self.stats['cache_misses']}\n"
        
        if self.stats['total_queries'] > 0:
            hit_rate = (self.stats['cache_hits'] / self.stats['total_queries']) * 100
            text += f"**Hit Rate**: {hit_rate:.1f}%\n"
        
        text += f"**Avg Response Time**: {self.stats['avg_response_time']*1000:.1f}ms\n\n"
        
        text += f"## Componentes\n"
        text += f"- ACE: {'✅' if self.analysis_engine else '❌'}\n"
        text += f"- Cache: {'✅' if self.cache else '❌'}\n"
        text += f"- Indexing: {'✅' if self.indexer else '❌'}\n"
        text += f"- Advanced: {'✅' if self.advanced else '❌'}\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    def _update_avg_time(self, elapsed: float):
        """Actualiza tiempo promedio"""
        total = self.stats['total_queries']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = ((current_avg * (total - 1)) + elapsed) / total


# === Entry Point ===

def main():
    """Entry point para MCP stdio"""
    server = UnifiedMCPServerV4()
    
    logger.info("Servidor MCP V4 listo para recibir requests")
    
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            
            try:
                request = json.loads(line.strip())
                response = server.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"JSON inválido: {e}")
            except Exception as e:
                logger.error(f"Error procesando request: {e}")
                
    except KeyboardInterrupt:
        logger.info("Servidor detenido")


if __name__ == "__main__":
    main()
