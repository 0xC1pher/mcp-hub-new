#!/usr/bin/env python3
"""
MCP v3 Enhanced - Integra técnicas de Grok y memoria persistente avanzada
Mantiene todas las funcionalidades v2 + nuevas técnicas avanzadas
"""

import json
import time
import hashlib
import threading
import sqlite3
import pickle
# import numpy as np  # No usado actualmente
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class GrokInspiredReasoning:
    """Sistema de razonamiento inspirado en Grok para análisis profundo"""
    
    def __init__(self):
        self.reasoning_patterns = {}
        self.concept_graph = defaultdict(set)
        self.abstraction_levels = {}
        self.insight_cache = {}
        
    def analyze_deep_patterns(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        """Análisis profundo de patrones usando técnicas Grok"""
        
        # 1. Extracción de conceptos clave
        concepts = self._extract_concepts(query, context)
        
        # 2. Construcción de grafo conceptual
        concept_relations = self._build_concept_graph(concepts, context)
        
        # 3. Análisis de abstracciones múltiples
        abstractions = self._multi_level_abstraction(concepts, concept_relations)
        
        # 4. Generación de insights
        insights = self._generate_insights(abstractions, query)
        
        return {
            'concepts': concepts,
            'relations': concept_relations,
            'abstractions': abstractions,
            'insights': insights,
            'confidence': self._calculate_confidence(insights)
        }
    
    def _extract_concepts(self, query: str, context: List[Dict]) -> List[str]:
        """Extrae conceptos clave usando análisis semántico"""
        concepts = set()
        
        # Conceptos del query
        query_words = query.lower().split()
        medical_concepts = ['paciente', 'medico', 'cita', 'historia', 'diagnostico', 'tratamiento']
        
        for word in query_words:
            if len(word) > 3 and (word in medical_concepts or word.endswith('cion')):
                concepts.add(word)
        
        # Conceptos del contexto
        for item in context:
            content = item.get('content', '').lower()
            for concept in medical_concepts:
                if concept in content:
                    concepts.add(concept)
        
        return list(concepts)
    
    def _build_concept_graph(self, concepts: List[str], context: List[Dict]) -> Dict[str, Set[str]]:
        """Construye grafo de relaciones conceptuales"""
        relations = defaultdict(set)
        
        for item in context:
            content = item.get('content', '').lower()
            present_concepts = [c for c in concepts if c in content]
            
            # Crear relaciones entre conceptos co-ocurrentes
            for i, concept1 in enumerate(present_concepts):
                for concept2 in present_concepts[i+1:]:
                    relations[concept1].add(concept2)
                    relations[concept2].add(concept1)
        
        return dict(relations)
    
    def _multi_level_abstraction(self, concepts: List[str], relations: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Análisis de abstracciones múltiples"""
        abstractions = {
            'level_1': concepts,  # Conceptos directos
            'level_2': [],        # Agrupaciones conceptuales
            'level_3': []         # Meta-conceptos
        }
        
        # Nivel 2: Agrupar conceptos relacionados
        concept_groups = []
        processed = set()
        
        for concept in concepts:
            if concept in processed:
                continue
                
            group = {concept}
            related = relations.get(concept, set())
            group.update(related)
            concept_groups.append(list(group))
            processed.update(group)
        
        abstractions['level_2'] = concept_groups
        
        # Nivel 3: Meta-conceptos médicos
        medical_meta = []
        if any(c in ['paciente', 'medico', 'cita'] for c in concepts):
            medical_meta.append('flujo_atencion_medica')
        if any(c in ['diagnostico', 'tratamiento', 'historia'] for c in concepts):
            medical_meta.append('proceso_clinico')
        
        abstractions['level_3'] = medical_meta
        
        return abstractions
    
    def _generate_insights(self, abstractions: Dict[str, Any], query: str) -> List[str]:
        """Genera insights basados en abstracciones"""
        insights = []
        
        # Insights basados en conceptos nivel 1
        concepts = abstractions['level_1']
        if 'paciente' in concepts and 'medico' in concepts:
            insights.append("Interacción médico-paciente identificada")
        
        if 'cita' in concepts and 'historia' in concepts:
            insights.append("Flujo de documentación clínica detectado")
        
        # Insights basados en agrupaciones nivel 2
        groups = abstractions['level_2']
        if len(groups) > 1:
            insights.append(f"Múltiples dominios conceptuales ({len(groups)}) identificados")
        
        # Insights basados en meta-conceptos nivel 3
        meta_concepts = abstractions['level_3']
        for meta in meta_concepts:
            if meta == 'flujo_atencion_medica':
                insights.append("Optimización del flujo de atención médica recomendada")
            elif meta == 'proceso_clinico':
                insights.append("Mejoras en proceso clínico sugeridas")
        
        # Insight por defecto si no hay otros
        if not insights:
            insights.append("Análisis conceptual básico completado")
        
        return insights
    
    def _calculate_confidence(self, insights: List[str]) -> float:
        """Calcula confianza del análisis"""
        base_confidence = 0.5
        
        # Aumentar confianza por número de insights
        insight_bonus = min(len(insights) * 0.1, 0.3)
        
        # Aumentar confianza por insights específicos
        specific_bonus = 0.0
        for insight in insights:
            if 'flujo' in insight.lower() or 'proceso' in insight.lower():
                specific_bonus += 0.1
        
        return min(base_confidence + insight_bonus + specific_bonus, 0.95)

class AdvancedMemoryPersistence:
    """Sistema de memoria persistente avanzada con técnicas de retención"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.memory_layers = {
            'episodic': {},      # Memoria episódica (eventos específicos)
            'semantic': {},      # Memoria semántica (conocimiento general)
            'procedural': {},    # Memoria procedimental (cómo hacer cosas)
            'working': {}        # Memoria de trabajo (temporal)
        }
        
        self.consolidation_queue = deque()
        self.forgetting_curve = {}
        self.importance_weights = {}
        
        self._init_persistent_storage()
        self._start_consolidation_process()
    
    def _init_persistent_storage(self):
        """Inicializa almacenamiento persistente SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla para memoria episódica
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY,
                event_id TEXT UNIQUE,
                content BLOB,
                timestamp REAL,
                importance REAL,
                access_count INTEGER DEFAULT 0,
                last_access REAL,
                consolidation_level INTEGER DEFAULT 0
            )
        ''')
        
        # Tabla para memoria semántica
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY,
                concept TEXT,
                knowledge BLOB,
                confidence REAL,
                sources TEXT,
                created_at REAL,
                updated_at REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_episodic_memory(self, event_id: str, content: Dict, importance: float = 0.5):
        """Almacena memoria episódica con importancia"""
        
        # Calcular importancia automática si no se proporciona
        if importance == 0.5:
            importance = self._calculate_importance(content)
        
        memory_item = {
            'content': content,
            'timestamp': time.time(),
            'importance': importance,
            'access_count': 0,
            'consolidation_level': 0
        }
        
        # Almacenar en memoria de trabajo primero
        self.memory_layers['working'][event_id] = memory_item
        
        # Agregar a cola de consolidación
        self.consolidation_queue.append((event_id, memory_item))
        
        logger.info(f"💾 Memoria episódica almacenada: {event_id} (importancia: {importance:.2f})")
    
    def _calculate_importance(self, content: Dict) -> float:
        """Calcula importancia automática del contenido"""
        importance = 0.5  # Base
        
        # Aumentar importancia por palabras clave médicas
        content_str = str(content).lower()
        medical_keywords = ['paciente', 'medico', 'diagnostico', 'tratamiento', 'emergencia']
        
        for keyword in medical_keywords:
            if keyword in content_str:
                importance += 0.1
        
        # Aumentar importancia por análisis o insights
        if 'analysis' in content or 'insights' in content:
            importance += 0.2
        
        return min(importance, 1.0)
    
    def _start_consolidation_process(self):
        """Inicia proceso de consolidación en background"""
        def consolidation_worker():
            while True:
                try:
                    if self.consolidation_queue:
                        event_id, memory_item = self.consolidation_queue.popleft()
                        self._consolidate_memory(event_id, memory_item)
                    time.sleep(1)  # Pausa entre consolidaciones
                except Exception as e:
                    logger.error(f"Error en consolidación: {e}")
        
        consolidation_thread = threading.Thread(target=consolidation_worker, daemon=True)
        consolidation_thread.start()
    
    def _consolidate_memory(self, event_id: str, memory_item: Dict):
        """Consolida memoria de working a episodic storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serializar contenido
            content_blob = pickle.dumps(memory_item['content'])
            
            cursor.execute('''
                INSERT OR REPLACE INTO episodic_memory 
                (event_id, content, timestamp, importance, access_count, last_access, consolidation_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                content_blob,
                memory_item['timestamp'],
                memory_item['importance'],
                memory_item['access_count'],
                memory_item.get('last_access', time.time()),
                1  # Nivel de consolidación inicial
            ))
            
            conn.commit()
            conn.close()
            
            # Mover de working a episodic
            if event_id in self.memory_layers['working']:
                self.memory_layers['episodic'][event_id] = self.memory_layers['working'].pop(event_id)
            
            logger.debug(f"💾 Memoria consolidada: {event_id}")
            
        except Exception as e:
            logger.error(f"Error consolidando memoria {event_id}: {e}")

class MCPv3EnhancedServer:
    """Servidor MCP v3 con técnicas Grok y memoria persistente avanzada"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.version = "3.0.0"
        
        # Importar y mantener todas las técnicas v2
        try:
            from advanced_techniques import UnifiedAdvancedSystem
            from context_indexing_system import ContextIndexingSystem
            from unified_mcp_server import UnifiedMCPServer
            
            # Heredar funcionalidades v2
            self.v2_server = UnifiedMCPServer(str(self.project_root))
            logger.info("✅ Funcionalidades MCP v2 heredadas")
            
        except ImportError as e:
            logger.error(f"❌ Error importando v2: {e}")
            self.v2_server = None
        
        # Inicializar nuevas técnicas v3
        self.grok_reasoning = GrokInspiredReasoning()
        
        # Sistema de memoria persistente avanzada
        memory_db_path = self.project_root / "mcp-hub" / "data" / "cache" / "mcp_v3_memory.db"
        memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.advanced_memory = AdvancedMemoryPersistence(str(memory_db_path))
        
        # Nuevas métricas v3
        self.v3_metrics = {
            'grok_analyses': 0,
            'deep_insights': 0,
            'memory_consolidations': 0,
            'reasoning_patterns': 0
        }
        
        logger.info("🚀 MCP v3 Enhanced iniciado")
        logger.info(f"📊 Versión: {self.version}")
        logger.info("🧠 Técnicas Grok activadas")
        logger.info("💾 Memoria persistente avanzada activada")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests con análisis Grok y memoria avanzada"""
        
        # Primero procesar con v2 (compatibilidad)
        if self.v2_server:
            v2_response = self.v2_server.handle_request(request)
        else:
            v2_response = {'error': 'v2 no disponible'}
        
        # Análisis Grok adicional para queries complejas
        if request.get('method') == 'tools/call':
            tool_name = request.get('params', {}).get('name', '')
            
            if tool_name == 'grok_analysis':
                return self._grok_analysis(request.get('params', {}).get('arguments', {}))
            elif tool_name == 'advanced_memory_query':
                return self._advanced_memory_query(request.get('params', {}).get('arguments', {}))
            elif tool_name in ['context_query', 'code_review']:
                # Enhancer respuesta v2 con análisis Grok
                return self._enhance_with_grok(v2_response, request)
        
        return v2_response
    
    def _grok_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis profundo usando técnicas Grok"""
        query = args.get('query', '')
        context_limit = args.get('context_limit', 10)
        
        # Obtener contexto relevante
        if self.v2_server and hasattr(self.v2_server, 'context_indexer'):
            context_results = self.v2_server.context_indexer.search_similar_contexts(query)[:context_limit]
        else:
            context_results = []
        
        # Análisis Grok
        grok_analysis = self.grok_reasoning.analyze_deep_patterns(query, context_results)
        
        # Almacenar en memoria episódica
        event_id = f"grok_analysis_{int(time.time())}"
        self.advanced_memory.store_episodic_memory(
            event_id, 
            {'query': query, 'analysis': grok_analysis},
            importance=grok_analysis['confidence']
        )
        
        self.v3_metrics['grok_analyses'] += 1
        
        # Formatear respuesta
        response = f"""
🧠 **ANÁLISIS GROK PROFUNDO**

🔍 **Query**: {query}

🎯 **Conceptos Identificados**: {', '.join(grok_analysis['concepts'])}

🕸️ **Relaciones Conceptuales**:
{self._format_concept_relations(grok_analysis['relations'])}

🔬 **Niveles de Abstracción**:
{self._format_abstractions(grok_analysis['abstractions'])}

💡 **Insights Generados**:
{self._format_insights(grok_analysis['insights'])}

📊 **Confianza**: {grok_analysis['confidence']:.2f}

💾 **Memoria**: Análisis almacenado como {event_id}
        """
        
        return {'content': [{'type': 'text', 'text': response}]}
    
    def _enhance_with_grok(self, v2_response: Dict, request: Dict) -> Dict[str, Any]:
        """Mejora respuesta v2 con análisis Grok"""
        
        if 'error' in v2_response:
            return v2_response
        
        # Extraer query del request
        args = request.get('params', {}).get('arguments', {})
        query = args.get('query', args.get('task_description', ''))
        
        if not query:
            return v2_response
        
        # Análisis Grok rápido
        try:
            grok_analysis = self.grok_reasoning.analyze_deep_patterns(query, [])
            
            # Agregar insights Grok a la respuesta
            original_content = v2_response.get('content', [{}])[0].get('text', '')
            
            enhanced_content = f"""{original_content}

---
🧠 **ANÁLISIS GROK ADICIONAL**:
💡 **Insights**: {', '.join(grok_analysis['insights'][:3])}
🎯 **Conceptos clave**: {', '.join(grok_analysis['concepts'][:5])}
📊 **Confianza**: {grok_analysis['confidence']:.2f}
            """
            
            v2_response['content'][0]['text'] = enhanced_content
            self.v3_metrics['deep_insights'] += 1
            
        except Exception as e:
            logger.warning(f"Error en enhancement Grok: {e}")
        
        return v2_response
    
    def _advanced_memory_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Consulta avanzada de memoria persistente"""
        query = args.get('query', '')
        memory_type = args.get('memory_type', 'episodic')
        limit = args.get('limit', 5)
        
        try:
            # Buscar en memoria persistente
            memories = self._search_persistent_memory(query, memory_type, limit)
            
            if not memories:
                return {
                    'content': [{'type': 'text', 'text': f"🔍 No se encontraron memorias para: '{query}' en tipo '{memory_type}'"}]
                }
            
            # Formatear respuesta
            response = f"""
💾 **CONSULTA DE MEMORIA AVANZADA**

🔍 **Query**: {query}
🧠 **Tipo de memoria**: {memory_type}
📊 **Resultados encontrados**: {len(memories)}

"""
            
            for i, memory in enumerate(memories, 1):
                timestamp = datetime.fromtimestamp(memory.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M')
                importance = memory.get('importance', 0)
                access_count = memory.get('access_count', 0)
                
                response += f"""
**{i}. Memoria #{memory.get('id', 'N/A')}**
📅 **Fecha**: {timestamp}
⭐ **Importancia**: {importance:.2f}
👁️ **Accesos**: {access_count}
📝 **Contenido**: {str(memory.get('content', ''))[:200]}...

"""
            
            self.v3_metrics['memory_consolidations'] += 1
            
            return {'content': [{'type': 'text', 'text': response}]}
            
        except Exception as e:
            logger.error(f"Error en consulta de memoria: {e}")
            return {'error': f'Error en consulta de memoria: {str(e)}'}
    
    def _search_persistent_memory(self, query: str, memory_type: str, limit: int) -> List[Dict]:
        """Busca en memoria persistente SQLite"""
        try:
            import sqlite3
            
            db_path = self.project_root / "mcp-hub" / "data" / "cache" / "mcp_v3_memory.db"
            if not db_path.exists():
                return []
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            if memory_type == 'episodic':
                cursor.execute('''
                    SELECT event_id, content, timestamp, importance, access_count
                    FROM episodic_memory 
                    WHERE content LIKE ? 
                    ORDER BY importance DESC, timestamp DESC 
                    LIMIT ?
                ''', (f'%{query}%', limit))
                
                results = []
                for row in cursor.fetchall():
                    try:
                        content = pickle.loads(row[1]) if row[1] else {}
                    except:
                        content = {'raw': str(row[1])}
                    
                    results.append({
                        'id': row[0],
                        'content': content,
                        'timestamp': row[2],
                        'importance': row[3],
                        'access_count': row[4]
                    })
                
                conn.close()
                return results
            
            conn.close()
            return []
            
        except Exception as e:
            logger.error(f"Error buscando en memoria persistente: {e}")
            return []
    
    def _format_concept_relations(self, relations: Dict[str, Set[str]]) -> str:
        """Formatea relaciones conceptuales para visualización"""
        if not relations:
            return "No se encontraron relaciones conceptuales"
        
        formatted = ""
        for concept, related in relations.items():
            if related:
                formatted += f"• **{concept}** → {', '.join(related)}\n"
        
        return formatted if formatted else "Relaciones conceptuales básicas"
    
    def _format_abstractions(self, abstractions: Dict[str, Any]) -> str:
        """Formatea abstracciones para visualización"""
        formatted = ""
        
        for level, content in abstractions.items():
            if content:
                if level == 'level_1':
                    formatted += f"**Nivel 1 (Conceptos)**: {', '.join(content)}\n"
                elif level == 'level_2':
                    formatted += f"**Nivel 2 (Agrupaciones)**: {len(content)} grupos identificados\n"
                elif level == 'level_3':
                    formatted += f"**Nivel 3 (Meta-conceptos)**: {', '.join(content)}\n"
        
        return formatted if formatted else "Abstracciones básicas identificadas"
    
    def _format_insights(self, insights: List[str]) -> str:
        """Formatea insights para visualización"""
        if not insights:
            return "No se generaron insights específicos"
        
        formatted = ""
        for i, insight in enumerate(insights, 1):
            formatted += f"{i}. {insight}\n"
        
        return formatted
    
    def get_v3_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas específicas de v3"""
        
        v2_stats = {}
        if self.v2_server and hasattr(self.v2_server, '_system_stats'):
            try:
                v2_stats = self.v2_server._system_stats().get('content', [{}])[0].get('text', '')
            except:
                v2_stats = "v2 stats no disponibles"
        
        return {
            'version': self.version,
            'v2_compatibility': self.v2_server is not None,
            'v3_metrics': self.v3_metrics,
            'grok_patterns': len(self.grok_reasoning.reasoning_patterns),
            'memory_layers': {k: len(v) for k, v in self.advanced_memory.memory_layers.items()},
            'v2_inherited_stats': v2_stats
        }

# Instancia global
mcp_v3_server = None

def get_mcp_v3_server(project_root: str = None) -> MCPv3EnhancedServer:
    """Obtiene instancia singleton del servidor v3"""
    global mcp_v3_server
    if mcp_v3_server is None:
        mcp_v3_server = MCPv3EnhancedServer(project_root)
    return mcp_v3_server
