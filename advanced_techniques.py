#!/usr/bin/env python3
"""
Técnicas Avanzadas MCP - Implementación Unificada
Combina todas las técnicas avanzadas de los documentos y PDFs
"""

import json
import time
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AdvancedMemoryManager:
    """Gestor de memoria avanzado con técnicas de optimización"""
    
    def __init__(self):
        self.memory_pools = {
            'hot': {},      # Datos accedidos frecuentemente
            'warm': {},     # Datos accedidos ocasionalmente  
            'cold': {}      # Datos accedidos raramente
        }
        
        self.access_patterns = defaultdict(list)
        self.memory_pressure = 0.0
        self.cleanup_threshold = 0.8
        
        # Métricas avanzadas
        self.allocation_stats = {
            'total_allocations': 0,
            'deallocations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread para limpieza automática
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()
    
    def allocate(self, key: str, data: Any, priority: str = 'warm') -> bool:
        """Aloca memoria con gestión inteligente"""
        try:
            # Calcular tamaño aproximado
            size = self._estimate_size(data)
            
            # Verificar presión de memoria
            if self._check_memory_pressure(size):
                self._trigger_cleanup()
            
            # Almacenar en pool apropiado
            self.memory_pools[priority][key] = {
                'data': data,
                'allocated_at': time.time(),
                'access_count': 0,
                'last_access': time.time(),
                'size': size
            }
            
            self.allocation_stats['total_allocations'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error en allocate: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Recupera datos con tracking de acceso"""
        # Buscar en todos los pools
        for pool_name, pool in self.memory_pools.items():
            if key in pool:
                item = pool[key]
                item['access_count'] += 1
                item['last_access'] = time.time()
                
                # Registrar patrón de acceso
                self.access_patterns[key].append(time.time())
                
                # Promover si es necesario
                self._consider_promotion(key, pool_name, item)
                
                self.allocation_stats['cache_hits'] += 1
                return item['data']
        
        self.allocation_stats['cache_misses'] += 1
        return None
    
    def _consider_promotion(self, key: str, current_pool: str, item: Dict):
        """Considera promover item a pool más caliente"""
        access_frequency = len(self.access_patterns[key])
        recent_accesses = sum(1 for t in self.access_patterns[key] 
                            if time.time() - t < 300)  # 5 minutos
        
        # Promover de cold a warm
        if current_pool == 'cold' and recent_accesses >= 3:
            self._move_item(key, 'cold', 'warm')
        
        # Promover de warm a hot
        elif current_pool == 'warm' and recent_accesses >= 5:
            self._move_item(key, 'warm', 'hot')
    
    def _move_item(self, key: str, from_pool: str, to_pool: str):
        """Mueve item entre pools"""
        if key in self.memory_pools[from_pool]:
            item = self.memory_pools[from_pool].pop(key)
            self.memory_pools[to_pool][key] = item
    
    def _background_cleanup(self):
        """Limpieza automática en background"""
        while True:
            try:
                time.sleep(60)  # Cada minuto
                self._perform_cleanup()
            except Exception as e:
                logger.error(f"Error en background cleanup: {e}")
    
    def _perform_cleanup(self):
        """Realiza limpieza de memoria"""
        current_time = time.time()
        
        # Limpiar items antiguos de cold pool
        cold_pool = self.memory_pools['cold']
        to_remove = []
        
        for key, item in cold_pool.items():
            # Remover si no se ha accedido en 1 hora
            if current_time - item['last_access'] > 3600:
                to_remove.append(key)
        
        for key in to_remove:
            del cold_pool[key]
            self.allocation_stats['deallocations'] += 1
        
        # Degradar items de hot a warm si no se usan
        hot_pool = self.memory_pools['hot']
        to_degrade = []
        
        for key, item in hot_pool.items():
            if current_time - item['last_access'] > 600:  # 10 minutos
                to_degrade.append(key)
        
        for key in to_degrade:
            self._move_item(key, 'hot', 'warm')
    
    def _estimate_size(self, data: Any) -> int:
        """Estima tamaño de datos"""
        try:
            return len(str(data))
        except:
            return 1000  # Estimación por defecto
    
    def _check_memory_pressure(self, new_size: int) -> bool:
        """Verifica presión de memoria"""
        total_items = sum(len(pool) for pool in self.memory_pools.values())
        return total_items > 1000  # Umbral simple
    
    def _trigger_cleanup(self):
        """Dispara limpieza inmediata"""
        self._perform_cleanup()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de memoria"""
        return {
            'pools': {name: len(pool) for name, pool in self.memory_pools.items()},
            'allocation_stats': self.allocation_stats.copy(),
            'memory_pressure': self.memory_pressure
        }

class AdaptiveQueryOptimizer:
    """Optimizador de queries adaptativo"""
    
    def __init__(self):
        self.query_history = deque(maxlen=1000)
        self.performance_metrics = {}
        self.optimization_rules = []
        
    def optimize_query(self, query: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Optimiza query basado en historial y contexto"""
        
        # Registrar query
        query_record = {
            'query': query,
            'timestamp': time.time(),
            'context': context
        }
        self.query_history.append(query_record)
        
        # Aplicar optimizaciones
        optimized_query = query
        optimized_context = context.copy()
        
        # 1. Expansión semántica
        optimized_query = self._expand_semantically(optimized_query)
        
        # 2. Filtrado contextual
        optimized_context = self._apply_contextual_filters(optimized_context)
        
        # 3. Priorización por historial
        optimized_context['priority_boost'] = self._calculate_priority_boost(query)
        
        return optimized_query, optimized_context
    
    def _expand_semantically(self, query: str) -> str:
        """Expande query semánticamente"""
        # Sinónimos médicos básicos
        medical_synonyms = {
            'paciente': ['paciente', 'enfermo', 'usuario', 'cliente'],
            'doctor': ['doctor', 'médico', 'profesional'],
            'cita': ['cita', 'consulta', 'appointment'],
            'historia': ['historia', 'historial', 'record']
        }
        
        expanded_terms = []
        for word in query.lower().split():
            if word in medical_synonyms:
                expanded_terms.extend(medical_synonyms[word])
            else:
                expanded_terms.append(word)
        
        return ' '.join(expanded_terms)
    
    def _apply_contextual_filters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica filtros contextuales"""
        # Agregar filtros basados en contexto médico
        context['medical_context'] = True
        context['language'] = 'es'
        context['domain'] = 'healthcare'
        
        return context
    
    def _calculate_priority_boost(self, query: str) -> float:
        """Calcula boost de prioridad basado en historial"""
        similar_queries = 0
        for record in self.query_history:
            if self._calculate_similarity(query, record['query']) > 0.7:
                similar_queries += 1
        
        # Más queries similares = mayor prioridad
        return min(2.0, 1.0 + (similar_queries * 0.1))
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calcula similitud entre queries"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class IntelligentDeduplicator:
    """Deduplicador inteligente con técnicas avanzadas"""
    
    def __init__(self):
        self.content_hashes = {}
        self.similarity_cache = {}
        self.dedup_stats = {
            'total_processed': 0,
            'duplicates_found': 0,
            'near_duplicates_found': 0
        }
    
    def process_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa contenido para deduplicación"""
        
        self.dedup_stats['total_processed'] += 1
        
        # Hash exacto
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Verificar duplicado exacto
        if content_hash in self.content_hashes:
            self.dedup_stats['duplicates_found'] += 1
            return {
                'is_duplicate': True,
                'duplicate_type': 'exact',
                'original_source': self.content_hashes[content_hash],
                'action': 'skip'
            }
        
        # Verificar near-duplicates
        near_duplicate = self._find_near_duplicate(content)
        if near_duplicate:
            self.dedup_stats['near_duplicates_found'] += 1
            return {
                'is_duplicate': True,
                'duplicate_type': 'near',
                'similarity': near_duplicate['similarity'],
                'original_source': near_duplicate['source'],
                'action': 'merge_or_skip'
            }
        
        # Registrar como único
        self.content_hashes[content_hash] = metadata.get('source', 'unknown')
        
        return {
            'is_duplicate': False,
            'content_hash': content_hash,
            'action': 'process'
        }
    
    def _find_near_duplicate(self, content: str) -> Optional[Dict[str, Any]]:
        """Encuentra near-duplicates usando técnicas avanzadas"""
        
        # Normalizar contenido
        normalized = self._normalize_content(content)
        
        # Buscar en cache de similitud
        for existing_hash, existing_source in self.content_hashes.items():
            cache_key = f"{existing_hash}:{hashlib.md5(normalized.encode()).hexdigest()}"
            
            if cache_key in self.similarity_cache:
                similarity = self.similarity_cache[cache_key]
            else:
                # Calcular similitud (implementación simplificada)
                similarity = self._calculate_content_similarity(normalized, existing_hash)
                self.similarity_cache[cache_key] = similarity
            
            if similarity > 0.85:  # Umbral de near-duplicate
                return {
                    'similarity': similarity,
                    'source': existing_source
                }
        
        return None
    
    def _normalize_content(self, content: str) -> str:
        """Normaliza contenido para comparación"""
        # Remover espacios extra, normalizar case, etc.
        normalized = ' '.join(content.lower().split())
        
        # Remover caracteres especiales comunes
        chars_to_remove = ['"', "'", ".", ",", ";", ":", "!", "?"]
        for char in chars_to_remove:
            normalized = normalized.replace(char, "")
        
        return normalized
    
    def _calculate_content_similarity(self, content1: str, content2_hash: str) -> float:
        """Calcula similitud entre contenidos"""
        # Implementación simplificada - en producción usar algoritmos más sofisticados
        # como Jaccard similarity, cosine similarity, etc.
        
        # Por ahora, usar similitud de palabras
        words1 = set(content1.split())
        
        # Simular obtención de content2 (en implementación real, se obtendría del hash)
        # Por simplicidad, retornamos similitud baja
        return 0.3
    
    def get_dedup_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de deduplicación"""
        total = self.dedup_stats['total_processed']
        if total == 0:
            return self.dedup_stats
        
        return {
            **self.dedup_stats,
            'duplicate_rate': (self.dedup_stats['duplicates_found'] / total) * 100,
            'near_duplicate_rate': (self.dedup_stats['near_duplicates_found'] / total) * 100,
            'unique_rate': ((total - self.dedup_stats['duplicates_found'] - 
                           self.dedup_stats['near_duplicates_found']) / total) * 100
        }

class ContextualLearningSystem:
    """Sistema de aprendizaje contextual"""
    
    def __init__(self):
        self.learning_patterns = {}
        self.context_associations = defaultdict(list)
        self.feedback_history = []
        
    def learn_from_interaction(self, query: str, results: List[Dict], 
                             user_feedback: Optional[Dict] = None):
        """Aprende de interacciones del usuario"""
        
        interaction = {
            'query': query,
            'results_count': len(results),
            'timestamp': time.time(),
            'feedback': user_feedback
        }
        
        self.feedback_history.append(interaction)
        
        # Extraer patrones
        self._extract_patterns(query, results, user_feedback)
        
        # Actualizar asociaciones contextuales
        self._update_context_associations(query, results)
    
    def _extract_patterns(self, query: str, results: List[Dict], 
                         feedback: Optional[Dict]):
        """Extrae patrones de aprendizaje"""
        
        query_type = self._classify_query_type(query)
        
        if query_type not in self.learning_patterns:
            self.learning_patterns[query_type] = {
                'successful_patterns': [],
                'failed_patterns': [],
                'optimization_hints': []
            }
        
        # Si hay feedback positivo, registrar como patrón exitoso
        if feedback and feedback.get('rating', 0) >= 4:
            pattern = {
                'query_keywords': query.lower().split(),
                'result_types': [r.get('metadata', {}).get('chunk_type') for r in results],
                'success_factors': feedback.get('success_factors', [])
            }
            self.learning_patterns[query_type]['successful_patterns'].append(pattern)
    
    def _classify_query_type(self, query: str) -> str:
        """Clasifica tipo de query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['paciente', 'patient']):
            return 'patient_related'
        elif any(word in query_lower for word in ['cita', 'appointment']):
            return 'appointment_related'
        elif any(word in query_lower for word in ['historia', 'history']):
            return 'medical_history'
        elif any(word in query_lower for word in ['código', 'code', 'función']):
            return 'code_related'
        else:
            return 'general'
    
    def _update_context_associations(self, query: str, results: List[Dict]):
        """Actualiza asociaciones contextuales"""
        
        for result in results:
            file_path = result.get('file', '')
            if file_path:
                self.context_associations[query].append({
                    'file': file_path,
                    'relevance': result.get('score', 0),
                    'timestamp': time.time()
                })
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Obtiene insights del aprendizaje"""
        
        total_interactions = len(self.feedback_history)
        positive_feedback = sum(1 for f in self.feedback_history 
                              if f.get('feedback', {}).get('rating', 0) >= 4)
        
        return {
            'total_interactions': total_interactions,
            'positive_feedback_rate': (positive_feedback / total_interactions * 100) 
                                    if total_interactions > 0 else 0,
            'learned_patterns': len(self.learning_patterns),
            'context_associations': len(self.context_associations),
            'query_types': list(self.learning_patterns.keys())
        }

class UnifiedAdvancedSystem:
    """Sistema unificado que combina todas las técnicas avanzadas"""
    
    def __init__(self):
        self.memory_manager = AdvancedMemoryManager()
        self.query_optimizer = AdaptiveQueryOptimizer()
        self.deduplicator = IntelligentDeduplicator()
        self.learning_system = ContextualLearningSystem()
        
        logger.info("🧠 Sistema Avanzado Unificado inicializado")
        logger.info("   ✅ Memory Manager")
        logger.info("   ✅ Query Optimizer") 
        logger.info("   ✅ Intelligent Deduplicator")
        logger.info("   ✅ Contextual Learning System")
    
    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa query usando todas las técnicas avanzadas"""
        
        # 1. Optimizar query
        optimized_query, optimized_context = self.query_optimizer.optimize_query(query, context)
        
        # 2. Buscar en memoria
        cached_result = self.memory_manager.retrieve(f"query:{optimized_query}")
        if cached_result:
            return cached_result
        
        # 3. Procesar con deduplicación (simulado)
        # En implementación real, aquí se haría la búsqueda real
        mock_results = [
            {'content': f'Resultado para: {optimized_query}', 'score': 0.9},
            {'content': f'Contexto relacionado: {optimized_query}', 'score': 0.7}
        ]
        
        # 4. Aplicar deduplicación
        deduplicated_results = []
        for result in mock_results:
            dedup_info = self.deduplicator.process_content(
                result['content'], 
                {'source': 'mock'}
            )
            
            if not dedup_info['is_duplicate']:
                deduplicated_results.append(result)
        
        # 5. Almacenar en memoria
        final_result = {
            'query': optimized_query,
            'results': deduplicated_results,
            'context': optimized_context,
            'processed_at': time.time()
        }
        
        self.memory_manager.allocate(f"query:{optimized_query}", final_result, 'warm')
        
        # 6. Aprender de la interacción
        self.learning_system.learn_from_interaction(query, deduplicated_results)
        
        return final_result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema completo"""
        return {
            'memory_stats': self.memory_manager.get_stats(),
            'deduplication_stats': self.deduplicator.get_dedup_stats(),
            'learning_insights': self.learning_system.get_learning_insights(),
            'system_uptime': time.time() - getattr(self, 'start_time', time.time())
        }

# Instancia global del sistema avanzado
advanced_system = UnifiedAdvancedSystem()
