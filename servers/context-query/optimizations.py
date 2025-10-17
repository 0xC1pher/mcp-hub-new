#!/usr/bin/env python3
"""
Módulo de Optimizaciones Avanzadas para MCP Context Query Server
Implementa estrategias de optimización según OPTIMIZATION-STRATEGIES.md
"""

import time
import json
import threading
import psutil
import os
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import re
import math

logger = logging.getLogger(__name__)

class TokenBudgetManager:
    """Gestión inteligente de presupuesto de tokens"""

    def __init__(self, max_tokens: int = 4000, reserved_tokens: int = 500):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimación aproximada de tokens (1 token ≈ 4 caracteres en inglés)"""
        if not text:
            return 0
        # Estimación simple: ~4 caracteres por token
        return max(1, len(text) // 4)

    def prioritize_content(self, sections: List[Dict]) -> List[Dict]:
        """Prioriza secciones basado en relevancia"""
        prioritized = []
        for section in sections:
            priority = self.calculate_priority(section)
            token_count = self.estimate_tokens(section.get('content', ''))
            prioritized.append({
                **section,
                'priority': priority,
                'token_count': token_count
            })

        return sorted(prioritized, key=lambda x: x['priority'], reverse=True)

    def calculate_priority(self, section: Dict) -> float:
        """Calcula prioridad de una sección"""
        factors = {
            'relevance_score': section.get('relevance', 0),
            'recency': self.get_recency_score(section.get('last_updated')),
            'context_density': self.get_context_density(section.get('content', '')),
            'access_count': section.get('access_count', 0)
        }

        # Pesos para cada factor
        weights = {
            'relevance_score': 0.4,
            'recency': 0.2,
            'context_density': 0.3,
            'access_count': 0.1
        }

        return sum(score * weights[factor] for factor, score in factors.items())

    def get_recency_score(self, last_updated: Optional[float]) -> float:
        """Calcula score basado en recencia"""
        if not last_updated:
            return 0.5  # Score neutral

        days_since_update = (time.time() - last_updated) / (24 * 3600)
        # Decaimiento exponencial: más reciente = score más alto
        return math.exp(-days_since_update / 30)  # 30 días de vida media

    def get_context_density(self, content: str) -> float:
        """Calcula densidad de información del contexto"""
        if not content:
            return 0

        # Métricas de densidad
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        lists = len(re.findall(r'^[-*+]\s', content, re.MULTILINE))

        # Score basado en estructura y contenido
        density_score = min(1.0, (
            (word_count / 100) * 0.3 +  # Longitud
            (sentence_count / 10) * 0.2 +  # Estructura
            code_blocks * 0.3 +  # Código
            lists * 0.2  # Listas
        ))

        return density_score

    def allocate_tokens(self, sections: List[Dict]) -> List[Dict]:
        """Asigna tokens disponibles a secciones priorizadas"""
        prioritized = self.prioritize_content(sections)
        allocated_sections = []
        remaining_tokens = self.available_tokens

        for section in prioritized:
            if remaining_tokens <= 0:
                break

            allocated_tokens = min(section['token_count'], remaining_tokens)
            if allocated_tokens > 0:
                allocated_sections.append({
                    **section,
                    'allocated_tokens': allocated_tokens,
                    'content_truncated': allocated_tokens < section['token_count']
                })
                remaining_tokens -= allocated_tokens

        return allocated_sections


class SemanticChunker:
    """Chunking semántico avanzado"""

    def __init__(self, chunk_size: int = 512, overlap_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def semantic_chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Divide texto en chunks semánticos"""
        if metadata is None:
            metadata = {}

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Crear chunk actual
                chunk_content = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'tokens': current_tokens,
                    'metadata': {**metadata, 'chunk_index': len(chunks)},
                    'sentences': len(current_chunk)
                })

                # Mantener solapamiento
                overlap_sentences = self.get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = self.estimate_tokens(' '.join(overlap_sentences))

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Último chunk
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'tokens': current_tokens,
                'metadata': {**metadata, 'chunk_index': len(chunks)},
                'sentences': len(current_chunk)
            })

        return chunks

    def split_into_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones"""
        # Patrón mejorado para dividir oraciones
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())

        # Filtrar oraciones vacías y muy cortas
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Obtiene oraciones para solapamiento"""
        overlap_tokens = 0
        overlap_sentences = []

        # Tomar oraciones desde el final hasta alcanzar el overlap deseado
        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap_size:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def estimate_tokens(self, text: str) -> int:
        """Estimación simple de tokens"""
        return max(1, len(text) // 4)


class MultiLevelCache:
    """Cache multinivel (L1, L2, Disk)"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Configuración de niveles
        self.l1_cache = {}  # Memoria rápida
        self.l2_cache = {}  # Memoria media
        self.l1_max_size = 100
        self.l2_max_size = 1000
        self.disk_cache_index = os.path.join(cache_dir, "index.json")

        # Cargar índice de disco
        self.disk_index = self.load_disk_index()

        # Estadísticas
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'sets': 0
        }

    def get(self, key: str) -> Any:
        """Obtiene valor de cache con promoción automática"""
        # L1 Cache
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if time.time() < entry['expires']:
                self.stats['l1_hits'] += 1
                return entry['value']
            else:
                del self.l1_cache[key]

        # L2 Cache
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if time.time() < entry['expires']:
                self.stats['l2_hits'] += 1
                # Promover a L1
                self.promote_to_l1(key, entry['value'], entry['expires'] - time.time())
                return entry['value']
            else:
                del self.l2_cache[key]

        # Disk Cache
        disk_value = self.get_disk_cache(key)
        if disk_value:
            self.stats['disk_hits'] += 1
            # Promover a L2
            self.promote_to_l2(key, disk_value)
            return disk_value

        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600000) -> None:
        """Almacena valor en cache"""
        expires = time.time() + ttl

        # Siempre guardar en L1
        self.l1_cache[key] = {'value': value, 'expires': expires}
        self.stats['sets'] += 1

        # Mantener tamaño de caches
        self.maintain_cache_size()

    def promote_to_l1(self, key: str, value: Any, ttl: int = 3600000) -> None:
        """Promueve entrada a L1"""
        expires = time.time() + ttl
        self.l1_cache[key] = {'value': value, 'expires': expires}
        self.maintain_cache_size()

    def promote_to_l2(self, key: str, value: Any, ttl: int = 3600000) -> None:
        """Promueve entrada a L2"""
        expires = time.time() + ttl
        self.l2_cache[key] = {'value': value, 'expires': expires}

    def get_disk_cache(self, key: str) -> Any:
        """Obtiene valor del cache en disco"""
        if key not in self.disk_index:
            return None

        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entry = json.load(f)

                if time.time() < entry.get('expires', 0):
                    return entry['value']
                else:
                    # Expirado, limpiar
                    os.remove(cache_file)
                    del self.disk_index[key]
                    self.save_disk_index()
        except Exception as e:
            logger.warning(f"Error leyendo cache en disco para {key}: {e}")

        return None

    def save_to_disk(self, key: str, value: Any, ttl: int = 86400000) -> None:
        """Guarda en cache de disco"""
        expires = time.time() + ttl
        entry = {'value': value, 'expires': expires}

        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)

            self.disk_index[key] = expires
            self.save_disk_index()
        except Exception as e:
            logger.error(f"Error guardando en cache de disco: {e}")

    def load_disk_index(self) -> Dict:
        """Carga índice de cache de disco"""
        try:
            if os.path.exists(self.disk_cache_index):
                with open(self.disk_cache_index, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error cargando índice de disco: {e}")

        return {}

    def save_disk_index(self) -> None:
        """Guarda índice de cache de disco"""
        try:
            with open(self.disk_cache_index, 'w', encoding='utf-8') as f:
                json.dump(self.disk_index, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando índice de disco: {e}")

    def maintain_cache_size(self) -> None:
        """Mantiene tamaño máximo de caches"""
        # Limpiar expirados y mantener tamaño L1
        current_time = time.time()

        # L1 Cache
        expired_keys = [k for k, v in self.l1_cache.items() if current_time >= v['expires']]
        for key in expired_keys:
            del self.l1_cache[key]

        if len(self.l1_cache) > self.l1_max_size:
            # Remover entradas más antiguas (FIFO simple)
            keys_to_remove = list(self.l1_cache.keys())[:len(self.l1_cache) - self.l1_max_size]
            for key in keys_to_remove:
                del self.l1_cache[key]

        # L2 Cache
        expired_keys = [k for k, v in self.l2_cache.items() if current_time >= v['expires']]
        for key in expired_keys:
            del self.l2_cache[key]

        if len(self.l2_cache) > self.l2_max_size:
            keys_to_remove = list(self.l2_cache.keys())[:len(self.l2_cache) - self.l2_max_size]
            for key in keys_to_remove:
                del self.l2_cache[key]

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de cache"""
        total_requests = sum(self.stats.values())
        hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['disk_hits']) / max(1, total_requests)

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'disk_size': len(self.disk_index)
        }


class QueryOptimizer:
    """Optimización avanzada de consultas"""

    def __init__(self):
        self.query_cache = {}
        self.semantic_cache = {}
        self.synonyms = {
            'codigo': ['code', 'programacion', 'programming', 'desarrollo', 'development'],
            'seguridad': ['security', 'auth', 'autenticacion', 'authentication', 'login'],
            'base': ['database', 'db', 'datos', 'data', 'postgresql', 'sqlite'],
            'modelo': ['model', 'models', 'estructura', 'structure'],
            'vista': ['view', 'template', 'html', 'frontend'],
            'api': ['rest', 'endpoint', 'endpoints', 'django-rest-framework'],
            'testing': ['test', 'pruebas', 'tests', 'pytest', 'unittest'],
            'deploy': ['deployment', 'despliegue', 'produccion', 'production'],
            'git': ['version', 'control', 'commit', 'merge', 'branch']
        }

    def optimize_query(self, query: str, context: Dict = None) -> Dict:
        """Optimiza consulta completa"""
        if context is None:
            context = {}

        start_time = time.time()

        # 1. Normalización
        normalized_query = self.normalize_query(query)

        # 2. Verificar cache semántico
        cached_result = self.check_semantic_cache(normalized_query)
        if cached_result:
            return cached_result

        # 3. Expansión de consulta
        expanded_query = self.expand_query(normalized_query, context)

        # 4. Filtrado por relevancia
        filtered_query = self.filter_by_relevance(expanded_query)

        # 5. Cachear resultado
        result = {
            'original_query': query,
            'normalized_query': normalized_query,
            'expanded_terms': expanded_query,
            'filtered_terms': filtered_query,
            'optimization_time': time.time() - start_time
        }

        # Cache semántico
        self.semantic_cache[normalized_query] = result

        return result

    def normalize_query(self, query: str) -> str:
        """Normaliza consulta"""
        # Convertir a minúsculas, remover puntuación
        normalized = re.sub(r'[^\w\s]', '', query.lower())

        # Remover palabras comunes
        stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o',
            'pero', 'que', 'como', 'en', 'de', 'a', 'por', 'para', 'con', 'sin',
            'me', 'te', 'se', 'nos', 'les', 'lo', 'le', 'la', 'los', 'las'
        }

        words = [word for word in normalized.split() if word not in stop_words and len(word) > 2]
        return ' '.join(words)

    def expand_query(self, normalized_query: str, context: Dict) -> Dict:
        """Expande consulta con sinónimos y términos de contexto"""
        terms = normalized_query.split()

        # Obtener sinónimos
        synonyms = set()
        for term in terms:
            if term in self.synonyms:
                synonyms.update(self.synonyms[term])

        # Extraer términos de contexto
        context_terms = self.extract_context_terms(context)

        all_terms = set(terms + list(synonyms) + context_terms)

        # Calcular pesos
        weights = self.calculate_term_weights(terms, list(synonyms), context_terms)

        return {
            'original_terms': terms,
            'synonyms': list(synonyms),
            'context_terms': context_terms,
            'all_terms': list(all_terms),
            'weights': weights
        }

    def extract_context_terms(self, context: Dict) -> List[str]:
        """Extrae términos relevantes del contexto"""
        context_terms = []

        # Términos del historial reciente
        if 'recent_queries' in context:
            for query in context['recent_queries'][-3:]:  # Últimas 3 consultas
                words = query.lower().split()
                context_terms.extend(words)

        # Preferencias del usuario
        if 'user_preferences' in context:
            if context['user_preferences'].get('focus_area'):
                context_terms.append(context['user_preferences']['focus_area'])

        return list(set(context_terms))  # Remover duplicados

    def calculate_term_weights(self, original_terms: List[str], synonyms: List[str], context_terms: List[str]) -> Dict[str, float]:
        """Calcula pesos para términos"""
        weights = {}

        # Términos originales tienen peso máximo
        for term in original_terms:
            weights[term] = 1.0

        # Sinónimos tienen peso medio
        for term in synonyms:
            weights[term] = 0.7

        # Términos de contexto tienen peso bajo
        for term in context_terms:
            weights[term] = 0.4

        return weights

    def filter_by_relevance(self, expanded_query: Dict) -> List[str]:
        """Filtra términos por relevancia"""
        all_terms = expanded_query['all_terms']
        weights = expanded_query['weights']

        # Filtrar términos con peso suficiente
        relevant_terms = [term for term in all_terms if weights.get(term, 0) >= 0.5]

        # Limitar a términos más relevantes
        return sorted(relevant_terms, key=lambda x: weights.get(x, 0), reverse=True)[:10]

    def check_semantic_cache(self, normalized_query: str) -> Optional[Dict]:
        """Verifica cache semántico"""
        if normalized_query in self.semantic_cache:
            entry = self.semantic_cache[normalized_query]
            # Verificar si no está expirado (5 minutos)
            if time.time() - entry.get('timestamp', 0) < 300:
                return entry

        return None


class AdaptiveRateLimiter:
    """Rate limiting adaptativo"""

    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            'per_second': 10,
            'per_minute': 100,
            'per_hour': 1000
        }
        self.penalties = defaultdict(int)  # Penalizaciones por cliente

    def check_limit(self, client_id: str) -> bool:
        """Verifica límites de tasa para un cliente"""
        current_time = time.time()

        # Limpiar requests antiguos
        self.cleanup_old_requests(client_id, current_time)

        client_requests = self.requests[client_id]

        # Verificar límites con penalización
        penalty_multiplier = 1 + (self.penalties[client_id] * 0.5)

        # Por segundo
        recent_second = [t for t in client_requests if current_time - t < 1]
        if len(recent_second) >= (self.limits['per_second'] * penalty_multiplier):
            self.apply_penalty(client_id)
            return False

        # Por minuto
        recent_minute = [t for t in client_requests if current_time - t < 60]
        if len(recent_minute) >= (self.limits['per_minute'] * penalty_multiplier):
            self.apply_penalty(client_id)
            return False

        # Por hora
        recent_hour = [t for t in client_requests if current_time - t < 3600]
        if len(recent_hour) >= (self.limits['per_hour'] * penalty_multiplier):
            self.apply_penalty(client_id)
            return False

        # Registrar request
        client_requests.append(current_time)
        return True

    def cleanup_old_requests(self, client_id: str, current_time: float) -> None:
        """Limpia requests antiguos"""
        client_requests = self.requests[client_id]

        # Mantener solo requests de la última hora
        while client_requests and current_time - client_requests[0] > 3600:
            client_requests.popleft()

        # Remover deque vacío
        if not client_requests:
            del self.requests[client_id]

    def apply_penalty(self, client_id: str) -> None:
        """Aplica penalización por violación de límites"""
        self.penalties[client_id] = min(self.penalties[client_id] + 1, 5)  # Máximo 5 penalizaciones

        # Reducir penalización gradualmente
        threading.Timer(300, lambda: self.reduce_penalty(client_id)).start()  # 5 minutos

    def reduce_penalty(self, client_id: str) -> None:
        """Reduce penalización gradualmente"""
        if self.penalties[client_id] > 0:
            self.penalties[client_id] -= 1


class ResourceMonitor:
    """Monitoreo de recursos del sistema"""

    def __init__(self):
        self.metrics = {
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'response_time': deque(maxlen=100),
            'cache_hit_rate': 0,
            'active_connections': 0
        }
        self.monitoring_active = False
        self.monitor_thread = None

    def start_monitoring(self) -> None:
        """Inicia monitoreo en background"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoreo de recursos iniciado")

    def stop_monitoring(self) -> None:
        """Detiene monitoreo"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoreo de recursos detenido")

    def monitor_loop(self) -> None:
        """Loop principal de monitoreo"""
        while self.monitoring_active:
            try:
                self.collect_metrics()
                self.optimize_based_on_metrics()
                time.sleep(5)  # Cada 5 segundos
            except Exception as e:
                logger.error(f"Error en monitoreo: {e}")

    def collect_metrics(self) -> None:
        """Recolecta métricas del sistema"""
        current_time = time.time()

        # Memoria
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'].append({
            'timestamp': current_time,
            'percent': memory.percent,
            'used': memory.used,
            'available': memory.available
        })

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics['cpu_usage'].append({
            'timestamp': current_time,
            'percent': cpu_percent
        })

    def optimize_based_on_metrics(self) -> None:
        """Optimiza basado en métricas recolectadas"""
        avg_memory = self.get_average_memory_usage()
        avg_cpu = self.get_average_cpu_usage()

        # Optimizaciones basadas en métricas
        if avg_memory > 80:  # 80% de memoria
            logger.warning("Uso alto de memoria detectado, activando optimizaciones")
            self.trigger_memory_optimization()

        if avg_cpu > 90:  # 90% de CPU
            logger.warning("Uso alto de CPU detectado, reduciendo concurrencia")
            self.trigger_cpu_optimization()

    def trigger_memory_optimization(self) -> None:
        """Optimiza uso de memoria"""
        # Forzar garbage collection
        import gc
        gc.collect()

        # Limpiar caches si es necesario
        logger.info("Optimización de memoria ejecutada")

    def trigger_cpu_optimization(self) -> None:
        """Optimiza uso de CPU"""
        # Reducir threads o procesos si es necesario
        logger.info("Optimización de CPU ejecutada")

    def get_average_memory_usage(self) -> float:
        """Obtiene uso promedio de memoria"""
        if not self.metrics['memory_usage']:
            return 0

        recent_memory = list(self.metrics['memory_usage'])[-10:]  # Últimos 10 registros
        return sum(m['percent'] for m in recent_memory) / len(recent_memory)

    def get_average_cpu_usage(self) -> float:
        """Obtiene uso promedio de CPU"""
        if not self.metrics['cpu_usage']:
            return 0

        recent_cpu = list(self.metrics['cpu_usage'])[-10:]  # Últimos 10 registros
        return sum(c['percent'] for c in recent_cpu) / len(recent_cpu)

    def record_response_time(self, response_time: float) -> None:
        """Registra tiempo de respuesta"""
        self.metrics['response_time'].append({
            'timestamp': time.time(),
            'time': response_time
        })

    def update_cache_hit_rate(self, hit_rate: float) -> None:
        """Actualiza hit rate de cache"""
        self.metrics['cache_hit_rate'] = hit_rate

    def get_metrics_summary(self) -> Dict:
        """Obtiene resumen de métricas"""
        return {
            'memory_avg_percent': self.get_average_memory_usage(),
            'cpu_avg_percent': self.get_average_cpu_usage(),
            'response_time_avg': self.get_average_response_time(),
            'cache_hit_rate': self.metrics['cache_hit_rate'],
            'memory_samples': len(self.metrics['memory_usage']),
            'cpu_samples': len(self.metrics['cpu_usage'])
        }

    def get_average_response_time(self) -> float:
        """Obtiene tiempo promedio de respuesta"""
        if not self.metrics['response_time']:
            return 0

        recent_responses = list(self.metrics['response_time'])[-20:]  # Últimos 20
        return sum(r['time'] for r in recent_responses) / len(recent_responses)


class OptimizedFuzzySearch:
    """Búsqueda fuzzy optimizada"""

    def __init__(self):
        self.indexed_terms = {}
        self.ngram_index = defaultdict(set)
        self.documents = {}

    def build_index(self, documents: Dict[str, Dict]) -> None:
        """Construye índices para búsqueda fuzzy"""
        # Limpiar índices previos para evitar referencias obsoletas
        self.indexed_terms.clear()
        self.ngram_index.clear()
        self.documents = {}

        for doc_id, doc_data in documents.items():
            self.documents[doc_id] = doc_data
            content = doc_data.get('content', '')

            terms = self.extract_terms(content)
            for term in terms:
                if term not in self.indexed_terms:
                    self.indexed_terms[term] = set()
                self.indexed_terms[term].add(doc_id)

                # Construir índice de n-gramas
                self.build_ngram_index(term, doc_id)

    def search(self, query: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Búsqueda fuzzy con scoring"""
        query_terms = self.extract_terms(query)
        candidates = set()

        # Búsqueda exacta primero
        for term in query_terms:
            if term in self.indexed_terms:
                candidates.update(self.indexed_terms[term])

        results = []

        # Si pocos resultados, hacer búsqueda fuzzy
        if len(candidates) < 5:
            fuzzy_matches = self.fuzzy_search(query_terms, threshold)
            candidates.update(fuzzy_matches.keys())

        # Calcular scores para todos los candidatos
        for doc_id in candidates:
            score = self.calculate_relevance_score(query, self.documents[doc_id])
            results.append((doc_id, score))

        # Ordenar por score descendente
        return sorted(results, key=lambda x: x[1], reverse=True)

    def fuzzy_search(self, query_terms: List[str], threshold: float) -> Dict[str, float]:
        """Búsqueda fuzzy usando n-gramas"""
        matches = {}

        for query_term in query_terms:
            query_ngrams = self.get_ngrams(query_term)

            for ngram, doc_ids in self.ngram_index.items():
                if self.ngram_similarity(query_ngrams, self.get_ngrams(ngram)) >= threshold:
                    for doc_id in doc_ids:
                        if doc_id not in matches:
                            matches[doc_id] = 0
                        matches[doc_id] += 1

        return matches

    def get_ngrams(self, term: str, n: int = 2) -> Set[str]:
        """Genera n-gramas para un término"""
        term = term.lower()
        ngrams = set()

        for i in range(len(term) - n + 1):
            ngrams.add(term[i:i+n])

        return ngrams

    def ngram_similarity(self, ngrams1: Set[str], ngrams2: Set[str]) -> float:
        """Calcula similitud usando n-gramas"""
        if not ngrams1 or not ngrams2:
            return 0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0

    def calculate_relevance_score(self, query: str, document: Dict) -> float:
        """Calcula score de relevancia"""
        content = document.get('content', '').lower()
        query_lower = query.lower()

        # Score exacto
        exact_matches = sum(1 for word in query_lower.split() if word in content)
        exact_score = exact_matches / len(query_lower.split()) if query_lower.split() else 0

        # Score parcial
        partial_score = 0
        for word in query_lower.split():
            if any(word in sentence for sentence in content.split('.')):
                partial_score += 0.5

        # Score semántico (simplificado)
        semantic_score = 0
        semantic_terms = ['importante', 'crucial', 'clave', 'fundamental']
        for term in semantic_terms:
            if term in content:
                semantic_score += 0.2

        # Score de contexto
        context_score = min(0.3, len(content) / 1000)  # Bonus por longitud razonable

        # Score de recencia
        recency_score = document.get('recency_score', 0.5)

        # Combinar scores con pesos
        weights = {
            'exact': 0.4,
            'partial': 0.3,
            'semantic': 0.2,
            'context': 0.05,
            'recency': 0.05
        }

        final_score = (
            exact_score * weights['exact'] +
            partial_score * weights['partial'] +
            semantic_score * weights['semantic'] +
            context_score * weights['context'] +
            recency_score * weights['recency']
        )

        return min(1.0, final_score)  # Máximo 1.0

    def extract_terms(self, text: str) -> List[str]:
        """Extrae términos del texto"""
        # Normalizar y tokenizar
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        terms = text.split()

        # Filtrar términos cortos y comunes
        stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'en', 'de', 'a', 'por', 'con'}
        filtered_terms = [term for term in terms if len(term) > 2 and term not in stop_words]

        return list(set(filtered_terms))  # Remover duplicados

    def build_ngram_index(self, term: str, doc_id: str) -> None:
        """Construye índice de n-gramas"""
        ngrams = self.get_ngrams(term)
        for ngram in ngrams:
            self.ngram_index[ngram].add(doc_id)

    def has_index(self) -> bool:
        """Indica si el índice contiene documentos cargados"""
        return len(self.documents) > 0

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Obtiene un documento previamente indexado"""
        return self.documents.get(doc_id)


class RelevanceScorer:
    """Sistema de puntuación de relevancia avanzado"""

    def __init__(self):
        self.weights = {
            'exact_match': 1.0,
            'partial_match': 0.7,
            'semantic_match': 0.5,
            'context_match': 0.3,
            'recency': 0.2,
            'user_preference': 0.4,
            'content_quality': 0.3
        }

    def calculate_relevance(self, query: str, document: Dict, context: Dict = None) -> float:
        """Calcula relevancia completa"""
        if context is None:
            context = {}

        scores = {
            'exact_match': self.get_exact_match_score(query, document),
            'partial_match': self.get_partial_match_score(query, document),
            'semantic_match': self.get_semantic_match_score(query, document),
            'context_match': self.get_context_match_score(context, document),
            'recency': self.get_recency_score(document),
            'user_preference': self.get_user_preference_score(context, document),
            'content_quality': self.get_content_quality_score(document)
        }

        # Calcular score final
        final_score = sum(scores[key] * self.weights[key] for key in scores)

        # Normalizar a 0-1
        return min(1.0, max(0.0, final_score))

    def get_exact_match_score(self, query: str, document: Dict) -> float:
        """Score de coincidencia exacta"""
        query_terms = set(query.lower().split())
        content = document.get('content', '').lower()

        matches = sum(1 for term in query_terms if term in content)
        return matches / len(query_terms) if query_terms else 0

    def get_partial_match_score(self, query: str, document: Dict) -> float:
        """Score de coincidencia parcial"""
        query_terms = query.lower().split()
        content = document.get('content', '').lower()

        score = 0
        for term in query_terms:
            # Buscar términos similares
            if term in content:
                score += 1.0
            elif len(term) > 3:
                # Buscar substrings
                if any(term in word for word in content.split()):
                    score += 0.5

        return score / len(query_terms) if query_terms else 0

    def get_semantic_match_score(self, query: str, document: Dict) -> float:
        """Score semántico basado en categorías"""
        semantic_map = {
            'codigo': ['python', 'django', 'function', 'class', 'import', 'def'],
            'seguridad': ['auth', 'login', 'password', 'encrypt', 'secure'],
            'base_datos': ['model', 'database', 'query', 'sql', 'postgresql'],
            'testing': ['test', 'pytest', 'unittest', 'coverage'],
            'api': ['rest', 'endpoint', 'json', 'request', 'response']
        }

        query_lower = query.lower()
        content = document.get('content', '').lower()

        score = 0
        for category, terms in semantic_map.items():
            query_has_category = any(term in query_lower for term in terms)
            content_has_category = any(term in content for term in terms)

            if query_has_category and content_has_category:
                score += 0.5

        return min(1.0, score)

    def get_context_match_score(self, context: Dict, document: Dict) -> float:
        """Score basado en contexto de usuario"""
        score = 0

        # Historial reciente
        if 'recent_queries' in context:
            recent_terms = set()
            for q in context['recent_queries'][-3:]:
                recent_terms.update(q.lower().split())

            content = document.get('content', '').lower()
            matches = sum(1 for term in recent_terms if term in content)
            score += min(0.5, matches / len(recent_terms) if recent_terms else 0)

        # Preferencias de usuario
        if 'user_preferences' in context:
            prefs = context['user_preferences']
            if prefs.get('preferred_sections'):
                section_id = document.get('section_id', '')
                if section_id in prefs['preferred_sections']:
                    score += 0.3

        return min(1.0, score)

    def get_recency_score(self, document: Dict) -> float:
        """Score basado en recencia"""
        last_updated = document.get('last_updated', time.time())
        days_old = (time.time() - last_updated) / (24 * 3600)

        # Decaimiento exponencial
        return math.exp(-days_old / 30)  # 30 días de vida media

    def get_user_preference_score(self, context: Dict, document: Dict) -> float:
        """Score basado en preferencias de usuario"""
        if 'user_profile' not in context:
            return 0.5  # Score neutral

        profile = context['user_profile']
        score = 0

        # Nivel de experiencia
        experience_level = profile.get('experience_level', 'intermediate')
        if experience_level == 'beginner' and document.get('difficulty') == 'basic':
            score += 0.3
        elif experience_level == 'advanced' and document.get('difficulty') == 'advanced':
            score += 0.3

        # Áreas de interés
        interests = profile.get('interests', [])
        section_id = document.get('section_id', '')
        if any(interest in section_id for interest in interests):
            score += 0.4

        return min(1.0, score)

    def get_content_quality_score(self, document: Dict) -> float:
        """Score basado en calidad del contenido"""
        content = document.get('content', '')
        score = 0

        # Longitud apropiada
        word_count = len(content.split())
        if 50 <= word_count <= 2000:
            score += 0.3
        elif word_count > 2000:
            score += 0.1  # Penalización por contenido muy largo

        # Estructura
        has_lists = bool(re.search(r'^[-*+]\s', content, re.MULTILINE))
        has_headers = bool(re.search(r'^#{1,6}\s', content, re.MULTILINE))
        has_code = bool(re.search(r'```.*?```', content, re.DOTALL))

        if has_lists or has_headers:
            score += 0.3
        if has_code:
            score += 0.4

        # Actualidad
        last_updated = document.get('last_updated', time.time())
        months_old = (time.time() - last_updated) / (30 * 24 * 3600)
        if months_old < 6:
            score += 0.2

        return min(1.0, score)


# Instancias globales para uso en el servidor
token_budget = TokenBudgetManager()
semantic_chunker = SemanticChunker()
cache = MultiLevelCache()
query_optimizer = QueryOptimizer()
rate_limiter = AdaptiveRateLimiter()
resource_monitor = ResourceMonitor()
fuzzy_search = OptimizedFuzzySearch()
relevance_scorer = RelevanceScorer()
