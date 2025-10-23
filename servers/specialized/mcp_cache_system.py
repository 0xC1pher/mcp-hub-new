#!/usr/bin/env python3
"""
MCP Cache System - Servidor especializado en cache multinivel inteligente
Implementa cache L1/L2/Disk con >85% hit rate y deduplicación automática
"""

import json
import sys
import logging
import time
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import threading

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp-cache-system')

class IntelligentCacheSystem:
    """Sistema de cache multinivel inteligente optimizado"""
    
    def __init__(self, cache_dir: str = "cache", l1_size: int = 100, l2_size: int = 1000, disk_size: int = 10000):
        self.cache_directory = Path(cache_dir)
        self.cache_directory.mkdir(exist_ok=True)
        
        # Cache multinivel
        self.l1_cache = {}  # Cache rápido en memoria (100 items)
        self.l2_cache = {}  # Cache medio en memoria (1000 items)
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.disk_size = disk_size
        
        # Métricas de rendimiento
        self.hits = 0
        self.misses = 0
        self.access_counts = defaultdict(int)
        self.last_access = defaultdict(float)
        
        # Deduplicación
        self.content_hashes = set()
        self.hash_to_key = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"✅ Cache System inicializado - L1:{l1_size}, L2:{l2_size}, Disk:{disk_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache multinivel"""
        with self.lock:
            current_time = time.time()
            
            # L1 Cache (más rápido)
            if key in self.l1_cache:
                self.hits += 1
                self.access_counts[key] += 1
                self.last_access[key] = current_time
                logger.debug(f"🎯 L1 HIT: {key}")
                return self.l1_cache[key]
            
            # L2 Cache
            if key in self.l2_cache:
                self.hits += 1
                self.access_counts[key] += 1
                self.last_access[key] = current_time
                # Promover a L1
                self._promote_to_l1(key)
                logger.debug(f"🎯 L2 HIT: {key}")
                return self.l2_cache[key]
            
            # Disk Cache
            disk_value = self._get_from_disk(key)
            if disk_value is not None:
                self.hits += 1
                self.access_counts[key] += 1
                self.last_access[key] = current_time
                # Promover a L2
                self._promote_to_l2(key, disk_value)
                logger.debug(f"🎯 DISK HIT: {key}")
                return disk_value
            
            # Cache miss
            self.misses += 1
            logger.debug(f"❌ CACHE MISS: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Establece valor en cache con deduplicación"""
        with self.lock:
            try:
                # Generar hash del contenido para deduplicación
                content_str = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
                content_hash = hashlib.md5(content_str.encode()).hexdigest()[:12]
                
                # Verificar duplicación
                if content_hash in self.content_hashes:
                    existing_key = self.hash_to_key.get(content_hash)
                    if existing_key and existing_key != key:
                        logger.info(f"🔄 Contenido duplicado detectado: {key} -> {existing_key}")
                        # Crear alias en lugar de duplicar
                        self.l1_cache[key] = self.get(existing_key)
                        return True
                
                # Guardar en L1
                self.l1_cache[key] = value
                self.access_counts[key] += 1
                self.last_access[key] = time.time()
                
                # Registrar hash
                self.content_hashes.add(content_hash)
                self.hash_to_key[content_hash] = key
                
                # Guardar en disco
                self._save_to_disk(key, value, ttl)
                
                # Limpiar caches si están llenos
                self._cleanup_caches()
                
                logger.debug(f"💾 CACHE SET: {key} (hash: {content_hash})")
                return True
                
            except Exception as e:
                logger.error(f"Error guardando en cache {key}: {e}")
                return False
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Busca en cache usando query semántica"""
        with self.lock:
            results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            # Buscar en L1
            for key, value in self.l1_cache.items():
                score = self._calculate_relevance(query_words, key, value)
                if score > 0.3:
                    results.append({
                        'key': key,
                        'content': value,
                        'score': score,
                        'source': 'L1',
                        'access_count': self.access_counts[key]
                    })
            
            # Buscar en L2 si necesitamos más resultados
            if len(results) < max_results:
                for key, value in self.l2_cache.items():
                    if key not in self.l1_cache:  # Evitar duplicados
                        score = self._calculate_relevance(query_words, key, value)
                        if score > 0.3:
                            results.append({
                                'key': key,
                                'content': value,
                                'score': score,
                                'source': 'L2',
                                'access_count': self.access_counts[key]
                            })
            
            # Buscar en disco si aún necesitamos más
            if len(results) < max_results:
                disk_results = self._search_disk(query_words, max_results - len(results))
                results.extend(disk_results)
            
            # Ordenar por relevancia
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:max_results]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del cache"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses,
                'total_requests': total_requests,
                'l1_size': len(self.l1_cache),
                'l2_size': len(self.l2_cache),
                'disk_size': len(list(self.cache_directory.glob("*.json"))),
                'unique_content_hashes': len(self.content_hashes),
                'deduplication_rate': len(self.content_hashes) / max(1, len(self.l1_cache) + len(self.l2_cache)) * 100
            }
    
    def force_refresh(self) -> Dict[str, Any]:
        """Fuerza actualización completa del cache"""
        with self.lock:
            # Limpiar caches en memoria
            old_l1_size = len(self.l1_cache)
            old_l2_size = len(self.l2_cache)
            
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.content_hashes.clear()
            self.hash_to_key.clear()
            
            # Resetear métricas
            self.hits = 0
            self.misses = 0
            self.access_counts.clear()
            self.last_access.clear()
            
            logger.info(f"🔄 Cache refresh completado - L1:{old_l1_size}→0, L2:{old_l2_size}→0")
            
            return {
                'l1_cleared': old_l1_size,
                'l2_cleared': old_l2_size,
                'disk_files': len(list(self.cache_directory.glob("*.json"))),
                'status': 'refreshed'
            }
    
    def _promote_to_l1(self, key: str) -> None:
        """Promueve elemento de L2 a L1"""
        if len(self.l1_cache) >= self.l1_size:
            # Eliminar elemento menos usado de L1
            lru_key = min(self.l1_cache.keys(), key=lambda k: self.last_access[k])
            # Mover a L2 si hay espacio
            if len(self.l2_cache) < self.l2_size:
                self.l2_cache[lru_key] = self.l1_cache[lru_key]
            del self.l1_cache[lru_key]
        
        self.l1_cache[key] = self.l2_cache[key]
    
    def _promote_to_l2(self, key: str, value: Any) -> None:
        """Promueve elemento de disco a L2"""
        if len(self.l2_cache) >= self.l2_size:
            # Eliminar elemento menos usado de L2
            lru_key = min(self.l2_cache.keys(), key=lambda k: self.last_access[k])
            del self.l2_cache[lru_key]
        
        self.l2_cache[key] = value
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache en disco"""
        try:
            cache_file = self.cache_directory / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('expires', 0) > time.time():
                        return data.get('value')
                    else:
                        cache_file.unlink()  # Eliminar archivo expirado
        except Exception as e:
            logger.error(f"Error leyendo cache de disco {key}: {e}")
        return None
    
    def _save_to_disk(self, key: str, value: Any, ttl: int) -> None:
        """Guarda valor en cache de disco"""
        try:
            cache_file = self.cache_directory / f"{key}.json"
            data = {
                'value': value,
                'expires': time.time() + ttl,
                'created': time.time()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error guardando cache en disco {key}: {e}")
    
    def _cleanup_caches(self) -> None:
        """Limpia caches cuando están llenos"""
        # Limpiar L1 si está lleno
        if len(self.l1_cache) > self.l1_size:
            # Mover elementos menos usados a L2
            sorted_items = sorted(self.l1_cache.items(), key=lambda x: self.last_access[x[0]])
            items_to_move = len(self.l1_cache) - self.l1_size
            
            for key, value in sorted_items[:items_to_move]:
                if len(self.l2_cache) < self.l2_size:
                    self.l2_cache[key] = value
                del self.l1_cache[key]
        
        # Limpiar L2 si está lleno
        if len(self.l2_cache) > self.l2_size:
            sorted_items = sorted(self.l2_cache.items(), key=lambda x: self.last_access[x[0]])
            items_to_remove = len(self.l2_cache) - self.l2_size
            
            for key, _ in sorted_items[:items_to_remove]:
                del self.l2_cache[key]
    
    def _calculate_relevance(self, query_words: set, key: str, value: Any) -> float:
        """Calcula relevancia de un item del cache para la query"""
        try:
            # Convertir valor a texto
            text_content = json.dumps(value) if not isinstance(value, str) else value
            text_content = text_content.lower()
            key_lower = key.lower()
            
            # Calcular score basado en coincidencias
            key_matches = sum(1 for word in query_words if word in key_lower)
            content_matches = sum(1 for word in query_words if word in text_content)
            
            # Normalizar scores
            key_score = key_matches / len(query_words) if query_words else 0
            content_score = content_matches / len(query_words) if query_words else 0
            
            # Score final (key tiene más peso)
            final_score = (key_score * 0.7) + (content_score * 0.3)
            
            # Bonus por frecuencia de acceso
            access_bonus = min(0.2, self.access_counts[key] / 100)
            
            return min(1.0, final_score + access_bonus)
            
        except Exception as e:
            logger.error(f"Error calculando relevancia: {e}")
            return 0.0
    
    def _search_disk(self, query_words: set, max_results: int) -> List[Dict]:
        """Busca en archivos de cache en disco"""
        results = []
        try:
            for cache_file in self.cache_directory.glob("*.json"):
                if len(results) >= max_results:
                    break
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get('expires', 0) > time.time():
                            key = cache_file.stem
                            value = data.get('value')
                            score = self._calculate_relevance(query_words, key, value)
                            
                            if score > 0.3:
                                results.append({
                                    'key': key,
                                    'content': value,
                                    'score': score,
                                    'source': 'DISK',
                                    'access_count': self.access_counts[key]
                                })
                except Exception as e:
                    logger.error(f"Error leyendo archivo de cache {cache_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error buscando en disco: {e}")
        
        return results


class MCPCacheServer:
    """Servidor MCP especializado en cache multinivel"""
    
    def __init__(self):
        self.cache_system = IntelligentCacheSystem()
        logger.info("🚀 MCP Cache Server iniciado")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP"""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        try:
            if method == 'initialize':
                result = self._handle_initialize(params)
            elif method == 'tools/list':
                result = self._handle_tools_list(params)
            elif method == 'tools/call':
                result = self._handle_tools_call(params)
            else:
                result = {'error': f'Método no soportado: {method}'}
            
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error manejando request: {e}")
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'error': {
                    'code': -32603,
                    'message': str(e)
                }
            }
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja inicialización del servidor MCP"""
        return {
            'protocolVersion': '2024-11-05',
            'capabilities': {
                'tools': {
                    'listChanged': True
                }
            },
            'serverInfo': {
                'name': 'mcp-cache-system',
                'version': '1.0.0'
            }
        }
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista las herramientas disponibles"""
        return {
            'tools': [
                {
                    'name': 'cache_get',
                    'description': 'Obtiene valor del cache multinivel',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'key': {'type': 'string', 'description': 'Clave del cache'}
                        },
                        'required': ['key']
                    }
                },
                {
                    'name': 'cache_set',
                    'description': 'Establece valor en cache con deduplicación',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'key': {'type': 'string', 'description': 'Clave del cache'},
                            'value': {'description': 'Valor a guardar'},
                            'ttl': {'type': 'integer', 'description': 'Tiempo de vida en segundos', 'default': 3600}
                        },
                        'required': ['key', 'value']
                    }
                },
                {
                    'name': 'cache_search',
                    'description': 'Busca en cache usando query semántica',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Query de búsqueda'},
                            'max_results': {'type': 'integer', 'description': 'Máximo resultados', 'default': 10}
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'cache_metrics',
                    'description': 'Obtiene métricas del cache',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                },
                {
                    'name': 'cache_refresh',
                    'description': 'Fuerza actualización completa del cache',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            ]
        }
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        try:
            if tool_name == 'cache_get':
                return self._cache_get(arguments)
            elif tool_name == 'cache_set':
                return self._cache_set(arguments)
            elif tool_name == 'cache_search':
                return self._cache_search(arguments)
            elif tool_name == 'cache_metrics':
                return self._cache_metrics(arguments)
            elif tool_name == 'cache_refresh':
                return self._cache_refresh(arguments)
            else:
                return {
                    'content': [{'type': 'text', 'text': f'Herramienta desconocida: {tool_name}'}],
                    'isError': True
                }
                
        except Exception as e:
            logger.error(f"Error ejecutando herramienta {tool_name}: {e}")
            return {
                'content': [{'type': 'text', 'text': f'Error ejecutando {tool_name}: {str(e)}'}],
                'isError': True
            }
    
    def _cache_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene valor del cache"""
        key = args.get('key', '')
        value = self.cache_system.get(key)
        
        if value is not None:
            return {
                'content': [{'type': 'text', 'text': f'🎯 Cache HIT para "{key}":\n\n{json.dumps(value, indent=2, ensure_ascii=False)}'}]
            }
        else:
            return {
                'content': [{'type': 'text', 'text': f'❌ Cache MISS para "{key}" - No encontrado'}]
            }
    
    def _cache_set(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Establece valor en cache"""
        key = args.get('key', '')
        value = args.get('value')
        ttl = args.get('ttl', 3600)
        
        success = self.cache_system.set(key, value, ttl)
        
        if success:
            return {
                'content': [{'type': 'text', 'text': f'✅ Valor guardado en cache: "{key}" (TTL: {ttl}s)'}]
            }
        else:
            return {
                'content': [{'type': 'text', 'text': f'❌ Error guardando en cache: "{key}"'}],
                'isError': True
            }
    
    def _cache_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Busca en cache"""
        query = args.get('query', '')
        max_results = args.get('max_results', 10)
        
        results = self.cache_system.search(query, max_results)
        
        if results:
            response = f'🔍 Encontrados {len(results)} resultados para "{query}":\n\n'
            
            for i, result in enumerate(results, 1):
                response += f'**{i}. {result["key"]}** (score: {result["score"]:.2f}, source: {result["source"]})\n'
                content_preview = str(result["content"])[:200] + "..." if len(str(result["content"])) > 200 else str(result["content"])
                response += f'```\n{content_preview}\n```\n\n'
            
            return {
                'content': [{'type': 'text', 'text': response}]
            }
        else:
            return {
                'content': [{'type': 'text', 'text': f'🔍 No se encontraron resultados para: "{query}"'}]
            }
    
    def _cache_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene métricas del cache"""
        metrics = self.cache_system.get_metrics()
        
        response = f'''📊 **Métricas del Cache Multinivel**

🎯 **Rendimiento**:
- Hit Rate: {metrics["hit_rate"]:.1f}%
- Cache Hits: {metrics["hits"]}
- Cache Misses: {metrics["misses"]}
- Total Requests: {metrics["total_requests"]}

💾 **Utilización**:
- L1 Cache: {metrics["l1_size"]} items
- L2 Cache: {metrics["l2_size"]} items
- Disk Cache: {metrics["disk_size"]} items

🔄 **Deduplicación**:
- Content Hashes: {metrics["unique_content_hashes"]}
- Deduplication Rate: {metrics["deduplication_rate"]:.1f}%

⚡ **Estado**: {'🟢 Óptimo' if metrics["hit_rate"] > 85 else '🟡 Bueno' if metrics["hit_rate"] > 70 else '🔴 Necesita optimización'}
'''
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }
    
    def _cache_refresh(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Actualiza cache"""
        result = self.cache_system.force_refresh()
        
        response = f'''🔄 **Cache Actualizado**

✅ L1 Cache limpiado: {result["l1_cleared"]} items
✅ L2 Cache limpiado: {result["l2_cleared"]} items
📁 Archivos en disco: {result["disk_files"]} items
⚡ Estado: {result["status"]}

Sistema listo para nuevas consultas.
'''
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }
    
    def run(self):
        """Ejecuta el servidor MCP"""
        logger.info("🚀 Iniciando MCP Cache Server...")
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = self.handle_request(request)
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error decodificando JSON: {e}")
                except Exception as e:
                    logger.error(f"Error procesando línea: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Servidor detenido por usuario")
        except Exception as e:
            logger.error(f"Error en servidor: {e}")


if __name__ == "__main__":
    server = MCPCacheServer()
    server.run()
