#!/usr/bin/env python3
"""
Sistema de Cache Inteligente Multinivel
Implementa cache L1/L2/Disk con alimentación automática desde directorio
y chunking inteligente para máximo rendimiento.
"""

import json
import os
import time
import hashlib
import threading
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheItem:
    """Elemento del cache con metadata"""
    key: str
    content: str
    chunks: List[str]
    metadata: Dict[str, Any]
    access_count: int
    last_accessed: float
    created_at: float
    file_hash: str
    file_path: str
    size_bytes: int

@dataclass
class CacheMetrics:
    """Métricas del sistema de cache"""
    l1_hits: int = 0
    l2_hits: int = 0
    disk_hits: int = 0
    misses: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        hits = self.l1_hits + self.l2_hits + self.disk_hits
        return hits / self.total_requests
    
    @property
    def l1_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.l1_hits / self.total_requests

class IntelligentCacheSystem:
    """Sistema de cache inteligente multinivel con alimentación automática"""
    
    def __init__(self, 
                 source_directory: str,
                 cache_directory: str = None,
                 l1_size: int = 100,
                 l2_size: int = 1000,
                 disk_size: int = 10000,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.source_directory = Path(source_directory)
        self.cache_directory = Path(cache_directory or (self.source_directory / ".cache"))
        
        # Configuración de cache
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.disk_size = disk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Cache multinivel
        self.l1_cache: OrderedDict[str, CacheItem] = OrderedDict()  # Acceso instantáneo
        self.l2_cache: OrderedDict[str, CacheItem] = OrderedDict()  # Datos frecuentes
        self.disk_cache_index: Dict[str, str] = {}  # key -> file_path
        
        # Índices para búsqueda rápida
        self.content_hash_index: Dict[str, str] = {}  # content_hash -> key
        self.file_path_index: Dict[str, str] = {}    # file_path -> key
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> keys
        
        # Métricas y monitoreo
        self.metrics = CacheMetrics()
        self.file_timestamps: Dict[str, float] = {}
        
        # Threading para operaciones asíncronas
        self.lock = threading.RLock()
        self.background_thread = None
        self.stop_background = False
        
        # Inicializar sistema
        self._initialize_cache_system()
    
    def _initialize_cache_system(self):
        """Inicializa el sistema de cache"""
        
        # Crear directorios necesarios
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        (self.cache_directory / "l1").mkdir(exist_ok=True)
        (self.cache_directory / "l2").mkdir(exist_ok=True)
        (self.cache_directory / "disk").mkdir(exist_ok=True)
        
        # Cargar índices existentes
        self._load_cache_indexes()
        
        # Cargar cache desde disco
        self._load_existing_cache()
        
        # Iniciar alimentación automática
        self._start_auto_feeding()
        
        logger.info(f"Cache inteligente inicializado:")
        logger.info(f"  - Directorio fuente: {self.source_directory}")
        logger.info(f"  - Cache L1: {len(self.l1_cache)}/{self.l1_size} items")
        logger.info(f"  - Cache L2: {len(self.l2_cache)}/{self.l2_size} items")
        logger.info(f"  - Cache Disk: {len(self.disk_cache_index)} items")
    
    def _load_cache_indexes(self):
        """Carga índices de cache desde disco"""
        
        indexes_file = self.cache_directory / "indexes.json"
        if indexes_file.exists():
            try:
                with open(indexes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.content_hash_index = data.get('content_hash_index', {})
                self.file_path_index = data.get('file_path_index', {})
                self.file_timestamps = data.get('file_timestamps', {})
                
                # Reconstruir keyword_index
                keyword_data = data.get('keyword_index', {})
                for keyword, keys in keyword_data.items():
                    self.keyword_index[keyword] = set(keys)
                
                logger.info(f"Índices cargados: {len(self.content_hash_index)} items")
                
            except Exception as e:
                logger.warning(f"Error cargando índices: {e}")
    
    def _save_cache_indexes(self):
        """Guarda índices de cache en disco"""
        
        try:
            # Convertir sets a listas para JSON
            keyword_data = {k: list(v) for k, v in self.keyword_index.items()}
            
            data = {
                'content_hash_index': self.content_hash_index,
                'file_path_index': self.file_path_index,
                'file_timestamps': self.file_timestamps,
                'keyword_index': keyword_data,
                'last_updated': time.time()
            }
            
            indexes_file = self.cache_directory / "indexes.json"
            with open(indexes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error guardando índices: {e}")
    
    def _load_existing_cache(self):
        """Carga cache existente desde disco"""
        
        # Cargar L2 cache
        l2_dir = self.cache_directory / "l2"
        for cache_file in l2_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    item = pickle.load(f)
                    if isinstance(item, CacheItem):
                        self.l2_cache[item.key] = item
                        if len(self.l2_cache) >= self.l2_size:
                            break
            except Exception as e:
                logger.warning(f"Error cargando cache L2 {cache_file}: {e}")
        
        # Indexar disk cache
        disk_dir = self.cache_directory / "disk"
        for cache_file in disk_dir.glob("*.pkl"):
            key = cache_file.stem
            self.disk_cache_index[key] = str(cache_file)
        
        logger.info(f"Cache cargado - L2: {len(self.l2_cache)}, Disk: {len(self.disk_cache_index)}")
    
    def _start_auto_feeding(self):
        """Inicia alimentación automática desde directorio fuente"""
        
        if self.background_thread is None or not self.background_thread.is_alive():
            self.background_thread = threading.Thread(
                target=self._auto_feed_worker,
                daemon=True
            )
            self.background_thread.start()
            logger.info("Alimentación automática iniciada")
    
    def _auto_feed_worker(self):
        """Worker para alimentación automática del cache"""
        
        while not self.stop_background:
            try:
                self._scan_and_cache_directory()
                time.sleep(30)  # Escanear cada 30 segundos
            except Exception as e:
                logger.error(f"Error en alimentación automática: {e}")
                time.sleep(60)  # Esperar más tiempo si hay error
    
    def _scan_and_cache_directory(self):
        """Escanea directorio fuente y cachea archivos nuevos/modificados"""
        
        if not self.source_directory.exists():
            return
        
        files_processed = 0
        files_skipped = 0
        
        # Buscar archivos de texto
        text_extensions = {'.md', '.txt', '.py', '.js', '.json', '.yaml', '.yml', '.xml', '.html', '.css'}
        
        for file_path in self.source_directory.rglob('*'):
            if not file_path.is_file() or file_path.suffix.lower() not in text_extensions:
                continue
            
            # Verificar si necesita actualización
            if self._needs_caching(file_path):
                try:
                    self._cache_file_content(file_path)
                    files_processed += 1
                except Exception as e:
                    logger.warning(f"Error cacheando {file_path}: {e}")
            else:
                files_skipped += 1
        
        if files_processed > 0:
            logger.info(f"Alimentación automática: {files_processed} archivos procesados, {files_skipped} omitidos")
            self._save_cache_indexes()
    
    def _needs_caching(self, file_path: Path) -> bool:
        """Verifica si un archivo necesita ser cacheado"""
        
        file_str = str(file_path)
        
        # Verificar si ya está en el índice
        if file_str in self.file_path_index:
            # Verificar timestamp
            current_mtime = file_path.stat().st_mtime
            cached_mtime = self.file_timestamps.get(file_str, 0)
            
            return current_mtime > cached_mtime
        
        return True  # Archivo nuevo
    
    def _cache_file_content(self, file_path: Path):
        """Cachea el contenido de un archivo"""
        
        try:
            # Leer contenido
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return  # Archivo vacío
            
            # Generar hash del contenido
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Verificar si ya existe este contenido
            if content_hash in self.content_hash_index:
                # Solo actualizar timestamp
                self.file_timestamps[str(file_path)] = file_path.stat().st_mtime
                return
            
            # Crear chunks
            chunks = self._create_chunks(content)
            
            # Generar clave única
            cache_key = f"file_{content_hash[:16]}"
            
            # Crear item de cache
            cache_item = CacheItem(
                key=cache_key,
                content=content,
                chunks=chunks,
                metadata={
                    'file_extension': file_path.suffix,
                    'file_name': file_path.name,
                    'chunk_count': len(chunks),
                    'language': self._detect_language(file_path)
                },
                access_count=0,
                last_accessed=time.time(),
                created_at=time.time(),
                file_hash=content_hash,
                file_path=str(file_path),
                size_bytes=len(content.encode('utf-8'))
            )
            
            # Guardar en cache
            self._store_cache_item(cache_item)
            
            # Actualizar índices
            self.content_hash_index[content_hash] = cache_key
            self.file_path_index[str(file_path)] = cache_key
            self.file_timestamps[str(file_path)] = file_path.stat().st_mtime
            
            # Indexar keywords
            self._index_keywords(cache_key, content)
            
            logger.debug(f"Archivo cacheado: {file_path.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"Error cacheando archivo {file_path}: {e}")
    
    def _create_chunks(self, content: str) -> List[str]:
        """Crea chunks del contenido con solapamiento inteligente"""
        
        if len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # Si no es el último chunk, buscar punto de corte natural
            if end < len(content):
                # Buscar salto de línea cercano
                for i in range(min(100, len(content) - end)):
                    if content[end + i] == '\n':
                        end = end + i + 1
                        break
            
            chunk = content[start:end]
            chunks.append(chunk)
            
            # Calcular siguiente inicio con solapamiento
            if end >= len(content):
                break
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _detect_language(self, file_path: Path) -> str:
        """Detecta el lenguaje/tipo del archivo"""
        
        extension_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.html': 'html',
            '.css': 'css',
            '.xml': 'xml',
            '.txt': 'text'
        }
        
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    def _index_keywords(self, cache_key: str, content: str):
        """Indexa keywords del contenido para búsqueda rápida"""
        
        # Extraer palabras significativas
        import re
        words = re.findall(r'\b\w{3,}\b', content.lower())
        
        # Filtrar palabras comunes
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        significant_words = set()
        for word in words:
            if len(word) >= 4 and word not in stop_words:
                significant_words.add(word)
        
        # Indexar palabras significativas
        for word in significant_words:
            self.keyword_index[word].add(cache_key)
    
    def _store_cache_item(self, item: CacheItem):
        """Almacena item en el nivel de cache apropiado"""
        
        with self.lock:
            # Intentar L1 primero
            if len(self.l1_cache) < self.l1_size:
                self.l1_cache[item.key] = item
                return
            
            # Mover item menos usado de L1 a L2
            if self.l1_cache:
                lru_key = next(iter(self.l1_cache))
                lru_item = self.l1_cache.pop(lru_key)
                self._move_to_l2(lru_item)
            
            # Agregar nuevo item a L1
            self.l1_cache[item.key] = item
    
    def _move_to_l2(self, item: CacheItem):
        """Mueve item de L1 a L2"""
        
        if len(self.l2_cache) >= self.l2_size:
            # Mover item menos usado de L2 a disco
            lru_key = next(iter(self.l2_cache))
            lru_item = self.l2_cache.pop(lru_key)
            self._move_to_disk(lru_item)
        
        self.l2_cache[item.key] = item
    
    def _move_to_disk(self, item: CacheItem):
        """Mueve item de L2 a cache de disco"""
        
        try:
            disk_file = self.cache_directory / "disk" / f"{item.key}.pkl"
            with open(disk_file, 'wb') as f:
                pickle.dump(item, f)
            
            self.disk_cache_index[item.key] = str(disk_file)
            
            # Limpiar cache de disco si excede límite
            if len(self.disk_cache_index) > self.disk_size:
                self._cleanup_disk_cache()
                
        except Exception as e:
            logger.error(f"Error moviendo a disco: {e}")
    
    def _cleanup_disk_cache(self):
        """Limpia cache de disco eliminando items más antiguos"""
        
        try:
            # Obtener items con timestamps
            disk_items = []
            for key, file_path in list(self.disk_cache_index.items()):
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    disk_items.append((mtime, key, file_path))
                else:
                    # Archivo no existe, remover del índice
                    del self.disk_cache_index[key]
            
            # Ordenar por timestamp (más antiguos primero)
            disk_items.sort()
            
            # Eliminar 10% de los más antiguos
            items_to_remove = len(disk_items) // 10
            for _, key, file_path in disk_items[:items_to_remove]:
                try:
                    os.remove(file_path)
                    del self.disk_cache_index[key]
                except Exception as e:
                    logger.warning(f"Error eliminando {file_path}: {e}")
            
            logger.info(f"Cache de disco limpiado: {items_to_remove} items eliminados")
            
        except Exception as e:
            logger.error(f"Error limpiando cache de disco: {e}")
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Busca en el cache usando query inteligente"""
        
        with self.lock:
            self.metrics.total_requests += 1
            
            # Normalizar query
            query_lower = query.lower()
            query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
            
            # Buscar por keywords
            candidate_keys = set()
            for word in query_words:
                if word in self.keyword_index:
                    candidate_keys.update(self.keyword_index[word])
            
            if not candidate_keys:
                self.metrics.misses += 1
                return []
            
            # Obtener items candidatos
            results = []
            for key in candidate_keys:
                item = self._get_cache_item(key)
                if item:
                    # Calcular relevancia
                    relevance = self._calculate_relevance(query_words, item)
                    if relevance > 0:
                        results.append({
                            'key': key,
                            'content': item.content,
                            'chunks': item.chunks,
                            'metadata': item.metadata,
                            'relevance': relevance,
                            'file_path': item.file_path
                        })
            
            # Ordenar por relevancia
            results.sort(key=lambda x: x['relevance'], reverse=True)
            
            return results[:max_results]
    
    def _get_cache_item(self, key: str) -> Optional[CacheItem]:
        """Obtiene item del cache multinivel"""
        
        # Buscar en L1
        if key in self.l1_cache:
            item = self.l1_cache[key]
            item.access_count += 1
            item.last_accessed = time.time()
            # Mover al final (más reciente)
            self.l1_cache.move_to_end(key)
            self.metrics.l1_hits += 1
            return item
        
        # Buscar en L2
        if key in self.l2_cache:
            item = self.l2_cache.pop(key)
            item.access_count += 1
            item.last_accessed = time.time()
            # Promover a L1
            self._promote_to_l1(item)
            self.metrics.l2_hits += 1
            return item
        
        # Buscar en disco
        if key in self.disk_cache_index:
            try:
                disk_file = self.disk_cache_index[key]
                with open(disk_file, 'rb') as f:
                    item = pickle.load(f)
                
                item.access_count += 1
                item.last_accessed = time.time()
                # Promover a L2
                self._promote_to_l2(item)
                self.metrics.disk_hits += 1
                return item
                
            except Exception as e:
                logger.warning(f"Error cargando desde disco {key}: {e}")
                # Remover entrada inválida
                del self.disk_cache_index[key]
        
        return None
    
    def _promote_to_l1(self, item: CacheItem):
        """Promueve item a L1 cache"""
        
        if len(self.l1_cache) >= self.l1_size:
            # Mover LRU de L1 a L2
            lru_key = next(iter(self.l1_cache))
            lru_item = self.l1_cache.pop(lru_key)
            self._move_to_l2(lru_item)
        
        self.l1_cache[item.key] = item
    
    def _promote_to_l2(self, item: CacheItem):
        """Promueve item a L2 cache"""
        
        if len(self.l2_cache) >= self.l2_size:
            # Mover LRU de L2 a disco
            lru_key = next(iter(self.l2_cache))
            lru_item = self.l2_cache.pop(lru_key)
            self._move_to_disk(lru_item)
        
        self.l2_cache[item.key] = item
    
    def _calculate_relevance(self, query_words: Set[str], item: CacheItem) -> float:
        """Calcula relevancia de un item para la query"""
        
        content_lower = item.content.lower()
        content_words = set(re.findall(r'\b\w{3,}\b', content_lower))
        
        # Coincidencias exactas
        exact_matches = len(query_words.intersection(content_words))
        
        # Factor de frecuencia de acceso
        access_factor = min(1.0, item.access_count / 10)
        
        # Factor de recencia
        age_hours = (time.time() - item.last_accessed) / 3600
        recency_factor = max(0.1, 1.0 - (age_hours / 168))  # Decae en 1 semana
        
        # Relevancia combinada
        relevance = (exact_matches / len(query_words)) * 0.7 + access_factor * 0.2 + recency_factor * 0.1
        
        return relevance
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema de cache"""
        
        return {
            'hit_rate': self.metrics.hit_rate,
            'l1_hit_rate': self.metrics.l1_hit_rate,
            'total_requests': self.metrics.total_requests,
            'l1_hits': self.metrics.l1_hits,
            'l2_hits': self.metrics.l2_hits,
            'disk_hits': self.metrics.disk_hits,
            'misses': self.metrics.misses,
            'cache_sizes': {
                'l1': len(self.l1_cache),
                'l2': len(self.l2_cache),
                'disk': len(self.disk_cache_index)
            },
            'cache_limits': {
                'l1': self.l1_size,
                'l2': self.l2_size,
                'disk': self.disk_size
            },
            'total_indexed_files': len(self.file_path_index),
            'total_keywords': len(self.keyword_index)
        }
    
    def force_refresh(self):
        """Fuerza actualización completa del cache"""
        
        logger.info("Iniciando actualización forzada del cache...")
        self._scan_and_cache_directory()
        self._save_cache_indexes()
        logger.info("Actualización forzada completada")
    
    def shutdown(self):
        """Cierra el sistema de cache limpiamente"""
        
        self.stop_background = True
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
        
        self._save_cache_indexes()
        logger.info("Sistema de cache cerrado")

# Función de utilidad para testing
def test_cache_system():
    """Función de prueba del sistema de cache"""
    
    import tempfile
    import shutil
    
    # Crear directorio temporal
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Crear archivos de prueba
        (temp_dir / "test1.py").write_text("def hello(): return 'world'")
        (temp_dir / "test2.md").write_text("# Test\nThis is a test file")
        
        # Inicializar cache
        cache = IntelligentCacheSystem(
            source_directory=str(temp_dir),
            l1_size=5,
            l2_size=10,
            disk_size=50
        )
        
        # Esperar a que se cacheen los archivos
        time.sleep(2)
        
        # Buscar contenido
        results = cache.search("hello world")
        print(f"Resultados encontrados: {len(results)}")
        
        for result in results:
            print(f"- {result['file_path']}: {result['relevance']:.2f}")
        
        # Mostrar métricas
        metrics = cache.get_metrics()
        print(f"Métricas: {json.dumps(metrics, indent=2)}")
        
        cache.shutdown()
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cache_system()
