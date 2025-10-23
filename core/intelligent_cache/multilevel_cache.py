#!/usr/bin/env python3
"""
🚀 Cache Inteligente Multinivel
L1: 100 items en memoria (acceso instantáneo)
L2: 1000 items en disco SSD (< 5ms)
L3: 10000+ items comprimidos (< 50ms)
Algoritmo LRU con scoring de relevancia
"""
import json
import time
import hashlib
import sqlite3
import zstandard as zstd
import msgpack
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
import threading
import os
from pathlib import Path

class CacheItem:
    """Item del cache con metadata"""
    def __init__(self, key: str, data: Any, score: float = 1.0):
        self.key = key
        self.data = data
        self.score = score
        self.access_count = 1
        self.last_accessed = time.time()
        self.created_at = time.time()
        self.size_bytes = len(str(data))
    
    def update_access(self, score_boost: float = 0.1):
        """Actualiza estadísticas de acceso"""
        self.access_count += 1
        self.last_accessed = time.time()
        self.score = min(10.0, self.score + score_boost)
    
    def calculate_relevance_score(self) -> float:
        """Calcula score de relevancia para LRU inteligente"""
        time_decay = max(0.1, 1.0 - (time.time() - self.last_accessed) / 3600)  # Decay por hora
        frequency_boost = min(2.0, self.access_count / 10.0)  # Boost por frecuencia
        return self.score * time_decay * frequency_boost

class L1MemoryCache:
    """Cache L1 - Memoria RAM (100 items, acceso instantáneo)"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene item del cache L1"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                item.update_access()
                # Mover al final (más reciente)
                self.cache.move_to_end(key)
                self.hits += 1
                return item.data
            
            self.misses += 1
            return None
    
    def put(self, key: str, data: Any, score: float = 1.0) -> None:
        """Almacena item en cache L1"""
        with self.lock:
            if key in self.cache:
                # Actualizar existente
                self.cache[key].data = data
                self.cache[key].update_access()
                self.cache.move_to_end(key)
            else:
                # Nuevo item
                if len(self.cache) >= self.max_size:
                    self._evict_lru()
                
                self.cache[key] = CacheItem(key, data, score)
    
    def _evict_lru(self) -> None:
        """Expulsa el item menos relevante"""
        if not self.cache:
            return
        
        # Encontrar item con menor score de relevancia
        min_score = float('inf')
        lru_key = None
        
        for key, item in self.cache.items():
            relevance = item.calculate_relevance_score()
            if relevance < min_score:
                min_score = relevance
                lru_key = key
        
        if lru_key:
            del self.cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache L1"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'level': 'L1_Memory',
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': round(hit_rate, 2),
                'memory_usage_mb': sum(item.size_bytes for item in self.cache.values()) / (1024 * 1024)
            }

class L2DiskCache:
    """Cache L2 - Disco SSD (1000 items, < 5ms)"""
    
    def __init__(self, cache_dir: str = "cache_l2", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.db_path = self.cache_dir / "l2_index.db"
        self.compressor = zstd.ZstdCompressor(level=1)  # Compresión rápida
        self.decompressor = zstd.ZstdDecompressor()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self._init_database()
    
    def _init_database(self):
        """Inicializa base de datos de índice"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS l2_cache (
                key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 1,
                last_accessed REAL DEFAULT (julianday('now')),
                created_at REAL DEFAULT (julianday('now')),
                size_bytes INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON l2_cache(last_accessed)
        ''')
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene item del cache L2"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, access_count, score
                FROM l2_cache 
                WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            
            if result:
                file_path, access_count, score = result
                
                # Leer archivo comprimido
                try:
                    with open(self.cache_dir / file_path, 'rb') as f:
                        compressed_data = f.read()
                    
                    packed_data = self.decompressor.decompress(compressed_data)
                    data = msgpack.unpackb(packed_data, raw=False)
                    
                    # Actualizar estadísticas
                    cursor.execute('''
                        UPDATE l2_cache 
                        SET access_count = ?, last_accessed = julianday('now'), score = ?
                        WHERE key = ?
                    ''', (access_count + 1, min(10.0, score + 0.1), key))
                    
                    conn.commit()
                    conn.close()
                    
                    self.hits += 1
                    return data
                    
                except (FileNotFoundError, Exception):
                    # Archivo corrupto o no encontrado, limpiar entrada
                    cursor.execute('DELETE FROM l2_cache WHERE key = ?', (key,))
                    conn.commit()
            
            conn.close()
            self.misses += 1
            return None
    
    def put(self, key: str, data: Any, score: float = 1.0) -> None:
        """Almacena item en cache L2"""
        with self.lock:
            # Verificar si necesitamos limpiar espacio
            self._cleanup_if_needed()
            
            # Generar nombre de archivo único
            file_hash = hashlib.md5(key.encode()).hexdigest()
            file_path = f"{file_hash}.cache"
            
            # Comprimir y guardar datos
            packed_data = msgpack.packb(data, use_bin_type=True)
            compressed_data = self.compressor.compress(packed_data)
            
            with open(self.cache_dir / file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Actualizar índice
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO l2_cache 
                (key, file_path, score, size_bytes, last_accessed)
                VALUES (?, ?, ?, ?, julianday('now'))
            ''', (key, file_path, score, len(compressed_data)))
            
            conn.commit()
            conn.close()
    
    def _cleanup_if_needed(self) -> None:
        """Limpia cache si excede el tamaño máximo"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM l2_cache')
        current_size = cursor.fetchone()[0]
        
        if current_size >= self.max_size:
            # Eliminar 20% de los items menos relevantes
            cleanup_count = int(self.max_size * 0.2)
            
            cursor.execute('''
                SELECT key, file_path, 
                       (score * (julianday('now') - last_accessed + 1) * access_count) as relevance
                FROM l2_cache 
                ORDER BY relevance ASC 
                LIMIT ?
            ''', (cleanup_count,))
            
            to_delete = cursor.fetchall()
            
            for key, file_path, _ in to_delete:
                # Eliminar archivo
                try:
                    os.remove(self.cache_dir / file_path)
                except FileNotFoundError:
                    pass
                
                # Eliminar entrada de DB
                cursor.execute('DELETE FROM l2_cache WHERE key = ?', (key,))
            
            conn.commit()
        
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache L2"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*), SUM(size_bytes), AVG(access_count)
                FROM l2_cache
            ''')
            
            stats = cursor.fetchone()
            conn.close()
            
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'level': 'L2_Disk',
                'size': stats[0] or 0,
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': round(hit_rate, 2),
                'disk_usage_mb': (stats[1] or 0) / (1024 * 1024),
                'avg_access_count': round(stats[2] or 0, 2)
            }

class L3CompressedCache:
    """Cache L3 - Comprimido (10000+ items, < 50ms)"""
    
    def __init__(self, db_path: str = "cache_l3.db", max_size: int = 10000):
        self.db_path = db_path
        self.max_size = max_size
        self.compressor = zstd.ZstdCompressor(level=6)  # Compresión alta
        self.decompressor = zstd.ZstdDecompressor()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self._init_database()
    
    def _init_database(self):
        """Inicializa base de datos L3"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS l3_cache (
                key TEXT PRIMARY KEY,
                compressed_data BLOB NOT NULL,
                score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 1,
                last_accessed REAL DEFAULT (julianday('now')),
                created_at REAL DEFAULT (julianday('now')),
                original_size INTEGER DEFAULT 0,
                compressed_size INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_l3_last_accessed ON l3_cache(last_accessed)
        ''')
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene item del cache L3"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT compressed_data, access_count, score
                FROM l3_cache 
                WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            
            if result:
                compressed_data, access_count, score = result
                
                try:
                    # Descomprimir datos
                    packed_data = self.decompressor.decompress(compressed_data)
                    data = msgpack.unpackb(packed_data, raw=False)
                    
                    # Actualizar estadísticas
                    cursor.execute('''
                        UPDATE l3_cache 
                        SET access_count = ?, last_accessed = julianday('now'), score = ?
                        WHERE key = ?
                    ''', (access_count + 1, min(10.0, score + 0.05), key))
                    
                    conn.commit()
                    conn.close()
                    
                    self.hits += 1
                    return data
                    
                except Exception:
                    # Datos corruptos, eliminar entrada
                    cursor.execute('DELETE FROM l3_cache WHERE key = ?', (key,))
                    conn.commit()
            
            conn.close()
            self.misses += 1
            return None
    
    def put(self, key: str, data: Any, score: float = 1.0) -> None:
        """Almacena item en cache L3"""
        with self.lock:
            # Comprimir datos
            packed_data = msgpack.packb(data, use_bin_type=True)
            compressed_data = self.compressor.compress(packed_data)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar si necesitamos limpiar espacio
            cursor.execute('SELECT COUNT(*) FROM l3_cache')
            current_size = cursor.fetchone()[0]
            
            if current_size >= self.max_size:
                self._cleanup_old_entries(cursor, int(self.max_size * 0.1))
            
            # Insertar nuevo item
            cursor.execute('''
                INSERT OR REPLACE INTO l3_cache 
                (key, compressed_data, score, original_size, compressed_size, last_accessed)
                VALUES (?, ?, ?, ?, ?, julianday('now'))
            ''', (key, compressed_data, score, len(packed_data), len(compressed_data)))
            
            conn.commit()
            conn.close()
    
    def _cleanup_old_entries(self, cursor, cleanup_count: int) -> None:
        """Limpia entradas antiguas"""
        cursor.execute('''
            DELETE FROM l3_cache 
            WHERE key IN (
                SELECT key FROM l3_cache 
                ORDER BY (score * access_count * (julianday('now') - last_accessed + 1)) ASC 
                LIMIT ?
            )
        ''', (cleanup_count,))
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache L3"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*), SUM(compressed_size), SUM(original_size), AVG(access_count)
                FROM l3_cache
            ''')
            
            stats = cursor.fetchone()
            conn.close()
            
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            compression_ratio = 0
            if stats[2] and stats[1]:  # original_size y compressed_size
                compression_ratio = (1 - stats[1] / stats[2]) * 100
            
            return {
                'level': 'L3_Compressed',
                'size': stats[0] or 0,
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': round(hit_rate, 2),
                'compressed_size_mb': (stats[1] or 0) / (1024 * 1024),
                'original_size_mb': (stats[2] or 0) / (1024 * 1024),
                'compression_ratio_percent': round(compression_ratio, 2),
                'avg_access_count': round(stats[3] or 0, 2)
            }

class MultiLevelIntelligentCache:
    """Cache Inteligente Multinivel con LRU y scoring de relevancia"""
    
    def __init__(self, 
                 l1_size: int = 100, 
                 l2_size: int = 1000, 
                 l3_size: int = 10000,
                 cache_dir: str = "intelligent_cache"):
        
        self.l1_cache = L1MemoryCache(l1_size)
        self.l2_cache = L2DiskCache(f"{cache_dir}/l2", l2_size)
        self.l3_cache = L3CompressedCache(f"{cache_dir}/l3.db", l3_size)
        
        self.total_requests = 0
        self.total_hits = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene item del cache multinivel"""
        self.total_requests += 1
        
        # Intentar L1 (memoria)
        data = self.l1_cache.get(key)
        if data is not None:
            self.total_hits += 1
            return data
        
        # Intentar L2 (disco)
        data = self.l2_cache.get(key)
        if data is not None:
            # Promover a L1 para acceso futuro
            self.l1_cache.put(key, data, score=2.0)  # Score boost por promoción
            self.total_hits += 1
            return data
        
        # Intentar L3 (comprimido)
        data = self.l3_cache.get(key)
        if data is not None:
            # Promover a L2 y L1
            self.l2_cache.put(key, data, score=1.5)
            self.l1_cache.put(key, data, score=1.5)
            self.total_hits += 1
            return data
        
        return None
    
    def put(self, key: str, data: Any, score: float = 1.0) -> None:
        """Almacena item en todos los niveles del cache"""
        # Almacenar en todos los niveles
        self.l1_cache.put(key, data, score)
        self.l2_cache.put(key, data, score)
        self.l3_cache.put(key, data, score)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Estadísticas completas del cache multinivel"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        
        overall_hit_rate = (self.total_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'overall': {
                'total_requests': self.total_requests,
                'total_hits': self.total_hits,
                'overall_hit_rate_percent': round(overall_hit_rate, 2),
                'target_hit_rate_percent': 85.0,
                'performance_status': '✅ EXCELLENT' if overall_hit_rate >= 85 else 
                                    '🟡 GOOD' if overall_hit_rate >= 70 else '🔴 NEEDS IMPROVEMENT'
            },
            'levels': {
                'L1': l1_stats,
                'L2': l2_stats,
                'L3': l3_stats
            }
        }
    
    def cleanup_all_levels(self) -> Dict[str, int]:
        """Limpia todos los niveles del cache"""
        # L1 se limpia automáticamente por LRU
        # L2 y L3 tienen limpieza automática, pero podemos forzar una limpieza manual
        
        return {
            'L1': 0,  # L1 no necesita limpieza manual
            'L2': 0,  # L2 se limpia automáticamente
            'L3': 0   # L3 se limpia automáticamente
        }

# Instancia global del cache
intelligent_cache = MultiLevelIntelligentCache()

def get_cache_instance() -> MultiLevelIntelligentCache:
    """Obtiene la instancia global del cache inteligente"""
    return intelligent_cache
