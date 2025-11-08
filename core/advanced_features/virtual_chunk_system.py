"""
Virtual Chunk System - Sistema de chunks virtuales con almacenamiento MP4 optimizado
Implementa la arquitectura de chunks virtuales que referencian contenido sin duplicación
con almacenamiento eficiente en formato MP4 como contenedor de vectores
"""

import os
import struct
import hashlib
import mmap
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, BinaryIO
from dataclasses import dataclass, asdict
from enum import Enum
import json
import zlib
import time
from concurrent.futures import ThreadPoolExecutor
import threading


class ChunkType(Enum):
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    BINARY = "binary"


@dataclass
class VirtualChunkMetadata:
    """Metadata para un chunk virtual"""
    chunk_id: str
    file_path: str
    start_offset: int
    end_offset: int
    size: int
    chunk_type: ChunkType
    vector_hash: str
    content_hash: str
    embedding_dim: int
    timestamp: float
    keywords: List[str]
    complexity_score: float

    def to_bytes(self) -> bytes:
        """Serializa metadata a bytes para almacenamiento eficiente"""
        data = {
            'chunk_id': self.chunk_id,
            'file_path': self.file_path,
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'size': self.size,
            'chunk_type': self.chunk_type.value,
            'vector_hash': self.vector_hash,
            'content_hash': self.content_hash,
            'embedding_dim': self.embedding_dim,
            'timestamp': self.timestamp,
            'keywords': self.keywords[:10],  # Máximo 10 keywords
            'complexity_score': self.complexity_score
        }
        json_data = json.dumps(data, separators=(',', ':')).encode('utf-8')
        return zlib.compress(json_data, level=6)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VirtualChunkMetadata':
        """Deserializa metadata desde bytes"""
        json_data = zlib.decompress(data)
        data_dict = json.loads(json_data.decode('utf-8'))

        return cls(
            chunk_id=data_dict['chunk_id'],
            file_path=data_dict['file_path'],
            start_offset=data_dict['start_offset'],
            end_offset=data_dict['end_offset'],
            size=data_dict['size'],
            chunk_type=ChunkType(data_dict['chunk_type']),
            vector_hash=data_dict['vector_hash'],
            content_hash=data_dict['content_hash'],
            embedding_dim=data_dict['embedding_dim'],
            timestamp=data_dict['timestamp'],
            keywords=data_dict['keywords'],
            complexity_score=data_dict['complexity_score']
        )


class VirtualChunk:
    """
    Chunk virtual que referencia contenido sin almacenar duplicados
    Solo almacena offset y metadata, el contenido se lee on-demand
    """

    def __init__(self, metadata: VirtualChunkMetadata, mp4_storage: 'MP4VectorStorage' = None):
        self.metadata = metadata
        self._mp4_storage = mp4_storage
        self._cached_text = None
        self._cached_vector = None
        self._access_count = 0
        self._last_access = time.time()

    def get_text(self, use_cache: bool = True) -> str:
        """
        Obtiene el texto del chunk leyendo desde el archivo original
        Utiliza cache para optimizar accesos repetidos
        """
        self._access_count += 1
        self._last_access = time.time()

        if use_cache and self._cached_text is not None:
            return self._cached_text

        try:
            with open(self.metadata.file_path, 'r', encoding='utf-8') as f:
                f.seek(self.metadata.start_offset)
                content = f.read(self.metadata.end_offset - self.metadata.start_offset)

                # Verificar integridad
                computed_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                if computed_hash != self.metadata.content_hash:
                    raise ValueError(f"Content hash mismatch for chunk {self.metadata.chunk_id}")

                if use_cache:
                    self._cached_text = content

                return content

        except Exception as e:
            raise RuntimeError(f"Failed to read chunk {self.metadata.chunk_id}: {e}")

    def get_vector(self, use_cache: bool = True) -> np.ndarray:
        """
        Obtiene el vector embedding del chunk desde MP4 storage
        """
        self._access_count += 1
        self._last_access = time.time()

        if use_cache and self._cached_vector is not None:
            return self._cached_vector

        if self._mp4_storage is None:
            raise RuntimeError("No MP4 storage available for vector retrieval")

        vector = self._mp4_storage.get_vector(self.metadata.chunk_id)

        if use_cache:
            self._cached_vector = vector

        return vector

    def get_context_window(self, window_size: int = 200) -> str:
        """
        Obtiene ventana de contexto alrededor del chunk
        """
        try:
            with open(self.metadata.file_path, 'r', encoding='utf-8') as f:
                # Calcular offsets de ventana
                start_window = max(0, self.metadata.start_offset - window_size)
                end_window = self.metadata.end_offset + window_size

                # Leer contenido con ventana
                f.seek(start_window)
                total_content = f.read(end_window - start_window)

                # Marcar el chunk principal
                chunk_start_in_window = self.metadata.start_offset - start_window
                chunk_end_in_window = chunk_start_in_window + self.metadata.size

                before = total_content[:chunk_start_in_window]
                chunk = total_content[chunk_start_in_window:chunk_end_in_window]
                after = total_content[chunk_end_in_window:]

                return f"[CONTEXT_BEFORE]{before}[/CONTEXT_BEFORE]\n[MAIN_CHUNK]{chunk}[/MAIN_CHUNK]\n[CONTEXT_AFTER]{after}[/CONTEXT_AFTER]"

        except Exception as e:
            return self.get_text()  # Fallback al contenido principal

    def invalidate_cache(self):
        """Invalida el cache del chunk"""
        self._cached_text = None
        self._cached_vector = None

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de uso del chunk"""
        return {
            'chunk_id': self.metadata.chunk_id,
            'access_count': self._access_count,
            'last_access': self._last_access,
            'cached_text': self._cached_text is not None,
            'cached_vector': self._cached_vector is not None,
            'size': self.metadata.size,
            'complexity': self.metadata.complexity_score
        }


class MP4VectorStorage:
    """
    Storage optimizado usando MP4 como contenedor de vectores
    No almacena texto, solo vectores embeddings y metadata
    """

    # MP4 Magic Numbers y Headers
    MP4_FTYP = b'ftyp'
    MP4_MDAT = b'mdat'
    MP4_MOOV = b'moov'
    MP4_CUSTOM = b'vctr'  # Custom atom para vectores

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.vector_index: Dict[str, Tuple[int, int]] = {}  # chunk_id -> (offset, size)
        self.metadata_index: Dict[str, VirtualChunkMetadata] = {}
        self._lock = threading.RLock()
        self._memory_map = None
        self._file_handle = None

        self._init_storage()

    def _init_storage(self):
        """Inicializa el almacenamiento MP4"""
        if not os.path.exists(self.storage_path):
            self._create_mp4_container()
        else:
            self._load_existing_container()

    def _create_mp4_container(self):
        """Crea un contenedor MP4 vacío para vectores"""
        with open(self.storage_path, 'wb') as f:
            # Escribir header FTYP (File Type Box)
            ftyp_data = b'vctr' + b'\x00' * 4 + b'vctr'  # Custom brand
            self._write_mp4_box(f, self.MP4_FTYP, ftyp_data)

            # Escribir MDAT placeholder (Media Data)
            self._write_mp4_box(f, self.MP4_MDAT, b'')

            # Escribir índice vacío
            index_data = json.dumps({
                'version': '1.0',
                'created': time.time(),
                'vector_index': {},
                'metadata_index': {}
            }).encode('utf-8')

            compressed_index = zlib.compress(index_data, level=9)
            self._write_mp4_box(f, self.MP4_CUSTOM, compressed_index)

    def _write_mp4_box(self, f: BinaryIO, box_type: bytes, data: bytes):
        """Escribe un box MP4 con el formato correcto"""
        size = len(data) + 8  # 4 bytes size + 4 bytes type + data
        f.write(struct.pack('>I', size))  # Big-endian uint32
        f.write(box_type)
        f.write(data)

    def _load_existing_container(self):
        """Carga un contenedor MP4 existente"""
        try:
            with open(self.storage_path, 'rb') as f:
                # Buscar el box de índice personalizado
                while True:
                    try:
                        size_data = f.read(4)
                        if len(size_data) < 4:
                            break

                        size = struct.unpack('>I', size_data)[0]
                        box_type = f.read(4)

                        if box_type == self.MP4_CUSTOM:
                            # Encontrado el índice
                            index_data = f.read(size - 8)
                            decompressed = zlib.decompress(index_data)
                            index_dict = json.loads(decompressed.decode('utf-8'))

                            # Cargar índices
                            self.vector_index = index_dict.get('vector_index', {})

                            # Reconstruir metadata index
                            metadata_raw = index_dict.get('metadata_index', {})
                            for chunk_id, meta_data in metadata_raw.items():
                                self.metadata_index[chunk_id] = VirtualChunkMetadata(**meta_data)

                            break
                        else:
                            # Saltar este box
                            f.seek(size - 8, 1)

                    except Exception as e:
                        print(f"Error reading MP4 box: {e}")
                        break

        except Exception as e:
            print(f"Error loading MP4 container: {e}")
            self._create_mp4_container()

    def store_vector(self, chunk_id: str, vector: np.ndarray, metadata: VirtualChunkMetadata):
        """
        Almacena un vector en el contenedor MP4
        """
        with self._lock:
            try:
                # Preparar vector para almacenamiento
                vector_bytes = vector.astype(np.float32).tobytes()
                vector_hash = hashlib.md5(vector_bytes).hexdigest()

                # Verificar si ya existe
                if chunk_id in self.vector_index:
                    print(f"Warning: Overwriting existing vector for chunk {chunk_id}")

                # Escribir vector al final del archivo
                with open(self.storage_path, 'r+b') as f:
                    f.seek(0, 2)  # Ir al final
                    vector_offset = f.tell()

                    # Header del vector: [chunk_id_len][chunk_id][vector_dim][vector_data]
                    chunk_id_bytes = chunk_id.encode('utf-8')
                    header = struct.pack('>II', len(chunk_id_bytes), len(vector))

                    f.write(header)
                    f.write(chunk_id_bytes)
                    f.write(vector_bytes)

                    vector_size = len(header) + len(chunk_id_bytes) + len(vector_bytes)

                # Actualizar índices
                self.vector_index[chunk_id] = (vector_offset, vector_size)
                self.metadata_index[chunk_id] = metadata

                # Actualizar metadata con hash
                metadata.vector_hash = vector_hash

                # Guardar índices actualizados
                self._save_indexes()

                return True

            except Exception as e:
                print(f"Error storing vector for chunk {chunk_id}: {e}")
                return False

    def get_vector(self, chunk_id: str) -> np.ndarray:
        """
        Obtiene un vector del almacenamiento MP4
        """
        with self._lock:
            if chunk_id not in self.vector_index:
                raise KeyError(f"Chunk {chunk_id} not found in vector storage")

            offset, size = self.vector_index[chunk_id]
            metadata = self.metadata_index[chunk_id]

            try:
                # Usar memory mapping para lectura eficiente
                if self._memory_map is None:
                    self._file_handle = open(self.storage_path, 'rb')
                    self._memory_map = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)

                # Leer header
                self._memory_map.seek(offset)
                chunk_id_len, vector_dim = struct.unpack('>II', self._memory_map.read(8))

                # Leer chunk_id (para verificación)
                stored_chunk_id = self._memory_map.read(chunk_id_len).decode('utf-8')
                if stored_chunk_id != chunk_id:
                    raise ValueError(f"Chunk ID mismatch: expected {chunk_id}, got {stored_chunk_id}")

                # Leer vector data
                vector_bytes = self._memory_map.read(vector_dim * 4)  # float32 = 4 bytes
                vector = np.frombuffer(vector_bytes, dtype=np.float32)

                # Verificar integridad
                computed_hash = hashlib.md5(vector_bytes).hexdigest()
                if computed_hash != metadata.vector_hash:
                    raise ValueError(f"Vector hash mismatch for chunk {chunk_id}")

                return vector

            except Exception as e:
                raise RuntimeError(f"Failed to read vector for chunk {chunk_id}: {e}")

    def _save_indexes(self):
        """Guarda los índices actualizados en el MP4"""
        try:
            # Preparar datos del índice
            metadata_serializable = {}
            for chunk_id, metadata in self.metadata_index.items():
                metadata_serializable[chunk_id] = asdict(metadata)
                metadata_serializable[chunk_id]['chunk_type'] = metadata.chunk_type.value

            index_data = {
                'version': '1.0',
                'updated': time.time(),
                'vector_index': self.vector_index,
                'metadata_index': metadata_serializable
            }

            compressed_index = zlib.compress(json.dumps(index_data).encode('utf-8'), level=9)

            # Reescribir solo la sección del índice
            # Por simplicidad, reescribimos todo el archivo
            # En implementación optimizada, se mantendría el MDAT y solo se actualizaría MOOV
            temp_path = self.storage_path + '.tmp'

            with open(temp_path, 'wb') as temp_f, open(self.storage_path, 'rb') as orig_f:
                # Copiar header FTYP
                orig_f.seek(0)
                size = struct.unpack('>I', orig_f.read(4))[0]
                orig_f.seek(0)
                temp_f.write(orig_f.read(size))

                # Copiar MDAT (datos de vectores)
                while True:
                    size_data = orig_f.read(4)
                    if len(size_data) < 4:
                        break

                    size = struct.unpack('>I', size_data)[0]
                    box_type = orig_f.read(4)

                    if box_type == self.MP4_CUSTOM:
                        # Saltar el índice viejo
                        orig_f.seek(size - 8, 1)
                        break
                    else:
                        # Copiar box completo
                        orig_f.seek(-8, 1)  # Retroceder
                        temp_f.write(orig_f.read(size))

                # Escribir nuevo índice
                self._write_mp4_box(temp_f, self.MP4_CUSTOM, compressed_index)

            # Reemplazar archivo original
            os.replace(temp_path, self.storage_path)

            # Invalidar memory map
            if self._memory_map:
                self._memory_map.close()
                self._memory_map = None
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None

        except Exception as e:
            print(f"Error saving indexes: {e}")

    def remove_vector(self, chunk_id: str) -> bool:
        """Elimina un vector del almacenamiento"""
        with self._lock:
            if chunk_id not in self.vector_index:
                return False

            try:
                # Remover de índices
                del self.vector_index[chunk_id]
                if chunk_id in self.metadata_index:
                    del self.metadata_index[chunk_id]

                # Guardar índices actualizados
                self._save_indexes()

                return True

            except Exception as e:
                print(f"Error removing vector {chunk_id}: {e}")
                return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del almacenamiento"""
        file_size = os.path.getsize(self.storage_path) if os.path.exists(self.storage_path) else 0

        vector_sizes = []
        total_dims = 0

        for metadata in self.metadata_index.values():
            vector_sizes.append(metadata.size)
            total_dims += metadata.embedding_dim

        return {
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'total_vectors': len(self.vector_index),
            'avg_vector_dim': total_dims / len(self.metadata_index) if self.metadata_index else 0,
            'total_chunks_size': sum(vector_sizes),
            'avg_chunk_size': np.mean(vector_sizes) if vector_sizes else 0,
            'storage_efficiency': (sum(vector_sizes) / file_size * 100) if file_size > 0 else 0
        }

    def __del__(self):
        """Cleanup al destruir el objeto"""
        if self._memory_map:
            self._memory_map.close()
        if self._file_handle:
            self._file_handle.close()


class VirtualChunkManager:
    """
    Administrador principal del sistema de chunks virtuales
    Coordina la creación, almacenamiento y retrieval de chunks
    """

    def __init__(self, storage_path: str):
        self.storage = MP4VectorStorage(storage_path)
        self.chunks: Dict[str, VirtualChunk] = {}
        self._lock = threading.RLock()
        self.stats = {
            'chunks_created': 0,
            'chunks_accessed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def create_virtual_chunk(self,
                           chunk_id: str,
                           file_path: str,
                           start_offset: int,
                           end_offset: int,
                           vector: np.ndarray,
                           chunk_type: ChunkType = ChunkType.TEXT,
                           keywords: List[str] = None,
                           complexity_score: float = 0.0) -> VirtualChunk:
        """
        Crea un nuevo chunk virtual
        """
        with self._lock:
            try:
                # Leer contenido para generar hash
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(start_offset)
                    content = f.read(end_offset - start_offset)
                    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

                # Crear metadata
                metadata = VirtualChunkMetadata(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    size=end_offset - start_offset,
                    chunk_type=chunk_type,
                    vector_hash="",  # Se establecerá en store_vector
                    content_hash=content_hash,
                    embedding_dim=len(vector),
                    timestamp=time.time(),
                    keywords=keywords or [],
                    complexity_score=complexity_score
                )

                # Almacenar vector
                if self.storage.store_vector(chunk_id, vector, metadata):
                    # Crear chunk virtual
                    chunk = VirtualChunk(metadata, self.storage)
                    self.chunks[chunk_id] = chunk
                    self.stats['chunks_created'] += 1

                    return chunk
                else:
                    raise RuntimeError(f"Failed to store vector for chunk {chunk_id}")

            except Exception as e:
                raise RuntimeError(f"Failed to create virtual chunk {chunk_id}: {e}")

    def get_chunk(self, chunk_id: str) -> Optional[VirtualChunk]:
        """Obtiene un chunk virtual por ID"""
        with self._lock:
            self.stats['chunks_accessed'] += 1

            if chunk_id in self.chunks:
                self.stats['cache_hits'] += 1
                return self.chunks[chunk_id]

            # Si no está en memoria, intentar cargar desde storage
            if chunk_id in self.storage.metadata_index:
                metadata = self.storage.metadata_index[chunk_id]
                chunk = VirtualChunk(metadata, self.storage)
                self.chunks[chunk_id] = chunk
                self.stats['cache_misses'] += 1
                return chunk

            return None

    def search_similar_chunks(self,
                            query_vector: np.ndarray,
                            top_k: int = 10,
                            chunk_type_filter: ChunkType = None) -> List[Tuple[VirtualChunk, float]]:
        """
        Busca chunks similares usando el vector query
        """
        similarities = []

        # Obtener todos los chunks que coincidan con el filtro
        candidate_chunks = []
        for chunk_id, metadata in self.storage.metadata_index.items():
            if chunk_type_filter is None or metadata.chunk_type == chunk_type_filter:
                candidate_chunks.append(chunk_id)

        # Calcular similitudes en paralelo
        def calculate_similarity(chunk_id):
            try:
                chunk = self.get_chunk(chunk_id)
                if chunk:
                    chunk_vector = chunk.get_vector()

                    # Cosine similarity
                    norm_query = query_vector / np.linalg.norm(query_vector)
                    norm_chunk = chunk_vector / np.linalg.norm(chunk_vector)
                    similarity = np.dot(norm_query, norm_chunk)

                    return (chunk, float(similarity))
            except Exception as e:
                print(f"Error calculating similarity for {chunk_id}: {e}")
            return None

        # Usar ThreadPoolExecutor para paralelización
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(calculate_similarity, candidate_chunks)
            similarities = [r for r in results if r is not None]

        # Ordenar por similitud y devolver top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def incremental_reindex(self, file_path: str) -> Dict[str, Any]:
        """
        Reindexado incremental inteligente para un archivo modificado
        """
        reindex_stats = {
            'chunks_removed': 0,
            'chunks_added': 0,
            'chunks_updated': 0,
            'processing_time': 0
        }

        start_time = time.time()

        with self._lock:
            try:
                # Encontrar chunks existentes para este archivo
                existing_chunks = []
                for chunk_id, metadata in self.storage.metadata_index.items():
                    if metadata.file_path == file_path:
                        existing_chunks.append(chunk_id)

                # Por simplicidad, removemos todos los chunks existentes del archivo
                # En implementación optimizada, se compararían hashes para determinar cambios
                for chunk_id in existing_chunks:
                    if self.remove_chunk(chunk_id):
                        reindex_stats['chunks_removed'] += 1

                print(f"Removed {len(existing_chunks)} existing chunks for {file_path}")
                print("Note: Automatic re-chunking would be implemented here")
                print("This would involve:")
                print("- Re-analyzing the file content")
                print("- Creating new chunks with updated offsets")
                print("- Generating new embeddings")
                print("- Storing updated vectors")

                reindex_stats['processing_time'] = time.time() - start_time
                return reindex_stats

            except Exception as e:
                reindex_stats['error'] = str(e)
                reindex_stats['processing_time'] = time.time() - start_time
                return reindex_stats

    def remove_chunk(self, chunk_id: str) -> bool:
        """Elimina un chunk virtual"""
        with self._lock:
            try:
                # Remover del storage
                if self.storage.remove_vector(chunk_id):
                    # Remover de memoria
                    if chunk_id in self.chunks:
                        del self.chunks[chunk_id]
                    return True
                return False

            except Exception as e:
                print(f"Error removing chunk {chunk_id}: {e}")
                return False

    def validate_chunk_integrity(self, chunk_id: str) -> Dict[str, Any]:
        """
        Valida la integridad de un chunk virtual
        """
        validation_result = {
            'chunk_id': chunk_id,
            'exists': False,
            'content_hash_valid': False,
            'vector_hash_valid': False,
            'file_accessible': False,
            'errors': []
        }

        try:
            chunk = self.get_chunk(chunk_id)
            if not chunk:
                validation_result['errors'].append(f"Chunk {chunk_id} not found")
                return validation_result

            validation_result['exists'] = True

            # Validar acceso al archivo
            try:
                content = chunk.get_text(use_cache=False)
                validation_result['file_accessible'] = True

                # Validar hash de contenido
                computed_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                if computed_hash == chunk.metadata.content_hash:
                    validation_result['content_hash_valid'] = True
                else:
                    validation_result['errors'].append("Content hash mismatch")

            except Exception as e:
                validation_result['errors'].append(f"File access error: {e}")

            # Validar vector
            try:
                vector = chunk.get_vector(use_cache=False)
                vector_bytes = vector.astype(np.float32).tobytes()
                computed_vector_hash = hashlib.md5(vector_bytes).hexdigest()

                if computed_vector_hash == chunk.metadata.vector_hash:
                    validation_result['vector_hash_valid'] = True
                else:
                    validation_result['errors'].append("Vector hash mismatch")

            except Exception as e:
                validation_result['errors'].append(f"Vector access error: {e}")

        except Exception as e:
            validation_result['errors'].append(f"General validation error: {e}")

        return validation_result

    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        storage_stats = self.storage.get_storage_stats()

        # Estadísticas de chunks en memoria
        memory_chunks = len(self.chunks)
        cached_text_count = sum(1 for chunk in self.chunks.values() if chunk._cached_text is not None)
        cached_vector_count = sum(1 for chunk in self.chunks.values() if chunk._cached_vector is not None)

        # Estadísticas de acceso
        total_accesses = sum(chunk._access_count for chunk in self.chunks.values())

        return {
            **storage_stats,
            'memory_chunks': memory_chunks,
            'cached_text_chunks': cached_text_count,
            'cached_vector_chunks': cached_vector_count,
            'total_chunk_accesses': total_accesses,
            'manager_stats': self.stats,
            'cache_hit_rate': (self.stats['cache_hits'] / max(1, self.stats['chunks_accessed'])) * 100
        }


# Funciones de utilidad
def create_virtual_chunk_system(storage_path: str) -> VirtualChunkManager:
    """Crea una instancia del sistema de chunks virtuales"""
    return VirtualChunkManager(storage_path)


def estimate_storage_savings(text_files: List[str],
                           avg_embedding_dim: int = 384) -> Dict[str, Any]:
    """
    Estima el ahorro de almacenamiento usando chunks virtuales vs almacenamiento tradicional
    """
    total_text_size = 0
    estimated_chunks = 0

    for file_path in text_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_text_size += file_size
            # Estimar chunks (asumiendo 800 chars promedio por chunk)
            estimated_chunks += max(1, file_size // 800)

    # Cálculos de almacenamiento
    # Tradicional: texto completo + vectores
    traditional_text_storage = total_text_size
    traditional_vector_storage = estimated_chunks * avg_embedding_dim * 4  # float32
    traditional_total = traditional_text_storage + traditional_vector_storage

    # Virtual chunks: solo vectores + metadata + índices
    virtual_vector_storage = estimated_chunks * avg_embedding_dim * 4
    virtual_metadata_storage = estimated_chunks * 200  # ~200 bytes por metadata comprimida
    virtual_index_storage = estimated_chunks * 50  # ~50 bytes por entrada de índice
    virtual_total = virtual_vector_storage + virtual_metadata_storage + virtual_index_storage

    savings_bytes = traditional_total - virtual_total
    savings_percentage = (savings_bytes / traditional_total * 100) if traditional_total > 0 else 0

    return {
        'traditional_storage': {
            'text_bytes': traditional_text_storage,
            'vector_bytes': traditional_vector_storage,
            'total_bytes': traditional_total,
            'total_mb': traditional_total / (1024 * 1024)
        },
        'virtual_storage': {
            'vector_bytes': virtual_vector_storage,
            'metadata_bytes': virtual_metadata_storage,
            'index_bytes': virtual_index_storage,
            'total_bytes': virtual_total,
            'total_mb': virtual_total / (1024 * 1024)
        },
        'savings': {
            'bytes': savings_bytes,
            'mb': savings_bytes / (1024 * 1024),
            'percentage': savings_percentage
        },
        'statistics': {
            'total_files': len(text_files),
            'total_text_size_mb': total_text_size / (1024 * 1024),
            'estimated_chunks': estimated_chunks,
            'avg_chunk_size': total_text_size // max(1, estimated_chunks)
        }
    }


if __name__ == "__main__":
    # Ejemplo completo de uso del sistema de chunks virtuales

    print("🚀 Virtual Chunk System - Demo")
    print("="*50)

    # 1. Crear el sistema
    storage_path = "demo_vectors.mp4"
    manager = create_virtual_chunk_system(storage_path)

    # 2. Crear archivo de ejemplo
    demo_file = "demo_content.txt"
    demo_content = """# Introducción a Machine Learning

Machine learning es una rama de la inteligencia artificial (IA) que se centra en el desarrollo de algoritmos y modelos estadísticos que permiten a las computadoras realizar tareas específicas sin ser explícitamente programadas para ello.

## Tipos de Machine Learning

### 1. Aprendizaje Supervisado
El aprendizaje supervisado utiliza datos etiquetados para entrenar modelos. Los algoritmos aprenden de ejemplos de entrada-salida para hacer predicciones sobre nuevos datos.

### 2. Aprendizaje No Supervisado
Este tipo de aprendizaje trabaja con datos sin etiquetas, buscando patrones ocultos o estructuras en los datos.

### 3. Aprendizaje por Refuerzo
El agente aprende a través de la interacción con un entorno, recibiendo recompensas o penalizaciones por sus acciones.

## Aplicaciones Prácticas

- Reconocimiento de imágenes
- Procesamiento de lenguaje natural
- Sistemas de recomendación
- Detección de fraudes
- Vehículos autónomos

## Algoritmos Populares

```python
# Ejemplo de regresión lineal
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[5]])
print(f"Predicción: {prediction}")
```

Esta introducción cubre los conceptos básicos del machine learning y sus aplicaciones más comunes en la industria actual.
"""

    # Escribir archivo de demo
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(demo_content)

    print(f"✅ Created demo file: {demo_file}")

    # 3. Crear chunks virtuales simulando un proceso de chunking
    demo_chunks = [
        {
            'id': 'chunk_intro',
            'start': 0,
            'end': 200,
            'type': ChunkType.MARKDOWN,
            'keywords': ['machine learning', 'inteligencia artificial', 'algoritmos']
        },
        {
            'id': 'chunk_supervised',
            'start': 200,
            'end': 400,
            'type': ChunkType.MARKDOWN,
            'keywords': ['aprendizaje supervisado', 'datos etiquetados', 'predicciones']
        },
        {
            'id': 'chunk_unsupervised',
            'start': 400,
            'end': 550,
            'type': ChunkType.MARKDOWN,
            'keywords': ['aprendizaje no supervisado', 'patrones', 'datos sin etiquetas']
        },
        {
            'id': 'chunk_code',
            'start': 900,
            'end': 1200,
            'type': ChunkType.CODE,
            'keywords': ['python', 'sklearn', 'regresion lineal', 'codigo']
        }
    ]

    print(f"\n📚 Creating {len(demo_chunks)} virtual chunks...")

    # Crear vectores de ejemplo (simulados)
    for chunk_info in demo_chunks:
        # Simular embedding (en caso real se usaría un modelo real)
        np.random.seed(hash(chunk_info['id']) % 2**32)
        demo_vector = np.random.normal(0, 1, 384).astype(np.float32)

        try:
            chunk = manager.create_virtual_chunk(
                chunk_id=chunk_info['id'],
                file_path=demo_file,
                start_offset=chunk_info['start'],
                end_offset=chunk_info['end'],
                vector=demo_vector,
                chunk_type=chunk_info['type'],
                keywords=chunk_info['keywords'],
                complexity_score=np.random.uniform(0.3, 0.9)
            )

            print(f"   ✅ Created chunk: {chunk_info['id']}")

        except Exception as e:
            print(f"   ❌ Error creating chunk {chunk_info['id']}: {e}")

    # 4. Demostrar retrieval de chunks
    print(f"\n🔍 Testing chunk retrieval...")

    for chunk_id in ['chunk_intro', 'chunk_code']:
        chunk = manager.get_chunk(chunk_id)
        if chunk:
            print(f"\n--- Chunk: {chunk_id} ---")
            print(f"Type: {chunk.metadata.chunk_type.value}")
            print(f"Keywords: {chunk.metadata.keywords}")
            print(f"Size: {chunk.metadata.size} bytes")
            print(f"Complexity: {chunk.metadata.complexity_score:.3f}")

            # Obtener contenido
            try:
                content = chunk.get_text()
                print(f"Content preview: {content[:100]}...")
            except Exception as e:
                print(f"Error reading content: {e}")

            # Obtener vector
            try:
                vector = chunk.get_vector()
                print(f"Vector shape: {vector.shape}")
                print(f"Vector preview: {vector[:5]}")
            except Exception as e:
                print(f"Error reading vector: {e}")

    # 5. Demostrar búsqueda por similitud
    print(f"\n🎯 Testing similarity search...")

    # Crear query vector (simulado)
    query_vector = np.random.normal(0, 1, 384).astype(np.float32)

    try:
        similar_chunks = manager.search_similar_chunks(query_vector, top_k=3)

        print("Top similar chunks:")
        for i, (chunk, similarity) in enumerate(similar_chunks, 1):
            print(f"  {i}. {chunk.metadata.chunk_id} - Similarity: {similarity:.4f}")
            print(f"     Keywords: {chunk.metadata.keywords[:3]}")

    except Exception as e:
        print(f"Error in similarity search: {e}")

    # 6. Validar integridad
    print(f"\n🛡️ Testing chunk integrity...")

    for chunk_id in ['chunk_intro', 'chunk_supervised']:
        validation = manager.validate_chunk_integrity(chunk_id)

        status = "✅" if not validation['errors'] else "❌"
        print(f"{status} Chunk {chunk_id}:")
        print(f"   Exists: {validation['exists']}")
        print(f"   Content hash valid: {validation['content_hash_valid']}")
        print(f"   Vector hash valid: {validation['vector_hash_valid']}")
        print(f"   File accessible: {validation['file_accessible']}")

        if validation['errors']:
            print(f"   Errors: {validation['errors']}")

    # 7. Mostrar estadísticas del sistema
    print(f"\n📊 System Statistics:")
    stats = manager.get_system_stats()

    print(f"   Storage file: {storage_path}")
    print(f"   File size: {stats['file_size_mb']:.2f} MB")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Memory chunks: {stats['memory_chunks']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"   Storage efficiency: {stats['storage_efficiency']:.1f}%")

    # 8. Estimar ahorros
    print(f"\n💾 Storage Savings Analysis:")
    savings = estimate_storage_savings([demo_file])

    print(f"   Traditional storage: {savings['traditional_storage']['total_mb']:.2f} MB")
    print(f"   Virtual chunk storage: {savings['virtual_storage']['total_mb']:.2f} MB")
    print(f"   Savings: {savings['savings']['mb']:.2f} MB ({savings['savings']['percentage']:.1f}%)")

    # 9. Cleanup
    print(f"\n🧹 Cleaning up demo files...")

    try:
        if os.path.exists(demo_file):
            os.remove(demo_file)
            print(f"   ✅ Removed {demo_file}")

        if os.path.exists(storage_path):
            os.remove(storage_path)
            print(f"   ✅ Removed {storage_path}")

    except Exception as e:
        print(f"   ❌ Cleanup error: {e}")

    print(f"\n🎉 Virtual Chunk System Demo Completed!")
    print("="*50)
