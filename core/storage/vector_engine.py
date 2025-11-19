"""
Vector Engine for MCP v5
Handles HNSW indexing, vector search, and embedding operations
"""

import numpy as np
import hnswlib
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
import logging
import pickle

logger = logging.getLogger(__name__)


class VectorEngine:
    """
    Manages vector embeddings and HNSW-based similarity search
    """
    
    def __init__(self, config: Dict):
        """
        Initialize vector engine
        
        Args:
            config: Configuration dict with embedding and HNSW params
        """
        self.config = config
        self.dimension = config.get('embedding', {}).get('dimension', 384)
        self.model_name = config.get('embedding', {}).get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.normalize = config.get('embedding', {}).get('normalize', True)
        self.dtype = config.get('embedding', {}).get('dtype', 'float16')
        
        # HNSW parameters
        hnsw_config = config.get('hnsw', {})
        self.ef_construction = hnsw_config.get('ef_construction', 200)
        self.M = hnsw_config.get('M', 16)
        self.ef_search = hnsw_config.get('ef_search', 50)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # HNSW index
        self.index: Optional[hnswlib.Index] = None
        self.id_to_chunk_id: Dict[int, str] = {}
        self.chunk_id_to_id: Dict[str, int] = {}
        
        logger.info(f"VectorEngine initialized with dimension={self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text
        
        Args:
            text: Input text
        
        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if self.normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        if self.dtype == 'float16':
            embedding = embedding.astype(np.float16)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts
        
        Args:
            texts: List of input texts
        
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        if self.dtype == 'float16':
            embeddings = embeddings.astype(np.float16)
        
        return embeddings
    
    def create_index(self, num_elements: int):
        """
        Create new HNSW index
        
        Args:
            num_elements: Expected number of vectors
        """
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.index.set_ef(self.ef_search)
        
        logger.info(f"Created HNSW index for {num_elements} elements")
    
    def add_vectors(self, vectors: np.ndarray, chunk_ids: List[str]):
        """
        Add vectors to HNSW index
        
        Args:
            vectors: Array of vectors (N x dimension)
            chunk_ids: List of chunk IDs corresponding to vectors
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        # Convert float16 to float32 for HNSW
        if vectors.dtype == np.float16:
            vectors = vectors.astype(np.float32)
        
        # Create internal IDs
        internal_ids = np.arange(len(chunk_ids))
        
        # Add to index
        self.index.add_items(vectors, internal_ids)
        
        # Update mappings
        for internal_id, chunk_id in zip(internal_ids, chunk_ids):
            self.id_to_chunk_id[int(internal_id)] = chunk_id
            self.chunk_id_to_id[chunk_id] = int(internal_id)
        
        logger.info(f"Added {len(vectors)} vectors to index")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            (chunk_ids, distances) tuple
        """
        if self.index is None:
            raise ValueError("Index not initialized")
        
        # Convert float16 to float32 for search
        if query_vector.dtype == np.float16:
            query_vector = query_vector.astype(np.float32)
        
        # Reshape for HNSW
        query_vector = query_vector.reshape(1, -1)
        
        # Search
        labels, distances = self.index.knn_query(query_vector, k=top_k)
        
        # Convert internal IDs to chunk IDs
        chunk_ids = [self.id_to_chunk_id[int(label)] for label in labels[0]]
        scores = [float(1.0 - dist) for dist in distances[0]]  # Convert distance to similarity
        
        return chunk_ids, scores
    
    def serialize_index(self) -> bytes:
        """
        Serialize HNSW index to bytes
        
        Returns:
            Serialized index as bytes
        """
        if self.index is None:
            return b''
        
        # Save index to temporary file then read as bytes
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.index.save_index(tmp_path)
            
            with open(tmp_path, 'rb') as f:
                index_bytes = f.read()
            
            # Also serialize the ID mappings
            mappings = {
                'id_to_chunk_id': self.id_to_chunk_id,
                'chunk_id_to_id': self.chunk_id_to_id
            }
            mappings_bytes = pickle.dumps(mappings)
            
            # Combine with separator
            separator = len(index_bytes).to_bytes(8, 'big')
            return separator + index_bytes + mappings_bytes
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def load_index_from_bytes(self, data: bytes, num_elements: int):
        """
        Load HNSW index from bytes
        
        Args:
            data: Serialized index data
            num_elements: Number of elements in index
        """
        import tempfile
        import os
        
        # Extract separator
        separator = int.from_bytes(data[:8], 'big')
        
        # Extract index bytes and mappings
        index_bytes = data[8:8+separator]
        mappings_bytes = data[8+separator:]
        
        # Load mappings
        mappings = pickle.loads(mappings_bytes)
        self.id_to_chunk_id = mappings['id_to_chunk_id']
        self.chunk_id_to_id = mappings['chunk_id_to_id']
        
        # Load index from temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(index_bytes)
            tmp_path = tmp.name
        
        try:
            self.index = hnswlib.Index(space='cosine', dim=self.dimension)
            self.index.load_index(tmp_path, max_elements=num_elements)
            self.index.set_ef(self.ef_search)
            
            logger.info(f"Loaded HNSW index with {len(self.id_to_chunk_id)} vectors")
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'ready',
            'num_vectors': len(self.id_to_chunk_id),
            'dimension': self.dimension,
            'ef_construction': self.ef_construction,
            'M': self.M,
            'ef_search': self.ef_search,
            'model': self.model_name
        }
