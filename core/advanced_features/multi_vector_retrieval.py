"""
Multi-Vector Retrieval (MVR) - Sistema de retrieval con múltiples vectores
Implementa búsqueda semántica con diferentes tipos de embeddings y estrategias de fusión
"""

import numpy as np
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json


class VectorType(Enum):
    SEMANTIC = "semantic"        # Embeddings semánticos generales
    KEYWORD = "keyword"          # Embeddings basados en keywords
    STRUCTURAL = "structural"    # Embeddings de estructura
    CONTEXTUAL = "contextual"    # Embeddings contextuales
    DENSE = "dense"             # Vectores densos tradicionales
    SPARSE = "sparse"           # Vectores sparse (TF-IDF like)


class FusionStrategy(Enum):
    WEIGHTED_SUM = "weighted_sum"
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    CombSUM = "combsum"
    MAX_SCORE = "max_score"
    NEURAL_FUSION = "neural_fusion"


@dataclass
class VectorDocument:
    doc_id: str
    content: str
    vectors: Dict[VectorType, np.ndarray]
    metadata: Dict[str, Any]
    vector_hashes: Dict[VectorType, str]


@dataclass
class SearchResult:
    doc_id: str
    content: str
    score: float
    vector_scores: Dict[VectorType, float]
    metadata: Dict[str, Any]
    explanation: Dict[str, Any]


class MultiVectorEmbedder:
    """Genera múltiples tipos de embeddings para un documento"""

    def __init__(self):
        # Simulamos diferentes embedders
        # En implementación real, aquí irían modelos como SentenceTransformer, etc.
        self.embedding_dims = {
            VectorType.SEMANTIC: 384,
            VectorType.KEYWORD: 256,
            VectorType.STRUCTURAL: 128,
            VectorType.CONTEXTUAL: 512,
            VectorType.DENSE: 768,
            VectorType.SPARSE: 1000
        }

    def generate_embeddings(self,
                          text: str,
                          vector_types: List[VectorType],
                          metadata: Dict[str, Any] = None) -> Dict[VectorType, np.ndarray]:
        """Genera múltiples tipos de embeddings para un texto"""

        embeddings = {}

        for vector_type in vector_types:
            if vector_type == VectorType.SEMANTIC:
                embeddings[vector_type] = self._generate_semantic_embedding(text)
            elif vector_type == VectorType.KEYWORD:
                embeddings[vector_type] = self._generate_keyword_embedding(text)
            elif vector_type == VectorType.STRUCTURAL:
                embeddings[vector_type] = self._generate_structural_embedding(text, metadata)
            elif vector_type == VectorType.CONTEXTUAL:
                embeddings[vector_type] = self._generate_contextual_embedding(text, metadata)
            elif vector_type == VectorType.DENSE:
                embeddings[vector_type] = self._generate_dense_embedding(text)
            elif vector_type == VectorType.SPARSE:
                embeddings[vector_type] = self._generate_sparse_embedding(text)

        return embeddings

    def _generate_semantic_embedding(self, text: str) -> np.ndarray:
        """Genera embedding semántico (simulado)"""
        # En implementación real: usar SentenceTransformer o similar
        np.random.seed(hash(text) % 2**32)
        return np.random.normal(0, 1, self.embedding_dims[VectorType.SEMANTIC]).astype(np.float32)

    def _generate_keyword_embedding(self, text: str) -> np.ndarray:
        """Genera embedding basado en keywords extraídos"""
        # Extrae keywords y crea embedding basado en ellos
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3][:20]  # Top 20 palabras relevantes

        # Crear vector basado en hash de keywords
        vector = np.zeros(self.embedding_dims[VectorType.KEYWORD])
        for i, keyword in enumerate(keywords):
            idx = hash(keyword) % len(vector)
            vector[idx] += 1.0 / (i + 1)  # Peso descendente

        # Normalizar
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).astype(np.float32)

    def _generate_structural_embedding(self, text: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Genera embedding basado en estructura del documento"""
        features = np.zeros(self.embedding_dims[VectorType.STRUCTURAL])

        # Características estructurales
        lines = text.split('\n')
        features[0] = len(lines) / 100.0  # Número de líneas normalizado
        features[1] = len(text) / 10000.0  # Longitud normalizada
        features[2] = text.count('.') / len(text) if text else 0  # Densidad de puntos
        features[3] = text.count('?') / len(text) if text else 0  # Densidad de preguntas
        features[4] = text.count('\n\n') / len(lines) if lines else 0  # Párrafos

        # Headers (si es markdown)
        features[5] = text.count('#') / len(text) if text else 0

        # Código (si contiene)
        features[6] = text.count('def ') / len(text) if text else 0
        features[7] = text.count('class ') / len(text) if text else 0

        # Listas
        features[8] = text.count('- ') / len(text) if text else 0
        features[9] = text.count('* ') / len(text) if text else 0

        # Rellenar resto con características derivadas
        for i in range(10, len(features)):
            features[i] = np.sin(i * 0.1) * np.sum(features[:10])

        return features.astype(np.float32)

    def _generate_contextual_embedding(self, text: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Genera embedding contextual considerando metadata"""
        # Combina embedding semántico con información contextual
        base_embedding = self._generate_semantic_embedding(text)

        # Extender a dimensión contextual
        contextual_embedding = np.zeros(self.embedding_dims[VectorType.CONTEXTUAL])
        contextual_embedding[:len(base_embedding)] = base_embedding

        # Agregar información contextual
        if metadata:
            context_features = []

            file_type = metadata.get('file_type')
            if file_type:
                context_features.append(hash(file_type) % 1000)

            timestamp = metadata.get('timestamp')
            if isinstance(timestamp, (int, float)) and timestamp:
                context_features.append(int(timestamp) % 1000)

            size = metadata.get('size')
            if isinstance(size, (int, float)) and size > 0:
                context_features.append(int(size) % 1000)

            domain = metadata.get('domain')
            if domain:
                context_features.append(hash(str(domain)) % 1000)

            # Rellenar características contextuales
            start_idx = len(base_embedding)
            for i, feature in enumerate(context_features):
                if start_idx + i < len(contextual_embedding):
                    contextual_embedding[start_idx + i] = feature / 1000.0

        return contextual_embedding.astype(np.float32)

    def _generate_dense_embedding(self, text: str) -> np.ndarray:
        """Genera embedding denso de alta dimensión"""
        # Simula un embedding como BERT/RoBERTa
        np.random.seed(hash(text + "dense") % 2**32)
        return np.random.normal(0, 1, self.embedding_dims[VectorType.DENSE]).astype(np.float32)

    def _generate_sparse_embedding(self, text: str) -> np.ndarray:
        """Genera embedding sparse tipo TF-IDF"""
        words = text.lower().split()
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Crear vector sparse
        vector = np.zeros(self.embedding_dims[VectorType.SPARSE])

        for word, count in word_counts.items():
            idx = hash(word) % len(vector)
            # TF-IDF simplificado: tf * log(N/df) - aquí solo TF
            vector[idx] += count / len(words)

        return vector.astype(np.float32)


class MultiVectorIndex:
    """Índice para múltiples tipos de vectores"""

    def __init__(self, vector_types: List[VectorType]):
        self.vector_types = vector_types
        self.documents: Dict[str, VectorDocument] = {}
        self.embedder = MultiVectorEmbedder()

        # Índices por tipo de vector (simulando FAISS/HNSW)
        self.indexes = {vt: [] for vt in vector_types}
        self.doc_ids_by_index = {vt: [] for vt in vector_types}

    def add_document(self,
                    doc_id: str,
                    content: str,
                    metadata: Dict[str, Any] = None) -> bool:
        """Añade un documento al índice con todos los tipos de vectores"""

        try:
            # Generar embeddings
            embeddings = self.embedder.generate_embeddings(
                content, self.vector_types, metadata
            )

            # Generar hashes de vectores para integridad
            vector_hashes = {}
            for vector_type, vector in embeddings.items():
                vector_bytes = vector.tobytes()
                vector_hashes[vector_type] = hashlib.md5(vector_bytes).hexdigest()

            # Crear documento vectorizado
            doc = VectorDocument(
                doc_id=doc_id,
                content=content,
                vectors=embeddings,
                metadata=metadata or {},
                vector_hashes=vector_hashes
            )

            # Almacenar documento
            self.documents[doc_id] = doc

            # Añadir a índices
            for vector_type in self.vector_types:
                self.indexes[vector_type].append(embeddings[vector_type])
                self.doc_ids_by_index[vector_type].append(doc_id)

            return True

        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")
            return False

    def remove_document(self, doc_id: str) -> bool:
        """Elimina un documento del índice"""

        if doc_id not in self.documents:
            return False

        try:
            # Remover de índices
            for vector_type in self.vector_types:
                if doc_id in self.doc_ids_by_index[vector_type]:
                    idx = self.doc_ids_by_index[vector_type].index(doc_id)
                    del self.indexes[vector_type][idx]
                    del self.doc_ids_by_index[vector_type][idx]

            # Remover documento
            del self.documents[doc_id]
            return True

        except Exception as e:
            print(f"Error removing document {doc_id}: {e}")
            return False


class MultiVectorRetriever:
    """Sistema de retrieval con múltiples vectores y estrategias de fusión"""

    def __init__(self,
                 vector_types: List[VectorType],
                 fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
                 vector_weights: Dict[VectorType, float] = None):

        self.vector_types = vector_types
        self.fusion_strategy = fusion_strategy
        self.vector_weights = vector_weights or self._default_weights()
        self.index = MultiVectorIndex(vector_types)

    def _default_weights(self) -> Dict[VectorType, float]:
        """Pesos por defecto para cada tipo de vector"""
        return {
            VectorType.SEMANTIC: 0.3,
            VectorType.KEYWORD: 0.2,
            VectorType.STRUCTURAL: 0.1,
            VectorType.CONTEXTUAL: 0.25,
            VectorType.DENSE: 0.1,
            VectorType.SPARSE: 0.05
        }

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Añade documento al índice"""
        return self.index.add_document(doc_id, content, metadata)

    def search(self,
              query: str,
              top_k: int = 10,
              query_metadata: Dict[str, Any] = None,
              vector_type_boost: Dict[VectorType, float] = None) -> List[SearchResult]:
        """
        Realiza búsqueda multi-vector

        Args:
            query: Query de búsqueda
            top_k: Número de resultados a devolver
            query_metadata: Metadata del query para embeddings contextuales
            vector_type_boost: Boost adicional por tipo de vector

        Returns:
            Lista de resultados ordenados por score
        """

        if not self.index.documents:
            return []

        # 1. Generar embeddings del query
        query_embeddings = self.index.embedder.generate_embeddings(
            query, self.vector_types, query_metadata
        )

        # 2. Calcular similarities por tipo de vector
        all_similarities = {}

        for vector_type in self.vector_types:
            similarities = self._calculate_similarities(
                query_embeddings[vector_type],
                self.index.indexes[vector_type],
                vector_type
            )
            all_similarities[vector_type] = similarities

        # 3. Aplicar fusion strategy
        final_scores = self._apply_fusion_strategy(
            all_similarities,
            vector_type_boost or {}
        )

        # 4. Obtener top-k resultados
        top_doc_indices = np.argsort(final_scores)[-top_k:][::-1]

        # 5. Crear objetos de resultado
        results = []
        for idx in top_doc_indices:
            if final_scores[idx] > 0:  # Solo resultados con score positivo
                doc_id = list(self.index.documents.keys())[idx]
                doc = self.index.documents[doc_id]

                # Calcular scores individuales
                vector_scores = {}
                for vector_type in self.vector_types:
                    vector_scores[vector_type] = float(all_similarities[vector_type][idx])

                # Crear explicación del score
                explanation = self._create_score_explanation(
                    vector_scores,
                    final_scores[idx],
                    vector_type_boost or {}
                )

                result = SearchResult(
                    doc_id=doc_id,
                    content=doc.content,
                    score=float(final_scores[idx]),
                    vector_scores=vector_scores,
                    metadata=doc.metadata,
                    explanation=explanation
                )

                results.append(result)

        return results

    def _calculate_similarities(self,
                               query_vector: np.ndarray,
                               doc_vectors: List[np.ndarray],
                               vector_type: VectorType) -> np.ndarray:
        """Calcula similarities entre query y documentos para un tipo de vector"""

        if not doc_vectors:
            return np.array([])

        # Convertir lista a matriz
        doc_matrix = np.vstack(doc_vectors)

        # Normalizar vectores
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)

        # Calcular cosine similarity
        similarities = np.dot(doc_norms, query_norm)

        # Aplicar transformaciones específicas por tipo
        if vector_type == VectorType.SPARSE:
            # Para sparse, usar también dot product sin normalización
            raw_dot = np.dot(doc_matrix, query_vector)
            similarities = 0.7 * similarities + 0.3 * (raw_dot / np.max(raw_dot + 1e-8))
        elif vector_type == VectorType.STRUCTURAL:
            # Para structural, aplicar sigmoid para suavizar
            similarities = 1 / (1 + np.exp(-5 * similarities))

        return np.maximum(similarities, 0)  # No negative similarities

    def _apply_fusion_strategy(self,
                              all_similarities: Dict[VectorType, np.ndarray],
                              vector_type_boost: Dict[VectorType, float]) -> np.ndarray:
        """Aplica la estrategia de fusión seleccionada"""

        if not all_similarities:
            return np.array([])

        # Obtener número de documentos
        n_docs = len(next(iter(all_similarities.values())))

        if self.fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum_fusion(all_similarities, vector_type_boost)

        elif self.fusion_strategy == FusionStrategy.RRF:
            return self._reciprocal_rank_fusion(all_similarities, vector_type_boost)

        elif self.fusion_strategy == FusionStrategy.CombSUM:
            return self._combsum_fusion(all_similarities, vector_type_boost)

        elif self.fusion_strategy == FusionStrategy.MAX_SCORE:
            return self._max_score_fusion(all_similarities, vector_type_boost)

        elif self.fusion_strategy == FusionStrategy.NEURAL_FUSION:
            return self._neural_fusion(all_similarities, vector_type_boost)

        else:
            # Default: weighted sum
            return self._weighted_sum_fusion(all_similarities, vector_type_boost)

    def _weighted_sum_fusion(self,
                            all_similarities: Dict[VectorType, np.ndarray],
                            boost: Dict[VectorType, float]) -> np.ndarray:
        """Fusión por suma ponderada"""

        n_docs = len(next(iter(all_similarities.values())))
        final_scores = np.zeros(n_docs)

        total_weight = 0
        for vector_type, similarities in all_similarities.items():
            weight = self.vector_weights.get(vector_type, 0.1)
            weight *= boost.get(vector_type, 1.0)

            final_scores += weight * similarities
            total_weight += weight

        # Normalizar
        if total_weight > 0:
            final_scores /= total_weight

        return final_scores

    def _reciprocal_rank_fusion(self,
                               all_similarities: Dict[VectorType, np.ndarray],
                               boost: Dict[VectorType, float]) -> np.ndarray:
        """Reciprocal Rank Fusion (RRF)"""

        n_docs = len(next(iter(all_similarities.values())))
        final_scores = np.zeros(n_docs)
        k = 60  # Parámetro RRF estándar

        for vector_type, similarities in all_similarities.items():
            # Obtener ranking
            ranked_indices = np.argsort(similarities)[::-1]

            # Calcular RRF scores
            rrf_scores = np.zeros(n_docs)
            for rank, doc_idx in enumerate(ranked_indices):
                rrf_scores[doc_idx] = 1.0 / (k + rank + 1)

            # Aplicar peso y boost
            weight = self.vector_weights.get(vector_type, 0.1)
            weight *= boost.get(vector_type, 1.0)

            final_scores += weight * rrf_scores

        return final_scores

    def _combsum_fusion(self,
                       all_similarities: Dict[VectorType, np.ndarray],
                       boost: Dict[VectorType, float]) -> np.ndarray:
        """CombSUM fusion - suma directa de scores normalizados"""

        n_docs = len(next(iter(all_similarities.values())))
        final_scores = np.zeros(n_docs)

        for vector_type, similarities in all_similarities.items():
            # Normalizar scores a [0,1]
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)

            if max_sim > min_sim:
                norm_similarities = (similarities - min_sim) / (max_sim - min_sim)
            else:
                norm_similarities = similarities

            # Aplicar boost
            weight = boost.get(vector_type, 1.0)
            final_scores += weight * norm_similarities

        return final_scores

    def _max_score_fusion(self,
                         all_similarities: Dict[VectorType, np.ndarray],
                         boost: Dict[VectorType, float]) -> np.ndarray:
        """Max Score fusion - toma el score máximo entre tipos"""

        n_docs = len(next(iter(all_similarities.values())))
        final_scores = np.zeros(n_docs)

        for vector_type, similarities in all_similarities.items():
            boosted_similarities = similarities * boost.get(vector_type, 1.0)
            final_scores = np.maximum(final_scores, boosted_similarities)

        return final_scores

    def _neural_fusion(self,
                      all_similarities: Dict[VectorType, np.ndarray],
                      boost: Dict[VectorType, float]) -> np.ndarray:
        """Neural fusion - combinación no lineal simple"""

        n_docs = len(next(iter(all_similarities.values())))

        # Crear matriz de características
        features = []
        for vector_type in self.vector_types:
            if vector_type in all_similarities:
                boosted_sims = all_similarities[vector_type] * boost.get(vector_type, 1.0)
                features.append(boosted_sims)

        if not features:
            return np.zeros(n_docs)

        feature_matrix = np.vstack(features).T  # (n_docs, n_vector_types)

        # Red neuronal simple (una capa)
        # Pesos simulados - en implementación real entrenar estos pesos
        weights = np.random.rand(len(features), 1) * 0.5 + 0.5
        bias = 0.1

        # Forward pass
        linear_output = np.dot(feature_matrix, weights).flatten() + bias

        # Activación sigmoid
        final_scores = 1 / (1 + np.exp(-linear_output))

        return final_scores

    def _create_score_explanation(self,
                                 vector_scores: Dict[VectorType, float],
                                 final_score: float,
                                 boost: Dict[VectorType, float]) -> Dict[str, Any]:
        """Crea explicación detallada del score"""

        explanation = {
            "final_score": final_score,
            "fusion_strategy": self.fusion_strategy.value,
            "vector_contributions": {}
        }

        total_weighted_score = 0
        for vector_type, score in vector_scores.items():
            weight = self.vector_weights.get(vector_type, 0.1)
            boost_factor = boost.get(vector_type, 1.0)
            effective_weight = weight * boost_factor
            contribution = effective_weight * score

            explanation["vector_contributions"][vector_type.value] = {
                "raw_score": score,
                "weight": weight,
                "boost": boost_factor,
                "effective_weight": effective_weight,
                "contribution": contribution
            }

            total_weighted_score += contribution

        explanation["total_weighted_score"] = total_weighted_score
        explanation["normalization_factor"] = final_score / total_weighted_score if total_weighted_score > 0 else 1.0

        return explanation

    def get_index_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice"""

        stats = {
            "total_documents": len(self.index.documents),
            "vector_types": [vt.value for vt in self.vector_types],
            "fusion_strategy": self.fusion_strategy.value,
            "vector_weights": {vt.value: w for vt, w in self.vector_weights.items()},
            "index_sizes": {}
        }

        for vector_type in self.vector_types:
            stats["index_sizes"][vector_type.value] = len(self.index.indexes[vector_type])

        return stats


# Funciones de conveniencia
def create_mvr_system(vector_types: List[str] = None,
                     fusion_strategy: str = "weighted_sum") -> MultiVectorRetriever:
    """Crea sistema MVR con configuración por defecto"""

    if vector_types is None:
        vector_types = ["semantic", "keyword", "contextual"]

    vt_enums = [VectorType(vt) for vt in vector_types]
    fs_enum = FusionStrategy(fusion_strategy)

    return MultiVectorRetriever(vt_enums, fs_enum)


if __name__ == "__main__":
    # Ejemplo de uso
    mvr = create_mvr_system()

    # Añadir documentos de ejemplo
    docs = [
        ("doc1", "Python es un lenguaje de programación versátil", {"type": "programming"}),
        ("doc2", "Machine learning utiliza algoritmos para aprender patrones", {"type": "ai"}),
        ("doc3", "Los vectores son estructuras matemáticas fundamentales", {"type": "math"}),
        ("doc4", "La búsqueda semántica mejora la relevancia", {"type": "search"}),
        ("doc5", "Los embeddings capturan significado semántico", {"type": "nlp"})
    ]

    for doc_id, content, metadata in docs:
        mvr.add_document(doc_id, content, metadata)

    # Realizar búsqueda
    results = mvr.search("algoritmos de aprendizaje automático", top_k=3)

    print("Resultados de búsqueda MVR:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.doc_id} (Score: {result.score:.4f})")
        print(f"   Contenido: {result.content}")
        print(f"   Scores por vector: {result.vector_scores}")
        print(f"   Metadata: {result.metadata}")

    # Mostrar estadísticas
    stats = mvr.get_index_stats()
    print(f"\nEstadísticas del índice MVR:")
    print(json.dumps(stats, indent=2))
