"""Orquestador principal de características avanzadas.

Este módulo implementa el `AdvancedFeaturesOrchestrator`, encargado de coordinar
los subsistemas descritos en `feature.md`:

* Dynamic Chunking adaptativo
* Query Expansion automática
* Multi-Vector Retrieval con fusiones configurables
* Confidence Calibration dinámica
* Virtual Chunk System con almacenamiento MP4 (opcional)

El orquestador expone una API asincrónica que permite procesar consultas de
forma coordinada, gestionar caches multinivel y recopilar métricas de uso para
optimizar la configuración en ejecuciones posteriores.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import AdvancedConfig, ProcessingMode
from .confidence_calibration import (
    ConfidenceScore,
    DynamicConfidenceCalibrator,
    create_calibrator,
)
from .dynamic_chunking import AdaptiveChunk, DynamicChunkingSystem
from .multi_vector_retrieval import (
    MultiVectorRetriever,
    SearchResult,
    VectorType,
    create_mvr_system,
)
from .query_expansion import AutoQueryExpander, QueryExpansion
from .virtual_chunk_system import (
    ChunkType,
    VirtualChunk,
    VirtualChunkManager,
    create_virtual_chunk_system,
)


class FeatureState(Enum):
    """Estado de ejecución de una característica."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Resultado estructurado del procesamiento avanzado."""

    query: str
    expanded_queries: List[str] = field(default_factory=list)
    expansion_details: Optional[QueryExpansion] = None
    chunks: List[AdaptiveChunk] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    confidence_scores: List[ConfidenceScore] = field(default_factory=list)
    feature_status: Dict[str, FeatureState] = field(default_factory=dict)
    processing_time: float = 0.0
    cache_hit: bool = False
    failure_modes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedFeaturesOrchestrator:
    """Coordinador principal de las características avanzadas del MCP."""

    _DEFAULT_CACHE_ENTRIES = 100

    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.processing_mode = config.processing_mode

        # Subsistemas configurables
        self.chunking_system: Optional[DynamicChunkingSystem] = None
        if config.enable_dynamic_chunking:
            self.chunking_system = DynamicChunkingSystem()

        self.query_expander: Optional[AutoQueryExpander] = None
        if config.enable_query_expansion:
            self.query_expander = AutoQueryExpander()

        self.mvr_system: Optional[MultiVectorRetriever] = None
        if config.enable_mvr:
            self.mvr_system = create_mvr_system(
                self._default_vector_types(), self._default_fusion_strategy()
            )

        self.confidence_calibrator: Optional[DynamicConfidenceCalibrator] = None
        if config.enable_confidence_calibration:
            self.confidence_calibrator = create_calibrator()

        self.virtual_chunk_manager: Optional[VirtualChunkManager] = None
        if config.enable_virtual_chunks:
            self.virtual_chunk_manager = create_virtual_chunk_system(
                str(self._default_storage_path())
            )

        # Cache multinivel (simple LRU en memoria + contadores de uso)
        self._query_cache: "OrderedDict[str, ProcessingResult]" = OrderedDict()
        self._cache_limit = max(self._DEFAULT_CACHE_ENTRIES, config.max_search_results * 5)

        # Métricas operativas
        self._operations_count = 0
        self._feedback_history: deque = deque(maxlen=2000)
        self._indexed_documents: set[str] = set()
        self._metrics: Dict[str, Any] = {
            "avg_processing_time_ms": 0.0,
            "last_operation_ts": None,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
        }
        self._feature_usage: Dict[str, int] = {
            "dynamic_chunking": 0,
            "multi_vector_retrieval": 0,
            "virtual_chunks": 0,
            "query_expansion": 0,
            "confidence_calibration": 0,
        }
        self._error_counts: Dict[str, int] = {
            "dynamic_chunking": 0,
            "multi_vector_retrieval": 0,
            "virtual_chunks": 0,
            "query_expansion": 0,
            "confidence_calibration": 0,
        }

        # Sincronización para operaciones concurrentes
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrent_operations))

        # Umbrales dinámicos
        self._confidence_threshold = self._resolve_confidence_threshold()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    async def process_advanced(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Ejecuta el pipeline completo para una consulta."""

        start_time = time.time()
        context = context or {}
        documents = documents or []

        cache_key = self._make_cache_key(query, documents, context)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result.processing_time = time.time() - start_time
            cached_result.cache_hit = True
            self._register_operation(cached=True, duration=cached_result.processing_time)
            return cached_result

        feature_status: Dict[str, FeatureState] = self._initial_feature_status()
        failure_modes: List[str] = []

        expansion: Optional[QueryExpansion] = None
        if self.query_expander:
            try:
                expansion = await asyncio.to_thread(
                    self.query_expander.expand_query,
                    query,
                    max_expansions=self.config.max_expansions,
                    domain_hint=context.get("domain"),
                )
                feature_status["query_expansion"] = FeatureState.ENABLED
                self._record_feature_event("query_expansion", success=True)
            except Exception:  # pragma: no cover - fallback defensivo
                feature_status["query_expansion"] = FeatureState.ERROR
                self._record_feature_event("query_expansion", success=False)

        chunk_results: List[AdaptiveChunk] = []
        chunk_origins: Dict[str, Dict[str, Any]] = {}
        if self.chunking_system and documents:
            chunk_results, chunk_origins = await self._generate_chunks(
                documents, feature_status
            )
            status = feature_status.get("dynamic_chunking")
            if status == FeatureState.ENABLED and chunk_results:
                self._record_feature_event("dynamic_chunking", success=True)
            elif status == FeatureState.ERROR:
                self._record_feature_event("dynamic_chunking", success=False)

        if self.virtual_chunk_manager and chunk_results:
            self._register_virtual_chunks(
                documents, chunk_results, chunk_origins, feature_status
            )

        if self.mvr_system:
            self._ensure_documents_indexed(documents)

        mvr_results = await self._run_retrieval(query, expansion, context, feature_status)

        confidence_scores: List[ConfidenceScore] = []
        if self.confidence_calibrator and mvr_results:
            try:
                confidence_scores = self._calibrate_results(mvr_results)
                feature_status["confidence_calibration"] = FeatureState.ENABLED
                self._record_feature_event("confidence_calibration", success=True)
            except Exception:
                confidence_scores = []
                feature_status["confidence_calibration"] = FeatureState.ERROR
                self._record_feature_event("confidence_calibration", success=False)
        elif self.confidence_calibrator:
            feature_status["confidence_calibration"] = FeatureState.SKIPPED

        failure_modes.extend(self._detect_failure_modes(query, expansion, mvr_results))

        processing_time = time.time() - start_time
        result = ProcessingResult(
            query=query,
            expanded_queries=list(expansion.expanded_queries) if expansion else [],
            expansion_details=expansion,
            chunks=chunk_results,
            search_results=mvr_results,
            confidence_scores=confidence_scores,
            feature_status=feature_status,
            processing_time=processing_time,
            failure_modes=failure_modes,
            metadata={
                "context": context,
                "processing_mode": self.processing_mode.value,
                "documents_indexed": len(documents),
            },
        )

        self._store_cached_result(cache_key, result)
        self._register_operation(cached=False, duration=processing_time)

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Retorna un resumen del estado actual del orquestador."""

        feature_status = {
            "dynamic_chunking": self._feature_state_value(self.chunking_system),
            "multi_vector_retrieval": self._feature_state_value(self.mvr_system),
            "virtual_chunks": self._feature_state_value(self.virtual_chunk_manager),
            "query_expansion": self._feature_state_value(self.query_expander),
            "confidence_calibration": self._feature_state_value(self.confidence_calibrator),
        }

        return {
            "config": {
                "processing_mode": self.processing_mode.value,
                "enabled_features": [
                    k for k, v in feature_status.items() if v == FeatureState.ENABLED.value
                ],
                "max_concurrent_operations": self.config.max_concurrent_operations,
                "max_search_results": self.config.max_search_results,
            },
            "feature_status": feature_status,
            "statistics": {
                "total_operations": self._operations_count,
                "avg_processing_time_ms": self._metrics["avg_processing_time_ms"],
                "cache_hits": self._metrics["cache_hits"],
                "cache_misses": self._metrics["cache_misses"],
                "confidence_threshold": self._confidence_threshold,
                "cache_hit_rate": self._metrics.get("cache_hit_rate", 0.0),
                "feature_usage": dict(self._feature_usage),
                "error_counts": dict(self._error_counts),
            },
        }

    def add_feedback(
        self,
        query: str,
        result_doc_id: str,
        relevance_score: float,
        was_helpful: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra feedback del usuario y actualiza calibradores."""

        context = context or {}
        payload = {
            "query": query,
            "result_doc_id": result_doc_id,
            "relevance_score": relevance_score,
            "was_helpful": was_helpful,
            "context": context,
            "timestamp": time.time(),
        }
        self._feedback_history.append(payload)

        if self.confidence_calibrator:
            try:
                self.confidence_calibrator.add_feedback(
                    predicted_confidence=float(min(max(relevance_score, 0.0), 1.0)),
                    actual_correctness=bool(was_helpful),
                    context=context,
                )
            except Exception:  # pragma: no cover - protección
                pass

    def optimize_configuration(self) -> Dict[str, Any]:
        """Calcula recomendaciones de ajuste basadas en métricas de uso."""

        recommendations: List[str] = []
        hit_rate = self._metrics.get("cache_hit_rate", 0.0)
        if hit_rate < 50 and len(self._query_cache) == self._cache_limit:
            recommendations.append(
                "Incrementa cache_size_mb para mejorar el hit rate de cache."
            )
        if self.confidence_calibrator and len(self._feedback_history) < 50:
            recommendations.append(
                "Recopila más feedback para mejorar la calibración de confianza."
            )
        if self.virtual_chunk_manager is None and self.config.enable_virtual_chunks:
            recommendations.append(
                "Verifica permisos de escritura para habilitar virtual chunks."
            )

        return {
            "current_performance": {
                "avg_processing_time_ms": self._metrics["avg_processing_time_ms"],
                "avg_processing_time": self._metrics["avg_processing_time_ms"] / 1000.0,
                "cache_hit_rate": hit_rate,
                "operations": self._operations_count,
            },
            "recommendations": recommendations,
            "auto_applied": [],
        }

    # ------------------------------------------------------------------
    # Métodos internos auxiliares
    # ------------------------------------------------------------------
    async def _generate_chunks(
        self,
        documents: List[Dict[str, Any]],
        feature_status: Dict[str, FeatureState],
    ) -> Tuple[List[AdaptiveChunk], Dict[str, Dict[str, Any]]]:
        """Genera chunks adaptativos con concurrencia controlada."""

        chunk_lists: List[List[AdaptiveChunk]] = []
        origin_map: Dict[str, Dict[str, Any]] = {}
        for index, document in enumerate(documents):
            await self._semaphore.acquire()
            try:
                chunks = await asyncio.to_thread(
                    self.chunking_system.adaptive_chunking,
                    text=document.get("content", ""),
                    file_path=document.get("path", f"doc_{index}.txt"),
                )
                for chunk in chunks:
                    origin_map[chunk.metadata.chunk_id] = document
                chunk_lists.append(chunks)
            except Exception:
                feature_status["dynamic_chunking"] = FeatureState.ERROR
                self._record_feature_event("dynamic_chunking", success=False)
            finally:
                self._semaphore.release()

        if feature_status.get("dynamic_chunking") != FeatureState.ERROR:
            feature_status["dynamic_chunking"] = FeatureState.ENABLED

        flattened = [chunk for chunks in chunk_lists for chunk in chunks]
        return flattened, origin_map

    def _register_virtual_chunks(
        self,
        documents: List[Dict[str, Any]],
        chunks: List[AdaptiveChunk],
        chunk_origins: Dict[str, Dict[str, Any]],
        feature_status: Dict[str, FeatureState],
    ) -> None:
        """Registra chunks en el sistema virtual si es posible."""

        if not self.virtual_chunk_manager:
            return

        registered_any = False
        error_occurred = False

        for chunk in chunks:
            document = chunk_origins.get(chunk.metadata.chunk_id)
            if not document:
                continue

            file_path = document.get("path")
            if not file_path or not os.path.exists(file_path):
                continue

            try:
                start_offset, end_offset = self._locate_offsets_in_file(
                    file_path, chunk.content
                )
                vector = self._build_virtual_vector(chunk.content)
                chunk_identifier = f"{document.get('id', file_path)}:{chunk.metadata.chunk_id}"
                self.virtual_chunk_manager.create_virtual_chunk(
                    chunk_id=chunk_identifier,
                    file_path=file_path,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    vector=vector,
                    chunk_type=self._infer_chunk_type(document),
                    keywords=chunk.metadata.keywords,
                    complexity_score=chunk.metadata.complexity_score,
                )
                registered_any = True
            except Exception:
                error_occurred = True

        if registered_any:
            feature_status["virtual_chunks"] = FeatureState.ENABLED
            self._record_feature_event("virtual_chunks", success=True)
        elif error_occurred:
            feature_status["virtual_chunks"] = FeatureState.ERROR
            self._record_feature_event("virtual_chunks", success=False)
        else:
            feature_status["virtual_chunks"] = FeatureState.SKIPPED

    async def _run_retrieval(
        self,
        query: str,
        expansion: Optional[QueryExpansion],
        context: Dict[str, Any],
        feature_status: Dict[str, FeatureState],
    ) -> List[SearchResult]:
        """Ejecuta búsquedas multi-vector y fusiona resultados."""

        if not self.mvr_system:
            feature_status["multi_vector_retrieval"] = FeatureState.SKIPPED
            return []

        search_queries = [query]
        if expansion:
            search_queries.extend(expansion.expanded_queries[:3])

        results_map: Dict[str, SearchResult] = {}
        for search_query in search_queries:
            await self._semaphore.acquire()
            try:
                partial_results = await asyncio.to_thread(
                    self.mvr_system.search,
                    search_query,
                    top_k=self.config.max_search_results,
                    query_metadata=context,
                )
            except Exception:
                feature_status["multi_vector_retrieval"] = FeatureState.ERROR
                self._record_feature_event("multi_vector_retrieval", success=False)
                partial_results = []
            finally:
                self._semaphore.release()

            for result in partial_results:
                stored = results_map.get(result.doc_id)
                if stored is None or result.score > stored.score:
                    results_map[result.doc_id] = result

        aggregated = sorted(results_map.values(), key=lambda r: r.score, reverse=True)
        feature_status["multi_vector_retrieval"] = (
            FeatureState.ENABLED if aggregated else FeatureState.SKIPPED
        )

        status = feature_status.get("multi_vector_retrieval")
        if status == FeatureState.ENABLED:
            self._record_feature_event("multi_vector_retrieval", success=True)

        return aggregated

    def _calibrate_results(self, results: List[SearchResult]) -> List[ConfidenceScore]:
        """Calibra los resultados usando el sistema de confianza."""

        calibrated: List[ConfidenceScore] = []
        for result in results[: self.config.max_search_results]:
            raw_score = float(max(0.0, min(1.0, result.score)))
            try:
                calibrated_score = self.confidence_calibrator.calibrate_confidence(
                    raw_score,
                    context={"doc_id": result.doc_id},
                )
            except Exception:
                calibrated_score = ConfidenceScore(
                    raw_score=raw_score,
                    calibrated_score=raw_score,
                    confidence_level=None,  # type: ignore[arg-type]
                    calibration_method=self.confidence_calibrator.current_best_method,  # type: ignore[union-attr]
                    uncertainty_estimate=0.5,
                )
            calibrated.append(calibrated_score)

        return calibrated

    def _detect_failure_modes(
        self,
        query: str,
        expansion: Optional[QueryExpansion],
        results: List[SearchResult],
    ) -> List[str]:
        """Identifica posibles modos de fallo para la consulta."""

        failures: List[str] = []
        if not query.strip():
            failures.append("EMPTY_QUERY")
        if not results:
            failures.append("NO_RESULTS")
        elif results[0].score < self._confidence_threshold:
            failures.append("LOW_CONFIDENCE_RESULT")
        if expansion and expansion.confidence_score < 0.3:
            failures.append("WEAK_EXPANSION")
        return failures

    def _ensure_documents_indexed(self, documents: Iterable[Dict[str, Any]]) -> None:
        """Añade documentos al índice MVR si aún no existen."""

        if not self.mvr_system:
            return

        for idx, document in enumerate(documents):
            doc_id = str(document.get("id", f"doc-{idx}"))
            if doc_id in self._indexed_documents:
                continue

            content = document.get("content")
            if not content:
                continue

            metadata = {
                "path": document.get("path"),
                "file_type": document.get("type"),
                "domain": document.get("domain"),
                "timestamp": document.get("timestamp"),
                "size": len(content),
            }
            try:
                self.mvr_system.add_document(doc_id, content, metadata)
                self._indexed_documents.add(doc_id)
                self._record_feature_event("document_indexing", success=True)
            except Exception:
                self._record_feature_event("document_indexing", success=False)
                continue

    def _register_operation(self, cached: bool, duration: float) -> None:
        """Actualiza métricas agregadas tras una operación."""

        self._operations_count += 1
        metric_key = "cache_hits" if cached else "cache_misses"
        self._metrics[metric_key] += 1
        total_time = (
            self._metrics["avg_processing_time_ms"] * (self._operations_count - 1)
            + duration * 1000
        )
        self._metrics["avg_processing_time_ms"] = total_time / max(self._operations_count, 1)
        total_cache_requests = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        if total_cache_requests:
            self._metrics["cache_hit_rate"] = (
                self._metrics["cache_hits"] / total_cache_requests * 100
            )
        else:
            self._metrics["cache_hit_rate"] = 0.0
        self._metrics["last_operation_ts"] = time.time()

    def _make_cache_key(
        self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Construye un identificador único para el cache de consultas."""

        doc_signature = tuple(
            sorted(str(doc.get("id", idx)) for idx, doc in enumerate(documents))
        )
        context_items = tuple(sorted((k, str(v)) for k, v in context.items()))
        payload = f"{query}|{doc_signature}|{context_items}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[ProcessingResult]:
        """Obtiene una copia del resultado almacenado en cache."""

        result = self._query_cache.get(cache_key)
        if result is None:
            return None
        self._query_cache.move_to_end(cache_key)
        return self._clone_result(result)

    def _store_cached_result(self, cache_key: str, result: ProcessingResult) -> None:
        """Guarda una copia del resultado en el cache LRU."""

        self._query_cache[cache_key] = self._clone_result(result)
        self._query_cache.move_to_end(cache_key)
        if len(self._query_cache) > self._cache_limit:
            self._query_cache.popitem(last=False)

    def _clone_result(self, result: ProcessingResult) -> ProcessingResult:
        """Crea una copia desacoplada de un resultado."""

        cloned_chunks = [replace(chunk) for chunk in result.chunks]
        cloned_results = [replace(res) for res in result.search_results]
        cloned_scores = [replace(score) for score in result.confidence_scores]
        return ProcessingResult(
            query=result.query,
            expanded_queries=list(result.expanded_queries),
            expansion_details=result.expansion_details,
            chunks=cloned_chunks,
            search_results=cloned_results,
            confidence_scores=cloned_scores,
            feature_status=dict(result.feature_status),
            processing_time=result.processing_time,
            cache_hit=result.cache_hit,
            failure_modes=list(result.failure_modes),
            metadata=dict(result.metadata),
        )

    def _initial_feature_status(self) -> Dict[str, FeatureState]:
        """Inicializa el estado de las características declaradas."""

        return {
            "dynamic_chunking": FeatureState.DISABLED,
            "multi_vector_retrieval": FeatureState.DISABLED,
            "virtual_chunks": FeatureState.DISABLED,
            "query_expansion": FeatureState.DISABLED,
            "confidence_calibration": FeatureState.DISABLED,
        }

    def _feature_state_value(self, component: Any) -> str:
        return FeatureState.ENABLED.value if component else FeatureState.DISABLED.value

    def _default_vector_types(self) -> List[str]:
        if self.processing_mode == ProcessingMode.FAST:
            return [VectorType.SEMANTIC.value, VectorType.KEYWORD.value]
        if self.processing_mode == ProcessingMode.COMPREHENSIVE:
            return [vt.value for vt in VectorType]
        return [
            VectorType.SEMANTIC.value,
            VectorType.KEYWORD.value,
            VectorType.CONTEXTUAL.value,
        ]

    def _default_fusion_strategy(self) -> str:
        if self.processing_mode == ProcessingMode.FAST:
            return "max_score"
        if self.processing_mode == ProcessingMode.COMPREHENSIVE:
            return "reciprocal_rank_fusion"
        return "weighted_sum"

    def _default_storage_path(self) -> Path:
        base = Path(__file__).resolve().parent / "storage"
        base.mkdir(parents=True, exist_ok=True)
        return base / "context_vectors.mp4"

    def _resolve_confidence_threshold(self) -> float:
        if self.processing_mode == ProcessingMode.FAST:
            return 0.65
        if self.processing_mode == ProcessingMode.COMPREHENSIVE:
            return 0.8
        return 0.72

    def _build_virtual_vector(self, text: str, dim: int = 384) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, dim).astype(np.float32)

    def _locate_offsets_in_file(self, file_path: str, snippet: str) -> Tuple[int, int]:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        start_offset = content.find(snippet)
        if start_offset == -1:
            start_offset = 0
        end_offset = start_offset + len(snippet)
        return start_offset, end_offset

    def _record_feature_event(self, feature: str, success: bool) -> None:
        if success:
            self._feature_usage[feature] = self._feature_usage.get(feature, 0) + 1
        else:
            self._error_counts[feature] = self._error_counts.get(feature, 0) + 1

    def _infer_chunk_type(self, document: Dict[str, Any]) -> ChunkType:
        mapping = {
            "code": ChunkType.CODE,
            "markdown": ChunkType.MARKDOWN,
            "json": ChunkType.JSON,
        }
        doc_type = str(document.get("type", "text")).lower()
        return mapping.get(doc_type, ChunkType.TEXT)
