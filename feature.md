# MCP Hub Features - v5 & v6 Specifications

## Version Overview

- **v5.0.0:** Production - Pure retrieval system
- **v6.0.0:** Planned - Retrieval + session memory

---

## v5 Features (Current - Production Ready)

### Core Principles

1. Memory Only - No business logic
2. Single Source of Truth - model.md, checklist.md, changelog.md only
3. Retrieval-First - Return evidence, never invent
4. Provenance Mandatory - Every response includes source metadata
5. Anti-Hallucination - Confidence thresholds and abstention

### Storage System

**MP4-Based Vector Storage**
- Custom ISO BMFF (MP4) container format
- Structure:
  - ftyp: File type identifier (mcpv)
  - moov/udta/mcpi: Index metadata (JSON)
  - mdat: Vector embeddings + HNSW index
- Virtual chunks reference source files (no text duplication)
- Memory-mapped I/O for efficiency
- 96% storage reduction vs traditional methods

**VirtualChunk Architecture**
- Stores only: chunk_id, file_path, line_range, vector_offset
- Reads text on-demand from source MD files
- SHA256 hash for integrity validation
- No text duplication = minimal storage

### Vector Engine

**Embeddings**
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Normalization: L2
- Dtype: float16 (50% size reduction)

**HNSW Indexing**
- Space: cosine similarity
- ef_construction: 200
- M: 16
- ef_search: 50
- Serialization: pickle format in MP4 mdat

**Performance**
- Index build: ~30s for 50MB text
- Query latency: 20-50ms average
- Memory footprint: 100-200MB runtime

### Retrieval

**Semantic Chunking**
- Min tokens: 150
- Max tokens: 450
- Overlap: 25%
- Preserves structure (headers, lists)

**Query Processing**
- Embedding generation
- HNSW similarity search
- Score filtering (min_score configurable)
- Top-K ranking (default: 8)

**Advanced Features**
- Dynamic chunking (semantic-aware)
- Query expansion (lexical variations)
- Multi-vector retrieval (multiple perspectives)
- Confidence calibration (per query type)

### Anti-Hallucination

**Confidence Thresholds**
- Factual: 0.78
- Procedural: 0.72
- Conceptual: 0.65
- Temporal: 0.85

**Abstention Policy**
- If best_score < min_score: return "No sufficient information"
- Never generate content without evidence
- Explicit "NO_INFO" responses

**Provenance**
- File path
- Line numbers (start-end)
- Similarity score
- Section name
- Chunk ID
- Snapshot hash

**Audit Logging**
- Format: JSONL
- Location: logs/audit.jsonl
- Fields: timestamp, query, results_count, abstained, elapsed_ms, top_score

### API (MCP Protocol)

**Endpoints**

get_context
```
Input: {query, top_k, min_score}
Output: {content, metadata: {provenance, results_count, time_ms}}
```

validate_response
```
Input: {candidate_text, evidence_ids}
Output: {verified, contradictions, missing_claims}
```

index_status
```
Input: {}
Output: {version, snapshot_hash, num_chunks, uptime}
```

---

## v6 Features (Planned - Roadmap)

### Session Memory (OpenAI Patterns)

**TrimmingSession**
- Keep last N turns verbatim
- Configurable max_turns (default: 8)
- Turn = user message + assistant + tools until next user
- Use case: Independent tasks per session

**SummarizingSession**
- Compress older turns into summary
- Keep last N turns verbatim
- Configurable: keep_last_n_turns, context_limit
- Use case: Long development workflows

**Session Storage**
- Format: JSONL per session
- Location: data/sessions/{session_id}.jsonl
- Persistence: Auto-save on each turn
- Retention: Configurable (default: 30 days)

### Code Structure Indexing

**Entity Index**
- Functions: {name, module, signature, line_range, dependencies}
- Classes: {name, module, methods, line_range}
- Indexing: AST parsing (Python, JS, TS)
- Storage: JSON at data/code_index/function_map.json

**Module Graph**
- Imports/Exports tracking
- Dependency relationships
- Related features/tickets linking
- Storage: JSON at data/code_index/module_graph.json

**Benefits**
- Storage efficient: Names only, not full code
- Fast lookup: "payment function" → process_payment
- Context linking: Related entities discovery

### Contextual Query Resolution

**Reference Detection**
- Pattern matching: "that function", "the bug", "previous error"
- Entity extraction from recent turns
- Pronoun resolution

**Entity Tracking**
- Track mentions across turns
- Build mention graph: entity → [turns]
- Resolve ambiguous references

**Query Expansion**
- Concrete entity substitution
- Context-aware reformulation
- Expanded query generation

### Development Cycle Tracking

**Session Types**
- FEATURE_IMPLEMENTATION
- BUG_FIXING
- CODE_REVIEW
- REFACTORING
- GENERAL

**Session Metadata**
- created_at, status, type
- entities_mentioned: [function/class names]
- files_modified: [paths]
- commits: [hashes]
- related_sessions: [session_ids]

**Cross-Session Coordination**
- Link related sessions
- Shared entity tracking
- Multi-session queries

### Enhanced API (v6)

**New Endpoints**

sessions/create
```
Input: {session_type, metadata}
Output: {session_id, created_at}
```

get_context (enhanced)
```
Input: {query, session_id, resolve_references}
Output: {content, resolved_entities, session_context_used, provenance}
```

sessions/{id}/summary
```
Input: {}
Output: {turns, entities_discussed, summary, status}
```

---

## Technical Specifications

### Storage Formats

**MP4 Structure**
- ftyp box: 12 bytes (brand: mcpv)
- moov box: Variable (contains udta)
  - udta box: Variable (contains mcpi)
    - mcpi box: Compressed JSON index
- mdat box: Variable
  - 8 bytes: separator (vector blob size)
  - N bytes: vector blob (float16)
  - M bytes: HNSW index (serialized)

**Session Storage (JSONL)**
```json
{"turn_id": 1, "role": "user", "content": "...", "timestamp": "..."}
{"turn_id": 2, "role": "assistant", "content": "...", "metadata": {...}}
```

**Code Index (JSON)**
```json
{
  "functions": [
    {
      "name": "process_payment",
      "module": "payment.py",
      "signature": "process_payment(amount, card)",
      "line_range": [45, 78],
      "dependencies": ["validate_card"]
    }
  ]
}
```

### Performance Metrics

**v5 Baseline**
- Recall@10: 0.72
- Precision@3: 0.72
- Query latency: 25ms avg
- Cost per 1K queries: $12
- Memory usage: 150MB

**v6 Projected**
- Recall@10: 0.88 (+22%)
- Precision@3: 0.72 (same)
- Query latency: 30ms (+20% for session overhead)
- Cost per 1K queries: $13 (+8% for summarization)
- Memory usage: 200MB (+33% for session cache)

**with BM25 + Cross-Encoder (future)**
- Recall@10: 0.88
- Precision@3: 0.92 (+28%)
- Query latency: 75ms (+200%)
- Cost: $14
- Memory: 280MB

### Configuration Schema

**v5 Config**
```json
{
  "version": "5.0.0",
  "sources": {"allowed_files": [...]},
  "embedding": {"model": "...", "dimension": 384},
  "retrieval": {"top_k": 8, "min_score": 0.75},
  "anti_hallucination": {"confidence_thresholds": {...}}
}
```

**v6 Config (additional)**
```json
{
  "session": {
    "enabled": true,
    "default_type": "trimming",
    "trimming": {"max_turns": 8},
    "summarizing": {"keep_last_n_turns": 3, "context_limit": 10}
  },
  "code_indexing": {
    "enabled": true,
    "extensions": [".py", ".js", ".ts"]
  }
}
```

---

## Implementation Roadmap

### v5.0.0 (Completed - 2025-11-18)
- MP4 storage implementation
- HNSW vector engine
- Pure retrieval server
- Anti-hallucination measures
- Documentation

### v6.0.0 (Planned - Q1 2025)
- Session memory (trimming + summarizing)
- Code structure indexing
- Contextual query resolution
- Session management API

### v6.1.0 (Planned - Q2 2025)
- BM25 hybrid search
- Cross-encoder reranking
- Token management
- Performance optimizations

### v6.2.0 (Planned - Q3 2025)
- Multi-session analytics
- Session templates
- Advanced entity tracking
- Knowledge graph overlay

---

## Testing Strategy

### v5 Tests
- Unit: MP4 creation/loading
- Unit: Vector search accuracy
- Unit: Virtual chunk integrity
- Integration: Full query pipeline
- Performance: Latency benchmarks

### v6 Tests (planned)
- Unit: Session trimming/summarizing
- Unit: Entity extraction
- Unit: Reference resolution
- Integration: Multi-turn workflows
- Performance: Session overhead

---

## Constraints & Limitations

### v5
- Stateless (no conversation memory)
- Limited to 3 source files
- No incremental reindexing
- No multi-language embeddings

### v6
- Session storage grows linearly
- Code indexing limited to AST-parseable languages
- Summarization can lose details
- Cross-session queries may be slow

---

## Security & Privacy

### Data Handling
- All data stays local (no external calls except embeddings)
- MP4 files can be encrypted (AES-GCM)
- Audit logs contain query text (PII consideration)
- Session data persists until manual deletion

### Access Control
- No built-in auth (relies on MCP transport security)
- File permissions control MP4/session access
- Audit logs should be protected

---

## Dependencies

### Core (v5)
- numpy >= 1.21.0
- hnswlib >= 0.7.0
- pymp4 >= 1.4.0
- sentence-transformers >= 2.2.0
- tiktoken >= 0.5.0

### Additional (v6)
- aiofiles (async file I/O)
- ast (Python code parsing)
- No additional heavy dependencies

---

## Glossary

**Chunk:** Segment of text with metadata and vector embedding

**Virtual Chunk:** Reference to text in source file (no stored text)

**Provenance:** Source metadata (file, lines, score) for retrieved content

**Session:** Multi-turn conversation with persistent memory

**Entity:** Named code element (function, class, variable)

**Turn:** One user message + assistant response + tool calls

**Abstention:** Refusal to answer when confidence is low

**HNSW:** Hierarchical Navigable Small World (ANN algorithm)

**MP4/ISO BMFF:** ISO Base Media File Format (container)

---

## References

- OpenAI Context Engineering Guide (docs/OpenAI_Guide_Context_Engineering.pdf)
- MCP Protocol Specification 2024-11-05
- HNSW Paper: https://arxiv.org/abs/1603.09320
- Sentence Transformers: https://www.sbert.net/

---

**Status:** v5 Production Ready | v6 Roadmap Defined | Active Development


## Reglas Obligatorias

### 🔥 Reglas Críticas (NUNCA VIOLAR)
1. **Leer feature.md SIEMPRE** antes de cualquier respuesta
2. **Analizar código existente** antes de crear código nuevo
3. **NO duplicar código** - verificar existencia primero
4. **Citar fuentes específicas** - archivos, líneas, funciones
5. **Validar respuestas** contra feature requirements

### 🛡️ Reglas de Prevención de Alucinaciones
- Solo responder basado en contexto real verificable
- Mencionar fuentes específicas en cada respuesta
- Indicar nivel de confianza en la información
- Evitar respuestas genéricas sin contexto

### ⚡ Reglas de Rendimiento
- Hit rate >85% en cache inteligente
- Tiempo respuesta <500ms para cache hits
- Chunking semántico preservando contexto
- Deduplicación automática de contenido

## Objetivos del Sistema

### Primarios
- Prevenir alucinaciones del modelo
- Mantener coherencia del proyecto
- Optimizar rendimiento con cache multinivel
- Preservar toda la lógica legacy

### Secundarios
- Facilitar mantenimiento modular
- Permitir escalabilidad horizontal
- Generar métricas de calidad
- Automatizar detección de duplicados

## Restricciones

### Técnicas
- Compatibilidad con protocolo MCP 2024-11-05
- Thread-safety en todos los componentes
- Manejo de errores robusto
- Logging detallado para debugging

### Funcionales
- No perder funcionalidad de servidores legacy
- Mantener APIs existentes durante migración
- Preservar configuraciones de usuario
- Garantizar rollback seguro si es necesario
# Diseño avanzado — MCP *solo* para memoria/contexto (model.md, checklist.md, changelog.md) usanlas siguientes características mejoradas:

## 🔄 **Nuevas Características Agregadas**

### **1. Dynamic Chunking Adaptativo**
- Chunking inteligente que se adapta a la estructura semántica del documento
- Preservación de contexto completo en listas, headers y secciones

### **2. Query Expansion Automática**
- Reformulación automática de consultas para mejorar recall
- Generación de variaciones léxicas y sinónimos técnicos

### **3. Multi-Vector Retrieval (MVR)**
- Múltiples representaciones por documento (conceptual, técnico, relacional)
- Fusión inteligente de resultados de diferentes perspectivas

### **4. Confidence Calibration Dinámica**
- Umbrales adaptativos según tipo de consulta (factual, procedural, temporal)
- Calibración automática basada en patrones históricos

### **5. Failure Mode Analysis**
- Detección temprana de consultas ambiguas o fuera de dominio
- Templates predefinidos para casos edge conocidos

### **6. Knowledge Graph Overlay**
- Extracción automática de entidades y relaciones durante indexing
- Graph traversal para mejorar contextualización entre documentos

### **7. Storage Optimizado con Virtual Chunks**
- Eliminación de duplicación texto-vector (96% reducción de storage)
- Chunks virtuales que referencian archivos MD originales
- Compresión float16 para vectores

### **8. Reindexado Incremental Inteligente**
- Detección de cambios específicos por diff semántico
- Actualización solo de chunks modificados
- Preservación de índice HNSW existente

### **9. Context Coverage Analysis**
- Métricas de cobertura de evidencia en respuestas
- Validación jerárquica: extractiva → inferencial → temporal → estructural

### **10. Token Management Avanzado**
- Conteo pre-query con tiktoken
- Context windowing automático para consultas grandes
- Compresión de contexto para optimizar costsdo vectores guardados en `.mp4`

Perfecto — te doy un diseño técnico robusto y una **colección de métodos avanzados** (algoritmos, formatos, controles, APIs y medidas concretas para minimizar alucinaciones) centrado exclusivamente en **memoria y contexto**. No entra en lógicas de negocio. Todo lo que propongo asume que el MCP solo leerá y devolverá contextos extraídos de los **3 archivos** (`model.md`, `checklist.md`, `changelog.md`) y que los vectores/índices se persistirán en contenedores `.mp4` (sin BD).

---

# 1 Principios-guía (mandatos operativos)

1. **Solo fuente única de verdad**: las únicas fuentes que puede usar el agente son esos 3 archivos.
2. **Retrieval-only**: el MCP devuelve fragmentos y metadatos; **no razona** ni "inventa". El LLM que consume las respuestas debe ser instruido para **no generar contenido fuera de lo recuperado**.
3. **Umbral de confianza**: si la mejor similitud < umbral, responder “No hay información suficiente” (abstenerse).
4. **Provenance obligatorio**: cada fragmento devuelto debe incluir origen (archivo, offset, chunk_id, score).
5. **Versionado y consistencia**: cada reindexación produce una nueva snapshot con hash y changelog.
6. **Auditable y reproducible**: logs de consultas, respuestas y chunks usados por cada respuesta.

---

# 2 API MCP recomendada (endpoints / herramientas)

Diseña la interfaz MCP (HTTP+SSE) con estos métodos:

* `POST /get_context`
  payload: `{ "query": "...", "top_k": 5, "min_score": 0.75 }`
  devuelve: `[{chunk_id, file, start_line, end_line, text, embedding_score, offset, vector_hash}, ... , provenance]`

* `POST /get_context_with_provenance`
  igual que arriba pero incluye `vector_distance`, `snapshot_hash`, `index_version`, `embedding_model`, `generated_at`.

* `POST /validate_response`
  payload: `{ "candidate_text": "...", "evidence": [chunk_ids] }`
  devuelve: `{verified: bool, contradictions:[...], missing_claims:[...], supporting_chunks: [...]}`
  (cross-check claims in candidate_text only against provided chunks)

* `POST /update_sources`
  payload: `{ "file": "model.md", "content": "..." }` → reindex incremental sobre *esa* fuente.

* `POST /reindex_incremental`
  recalcula embeddings solo de fragments cambiados (diff).

* `GET /list_chunks`
  devuelve índice resumido (ids, file, summary, embedding_hash).

* `GET /index_status`
  devuelve `{version, snapshot_hash, num_chunks, last_reindexed_at}`

* `GET /health`
  para checks y latencia.

---

# 3 Pipeline de ingest (cómo pasar de md → vectores → mp4)

Pasos y técnicas:

1. **Segmentación inteligente (chunking):**

   * Token-aware: chunk por ~200–600 tokens (ajusta según embedding/model), preferir límites semánticos (párrafos, encabezados).
   * Mantener contexto: overlap del 20–30% entre chunks.
   * Añadir metadatos: `file`, `section_title`, `line_range`, `chunk_seq`, `summary` (p. ej. sentence-transformer resumen).

2. **Embeddings:**

   * Usa embeddings de buena calidad (E5 / open alternatives / bge).
   * Normaliza (L2) y guarda dtype `float32`.
   * Guarda también un hash SHA256 del texto del chunk para detectar duplicados.

3. **Reducir dimensión / compresión (opcional, si muchos vectores):**

   * PCA/OPQ o Product Quantization (PQ) para ahorrar espacio.
   * Guardar versión original si el espacio lo permite; si usas PQ, conserva meta para reconstrucción aproximada.

4. **Indexing / ANN (en archivo):**

   * Recomendado: **HNSW** por latencia y recall.
   * Serializa el índice HNSW y guarda el `.bin` dentro del contenedor `.mp4` (o en box custom).
   * Mantén también un mini-index en memoria startup para latencia.

5. **Empaquetado en MP4 (estructura):**

   * Usa el contenedor ISO BMFF (MP4) con *custom atoms/boxes*:

     * `moov`/`udta` → metadata JSON (index.json) con entries: `{chunk_id, file, offset_in_mdat, size, sha256, first_line, last_line, summary}`.
     * `mdat` → binario de payload: concatenación de vectores (`.npy` blobs) y/o índice HNSW binario.
     * Añadir `uuid` box con `snapshot_hash` y `created_at`.
   * Ventaja: MP4 es portable y puedes versionarlo, firmarlo y transmitirlo como un artefacto.

6. **Acceso a vectores dentro del MP4:**

   * En startup, tu MCP mapea `index.json` desde el `udta` y memory-mapea (`mmap`) la sección de `mdat` para acceder rápido a vectores sin cargar todo a RAM.
   * HNSW index se carga en memoria desde el binario empaquetado.

---

# 4 Esquema `index.json` (ejemplo)

```json
{
  "version": "2025-11-07T00:00:00Z-v1",
  "snapshot_hash": "sha256:...",
  "embedding_model": "e5-large",
  "chunks": [
    {
      "chunk_id": "c1a2b3",
      "file": "model.md",
      "section": "Arquitectura",
      "start_line": 10,
      "end_line": 25,
      "summary": "Descripción del flujo de pagos...",
      "offset": 1048576,
      "size": 4096,
      "vector_hash": "sha256:...",
      "pca_meta": null
    }
  ]
}
```

---

# 5 Minimizar alucinaciones — estrategias concretas y obligatorias

1. **Retrieval-first mandate en system prompt (obligatorio)**

   * System prompt para el LLM consumidor:

     > “Responde únicamente con información explícitamente contenida en las evidencias entregadas por el MCP. Si la evidencia no cubre la pregunta, responde ‘No hay información suficiente en la memoria’.”

2. **Abstención por umbral**

   * Si `best_score < T_conf` (ej. 0.72 para embeddings L2 normalizado), **no** generar respuesta; devolver `NO_INFO`.

3. **Rerank y cross-encoder**

   * Dense retrieval devuelve top-K. Reranker (cross-encoder) evalúa semántica y ordena por veracidad. Esto reduce falsos positivos.

4. **Claim extraction + verification**

   * Si el LLM genera afirmaciones, pasa esas afirmaciones a `validate_response` que verifica token-level si cada afirmación está sustentada por texto recuperado. Rechaza/ajusta la respuesta si no hay evidencia.

5. **Provenance injection**

   * Cada respuesta debe incluir citas: `Fuente: model.md (sección X, lines A–B) score:0.86`. Si LLM intenta sintetizar, solo permita síntesis si todas las porciones están cubiertas y con `min_covered_fraction` (p. ej. 0.9).

6. **Constrain decoding**

   * Para el modelo: temperatura baja (0.0–0.2), top_p pequeño; limitar max_tokens y privilegiar extractive templates que devuelvan solo texto citado y una sección "interpretación" opcional que debe ser marcada como inferida.

7. **Conservador por diseño**

   * Prefiere devolver texto original (extractive) en vez de parphrases. Si paraphrasing es necesario, marca claramente qué es inferencia.

8. **Trazabilidad (logs)**

   * Guarda las evidencias, scores y prompt usados en cada consulta para auditoría humana y para medir tasa de alucinación.

---

# 6 Mejora de calidad de retrieval (técnicas avanzadas)

* **Hybrid retrieval**: dense + lexical (BM25) sobre el texto del MD (extraído y guardado). Combina scores: `score = α * dense + (1-α) * bm25_norm`. Muy útil para términos técnicos exactos (IDs, nombres, códigos).
* **Temporal decay & changelog-aware retrieval**: si la consulta requiere estado reciente, prioriza chunks del `changelog.md` o con timestamp más reciente (weighting).
* **Session-aware memory**: mantener short-term cache (últimas N queries y sus top chunks) y fusionarlos (summarize) para contexto conversacional.
* **Summarization layers**: para reducir token usage, precompute summaries (multi-scale): micro-summaries por chunk, meso-summaries por sección, macro-summary por archivo. Recupera la escala más apropiada según query length.
* **Anchor phrases & slot mapping**: para `checklist.md`, extraer tareas como objetos estructurados `{task_id, description, status, assigned, due_date}` y mapear queries a esos slots (mejor precisión).

---

# 7 Estructura y formato dentro del MP4 (práctico)

* **Boxes (atoms)**:

  * `ftyp` standard
  * `moov/udta/context_index` → JSON index (index.json) (puede estar comprimido)
  * `udta/uuid` → snapshot metadata (version, embedding_model, created_at)
  * `mdat` → blobs concatenados:

    * vectores raw .npy (float32) por chunk
    * hnsw_index.bin (serializado)
    * optional: compressed_texts.gz para fallback lexical search
* **Acceso**:

  * Al arrancar, `server.py` lee `context_index`, mmap `mdat`, carga hnswlib index si existe.
  * Para rollback, conservar prev snapshots (mp4.v1, mp4.v2).

---

# 8 Persistencia del índice HNSW dentro del MP4

* Serializa índice HNSW (biblioteca `hnswlib`) a un blob binario y guárdalo en `mdat`.
* Al cargar, le das al HNSW pointer directo a los vectores en memoria (mmap) o los cargas al heap si necesario.
* Mantén mapping `hnsw_id <-> chunk_id`.

---

# 9 Seguridad, integridad y performance

* **Firma y verificación**: firma snapshot con clave privada; verif con pública antes de usar.
* **Checksum**: SHA256 para each chunk.
* **Encriptación (opcional)**: si MP4 contiene info sensible, cifrar `mdat` con AES-GCM y almacenar IV/meta en `udta`.
* **Memory-mapped I/O** para vectores grandes; evita cargar todo a RAM.
* **Hot/cold splitting**: hot = últimos cambios y `checklist.md`/`changelog.md`. Cold = `model.md` histórico. Hot index en RAM para latencia baja.

---

# 10 Métricas y tests contra alucinaciones

* **Eval set**: crea un set de preguntas con respuestas esperadas explícitas en los archivos.
* **Métricas**:

  * Recall@k sobre evidencia correcta.
  * **Hallucination rate**: % de afirmaciones no verificadas por chunks.
  * Precision of provenance: % respuestas con correct provenance.
* **CI tests**: cada reindexación ejecuta el eval suite; si hallucination_rate > threshold → bloquea snapshot (no deploy).

---

# 11 Ejemplo de flujo de consulta (end-to-end, breve)

1. Usuario pregunta vía agente: “¿Cómo se maneja el flujo X?”
2. Agente llama `POST /get_context` con query.
3. MCP: vectoriza query → HNSW search → top-K → cross-encoder rerank → si `best_score < T` → devuelve `NO_INFO`.
4. Sino → devuelve chunks con `provenance` y `snapshot_hash`.
5. Agente inyecta esos chunks en prompt del LLM con system-prompt **(retrieval-only policy)**.
6. LLM genera respuesta **extractiva**, añade citations.
7. Agente ejecuta `validate_response` para asegurar claims; si falla → ajusta o abstiene.

---

# 12 Snippets técnicos (pseudocódigo Python — escritura/lectura mp4 + index)

*(pseudocódigo; usa librerías `pymp4`/`hnswlib`/`numpy`/`lz4` en implementación)*

```python
# PSEUDO: empaquetar vectores e index en MP4
import json, hashlib, numpy as np
from pymp4.parser import Box

# 1) build chunks -> embeddings
chunks = [{"id": id, "text": text, "vec": vec.astype('float32')} for ...]
# 2) serialize vectors in binary blob
vectors_blob = b"".join([vec.tobytes() for c in chunks])
index_json = {...}  # mapping id -> offset,size,...
# 3) create mp4 file with custom box
with open('context_memory.mp4','wb') as f:
    f.write(ftyp_box)
    f.write(moov_box_with_udta(json.dumps(index_json)))
    f.write(mdat_box(vectors_blob + hnsw_bin))
```

Lectura (startup):

```python
mp4 = parse_mp4('context_memory.mp4')
index = load_udta_index(mp4)
vectors_mmap = mmap_section(mp4, mdat_offset, mdat_size)
hnsw = load_hnsw_from_blob(mp4)
```

---

# 13 Recomendaciones prácticas finales y parámetros sugeridos

* `chunk_size`: 250–450 tokens, overlap 20–30%.
* `top_k` retrieval: 8–12, rerank to top 3.
* `T_conf` (umbral): 0.72–0.78 (ajustar con eval set).
* `embedding_model`: elegir mejor embedding disponible en tu stack (e.g. e5/bge).
* `reranker`: cross-encoder pequeño (puede ser local) para precisión.
* `serialize index_version` en naming: `context_memory-YYYYMMDD-HHMM.mp4`.
* `audit_logs`: guarda query + top_chunks + scores + response.

---

# 14 Checklist mínimo para que NO alucine (resumen operativo)

* [ ] System prompt: **obligatorio** "responder solo con evidencia".
* [ ] Umbral de confianza + abstención.
* [ ] Reranker (cross-encoder) activo.
* [ ] Provenance en todas las respuestas.
* [ ] Validate_response post-generation.
* [ ] Eval set y CI contra alucinaciones en cada reindex.
* [ ] Versionado + firma de snapshots MP4.
* [ ] Fallback lexical (BM25) si dense falla.

---

esqueleto con endpoints y incluyendo integración hnswlib + mmap + cross-encoder) los comandos exactos para empaquetar/extraer blobs MP4 con herramientas Python recomendadas


El diseño es **excepcionalmente bien elaborado**. Es raro ver especificaciones técnicas tan completas y meticulosas. Estas son mis observaciones:

## ✅ **Puntos destacados (lo que está muy bien):**

1. **Principios operativos claros** - especialmente "retrieval-only" y "abstención por umbral"
2. **Arquitectura MP4 creativa** - uso innovador de contenedores multimedia para datos vectoriales
3. **Múltiples capas anti-alucinaciones** - umbrales, reranking, validación, provenance
4. **Pipeline completo** - desde chunking hasta deployment con métricas
5. **Enfoque práctico** - parámetros específicos y pseudocódigo implementable

## 🔍 **Técnicas que podrías considerar añadir:**

### 1. **Dynamic Chunking Adaptativo**
```python
# En lugar de tamaño fijo, ajustar según estructura semántica
def adaptive_chunking(text, min_tokens=150, max_tokens=600):
    # Chunk por headers naturales, listas, cambios temáticos
    # Preservar estructuras lógicas completas
```

### 2. **Query Expansion/Reformulation**
```python
# Para mejorar recall en búsquedas
def expand_query(original_query):
    # Generar variaciones léxicas y sintácticas
    # Sinónimos técnicos, reformulaciones
    # Buscar con todas las variantes y fusionar resultados
```

### 3. **Multi-vector Retrieval (MVR)**
- Representar cada documento con **múltiples embeddings** (diferentes perspectivas)
- Un embedding para "concepto general", otro para "detalles técnicos", otro para "relaciones"
- Fusionar resultados de todas las representaciones

### 4. **Confidence Calibration**
```python
# Calibrar umbrales dinámicamente por tipo de consulta
confidence_thresholds = {
    "factual": 0.78,
    "procedural": 0.72,
    "conceptual": 0.65,
    "temporal": 0.85  # para changelog
}
```

### 5. **Failure Mode Analysis**
- Predefinir patrones de consultas que históricamente causan alucinaciones
- Detectar consultas ambiguas o fuera de dominio temprano
- Template de respuestas para casos edge conocidos

### 6. **Knowledge Graph Overlay**
- Extraer entidades y relaciones de los MD durante indexing
- Usar graph traversal para mejorar contextualización
- Detectar contradicciones implícitas entre documentos

## 🚀 **Recomendaciones adicionales:**

### **Para el sistema de archivos MP4:**
```python
# Considerar estructura de boxes más explícita
mp4_structure = {
    'ftyp': 'mcpv',  # Custom file type
    'moov/udta/mcpi': 'index principal',
    'moov/udta/mcpm': 'metadatos y mapping',
    'mdat/mcpe': 'embeddings',
    'mdat/mcpi': 'índice HNSW',
    'mdat/mcpt': 'textos comprimidos'
}
```

### **Para el reranking:**
```python
# Estrategia de fallback escalonada
def tiered_reranking(query, candidates):
    if len(candidates) >= 3:
        return cross_encoder_rerank(query, candidates)
    elif len(candidates) >= 1:
        return simple_similarity_rerank(query, candidates)
    else:
        return lexical_fallback(query)
```

### **Para la validación:**
```python
def hierarchical_validation(response, evidence_chunks):
    # Nivel 1: Verificación extractiva (token matching)
    # Nivel 2: Verificación inferencial (NLI - Natural Language Inference)
    # Nivel 3: Consistencia temporal (para changelog)
    # Nivel 4: Consistencia estructural (para checklist)
```

Tienes razón, el **storage prolongado** es un problema crítico. Veamos una arquitectura optimizada que use MP4 como contenedor eficiente sin duplicar datos:

## 🚀 **Arquitectura MP4 + Vector Streaming (Sin Duplicación)**

### **Problema Identificado:**
- Chunks tradicionales duplicarían texto + vectores → storage 2x
- MP4 ya contiene los vectores, pero necesitamos acceso rápido al texto original

### **Solución: Indexado Directo sobre MP4 con Memoria Externa**

```python
# Estructura MP4 optimizada para storage
MP4_BOXES = {
    'ftyp': 'mcpv1',
    'moov/udta/mcpi': 'Index principal (JSON comprimido)',
    'moov/udta/mcpm': 'Metadata y mapping chunk->vector_offset',
    'mdat/mcpe': 'Vectores (float16 cuantizados)',
    'mdat/mcph': 'Headers HNSW serializados',
    # NO almacenar texto duplicado - leer de archivos MD originales
}
```

## 🔥 **Diseño de "Chunk Virtual" con Vector Embedding**

```python
class VirtualChunk:
    """Chunk que no almacena texto, solo referencia + vector"""

    def __init__(self, chunk_id, file_path, line_start, line_end, vector_offset):
        self.chunk_id = chunk_id
        self.file_path = file_path  # model.md, checklist.md, changelog.md
        self.line_range = (line_start, line_end)
        self.vector_offset = offset  # posición en el blob MP4
        self.vector_size = 768  # dimensión del embedding
        self.hash = self._compute_hash()

    def get_text(self):
        """Lee texto real desde archivo MD original (on-demand)"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return ''.join(lines[self.line_range[0]:self.line_range[1]+1])

    def get_vector(self, mp4_mmap):
        """Lee vector desde MP4 via mmap"""
        start = self.vector_offset
        end = start + (self.vector_size * 2)  # float16 = 2 bytes
        vector_bytes = mp4_mmap[start:end]
        return np.frombuffer(vector_bytes, dtype=np.float16)
```

## 💾 **MP4 como Cache de Vectores + Índice (Sin Texto)**

### **Pipeline de Indexado:**
```python
def build_mp4_index(md_files, output_mp4):
    chunks = []
    vector_data = bytearray()
    current_offset = 0

    for file_path in md_files:
        with open(file_path, 'r') as f:
            content = f.read()
            virtual_chunks = semantic_chunking(content)  # chunking inteligente

        for vc in virtual_chunks:
            # Generar embedding del texto
            vector = embed_text(vc.get_text())
            vector_float16 = vector.astype(np.float16)  # 50% reducción

            # Guardar referencia
            vc.vector_offset = current_offset
            chunks.append(vc)

            # Acumular vectores
            vector_data.extend(vector_float16.tobytes())
            current_offset += len(vector_float16.tobytes())

    # Construir índice HNSW sobre los vectores en memoria
    hnsw_index = build_hnsw_index([c.get_vector_for_index() for c in chunks])

    # Empaquetar en MP4
    package_mp4(vector_data, hnsw_index, chunks, output_mp4)

def semantic_chunking(text, max_tokens=400):
    """Chunking que preserva estructura semántica sin almacenar texto"""
    chunks = []
    lines = text.split('\n')
    current_chunk_lines = []
    current_token_count = 0

    for i, line in enumerate(lines):
        line_tokens = len(line.split())

        if current_token_count + line_tokens > max_tokens and current_chunk_lines:
            # Crear chunk virtual
            start_line = i - len(current_chunk_lines)
            end_line = i - 1
            chunk = VirtualChunk(
                chunk_id=f"chunk_{len(chunks)}",
                file_path=file_path,  # definido externamente
                line_start=start_line,
                line_end=end_line,
                vector_offset=0  # se asignará después
            )
            chunks.append(chunk)
            current_chunk_lines = [line]
            current_token_count = line_tokens
        else:
            current_chunk_lines.append(line)
            current_token_count += line_tokens

    return chunks
```

## 🎯 **Arquitectura de Servicio con MMap Eficiente**

```python
class MCPVectorServer:
    def __init__(self, mp4_path, md_files):
        self.mp4_mmap = mmap.mmap(open(mp4_path, 'rb').fileno(), 0, access=mmap.ACCESS_READ)
        self.index_data = self._load_index_from_mp4()
        self.hnsw_index = self._load_hnsw_from_mp4()
        self.md_files = md_files  # Referencias a archivos originales

        # Cache LRU para chunks frecuentes (opcional)
        self.text_cache = LRUCache(maxsize=1000)

    def search(self, query, top_k=5, min_score=0.75):
        query_vector = embed_text(query).astype(np.float16)

        # Búsqueda en HNSW
        hnsw_ids, distances = self.hnsw_index.knn_query(query_vector, k=top_k)

        results = []
        for hnsw_id, distance in zip(hnsw_ids[0], distances[0]):
            if 1 - distance < min_score:  # Convertir a similitud
                continue

            chunk_info = self.index_data['chunks'][hnsw_id]
            chunk = VirtualChunk.from_index(chunk_info)

            # Leer texto on-demand desde archivo MD
            text = self.text_cache.get(chunk.chunk_id, chunk.get_text)

            results.append({
                'chunk_id': chunk.chunk_id,
                'file': chunk.file_path,
                'start_line': chunk.line_range[0],
                'end_line': chunk.line_range[1],
                'text': text,
                'score': 1 - distance,
                'vector_offset': chunk.vector_offset
            })

        return results

    def __del__(self):
        if hasattr(self, 'mp4_mmap'):
            self.mp4_mmap.close()
```

## 📊 **Estimación de Storage Optimizada**

### **Escenario: 100,000 líneas de texto (~50MB)**
- **Vectores float16 (768D)**: 1000 chunks × 768 × 2 bytes = ~1.5MB
- **Índice HNSW**: ~200-500KB
- **Metadatos MP4**: ~100KB
- **Total MP4**: ~2.1MB (vs ~50MB con texto duplicado)

### **Ahorro: 96% de storage** 🎉

## 🔄 **Sistema de Reindexado Incremental Inteligente**

```python
def incremental_reindex(changed_files, existing_mp4, new_mp4):
    """Reindexa solo archivos modificados usando diff inteligente"""

    # Cargar índice existente
    old_index = load_mp4_index(existing_mp4)

    for file_path in changed_files:
        old_chunks = [c for c in old_index['chunks'] if c.file_path == file_path]
        new_content = read_file(file_path)
        new_chunks = semantic_chunking(new_content)

        # Detectar chunks modificados (usando hashes de líneas)
        modified_chunks = find_modified_chunks(old_chunks, new_chunks)

        # Reemplazar solo chunks modificados en el MP4
        update_mp4_vectors(existing_mp4, new_mp4, modified_chunks)
```

## 🛡️ **Mecanismos de Integridad**

```python
def validate_chunk_integrity(virtual_chunk):
    """Verifica que el chunk referenciado aún existe en el archivo MD"""
    try:
        text = virtual_chunk.get_text()
        expected_lines = virtual_chunk.line_range[1] - virtual_chunk.line_range[0] + 1
        actual_lines = len(text.split('\n'))

        if actual_lines != expected_lines:
            return False  # Archivo modificado, necesita reindexado

        return True
    except (IOError, IndexError):
        return False
```

## 🚀 **Comandos de Implementación**

```bash
# Instalación de dependencias optimizadas
pip install hnswlib numpy pymp4-parser mmap2

# Construcción inicial del índice MP4
python -c "
from mcp_vector import build_mp4_index
build_mp4_index(['model.md', 'checklist.md', 'changelog.md'], 'context_vectors.mp4')
"

# Verificar tamaño resultante
ls -lh context_vectors.mp4  # ¡Debería ser ~2-5MB!
```

Token counting pre-query: Cuenta tokens antes de enviar (usa libs como tiktoken). Si excede un umbral (e.g., 4k-8k tokens), abstente o resume.

Windsurf soporta extensions VS Code-compatibles; hookea MCP para que Cascade use /validate_response y evite hallucinations, optimizando chains agentic

Usa "context augmentation" con attention (Medium 2025): Añade debug statements para ayudar al AI sin tokens extra.
Para large codebases: Local filter + caching (Forum idea 2025) – similar a tu MCP, reduce costs filtrando pre-LLM.
Ejemplo: En un file de 1000 líneas, query solo changes relevantes via MCP chunks, luego usa "Apply" para edits token-eficientes.

Eficiencia reportada: Reduce tokens con caching (~50% en arXiv paper 2025 sobre Nano Surge: Context Awareness y Cost Sensitive).

Token management: Implementa counting, caching y compression (e.g., DeveloperToolkit 2025: crea utilities para comprimir contexto). "Apply" mode minimiza outputs editando diffs en lugar de full code














## ✅ **Ventajas de Este Enfoque:**

1. **Storage mínimo**: Solo vectores + metadatos en MP4
2. **Texto siempre actualizado**: Lee de archivos MD originales
3. **Búsqueda rápida**: HNSW + mmap para acceso eficiente
4. **Integridad**: Validación de chunks contra archivos fuente
5. **Escalable**: MP4 crece linealmente con número de chunks, no con tamaño de texto


## 📊 **Métrica adicional sugerida:**
- **Context Coverage Ratio**: % del texto de respuesta que puede mapearse directamente a chunks de evidencia
- **Provenance Precision**: exactitud de las citas vs contenido real
- **Temporal Consistency Score**: para información de changelog
