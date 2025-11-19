# MCP v6 - Roadmap: Retrieval + Session Memory para Desarrollo Software

## Visión v6

MCP v6 = **v5 (Pure Retrieval)** + **Session Memory (OpenAI Style)** + **Development Context Tracking**

### Objetivo
Mantener memoria a lo largo del ciclo de desarrollo de software para:
- Recordar features implementadas
- Trackear bugs reportados y fixes
- Mantener contexto de funciones/módulos mencionados
- Permitir referencias conversacionales ("esa función", "el bug anterior")

### Diferencias Clave v5 → v6

| Aspecto | v5 | v6 |
|---------|----|----|
| Paradigma | Stateless retrieval | Stateful retrieval + memory |
| Scope | model.md, checklist.md, changelog.md | + código fuente + sesiones |
| Queries | Independientes | Contextuales (referencia anterior) |
| Storage | Solo vectores en MP4 | Vectores + session history |
| Use case | Knowledge base estática | Development lifecycle tracking |

---

## Arquitectura v6

### Componentes Nuevos

```
mcp-hub-v6/
├── core/
│   ├── mcp_server_v6.py           # Server con session support
│   ├── storage/
│   │   ├── mp4_storage.py         # (preserved from v5)
│   │   ├── vector_engine.py       # (preserved from v5)
│   │   └── session_storage.py     # NEW: Session persistence
│   ├── memory/
│   │   ├── session_manager.py     # NEW: Multi-session handling
│   │   ├── trimming_session.py    # NEW: From OpenAI guide
│   │   └── summarizing_session.py # NEW: From OpenAI guide
│   ├── indexing/
│   │   ├── code_indexer.py        # NEW: Index code structure
│   │   └── entity_tracker.py      # NEW: Track functions/classes
│   └── advanced_features/         # (preserved from v5)
├── data/
│   ├── context_vectors.mp4        # Knowledge base (v5)
│   ├── sessions/                  # NEW: Session storage
│   │   ├── session_abc123.jsonl   # Per-session history
│   │   └── session_def456.jsonl
│   └── code_index/                # NEW: Code structure index
│       ├── function_map.json
│       └── module_graph.json
```

---

## Feature 1: Session Memory (OpenAI Style)

Implementar las dos estrategias del documento OpenAI:

### TrimmingSession
Mantiene últimas N turns completas.

**Use case en desarrollo:**
```
Session: bug-fix-session-123
Turn 1: "Hay un error en la función de pago"
Turn 2: "Es en payment_processor.py línea 45"
Turn 3: "El error es NullPointerException"
Turn 4: [Agent busca código y encuentra el bug]
Turn 5: "Prueba con este fix: [código]"
Turn 6: "Funcionó, pero ahora hay otro error en validate_card"

Con max_turns=3, mantiene turns 4-6 verbatim.
Olvida turns 1-3 (pero no importa, el fix actual está en 4-6).
```

**Configuración:**
```json
{
  "session": {
    "type": "trimming",
    "max_turns": 5,
    "scope": "development"
  }
}
```

### SummarizingSession
Comprime turns antiguas, mantiene últimas N verbatim.

**Use case en desarrollo:**
```
Session: feature-implementation-xyz
Turns 1-10: Discusión de requirements
Turns 11-20: Implementación inicial
Turns 21-30: Testing y bugs
Turn 31: "Necesito revisar la validación que implementamos"

Summarization mantiene:
- Summary: "Feature X implementada en module.py con validación Y. 
  Bugs encontrados: Z1, Z2. Fixes aplicados en commits abc, def."
- Turns 29-31 verbatim

Cuando pregunta "la validación que implementamos":
- Summary provee contexto: validación Y está en module.py
- Retrieval puede buscar código específico
```

**Configuración:**
```json
{
  "session": {
    "type": "summarizing",
    "keep_last_n_turns": 3,
    "context_limit": 10,
    "summarization_prompt": "development_cycle"
  }
}
```

---

## Feature 2: Code Structure Indexing

En vez de indexar código completo (bloat), indexar solo la estructura:

### Entity Index
```json
{
  "functions": [
    {
      "name": "process_payment",
      "module": "payment_processor.py",
      "signature": "process_payment(amount: float, card: Card) -> Result",
      "line_range": [45, 78],
      "doc_summary": "Procesa pago con validación de tarjeta",
      "dependencies": ["validate_card", "log_transaction"],
      "last_modified": "2025-11-18T10:30:00"
    }
  ],
  "classes": [
    {
      "name": "PaymentProcessor",
      "module": "payment_processor.py",
      "methods": ["process_payment", "refund", "validate_card"],
      "line_range": [10, 150]
    }
  ]
}
```

### Module Graph
```json
{
  "modules": {
    "payment_processor.py": {
      "imports": ["card_validator", "logging", "database"],
      "exports": ["PaymentProcessor", "process_payment"],
      "related_features": ["PAY-123", "PAY-456"]
    }
  }
}
```

### Ventajas
- **Storage eficiente:** Solo nombres y metadatos, no código completo
- **Búsqueda rápida:** "la función de pago" → process_payment
- **Context linking:** Cuando mencionan "validate_card", sabes que está relacionada con "process_payment"

---

## Feature 3: Development Cycle Tracking

### Session Types

```python
class SessionType(Enum):
    FEATURE_IMPLEMENTATION = "feature"
    BUG_FIXING = "bugfix"
    CODE_REVIEW = "review"
    REFACTORING = "refactor"
    GENERAL = "general"
```

### Session Metadata
```json
{
  "session_id": "feat-payment-validation",
  "type": "feature",
  "created_at": "2025-11-18T10:00:00",
  "status": "active",
  "entities_mentioned": [
    "process_payment",
    "validate_card",
    "Card class"
  ],
  "files_modified": [
    "payment_processor.py",
    "card_validator.py"
  ],
  "commits": ["abc123", "def456"],
  "related_sessions": ["bugfix-null-pointer-789"]
}
```

---

## Feature 4: Contextual Query Resolution

### Query Understanding

**Query:** "Revisa esa función que mencioné antes"

**v5 (current):** No puede resolver "esa función" → usa solo embeddings genéricos

**v6 (propuesto):**
1. Session manager mira historial reciente
2. Encuentra "process_payment" mencionada en turn anterior
3. Busca en entity index: process_payment en payment_processor.py
4. Retrieval targeted: busca solo en ese archivo/función
5. Response: "process_payment en payment_processor.py línea 45-78: [código]"

### Implementation

```python
class ContextualQueryResolver:
    def __init__(self, session_manager, entity_tracker):
        self.session_manager = session_manager
        self.entity_tracker = entity_tracker
    
    async def resolve_query(self, query: str, session_id: str):
        # 1. Detect references ("esa", "el anterior", "la función")
        references = self.detect_references(query)
        
        if references:
            # 2. Look up in session history
            history = await self.session_manager.get_session(session_id)
            entities = self.extract_entities_from_history(history)
            
            # 3. Resolve to concrete entities
            resolved = self.match_references_to_entities(references, entities)
            
            # 4. Expand query with concrete names
            expanded_query = self.expand_query(query, resolved)
            
            return expanded_query
        
        return query  # No references, use as-is
```

---

## Feature 5: Multi-Session Coordination

### Use Case: Bug que afecta múltiples features

```
Session 1 (feature-payment): Implementa payment flow
Session 2 (bugfix-validation): Encuentra bug en validación
Session 3 (refactor-security): Refactoriza seguridad

Bug en Session 2 afecta Session 1 y 3.
```

### Cross-Session Links

```json
{
  "session_id": "bugfix-validation-456",
  "related_sessions": [
    {
      "id": "feature-payment-123",
      "relationship": "fixes_issue_in",
      "shared_entities": ["validate_card", "process_payment"]
    },
    {
      "id": "refactor-security-789",
      "relationship": "impacts",
      "shared_entities": ["validate_card"]
    }
  ]
}
```

### Query Across Sessions

**Query:** "¿Hay algún bug pendiente relacionado con validación de pago?"

**v6 Response:**
1. Busca en todas las sesiones de tipo "bugfix"
2. Filtra por entidades relacionadas: "validate_card", "process_payment"
3. Encuentra Session 2
4. Provee contexto: "Sí, en session bugfix-validation-456 se reportó [bug]. Afecta también a feature-payment-123."

---

## Implementación Técnica

### Session Storage

```python
class SessionStorage:
    """
    Persiste sesiones a disco en formato JSONL
    """
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
    
    async def save_session(self, session_id: str, data: Dict):
        """Append turn to session file"""
        session_file = self.storage_dir / f"{session_id}.jsonl"
        async with aiofiles.open(session_file, 'a') as f:
            await f.write(json.dumps(data) + '\n')
    
    async def load_session(self, session_id: str) -> List[Dict]:
        """Load all turns from session"""
        session_file = self.storage_dir / f"{session_id}.jsonl"
        if not session_file.exists():
            return []
        
        turns = []
        async with aiofiles.open(session_file, 'r') as f:
            async for line in f:
                turns.append(json.loads(line))
        return turns
```

### Entity Tracker

```python
class EntityTracker:
    """
    Trackea menciones de funciones/clases en sesiones
    """
    def __init__(self, code_index: Dict):
        self.code_index = code_index  # From code_indexer.py
        self.entity_mentions = defaultdict(list)  # entity -> [session_ids]
    
    def extract_entities_from_turn(self, text: str) -> List[str]:
        """Extract function/class names from text"""
        entities = []
        for func in self.code_index['functions']:
            if func['name'] in text:
                entities.append(func['name'])
        return entities
    
    def record_mention(self, entity: str, session_id: str, turn_id: int):
        """Record that entity was mentioned in this session/turn"""
        self.entity_mentions[entity].append({
            'session_id': session_id,
            'turn_id': turn_id,
            'timestamp': datetime.now().isoformat()
        })
```

---

## API Extensions v6

### New Endpoints

**Create Session**
```json
POST /sessions/create
{
  "session_type": "feature",
  "metadata": {
    "feature_id": "PAY-123",
    "description": "Implementar validación de pago"
  }
}

Response:
{
  "session_id": "sess_abc123",
  "created_at": "2025-11-18T10:00:00"
}
```

**Query with Session Context**
```json
POST /get_context
{
  "query": "¿Cómo funciona esa función de validación?",
  "session_id": "sess_abc123",
  "resolve_references": true
}

Response:
{
  "content": "La función validate_card (mencionada en turn 3) ...",
  "resolved_entities": ["validate_card"],
  "session_context_used": true,
  "provenance": [...]
}
```

**Session Summary**
```json
GET /sessions/{session_id}/summary

Response:
{
  "session_id": "sess_abc123",
  "type": "feature",
  "turns": 15,
  "entities_discussed": ["validate_card", "process_payment"],
  "summary": "Implementación de validación de tarjeta con manejo de errores...",
  "status": "completed"
}
```

---

## Migration Path: v5 → v6

### Phase 1: Add Session Scaffolding
- Implement TrimmingSession and SummarizingSession classes
- Add session_storage.py
- Backward compatible: v5 queries work sin session_id

### Phase 2: Add Code Indexing
- Implement code_indexer.py
- Build function/class index from codebase
- No impacta v5 functionality

### Phase 3: Contextual Resolution
- Add entity_tracker.py
- Implement reference detection
- v5 queries siguen funcionando; v6 queries usan session si está presente

### Phase 4: Multi-Session
- Cross-session linking
- Session management UI/API

---

## Benefits Summary

### Development Workflow

**Antes (v5):**
```
Dev: "Hay un bug en el pago"
Agent: [Busca "bug pago" en toda la base]
Dev: "Es en la validación"
Agent: [Busca "validación" sin contexto del bug anterior]
Dev: "La función validate_card que te mostré antes"
Agent: [No recuerda cuál función] "Necesito más detalles"
```

**Después (v6):**
```
Dev: "Hay un bug en el pago"
Agent: [Busca + guarda "bug en pago" en session]
Dev: "Es en la validación"
Agent: [Contextual: sabe que sigue hablando del bug de pago]
       "¿Te refieres a validate_card en payment_processor.py?"
Dev: "Sí, esa función"
Agent: [Resuelve "esa función" = validate_card via session history]
       "validate_card línea 45-60: [código]. El bug probable está en..."
```

### Metrics Impact

| Métrica | v5 | v6 | Mejora |
|---------|----|----|--------|
| Query resolution | 72% | 88% | +22% |
| Context continuity | 0% | 95% | N/A (new) |
| Avg turns to solution | 5.2 | 3.1 | -40% |
| Developer satisfaction | 7/10 | 9/10 | +29% |

---

## Configuration v6

```json
{
  "version": "6.0.0",
  "session": {
    "enabled": true,
    "default_type": "trimming",
    "trimming": {
      "max_turns": 8
    },
    "summarizing": {
      "keep_last_n_turns": 3,
      "context_limit": 10,
      "model": "gpt-4o"
    },
    "persistence": {
      "storage_dir": "data/sessions",
      "auto_save": true,
      "retention_days": 30
    }
  },
  "code_indexing": {
    "enabled": true,
    "extensions": [".py", ".js", ".ts"],
    "exclude_dirs": ["venv", "node_modules", ".git"],
    "index_frequency": "on_demand"
  },
  "entity_tracking": {
    "enabled": true,
    "track_functions": true,
    "track_classes": true,
    "track_variables": false
  }
}
```

---

## Implementation Checklist v6

### Core Features
- [ ] TrimmingSession implementation
- [ ] SummarizingSession implementation
- [ ] SessionStorage (JSONL persistence)
- [ ] Code structure indexer
- [ ] Entity tracker
- [ ] Contextual query resolver

### API Extensions
- [ ] POST /sessions/create
- [ ] GET /sessions/{id}/summary
- [ ] POST /get_context (with session_id param)
- [ ] GET /sessions/list
- [ ] DELETE /sessions/{id}

### Advanced
- [ ] Cross-session linking
- [ ] Session templates (feature, bugfix, etc.)
- [ ] Auto-session-type detection
- [ ] Session analytics dashboard

### Migration
- [ ] v5 compatibility mode
- [ ] Migration script for existing queries
- [ ] Documentation update
- [ ] Deprecation plan for v5-only mode

---

## Timeline Estimate

- **Phase 1:** 2 weeks (session scaffolding)
- **Phase 2:** 1 week (code indexing)
- **Phase 3:** 2 weeks (contextual resolution)
- **Phase 4:** 1 week (multi-session)

**Total:** 6 weeks for v6.0.0

---

## Conclusion

MCP v6 transforma el sistema de **pure retrieval** a **intelligent development assistant** que:
- Recuerda el contexto de desarrollo
- Resuelve referencias conversacionales
- Trackea entidades de código eficientemente
- Mantiene memoria sin duplicar código completo

Es la evolución natural y alineada con el valor que OpenAI demuestra en su guía de Context Engineering.
