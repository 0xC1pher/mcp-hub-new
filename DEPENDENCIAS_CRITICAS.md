# 🔗 DEPENDENCIAS CRÍTICAS - MCP HUB

## 📊 Mapa de Dependencias entre Componentes Legacy

### 🏗️ ARQUITECTURA ACTUAL
```
unified_mcp_server.py (CORE)
├── ConsolidatedACESystem
│   ├── AnalysisEngine
│   ├── CurationEngine  
│   └── EvolutionEngine
├── AdvancedQueryOptimizer
├── AdvancedScorer
├── MemoryManager
└── ContextCurator

optimized_mcp_server.py (PERFORMANCE)
├── TokenBudgetManager ⭐ (CRÍTICO)
├── MultiLevelCache ⭐ (CRÍTICO)
├── SemanticChunker ⭐ (CRÍTICO)
├── CacheIndexingSystem
├── AdvancedSemanticProcessor
└── QueryOptimizer

enhanced_mcp_server.py (SAFETY)
├── HallucinationDetector ⭐ (CRÍTICO)
├── ContextValidator ⭐ (CRÍTICO)
├── ModelGuidanceEngine
└── EnhancedMCPServer
```

## 🔄 DEPENDENCIAS CRÍTICAS IDENTIFICADAS

### 1. **TokenBudgetManager** (`optimized_mcp_server.py`)
```python
DEPENDE DE:
- Ninguna dependencia externa crítica
- Configuración: max_tokens, reserved_tokens

USADO POR:
- OptimizedMCPContextServer (línea 1715)
- Todos los métodos de procesamiento de queries

CRITICIDAD: ⭐⭐⭐ ALTA
RAZÓN: Previene overflow de tokens y optimiza rendimiento
```

### 2. **MultiLevelCache** (`optimized_mcp_server.py`)
```python
DEPENDE DE:
- CacheIndexingSystem (para indexado inteligente)
- hashlib (para hashing de keys)
- json (para serialización)

USADO POR:
- QueryOptimizer (línea 1718)
- Todos los métodos de búsqueda

CRITICIDAD: ⭐⭐⭐ ALTA
RAZÓN: >85% hit rate objetivo, rendimiento crítico
```

### 3. **SemanticChunker** (`optimized_mcp_server.py`)
```python
DEPENDE DE:
- re (regex para procesamiento)
- Configuración: chunk_size, overlap

USADO POR:
- OptimizedMCPContextServer (línea 1716)
- Procesamiento de documentos largos

CRITICIDAD: ⭐⭐⭐ ALTA
RAZÓN: Chunking inteligente esencial para contexto
```

### 4. **ConsolidatedACESystem** (`unified_mcp_server.py`)
```python
DEPENDE DE:
- AnalysisEngine
- CurationEngine  
- EvolutionEngine
- AdvancedScorer

USADO POR:
- UnifiedMCPContextServer (línea 680)
- Procesamiento principal de queries

CRITICIDAD: ⭐⭐⭐ ALTA
RAZÓN: Sistema central de análisis y evolución
```

### 5. **HallucinationDetector + ContextValidator** (`enhanced_mcp_server.py`)
```python
DEPENDE DE:
- Base de conocimiento verificada
- Patrones de detección de inconsistencias

USADO POR:
- EnhancedMCPServer (líneas 511-512)
- Validación de todas las respuestas

CRITICIDAD: ⭐⭐⭐ CRÍTICA
RAZÓN: Prevención de alucinaciones, seguridad del sistema
```

## ⚠️ RIESGOS DE MIGRACIÓN

### 🚨 ALTO RIESGO
1. **TokenBudgetManager** - Si se pierde, overflow de tokens garantizado
2. **HallucinationDetector** - Si se pierde, alucinaciones no detectadas
3. **MultiLevelCache** - Si se pierde, rendimiento degradado >50%

### 🟡 MEDIO RIESGO
1. **SemanticChunker** - Chunking básico puede funcionar temporalmente
2. **AdvancedScorer** - Scoring básico disponible como fallback

### 🟢 BAJO RIESGO
1. **MemoryManager** - Gestión básica de memoria suficiente inicialmente
2. **ContextCurator** - Funcionalidad opcional, mejora calidad

## 📋 ORDEN DE MIGRACIÓN RECOMENDADO

### PRIORIDAD 1 (CRÍTICA)
1. **TokenBudgetManager** → `core/shared/token_manager.py`
2. **HallucinationDetector + ContextValidator** → `core/shared/safety_system.py`

### PRIORIDAD 2 (ALTA)
3. **MultiLevelCache** → `core/intelligent_cache/cache_system.py`
4. **SemanticChunker** → `core/shared/chunking_system.py`

### PRIORIDAD 3 (MEDIA)
5. **ConsolidatedACESystem** → `core/context_query/ace_system.py`
6. **AdvancedScorer** → `core/shared/advanced_scorer.py`

### PRIORIDAD 4 (BAJA)
7. **MemoryManager** → `core/shared/memory_manager.py`
8. **QueryOptimizer** → `core/context_query/query_optimizer.py`

## 🔧 CONFIGURACIONES CRÍTICAS

### TokenBudgetManager
```python
max_tokens: 4000        # Límite máximo
reserved_tokens: 500    # Tokens reservados para respuesta
priority_threshold: 0.8 # Umbral de priorización
```

### MultiLevelCache
```python
l1_size: 100           # Cache L1 en memoria
l2_size: 1000          # Cache L2 en disco
disk_cache_mb: 50      # Límite cache en disco
```

### SemanticChunker
```python
chunk_size: 600        # Tamaño óptimo de chunk
overlap: 50            # Overlap entre chunks
min_chunk_size: 100    # Tamaño mínimo
```

## ✅ VALIDACIÓN POST-MIGRACIÓN

### Tests Obligatorios
- [ ] TokenBudgetManager mantiene límites
- [ ] Cache mantiene >85% hit rate
- [ ] HallucinationDetector funciona
- [ ] SemanticChunker produce chunks válidos
- [ ] Sistema ACE procesa correctamente

### Métricas de Éxito
- [ ] Tiempo respuesta <100ms mantenido
- [ ] Uso memoria <50MB mantenido  
- [ ] 0 alucinaciones detectadas
- [ ] Compatibilidad 100% con APIs existentes
