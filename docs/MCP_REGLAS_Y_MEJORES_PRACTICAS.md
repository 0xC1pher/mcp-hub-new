# 📋 Reglas y Mejores Prácticas - MCP v2.0

## Sistema de Base de Datos Vectorizada para Yari-System

---

## 🎯 Reglas Fundamentales del Sistema

### 1. **Búsqueda Híbrida por Defecto**

**Regla:** Siempre usar búsqueda híbrida (semántica + keywords) a menos que se especifique lo contrario.

**Razón:** Balance óptimo entre precisión semántica y coincidencia exacta.

**Implementación:**
```python
# ✅ CORRECTO
response = mcp.query(
    query_text="crear paciente",
    search_mode='hybrid'  # Por defecto
)

# ❌ EVITAR (solo en casos específicos)
response = mcp.query(
    query_text="crear paciente",
    search_mode='semantic'  # Solo para conceptos abstractos
)
```

**Excepciones:**
- **Semántica pura:** Conceptos médicos abstractos ("síntomas de diabetes")
- **Keywords:** Búsqueda de código exacto ("def crear_paciente")

---

### 2. **Límite de Resultados: 5 por Defecto**

**Regla:** Limitar resultados a 5 por consulta para evitar sobrecarga cognitiva.

**Razón:** Estudios muestran que usuarios solo revisan los primeros 3-5 resultados.

**Implementación:**
```python
# ✅ CORRECTO
response = mcp.query(
    query_text="historia clínica",
    n_results=5  # Por defecto
)

# ⚠️ USAR CON PRECAUCIÓN
response = mcp.query(
    query_text="historia clínica",
    n_results=20  # Solo si es necesario
)
```

**Excepciones:**
- Análisis exhaustivo de código
- Generación de documentación completa
- Auditorías de seguridad

---

### 3. **Cache Automático Siempre Activado**

**Regla:** El cache debe estar activado por defecto en todas las consultas.

**Razón:** Mejora rendimiento 100-1000x en consultas repetidas.

**Implementación:**
```python
# ✅ CORRECTO
response = mcp.query(
    query_text="módulo de pacientes",
    use_cache=True  # Por defecto
)

# ❌ SOLO PARA DEBUGGING
response = mcp.query(
    query_text="módulo de pacientes",
    use_cache=False  # Evitar en producción
)
```

**Excepciones:**
- Debugging de resultados
- Testing de algoritmos
- Verificación de indexación

---

### 4. **Respuestas en Español con Contexto Médico**

**Regla:** Todas las respuestas deben estar en español y considerar terminología médica.

**Razón:** Sistema diseñado para personal médico hispanohablante.

**Implementación:**
```python
# Sistema automáticamente:
# - Expande queries con sinónimos médicos
# - Prioriza contenido en español
# - Reconoce términos médicos

# Ejemplo de expansión automática:
query = "paciente"
# Expandido a: "paciente enfermo usuario asegurado"
```

**Diccionario de Sinónimos Médicos:**
- `paciente` → enfermo, usuario, asegurado
- `doctor` → médico, profesional, especialista
- `consulta` → cita, atención, visita
- `diagnóstico` → evaluación, valoración
- `tratamiento` → terapia, medicación
- `emergencia` → urgencia, crítico

---

### 5. **Prioridad: Precisión sobre Velocidad**

**Regla:** En contextos médicos, priorizar precisión de resultados sobre velocidad de respuesta.

**Razón:** Información médica incorrecta puede tener consecuencias graves.

**Implementación:**
```python
# Sistema automáticamente:
# - Usa chunking semántico (no por tamaño)
# - Valida relevancia de resultados
# - Deduplica contenido similar
# - Post-procesa para mejorar precisión

# Umbral mínimo de similitud
min_similarity = 0.5  # 50% de similitud mínima
```

**Métricas de Calidad:**
- **Similitud mínima:** 0.5 (50%)
- **Relevancia combinada:** (similitud + keywords) / 2
- **Deduplicación:** Contenido único por preview de 100 chars

---

## 🔧 Mejores Prácticas de Uso

### Indexación

#### ✅ DO: Indexación Incremental Regular

```bash
# Cron job diario a las 2 AM
0 2 * * * cd /path/to/softmedic && python manage.py mcp_index index
```

**Beneficios:**
- Solo procesa archivos nuevos/modificados
- 10x más rápido que reindexación completa
- Mantiene índice actualizado

#### ❌ DON'T: Reindexación Completa Frecuente

```bash
# ❌ EVITAR esto diariamente
python manage.py mcp_index reindex
```

**Razones:**
- Consume muchos recursos
- Innecesario si archivos no cambiaron
- Elimina cache de índice

---

### Consultas

#### ✅ DO: Consultas Específicas y Contextuales

```python
# ✅ BUENA CONSULTA
response = mcp.query(
    query_text="cómo validar datos de paciente en el formulario de registro",
    n_results=5,
    search_mode='hybrid'
)

# ❌ MALA CONSULTA
response = mcp.query(
    query_text="paciente",  # Demasiado genérica
    n_results=50  # Demasiados resultados
)
```

**Características de Buenas Consultas:**
- Específicas y descriptivas
- Incluyen contexto (dónde, qué, cómo)
- Usan terminología del dominio
- Longitud: 5-15 palabras

#### ✅ DO: Usar Filtros Cuando Sea Posible

```python
# ✅ FILTRAR POR CATEGORÍA
response = mcp.query(
    query_text="configuración de base de datos",
    filters={'category': 'config'}
)

# ✅ FILTRAR POR TIPO DE ARCHIVO
response = mcp.query(
    query_text="modelo de paciente",
    filters={'file_type': '.py'}
)
```

---

### Optimización

#### ✅ DO: Optimización Semanal

```bash
# Cron job semanal (domingos a las 3 AM)
0 3 * * 0 cd /path/to/softmedic && python manage.py mcp_index optimize
```

**Acciones de Optimización:**
- Limpia cache expirado
- Reorganiza datos entre niveles de cache
- Actualiza índices incrementales
- Verifica integridad

#### ✅ DO: Monitoreo de Métricas

```python
# Verificar estadísticas regularmente
stats = mcp.get_system_stats()

# Alertar si:
if stats['cache_stats']['l1']['hit_rate'] < 50:
    alert("Hit rate bajo en cache L1")

if stats['query_stats']['avg_response_time_ms'] > 500:
    alert("Tiempo de respuesta alto")
```

---

### Cache

#### ✅ DO: Configurar Tamaños Según Recursos

```python
# Servidor con 8GB RAM
mcp.cache = SmartCache(
    l1_size=1000,   # 1000 items
    l2_size=5000,   # 5000 items
    cache_dir='./cache'
)

# Servidor con 4GB RAM
mcp.cache = SmartCache(
    l1_size=500,    # 500 items
    l2_size=2000,   # 2000 items
    cache_dir='./cache'
)
```

#### ✅ DO: Limpiar Cache Periódicamente

```python
# Limpiar cache expirado
mcp.cache.cleanup_expired()

# Resetear cache específico si hay problemas
mcp.reset_cache(level='l1')  # Solo L1
mcp.reset_cache(level='l2')  # Solo L2
mcp.reset_cache()  # Todo el cache
```

---

## 🚫 Anti-Patrones (Qué NO Hacer)

### 1. ❌ Consultas Demasiado Genéricas

```python
# ❌ MAL
response = mcp.query("paciente")
response = mcp.query("código")
response = mcp.query("función")

# ✅ BIEN
response = mcp.query("validación de datos de paciente en formulario")
response = mcp.query("código de autenticación de usuarios")
response = mcp.query("función para calcular edad del paciente")
```

### 2. ❌ Deshabilitar Cache sin Razón

```python
# ❌ MAL (en producción)
for query in queries:
    response = mcp.query(query, use_cache=False)

# ✅ BIEN
for query in queries:
    response = mcp.query(query, use_cache=True)
```

### 3. ❌ Reindexar en Cada Consulta

```python
# ❌ MAL
def search(query):
    mcp.initialize_index(force_reindex=True)  # ¡NO!
    return mcp.query(query)

# ✅ BIEN
def search(query):
    return mcp.query(query)  # El índice ya está actualizado
```

### 4. ❌ Ignorar Errores de Health Check

```python
# ❌ MAL
health = mcp.health_check()
# Ignorar si status != 'healthy'

# ✅ BIEN
health = mcp.health_check()
if health['status'] != 'healthy':
    logger.error(f"MCP degradado: {health}")
    # Tomar acción correctiva
```

### 5. ❌ No Monitorear Métricas

```python
# ❌ MAL
# Nunca revisar estadísticas

# ✅ BIEN
# Monitoreo regular
stats = mcp.get_system_stats()
if stats['cache_stats']['l1']['hit_rate'] < 50:
    # Investigar por qué el hit rate es bajo
    pass
```

---

## 📊 Umbrales y Límites Recomendados

### Rendimiento

| Métrica | Óptimo | Aceptable | Crítico |
|---------|--------|-----------|---------|
| **Tiempo de respuesta (cache)** | < 20ms | < 50ms | > 100ms |
| **Tiempo de respuesta (DB)** | < 200ms | < 500ms | > 1000ms |
| **Hit rate L1** | > 70% | > 50% | < 30% |
| **Hit rate L2** | > 60% | > 40% | < 20% |
| **Memoria usada** | < 300MB | < 500MB | > 1GB |

### Almacenamiento

| Componente | Tamaño Típico | Máximo Recomendado |
|------------|---------------|-------------------|
| **ChromaDB** | 100-200MB | 500MB |
| **Cache L3** | 50-100MB | 300MB |
| **Índice** | < 1MB | 5MB |
| **Total** | 150-300MB | 800MB |

### Consultas

| Parámetro | Recomendado | Máximo |
|-----------|-------------|--------|
| **n_results** | 5 | 20 |
| **Longitud query** | 5-15 palabras | 50 palabras |
| **Consultas/minuto** | < 60 | 100 |

---

## 🔐 Seguridad y Privacidad

### Archivos Excluidos Automáticamente

El sistema **NO indexa**:
- `.env` y archivos de configuración sensibles
- `secrets/`, `private/`, `confidential/`
- Archivos en `.gitignore`
- Directorios: `venv/`, `node_modules/`, `__pycache__/`
- Archivos binarios: `.pyc`, `.so`, `.dll`
- Bases de datos: `.db`, `.sqlite3`

### Datos Médicos Sensibles

```python
# ✅ CORRECTO: No indexar datos de pacientes reales
# El sistema solo indexa CÓDIGO, no datos

# Los datos médicos están en:
# - Base de datos (no indexada)
# - Archivos de backup (excluidos)

# El MCP indexa:
# - Código fuente (.py, .js)
# - Documentación (.md, .txt)
# - Configuraciones (.json, .yaml)
```

---

## 🎓 Casos de Uso Recomendados

### 1. Asistente de Desarrollo

```python
# Encontrar cómo implementar una funcionalidad
response = mcp.query(
    "cómo crear un nuevo paciente en el sistema",
    search_mode='hybrid',
    n_results=5
)

# Mostrar código relevante al desarrollador
for result in response['results']:
    print(f"Archivo: {result['metadata']['source']}")
    print(f"Código: {result['content'][:200]}...")
```

### 2. Documentación Automática

```python
# Generar documentación de un módulo
context = mcp.get_context_for_module(
    module_name='pacientes',
    context_type='code',
    n_results=10
)

# Usar contexto para generar docs con IA
docs = generate_documentation(context)
```

### 3. Code Review Automático

```python
# Buscar patrones similares en el código
response = mcp.query(
    "validación de formularios de pacientes",
    search_mode='hybrid'
)

# Comparar con implementación actual
for result in response['results']:
    compare_implementations(current_code, result['content'])
```

### 4. Búsqueda de Configuración

```python
# Encontrar configuraciones específicas
response = mcp.query(
    "configuración de conexión a base de datos PostgreSQL",
    filters={'category': 'config'}
)
```

---

## 📈 Métricas de Éxito

### KPIs del Sistema

1. **Rendimiento:**
   - Tiempo promedio < 200ms
   - Hit rate cache > 70%
   - Throughput > 50 q/s

2. **Calidad:**
   - Precisión > 80%
   - Resultados relevantes > 90%
   - Deduplicación > 95%

3. **Recursos:**
   - Memoria < 500MB
   - Disco < 300MB
   - CPU < 30% (idle)

4. **Disponibilidad:**
   - Uptime > 99.9%
   - Health check: healthy
   - Errores < 0.1%

---

## 🔄 Mantenimiento Regular

### Diario
```bash
# Indexación incremental
0 2 * * * python manage.py mcp_index index
```

### Semanal
```bash
# Optimización completa
0 3 * * 0 python manage.py mcp_index optimize

# Verificación de salud
0 4 * * 0 python manage.py mcp_index health > /var/log/mcp_health.log
```

### Mensual
```bash
# Estadísticas completas
python manage.py mcp_index stats --json > stats_$(date +%Y%m).json

# Benchmark de rendimiento
python benchmark_mcp.py
```

---

## 🆘 Troubleshooting

### Problema: Consultas Lentas

**Diagnóstico:**
```bash
python manage.py mcp_index stats
```

**Solución:**
```bash
# Si hit rate < 50%
python manage.py mcp_index optimize

# Si documentos < 100
python manage.py mcp_index index
```

### Problema: Alto Uso de Memoria

**Solución:**
```python
# Reducir tamaños de cache
mcp.cache = SmartCache(l1_size=200, l2_size=800)
```

### Problema: Resultados Irrelevantes

**Solución:**
```python
# Aumentar umbral de similitud
results = [r for r in response['results'] if r['similarity'] > 0.7]

# Usar búsqueda semántica pura
response = mcp.query(query, search_mode='semantic')
```

---

## 📚 Referencias

- **Documentación completa:** `docs/MCP_OPTIMIZADO_V2.md`
- **Quick start:** `README_MCP_V2.md`
- **API Reference:** Docstrings en `mcp_core/*.py`
- **Benchmark:** `benchmark_mcp.py`

---

## ✅ Checklist de Implementación

- [ ] Instalar dependencias: `pip install -r requirements-mcp.txt`
- [ ] Ejecutar setup: `python setup_mcp_v2.py`
- [ ] Indexar proyecto: `python manage.py mcp_index index`
- [ ] Verificar salud: `python manage.py mcp_index health`
- [ ] Configurar cron jobs para mantenimiento
- [ ] Establecer monitoreo de métricas
- [ ] Documentar casos de uso específicos
- [ ] Entrenar al equipo en mejores prácticas

---

**Última actualización:** 2025-01-19  
**Versión:** 2.0.0  
**Autor:** Sistema MCP Optimizado para Yari-System
