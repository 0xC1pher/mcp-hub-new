# 🔧 Solución al Error del MCP v2.0

## ❌ Error Detectado

```
ERROR - Error importando mcp_core: No module named 'chromadb'
```

---

## 🎯 Causa del Problema

El servidor MCP v2.0 requiere dependencias adicionales que no están instaladas:
- `chromadb` - Base de datos vectorial
- `sentence-transformers` - Embeddings multilingües
- `transformers` - Modelos de lenguaje
- `torch` - Framework de deep learning

---

## ✅ Solución (En Progreso)

### Paso 1: Instalar Dependencias

```bash
pip install -r requirements-mcp.txt
```

**Estado:** ⏳ Instalando ahora...

### Paso 2: Verificar Instalación

Después de que termine la instalación, ejecuta:

```bash
python manage.py mcp_index health
```

### Paso 3: Indexar Proyecto

```bash
python manage.py mcp_index index
```

---

## 🚀 Alternativa Rápida

Si la instalación tarda mucho o falla, puedes:

### Opción 1: Usar Solo el MCP v1.0

Edita `mcp_config.json` y deshabilita el v2.0:

```json
{
  "mcpServers": {
    "softmedic-context": {
      "command": "python",
      "args": ["...\\optimized_mcp_server.py"],
      "disabled": false
    },
    "softmedic-vector-v2": {
      "command": "python",
      "args": ["...\\mcp_server.py"],
      "disabled": true  // ← Cambiar a true
    }
  }
}
```

### Opción 2: Instalar Dependencias Mínimas

Si torch es muy pesado, puedes instalar solo lo esencial:

```bash
pip install chromadb==0.4.22
pip install sentence-transformers==2.2.2
```

---

## 📊 Tamaño de Dependencias

| Paquete | Tamaño Aproximado |
|---------|-------------------|
| `chromadb` | ~50MB |
| `sentence-transformers` | ~100MB |
| `transformers` | ~200MB |
| `torch` | ~1.5GB ⚠️ |
| **Total** | **~1.85GB** |

**Nota:** La instalación puede tardar 5-15 minutos dependiendo de tu conexión.

---

## 🔍 Verificar Estado de Instalación

### Ver Progreso

```bash
# En otra terminal
pip list | grep -E "chromadb|sentence-transformers|torch"
```

### Verificar si ChromaDB está instalado

```bash
python -c "import chromadb; print('✓ ChromaDB instalado')"
```

---

## 🐛 Problemas Comunes

### Error: "No space left on device"

**Solución:** Libera espacio en disco (necesitas ~2GB libres)

### Error: "Microsoft Visual C++ required"

**Solución (Windows):**
1. Descarga Visual C++ Redistributable
2. O instala torch precompilado: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Error: "Connection timeout"

**Solución:**
```bash
pip install --timeout=300 -r requirements-mcp.txt
```

---

## ✅ Después de la Instalación

### 1. Reiniciar Windsurf

Para que cargue el servidor MCP correctamente.

### 2. Verificar Salud del Sistema

```bash
python manage.py mcp_index health
```

Deberías ver:

```
✓ MCP Service: healthy
✓ Vector Store: 4 collections
✓ Cache: operational
✓ Indexer: ready
```

### 3. Ejecutar Benchmark

```bash
python benchmark_mcp.py
```

---

## 📝 Mejoras Aplicadas al Servidor

He actualizado `mcp_core/mcp_server.py` para:

✅ **No fallar inmediatamente** si faltan dependencias  
✅ **Mostrar mensaje claro** con instrucciones de instalación  
✅ **Listar dependencias requeridas** explícitamente  
✅ **Permitir verificación** antes de intentar iniciar  

---

## 🎯 Estado Actual

- ✅ Servidor MCP v1.0: **Funcionando**
- ⏳ Servidor MCP v2.0: **Instalando dependencias...**
- ✅ Configuración: **Correcta**
- ✅ Código: **Sin errores**

---

## 📞 Próximos Pasos

1. ⏳ **Esperar** a que termine la instalación (~5-10 min)
2. ✅ **Verificar** con `python manage.py mcp_index health`
3. ✅ **Indexar** con `python manage.py mcp_index index`
4. ✅ **Reiniciar** Windsurf
5. ✅ **Probar** el nuevo MCP v2.0

---

## 💡 Tip

Mientras se instalan las dependencias, puedes seguir usando el MCP v1.0 (`softmedic-context`) que ya está funcionando.

---

**Última actualización:** 2025-01-19 22:18  
**Estado:** Instalando dependencias...
