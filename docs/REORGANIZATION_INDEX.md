# ÍNDICE DE ARCHIVOS MCP REORGANIZADOS

## Estructura Organizada:

### 📁 mcp-hub/
```
mcp-hub/
├── data/                    # Datos y cache
│   ├── chroma_db/          # Base de datos vectorial
│   ├── cache/              # Cache multinivel
│   └── mcp_context.db      # Base de datos de contexto
├── docs/                   # Documentación
├── config/                 # Configuración
├── scripts/                # Scripts de inicio y setup
├── servers/                # Servidores MCP
├── tests/                  # Archivos de prueba
└── backup/                 # Respaldos

```

## Archivos Reorganizados:

### Datos movidos:
- chroma_db/ → mcp-hub/data/chroma_db/
- cache/ → mcp-hub/data/cache/
- mcp_context.db → mcp-hub/data/mcp_context.db

### Documentación movida:
- docs/MCP_*.md → mcp-hub/docs/
- *MCP*.md → mcp-hub/docs/

### Configuración movida:
- requirements-mcp.txt → mcp-hub/config/
- setup_mcp_v2.py → mcp-hub/scripts/
- benchmark_mcp.py → mcp-hub/scripts/

### Directorios eliminados:
- mmcp-hub/ (duplicado)

## Fecha de reorganización: 2025-10-21 23:26:47
