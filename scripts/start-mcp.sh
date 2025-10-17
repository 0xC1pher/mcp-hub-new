#!/bin/bash

# Script de inicio del servidor MCP Context Query para SoftMedic
# Versión: 1.0.0

set -e  # Salir si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función de logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "servers/context-query/server.py" ]; then
    error "No se encuentra server.py. Ejecuta desde el directorio mcp-hub/"
    exit 1
fi

log "🚀 Iniciando SoftMedic MCP Context Query Server..."

# Verificar archivos requeridos
log "🔍 Verificando archivos requeridos..."

files_ok=true

if [ ! -f "servers/context-query/manifest.json" ]; then
    error "Falta manifest.json"
    files_ok=false
fi

if [ ! -f "servers/context-query/context/project-guidelines.md" ]; then
    error "Falta project-guidelines.md"
    files_ok=false
fi

if [ ! -f "servers/context-query/index/keyword-to-sections.json" ]; then
    error "Falta keyword-to-sections.json"
    files_ok=false
fi

if [ "$files_ok" = false ]; then
    error "Archivos faltantes. Revisa la estructura del proyecto."
    exit 1
fi

success "Archivos requeridos encontrados"

# Ejecutar validación del índice
log "🔍 Validando sincronización del índice..."
if [ -f "scripts/validate-index.py" ]; then
    if python3 scripts/validate-index.py; then
        success "Índice validado correctamente"
    else
        error "Error de validación del índice. Corrige antes de continuar."
        exit 1
    fi
else
    warning "Script de validación no encontrado, omitiendo validación"
fi

# Verificar puerto disponible
PORT=${PORT:-8081}
log "🔌 Verificando puerto $PORT..."

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    error "Puerto $PORT ya está en uso"
    exit 1
fi

success "Puerto $PORT disponible"

# Crear directorio de logs si no existe
mkdir -p logs

# Iniciar servidor
log "🌟 Iniciando servidor MCP..."
log "📋 Endpoints disponibles:"
log "   GET  /manifest         - Manifest MCP"
log "   GET  /health           - Health check"
log "   POST /tools/context.query - Consulta de contexto"
log ""
log "📁 Presiona Ctrl+C para detener"
log ""

# Ejecutar servidor
exec python3 servers/context-query/server.py $PORT
