@echo off
REM Script de inicio para Enhanced MCP Server con Context Feedback System
REM Versión 2.0.0 - Prevención de alucinaciones y coherencia de proyecto

echo ========================================
echo  Enhanced MCP Server v2.0.0
echo  Context Feedback System Activado
echo ========================================
echo.

REM Verificar que Python está disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado. Instale Python 3.8+ y agregue al PATH.
    pause
    exit /b 1
)

REM Cambiar al directorio del servidor
cd /d "%~dp0servers\context-query"

REM Verificar archivos esenciales
if not exist "enhanced_mcp_server.py" (
    echo ERROR: enhanced_mcp_server.py no encontrado
    pause
    exit /b 1
)

if not exist "context_feedback_system.py" (
    echo ERROR: context_feedback_system.py no encontrado
    pause
    exit /b 1
)

if not exist "feature.md" (
    echo ADVERTENCIA: feature.md no encontrado, se creará automáticamente
)

echo Verificando dependencias...
python -c "import json, sys, logging, time, pathlib" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Dependencias de Python faltantes
    pause
    exit /b 1
)

echo.
echo Características activas:
echo   ✅ Prevención de alucinaciones
echo   ✅ Análisis de código existente obligatorio  
echo   ✅ Lectura de feature.md antes de responder
echo   ✅ Gestión de tareas con retroalimentación
echo   ✅ Detección de duplicación de código
echo   ✅ Ciclo 2 tareas → contexto → 1 tarea → contexto
echo.

echo Iniciando Enhanced MCP Server...
echo Presione Ctrl+C para detener
echo.

REM Ejecutar el servidor mejorado
python enhanced_mcp_server.py

echo.
echo Servidor detenido.
pause
