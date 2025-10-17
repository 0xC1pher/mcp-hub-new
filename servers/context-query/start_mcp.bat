@echo off
echo ========================================
echo   SoftMedic MCP Context Hub v2.0
echo ========================================
echo.
echo Iniciando servidor MCP Context Query...
echo.

cd /d "%~dp0"

echo Verificando archivos necesarios...
if not exist "context\project-guidelines.md" (
    echo ERROR: No se encuentra project-guidelines.md
    pause
    exit /b 1
)

if not exist "index\keyword-to-sections.json" (
    echo ERROR: No se encuentra keyword-to-sections.json
    pause
    exit /b 1
)

echo Archivos verificados correctamente.
echo.
echo Iniciando servidor...
echo Presiona Ctrl+C para detener el servidor.
echo.

python simple_mcp_server.py

echo.
echo Servidor detenido.
pause
