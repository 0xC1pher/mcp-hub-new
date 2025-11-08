@echo off
REM ============================================
REM MCP HUB v4.0 - Sistema Principal Unificado
REM ============================================

cd /d "%~dp0"

:MAIN_MENU
cls
echo ================================================================================
echo  MCP HUB v4.0 - Sistema Principal
echo ================================================================================
echo.
echo  PRIMERA VEZ? Lee: EMPIEZA_AQUI.md
echo  PERDIDO? Este archivo es todo lo que necesitas
echo.
echo ================================================================================
echo  MENU PRINCIPAL
echo ================================================================================
echo.
echo [1] Iniciar Sistema MCP    ^<- Inicia el sistema principal
echo [2] Demo Completo          ^<- Ver todas las caracteristicas
echo [3] Instalar Dependencias  ^<- EJECUTA ESTO PRIMERO (primera vez)
echo [4] Configurar IDE         ^<- Windsurf o VS Code
echo [5] Debug Interactivo      ^<- Probar queries especificas
echo [6] Sistema Legacy v3      ^<- Sistema anterior (compatibilidad)
echo.
echo [?] Ayuda
echo [0] Salir
echo.
set /p choice="Selecciona una opcion (0-6 o ?): "

if "%choice%"=="1" goto START_SYSTEM
if "%choice%"=="2" goto DEMO_SYSTEM
if "%choice%"=="3" goto INSTALL_DEPS
if "%choice%"=="4" goto IDE_CONFIG
if "%choice%"=="5" goto DEBUG_MODE
if "%choice%"=="6" goto LEGACY_SYSTEM
if "%choice%"=="?" goto SHOW_HELP
if "%choice%"=="0" goto END
echo.
echo  Opcion invalida. Selecciona 0-6 o ?
timeout /t 2 >nul
goto MAIN_MENU

:SHOW_HELP
cls
echo ================================================================================
echo  AYUDA RAPIDA - MCP HUB v4.0
echo ================================================================================
echo.
echo  QUE OPCION USAR?
echo  ---------------
echo  [3] PRIMERO - Instala dependencias (solo primera vez)
echo  [1] DESPUES - Inicia el sistema principal
echo  [2] DEMO    - Ver que puede hacer el sistema
echo  [4] IDE     - Configurar tu editor (opcional)
echo  [5] DEBUG   - Herramienta de testing (desarrollo)
echo  [6] LEGACY  - Sistema v3 anterior (compatibilidad)
echo.
echo  FLUJO RECOMENDADO PRIMERA VEZ:
echo  ------------------------------
echo  1. Opcion [3] - Instalar dependencias (2-3 minutos)
echo  2. Opcion [1] - Iniciar sistema
echo  3. Listo!
echo.
echo  QUE ES CADA MODO:
echo  -----------------
echo  Sistema Principal [1]: Sistema completo v4.0 con todas las caracteristicas
echo    - Dynamic Chunking Adaptativo
echo    - Multi-Vector Retrieval (MVR)
echo    - Query Expansion Automatica
echo    - Confidence Calibration Dinamica
echo    - 10+ caracteristicas avanzadas
echo.
echo  Sistema Legacy [6]: Version anterior v3.0 (solo compatibilidad)
echo    - Sistema estable probado
echo    - Menos caracteristicas
echo    - Disponible si el nuevo tiene problemas
echo.
echo  RECOMENDACION: Usa [1] Sistema Principal
echo.
echo  DOCUMENTACION:
echo  - EMPIEZA_AQUI.md ........... Guia completa
echo  - MAPA_DE_ARCHIVOS.txt ...... Estructura del proyecto
echo  - core\advanced_features\README.md .. Detalles tecnicos
echo.
echo ================================================================================
pause
goto MAIN_MENU

:START_SYSTEM
cls
echo ================================================================================
echo  Iniciando MCP HUB v4.0 - Sistema Principal
echo ================================================================================
echo.
echo  Verificando sistema...
python -c "from core.advanced_features import create_orchestrator; print('Sistema verificado correctamente')" 2>nul
if errorlevel 1 (
    echo  Error: Sistema no disponible
    echo.
    echo  Posibles soluciones:
    echo  1. Ejecuta opcion [3] para instalar dependencias
    echo  2. Verifica que Python este instalado
    echo  3. Usa opcion [6] para sistema legacy (alternativa)
    echo.
    pause
    goto MAIN_MENU
)

echo  Sistema OK
echo.
echo  Modos disponibles:
echo  [1] Fast         - Rapido, caracteristicas basicas
echo  [2] Balanced     - Equilibrado (recomendado)
echo  [3] Complete     - Completo, todas las caracteristicas
echo.
set /p mode="Selecciona modo (1-3, Enter=Balanced): "

if "%mode%"=="" set mode=2
if "%mode%"=="1" set MODE_NAME=fast
if "%mode%"=="2" set MODE_NAME=balanced
if "%mode%"=="3" set MODE_NAME=comprehensive
if not defined MODE_NAME set MODE_NAME=balanced

echo.
echo  Iniciando sistema en modo %MODE_NAME%...
echo.
python -m core.advanced_features --mode %MODE_NAME%
if errorlevel 1 (
    echo.
    echo  El sistema encontro un error durante la ejecucion
    echo  Revisa los logs en: logs\
)
pause
goto MAIN_MENU

:DEMO_SYSTEM
cls
echo ================================================================================
echo  Demo Completo - MCP HUB v4.0
echo ================================================================================
echo.
echo  Este demo mostrara todas las caracteristicas del sistema:
echo  - Dynamic Chunking Adaptativo
echo  - Multi-Vector Retrieval (MVR)
echo  - Query Expansion Automatica
echo  - Confidence Calibration Dinamica
echo  - Virtual Chunks con MP4 Storage
echo  - Sistema de Feedback
echo  - Metricas de rendimiento
echo.
echo  Tiempo estimado: 5-10 minutos
echo.
set /p confirm="Continuar con el demo? (s/n): "
if /i not "%confirm%"=="s" goto MAIN_MENU

echo.
echo  Ejecutando demo completo...
python core\advanced_features\run_system.py
pause
goto MAIN_MENU

:INSTALL_DEPS
cls
echo ================================================================================
echo  Instalando Dependencias
echo ================================================================================
echo.
echo  Instalando dependencias necesarias para el sistema...
echo  Esto puede tomar 2-3 minutos
echo.
python install_deps.py
pause
goto MAIN_MENU

:IDE_CONFIG
cls
echo ================================================================================
echo  Configuracion de IDE
echo ================================================================================
echo.
echo  Configura tu entorno de desarrollo:
echo.
echo  [1] Windsurf IDE  - Configuracion automatica
echo  [2] VS Code       - Ya configurado (.vscode/)
echo  [3] Volver al menu
echo.
set /p ide_choice="Selecciona IDE (1-3): "

if "%ide_choice%"=="1" (
    echo.
    echo  Configurando Windsurf IDE...
    python scripts\start_mcp_windsurf.py
    pause
) else if "%ide_choice%"=="2" (
    echo.
    echo  VS Code - Configuracion
    echo  =======================
    echo  El proyecto ya esta configurado para VS Code
    echo.
    echo  Archivos de configuracion:
    echo  - .vscode\settings.json  (Configuracion del workspace)
    echo  - .vscode\launch.json    (Configuraciones de debug)
    echo  - .vscode\tasks.json     (Tareas automatizadas)
    echo.
    echo  Para usar:
    echo  1. Abre el proyecto en VS Code
    echo  2. Presiona F5 para debug
    echo  3. Usa Ctrl+Shift+P para tasks
    echo.
    pause
) else if "%ide_choice%"=="3" (
    goto MAIN_MENU
)
goto MAIN_MENU

:DEBUG_MODE
cls
echo ================================================================================
echo  Modo Debug Interactivo
echo ================================================================================
echo.
echo  Herramienta de testing para desarrollo
echo  Permite probar queries especificas y ver resultados detallados
echo.
python debug_query.py --interactive
pause
goto MAIN_MENU

:LEGACY_SYSTEM
cls
echo ================================================================================
echo  Sistema Legacy v3.0 (NO DISPONIBLE)
echo ================================================================================
echo.
echo  El sistema legacy fue eliminado en la limpieza.
echo.
echo  SOLUCION: Usa el sistema principal v4.0 (opcion [1])
echo  Es superior en todos los aspectos.
echo.
pause
goto MAIN_MENU

:END
cls
echo ================================================================================
echo  MCP HUB v4.0 - Gracias por usar el sistema
echo ================================================================================
echo.
echo  RECURSOS:
echo  - EMPIEZA_AQUI.md .............. Guia completa
echo  - MAPA_DE_ARCHIVOS.txt ......... Estructura del proyecto
echo  - core\advanced_features\README.md .. Documentacion tecnica
echo.
echo  INICIO RAPIDO:
echo  1. START_MCP.bat
echo  2. Opcion [3] (instalar dependencias)
echo  3. Opcion [1] (iniciar sistema)
echo.
echo  DEBUG:
echo  - Logs: logs\
echo  - Debug tool: python debug_query.py --interactive
echo  - Health check: python -c "from core.advanced_features import create_orchestrator; print('OK')"
echo.
echo ================================================================================
pause
exit /b 0
