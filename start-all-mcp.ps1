# Script PowerShell para ejecutar los 4 servidores MCP
# Usa rutas absolutas reales

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MCP HUB - Iniciando Servidores" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python no encontrado" -ForegroundColor Red
    Read-Host "Presione Enter para salir"
    exit 1
}

# Directorio base
$baseDir = "C:\Users\0x4171341\Desktop\CONSULTORIO\SoftMedic -Imca\mcp-hub"

# Definir servidores con rutas absolutas
$servers = @(
    @{
        Name = "Memory Context MCP"
        Path = "$baseDir\core\memory_context\memory_context_mcp.py"
        WorkDir = "$baseDir\core\memory_context"
    },
    @{
        Name = "Enhanced MCP Server"
        Path = "$baseDir\legacy\enhanced\enhanced_mcp_server.py"
        WorkDir = "$baseDir\legacy\enhanced"
    },
    @{
        Name = "Optimized MCP Server"
        Path = "$baseDir\legacy\optimized\optimized_mcp_server.py"
        WorkDir = "$baseDir\legacy\optimized"
    },
    @{
        Name = "Unified MCP Server"
        Path = "$baseDir\legacy\unified\unified_mcp_server.py"
        WorkDir = "$baseDir\legacy\unified"
    }
)

Write-Host "Verificando archivos de servidores..." -ForegroundColor Yellow
Write-Host ""

$allExist = $true
foreach ($server in $servers) {
    if (Test-Path $server.Path) {
        Write-Host "✓ $($server.Name): OK" -ForegroundColor Green
    } else {
        Write-Host "✗ $($server.Name): NO ENCONTRADO" -ForegroundColor Red
        Write-Host "  Ruta: $($server.Path)" -ForegroundColor Gray
        $allExist = $false
    }
}

Write-Host ""

if (-not $allExist) {
    Write-Host "ERROR: Algunos servidores no existen" -ForegroundColor Red
    Read-Host "Presione Enter para salir"
    exit 1
}

Write-Host "Iniciando servidores..." -ForegroundColor Yellow
Write-Host ""

$counter = 1
foreach ($server in $servers) {
    Write-Host "[$counter/4] Iniciando $($server.Name)..." -ForegroundColor Cyan
    
    # Iniciar servidor en nueva ventana
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$($server.WorkDir)'; python '$($server.Path)'"
    ) -WindowStyle Normal
    
    Start-Sleep -Milliseconds 500
    $counter++
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ 4 servidores MCP ejecutándose:" -ForegroundColor Green
Write-Host "  - Memory Context MCP" -ForegroundColor White
Write-Host "  - Enhanced MCP Server" -ForegroundColor White
Write-Host "  - Optimized MCP Server" -ForegroundColor White
Write-Host "  - Unified MCP Server" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Los servidores están corriendo en ventanas separadas" -ForegroundColor Yellow
Write-Host "Cierra cada ventana para detener el servidor correspondiente" -ForegroundColor Yellow
Write-Host ""

Read-Host "Presione Enter para cerrar este script"
