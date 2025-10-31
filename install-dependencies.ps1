# Script para instalar dependencias de los servidores MCP

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Instalando Dependencias MCP" -ForegroundColor Cyan
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

Write-Host ""
Write-Host "Instalando dependencias opcionales..." -ForegroundColor Yellow
Write-Host ""

# Lista de dependencias opcionales
$dependencies = @(
    @{Name="zstandard"; Description="Compresión avanzada para Memory Context MCP"},
    @{Name="msgpack"; Description="Serialización eficiente"},
    @{Name="numpy"; Description="Operaciones numéricas"}
)

$installed = 0
$failed = 0

foreach ($dep in $dependencies) {
    Write-Host "Instalando $($dep.Name)..." -ForegroundColor Cyan
    Write-Host "  → $($dep.Description)" -ForegroundColor Gray
    
    try {
        $result = pip install $dep.Name 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $($dep.Name) instalado" -ForegroundColor Green
            $installed++
        } else {
            Write-Host "  ⚠ $($dep.Name) falló (no crítico)" -ForegroundColor Yellow
            $failed++
        }
    } catch {
        Write-Host "  ⚠ $($dep.Name) falló (no crítico)" -ForegroundColor Yellow
        $failed++
    }
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Resumen de instalación:" -ForegroundColor Green
Write-Host "  ✓ Instalados: $installed" -ForegroundColor Green
Write-Host "  ⚠ Fallidos: $failed (no críticos)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($installed -gt 0) {
    Write-Host "✓ Los servidores MCP ahora tienen más funcionalidades" -ForegroundColor Green
} else {
    Write-Host "⚠ Los servidores funcionarán en modo básico" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Ahora puedes ejecutar: .\start-all-mcp.ps1" -ForegroundColor Cyan
Write-Host ""

Read-Host "Presione Enter para continuar"
