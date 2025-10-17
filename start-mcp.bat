@echo off
echo Iniciando SoftMedic MCP Context Hub...
cd /d "%~dp0"
python servers\context-query\server.py 8081
pause
