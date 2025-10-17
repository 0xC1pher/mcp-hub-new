#!/usr/bin/env python3
"""
Servidor MCP Context Query para SoftMedic - Versión Optimizada 2.0
Implementa protocolo MCP estándar con JSON-RPC sobre stdio
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/context-query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPContextServer:
    """Servidor MCP Context Query optimizado"""
    
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent
        self.guidelines_file = self.base_path / "context" / "project-guidelines.md"
        self.index_file = self.base_path / "index" / "keyword-to-sections.json"
        self.manifest_file = self.base_path / "manifest.json"
        
        # Cache simple
        self.cache = {}
        self.cache_timestamp = 0
        self.cache_ttl = 10
        
        logger.info("Servidor MCP Context Query iniciado")

    def _load_files(self):
        """Carga archivos con cache"""
        current_time = time.time()
        if current_time - self.cache_timestamp < self.cache_ttl and self.cache:
            return self.cache
            
        try:
            # Cargar guidelines
            guidelines_content = ""
            if self.guidelines_file.exists():
                with open(self.guidelines_file, 'r', encoding='utf-8') as f:
                    guidelines_content = f.read()
            
            # Cargar índice
            index_data = {}
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            
            # Cargar manifest
            manifest_data = {}
            if self.manifest_file.exists():
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
            
            self.cache = {
                'guidelines': guidelines_content,
                'index': index_data,
                'manifest': manifest_data
            }
            self.cache_timestamp = current_time
            
            return self.cache
            
        except Exception as e:
            logger.error(f"Error cargando archivos: {e}")
            return {'guidelines': '', 'index': {}, 'manifest': {}}

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extrae secciones del contenido usando delimitadores HTML"""
        sections = {}
        pattern = r'<!-- SECTION_ID:\s*([^>]+)\s*-->(.*?)(?=<!-- SECTION_ID:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for section_id, section_content in matches:
            sections[section_id.strip()] = section_content.strip()
        
        return sections

    def _find_relevant_sections(self, query: str, sections: Dict[str, str], index: Dict[str, List[str]]) -> List[str]:
        """Encuentra secciones relevantes basado en la consulta"""
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        relevant_sections = []
        
        # Buscar en índice
        for word in query_words:
            if word in index:
                relevant_sections.extend(index[word])
        
        # Buscar por palabras clave en contenido
        for section_id, content in sections.items():
            content_lower = content.lower()
            if any(word in content_lower for word in query_words):
                if section_id not in relevant_sections:
                    relevant_sections.append(section_id)
        
        # Eliminar duplicados y limitar a 2 secciones
        return list(dict.fromkeys(relevant_sections))[:2]

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja inicialización del servidor MCP"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "softmedic-context",
                "version": "2.0.0-optimized"
            }
        }

    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista las herramientas disponibles"""
        return {
            "tools": [
                {
                    "name": "context_query",
                    "description": "Obtiene fragmentos relevantes del contexto del proyecto basado en una consulta semántica.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Consulta sobre el contexto del proyecto"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }

    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "context_query":
            query = arguments.get("query", "")
            
            try:
                # Cargar archivos
                files_data = self._load_files()
                guidelines = files_data.get('guidelines', '')
                index = files_data.get('index', {})
                
                # Extraer secciones
                sections = self._extract_sections(guidelines)
                
                # Encontrar secciones relevantes
                relevant_section_ids = self._find_relevant_sections(query, sections, index)
                
                # Construir respuesta
                result_parts = []
                for section_id in relevant_section_ids:
                    if section_id in sections:
                        result_parts.append(f"**{section_id.replace('_', ' ').title()}:**\n\n{sections[section_id]}")
                
                if result_parts:
                    result = "\n\n---\n\n".join(result_parts)
                else:
                    result = f"No se encontró información relevante para la consulta: '{query}'. Las secciones disponibles son: {', '.join(sections.keys())}"
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error procesando consulta: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error procesando la consulta: {str(e)}"
                        }
                    ],
                    "isError": True
                }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Herramienta desconocida: {tool_name}"
                }
            ],
            "isError": True
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            else:
                result = {"error": f"Método no soportado: {method}"}
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error manejando request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    def run(self):
        """Ejecuta el servidor MCP"""
        logger.info("Iniciando servidor MCP Context Query...")
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = self.handle_request(request)
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error decodificando JSON: {e}")
                except Exception as e:
                    logger.error(f"Error procesando línea: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Servidor detenido por usuario")
        except Exception as e:
            logger.error(f"Error en servidor: {e}")

if __name__ == "__main__":
    import time
    import re
    
    server = MCPContextServer()
    server.run()
