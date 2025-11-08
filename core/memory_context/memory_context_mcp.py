#!/usr/bin/env python3
"""
🧠 Memory Context MCP Server
Solo maneja contexto de memoria - Sin modelo de negocio
Optimizado para almacenamiento eficiente
"""
import json
import sys
import os
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

# Agregar directorio raíz de mcp-hub al PYTHONPATH
current_dir = Path(__file__).resolve().parent
mcp_hub_root = current_dir.parent.parent  # Subir dos niveles: memory_context -> core -> mcp-hub
sys.path.insert(0, str(mcp_hub_root))

# Imports opcionales con fallbacks
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ zstandard no disponible - usando compresión básica")

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    print("⚠️ msgpack no disponible - usando JSON")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ numpy no disponible - funcionalidad limitada")

class MemoryContextMCP:
    def __init__(self, db_path: str = "memory_context.db"):
        self.db_path = db_path
        
        # Configurar compresión según disponibilidad
        if ZSTD_AVAILABLE:
            self.compressor = zstd.ZstdCompressor(level=3)
            self.decompressor = zstd.ZstdDecompressor()
        else:
            self.compressor = None
            self.decompressor = None
        
        self.init_database()
        
    def init_database(self):
        """Inicializa SQLite con esquema optimizado"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla principal de contextos (altamente comprimida)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_hash TEXT UNIQUE NOT NULL,
                compressed_data BLOB NOT NULL,
                metadata BLOB NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT (julianday('now')),
                created_at REAL DEFAULT (julianday('now')),
                context_type TEXT DEFAULT 'conversation'
            )
        ''')
        
        # Índice para búsquedas rápidas
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_context_hash ON memory_contexts(context_hash)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON memory_contexts(last_accessed)
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_context(self, context_data: Dict[str, Any]) -> str:
        """Genera hash único para el contexto"""
        content_str = json.dumps(context_data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def compress_data(self, data: Dict[str, Any]) -> bytes:
        """Comprime datos usando msgpack + zstd"""
        # Serializar con msgpack (más eficiente que JSON)
        packed = msgpack.packb(data, use_bin_type=True)
        # Comprimir con zstd
        compressed = self.compressor.compress(packed)
        return compressed
    
    def decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Descomprime datos"""
        packed = self.decompressor.decompress(compressed_data)
        return msgpack.unpackb(packed, raw=False)
    
    def store_context(self, context_data: Dict[str, Any]) -> str:
        """Almacena contexto en la base de datos"""
        context_hash = self.hash_context(context_data)
        
        # Comprimir datos principales
        compressed_data = self.compress_data(context_data)
        
        # Metadata liviana (sin comprimir para acceso rápido)
        metadata = {
            'data_size': len(json.dumps(context_data)),
            'compressed_size': len(compressed_data),
            'timestamp': time.time(),
            'keys': list(context_data.keys()) if isinstance(context_data, dict) else []
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO memory_contexts 
                (context_hash, compressed_data, metadata, last_accessed)
                VALUES (?, ?, ?, julianday('now'))
            ''', (context_hash, compressed_data, json.dumps(metadata)))
            
            conn.commit()
            return context_hash
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def retrieve_context(self, context_hash: str) -> Optional[Dict[str, Any]]:
        """Recupera contexto por hash"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT compressed_data, access_count 
            FROM memory_contexts 
            WHERE context_hash = ?
        ''', (context_hash,))
        
        result = cursor.fetchone()
        
        if result:
            compressed_data, access_count = result
            # Actualizar contador de acceso
            cursor.execute('''
                UPDATE memory_contexts 
                SET access_count = ?, last_accessed = julianday('now')
                WHERE context_hash = ?
            ''', (access_count + 1, context_hash))
            
            conn.commit()
            conn.close()
            
            return self.decompress_data(compressed_data)
        
        conn.close()
        return None
    
    def search_contexts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca contextos por contenido (búsqueda simple)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Búsqueda en metadata (más eficiente que descomprimir todo)
        cursor.execute('''
            SELECT context_hash, metadata, access_count, last_accessed
            FROM memory_contexts 
            WHERE metadata LIKE ? 
            ORDER BY last_accessed DESC 
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            context_hash, metadata_str, access_count, last_accessed = row
            metadata = json.loads(metadata_str)
            
            results.append({
                'context_hash': context_hash,
                'metadata': metadata,
                'access_count': access_count,
                'last_accessed': last_accessed
            })
        
        conn.close()
        return results
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Estadísticas del almacenamiento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_contexts,
                SUM(LENGTH(compressed_data)) as total_compressed_size,
                AVG(LENGTH(compressed_data)) as avg_compressed_size,
                MAX(access_count) as max_access_count
            FROM memory_contexts
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_contexts': stats[0],
            'total_compressed_size_bytes': stats[1],
            'average_context_size_bytes': stats[2],
            'most_accessed_context': stats[3]
        }
    
    def cleanup_old_contexts(self, max_age_days: int = 30, min_access_count: int = 1):
        """Limpia contextos antiguos o poco accedidos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM memory_contexts 
            WHERE 
                julianday('now') - last_accessed > ? 
                AND access_count < ?
        ''', (max_age_days, min_access_count))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count

class MemoryContextMCPServer:
    def __init__(self):
        self.memory_context = MemoryContextMCP()
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP"""
        method = request.get("method")
        params = request.get("params", {})
        
        try:
            if method == "initialize":
                return self.initialize(params)
            elif method == "tools/call":
                return self.handle_tool_call(params)
            else:
                return self.error_response(f"Unknown method: {method}")
        except Exception as e:
            return self.error_response(str(e))
    
    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inicialización MCP"""
        return {
            "jsonrpc": "2.0",
            "id": params.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    }
                },
                "serverInfo": {
                    "name": "memory-context-mcp",
                    "version": "1.0.0"
                },
                "tools": [
                    {
                        "name": "store_context",
                        "description": "Almacena contexto en memoria persistente",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "context_data": {"type": "object"}
                            },
                            "required": ["context_data"]
                        }
                    },
                    {
                        "name": "retrieve_context",
                        "description": "Recupera contexto por hash",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "context_hash": {"type": "string"}
                            },
                            "required": ["context_hash"]
                        }
                    },
                    {
                        "name": "search_contexts",
                        "description": "Busca contextos por query",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "default": 10}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "get_stats",
                        "description": "Obtiene estadísticas de memoria",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "cleanup_contexts",
                        "description": "Limpia contextos antiguos",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "max_age_days": {"type": "integer", "default": 30},
                                "min_access_count": {"type": "integer", "default": 1}
                            }
                        }
                    }
                ]
            }
        }
    
    def handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "store_context":
            result = self.memory_context.store_context(arguments["context_data"])
            return {
                "jsonrpc": "2.0",
                "id": params.get("id"),
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"Context stored with hash: {result}"
                    }]
                }
            }
        
        elif tool_name == "retrieve_context":
            context = self.memory_context.retrieve_context(arguments["context_hash"])
            if context:
                return {
                    "jsonrpc": "2.0",
                    "id": params.get("id"),
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": json.dumps(context, indent=2)
                        }]
                    }
                }
            else:
                return self.error_response("Context not found")
        
        elif tool_name == "search_contexts":
            results = self.memory_context.search_contexts(
                arguments.get("query", ""),
                arguments.get("limit", 10)
            )
            return {
                "jsonrpc": "2.0",
                "id": params.get("id"),
                "result": {
                    "content": [{
                        "type": "text", 
                        "text": json.dumps(results, indent=2)
                    }]
                }
            }
        
        elif tool_name == "get_stats":
            stats = self.memory_context.get_context_stats()
            return {
                "jsonrpc": "2.0",
                "id": params.get("id"),
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(stats, indent=2)
                    }]
                }
            }
        
        elif tool_name == "cleanup_contexts":
            deleted = self.memory_context.cleanup_old_contexts(
                arguments.get("max_age_days", 30),
                arguments.get("min_access_count", 1)
            )
            return {
                "jsonrpc": "2.0",
                "id": params.get("id"),
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"Deleted {deleted} old contexts"
                    }]
                }
            }
        
        else:
            return self.error_response(f"Unknown tool: {tool_name}")
    
    def error_response(self, message: str) -> Dict[str, Any]:
        """Respuesta de error estandarizada"""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": message
            }
        }

def main():
    """Loop principal del servidor MCP"""
    server = MemoryContextMCPServer()
    
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
