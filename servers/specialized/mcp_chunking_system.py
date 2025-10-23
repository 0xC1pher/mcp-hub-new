#!/usr/bin/env python3
"""
MCP Chunking System - Servidor especializado en chunking semántico optimizado
Implementa chunking inteligente por tipo de contenido con preservación de contexto
"""

import json
import sys
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mcp-chunking-system')

class SemanticChunker:
    """Chunking semántico inteligente optimizado"""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_cache = {}
        self.content_hashes = set()
        self.lock = threading.RLock()
        
        logger.info(f"✅ Semantic Chunker inicializado - Size:{chunk_size}, Overlap:{overlap}")
    
    def chunk_content(self, content: str, source_path: str = "") -> List[Dict]:
        """Divide contenido en chunks semánticos optimizados"""
        with self.lock:
            if not content or len(content.strip()) < 20:
                return []
            
            # Generar hash para deduplicación
            content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
            
            if content_hash in self.chunk_cache:
                logger.debug(f"🎯 Chunk cache hit: {content_hash}")
                return self.chunk_cache[content_hash]
            
            # Chunking inteligente por tipo
            chunks = self._intelligent_chunking(content, content_hash, source_path)
            
            # Cachear resultado
            if chunks:
                self.chunk_cache[content_hash] = chunks
                self.content_hashes.add(content_hash)
            
            logger.debug(f"📄 Creados {len(chunks)} chunks para {source_path}")
            return chunks
    
    def _intelligent_chunking(self, content: str, content_hash: str, source_path: str) -> List[Dict]:
        """Chunking inteligente basado en tipo de contenido"""
        if self._is_code_file(content, source_path):
            return self._chunk_code_optimized(content, content_hash)
        elif self._is_markdown(content):
            return self._chunk_markdown_optimized(content, content_hash)
        elif self._is_json_config(content, source_path):
            return self._chunk_json_optimized(content, content_hash)
        else:
            return self._chunk_text_optimized(content, content_hash)
    
    def _is_code_file(self, content: str, source_path: str) -> bool:
        """Detecta archivos de código"""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        if source_path and Path(source_path).suffix in code_extensions:
            return True
        
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'public ', 'private ']
        return sum(1 for indicator in code_indicators if indicator in content) >= 2
    
    def _is_markdown(self, content: str) -> bool:
        """Detecta markdown"""
        return content.count('#') > 1 or '```' in content or content.count('*') > 5
    
    def _is_json_config(self, content: str, source_path: str) -> bool:
        """Detecta archivos JSON/config"""
        if source_path and Path(source_path).suffix in {'.json', '.yaml', '.yml', '.toml'}:
            return True
        try:
            json.loads(content)
            return True
        except:
            return False
    
    def _chunk_code_optimized(self, content: str, base_hash: str) -> List[Dict]:
        """Chunking optimizado para código"""
        chunks = []
        lines = content.splitlines()
        
        i = 0
        chunk_id = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Detectar funciones/clases
            if re.match(r'^\s*(def|class|function|public|private)\s+(\w+)', line):
                function_lines, consumed = self._extract_complete_function(lines, i)
                
                if len('\n'.join(function_lines)) > 100:
                    chunk = self._create_optimized_chunk(
                        function_lines, f"{base_hash}_{chunk_id}", 'function', 
                        {'type': 'function', 'start_line': i + 1}
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                
                i += consumed
                continue
            
            # Agrupar imports/configuración
            if line.startswith(('import ', 'from ', 'const ', 'var ', 'let ')):
                related_lines, consumed = self._extract_related_lines(lines, i)
                
                if len('\n'.join(related_lines)) > 50:
                    chunk = self._create_optimized_chunk(
                        related_lines, f"{base_hash}_{chunk_id}", 'imports',
                        {'type': 'imports', 'start_line': i + 1}
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                
                i += consumed
                continue
            
            i += 1
        
        return chunks
    
    def _chunk_markdown_optimized(self, content: str, base_hash: str) -> List[Dict]:
        """Chunking optimizado para markdown"""
        chunks = []
        
        # Dividir por headers
        sections = re.split(r'\n(#{1,3}\s+[^\n]+)', content)
        
        chunk_id = 0
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i]
                content_part = sections[i + 1] if i + 1 < len(sections) else ""
                
                section_content = header + "\n" + content_part
                
                if len(section_content.strip()) > 100:
                    chunk = self._create_optimized_chunk(
                        [section_content], f"{base_hash}_{chunk_id}", 'markdown_section',
                        {'type': 'markdown_section', 'header': header.strip()}
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
        
        return chunks
    
    def _chunk_json_optimized(self, content: str, base_hash: str) -> List[Dict]:
        """Chunking optimizado para JSON/config"""
        try:
            data = json.loads(content)
            chunks = []
            chunk_id = 0
            
            # Si es un objeto, dividir por claves principales
            if isinstance(data, dict):
                for key, value in data.items():
                    chunk_content = json.dumps({key: value}, indent=2, ensure_ascii=False)
                    
                    if len(chunk_content) > 50:
                        chunk = self._create_optimized_chunk(
                            [chunk_content], f"{base_hash}_{chunk_id}", 'json_section',
                            {'type': 'json_section', 'key': key}
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_id += 1
            
            return chunks
            
        except json.JSONDecodeError:
            # Fallback a chunking de texto
            return self._chunk_text_optimized(content, base_hash)
    
    def _chunk_text_optimized(self, content: str, base_hash: str) -> List[Dict]:
        """Chunking optimizado para texto plano"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk_paras = []
        current_size = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            if current_size + len(paragraph) > self.chunk_size and current_chunk_paras:
                chunk_content = '\n\n'.join(current_chunk_paras)
                chunk = self._create_optimized_chunk(
                    [chunk_content], f"{base_hash}_{chunk_id}", 'text',
                    {'type': 'text', 'paragraph_count': len(current_chunk_paras)}
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
                
                current_chunk_paras = [paragraph]
                current_size = len(paragraph)
            else:
                current_chunk_paras.append(paragraph)
                current_size += len(paragraph)
        
        # Último chunk
        if current_chunk_paras:
            chunk_content = '\n\n'.join(current_chunk_paras)
            chunk = self._create_optimized_chunk(
                [chunk_content], f"{base_hash}_{chunk_id}", 'text',
                {'type': 'text', 'paragraph_count': len(current_chunk_paras)}
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _extract_complete_function(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """Extrae función completa"""
        function_lines = [lines[start_idx]]
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            
            if not line.strip() or line.strip().startswith('#'):
                function_lines.append(line)
                i += 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            if current_indent <= base_indent and line.strip():
                break
            
            function_lines.append(line)
            i += 1
        
        return function_lines, i - start_idx
    
    def _extract_related_lines(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """Extrae líneas relacionadas"""
        related_lines = []
        i = start_idx
        first_line = lines[i].strip()
        
        if first_line.startswith(('import ', 'from ')):
            # Agrupar imports
            while i < len(lines) and (
                lines[i].strip().startswith(('import ', 'from ')) or 
                not lines[i].strip()
            ):
                related_lines.append(lines[i])
                i += 1
        else:
            # Agrupar hasta cambio significativo
            while i < len(lines) and len(related_lines) < 10:
                line = lines[i]
                
                if re.match(r'^\s*(def|class)\s+', line):
                    break
                
                related_lines.append(line)
                i += 1
        
        return related_lines, i - start_idx
    
    def _create_optimized_chunk(self, lines: List[str], chunk_id: str, chunk_type: str, metadata: Dict = None) -> Optional[Dict]:
        """Crea chunk optimizado con deduplicación"""
        content = '\n'.join(lines) if isinstance(lines, list) else lines[0]
        content = content.strip()
        
        if len(content) < 50:
            return None
        
        # Hash para deduplicación
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        if content_hash in self.content_hashes:
            return None
        
        self.content_hashes.add(content_hash)
        
        # Métricas de calidad
        lines_count = content.count('\n') + 1
        words_count = len(content.split())
        
        return {
            'id': chunk_id,
            'content': content,
            'type': chunk_type,
            'size': len(content),
            'lines': lines_count,
            'words': words_count,
            'hash': content_hash,
            'quality_score': min(1.0, (words_count / 50) * 0.7 + (lines_count / 10) * 0.3),
            'metadata': metadata or {}
        }
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de chunking"""
        with self.lock:
            return {
                'cached_contents': len(self.chunk_cache),
                'unique_hashes': len(self.content_hashes),
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'deduplication_rate': len(self.content_hashes) / max(1, len(self.chunk_cache)) * 100
            }


class MCPChunkingServer:
    """Servidor MCP especializado en chunking semántico"""
    
    def __init__(self):
        self.chunker = SemanticChunker()
        logger.info("🚀 MCP Chunking Server iniciado")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP"""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        try:
            if method == 'initialize':
                result = self._handle_initialize(params)
            elif method == 'tools/list':
                result = self._handle_tools_list(params)
            elif method == 'tools/call':
                result = self._handle_tools_call(params)
            else:
                result = {'error': f'Método no soportado: {method}'}
            
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error manejando request: {e}")
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'error': {
                    'code': -32603,
                    'message': str(e)
                }
            }
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja inicialización"""
        return {
            'protocolVersion': '2024-11-05',
            'capabilities': {
                'tools': {
                    'listChanged': True
                }
            },
            'serverInfo': {
                'name': 'mcp-chunking-system',
                'version': '1.0.0'
            }
        }
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista herramientas disponibles"""
        return {
            'tools': [
                {
                    'name': 'chunk_content',
                    'description': 'Divide contenido en chunks semánticos optimizados',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'content': {'type': 'string', 'description': 'Contenido a dividir'},
                            'source_path': {'type': 'string', 'description': 'Ruta del archivo fuente'},
                            'chunk_size': {'type': 'integer', 'description': 'Tamaño de chunk', 'default': 600},
                            'overlap': {'type': 'integer', 'description': 'Solapamiento', 'default': 50}
                        },
                        'required': ['content']
                    }
                },
                {
                    'name': 'chunking_stats',
                    'description': 'Obtiene estadísticas de chunking',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            ]
        }
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        try:
            if tool_name == 'chunk_content':
                return self._chunk_content(arguments)
            elif tool_name == 'chunking_stats':
                return self._chunking_stats(arguments)
            else:
                return {
                    'content': [{'type': 'text', 'text': f'Herramienta desconocida: {tool_name}'}],
                    'isError': True
                }
                
        except Exception as e:
            logger.error(f"Error ejecutando herramienta {tool_name}: {e}")
            return {
                'content': [{'type': 'text', 'text': f'Error ejecutando {tool_name}: {str(e)}'}],
                'isError': True
            }
    
    def _chunk_content(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Divide contenido en chunks"""
        content = args.get('content', '')
        source_path = args.get('source_path', '')
        chunk_size = args.get('chunk_size', 600)
        overlap = args.get('overlap', 50)
        
        # Actualizar configuración si es necesaria
        if chunk_size != self.chunker.chunk_size or overlap != self.chunker.overlap:
            self.chunker.chunk_size = chunk_size
            self.chunker.overlap = overlap
        
        chunks = self.chunker.chunk_content(content, source_path)
        
        if chunks:
            response = f'📄 **Chunking Completado** para {source_path or "contenido"}\n\n'
            response += f'✅ **Creados {len(chunks)} chunks** (tamaño: {chunk_size}, overlap: {overlap})\n\n'
            
            for i, chunk in enumerate(chunks, 1):
                response += f'**Chunk {i}** ({chunk["type"]}, {chunk["size"]} chars, score: {chunk["quality_score"]:.2f})\n'
                preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                response += f'```\n{preview}\n```\n\n'
            
            return {
                'content': [{'type': 'text', 'text': response}]
            }
        else:
            return {
                'content': [{'type': 'text', 'text': f'⚠️ No se pudieron crear chunks para el contenido proporcionado'}]
            }
    
    def _chunking_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        stats = self.chunker.get_chunking_stats()
        
        response = f'''📊 **Estadísticas de Chunking Semántico**

💾 **Cache**:
- Contenidos cacheados: {stats["cached_contents"]}
- Hashes únicos: {stats["unique_hashes"]}
- Tasa de deduplicación: {stats["deduplication_rate"]:.1f}%

⚙️ **Configuración**:
- Tamaño de chunk: {stats["chunk_size"]} caracteres
- Solapamiento: {stats["overlap"]} caracteres

🎯 **Estado**: {'🟢 Óptimo' if stats["deduplication_rate"] > 80 else '🟡 Bueno' if stats["deduplication_rate"] > 60 else '🔴 Necesita optimización'}
'''
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }
    
    def run(self):
        """Ejecuta el servidor MCP"""
        logger.info("🚀 Iniciando MCP Chunking Server...")
        
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
    server = MCPChunkingServer()
    server.run()
