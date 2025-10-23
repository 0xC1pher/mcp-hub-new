#!/usr/bin/env python3
"""
🧩 Sistema de Chunking Inteligente
Chunking semántico por párrafos/funciones
Overlapping de 50-100 caracteres entre chunks
Metadata enriquecida por chunk
Indexación vectorial para búsqueda semántica
"""
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import json
import time

class SemanticChunk:
    """Representación de un chunk semántico"""
    
    def __init__(self, content: str, chunk_type: str, start_pos: int, end_pos: int, 
                 source_file: str = "", metadata: Dict[str, Any] = None):
        self.content = content
        self.chunk_type = chunk_type  # 'paragraph', 'function', 'class', 'code_block'
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.source_file = source_file
        self.metadata = metadata or {}
        self.chunk_id = self._generate_id()
        self.created_at = time.time()
        
    def _generate_id(self) -> str:
        """Genera ID único para el chunk"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.chunk_type}_{content_hash}_{self.start_pos}"
    
    def get_size_bytes(self) -> int:
        """Tamaño del chunk en bytes"""
        return len(self.content.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte chunk a diccionario"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'source_file': self.source_file,
            'metadata': self.metadata,
            'size_bytes': self.get_size_bytes(),
            'created_at': self.created_at
        }

class IntelligentSemanticChunker:
    """Chunker semántico inteligente mejorado"""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 75, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        # Patrones para diferentes tipos de contenido
        self.patterns = {
            'function_def': re.compile(r'^(def|function|async def)\s+\w+.*?:', re.MULTILINE),
            'class_def': re.compile(r'^class\s+\w+.*?:', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'paragraph': re.compile(r'\n\s*\n', re.MULTILINE),
            'heading': re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),
            'list_item': re.compile(r'^[\s]*[-*+]\s+.*$', re.MULTILINE),
            'numbered_list': re.compile(r'^[\s]*\d+\.\s+.*$', re.MULTILINE)
        }
    
    def chunk_text(self, text: str, source_file: str = "") -> List[SemanticChunk]:
        """Chunking principal del texto"""
        chunks = []
        
        # Detectar tipo de contenido
        content_type = self._detect_content_type(text)
        
        if content_type == 'code':
            chunks = self._chunk_code(text, source_file)
        elif content_type == 'markdown':
            chunks = self._chunk_markdown(text, source_file)
        else:
            chunks = self._chunk_generic_text(text, source_file)
        
        # Aplicar overlapping entre chunks
        chunks = self._apply_overlapping(chunks, text)
        
        # Enriquecer metadata
        chunks = self._enrich_metadata(chunks)
        
        return chunks
    
    def _detect_content_type(self, text: str) -> str:
        """Detecta el tipo de contenido"""
        # Contar patrones característicos
        code_indicators = len(self.patterns['function_def'].findall(text)) + \
                         len(self.patterns['class_def'].findall(text))
        
        markdown_indicators = len(self.patterns['heading'].findall(text)) + \
                            len(self.patterns['code_block'].findall(text))
        
        if code_indicators > 2:
            return 'code'
        elif markdown_indicators > 1:
            return 'markdown'
        else:
            return 'text'
    
    def _chunk_code(self, text: str, source_file: str) -> List[SemanticChunk]:
        """Chunking específico para código"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_type = 'code_block'
        start_line = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Detectar definiciones de función/clase
            if self.patterns['function_def'].match(line) or self.patterns['class_def'].match(line):
                # Guardar chunk anterior si existe
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    if len(chunk_content.strip()) >= self.min_chunk_size:
                        chunks.append(SemanticChunk(
                            content=chunk_content,
                            chunk_type=current_type,
                            start_pos=start_line,
                            end_pos=i-1,
                            source_file=source_file
                        ))
                
                # Extraer función/clase completa
                func_lines, end_line = self._extract_function_or_class(lines, i)
                chunk_content = '\n'.join(func_lines)
                
                chunk_type = 'function' if 'def ' in line else 'class'
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    chunk_type=chunk_type,
                    start_pos=i,
                    end_pos=end_line,
                    source_file=source_file,
                    metadata={'name': self._extract_name_from_definition(line)}
                ))
                
                i = end_line + 1
                current_chunk = []
                start_line = i
            else:
                current_chunk.append(line)
                
                # Si el chunk es muy grande, dividirlo
                if len('\n'.join(current_chunk)) > self.chunk_size:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(SemanticChunk(
                        content=chunk_content,
                        chunk_type='code_block',
                        start_pos=start_line,
                        end_pos=i,
                        source_file=source_file
                    ))
                    current_chunk = []
                    start_line = i + 1
                
                i += 1
        
        # Chunk final
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content.strip()) >= self.min_chunk_size:
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    chunk_type='code_block',
                    start_pos=start_line,
                    end_pos=len(lines)-1,
                    source_file=source_file
                ))
        
        return chunks
    
    def _chunk_markdown(self, text: str, source_file: str) -> List[SemanticChunk]:
        """Chunking específico para Markdown"""
        chunks = []
        
        # Dividir por headings principales
        sections = re.split(r'^(#{1,3}\s+.*?)$', text, flags=re.MULTILINE)
        
        current_content = ""
        current_heading = ""
        start_pos = 0
        
        for i, section in enumerate(sections):
            if re.match(r'^#{1,3}\s+', section):
                # Es un heading
                if current_content.strip():
                    # Guardar sección anterior
                    chunks.extend(self._chunk_section(
                        current_content, 
                        current_heading, 
                        start_pos, 
                        source_file
                    ))
                
                current_heading = section.strip()
                current_content = section + "\n"
                start_pos = text.find(section, start_pos)
            else:
                current_content += section
        
        # Última sección
        if current_content.strip():
            chunks.extend(self._chunk_section(
                current_content, 
                current_heading, 
                start_pos, 
                source_file
            ))
        
        return chunks
    
    def _chunk_section(self, content: str, heading: str, start_pos: int, source_file: str) -> List[SemanticChunk]:
        """Chunking de una sección de markdown"""
        chunks = []
        
        # Dividir por párrafos
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = ""
        chunk_start = start_pos
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(SemanticChunk(
                        content=current_chunk.strip(),
                        chunk_type='paragraph',
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(current_chunk),
                        source_file=source_file,
                        metadata={'heading': heading}
                    ))
                
                current_chunk = paragraph + "\n\n"
                chunk_start = start_pos + content.find(paragraph)
            else:
                current_chunk += paragraph + "\n\n"
        
        # Chunk final
        if current_chunk.strip():
            chunks.append(SemanticChunk(
                content=current_chunk.strip(),
                chunk_type='paragraph',
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk),
                source_file=source_file,
                metadata={'heading': heading}
            ))
        
        return chunks
    
    def _chunk_generic_text(self, text: str, source_file: str) -> List[SemanticChunk]:
        """Chunking genérico para texto plano"""
        chunks = []
        
        # Dividir por párrafos
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        start_pos = 0
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(SemanticChunk(
                        content=current_chunk.strip(),
                        chunk_type='paragraph',
                        start_pos=start_pos,
                        end_pos=start_pos + len(current_chunk),
                        source_file=source_file
                    ))
                
                current_chunk = paragraph + "\n\n"
                start_pos = text.find(paragraph, start_pos)
            else:
                current_chunk += paragraph + "\n\n"
        
        # Chunk final
        if current_chunk.strip():
            chunks.append(SemanticChunk(
                content=current_chunk.strip(),
                chunk_type='paragraph',
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk),
                source_file=source_file
            ))
        
        return chunks
    
    def _extract_function_or_class(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """Extrae función o clase completa"""
        func_lines = [lines[start_idx]]
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            
            # Línea vacía o comentario
            if not line.strip() or line.strip().startswith('#'):
                func_lines.append(line)
                i += 1
                continue
            
            # Calcular indentación
            line_indent = len(line) - len(line.lstrip())
            
            # Si la indentación es menor o igual, terminó la función/clase
            if line_indent <= indent_level and line.strip():
                break
            
            func_lines.append(line)
            i += 1
        
        return func_lines, i - 1
    
    def _extract_name_from_definition(self, line: str) -> str:
        """Extrae nombre de función o clase"""
        if 'def ' in line:
            match = re.search(r'def\s+(\w+)', line)
            return match.group(1) if match else 'unknown_function'
        elif 'class ' in line:
            match = re.search(r'class\s+(\w+)', line)
            return match.group(1) if match else 'unknown_class'
        return 'unknown'
    
    def _apply_overlapping(self, chunks: List[SemanticChunk], original_text: str) -> List[SemanticChunk]:
        """Aplica overlapping entre chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.content
            
            # Agregar overlap del chunk anterior
            if i > 0:
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk.content[-self.overlap:] if len(prev_chunk.content) > self.overlap else prev_chunk.content
                content = overlap_text + "\n" + content
            
            # Agregar overlap del chunk siguiente
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                overlap_text = next_chunk.content[:self.overlap] if len(next_chunk.content) > self.overlap else next_chunk.content
                content = content + "\n" + overlap_text
            
            # Crear nuevo chunk con overlapping
            overlapped_chunk = SemanticChunk(
                content=content,
                chunk_type=chunk.chunk_type,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                source_file=chunk.source_file,
                metadata=chunk.metadata.copy()
            )
            
            # Agregar metadata de overlapping
            overlapped_chunk.metadata['has_overlap'] = True
            overlapped_chunk.metadata['original_size'] = len(chunk.content)
            overlapped_chunk.metadata['overlapped_size'] = len(content)
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _enrich_metadata(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Enriquece metadata de los chunks"""
        for i, chunk in enumerate(chunks):
            # Estadísticas básicas
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'word_count': len(chunk.content.split()),
                'line_count': len(chunk.content.split('\n')),
                'char_count': len(chunk.content)
            })
            
            # Análisis de contenido
            if chunk.chunk_type == 'function':
                chunk.metadata.update(self._analyze_function(chunk.content))
            elif chunk.chunk_type == 'class':
                chunk.metadata.update(self._analyze_class(chunk.content))
            elif chunk.chunk_type == 'paragraph':
                chunk.metadata.update(self._analyze_paragraph(chunk.content))
            
            # Keywords extraction (simple)
            chunk.metadata['keywords'] = self._extract_keywords(chunk.content)
        
        return chunks
    
    def _analyze_function(self, content: str) -> Dict[str, Any]:
        """Analiza contenido de función"""
        return {
            'has_docstring': '"""' in content or "'''" in content,
            'has_return': 'return ' in content,
            'has_parameters': '(' in content.split('\n')[0] and ')' in content.split('\n')[0],
            'complexity_estimate': content.count('if ') + content.count('for ') + content.count('while ')
        }
    
    def _analyze_class(self, content: str) -> Dict[str, Any]:
        """Analiza contenido de clase"""
        return {
            'method_count': content.count('def '),
            'has_init': '__init__' in content,
            'has_docstring': '"""' in content or "'''" in content,
            'inheritance': '(' in content.split('\n')[0]
        }
    
    def _analyze_paragraph(self, content: str) -> Dict[str, Any]:
        """Analiza contenido de párrafo"""
        return {
            'sentence_count': content.count('.') + content.count('!') + content.count('?'),
            'has_code_snippets': '`' in content,
            'has_links': '[' in content and '](' in content,
            'has_lists': content.count('- ') + content.count('* ') + content.count('+ ')
        }
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extrae keywords simples del contenido"""
        # Palabras técnicas comunes
        technical_words = set()
        
        # Extraer palabras en mayúsculas o CamelCase
        camel_case_words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content)
        technical_words.update(camel_case_words)
        
        # Extraer palabras técnicas específicas
        tech_patterns = [
            r'\b\w*[Mm]anager\b', r'\b\w*[Ss]ystem\b', r'\b\w*[Cc]ache\b',
            r'\b\w*[Oo]ptimizer\b', r'\b\w*[Pp]rocessor\b', r'\b\w*[Hh]andler\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, content)
            technical_words.update(matches)
        
        return list(technical_words)[:10]  # Limitar a 10 keywords

class ChunkIndexer:
    """Indexador para búsqueda vectorial de chunks"""
    
    def __init__(self):
        self.chunk_index = {}
        self.keyword_index = {}
    
    def index_chunks(self, chunks: List[SemanticChunk]) -> None:
        """Indexa chunks para búsqueda rápida"""
        for chunk in chunks:
            # Índice por ID
            self.chunk_index[chunk.chunk_id] = chunk
            
            # Índice por keywords
            for keyword in chunk.metadata.get('keywords', []):
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(chunk.chunk_id)
    
    def search_by_keyword(self, keyword: str) -> List[SemanticChunk]:
        """Busca chunks por keyword"""
        chunk_ids = self.keyword_index.get(keyword, [])
        return [self.chunk_index[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunk_index]
    
    def search_by_content(self, query: str) -> List[SemanticChunk]:
        """Busca chunks por contenido (búsqueda simple)"""
        results = []
        query_lower = query.lower()
        
        for chunk in self.chunk_index.values():
            if query_lower in chunk.content.lower():
                results.append(chunk)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del indexador"""
        return {
            'total_chunks': len(self.chunk_index),
            'total_keywords': len(self.keyword_index),
            'avg_keywords_per_chunk': len(self.keyword_index) / len(self.chunk_index) if self.chunk_index else 0
        }

# Instancia global del chunker
semantic_chunker = IntelligentSemanticChunker()
chunk_indexer = ChunkIndexer()

def get_chunker_instance() -> IntelligentSemanticChunker:
    """Obtiene instancia global del chunker"""
    return semantic_chunker

def get_indexer_instance() -> ChunkIndexer:
    """Obtiene instancia global del indexer"""
    return chunk_indexer
