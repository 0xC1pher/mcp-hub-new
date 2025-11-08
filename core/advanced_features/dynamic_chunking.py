"""
Dynamic Chunking Adaptativo - Sistema de chunking inteligente y adaptativo
Implementa chunking semántico basado en contenido con ajuste dinámico de tamaños
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ContentType(Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    chunk_id: str
    content_type: ContentType
    complexity_score: float
    semantic_coherence: float
    size: int
    line_start: int
    line_end: int
    keywords: List[str]
    structure_depth: int
    overlap_chars: int


@dataclass
class AdaptiveChunk:
    content: str
    metadata: ChunkMetadata
    vector_hash: Optional[str] = None
    context_window: Optional[str] = None


class DynamicChunkingSystem:
    """Sistema de chunking adaptativo que ajusta el tamaño y método según el contenido"""

    def __init__(self,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1000,
                 overlap_ratio: float = 0.15,
                 complexity_threshold: float = 0.7):

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        self.complexity_threshold = complexity_threshold

        # Patrones para detección de contenido
        self.code_patterns = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+\s*[:\(]',
            r'function\s+\w+\s*\(',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'^\s*#.*$',
            r'^\s*//.*$',
            r'{\s*$',
            r'}\s*$'
        ]

        self.markdown_patterns = [
            r'^#+\s+.*$',
            r'^\*\s+.*$',
            r'^\-\s+.*$',
            r'^\d+\.\s+.*$',
            r'\*\*.*\*\*',
            r'`.*`',
            r'```.*```'
        ]

    def adaptive_chunking(self,
                         text: str,
                         file_path: str = "",
                         metadata: Dict[str, Any] = None) -> List[AdaptiveChunk]:
        """
        Realiza chunking adaptativo basado en el contenido

        Args:
            text: Texto a dividir en chunks
            file_path: Ruta del archivo (opcional)
            metadata: Metadatos adicionales

        Returns:
            Lista de AdaptiveChunk con chunking optimizado
        """

        # 1. Detectar tipo de contenido
        content_type = self._detect_content_type(text, file_path)

        # 2. Analizar complejidad del contenido
        complexity_score = self._analyze_complexity(text, content_type)

        # 3. Determinar estrategia de chunking
        chunk_strategy = self._determine_chunking_strategy(content_type, complexity_score)

        # 4. Aplicar chunking según estrategia
        if chunk_strategy == "semantic":
            chunks = self._semantic_chunking(text, content_type)
        elif chunk_strategy == "structural":
            chunks = self._structural_chunking(text, content_type)
        else:  # sliding_window
            chunks = self._sliding_window_chunking(text)

        # 5. Aplicar overlapping inteligente
        chunks_with_overlap = self._apply_intelligent_overlap(chunks, text)

        # 6. Generar metadata enriquecida
        adaptive_chunks = []
        for i, chunk in enumerate(chunks_with_overlap):
            metadata_obj = self._generate_chunk_metadata(
                chunk, i, content_type, complexity_score, text
            )

            adaptive_chunk = AdaptiveChunk(
                content=chunk["content"],
                metadata=metadata_obj,
                context_window=chunk.get("context", "")
            )

            adaptive_chunks.append(adaptive_chunk)

        return adaptive_chunks

    def _detect_content_type(self, text: str, file_path: str = "") -> ContentType:
        """Detecta el tipo de contenido basándose en patrones y extensión"""

        # Primero intentar por extensión de archivo
        if file_path:
            extension = file_path.lower().split('.')[-1]
            extension_map = {
                'py': ContentType.CODE,
                'js': ContentType.CODE,
                'ts': ContentType.CODE,
                'java': ContentType.CODE,
                'cpp': ContentType.CODE,
                'c': ContentType.CODE,
                'md': ContentType.MARKDOWN,
                'json': ContentType.JSON,
                'xml': ContentType.XML,
                'yaml': ContentType.YAML,
                'yml': ContentType.YAML
            }

            if extension in extension_map:
                return extension_map[extension]

        # Detectar por patrones de contenido
        lines = text.split('\n')
        code_score = 0
        markdown_score = 0

        for line in lines[:50]:  # Analizar primeras 50 líneas
            # Buscar patrones de código
            for pattern in self.code_patterns:
                if re.search(pattern, line, re.MULTILINE):
                    code_score += 1

            # Buscar patrones de markdown
            for pattern in self.markdown_patterns:
                if re.search(pattern, line, re.MULTILINE):
                    markdown_score += 1

        # Detectar JSON/XML por estructura
        text_stripped = text.strip()
        if text_stripped.startswith('{') and text_stripped.endswith('}'):
            return ContentType.JSON
        elif text_stripped.startswith('<') and text_stripped.endswith('>'):
            return ContentType.XML

        # Determinar tipo basado en scores
        if code_score > markdown_score and code_score > 2:
            return ContentType.CODE
        elif markdown_score > 2:
            return ContentType.MARKDOWN
        else:
            return ContentType.TEXT

    def _analyze_complexity(self, text: str, content_type: ContentType) -> float:
        """Analiza la complejidad del contenido para ajustar chunking"""

        complexity_factors = {
            'line_length_variance': 0,
            'nesting_depth': 0,
            'symbol_density': 0,
            'structural_elements': 0
        }

        lines = text.split('\n')

        # 1. Varianza en longitud de líneas
        line_lengths = [len(line) for line in lines if line.strip()]
        if line_lengths:
            complexity_factors['line_length_variance'] = np.std(line_lengths) / np.mean(line_lengths)

        # 2. Profundidad de anidamiento (para código)
        if content_type == ContentType.CODE:
            max_depth = 0
            current_depth = 0
            for line in lines:
                stripped = line.strip()
                if stripped.endswith('{') or stripped.endswith(':'):
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif stripped.startswith('}') or (stripped and not stripped.startswith(' ')):
                    current_depth = max(0, current_depth - 1)

            complexity_factors['nesting_depth'] = min(max_depth / 10.0, 1.0)

        # 3. Densidad de símbolos especiales
        special_chars = len([c for c in text if c in '{}[]().,;:"\'`'])
        complexity_factors['symbol_density'] = min(special_chars / len(text), 1.0)

        # 4. Elementos estructurales
        if content_type == ContentType.MARKDOWN:
            headers = len(re.findall(r'^#+\s+', text, re.MULTILINE))
            lists = len(re.findall(r'^\s*[\-\*\+]\s+', text, re.MULTILINE))
            complexity_factors['structural_elements'] = min((headers + lists) / 20.0, 1.0)

        # Calcular score final
        weights = [0.25, 0.3, 0.25, 0.2]
        complexity_score = sum(w * v for w, v in zip(weights, complexity_factors.values()))

        return min(complexity_score, 1.0)

    def _determine_chunking_strategy(self, content_type: ContentType, complexity: float) -> str:
        """Determina la estrategia de chunking óptima"""

        if content_type == ContentType.CODE and complexity > self.complexity_threshold:
            return "structural"
        elif content_type == ContentType.MARKDOWN:
            return "semantic"
        elif complexity > self.complexity_threshold:
            return "semantic"
        else:
            return "sliding_window"

    def _semantic_chunking(self, text: str, content_type: ContentType) -> List[Dict[str, Any]]:
        """Chunking semántico basado en estructura lógica"""

        chunks = []

        if content_type == ContentType.MARKDOWN:
            # Dividir por headers
            sections = re.split(r'^(#+\s+.*$)', text, flags=re.MULTILINE)
            current_chunk = ""

            for section in sections:
                if re.match(r'^#+\s+', section):
                    if current_chunk and len(current_chunk) > self.min_chunk_size:
                        chunks.append({"content": current_chunk.strip()})
                        current_chunk = section + "\n"
                    else:
                        current_chunk += section + "\n"
                else:
                    current_chunk += section

                if len(current_chunk) > self.max_chunk_size:
                    chunks.append({"content": current_chunk.strip()})
                    current_chunk = ""

            if current_chunk:
                chunks.append({"content": current_chunk.strip()})

        else:
            # Chunking por párrafos para texto normal
            paragraphs = text.split('\n\n')
            current_chunk = ""

            for paragraph in paragraphs:
                if len(current_chunk + paragraph) > self.max_chunk_size:
                    if current_chunk:
                        chunks.append({"content": current_chunk.strip()})
                        current_chunk = paragraph
                    else:
                        # Párrafo muy largo, usar sliding window
                        chunks.extend(self._sliding_window_chunking(paragraph))
                        current_chunk = ""
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph

            if current_chunk:
                chunks.append({"content": current_chunk.strip()})

        return chunks

    def _structural_chunking(self, text: str, content_type: ContentType) -> List[Dict[str, Any]]:
        """Chunking estructural para código"""

        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        brace_count = 0
        current_function = ""

        for i, line in enumerate(lines):
            current_chunk += line + "\n"

            # Detectar inicio de función/clase
            if re.match(r'^\s*(def|class|function)\s+', line):
                current_function = line.strip()

            # Contar llaves para detectar bloques
            brace_count += line.count('{') - line.count('}')

            # Condiciones para crear chunk
            should_chunk = False

            if brace_count == 0 and current_function and len(current_chunk) > self.min_chunk_size:
                should_chunk = True
            elif len(current_chunk) > self.max_chunk_size:
                should_chunk = True

            if should_chunk:
                chunks.append({"content": current_chunk.strip()})
                current_chunk = ""
                current_function = ""

        if current_chunk.strip():
            chunks.append({"content": current_chunk.strip()})

        return chunks

    def _sliding_window_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Chunking por ventana deslizante tradicional"""

        chunks = []
        overlap_size = int(self.max_chunk_size * self.overlap_ratio)

        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))

            # Ajustar límites para no cortar palabras
            if end < len(text):
                # Buscar el último espacio o salto de línea
                last_break = max(
                    text.rfind(' ', start, end),
                    text.rfind('\n', start, end)
                )
                if last_break > start:
                    end = last_break

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({"content": chunk_content})

            start = end - overlap_size

        return chunks

    def _apply_intelligent_overlap(self, chunks: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """Aplica overlapping inteligente entre chunks"""

        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            overlap_chars = int(len(chunk["content"]) * self.overlap_ratio)

            # Contexto anterior
            prev_context = ""
            if i > 0:
                prev_content = chunks[i-1]["content"]
                prev_context = prev_content[-overlap_chars:] if len(prev_content) > overlap_chars else prev_content

            # Contexto posterior
            next_context = ""
            if i < len(chunks) - 1:
                next_content = chunks[i+1]["content"]
                next_context = next_content[:overlap_chars] if len(next_content) > overlap_chars else next_content

            # Combinar contexto
            context = f"{prev_context}\n---\n{next_context}".strip()

            overlapped_chunk = {
                "content": chunk["content"],
                "context": context,
                "overlap_chars": len(context)
            }

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _generate_chunk_metadata(self,
                                chunk: Dict[str, Any],
                                index: int,
                                content_type: ContentType,
                                complexity_score: float,
                                full_text: str) -> ChunkMetadata:
        """Genera metadata enriquecida para el chunk"""

        content = chunk["content"]

        # Generar ID único
        chunk_id = hashlib.md5(f"{index}_{content[:100]}".encode()).hexdigest()[:12]

        # Extraer keywords
        keywords = self._extract_keywords(content, content_type)

        # Calcular coherencia semántica
        coherence = self._calculate_semantic_coherence(content)

        # Determinar líneas
        lines_before = full_text[:full_text.find(content)].count('\n') if content in full_text else 0
        line_start = lines_before + 1
        line_end = line_start + content.count('\n')

        # Profundidad estructural
        structure_depth = self._calculate_structure_depth(content, content_type)

        return ChunkMetadata(
            chunk_id=chunk_id,
            content_type=content_type,
            complexity_score=complexity_score,
            semantic_coherence=coherence,
            size=len(content),
            line_start=line_start,
            line_end=line_end,
            keywords=keywords,
            structure_depth=structure_depth,
            overlap_chars=chunk.get("overlap_chars", 0)
        )

    def _extract_keywords(self, content: str, content_type: ContentType) -> List[str]:
        """Extrae keywords relevantes del contenido"""

        keywords = []

        if content_type == ContentType.CODE:
            # Extraer nombres de funciones, clases y variables
            functions = re.findall(r'def\s+(\w+)', content)
            classes = re.findall(r'class\s+(\w+)', content)
            keywords.extend(functions + classes)

        elif content_type == ContentType.MARKDOWN:
            # Extraer headers
            headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            keywords.extend([h.strip() for h in headers])

        # Palabras frecuentes (simple)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Top 5 palabras más frecuentes
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords.extend([word for word, _ in top_words])

        return list(set(keywords))[:10]  # Máximo 10 keywords únicos

    def _calculate_semantic_coherence(self, content: str) -> float:
        """Calcula un score de coherencia semántica simple"""

        # Métricas simples de coherencia
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0

        # Coherencia basada en repetición de palabras clave
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            all_words.extend(words)

        if not all_words:
            return 0.5

        # Calcular diversidad vs repetición
        unique_words = set(all_words)
        repetition_ratio = len(all_words) / len(unique_words) if unique_words else 1

        # Score entre 0 y 1 (más repetición = más coherencia hasta cierto punto)
        coherence = min(repetition_ratio / 3.0, 1.0)

        return coherence

    def _calculate_structure_depth(self, content: str, content_type: ContentType) -> int:
        """Calcula la profundidad estructural del contenido"""

        if content_type == ContentType.MARKDOWN:
            headers = re.findall(r'^(#+)', content, re.MULTILINE)
            return max([len(h) for h in headers], default=0)

        elif content_type == ContentType.CODE:
            max_indent = 0
            for line in content.split('\n'):
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent // 4)  # Asumiendo indentación de 4 espacios

            return max_indent

        return 0

    def get_chunking_stats(self, chunks: List[AdaptiveChunk]) -> Dict[str, Any]:
        """Obtiene estadísticas del chunking realizado"""

        if not chunks:
            return {}

        sizes = [chunk.metadata.size for chunk in chunks]
        complexities = [chunk.metadata.complexity_score for chunk in chunks]
        coherences = [chunk.metadata.semantic_coherence for chunk in chunks]

        content_types = {}
        for chunk in chunks:
            ct = chunk.metadata.content_type.value
            content_types[ct] = content_types.get(ct, 0) + 1

        return {
            "total_chunks": len(chunks),
            "avg_size": np.mean(sizes),
            "size_std": np.std(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_complexity": np.mean(complexities),
            "avg_coherence": np.mean(coherences),
            "content_type_distribution": content_types,
            "total_keywords": sum(len(c.metadata.keywords) for c in chunks)
        }


# Función de conveniencia
def adaptive_chunking(text: str,
                     file_path: str = "",
                     min_chunk_size: int = 200,
                     max_chunk_size: int = 1000,
                     overlap_ratio: float = 0.15) -> List[AdaptiveChunk]:
    """
    Función de conveniencia para chunking adaptativo

    Args:
        text: Texto a dividir
        file_path: Ruta del archivo (opcional)
        min_chunk_size: Tamaño mínimo de chunk
        max_chunk_size: Tamaño máximo de chunk
        overlap_ratio: Ratio de overlapping

    Returns:
        Lista de chunks adaptativos
    """

    system = DynamicChunkingSystem(
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap_ratio=overlap_ratio
    )

    return system.adaptive_chunking(text, file_path)


if __name__ == "__main__":
    # Ejemplo de uso
    sample_text = """
    # Título Principal

    Este es un ejemplo de texto en markdown.

    ## Subsección

    - Item 1
    - Item 2
    - Item 3

    ### Código de ejemplo

    ```python
    def example_function():
        return "Hello World"
    ```

    ## Otra sección

    Más contenido aquí con información relevante.
    """

    chunker = DynamicChunkingSystem()
    chunks = chunker.adaptive_chunking(sample_text, "example.md")

    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.metadata.content_type.value}")
        print(f"  Size: {chunk.metadata.size}")
        print(f"  Complexity: {chunk.metadata.complexity_score:.3f}")
        print(f"  Coherence: {chunk.metadata.semantic_coherence:.3f}")
        print(f"  Keywords: {chunk.metadata.keywords}")
        print(f"  Content preview: {chunk.content[:100]}...")

    # Estadísticas
    stats = chunker.get_chunking_stats(chunks)
    print(f"\nChunking Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
