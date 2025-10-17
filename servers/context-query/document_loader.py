"""
Document Loader para Spec-Driven Development
Carga y procesa documentos markdown/PDF para 'entrenamiento' del sistema.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

class DocumentLoader:
    """Carga documentos para entrenamiento del sistema"""

    def __init__(self, docs_directory: str):
        self.docs_directory = Path(docs_directory)
        self.supported_extensions = ['.md', '.markdown', '.txt']  # Sin PDF por ahora (requiere dependencias)

    def load_all_documents(self) -> Dict[str, str]:
        """Carga todos los documentos soportados del directorio"""
        documents = {}

        if not self.docs_directory.exists():
            return documents

        for file_path in self.docs_directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content = self._read_file(file_path)
                    if content.strip():  # Solo archivos con contenido
                        documents[str(file_path)] = content
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")

        return documents

    def load_specific_files(self, file_paths: List[str]) -> Dict[str, str]:
        """Carga archivos específicos"""
        documents = {}

        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path.exists() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content = self._read_file(file_path)
                    documents[str(file_path)] = content
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")

        return documents

    def _read_file(self, file_path: Path) -> str:
        """Lee contenido de archivo"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def get_document_metadata(self, documents: Dict[str, str]) -> Dict[str, Dict]:
        """Obtiene metadata de documentos cargados"""
        metadata = {}

        for file_path, content in documents.items():
            path_obj = Path(file_path)
            lines = content.split('\n')

            metadata[file_path] = {
                'filename': path_obj.name,
                'extension': path_obj.suffix,
                'size': len(content),
                'lines': len(lines),
                'last_modified': os.path.getmtime(file_path),
                'has_headers': any(line.startswith('#') for line in lines[:10]),
                'estimated_reading_time': len(content.split()) / 200  # 200 palabras por minuto
            }

        return metadata

class TrainingManager:
    """Maneja el 'entrenamiento' del sistema con documentos"""

    def __init__(self, docs_directory: str):
        self.docs_directory = docs_directory
        self.document_loader = DocumentLoader(docs_directory)
        self.training_data = {}
        self.is_trained = False

    def train_system(self, force_retrain: bool = False) -> Dict[str, any]:
        """Entrena el sistema con documentos disponibles"""
        if self.is_trained and not force_retrain:
            return {
                'status': 'already_trained',
                'documents_loaded': len(self.training_data),
                'last_training': self.training_data.get('metadata', {}).get('training_timestamp')
            }

        # Cargar documentos
        documents = self.document_loader.load_all_documents()

        if not documents:
            return {'status': 'no_documents', 'error': 'No se encontraron documentos para entrenar'}

        # Obtener metadata
        metadata = self.document_loader.get_document_metadata(documents)

        # Almacenar datos de entrenamiento
        self.training_data = {
            'documents': documents,
            'metadata': metadata,
            'training_timestamp': json.dumps({'timestamp': __import__('time').time()}),
            'total_documents': len(documents),
            'total_content_size': sum(len(content) for content in documents.values())
        }

        self.is_trained = True

        return {
            'status': 'trained',
            'documents_loaded': len(documents),
            'total_size': self.training_data['total_content_size'],
            'training_timestamp': self.training_data['training_timestamp']
        }

    def get_training_status(self) -> Dict[str, any]:
        """Obtiene estado del entrenamiento"""
        if not self.is_trained:
            return {'status': 'not_trained'}

        return {
            'status': 'trained',
            'documents_count': len(self.training_data.get('documents', {})),
            'total_size': self.training_data.get('total_content_size', 0),
            'last_training': self.training_data.get('training_timestamp'),
            'metadata': self.training_data.get('metadata', {})
        }

    def get_document_content(self, filename: str) -> Optional[str]:
        """Obtiene contenido de un documento específico"""
        documents = self.training_data.get('documents', {})
        for file_path, content in documents.items():
            if Path(file_path).name == filename:
                return content
        return None

    def search_documents(self, query: str) -> List[Dict]:
        """Busca en documentos entrenados"""
        if not self.is_trained:
            return []

        results = []
        query_lower = query.lower()

        documents = self.training_data.get('documents', {})
        for file_path, content in documents.items():
            if query_lower in content.lower():
                # Encontrar contexto alrededor del match
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if query_lower in line.lower():
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        context = '\n'.join(lines[context_start:context_end])

                        results.append({
                            'filename': Path(file_path).name,
                            'line_number': i + 1,
                            'context': context,
                            'full_path': file_path
                        })
                        break  # Solo primera ocurrencia por archivo

        return results[:10]  # Limitar resultados
