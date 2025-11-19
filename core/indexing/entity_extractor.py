"""
Entity Extractor - Builds searchable entity index from code
Creates index of functions, classes, and their relationships
"""

from typing import Dict, List, Set, Any
from pathlib import Path
import logging

from .ast_parser import PythonASTParser

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts and indexes code entities (functions, classes)
    Builds dependencies and relationships between entities
    """
    
    def __init__(self):
        """Initialize entity extractor"""
        self.parser = PythonASTParser()
        self.entities = {
            'functions': {},  # name -> metadata
            'classes': {},    # name -> metadata
            'modules': {}     # file_path -> module info
        }
        self.dependencies = {}  # entity_name -> [dependencies]
    
    def index_file(self, file_path: str) -> bool:
        """
        Index a single Python file
        
        Args:
            file_path: Path to file
            
        Returns:
            True if successful
        """
        parsed = self.parser.parse_file(file_path)
        
        if not parsed:
            return False
        
        module_name = Path(file_path).stem
        
        # Store module info
        self.entities['modules'][file_path] = {
            'name': module_name,
            'path': file_path,
            'docstring': parsed.get('module_docstring'),
            'imports': parsed.get('imports', []),
            'function_count': len(parsed.get('functions', [])),
            'class_count': len(parsed.get('classes', []))
        }
        
        # Index functions
        for func in parsed.get('functions', []):
            func_full_name = f"{module_name}.{func['name']}"
            
            self.entities['functions'][func_full_name] = {
                'name': func['name'],
                'module': module_name,
                'file_path': file_path,
                'signature': func['signature'],
                'line_range': [func['line_start'], func['line_end']],
                'docstring': func['docstring'],
                'args': [arg['name'] for arg in func['args']],
                'return_type': func.get('return_type'),
                'is_async': func.get('is_async', False),
                'decorators': func.get('decorators', [])
            }
            
            # Extract  dependencies (function calls)
            calls = self.parser.get_function_calls(file_path, func['name'])
            if calls:
                self.dependencies[func_full_name] = calls
        
        # Index classes
        for cls in parsed.get('classes', []):
            cls_full_name = f"{module_name}.{cls['name']}"
            
            self.entities['classes'][cls_full_name] = {
                'name': cls['name'],
                'module': module_name,
                'file_path': file_path,
                'line_range': [cls['line_start'], cls['line_end']],
                'docstring': cls['docstring'],
                'bases': cls.get('bases', []),
                'methods': [m['name'] for m in cls.get('methods', [])],
                'method_details': cls.get('methods', []),
                'attributes': cls.get('attributes', []),
                'decorators': cls.get('decorators', [])
            }
            
            # Index class methods as separate entities
            for method in cls.get('methods', []):
                method_full_name = f"{cls_full_name}.{method['name']}"
                
                self.entities['functions'][method_full_name] = {
                    'name': method['name'],
                    'module': module_name,
                    'class': cls['name'],
                    'file_path': file_path,
                    'signature': method['signature'],
                    'line_range': [method['line_start'], method['line_end']],
                    'docstring': method['docstring'],
                    'args': [arg['name'] for arg in method['args']],
                    'return_type': method.get('return_type'),
                    'is_async': method.get('is_async', False),
                    'decorators': method.get('decorators', [])
                }
        
        logger.info(f"Indexed {file_path}: {len(parsed['functions'])} functions, {len(parsed['classes'])} classes")
        return True
    
    def index_directory(self, directory: str, extensions: List[str] = None) -> int:
        """
        Index all Python files in a directory
        
        Args:
            directory: Directory to index
            extensions: File extensions to index (default: ['.py'])
            
        Returns:
            Number of files indexed
        """
        if extensions is None:
            extensions = ['.py']
        
        directory_path = Path(directory)
        indexed_count = 0
        
        for ext in extensions:
            for file_path in directory_path.rglob(f'*{ext}'):
                if self.index_file(str(file_path)):
                    indexed_count += 1
        
        logger.info(f"Indexed {indexed_count} files in {directory}")
        return indexed_count
    
    def find_entity(self, query: str) -> List[Dict[str, Any]]:
        """
        Find entities matching a query
        
        Args:
            query: Search query (function/class name)
            
        Returns:
            List of matching entities
        """
        results = []
        query_lower = query.lower()
        
        # Search functions
        for name, entity in self.entities['functions'].items():
            if query_lower in name.lower() or query_lower in entity['name'].lower():
                results.append({
                    'type': 'function',
                    'full_name': name,
                    **entity
                })
        
        # Search classes
        for name, entity in self.entities['classes'].items():
            if query_lower in name.lower() or query_lower in entity['name'].lower():
                results.append({
                    'type': 'class',
                    'full_name': name,
                    **entity
                })
        
        return results
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info about a specific entity
        
        Args:
            entity_name: Full entity name (module.function or module.Class)
            
        Returns:
            Entity info or None
        """
        # Try functions first
        if entity_name in self.entities['functions']:
            entity = self.entities['functions'][entity_name]
            deps = self.dependencies.get(entity_name, [])
            return {
                'type': 'function',
                'full_name': entity_name,
                'dependencies': deps,
                **entity
            }
        
        # Try classes
        if entity_name in self.entities['classes']:
            entity = self.entities['classes'][entity_name]
            return {
                'type': 'class',
                'full_name': entity_name,
                **entity
            }
        
        return None
    
    def get_module_entities(self, module_name: str) -> Dict[str, List[str]]:
        """
        Get all entities in a module
        
        Args:
            module_name: Module name
            
        Returns:
            Dict with functions and classes lists
        """
        functions = [
            name for name, entity in self.entities['functions'].items()
            if entity['module'] == module_name and 'class' not in entity
        ]
        
        classes = [
            name for name, entity in self.entities['classes'].items()
            if entity['module'] == module_name
        ]
        
        return {
            'functions': functions,
            'classes': classes
        }
    
    def get_related_entities(self, entity_name: str) -> Dict[str, List[str]]:
        """
        Get entities related to a given entity
        
        Args:
            entity_name: Entity to find relations for
            
        Returns:
            Dict with dependencies and dependents
        """
        # Direct dependencies
        dependencies = self.dependencies.get(entity_name, [])
        
        # Find dependents (who depends on this entity)
        dependents = [
            name for name, deps in self.dependencies.items()
            if entity_name in deps or entity_name.split('.')[-1] in deps
        ]
        
        return {
            'dependencies': dependencies,
            'dependents': dependents
        }
    
    def to_json(self) -> Dict[str, Any]:
        """Export index to JSON-serializable dict"""
        return {
            'entities': self.entities,
            'dependencies': self.dependencies
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get indexing statistics"""
        return {
            'total_functions': len(self.entities['functions']),
            'total_classes': len(self.entities['classes']),
            'total_modules': len(self.entities['modules']),
            'total_dependencies': len(self.dependencies)
        }
