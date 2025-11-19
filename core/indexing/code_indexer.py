"""
Code Indexer - Main interface for code structure indexing
Manages entity extraction, persistence, and querying
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class CodeIndexer:
    """
    Main code indexing system
    Coordinates entity extraction and persistence
    """
    
    def __init__(self, index_dir: str = "data/code_index"):
        """
        Initialize code indexer
        
        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = EntityExtractor()
        self._lock = asyncio.Lock()
        
        self.index_metadata = {
            'created_at': None,
            'last_updated': None,
            'indexed_directories': [],
            'total_files': 0
        }
        
        logger.info(f"CodeIndexer initialized at {index_dir}")
    
    async def index_codebase(
        self,
        directories: List[str],
        extensions: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Index entire codebase
        
        Args:
            directories: List of directories to index
            extensions: File extensions to index
            exclude_patterns: Patterns to exclude (e.g., 'test_', '__pycache__')
            
        Returns:
            Indexing statistics
        """
        if extensions is None:
            extensions = ['.py']
        
        if exclude_patterns is None:
            exclude_patterns = ['test_', '__pycache__', '.pyc', 'venv', 'node_modules']
        
        async with self._lock:
            total_files = 0
            
            for directory in directories:
                dir_path = Path(directory)
                
                if not dir_path.exists():
                    logger.warning(f"Directory not found: {directory}")
                    continue
                
                # Index files
                for ext in extensions:
                    for file_path in dir_path.rglob(f'*{ext}'):
                        # Check exclude patterns
                        if any(pattern in str(file_path) for pattern in exclude_patterns):
                            continue
                        
                        if await asyncio.to_thread(self.extractor.index_file, str(file_path)):
                            total_files += 1
            
            # Update metadata
            now = datetime.now().isoformat()
            if not self.index_metadata['created_at']:
                self.index_metadata['created_at'] = now
            
            self.index_metadata['last_updated'] = now
            self.index_metadata['indexed_directories'] = directories
            self.index_metadata['total_files'] = total_files
            
            # Save index
            await self.save_index()
            
            stats = self.extractor.get_stats()
            stats['total_files'] = total_files
            
            logger.info(f"Indexed {total_files} files: {stats}")
            
            return stats
    
    async def save_index(self) -> bool:
        """
        Save index to disk
        
        Returns:
            True if successful
        """
        try:
            # Save entity index
            entities_file = self.index_dir / 'entities.json'
            index_data = self.extractor.to_json()
            
            async with asyncio.Lock():
                with open(entities_file, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=2)
            
            # Save metadata
            metadata_file = self.index_dir / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_metadata, f, indent=2)
            
            logger.info(f"Index saved to {self.index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    async def load_index(self) -> bool:
        """
        Load index from disk
        
        Returns:
            True if successful
        """
        try:
            entities_file = self.index_dir / 'entities.json'
            metadata_file = self.index_dir / 'metadata.json'
            
            if not entities_file.exists():
                logger.warning("No index file found")
                return False
            
            # Load entities
            with open(entities_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.extractor.entities = index_data.get('entities', {})
            self.extractor.dependencies = index_data.get('dependencies', {})
            
            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.index_metadata = json.load(f)
            
            logger.info(f"Index loaded from {self.index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for entities
        
        Args:
            query: Search query
            
        Returns:
            List of matching entities
        """
        async with self._lock:
            return await asyncio.to_thread(self.extractor.find_entity, query)
    
    async def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get specific entity info
        
        Args:
            entity_name: Full entity name
            
        Returns:
            Entity info or None
        """
        async with self._lock:
            return await asyncio.to_thread(self.extractor.get_entity_info, entity_name)
    
    async def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """
        Get all entities in a module
        
        Args:
            module_name: Module name
            
        Returns:
            Module entity list
        """
        async with self._lock:
            entities = await asyncio.to_thread(
                self.extractor.get_module_entities,
                module_name
            )
            
            # Add module metadata if available
            module_file = None
            for file_path, module_info in self.extractor.entities['modules'].items():
                if module_info['name'] == module_name:
                    module_file = file_path
                    break
            
            return {
                'module': module_name,
                'file_path': module_file,
                **entities
            }
    
    async def get_dependencies(self, entity_name: str) -> Dict[str, List[str]]:
        """
        Get entity dependencies and dependents
        
        Args:
            entity_name: Entity name
            
        Returns:
            Dependencies and dependents
        """
        async with self._lock:
            return await asyncio.to_thread(
                self.extractor.get_related_entities,
                entity_name
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        async with self._lock:
            stats = self.extractor.get_stats()
            return {
                **stats,
                **self.index_metadata
            }
    
    async def refresh_index(
        self,
        directories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Refresh index (re-index previously indexed directories)
        
        Args:
            directories: Directories to refresh (default: previously indexed)
            
        Returns:
            Refresh statistics
        """
        if directories is None:
            directories = self.index_metadata.get('indexed_directories', [])
        
        if not directories:
            logger.warning("No directories to refresh")
            return {}
        
        # Clear current index
        async with self._lock:
            self.extractor.entities = {
                'functions': {},
                'classes': {},
                'modules': {}
            }
            self.extractor.dependencies = {}
        
        # Re-index
        return await self.index_codebase(directories)
    
    def build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build complete dependency graph
        
        Returns:
            Graph as adjacency list
        """
        return dict(self.extractor.dependencies)
    
    def get_function_signature(self, function_name: str) -> Optional[str]:
        """
        Get function signature by name
        
        Args:
            function_name: Function name (can be partial)
            
        Returns:
            Function signature or None
        """
        # Search for matching function
        results = self.extractor.find_entity(function_name)
        
        for result in results:
            if result['type'] == 'function':
                return result.get('signature')
        
        return None
