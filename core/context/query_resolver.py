"""
Contextual Query Resolver - Resolves contextual references to concrete entities
Integrates pattern detection, entity tracking, and code index
"""

from typing import Dict, List, Any, Optional
import logging

from .pattern_detector import PatternDetector, ReferenceType
from .entity_tracker import EntityTracker

logger = logging.getLogger(__name__)


class ContextualQueryResolver:
    """
    Resolves contextual queries using session history and code index
    Expands queries with concrete entity names
    """
    
    def __init__(
        self,
        entity_tracker: EntityTracker,
        code_indexer: Any = None  # CodeIndexer from indexing module
    ):
        """
        Initialize contextual query resolver
        
        Args:
            entity_tracker: EntityTracker instance
            code_indexer: Optional CodeIndexer for entity validation
        """
        self.entity_tracker = entity_tracker
        self.code_indexer = code_indexer
        self.pattern_detector = PatternDetector()
        
        logger.info("ContextualQueryResolver initialized")
    
    async def resolve_query(
        self,
        query: str,
        current_turn: int,
        session_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Resolve contextual query to expanded query with concrete entities
        
        Args:
            query: Original user query
            current_turn: Current turn number
            session_history: Optional session history for additional context
            
        Returns:
            Resolution result with expanded query and metadata
        """
        # Detect references
        references = self.pattern_detector.detect_references(query)
        
        if not references:
            # No references, return as-is
            return {
                'original_query': query,
                'expanded_query': query,
                'resolved_entities': [],
                'has_context': False
            }
        
        # Resolve each reference
        resolved_entities = []
        replacements = {}  # position -> replacement text
        
        for ref in references:
            resolution = await self._resolve_reference(ref, current_turn, session_history)
            
            if resolution:
                resolved_entities.append(resolution)
                replacements[ref['position']] = resolution['entity_name']
        
        # Build expanded query
        expanded_query = self._apply_replacements(query, replacements)
        
        logger.info(
            f"Resolved query: '{query}' -> '{expanded_query}' "
            f"({len(resolved_entities)} entities)"
        )
        
        return {
            'original_query': query,
            'expanded_query': expanded_query,
            'resolved_entities': resolved_entities,
            'references': references,
            'has_context': True
        }
    
    async def _resolve_reference(
        self,
        reference: Dict[str, Any],
        current_turn: int,
        session_history: Optional[List[Dict]]
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a single contextual reference
        
        Args:
            reference: Reference dict from pattern detector
            current_turn: Current turn number
            session_history: Session history
            
        Returns:
            Resolution dict or None
        """
        ref_type = reference['type']
        target = reference.get('target', '').lower()
        
        # Map target to entity type
        entity_type = self._map_target_to_entity_type(target)
        
        if ref_type == ReferenceType.DEMONSTRATIVE.value:
            # "that function", "this class"
            entity = self.entity_tracker.get_last_mentioned_entity(
                entity_type=entity_type,
                before_turn=current_turn
            )
            
            if entity:
                return {
                    'entity_name': entity,
                    'entity_type': entity_type,
                    'reference_type': ref_type,
                    'confidence': 0.9
                }
        
        elif ref_type == ReferenceType.DEFINITE.value:
            # "the bug", "the error"
            entity = self.entity_tracker.get_last_mentioned_entity(
                entity_type=entity_type,
                before_turn=current_turn
            )
            
            if entity:
                return {
                    'entity_name': entity,
                    'entity_type': entity_type,
                    'reference_type': ref_type,
                    'confidence': 0.8
                }
        
        elif ref_type == ReferenceType.PRONOUN.value:
            # "it", "its"
            # Get the most recent entity regardless of type
            entity = self.entity_tracker.get_last_mentioned_entity(
                before_turn=current_turn
            )
            
            if entity:
                # Determine entity type from history
                history = self.entity_tracker.get_mention_history(entity)
                entity_type = history[-1].entity_type if history else 'unknown'
                
                return {
                    'entity_name': entity,
                    'entity_type': entity_type,
                    'reference_type': ref_type,
                    'confidence': 0.7
                }
        
        elif ref_type == ReferenceType.POSITIONAL.value:
            # "previous function", "last error"
            # Get recent mentions and pick based on position
            recent = self.entity_tracker.get_recent_mentions(
                entity_type=entity_type,
                limit=5
            )
            
            # Filter by turn < current_turn
            recent = [m for m in recent if m.turn_id < current_turn]
            
            modifier = reference.get('modifier', '').lower()
            
            if modifier in ('previous', 'last', 'anterior', 'último'):
                if recent:
                    return {
                        'entity_name': recent[0].entity_name,
                        'entity_type': recent[0].entity_type,
                        'reference_type': ref_type,
                        'confidence': 0.85
                    }
            
            elif modifier in ('earlier', 'prior', 'before', 'previo'):
                # Get second-to-last
                if len(recent) >= 2:
                    return {
                        'entity_name': recent[1].entity_name,
                        'entity_type': recent[1] entity_type,
                        'reference_type': ref_type,
                        'confidence': 0.75
                    }
        
        return None
    
    def _map_target_to_entity_type(self, target: str) -> Optional[str]:
        """Map target keyword to entity type"""
        mapping = {
            'function': 'function',
            'función': 'function',
            'class': 'class',
            'clase': 'class',
            'method': 'function',
            'método': 'function',
            'variable': 'variable',
            'file': 'file',
            'archivo': 'file',
            'module': 'module',
            'módulo': 'module',
            'bug': 'bug',
            'error': 'error',
            'issue': 'issue',
            'feature': 'feature'
        }
        
        return mapping.get(target.lower())
    
    def _apply_replacements(
        self,
        original: str,
        replacements: Dict[tuple, str]
    ) -> str:
        """
        Apply entity replacements to query
        
        Args:
            original: Original query
            replacements: Dict of (start, end) -> replacement
            
        Returns:
            Expanded query
        """
        if not replacements:
            return original
        
        # Sort by position (reverse to maintain indices)
        sorted_positions = sorted(replacements.keys(), reverse=True)
        
        result = original
        for (start, end) in sorted_positions:
            replacement = replacements[(start, end)]
            result = result[:start] + replacement + result[end:]
        
        return result
    
    async def validate_with_code_index(self, entity_name: str) -> bool:
        """
        Validate entity exists in code index
        
        Args:
            entity_name: Entity to validate
            
        Returns:
            True if entity exists in code
        """
        if not self.code_indexer:
            return True  # Assume valid if no indexer
        
        try:
            results = await self.code_indexer.search(entity_name)
            return len(results) > 0
        except Exception as e:
            logger.warning(f"Error validating entity {entity_name}: {e}")
            return False
    
    def get_expanded_context(
        self,
        entity_name: str,
        max_contexts: int = 3
    ) -> List[str]:
        """
        Get additional context for resolved entity
        
        Args:
            entity_name: Entity name
            max_contexts: Maximum contexts
            
        Returns:
            List of context strings
        """
        return self.entity_tracker.get_context_for_entity(
            entity_name,
            max_contexts=max_contexts
        )
