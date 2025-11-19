"""
Entity Tracker - Tracks entity mentions across session history
Maintains mention history to resolve contextual references
"""

from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EntityMention:
    """Represents a single entity mention in conversation"""
    
    def __init__(
        self,
        entity_name: str,
        entity_type: str,
        turn_id: int,
        context: str,
        timestamp: str = None
    ):
        self.entity_name = entity_name
        self.entity_type = entity_type  # 'function', 'class', 'variable', 'file'
        self.turn_id = turn_id
        self.context = context  # Surrounding text
        self.timestamp = timestamp or datetime.now().isoformat()
        self.mention_count = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'turn_id': self.turn_id,
            'context': self.context,
            'timestamp': self.timestamp,
            'mention_count': self.mention_count
        }


class EntityTracker:
    """
    Tracks entity mentions across session turns
    Provides mention history for contextual resolution
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize entity tracker
        
        Args:
            max_history: Maximum mentions to keep in history
        """
        self.max_history = max_history
        
        # entity_name -> list of mentions (most recent last)
        self.mentions: Dict[str, deque[EntityMention]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        
        # turn_id -> list of entities mentioned in that turn
        self.turn_entities: Dict[int, List[str]] = defaultdict(list)
        
        # entity_type -> set of entity names
        self.entities_by_type: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"EntityTracker initialized with max_history={max_history}")
    
    def record_mention(
        self,
        entity_name: str,
        entity_type: str,
        turn_id: int,
        context: str = ""
    ) -> None:
        """
        Record an entity mention
        
        Args:
            entity_name: Name of the entity
            entity_type: Type (function, class, etc.)
            turn_id: Turn number where mentioned
            context: Surrounding context
        """
        mention = EntityMention(entity_name, entity_type, turn_id, context)
        
        # Add to mentions
        self.mentions[entity_name].append(mention)
        
        # Track by turn
        if entity_name not in self.turn_entities[turn_id]:
            self.turn_entities[turn_id].append(entity_name)
        
        # Track by type
        self.entities_by_type[entity_type].add(entity_name)
        
        logger.debug(f"Recorded mention: {entity_name} ({entity_type}) at turn {turn_id}")
    
    def get_recent_mentions(
        self,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[EntityMention]:
        """
        Get recent entity mentions
        
        Args:
            entity_type: Filter by entity type (optional)
            limit: Maximum mentions to return
            
        Returns:
            List of recent mentions
        """
        all_mentions = []
        
        for entity_mentions in self.mentions.values():
            all_mentions.extend(entity_mentions)
        
        # Filter by type if specified
        if entity_type:
            all_mentions = [m for m in all_mentions if m.entity_type == entity_type]
        
        # Sort by timestamp (most recent first)
        all_mentions.sort(key=lambda m: m.timestamp, reverse=True)
        
        return all_mentions[:limit]
    
    def get_last_mentioned_entity(
        self,
        entity_type: Optional[str] = None,
        before_turn: Optional[int] = None
    ) -> Optional[str]:
        """
        Get the most recently mentioned entity
        
        Args:
            entity_type: Filter by type
            before_turn: Only consider mentions before this turn
            
        Returns:
            Entity name or None
        """
        recent = self.get_recent_mentions(entity_type=entity_type, limit=100)
        
        if before_turn is not None:
            recent = [m for m in recent if m.turn_id < before_turn]
        
        return recent[0].entity_name if recent else None
    
    def get_entities_in_turn(self, turn_id: int) -> List[str]:
        """
        Get all entities mentioned in a specific turn
        
        Args:
            turn_id: Turn number
            
        Returns:
            List of entity names
        """
        return self.turn_entities.get(turn_id, [])
    
    def get_mention_history(self, entity_name: str) -> List[EntityMention]:
        """
        Get complete mention history for an entity
        
        Args:
            entity_name: Entity to look up
            
        Returns:
            List of mentions
        """
        return list(self.mentions.get(entity_name, []))
    
    def get_entities_by_type(self, entity_type: str) -> Set[str]:
        """
        Get all entities of a specific type
        
        Args:
            entity_type: Entity type to filter
            
        Returns:
            Set of entity names
        """
        return self.entities_by_type.get(entity_type, set())
    
    def find_matching_entities(
        self,
        target: str,
        entity_type: Optional[str] = None,
        fuzzy: bool = True
    ) -> List[str]:
        """
        Find entities matching a target string
        
        Args:
            target: Search string
            entity_type: Optional type filter
            fuzzy: Allow fuzzy matching
            
        Returns:
            List of matching entity names
        """
        matches = []
        target_lower = target.lower()
        
        # Search in mentioned entities
        for entity_name in self.mentions.keys():
            # Type filter
            if entity_type:
                entity_mentions = self.mentions[entity_name]
                if not any(m.entity_type == entity_type for m in entity_mentions):
                    continue
            
            # Exact match
            if target_lower == entity_name.lower():
                matches.append(entity_name)
                continue
            
            # Fuzzy match
            if fuzzy and target_lower in entity_name.lower():
                matches.append(entity_name)
        
        # Sort by recency (most recent first)
        def get_last_mention_time(name):
            mentions = self.mentions[name]
            return mentions[-1].timestamp if mentions else ""
        
        matches.sort(key=get_last_mention_time, reverse=True)
        
        return matches
    
    def get_context_for_entity(
        self,
        entity_name: str,
        max_contexts: int = 3
    ) -> List[str]:
        """
        Get contexts where entity was mentioned
        
        Args:
            entity_name: Entity name
            max_contexts: Maximum contexts to return
            
        Returns:
            List of context strings
        """
        mentions = self.get_mention_history(entity_name)
        
        # Get most recent contexts
        contexts = [m.context for m in reversed(mentions) if m.context]
        
        return contexts[:max_contexts]
    
    def clear(self) -> None:
        """Clear all tracking data"""
        self.mentions.clear()
        self.turn_entities.clear()
        self.entities_by_type.clear()
        logger.info("EntityTracker cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return {
            'total_entities': len(self.mentions),
            'total_mentions': sum(len(mentions) for mentions in self.mentions.values()),
            'entities_by_type': {
                etype: len(entities)
                for etype, entities in self.entities_by_type.items()
            },
            'turns_tracked': len(self.turn_entities)
        }
