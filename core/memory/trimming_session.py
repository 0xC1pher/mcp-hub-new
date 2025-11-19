"""
Trimming Session - Keeps last N turns verbatim
Based on OpenAI Context Engineering guide
"""

import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TrimmingSession:
    """
    Session that keeps only the last N user turns verbatim.
    
    A turn = one user message + all subsequent items (assistant, tool calls/results)
    until the next user message.
    
    Use case: Independent tasks where only recent context matters
    """
    
    def __init__(self, session_id: str, max_turns: int = 8):
        """
        Initialize trimming session
        
        Args:
            session_id: Unique session identifier
            max_turns: Maximum number of user turns to keep
        """
        self.session_id = session_id
        self.max_turns = max(1, int(max_turns))
        self._items: deque[Dict[str, Any]] = deque()
        self._lock = asyncio.Lock()
        
        logger.info(f"TrimmingSession {session_id} initialized with max_turns={max_turns}")
    
    async def get_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get session items (trimmed to last N user turns)
        
        Args:
            limit: Optional limit on number of items to return
            
        Returns:
            List of items
        """
        async with self._lock:
            trimmed = self._trim_to_last_turns(list(self._items))
            
            if limit is not None and limit >= 0:
                return trimmed[-limit:]
            return trimmed
    
    async def add_items(self, items: List[Dict[str, Any]]) -> None:
        """
        Add items to session and trim to last N user turns
        
        Args:
            items: List of items to add
        """
        if not items:
            return
        
        async with self._lock:
            self._items.extend(items)
            trimmed = self._trim_to_last_turns(list(self._items))
            self._items.clear()
            self._items.extend(trimmed)
        
        logger.debug(f"Session {self.session_id}: Added {len(items)} items, now have {len(self._items)} items")
    
    async def pop_item(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return the most recent item
        
        Returns:
            Last item or None
        """
        async with self._lock:
            return self._items.pop() if self._items else None
    
    async def clear(self) -> None:
        """Clear all items from session"""
        async with self._lock:
            self._items.clear()
        logger.info(f"Session {self.session_id} cleared")
    
    async def set_max_turns(self, max_turns: int) -> None:
        """
        Update max_turns and re-trim
        
        Args:
            max_turns: New maximum turns
        """
        async with self._lock:
            self.max_turns = max(1, int(max_turns))
            trimmed = self._trim_to_last_turns(list(self._items))
            self._items.clear()
            self._items.extend(trimmed)
        
        logger.info(f"Session {self.session_id} max_turns updated to {self.max_turns}")
    
    def _is_user_message(self, item: Dict[str, Any]) -> bool:
        """Check if item is a user message"""
        role = item.get('role')
        if role is not None:
            return role == 'user'
        
        # Fallback: check 'type' field
        if item.get('type') == 'message' and item.get('role') == 'user':
            return True
        
        return False
    
    def _trim_to_last_turns(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep only the last max_turns user turns.
        
        Args:
            items: List of all items
            
        Returns:
            Trimmed list
        """
        if not items:
            return items
        
        count = 0
        start_idx = 0
        
        # Walk backward to find the Nth user message
        for i in range(len(items) - 1, -1, -1):
            if self._is_user_message(items[i]):
                count += 1
                if count == self.max_turns:
                    start_idx = i
                    break
        
        return items[start_idx:]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        async with self._lock:
            user_turns = sum(1 for item in self._items if self._is_user_message(item))
            
            return {
                'session_id': self.session_id,
                'type': 'trimming',
                'max_turns': self.max_turns,
                'total_items': len(self._items),
                'user_turns': user_turns,
                'assistant_items': len(self._items) - user_turns
            }
