"""
Session Manager - Manages multiple sessions
Coordinates session lifecycle and persistence
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import logging

from .session_storage import SessionStorage
from .trimming_session import TrimmingSession
from .summarizing_session import SummarizingSession

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Types of development sessions"""
    FEATURE_IMPLEMENTATION = "feature"
    BUG_FIXING = "bugfix"
    CODE_REVIEW = "review"
    REFACTORING = "refactor"
    GENERAL = "general"


class SessionManager:
    """
    Manages multiple sessions with persistence
    Coordinates between in-memory sessions and storage
    """
    
    def __init__(
        self,
        storage_dir: str,
        default_session_type: str = "trimming",
        default_max_turns: int = 8,
        default_keep_last: int = 3,
        default_context_limit: int = 10
    ):
        """
        Initialize session manager
        
        Args:
            storage_dir: Directory for session persistence
            default_session_type: 'trimming' or 'summarizing'
            default_max_turns: For trimming sessions
            default_keep_last: For summarizing sessions
            default_context_limit: For summarizing sessions
        """
        self.storage = SessionStorage(storage_dir)
        self.default_session_type = default_session_type
        self.default_max_turns = default_max_turns
        self.default_keep_last = default_keep_last
        self.default_context_limit = default_context_limit
        
        self._active_sessions: Dict[str, Any] = {}
        self._session_metadata: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"SessionManager initialized with storage at {storage_dir}")
    
    async def create_session(
        self,
        session_type: Optional[SessionType] = None,
        session_class: Optional[str] = None,
        metadata: Optional[Dict] = None,
        **session_kwargs
    ) -> str:
        """
        Create a new session
        
        Args:
            session_type: Type of development session
            session_class: 'trimming' or 'summarizing' (overrides default)
            metadata: Custom metadata for this session
            **session_kwargs: Additional session parameters
            
        Returns:
            session_id
        """
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        session_prefix = session_type.value if session_type else "gen"
        session_id = f"sess_{session_prefix}_{timestamp}"
        
        # Determine session class
        sess_class = session_class or self.default_session_type
        
        # Create session object
        if sess_class == "trimming":
            max_turns = session_kwargs.get('max_turns', self.default_max_turns)
            session_obj = TrimmingSession(session_id, max_turns=max_turns)
            
        elif sess_class == "summarizing":
            keep_last = session_kwargs.get('keep_last_n_turns', self.default_keep_last)
            limit = session_kwargs.get('context_limit', self.default_context_limit)
            summarizer = session_kwargs.get('summarizer', None)
            session_obj = SummarizingSession(
                session_id,
                keep_last_n_turns=keep_last,
                context_limit=limit,
                summarizer=summarizer
            )
        else:
            raise ValueError(f"Unknown session class: {sess_class}")
        
        # Store session
        async with self._lock:
            self._active_sessions[session_id] = session_obj
            self._session_metadata[session_id] = {
                'session_id': session_id,
                'type': session_type.value if session_type else 'general',
                'class': sess_class,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                **(metadata or {})
            }
        
        logger.info(f"Created {sess_class} session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Any]:
        """
        Get session object by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None
        """
        async with self._lock:
            return self._active_sessions.get(session_id)
    
    async def close_session(self, session_id: str, persist: bool = True) -> bool:
        """
        Close and optionally persist a session
        
        Args:
            session_id: Session identifier
            persist: Whether to save to disk
            
        Returns:
            True if successful
        """
        async with self._lock:
            session = self._active_sessions.get(session_id)
            
            if not session:
                logger.warning(f"Session {session_id} not found")
                return False
            
            if persist:
                # Get items and save
                items = await session.get_items()
                
                # For summarizing sessions, get full history
                if hasattr(session, 'get_full_history'):
                    history = await session.get_full_history()
                    # Save both items and metadata
                    turns = [
                        {
                            'turn_id': i,
                            **item.get('message', {}),
                            'metadata': item.get('metadata', {})
                        }
                        for i, item in enumerate(history)
                    ]
                else:
                    turns = [
                        {'turn_id': i, **item}
                        for i, item in enumerate(items)
                    ]
                
                await self.storage.save_session(session_id, turns)
            
            # Update metadata
            if session_id in self._session_metadata:
                self._session_metadata[session_id]['status'] = 'closed'
                self._session_metadata[session_id]['closed_at'] = datetime.now().isoformat()
            
            # Remove from active
            del self._active_sessions[session_id]
        
        logger.info(f"Closed session {session_id} (persist={persist})")
        return True
    
    async def load_session(self, session_id: str, session_class: str = "trimming") -> bool:
        """
        Load a persisted session into memory
        
        Args:
            session_id: Session identifier
            session_class: Type of session to create
            
        Returns:
            True if successful
        """
        turns = await self.storage.load_session(session_id)
        
        if not turns:
            logger.warning(f"No data found for session {session_id}")
            return False
        
        # Create session object
        if session_class == "trimming":
            session_obj = TrimmingSession(session_id, max_turns=self.default_max_turns)
        else:
            session_obj = SummarizingSession(
                session_id,
                keep_last_n_turns=self.default_keep_last,
                context_limit=self.default_context_limit
            )
        
        # Add turns
        items = [
            {k: v for k, v in turn.items() if k not in ('turn_id', 'timestamp', 'metadata')}
            for turn in turns
        ]
        await session_obj.add_items(items)
        
        # Store in active sessions
        async with self._lock:
            self._active_sessions[session_id] = session_obj
            self._session_metadata[session_id] = {
                'session_id': session_id,
                'loaded_at': datetime.now().isoformat(),
                'status': 'active'
            }
        
        logger.info(f"Loaded session {session_id} with {len(turns)} turns")
        return True
    
    async def list_sessions(self, status: Optional[str] = None) -> List[Dict]:
        """
        List sessions with metadata
        
        Args:
            status: Filter by status ('active', 'closed', or None for all)
            
        Returns:
            List of session metadata dicts
        """
        async with self._lock:
            sessions = list(self._session_metadata.values())
        
        if status:
            sessions = [s for s in sessions if s.get('status') == status]
        
        return sessions
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        Get summary of a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Summary dict or None
        """
        session = await self.get_session(session_id)
        
        if not session:
            # Try to get from storage
            return await self.storage.get_session_metadata(session_id)
        
        # Get stats from session object
        stats = await session.get_stats()
        
        # Combine with metadata
        async with self._lock:
            metadata = self._session_metadata.get(session_id, {})
        
        return {**metadata, **stats}
    
    async def delete_session(self, session_id: str, from_disk: bool = True) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session identifier
            from_disk: Also delete from disk
            
        Returns:
            True if successful
        """
        # Remove from memory
        async with self._lock:
            self._active_sessions.pop(session_id, None)
            self._session_metadata.pop(session_id, None)
        
        # Remove from disk
        if from_disk:
            await self.storage.delete_session(session_id)
        
        logger.info(f"Deleted session {session_id} (from_disk={from_disk})")
return True
