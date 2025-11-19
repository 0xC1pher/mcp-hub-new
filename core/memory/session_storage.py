"""
Session Storage - Persistence layer for session data
Stores session history in JSONL format
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SessionStorage:
    """
    Manages persistence of session data to disk
    Format: JSONL (one JSON object per line)
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize session storage
        
        Args:
            storage_dir: Directory for session files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionStorage initialized at {self.storage_dir}")
    
    def get_session_path(self, session_id: str) -> Path:
        """Get file path for a session"""
        return self.storage_dir / f"{session_id}.jsonl"
    
    async def append_turn(self, session_id: str, turn_data: Dict) -> bool:
        """
        Append a turn to session file
        
        Args:
            session_id: Session identifier
            turn_data: Turn data to append
            
        Returns:
            True if successful
        """
        try:
            session_file = self.get_session_path(session_id)
            
            # Add timestamp if not present
            if 'timestamp' not in turn_data:
                turn_data['timestamp'] = datetime.now().isoformat()
            
            # Append to file
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(turn_data) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Error appending turn to {session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> List[Dict]:
        """
        Load all turns from a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of turn dictionaries
        """
        try:
            session_file = self.get_session_path(session_id)
            
            if not session_file.exists():
                logger.info(f"Session {session_id} not found, returning empty")
                return []
            
            turns = []
            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        turns.append(json.loads(line))
            
            logger.info(f"Loaded {len(turns)} turns from {session_id}")
            return turns
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return []
    
    async def save_session(self, session_id: str, turns: List[Dict]) -> bool:
        """
        Save complete session (overwrites existing)
        
        Args:
            session_id: Session identifier
            turns: List of turn data
            
        Returns:
            True if successful
        """
        try:
            session_file = self.get_session_path(session_id)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                for turn in turns:
                    if 'timestamp' not in turn:
                        turn['timestamp'] = datetime.now().isoformat()
                    f.write(json.dumps(turn) + '\n')
            
            logger.info(f"Saved {len(turns)} turns to {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session file
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            session_file = self.get_session_path(session_id)
            
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Deleted session {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def list_sessions(self) -> List[str]:
        """
        List all session IDs
        
        Returns:
            List of session IDs
        """
        try:
            session_files = list(self.storage_dir.glob("*.jsonl"))
            session_ids = [f.stem for f in session_files]
            return sorted(session_ids)
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
    
    async def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """
        Get metadata about a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Metadata dictionary or None
        """
        try:
            session_file = self.get_session_path(session_id)
            
            if not session_file.exists():
                return None
            
            turns = await self.load_session(session_id)
            
            if not turns:
                return None
            
            first_turn = turns[0]
            last_turn = turns[-1]
            
            return {
                'session_id': session_id,
                'turn_count': len(turns),
                'created_at': first_turn.get('timestamp', 'unknown'),
                'last_activity': last_turn.get('timestamp', 'unknown'),
                'file_size': session_file.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for {session_id}: {e}")
            return None
