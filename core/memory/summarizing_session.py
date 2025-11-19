"""
Summarizing Session - Compresses old turns, keeps recent verbatim
Based on OpenAI Context Engineering guide
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SummarizingSession:
    """
    Session that summarizes older turns and keeps last N verbatim.
    
    When user turns exceed context_limit:
    - Everything before the last keep_last_n_turns is summarized
    - Summary injected as synthetic user→assistant pair
    - Last keep_last_n_turns kept verbatim
    
    Use case: Long development workflows where older context must be preserved
    """
    
    def __init__(
        self,
        session_id: str,
        keep_last_n_turns: int = 3,
        context_limit: int = 10,
        summarizer: Optional[Any] = None
    ):
        """
        Initialize summarizing session
        
        Args:
            session_id: Unique session identifier  
            keep_last_n_turns: Number of recent turns to keep verbatim
            context_limit: Max turns before triggering summarization
            summarizer: Optional summarizer object with summarize() method
        """
        assert context_limit >= 1, "context_limit must be >= 1"
        assert keep_last_n_turns >= 0, "keep_last_n_turns must be >= 0"
        assert keep_last_n_turns <= context_limit, "keep_last_n_turns must be <= context_limit"
        
        self.session_id = session_id
        self.keep_last_n_turns = keep_last_n_turns
        self.context_limit = context_limit
        self.summarizer = summarizer
        
        self._records: deque[Dict[str, Any]] = deque()
        self._lock = asyncio.Lock()
        
        logger.info(
            f"SummarizingSession {session_id} initialized: "
            f"keep_last={keep_last_n_turns}, limit={context_limit}"
        )
    
    async def get_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get session items (model-safe messages only)
        
        Args:
            limit: Optional limit on items
            
        Returns:
            List of messages
        """
        async with self._lock:
            data = list(self._records)
            msgs = [self._sanitize_for_model(rec['msg']) for rec in data]
            
            if limit:
                return msgs[-limit:]
            return msgs
    
    async def add_items(self, items: List[Dict[str, Any]]) -> None:
        """
        Add items and trigger summarization if needed
        
        Args:
            items: List of items to add
        """
        # 1. Ingest items
        async with self._lock:
            for item in items:
                msg, meta = self._split_msg_and_meta(item)
                self._records.append({'msg': msg, 'meta': meta})
            
            need_summary, boundary = self._summarize_decision_locked()
        
        # 2. No summarization needed
        if not need_summary:
            async with self._lock:
                self._normalize_synthetic_flags_locked()
            return
        
        # 3. Prepare summary (outside lock)
        async with self._lock:
            snapshot = list(self._records)
        
        prefix_msgs = [r['msg'] for r in snapshot[:boundary]]
        user_shadow, assistant_summary = await self._summarize(prefix_msgs)
        
        # 4. Apply summary atomically
        async with self._lock:
            still_need, new_boundary = self._summarize_decision_locked()
            
            if not still_need:
                self._normalize_synthetic_flags_locked()
                return
            
            snapshot = list(self._records)
            suffix = snapshot[new_boundary:]  # Keep last N turns
            
            # Replace with: synthetic pair + suffix
            self._records.clear()
            self._records.extend([
                {
                    'msg': {'role': 'user', 'content': user_shadow},
                    'meta': {
                        'synthetic': True,
                        'kind': 'history_summary_prompt',
                        'summary_for_turns': f'<all before idx {new_boundary}>'
                    }
                },
                {
                    'msg': {'role': 'assistant', 'content': assistant_summary},
                    'meta': {
                        'synthetic': True,
                        'kind': 'history_summary',
                        'summary_for_turns': f'<all before idx {new_boundary}>'
                    }
                }
            ])
            self._records.extend(suffix)
            self._normalize_synthetic_flags_locked()
        
        logger.info(
            f"Session {self.session_id}: Summarized {boundary} items, "
            f"kept {len(suffix)} recent turns"
        )
    
    async def pop_item(self) -> Optional[Dict[str, Any]]:
        """Pop latest message"""
        async with self._lock:
            if not self._records:
                return None
            rec = self._records.pop()
            return dict(rec['msg'])
    
    async def clear(self) -> None:
        """Clear all records"""
        async with self._lock:
            self._records.clear()
        logger.info(f"Session {self.session_id} cleared")
    
    async def get_full_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get full history with metadata (for debugging/analytics)
        
        Returns:
            List of {message, metadata} dicts
        """
        async with self._lock:
            data = list(self._records)
            out = [{'message': dict(rec['msg']), 'metadata': dict(rec['meta'])} for rec in data]
            
            if limit:
                return out[-limit:]
            return out
    
    def _split_msg_and_meta(self, item: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Split item into message and metadata"""
        allowed_keys = {'role', 'content', 'name'}
        
        msg = {k: v for k, v in item.items() if k in allowed_keys}
        extra = {k: v for k, v in item.items() if k not in allowed_keys}
        
        meta = dict(extra.pop('metadata', {}))
        meta.update(extra)
        
        # Defaults
        msg.setdefault('role', 'user')
        msg.setdefault('content', str(item))
        
        role = msg.get('role')
        if role in ('user', 'assistant') and 'synthetic' not in meta:
            meta['synthetic'] = False
        
        return msg, meta
    
    @staticmethod
    def _sanitize_for_model(msg: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only model-safe keys"""
        allowed = {'role', 'content', 'name'}
        return {k: v for k, v in msg.items() if k in allowed}
    
    @staticmethod
    def _is_real_user_turn_start(rec: Dict[str, Any]) -> bool:
        """Check if record starts a real user turn"""
        return (
            rec['msg'].get('role') == 'user'
            and not rec['meta'].get('synthetic', False)
        )
    
    def _summarize_decision_locked(self) -> Tuple[bool, int]:
        """
        Decide if summarization is needed and compute boundary
        
        Returns:
            (need_summary, boundary_idx)
        """
        user_starts: List[int] = [
            i for i, rec in enumerate(self._records)
            if self._is_real_user_turn_start(rec)
        ]
        
        real_turns = len(user_starts)
        
        # Not over limit
        if real_turns <= self.context_limit:
            return False, -1
        
        # Keep zero turns → summarize everything
        if self.keep_last_n_turns == 0:
            return True, len(self._records)
        
        # Keep last N turns
        if len(user_starts) < self.keep_last_n_turns:
            return False, -1
        
        boundary = user_starts[-self.keep_last_n_turns]
        
        if boundary <= 0:
            return False, -1
        
        return True, boundary
    
    def _normalize_synthetic_flags_locked(self) -> None:
        """Ensure all real user/assistant records have synthetic=False"""
        for rec in self._records:
            role = rec['msg'].get('role')
            if role in ('user', 'assistant') and 'synthetic' not in rec['meta']:
                rec['meta']['synthetic'] = False
    
    async def _summarize(self, prefix_msgs: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Summarize prefix messages
        
        Returns:
            (user_shadow, assistant_summary)
        """
        if not self.summarizer:
            return (
                "Summarize the conversation we had so far.",
                "Summary not available (no summarizer configured)."
            )
        
        clean_prefix = [self._sanitize_for_model(m) for m in prefix_msgs]
        return await self.summarizer.summarize(clean_prefix)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        async with self._lock:
            user_turns = sum(
                1 for rec in self._records
                if self._is_real_user_turn_start(rec)
            )
            synthetic_count = sum(
                1 for rec in self._records
                if rec['meta'].get('synthetic', False)
            )
            
            return {
                'session_id': self.session_id,
                'type': 'summarizing',
                'keep_last_n_turns': self.keep_last_n_turns,
                'context_limit': self.context_limit,
                'total_records': len(self._records),
                'real_user_turns': user_turns,
                'synthetic_items': synthetic_count
            }
