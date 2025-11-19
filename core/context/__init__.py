"""
Context resolution module initialization
Resolves contextual references in queries using session history and code index
"""

from .pattern_detector import PatternDetector
from .entity_tracker import EntityTracker
from .query_resolver import ContextualQueryResolver

__all__ = [
    'PatternDetector',
    'EntityTracker',
    'ContextualQueryResolver'
]
