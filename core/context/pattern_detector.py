"""
Pattern Detector - Detects contextual references in queries
Identifies patterns like "that function", "the bug", "previous error"
"""

import re
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReferenceType(Enum):
    """Types of contextual references"""
    DEMONSTRATIVE = "demonstrative"  # that, this, those
    DEFINITE = "definite"  # the function, the bug
  PRONOUN = "pronoun"  # it, its
    POSITIONAL = "positional"  # previous, last, earlier
    IMPLICIT = "implicit"  # no explicit reference but contextual


class PatternDetector:
    """
    Detects contextual references in user queries
    Identifies when user is referring to previously mentioned entities
    """
    
    def __init__(self):
        """Initialize pattern detector with regex patterns"""
        
        # Demonstrative patterns: "that function", "this class"
        self.demonstrative_patterns = [
            r'\b(that|this|those|these)\s+(\w+)',
            r'\b(esa|ese|esta|este|esos|estas)\s+(\w+)',  # Spanish
        ]
        
        # Definite article patterns: "the bug", "the error"
        self.definite_patterns = [
            r'\bthe\s+(\w+)',
            r'\b(la|el|los|las)\s+(\w+)',  # Spanish
        ]
        
        # Pronoun patterns: "it", "its method"
        self.pronoun_patterns = [
            r'\b(it|its|itself)\b',
            r'\b(lo|la|su|sus)\b',  # Spanish
        ]
        
        # Positional patterns: "previous function", "last bug"
        self.positional_patterns = [
            r'\b(previous|last|earlier|prior|above|before)\s+(\w+)',
            r'\b(anterior|último|previo)\s+(\w+)',  # Spanish
        ]
        
        # Code entity keywords
        self.code_keywords = [
            'function', 'class', 'method', 'variable', 'constant',
            'module', 'file', 'bug', 'error', 'issue', 'feature',
            'función', 'clase', 'método', 'variable', 'módulo', 'archivo'  # Spanish
        ]
        
        logger.info("PatternDetector initialized")
    
    def detect_references(self, query: str) -> List[Dict[str, Any]]:
        """
        Detect all contextual references in a query
        
        Args:
            query: User query text
            
        Returns:
            List of detected references with metadata
        """
        references = []
        query_lower = query.lower()
        
        # Detect demonstratives
        for pattern in self.demonstrative_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': ReferenceType.DEMONSTRATIVE.value,
                    'text': match.group(0),
                    'determiner': match.group(1),
                    'target': match.group(2),
                    'position': match.span()
                })
        
        # Detect definite articles
        for pattern in self.definite_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                target = match.group(2) if match.lastindex == 2 else match.group(1)
                if target in self.code_keywords:
                    references.append({
                        'type': ReferenceType.DEFINITE.value,
                        'text': match.group(0),
                        'target': target,
                        'position': match.span()
                    })
        
        # Detect pronouns
        for pattern in self.pronoun_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': ReferenceType.PRONOUN.value,
                    'text': match.group(0),
                    'position': match.span()
                })
        
        # Detect positional references
        for pattern in self.positional_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': ReferenceType.POSITIONAL.value,
                    'text': match.group(0),
                    'modifier': match.group(1),
                    'target': match.group(2),
                    'position': match.span()
                })
        
        logger.debug(f"Detected {len(references)} references in query: {query}")
        return references
    
    def has_references(self, query: str) -> bool:
        """
        Quick check if query contains contextual references
        
        Args:
            query: User query
            
        Returns:
            True if references detected
        """
        return len(self.detect_references(query)) > 0
    
    def extract_implicit_context(self, query: str) -> Dict[str, Any]:
        """
        Extract implicit contextual hints from query
        
        Args:
            query: User query
            
        Returns:
            Dict with context hints
        """
        query_lower = query.lower()
        context = {
            'needs_context': False,
            'keywords': [],
            'intent': None
        }
        
        # Check for code keywords
        found_keywords = [kw for kw in self.code_keywords if kw in query_lower]
        if found_keywords:
            context['keywords'] = found_keywords
            context['needs_context'] = True
        
        # Detect intent
        if any(word in query_lower for word in ['how', 'cómo', 'what', 'qué']):
            context['intent'] = 'information'
        elif any(word in query_lower for word in ['fix', 'arreglar', 'solve', 'resolver']):
            context['intent'] = 'solution'
        elif any(word in query_lower for word in ['find', 'buscar', 'where', 'dónde']):
            context['intent'] = 'search'
        
        return context
    
    def get_reference_targets(self, references: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract target entities from references
        
        Args:
            references: List of detected references
            
        Returns:
            Set of target entity types
        """
        targets = set()
        
        for ref in references:
            if 'target' in ref:
                targets.add(ref['target'])
        
        return targets
    
    def classify_query_type(self, query: str) -> str:
        """
        Classify query type based on patterns
        
        Args:
            query: User query
            
        Returns:
            Query type: 'contextual', 'direct', 'ambiguous'
        """
        references = self.detect_references(query)
        
        if len(references) > 0:
            return 'contextual'
        
        # Check for entity names (capital letters, snake_case, camelCase)
        if re.search(r'[A-Z][a-z]+|[a-z]+_[a-z]+|[a-z]+[A-Z][a-z]+', query):
            return 'direct'
        
        return 'ambiguous'
