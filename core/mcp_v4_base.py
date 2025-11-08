#!/usr/bin/env python3
"""
MCP V4 - MÓDULO BASE
Componentes fundamentales consolidados de todas las versiones
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detector de alucinaciones (de v1 Enhanced)"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'definitivamente', r'siempre es', r'nunca falla',
            r'garantizado que', r'imposible que'
        ]
    
    def detect_hallucination_risks(self, query: str, context: str, 
                                  proposed_response: str = "") -> Dict:
        risks = {
            'confidence_level': 'medium',
            'risk_factors': [],
            'recommendations': [],
            'context_gaps': []
        }
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, proposed_response.lower()):
                risks['risk_factors'].append(f'absolute_statement: {pattern}')
        
        if len(context.strip()) < 100:
            risks['context_gaps'].append('insufficient_context')
            risks['confidence_level'] = 'low'
        
        if len(query.split()) < 3:
            risks['risk_factors'].append('ambiguous_query')
        
        if risks['risk_factors'] or risks['context_gaps']:
            risks['recommendations'] = self._generate_recommendations(risks)
        
        return risks
    
    def _generate_recommendations(self, risks: Dict) -> List[str]:
        recommendations = []
        if 'insufficient_context' in risks['context_gaps']:
            recommendations.extend(['request_more_context', 'acknowledge_limitations'])
        if any('absolute_statement' in rf for rf in risks['risk_factors']):
            recommendations.extend(['use_qualified_language', 'provide_alternatives'])
        return recommendations


class ContextValidator:
    """Validador de contexto (de v1 Enhanced)"""
    
    def __init__(self):
        self.validation_rules = {
            'min_context_length': 50,
            'max_context_age': 3600,
            'required_sections': ['description', 'examples']
        }
    
    def validate_context_quality(self, context: str, metadata: Dict = None) -> Dict:
        validation = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'improvements': []
        }
        
        if len(context.strip()) < self.validation_rules['min_context_length']:
            validation['issues'].append('context_too_short')
            validation['is_valid'] = False
        
        structure_score = self._assess_structure(context)
        relevance_score = self._assess_relevance(context, metadata)
        
        validation['quality_score'] = structure_score * 0.4 + relevance_score * 0.6
        
        if validation['quality_score'] < 0.7:
            validation['improvements'] = self._suggest_improvements(context)
        
        return validation
    
    def _assess_structure(self, context: str) -> float:
        score = 0.0
        if re.search(r'(class|def|function)', context): score += 0.3
        if re.search(r'(""".*?"""|#.*)', context, re.DOTALL): score += 0.2
        if 'example' in context.lower() or '```' in context: score += 0.3
        if re.search(r'(#{1,3}\s+|^\s*\d+\.)', context, re.MULTILINE): score += 0.2
        return min(1.0, score)
    
    def _assess_relevance(self, context: str, metadata: Dict = None) -> float:
        if not metadata or not metadata.get('query'):
            return 0.5
        
        query = metadata['query']
        context_words = set(re.findall(r'\w+', context.lower()))
        query_words = set(re.findall(r'\w+', query.lower()))
        
        if not query_words:
            return 0.5
        
        overlap = len(context_words & query_words) / len(query_words)
        return min(1.0, overlap * 2)
    
    def _suggest_improvements(self, context: str) -> List[str]:
        improvements = []
        if len(context) < 200: improvements.append('add_more_detail')
        if not re.search(r'(example|ejemplo)', context.lower()): improvements.append('add_examples')
        if not re.search(r'(""".*?"""|#.*)', context, re.DOTALL): improvements.append('add_documentation')
        return improvements


class ResponseQualityMonitor:
    """Monitor de calidad de respuestas (de v1 Enhanced)"""
    
    def __init__(self):
        self.response_history = []
    
    def evaluate_response_quality(self, query: str, context: str, response: str) -> Dict:
        metrics = {
            'relevance': self._assess_relevance(query, response),
            'completeness': self._assess_completeness(response),
            'clarity': self._assess_clarity(response),
            'accuracy': self._assess_accuracy(context, response)
        }
        
        weights = {'relevance': 0.3, 'completeness': 0.25, 'clarity': 0.2, 'accuracy': 0.25}
        overall_score = sum(metrics[m] * weights[m] for m in metrics)
        
        return {
            'overall_score': overall_score,
            'metrics': metrics,
            'confidence': 'high' if overall_score >= 0.8 else 'low' if overall_score <= 0.5 else 'medium',
            'strengths': [f"high_{m}" for m, s in metrics.items() if s >= 0.8],
            'improvements': [f"improve_{m}" for m, s in metrics.items() if s < 0.6]
        }
    
    def _assess_relevance(self, query: str, response: str) -> float:
        query_words = set(re.findall(r'\w+', query.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        if not query_words: return 0.5
        overlap = len(query_words & response_words) / len(query_words)
        return min(1.0, overlap * 1.5)
    
    def _assess_completeness(self, response: str) -> float:
        base = min(1.0, len(response) / 500)
        if re.search(r'(#{1,3}|\d+\.|\*\s)', response): base += 0.2
        if '```' in response or 'example' in response.lower(): base += 0.2
        return min(1.0, base)
    
    def _assess_clarity(self, response: str) -> float:
        sentences = len(re.split(r'[.!?]+', response))
        words = len(response.split())
        if sentences == 0: return 0.0
        
        avg_len = words / sentences
        clarity = 0.8 if 10 <= avg_len <= 25 else 0.5
        
        if any(m in response for m in ['**', '*', '`', '```']):
            clarity += 0.2
        
        return min(1.0, clarity)
    
    def _assess_accuracy(self, context: str, response: str) -> float:
        return 0.85  # Default alto si no se detectan contradicciones
