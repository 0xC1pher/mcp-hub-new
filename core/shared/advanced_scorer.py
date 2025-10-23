#!/usr/bin/env python3
"""
🎯 Advanced Scorer - Migrado desde legacy/unified
Sistema de scoring avanzado con métricas de relevancia
Integrado con sistema de feedback
"""
import re
import math
from typing import Dict, List, Any, Optional
import time

class AdvancedScorer:
    """Sistema de scoring avanzado"""
    
    def __init__(self):
        self.feedback_integration = True
        self.scoring_history = {}
        self.performance_metrics = {
            'total_scores': 0,
            'avg_score': 0.0,
            'high_scores': 0,  # scores > 0.8
            'low_scores': 0    # scores < 0.3
        }
    
    def calculate_score(self, query: str, content: str, context: Dict[str, Any] = None) -> float:
        """Calcula score de relevancia avanzado"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        scores = {}
        
        # Exact match con frecuencia
        exact_count = content_lower.count(query_lower)
        scores['exact_match'] = min(1.0, exact_count * 0.5)
        
        # Partial matches
        query_words = query_lower.split()
        word_scores = []
        
        for word in query_words:
            if len(word) > 2:  # Ignorar palabras muy cortas
                word_count = content_lower.count(word)
                word_score = min(1.0, word_count * 0.3)
                
                # Bonus por posición (inicio del contenido)
                if content_lower.find(word) < len(content_lower) * 0.2:
                    word_score *= 1.2
                
                word_scores.append(word_score)
        
        scores['partial_match'] = sum(word_scores) / len(query_words) if query_words else 0
        
        # Context density
        scores['context_density'] = self._calculate_context_density(content)
        
        # Semantic similarity (básico)
        scores['semantic_similarity'] = self._calculate_semantic_similarity(query, content)
        
        # Feedback integration
        scores['feedback_boost'] = self._get_feedback_boost(query, content, context)
        
        # Weighted final score
        final_score = (
            scores['exact_match'] * 2.5 +
            scores['partial_match'] * 1.8 +
            scores['context_density'] * 1.0 +
            scores['semantic_similarity'] * 1.2 +
            scores['feedback_boost'] * 0.5
        ) / 7.0
        
        final_score = min(1.0, final_score)
        
        # Actualizar métricas
        self._update_metrics(final_score, scores)
        
        return final_score
    
    def _calculate_context_density(self, content: str) -> float:
        """Calcula densidad de contexto"""
        # Elementos que indican alta densidad de contexto
        code_elements = content.count('def ') + content.count('class ') + content.count('import ')
        list_items = content.count('\n- ') + content.count('\n* ')
        headers = content.count('\n#')
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content))  # CamelCase
        
        total_elements = code_elements + list_items + headers + (technical_terms / 10)
        content_length = len(content)
        
        if content_length == 0:
            return 0.0
        
        density = min(1.0, total_elements / (content_length / 100))
        return density
    
    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """Calcula similitud semántica básica"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost para términos técnicos relacionados
        technical_boost = self._get_technical_term_boost(query, content)
        
        return min(1.0, jaccard + technical_boost)
    
    def _get_technical_term_boost(self, query: str, content: str) -> float:
        """Boost para términos técnicos relacionados"""
        technical_patterns = {
            'cache': ['memory', 'storage', 'buffer', 'fast'],
            'optimization': ['performance', 'speed', 'efficient', 'fast'],
            'mcp': ['server', 'protocol', 'context', 'model'],
            'token': ['budget', 'limit', 'allocation', 'management'],
            'chunk': ['segment', 'piece', 'part', 'division']
        }
        
        query_lower = query.lower()
        content_lower = content.lower()
        boost = 0.0
        
        for main_term, related_terms in technical_patterns.items():
            if main_term in query_lower:
                for related in related_terms:
                    if related in content_lower:
                        boost += 0.05
        
        return min(0.3, boost)  # Máximo 30% boost
    
    def _get_feedback_boost(self, query: str, content: str, context: Dict[str, Any] = None) -> float:
        """Boost basado en feedback histórico"""
        if not self.feedback_integration or not context:
            return 0.0
        
        # Boost por acceso frecuente
        access_count = context.get('access_count', 0)
        access_boost = min(0.2, access_count * 0.01)
        
        # Boost por recencia
        last_accessed = context.get('last_accessed', 0)
        if last_accessed:
            hours_since_access = (time.time() - last_accessed) / 3600
            recency_boost = max(0, 0.1 - (hours_since_access * 0.01))
        else:
            recency_boost = 0
        
        # Boost por rating positivo
        user_rating = context.get('user_rating', 0)
        rating_boost = user_rating * 0.05 if user_rating > 0 else 0
        
        return access_boost + recency_boost + rating_boost
    
    def _update_metrics(self, final_score: float, component_scores: Dict[str, float]) -> None:
        """Actualiza métricas de rendimiento"""
        self.performance_metrics['total_scores'] += 1
        
        # Actualizar promedio
        current_avg = self.performance_metrics['avg_score']
        total = self.performance_metrics['total_scores']
        self.performance_metrics['avg_score'] = ((current_avg * (total - 1)) + final_score) / total
        
        # Contar scores altos y bajos
        if final_score > 0.8:
            self.performance_metrics['high_scores'] += 1
        elif final_score < 0.3:
            self.performance_metrics['low_scores'] += 1
    
    def calculate_batch_scores(self, query: str, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calcula scores para múltiples contenidos"""
        scored_contents = []
        
        for content_item in contents:
            content = content_item.get('content', '')
            context = content_item.get('context', {})
            
            score = self.calculate_score(query, content, context)
            
            scored_item = content_item.copy()
            scored_item['relevance_score'] = score
            scored_item['scoring_timestamp'] = time.time()
            
            scored_contents.append(scored_item)
        
        # Ordenar por score descendente
        return sorted(scored_contents, key=lambda x: x['relevance_score'], reverse=True)
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Estadísticas del sistema de scoring"""
        total = self.performance_metrics['total_scores']
        
        return {
            'total_scores_calculated': total,
            'average_score': round(self.performance_metrics['avg_score'], 3),
            'high_quality_percentage': round((self.performance_metrics['high_scores'] / total * 100), 2) if total > 0 else 0,
            'low_quality_percentage': round((self.performance_metrics['low_scores'] / total * 100), 2) if total > 0 else 0,
            'feedback_integration_enabled': self.feedback_integration,
            'scoring_distribution': {
                'excellent': self.performance_metrics['high_scores'],
                'poor': self.performance_metrics['low_scores'],
                'average': total - self.performance_metrics['high_scores'] - self.performance_metrics['low_scores']
            }
        }
    
    def calibrate_scoring(self, feedback_data: List[Dict[str, Any]]) -> None:
        """Calibra el sistema de scoring basado en feedback"""
        if not feedback_data:
            return
        
        # Analizar feedback para ajustar pesos
        high_rated = [item for item in feedback_data if item.get('user_rating', 0) >= 4]
        low_rated = [item for item in feedback_data if item.get('user_rating', 0) <= 2]
        
        # Ajustes básicos basados en patrones de feedback
        if len(high_rated) > len(low_rated):
            # Si hay más contenido bien valorado, ser más permisivo
            pass  # Mantener configuración actual
        else:
            # Si hay más contenido mal valorado, ser más estricto
            pass  # Ajustar pesos si es necesario
    
    def reset_metrics(self) -> None:
        """Reinicia métricas de rendimiento"""
        self.performance_metrics = {
            'total_scores': 0,
            'avg_score': 0.0,
            'high_scores': 0,
            'low_scores': 0
        }
        self.scoring_history.clear()

# Instancia global del scorer
advanced_scorer = AdvancedScorer()

def get_scorer_instance() -> AdvancedScorer:
    """Obtiene instancia global del advanced scorer"""
    return advanced_scorer
