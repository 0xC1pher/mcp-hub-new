#!/usr/bin/env python3
"""
🎯 Token Budget Manager - Migrado desde legacy/optimized
Gestión inteligente de presupuesto de tokens con límites dinámicos y priorización
Integrado con cache inteligente
"""
import time
import math
import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TokenBudgetManager:
    """Gestión inteligente de presupuesto de tokens"""
    
    def __init__(self, max_tokens: int = 4000, reserved_tokens: int = 500):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens
        
        # Integración con cache inteligente
        self.cache_integration = True
        self.cache_hit_bonus = 0.1  # Bonus de prioridad para items en cache

    def estimate_tokens(self, text: str) -> int:
        """Estimación aproximada de tokens (1 token ≈ 4 caracteres)"""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def calculate_priority(self, section: Dict) -> float:
        """Calcula prioridad de una sección"""
        factors = {
            'relevance_score': section.get('relevance', 0),
            'recency': self.get_recency_score(section.get('last_updated')),
            'context_density': self.get_context_density(section.get('content', '')),
            'access_count': section.get('access_count', 0),
            'cache_status': self.get_cache_bonus(section)
        }
        
        weights = {
            'relevance_score': 0.35,
            'recency': 0.2,
            'context_density': 0.25,
            'access_count': 0.1,
            'cache_status': 0.1
        }
        
        return sum(score * weights[factor] for factor, score in factors.items())

    def get_recency_score(self, last_updated: Optional[float]) -> float:
        """Calcula score basado en recencia"""
        if not last_updated:
            return 0.5
        days_since_update = (time.time() - last_updated) / (24 * 3600)
        return math.exp(-days_since_update / 30)

    def get_context_density(self, content: str) -> float:
        """Calcula densidad de información del contexto"""
        if not content:
            return 0
        
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        lists = len(re.findall(r'^[-*+]\s', content, re.MULTILINE))
        
        density_score = min(1.0, (
            (word_count / 100) * 0.3 +
            (sentence_count / 10) * 0.2 +
            code_blocks * 0.3 +
            lists * 0.2
        ))
        
        return density_score

    def get_cache_bonus(self, section: Dict) -> float:
        """Bonus de prioridad para items en cache"""
        if not self.cache_integration:
            return 0
        
        # Si está en cache, dar bonus
        if section.get('in_cache', False):
            return self.cache_hit_bonus
        
        return 0

    def allocate_tokens(self, sections: List[Dict]) -> List[Dict]:
        """Asigna tokens disponibles a secciones priorizadas"""
        prioritized = sorted(sections, key=lambda x: self.calculate_priority(x), reverse=True)
        
        allocated = []
        remaining_tokens = self.available_tokens
        
        for section in prioritized:
            token_count = self.estimate_tokens(section.get('content', ''))
            if token_count <= remaining_tokens:
                allocated.append(section)
                remaining_tokens -= token_count
            else:
                # Truncar contenido si es necesario
                if remaining_tokens > 100:  # Mínimo 100 tokens
                    truncated_content = self.truncate_content(section.get('content', ''), remaining_tokens)
                    section['content'] = truncated_content
                    allocated.append(section)
                break
        
        return allocated

    def truncate_content(self, content: str, max_tokens: int) -> str:
        """Trunca contenido manteniendo estructura"""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        
        # Truncar por párrafos para mantener estructura
        paragraphs = content.split('\n\n')
        truncated = []
        current_length = 0
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) <= max_chars:
                truncated.append(paragraph)
                current_length += len(paragraph)
            else:
                break
        
        return '\n\n'.join(truncated) + '\n\n[... contenido truncado ...]'

    def optimize_for_cache(self, sections: List[Dict], cache_instance=None) -> List[Dict]:
        """Optimiza asignación de tokens considerando el cache"""
        if not cache_instance:
            return self.allocate_tokens(sections)
        
        # Marcar secciones que están en cache
        for section in sections:
            section_key = section.get('id') or section.get('content_hash')
            if section_key and cache_instance.get(section_key):
                section['in_cache'] = True
                section['cache_level'] = 'hit'
            else:
                section['in_cache'] = False
                section['cache_level'] = 'miss'
        
        return self.allocate_tokens(sections)

    def adjust_budget_dynamically(self, performance_metrics: Dict[str, Any]) -> None:
        """Ajusta presupuesto dinámicamente basado en métricas"""
        hit_rate = performance_metrics.get('cache_hit_rate', 0)
        response_time = performance_metrics.get('avg_response_time', 0)
        
        # Si el hit rate es alto, podemos ser más agresivos con tokens
        if hit_rate > 0.85:
            self.available_tokens = min(self.max_tokens - 300, self.available_tokens + 100)
        elif hit_rate < 0.70:
            self.available_tokens = max(self.max_tokens - 800, self.available_tokens - 100)
        
        # Si el tiempo de respuesta es alto, reducir tokens para acelerar
        if response_time > 1000:  # ms
            self.available_tokens = max(self.max_tokens - 1000, self.available_tokens - 200)

    def get_budget_stats(self) -> Dict[str, Any]:
        """Estadísticas del presupuesto de tokens"""
        return {
            'max_tokens': self.max_tokens,
            'reserved_tokens': self.reserved_tokens,
            'available_tokens': self.available_tokens,
            'utilization_percent': ((self.max_tokens - self.available_tokens) / self.max_tokens) * 100,
            'cache_integration': self.cache_integration,
            'cache_hit_bonus': self.cache_hit_bonus
        }

    def validate_allocation(self, allocated_sections: List[Dict]) -> bool:
        """Valida que la asignación no exceda el presupuesto"""
        total_tokens = sum(self.estimate_tokens(section.get('content', '')) for section in allocated_sections)
        return total_tokens <= self.available_tokens

# Instancia global del token manager
token_manager = TokenBudgetManager()

def get_token_manager() -> TokenBudgetManager:
    """Obtiene instancia global del token manager"""
    return token_manager
