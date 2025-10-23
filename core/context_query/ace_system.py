#!/usr/bin/env python3
"""
🧠 Sistema ACE (Análisis, Curación, Evolución) - Migrado desde legacy/unified
Sistema consolidado para guía del modelo con análisis profundo
Integrado con cache multinivel
"""
import re
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Motor de análisis profundo para contexto y queries"""
    
    def __init__(self):
        self.complexity_patterns = {
            'high': ['algorithm', 'optimization', 'performance', 'architecture'],
            'medium': ['function', 'method', 'class', 'implementation'],
            'low': ['variable', 'parameter', 'simple', 'basic']
        }
    
    def deep_analyze(self, query: str, context: str) -> Dict:
        """Análisis profundo de query y contexto"""
        analysis = {
            'query_analysis': self._analyze_query(query),
            'context_analysis': self._analyze_context(context),
            'relationship_analysis': self._analyze_relationships(query, context),
            'complexity_score': self._calculate_complexity(query, context),
            'guidance_recommendations': self._generate_guidance(query, context),
            'timestamp': time.time()
        }
        
        return analysis
    
    def _analyze_query(self, query: str) -> Dict:
        """Análisis específico del query"""
        return {
            'intent': self._detect_intent(query),
            'complexity': self._assess_query_complexity(query),
            'key_concepts': self._extract_key_concepts(query),
            'ambiguity_level': self._assess_ambiguity(query),
            'technical_depth': self._assess_technical_depth(query)
        }
    
    def _analyze_context(self, context: str) -> Dict:
        """Análisis específico del contexto"""
        return {
            'content_type': self._detect_content_type(context),
            'structure_quality': self._assess_structure(context),
            'information_density': self._calculate_info_density(context),
            'technical_depth': self._assess_technical_depth(context),
            'completeness': self._assess_completeness(context)
        }
    
    def _analyze_relationships(self, query: str, context: str) -> Dict:
        """Análisis de relaciones entre query y contexto"""
        query_concepts = set(self._extract_key_concepts(query))
        context_concepts = set(self._extract_key_concepts(context))
        
        return {
            'concept_overlap': len(query_concepts & context_concepts),
            'relevance_score': self._calculate_relevance(query, context),
            'context_sufficiency': self._assess_context_sufficiency(query, context),
            'missing_elements': list(query_concepts - context_concepts),
            'semantic_alignment': self._calculate_semantic_alignment(query, context)
        }
    
    def _detect_intent(self, query: str) -> str:
        """Detecta la intención del query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'cómo', 'como']):
            return 'explanation'
        elif any(word in query_lower for word in ['create', 'build', 'make', 'implement']):
            return 'creation'
        elif any(word in query_lower for word in ['fix', 'debug', 'error', 'problem']):
            return 'debugging'
        elif any(word in query_lower for word in ['optimize', 'improve', 'enhance']):
            return 'optimization'
        elif any(word in query_lower for word in ['what', 'qué', 'que']):
            return 'information'
        else:
            return 'general'
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extrae conceptos clave del texto"""
        # Conceptos técnicos
        tech_concepts = re.findall(r'\b(?:class|function|method|variable|API|database|server|client|framework|cache|token|mcp|optimization)\b', text.lower())
        
        # Identificadores CamelCase y snake_case
        identifiers = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[a-z_]+[a-z]\b', text)
        
        # Combinar y filtrar
        all_concepts = tech_concepts + [id.lower() for id in identifiers if len(id) > 2]
        
        # Retornar conceptos únicos más frecuentes
        concept_counts = Counter(all_concepts)
        return [concept for concept, _ in concept_counts.most_common(10)]
    
    def _detect_content_type(self, context: str) -> str:
        """Detecta el tipo de contenido"""
        if 'def ' in context and 'class ' in context:
            return 'python_code'
        elif '```' in context and '#' in context:
            return 'markdown_documentation'
        elif '{' in context and '"' in context:
            return 'json_configuration'
        elif 'function' in context and '(' in context:
            return 'code_general'
        else:
            return 'text_content'
    
    def _assess_structure(self, context: str) -> float:
        """Evalúa la calidad de la estructura"""
        structure_indicators = [
            len(re.findall(r'#{1,3}\s+', context)) * 0.2,  # Headers
            len(re.findall(r'```.*?```', context, re.DOTALL)) * 0.3,  # Code blocks
            len(re.findall(r'^\s*[-*+]\s+', context, re.MULTILINE)) * 0.1,  # Lists
            len(re.findall(r'^\s*\d+\.\s+', context, re.MULTILINE)) * 0.1  # Numbered lists
        ]
        
        return min(1.0, sum(structure_indicators))
    
    def _calculate_info_density(self, context: str) -> float:
        """Calcula densidad de información"""
        if not context:
            return 0.0
        
        words = len(context.split())
        lines = context.count('\n') + 1
        chars = len(context)
        
        # Normalizar métricas
        word_density = min(1.0, words / 500)
        line_density = min(1.0, lines / 50)
        char_density = min(1.0, chars / 2000)
        
        return (word_density + line_density + char_density) / 3
    
    def _assess_technical_depth(self, text: str) -> float:
        """Evalúa profundidad técnica"""
        technical_indicators = [
            'class ', 'def ', 'import ', 'function', 'method',
            'variable', 'parameter', 'return', 'exception',
            'algorithm', 'optimization', 'performance', 'cache',
            'token', 'mcp', 'server', 'client'
        ]
        
        tech_count = sum(1 for indicator in technical_indicators if indicator in text.lower())
        return min(1.0, tech_count / 15)
    
    def _calculate_complexity(self, query: str, context: str) -> float:
        """Calcula score de complejidad general"""
        query_complexity = self._assess_query_complexity(query)
        context_complexity = self._assess_technical_depth(context)
        
        return (query_complexity + context_complexity) / 2
    
    def _assess_query_complexity(self, query: str) -> float:
        """Evalúa complejidad del query"""
        complexity_score = 0.0
        query_lower = query.lower()
        
        # Complejidad por patrones
        for level, patterns in self.complexity_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in query_lower)
            if level == 'high':
                complexity_score += matches * 0.3
            elif level == 'medium':
                complexity_score += matches * 0.2
            else:
                complexity_score += matches * 0.1
        
        # Complejidad por longitud y estructura
        word_count = len(query.split())
        if word_count > 10:
            complexity_score += 0.2
        elif word_count > 5:
            complexity_score += 0.1
        
        return min(1.0, complexity_score)

class CurationEngine:
    """Motor de curación y mejora de respuestas"""
    
    def __init__(self):
        self.curation_rules = {
            'technical_accuracy': True,
            'completeness_check': True,
            'clarity_enhancement': True,
            'example_inclusion': True
        }
    
    def curate(self, analysis: Dict, raw_response: str) -> Dict:
        """Cura y mejora la respuesta basada en análisis"""
        curated = {
            'original_response': raw_response,
            'curated_response': raw_response,
            'improvements_applied': [],
            'quality_score': 0.0,
            'curation_metadata': {}
        }
        
        # Aplicar mejoras basadas en análisis
        if analysis.get('query_analysis', {}).get('intent') == 'explanation':
            curated = self._enhance_explanation(curated, analysis)
        
        if analysis.get('context_analysis', {}).get('content_type') == 'python_code':
            curated = self._enhance_code_response(curated, analysis)
        
        # Verificar completitud
        curated = self._ensure_completeness(curated, analysis)
        
        # Calcular score de calidad
        curated['quality_score'] = self._calculate_quality_score(curated, analysis)
        
        return curated
    
    def _enhance_explanation(self, curated: Dict, analysis: Dict) -> Dict:
        """Mejora respuestas de explicación"""
        response = curated['curated_response']
        
        # Agregar estructura si falta
        if not re.search(r'^\d+\.|\*\s+|-\s+', response, re.MULTILINE):
            curated['improvements_applied'].append('added_structure')
        
        # Sugerir ejemplos si faltan
        if 'example' not in response.lower() and '```' not in response:
            curated['improvements_applied'].append('suggested_examples')
        
        return curated
    
    def _enhance_code_response(self, curated: Dict, analysis: Dict) -> Dict:
        """Mejora respuestas relacionadas con código"""
        response = curated['curated_response']
        
        # Verificar que incluya código si es necesario
        if analysis.get('context_analysis', {}).get('content_type') == 'python_code':
            if '```' not in response:
                curated['improvements_applied'].append('suggested_code_examples')
        
        return curated
    
    def _ensure_completeness(self, curated: Dict, analysis: Dict) -> Dict:
        """Asegura completitud de la respuesta"""
        missing_elements = analysis.get('relationship_analysis', {}).get('missing_elements', [])
        
        if missing_elements:
            curated['improvements_applied'].append('address_missing_elements')
            curated['curation_metadata']['missing_elements'] = missing_elements
        
        return curated
    
    def _calculate_quality_score(self, curated: Dict, analysis: Dict) -> float:
        """Calcula score de calidad de la respuesta curada"""
        base_score = 0.7  # Score base
        
        # Bonus por mejoras aplicadas
        improvements_bonus = len(curated['improvements_applied']) * 0.05
        
        # Bonus por relevancia del análisis
        relevance_bonus = analysis.get('relationship_analysis', {}).get('relevance_score', 0) * 0.2
        
        # Penalty por elementos faltantes
        missing_penalty = len(analysis.get('relationship_analysis', {}).get('missing_elements', [])) * 0.05
        
        final_score = base_score + improvements_bonus + relevance_bonus - missing_penalty
        return min(1.0, max(0.0, final_score))

class EvolutionTracker:
    """Rastreador de evolución del conocimiento"""
    
    def __init__(self):
        self.interaction_history = []
        self.learning_patterns = defaultdict(list)
        self.evolution_metrics = {
            'total_interactions': 0,
            'successful_responses': 0,
            'improvement_trends': {},
            'knowledge_gaps': set()
        }
    
    def track_evolution(self, interaction_data: Dict) -> None:
        """Rastrea la evolución basada en interacciones"""
        self.interaction_history.append({
            'timestamp': time.time(),
            'query': interaction_data.get('query', ''),
            'context_quality': interaction_data.get('context_quality', 0),
            'response_quality': interaction_data.get('response_quality', 0),
            'user_feedback': interaction_data.get('user_feedback'),
            'improvements_applied': interaction_data.get('improvements_applied', [])
        })
        
        self.evolution_metrics['total_interactions'] += 1
        
        # Actualizar patrones de aprendizaje
        self._update_learning_patterns(interaction_data)
        
        # Identificar gaps de conocimiento
        self._identify_knowledge_gaps(interaction_data)
        
        # Limpiar historial si es muy largo
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-500:]
    
    def _update_learning_patterns(self, interaction_data: Dict) -> None:
        """Actualiza patrones de aprendizaje"""
        query_type = interaction_data.get('query_type', 'general')
        response_quality = interaction_data.get('response_quality', 0)
        
        self.learning_patterns[query_type].append(response_quality)
        
        # Mantener solo los últimos 50 registros por tipo
        if len(self.learning_patterns[query_type]) > 50:
            self.learning_patterns[query_type] = self.learning_patterns[query_type][-25:]
    
    def _identify_knowledge_gaps(self, interaction_data: Dict) -> None:
        """Identifica gaps de conocimiento"""
        if interaction_data.get('response_quality', 0) < 0.6:
            missing_elements = interaction_data.get('missing_elements', [])
            self.evolution_metrics['knowledge_gaps'].update(missing_elements)
    
    def get_evolution_insights(self) -> Dict[str, Any]:
        """Obtiene insights de evolución"""
        return {
            'total_interactions': self.evolution_metrics['total_interactions'],
            'learning_trends': self._calculate_learning_trends(),
            'knowledge_gaps': list(self.evolution_metrics['knowledge_gaps']),
            'improvement_suggestions': self._generate_improvement_suggestions()
        }
    
    def _calculate_learning_trends(self) -> Dict[str, float]:
        """Calcula tendencias de aprendizaje"""
        trends = {}
        
        for query_type, qualities in self.learning_patterns.items():
            if len(qualities) >= 5:
                recent_avg = sum(qualities[-5:]) / 5
                older_avg = sum(qualities[:-5]) / len(qualities[:-5]) if len(qualities) > 5 else recent_avg
                trends[query_type] = recent_avg - older_avg
        
        return trends
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Genera sugerencias de mejora"""
        suggestions = []
        
        # Sugerencias basadas en gaps de conocimiento
        if len(self.evolution_metrics['knowledge_gaps']) > 5:
            suggestions.append('expand_knowledge_base')
        
        # Sugerencias basadas en tendencias
        trends = self._calculate_learning_trends()
        declining_areas = [area for area, trend in trends.items() if trend < -0.1]
        
        if declining_areas:
            suggestions.append(f'improve_handling_of_{declining_areas[0]}')
        
        return suggestions

class ConsolidatedACESystem:
    """Sistema ACE consolidado (Análisis, Curación, Evolución) para guía del modelo"""
    
    def __init__(self):
        self.analysis_engine = AnalysisEngine()
        self.curation_engine = CurationEngine()
        self.evolution_tracker = EvolutionTracker()
        self.knowledge_base = {}
        self.learning_patterns = defaultdict(list)
        self.cache_integration = True
    
    def process_query(self, query: str, context: str, cache_instance=None) -> Dict[str, Any]:
        """Procesamiento completo ACE de un query"""
        start_time = time.time()
        
        # ANÁLISIS
        analysis = self.analysis_engine.deep_analyze(query, context)
        
        # Integración con cache si está disponible
        if self.cache_integration and cache_instance:
            analysis['cache_status'] = self._check_cache_status(query, context, cache_instance)
        
        # CURACIÓN (simulada - en implementación real vendría después de la respuesta del modelo)
        curation_preview = {
            'recommended_structure': analysis['guidance_recommendations'],
            'quality_expectations': analysis['complexity_score'],
            'completeness_requirements': analysis['relationship_analysis']['missing_elements']
        }
        
        # EVOLUCIÓN
        interaction_data = {
            'query': query,
            'query_type': analysis['query_analysis']['intent'],
            'context_quality': analysis['context_analysis']['structure_quality'],
            'missing_elements': analysis['relationship_analysis']['missing_elements']
        }
        
        processing_time = time.time() - start_time
        
        return {
            'analysis': analysis,
            'curation_preview': curation_preview,
            'evolution_data': interaction_data,
            'processing_time_ms': processing_time * 1000,
            'recommendations': self._generate_comprehensive_recommendations(analysis)
        }
    
    def post_process_response(self, ace_result: Dict, model_response: str, user_feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """Post-procesamiento después de recibir respuesta del modelo"""
        # CURACIÓN de la respuesta real
        curated = self.curation_engine.curate(ace_result['analysis'], model_response)
        
        # EVOLUCIÓN con feedback
        if user_feedback:
            ace_result['evolution_data']['user_feedback'] = user_feedback
            ace_result['evolution_data']['response_quality'] = curated['quality_score']
        
        self.evolution_tracker.track_evolution(ace_result['evolution_data'])
        
        return {
            'curated_response': curated,
            'evolution_insights': self.evolution_tracker.get_evolution_insights(),
            'final_quality_score': curated['quality_score']
        }
    
    def _check_cache_status(self, query: str, context: str, cache_instance) -> Dict[str, Any]:
        """Verifica estado del cache para optimización"""
        query_hash = hash(query + context)
        
        return {
            'query_in_cache': cache_instance.get(str(query_hash)) is not None,
            'cache_hit_rate': getattr(cache_instance, 'get_comprehensive_stats', lambda: {})().get('overall', {}).get('overall_hit_rate_percent', 0)
        }
    
    def _generate_comprehensive_recommendations(self, analysis: Dict) -> List[str]:
        """Genera recomendaciones comprehensivas"""
        recommendations = []
        
        # Recomendaciones basadas en complejidad
        if analysis['complexity_score'] > 0.7:
            recommendations.append('provide_detailed_explanation')
            recommendations.append('include_step_by_step_approach')
        
        # Recomendaciones basadas en contexto
        if analysis['context_analysis']['structure_quality'] < 0.5:
            recommendations.append('request_better_context')
        
        # Recomendaciones basadas en elementos faltantes
        if analysis['relationship_analysis']['missing_elements']:
            recommendations.append('address_missing_concepts')
        
        # Recomendaciones basadas en intención
        intent = analysis['query_analysis']['intent']
        if intent == 'debugging':
            recommendations.extend(['provide_debugging_steps', 'include_testing_approach'])
        elif intent == 'creation':
            recommendations.extend(['provide_implementation_details', 'include_examples'])
        
        return recommendations
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Estadísticas del sistema ACE"""
        return {
            'analysis_engine': 'active',
            'curation_engine': 'active', 
            'evolution_tracker': self.evolution_tracker.get_evolution_insights(),
            'cache_integration': self.cache_integration,
            'knowledge_base_size': len(self.knowledge_base)
        }

# Instancia global del sistema ACE
ace_system = ConsolidatedACESystem()

def get_ace_system() -> ConsolidatedACESystem:
    """Obtiene instancia global del sistema ACE"""
    return ace_system
