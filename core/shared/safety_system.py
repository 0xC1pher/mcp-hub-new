#!/usr/bin/env python3
"""
🛡️ Safety System - Migrado desde legacy/enhanced
HallucinationDetector + ContextValidator para prevención de alucinaciones
Integrado con sistema de feedback
"""
import re
import time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """Detector de alucinaciones para prevenir respuestas incorrectas"""
    
    def __init__(self):
        self.known_facts = {}  # Base de conocimiento verificada
        self.suspicious_patterns = [
            r'definitivamente',
            r'siempre es',
            r'nunca falla',
            r'garantizado que',
            r'imposible que',
            r'100% seguro',
            r'absolutamente cierto',
            r'sin duda alguna'
        ]
        self.context_inconsistencies = []
        self.detection_stats = {
            'total_checks': 0,
            'risks_detected': 0,
            'false_positives': 0,
            'confirmed_hallucinations': 0
        }
    
    def detect_hallucination_risks(self, query: str, context: str, proposed_response: str = "") -> Dict:
        """Detecta riesgos de alucinación en la respuesta"""
        self.detection_stats['total_checks'] += 1
        
        risks = {
            'confidence_level': 'medium',
            'risk_factors': [],
            'recommendations': [],
            'context_gaps': [],
            'severity': 'low',
            'should_block': False
        }
        
        # Detectar patrones sospechosos
        suspicious_count = 0
        for pattern in self.suspicious_patterns:
            if re.search(pattern, proposed_response.lower()):
                risks['risk_factors'].append(f'absolute_statement: {pattern}')
                suspicious_count += 1
        
        # Detectar falta de contexto
        if len(context.strip()) < 100:
            risks['context_gaps'].append('insufficient_context')
            risks['confidence_level'] = 'low'
        
        # Detectar query ambigua
        if len(query.split()) < 3:
            risks['risk_factors'].append('ambiguous_query')
        
        # Detectar contradicciones con contexto conocido
        contradictions = self._detect_context_contradictions(proposed_response, context)
        if contradictions:
            risks['risk_factors'].extend(contradictions)
            risks['severity'] = 'high'
        
        # Detectar información no verificable
        unverifiable = self._detect_unverifiable_claims(proposed_response, context)
        if unverifiable:
            risks['risk_factors'].extend(unverifiable)
        
        # Evaluar severidad general
        total_risks = len(risks['risk_factors']) + len(risks['context_gaps'])
        if total_risks >= 3 or suspicious_count >= 2:
            risks['severity'] = 'high'
            risks['should_block'] = True
            self.detection_stats['risks_detected'] += 1
        elif total_risks >= 1:
            risks['severity'] = 'medium'
        
        # Generar recomendaciones
        if risks['risk_factors'] or risks['context_gaps']:
            risks['recommendations'] = self._generate_safety_recommendations(risks)
        
        return risks
    
    def _detect_context_contradictions(self, response: str, context: str) -> List[str]:
        """Detecta contradicciones con el contexto proporcionado"""
        contradictions = []
        
        # Buscar afirmaciones que contradigan el contexto
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Patrones de contradicción común
        contradiction_patterns = [
            (r'no existe', r'existe|implementado|disponible'),
            (r'imposible', r'posible|factible|viable'),
            (r'nunca', r'siempre|a veces|puede'),
            (r'no funciona', r'funciona|operativo|activo')
        ]
        
        for negative_pattern, positive_pattern in contradiction_patterns:
            if re.search(negative_pattern, response_lower) and re.search(positive_pattern, context_lower):
                contradictions.append(f'contradiction: {negative_pattern} vs context')
        
        return contradictions
    
    def _detect_unverifiable_claims(self, response: str, context: str) -> List[str]:
        """Detecta afirmaciones no verificables con el contexto"""
        unverifiable = []
        
        # Patrones de afirmaciones específicas que requieren verificación
        specific_claims = re.findall(r'(versión \d+\.\d+|desde \d{4}|exactamente \d+|precisamente)', response.lower())
        
        for claim in specific_claims:
            if claim not in context.lower():
                unverifiable.append(f'unverifiable_claim: {claim}')
        
        return unverifiable
    
    def _generate_safety_recommendations(self, risks: Dict) -> List[str]:
        """Genera recomendaciones para reducir riesgos"""
        recommendations = []
        
        if 'insufficient_context' in risks['context_gaps']:
            recommendations.append('request_more_context')
            recommendations.append('acknowledge_limitations')
        
        if any('absolute_statement' in rf for rf in risks['risk_factors']):
            recommendations.append('use_qualified_language')
            recommendations.append('provide_alternatives')
        
        if any('contradiction' in rf for rf in risks['risk_factors']):
            recommendations.append('verify_against_context')
            recommendations.append('resolve_contradictions')
        
        if any('unverifiable_claim' in rf for rf in risks['risk_factors']):
            recommendations.append('cite_sources')
            recommendations.append('qualify_uncertain_information')
        
        if risks['severity'] == 'high':
            recommendations.append('request_human_review')
        
        return recommendations
    
    def update_known_facts(self, facts: Dict[str, Any]) -> None:
        """Actualiza base de conocimiento verificada"""
        self.known_facts.update(facts)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Estadísticas del detector de alucinaciones"""
        total = self.detection_stats['total_checks']
        return {
            'total_checks': total,
            'risks_detected': self.detection_stats['risks_detected'],
            'risk_detection_rate': (self.detection_stats['risks_detected'] / total * 100) if total > 0 else 0,
            'false_positive_rate': (self.detection_stats['false_positives'] / total * 100) if total > 0 else 0,
            'confirmed_hallucinations': self.detection_stats['confirmed_hallucinations']
        }

class ContextValidator:
    """Validador de contexto para asegurar coherencia"""
    
    def __init__(self):
        self.context_history = []
        self.validation_rules = {
            'min_context_length': 50,
            'max_context_age': 3600,  # 1 hora
            'required_sections': ['description', 'examples'],
            'quality_threshold': 0.7
        }
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'avg_quality_score': 0.0
        }
    
    def validate_context_quality(self, context: str, metadata: Dict = None) -> Dict:
        """Valida la calidad del contexto proporcionado"""
        self.validation_stats['total_validations'] += 1
        
        validation = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'improvements': [],
            'confidence': 'medium',
            'validation_timestamp': time.time()
        }
        
        # Validar longitud mínima
        if len(context.strip()) < self.validation_rules['min_context_length']:
            validation['issues'].append('context_too_short')
            validation['is_valid'] = False
        
        # Validar estructura del contexto
        structure_score = self._assess_context_structure(context)
        validation['quality_score'] += structure_score * 0.4
        
        # Validar relevancia del contexto
        relevance_score = self._assess_context_relevance(context, metadata)
        validation['quality_score'] += relevance_score * 0.4
        
        # Validar completitud del contexto
        completeness_score = self._assess_context_completeness(context)
        validation['quality_score'] += completeness_score * 0.2
        
        # Determinar confianza
        if validation['quality_score'] >= 0.8:
            validation['confidence'] = 'high'
        elif validation['quality_score'] >= 0.6:
            validation['confidence'] = 'medium'
        else:
            validation['confidence'] = 'low'
        
        # Validar contra umbral de calidad
        if validation['quality_score'] < self.validation_rules['quality_threshold']:
            validation['is_valid'] = False
            self.validation_stats['failed_validations'] += 1
        else:
            self.validation_stats['passed_validations'] += 1
        
        # Generar mejoras sugeridas
        if validation['quality_score'] < 0.8:
            validation['improvements'] = self._suggest_context_improvements(context)
        
        # Actualizar estadísticas
        self._update_validation_stats(validation['quality_score'])
        
        return validation
    
    def _assess_context_structure(self, context: str) -> float:
        """Evalúa la estructura del contexto"""
        score = 0.0
        
        # Puntos por tener código estructurado
        if re.search(r'(class|def|function)', context):
            score += 0.3
        
        # Puntos por tener documentación
        if re.search(r'(""".*?"""|#.*)', context, re.DOTALL):
            score += 0.2
        
        # Puntos por tener ejemplos
        if 'example' in context.lower() or '```' in context:
            score += 0.3
        
        # Puntos por organización clara
        if re.search(r'(#{1,3}\s+|^\s*\d+\.)', context, re.MULTILINE):
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_context_relevance(self, context: str, metadata: Dict = None) -> float:
        """Evalúa la relevancia del contexto"""
        if not metadata:
            return 0.5
        
        query = metadata.get('query', '')
        if not query:
            return 0.5
        
        # Calcular overlapping de keywords
        context_words = set(re.findall(r'\w+', context.lower()))
        query_words = set(re.findall(r'\w+', query.lower()))
        
        if not query_words:
            return 0.5
        
        overlap = len(context_words & query_words) / len(query_words)
        
        # Bonus por términos técnicos específicos
        technical_bonus = self._get_technical_relevance_bonus(context, query)
        
        return min(1.0, (overlap * 1.5) + technical_bonus)
    
    def _assess_context_completeness(self, context: str) -> float:
        """Evalúa la completitud del contexto"""
        score = 0.0
        
        # Puntos por longitud adecuada
        if len(context) > 200:
            score += 0.3
        elif len(context) > 100:
            score += 0.2
        
        # Puntos por diversidad de información
        info_types = 0
        if re.search(r'(def|class|function)', context):
            info_types += 1
        if re.search(r'(example|ejemplo)', context.lower()):
            info_types += 1
        if re.search(r'(parameter|parámetro|argument)', context.lower()):
            info_types += 1
        if re.search(r'(return|devuelve|resultado)', context.lower()):
            info_types += 1
        
        score += min(0.4, info_types * 0.1)
        
        # Puntos por coherencia interna
        if not self._has_internal_contradictions(context):
            score += 0.3
        
        return min(1.0, score)
    
    def _get_technical_relevance_bonus(self, context: str, query: str) -> float:
        """Bonus por relevancia técnica específica"""
        technical_terms = {
            'mcp': 0.1, 'server': 0.1, 'cache': 0.1, 'token': 0.1,
            'optimization': 0.1, 'performance': 0.1, 'memory': 0.1
        }
        
        bonus = 0.0
        context_lower = context.lower()
        query_lower = query.lower()
        
        for term, value in technical_terms.items():
            if term in query_lower and term in context_lower:
                bonus += value
        
        return min(0.3, bonus)
    
    def _has_internal_contradictions(self, context: str) -> bool:
        """Detecta contradicciones internas en el contexto"""
        # Buscar patrones contradictorios simples
        context_lower = context.lower()
        
        contradictions = [
            ('no funciona', 'funciona correctamente'),
            ('no existe', 'está disponible'),
            ('imposible', 'es posible'),
            ('nunca', 'siempre')
        ]
        
        for neg, pos in contradictions:
            if neg in context_lower and pos in context_lower:
                return True
        
        return False
    
    def _suggest_context_improvements(self, context: str) -> List[str]:
        """Sugiere mejoras para el contexto"""
        improvements = []
        
        if len(context) < 200:
            improvements.append('add_more_detail')
        
        if not re.search(r'(example|ejemplo)', context.lower()):
            improvements.append('add_examples')
        
        if not re.search(r'(""".*?"""|#.*)', context, re.DOTALL):
            improvements.append('add_documentation')
        
        if not re.search(r'(parameter|parámetro)', context.lower()):
            improvements.append('add_parameter_info')
        
        if not re.search(r'(return|devuelve)', context.lower()):
            improvements.append('add_return_info')
        
        return improvements
    
    def _update_validation_stats(self, quality_score: float) -> None:
        """Actualiza estadísticas de validación"""
        total = self.validation_stats['total_validations']
        current_avg = self.validation_stats['avg_quality_score']
        
        # Actualizar promedio
        self.validation_stats['avg_quality_score'] = ((current_avg * (total - 1)) + quality_score) / total
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Estadísticas del validador de contexto"""
        total = self.validation_stats['total_validations']
        return {
            'total_validations': total,
            'passed_validations': self.validation_stats['passed_validations'],
            'failed_validations': self.validation_stats['failed_validations'],
            'pass_rate': (self.validation_stats['passed_validations'] / total * 100) if total > 0 else 0,
            'average_quality_score': round(self.validation_stats['avg_quality_score'], 3),
            'quality_threshold': self.validation_rules['quality_threshold']
        }

class ModelGuidanceEngine:
    """Motor de guía para el modelo para respuestas más precisas"""
    
    def __init__(self):
        self.guidance_templates = {
            'code_generation': {
                'structure': ['analysis', 'implementation', 'testing', 'documentation'],
                'avoid': ['hardcoded_values', 'missing_error_handling'],
                'include': ['type_hints', 'docstrings', 'examples']
            },
            'explanation': {
                'structure': ['overview', 'details', 'examples', 'summary'],
                'avoid': ['technical_jargon_without_explanation'],
                'include': ['step_by_step', 'visual_aids', 'analogies']
            },
            'debugging': {
                'structure': ['problem_identification', 'root_cause', 'solution', 'prevention'],
                'avoid': ['assumptions_without_verification'],
                'include': ['debugging_steps', 'testing_approach']
            }
        }
    
    def generate_guidance(self, query: str, context: str, intent: str = 'general') -> Dict:
        """Genera guías específicas para el modelo"""
        guidance = {
            'response_structure': [],
            'focus_areas': [],
            'avoid_patterns': [],
            'quality_checks': [],
            'context_preservation': {},
            'safety_guidelines': []
        }
        
        # Obtener template basado en intención
        template = self.guidance_templates.get(intent, {})
        
        if template:
            guidance['response_structure'] = template.get('structure', [])
            guidance['avoid_patterns'] = template.get('avoid', [])
            guidance['quality_checks'] = template.get('include', [])
        
        # Analizar áreas de enfoque específicas
        guidance['focus_areas'] = self._identify_focus_areas(query, context)
        
        # Reglas de preservación de contexto
        guidance['context_preservation'] = self._get_preservation_rules(context)
        
        # Directrices de seguridad
        guidance['safety_guidelines'] = self._get_safety_guidelines(query, context)
        
        return guidance
    
    def _identify_focus_areas(self, query: str, context: str) -> List[str]:
        """Identifica áreas de enfoque específicas"""
        focus_areas = []
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'cómo', 'como']):
            focus_areas.append('step_by_step_explanation')
        
        if any(word in query_lower for word in ['why', 'por qué', 'porque']):
            focus_areas.append('reasoning_and_rationale')
        
        if any(word in query_lower for word in ['error', 'bug', 'problema']):
            focus_areas.append('debugging_and_troubleshooting')
        
        if any(word in query_lower for word in ['optimize', 'improve', 'better']):
            focus_areas.append('optimization_and_improvement')
        
        return focus_areas
    
    def _get_preservation_rules(self, context: str) -> Dict[str, Any]:
        """Reglas para preservar información del contexto"""
        return {
            'maintain_technical_accuracy': True,
            'preserve_code_structure': '```' in context,
            'keep_examples': 'example' in context.lower(),
            'maintain_formatting': re.search(r'#{1,3}\s+', context) is not None
        }
    
    def _get_safety_guidelines(self, query: str, context: str) -> List[str]:
        """Directrices de seguridad específicas"""
        guidelines = [
            'verify_against_context',
            'avoid_absolute_statements',
            'acknowledge_limitations'
        ]
        
        if len(context) < 100:
            guidelines.append('request_more_context')
        
        if any(word in query.lower() for word in ['delete', 'remove', 'destroy']):
            guidelines.append('confirm_destructive_actions')
        
        return guidelines

class IntegratedSafetySystem:
    """Sistema de seguridad integrado"""
    
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.context_validator = ContextValidator()
        self.guidance_engine = ModelGuidanceEngine()
    
    def comprehensive_safety_check(self, query: str, context: str, proposed_response: str = "") -> Dict[str, Any]:
        """Verificación de seguridad completa"""
        # Validar contexto
        context_validation = self.context_validator.validate_context_quality(
            context, {'query': query}
        )
        
        # Detectar riesgos de alucinación
        hallucination_risks = self.hallucination_detector.detect_hallucination_risks(
            query, context, proposed_response
        )
        
        # Generar guía para el modelo
        model_guidance = self.guidance_engine.generate_guidance(query, context)
        
        return {
            'context_validation': context_validation,
            'hallucination_risks': hallucination_risks,
            'model_guidance': model_guidance,
            'overall_safety_score': self._calculate_overall_safety_score(
                context_validation, hallucination_risks
            ),
            'recommendations': self._consolidate_recommendations(
                context_validation, hallucination_risks, model_guidance
            )
        }
    
    def _calculate_overall_safety_score(self, context_val: Dict, halluc_risks: Dict) -> float:
        """Calcula score de seguridad general"""
        context_score = context_val.get('quality_score', 0.5)
        
        # Penalizar por riesgos de alucinación
        risk_penalty = len(halluc_risks.get('risk_factors', [])) * 0.1
        severity_penalty = 0.3 if halluc_risks.get('severity') == 'high' else 0.1
        
        safety_score = context_score - risk_penalty - severity_penalty
        return max(0.0, min(1.0, safety_score))
    
    def _consolidate_recommendations(self, context_val: Dict, halluc_risks: Dict, guidance: Dict) -> List[str]:
        """Consolida recomendaciones de todos los componentes"""
        all_recommendations = []
        
        all_recommendations.extend(context_val.get('improvements', []))
        all_recommendations.extend(halluc_risks.get('recommendations', []))
        all_recommendations.extend(guidance.get('safety_guidelines', []))
        
        # Eliminar duplicados manteniendo orden
        return list(dict.fromkeys(all_recommendations))
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Estadísticas del sistema de seguridad"""
        return {
            'hallucination_detection': self.hallucination_detector.get_detection_stats(),
            'context_validation': self.context_validator.get_validation_stats(),
            'system_status': 'active',
            'integration_health': 'good'
        }

# Instancia global del sistema de seguridad
safety_system = IntegratedSafetySystem()

def get_safety_system() -> IntegratedSafetySystem:
    """Obtiene instancia global del sistema de seguridad"""
    return safety_system
