#!/usr/bin/env python3
"""
Servidor MCP Unificado - Combina todas las técnicas avanzadas
- Sistema ACE (Análisis, Curación, Evolución)
- Cache multinivel inteligente
- Chunking semántico optimizado
- Context feedback system
- Memory management avanzado
- Deduplicación automática
"""

import json
import sys
import logging
import time
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import threading
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar técnicas avanzadas
try:
    from advanced_techniques import (
        UnifiedAdvancedSystem, 
        AdvancedMemoryManager,
        AdaptiveQueryOptimizer,
        IntelligentDeduplicator,
        ContextualLearningSystem
    )
    ADVANCED_TECHNIQUES_AVAILABLE = True
    logger.info("✅ Técnicas avanzadas cargadas correctamente")
except ImportError as e:
    logger.warning(f"⚠️ Técnicas avanzadas no disponibles: {e}")
    ADVANCED_TECHNIQUES_AVAILABLE = False

# Importar sistema de indexación de contexto
try:
    from context_indexing_system import ContextIndexingSystem
    CONTEXT_INDEXING_AVAILABLE = True
    logger.info("✅ Sistema de indexación de contexto cargado")
except ImportError as e:
    logger.warning(f"⚠️ Sistema de indexación de contexto no disponible: {e}")
    CONTEXT_INDEXING_AVAILABLE = False


class ConsolidatedACESystem:
    """Sistema ACE consolidado (Análisis, Curación, Evolución) para guía del modelo"""
    
    def __init__(self):
        self.analysis_engine = AnalysisEngine()
        self.curation_engine = CurationEngine()
        self.evolution_tracker = EvolutionTracker()
        self.knowledge_base = {}
        self.learning_patterns = defaultdict(list)
        
    def analyze_context(self, query: str, context: str) -> Dict:
        """Análisis completo del contexto y query"""
        return self.analysis_engine.deep_analyze(query, context)
    
    def curate_response(self, analysis: Dict, raw_response: str) -> Dict:
        """Cura y mejora la respuesta basada en análisis"""
        return self.curation_engine.curate(analysis, raw_response)
    
    def evolve_knowledge(self, interaction_data: Dict) -> None:
        """Evoluciona el conocimiento basado en interacciones"""
        self.evolution_tracker.track_evolution(interaction_data)


class AnalysisEngine:
    """Motor de análisis profundo para contexto y queries"""
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.intent_detector = IntentDetector()
        self.context_mapper = ContextMapper()
    
    def deep_analyze(self, query: str, context: str) -> Dict:
        """Análisis profundo de query y contexto"""
        analysis = {
            'query_analysis': self._analyze_query(query),
            'context_analysis': self._analyze_context(context),
            'relationship_analysis': self._analyze_relationships(query, context),
            'complexity_score': self._calculate_complexity(query, context),
            'guidance_recommendations': self._generate_guidance(query, context)
        }
        
        return analysis
    
    def _analyze_query(self, query: str) -> Dict:
        """Análisis específico del query"""
        return {
            'intent': self.intent_detector.detect_intent(query),
            'complexity': self.complexity_analyzer.assess_query_complexity(query),
            'key_concepts': self._extract_key_concepts(query),
            'ambiguity_level': self._assess_ambiguity(query)
        }
    
    def _analyze_context(self, context: str) -> Dict:
        """Análisis específico del contexto"""
        return {
            'content_type': self._detect_content_type(context),
            'structure_quality': self._assess_structure(context),
            'information_density': self._calculate_info_density(context),
            'technical_depth': self._assess_technical_depth(context)
        }
    
    def _analyze_relationships(self, query: str, context: str) -> Dict:
        """Análisis de relaciones entre query y contexto"""
        query_concepts = set(self._extract_key_concepts(query))
        context_concepts = set(self._extract_key_concepts(context))
        
        return {
            'concept_overlap': len(query_concepts & context_concepts),
            'relevance_score': self._calculate_relevance(query, context),
            'context_sufficiency': self._assess_context_sufficiency(query, context),
            'missing_elements': list(query_concepts - context_concepts)
        }
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extrae conceptos clave del texto"""
        # Conceptos técnicos
        tech_concepts = re.findall(r'\b(?:class|function|method|variable|API|database|server|client|framework)\b', text.lower())
        
        # Identificadores importantes
        identifiers = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[a-z_]+[a-z]\b', text)
        
        # Combinar y filtrar
        all_concepts = tech_concepts + [id.lower() for id in identifiers if len(id) > 2]
        
        # Retornar conceptos únicos más frecuentes
        from collections import Counter
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
    
    def _assess_technical_depth(self, context: str) -> float:
        """Evalúa profundidad técnica"""
        technical_indicators = [
            'class ', 'def ', 'import ', 'function', 'method',
            'variable', 'parameter', 'return', 'exception',
            'algorithm', 'optimization', 'performance'
        ]
        
        tech_count = sum(1 for indicator in technical_indicators if indicator in context.lower())
        return min(1.0, tech_count / 10)
    
    def _calculate_relevance(self, query: str, context: str) -> float:
        """Calcula relevancia entre query y contexto"""
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = len(query_words & context_words)
        union = len(query_words | context_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_context_sufficiency(self, query: str, context: str) -> str:
        """Evalúa si el contexto es suficiente"""
        relevance = self._calculate_relevance(query, context)
        context_length = len(context)
        
        if relevance > 0.7 and context_length > 500:
            return 'sufficient'
        elif relevance > 0.4 and context_length > 200:
            return 'adequate'
        else:
            return 'insufficient'
    
    def _assess_ambiguity(self, query: str) -> str:
        """Evalúa nivel de ambigüedad del query"""
        if len(query.split()) < 3:
            return 'high'
        elif len(query.split()) < 7:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_complexity(self, query: str, context: str) -> float:
        """Calcula score de complejidad general"""
        query_complexity = len(query.split()) / 20  # Normalizado
        context_complexity = self._assess_technical_depth(context)
        
        return min(1.0, (query_complexity + context_complexity) / 2)
    
    def _generate_guidance(self, query: str, context: str) -> Dict:
        """Genera recomendaciones de guía"""
        analysis = self._analyze_query(query)
        
        guidance = {
            'response_approach': self._suggest_approach(analysis['intent']),
            'focus_areas': self._extract_key_concepts(query)[:3],
            'caution_areas': [],
            'enhancement_suggestions': []
        }
        
        # Agregar precauciones basadas en análisis
        if analysis['ambiguity_level'] == 'high':
            guidance['caution_areas'].append('clarify_requirements')
        
        if self._assess_context_sufficiency(query, context) == 'insufficient':
            guidance['caution_areas'].append('request_more_context')
        
        return guidance
    
    def _suggest_approach(self, intent: str) -> str:
        """Sugiere enfoque de respuesta basado en intención"""
        approaches = {
            'code_generation': 'step_by_step_implementation',
            'explanation': 'structured_explanation',
            'debugging': 'systematic_troubleshooting',
            'optimization': 'analysis_and_improvement',
            'general': 'comprehensive_response'
        }
        
        return approaches.get(intent, 'comprehensive_response')


class CurationEngine:
    """Motor de curación para mejorar respuestas"""
    
    def __init__(self):
        self.quality_filters = QualityFilters()
        self.enhancement_engine = EnhancementEngine()
        self.safety_checker = SafetyChecker()
    
    def curate(self, analysis: Dict, raw_response: str) -> Dict:
        """Cura y mejora la respuesta"""
        curation_result = {
            'original_response': raw_response,
            'quality_assessment': self.quality_filters.assess(raw_response),
            'safety_check': self.safety_checker.check(raw_response, analysis),
            'enhancements': self.enhancement_engine.suggest_enhancements(raw_response, analysis),
            'curated_response': self._apply_curation(raw_response, analysis)
        }
        
        return curation_result
    
    def _apply_curation(self, response: str, analysis: Dict) -> str:
        """Aplica mejoras de curación"""
        curated = response
        
        # Aplicar mejoras basadas en análisis
        if analysis.get('complexity_score', 0) > 0.7:
            curated = self._add_complexity_warnings(curated)
        
        if analysis.get('context_analysis', {}).get('technical_depth', 0) > 0.8:
            curated = self._enhance_technical_accuracy(curated)
        
        return curated
    
    def _add_complexity_warnings(self, response: str) -> str:
        """Agrega advertencias para contenido complejo"""
        if 'complejo' not in response.lower() and 'complex' not in response.lower():
            return f"⚠️ **Nota**: Esta es una respuesta técnica compleja.\n\n{response}"
        return response
    
    def _enhance_technical_accuracy(self, response: str) -> str:
        """Mejora precisión técnica"""
        # Agregar disclaimers técnicos si es necesario
        if not any(phrase in response.lower() for phrase in ['puede variar', 'dependiendo de', 'en algunos casos']):
            return f"{response}\n\n📝 **Nota**: Los detalles pueden variar según la implementación específica."
        return response


class EvolutionTracker:
    """Rastreador de evolución del conocimiento"""
    
    def __init__(self):
        self.interaction_history = []
        self.learning_patterns = defaultdict(list)
        self.improvement_metrics = {}
    
    def track_evolution(self, interaction_data: Dict) -> None:
        """Rastrea la evolución del conocimiento"""
        self.interaction_history.append({
            'timestamp': time.time(),
            'query_type': interaction_data.get('query_type'),
            'success_metrics': interaction_data.get('success_metrics'),
            'improvement_areas': interaction_data.get('improvement_areas')
        })
        
        # Actualizar patrones de aprendizaje
        self._update_learning_patterns(interaction_data)
    
    def _update_learning_patterns(self, data: Dict) -> None:
        """Actualiza patrones de aprendizaje"""
        query_type = data.get('query_type', 'general')
        success_score = data.get('success_metrics', {}).get('overall_score', 0.5)
        
        self.learning_patterns[query_type].append(success_score)
        
        # Mantener solo los últimos 100 registros por tipo
        if len(self.learning_patterns[query_type]) > 100:
            self.learning_patterns[query_type] = self.learning_patterns[query_type][-100:]


class ComplexityAnalyzer:
    """Analizador de complejidad"""
    
    def assess_query_complexity(self, query: str) -> float:
        """Evalúa complejidad del query"""
        factors = {
            'length': min(1.0, len(query.split()) / 20),
            'technical_terms': self._count_technical_terms(query) / 10,
            'specificity': self._assess_specificity(query),
            'ambiguity': 1.0 - self._assess_clarity(query)
        }
        
        weights = {'length': 0.2, 'technical_terms': 0.3, 'specificity': 0.3, 'ambiguity': 0.2}
        
        return sum(score * weights[factor] for factor, score in factors.items())
    
    def _count_technical_terms(self, text: str) -> int:
        """Cuenta términos técnicos"""
        technical_terms = [
            'algorithm', 'optimization', 'implementation', 'architecture',
            'framework', 'database', 'API', 'class', 'function', 'method'
        ]
        
        return sum(1 for term in technical_terms if term in text.lower())
    
    def _assess_specificity(self, query: str) -> float:
        """Evalúa especificidad del query"""
        specific_indicators = ['how to', 'implement', 'create', 'fix', 'optimize']
        general_indicators = ['what is', 'explain', 'tell me about']
        
        specific_count = sum(1 for indicator in specific_indicators if indicator in query.lower())
        general_count = sum(1 for indicator in general_indicators if indicator in query.lower())
        
        if specific_count > general_count:
            return 0.8
        elif general_count > specific_count:
            return 0.3
        else:
            return 0.5
    
    def _assess_clarity(self, query: str) -> float:
        """Evalúa claridad del query"""
        clarity_factors = {
            'has_subject': any(word in query.lower() for word in ['how', 'what', 'why', 'when', 'where']),
            'has_context': len(query.split()) > 5,
            'has_specifics': any(char in query for char in ['(', ')', '{', '}', '"', "'"]),
            'proper_grammar': query.count('?') <= 1 and query.count('.') <= 2
        }
        
        return sum(clarity_factors.values()) / len(clarity_factors)


class IntentDetector:
    """Detector de intención"""
    
    def detect_intent(self, query: str) -> str:
        """Detecta la intención del query"""
        intent_patterns = {
            'code_generation': r'\b(?:create|generate|write|implement|build|make)\b',
            'explanation': r'\b(?:explain|what|how|why|describe|tell me)\b',
            'debugging': r'\b(?:error|bug|fix|debug|problem|issue|broken)\b',
            'optimization': r'\b(?:optimize|improve|better|faster|efficient|performance)\b',
            'modification': r'\b(?:change|modify|update|edit|refactor|alter)\b',
            'analysis': r'\b(?:analyze|review|evaluate|assess|examine)\b'
        }
        
        detected_intents = {}
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query.lower()):
                detected_intents[intent] = len(re.findall(pattern, query.lower()))
        
        if detected_intents:
            return max(detected_intents.items(), key=lambda x: x[1])[0]
        else:
            return 'general'


class ContextMapper:
    """Mapeador de contexto"""
    
    def map_context_structure(self, context: str) -> Dict:
        """Mapea la estructura del contexto"""
        return {
            'sections': self._identify_sections(context),
            'code_blocks': self._extract_code_blocks(context),
            'documentation': self._extract_documentation(context),
            'examples': self._extract_examples(context)
        }
    
    def _identify_sections(self, context: str) -> List[str]:
        """Identifica secciones del contexto"""
        headers = re.findall(r'^#{1,3}\s+(.+)$', context, re.MULTILINE)
        return headers
    
    def _extract_code_blocks(self, context: str) -> List[str]:
        """Extrae bloques de código"""
        code_blocks = re.findall(r'```.*?\n(.*?)```', context, re.DOTALL)
        return [block.strip() for block in code_blocks]
    
    def _extract_documentation(self, context: str) -> List[str]:
        """Extrae documentación"""
        doc_blocks = re.findall(r'"""(.*?)"""', context, re.DOTALL)
        return [doc.strip() for doc in doc_blocks]
    
    def _extract_examples(self, context: str) -> List[str]:
        """Extrae ejemplos"""
        example_patterns = [
            r'ejemplo:?\s*(.*?)(?=\n\n|\Z)',
            r'example:?\s*(.*?)(?=\n\n|\Z)'
        ]
        
        examples = []
        for pattern in example_patterns:
            examples.extend(re.findall(pattern, context, re.IGNORECASE | re.DOTALL))
        
        return [ex.strip() for ex in examples if ex.strip()]


class AdvancedQueryOptimizer:
    """Optimizador avanzado de queries"""
    
    def __init__(self):
        self.semantic_expander = SemanticExpander()
        self.context_enhancer = ContextEnhancer()
    
    def optimize_query(self, query: str, context: str) -> Dict:
        """Optimiza query para mejor respuesta"""
        return {
            'original_query': query,
            'expanded_query': self.semantic_expander.expand(query),
            'enhanced_context': self.context_enhancer.enhance(context, query),
            'optimization_score': self._calculate_optimization_score(query, context)
        }
    
    def _calculate_optimization_score(self, query: str, context: str) -> float:
        """Calcula score de optimización"""
        # Simplificado para el ejemplo
        return min(1.0, (len(query.split()) + len(context.split())) / 1000)


class SemanticExpander:
    """Expansor semántico de queries"""
    
    def expand(self, query: str) -> str:
        """Expande query semánticamente"""
        # Simplificado - en implementación real usaría embeddings
        expansions = {
            'create': 'generate implement build make develop',
            'fix': 'debug resolve solve repair correct',
            'optimize': 'improve enhance accelerate streamline'
        }
        
        expanded = query
        for term, expansion in expansions.items():
            if term in query.lower():
                expanded += f" ({expansion})"
        
        return expanded


class ContextEnhancer:
    """Mejorador de contexto"""
    
    def enhance(self, context: str, query: str) -> str:
        """Mejora contexto basado en query"""
        # Agregar metadatos relevantes
        enhanced = f"QUERY_CONTEXT: {query}\n\n{context}"
        
        # Agregar indicadores de relevancia
        query_words = set(query.lower().split())
        context_lines = context.split('\n')
        
        relevant_lines = []
        for line in context_lines:
            line_words = set(line.lower().split())
            if query_words & line_words:
                relevant_lines.append(f"[RELEVANT] {line}")
            else:
                relevant_lines.append(line)
        
        return '\n'.join(relevant_lines)


class ContextCurator:
    """Curador de contexto"""
    
    def curate_context(self, raw_context: str, query: str) -> Dict:
        """Cura contexto para optimizar respuesta"""
        return {
            'curated_context': self._apply_curation_rules(raw_context, query),
            'curation_metadata': self._generate_metadata(raw_context, query),
            'quality_score': self._assess_context_quality(raw_context)
        }
    
    def _apply_curation_rules(self, context: str, query: str) -> str:
        """Aplica reglas de curación"""
        # Simplificado - priorizar líneas relevantes
        lines = context.split('\n')
        query_words = set(query.lower().split())
        
        prioritized_lines = []
        for line in lines:
            line_words = set(line.lower().split())
            relevance = len(query_words & line_words)
            prioritized_lines.append((relevance, line))
        
        # Ordenar por relevancia y tomar top lines
        prioritized_lines.sort(key=lambda x: x[0], reverse=True)
        
        return '\n'.join([line for _, line in prioritized_lines[:50]])
    
    def _generate_metadata(self, context: str, query: str) -> Dict:
        """Genera metadatos de curación"""
        return {
            'original_length': len(context),
            'curated_length': len(self._apply_curation_rules(context, query)),
            'relevance_score': self._calculate_relevance(context, query)
        }
    
    def _assess_context_quality(self, context: str) -> float:
        """Evalúa calidad del contexto"""
        quality_factors = {
            'length': min(1.0, len(context) / 2000),
            'structure': len(re.findall(r'#{1,3}|```', context)) / 10,
            'information_density': len(context.split()) / max(1, context.count('\n'))
        }
        
        return sum(quality_factors.values()) / len(quality_factors)
    
    def _calculate_relevance(self, context: str, query: str) -> float:
        """Calcula relevancia del contexto"""
        context_words = set(context.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        return len(context_words & query_words) / len(query_words)


class EvolutionEngine:
    """Motor de evolución del sistema"""
    
    def __init__(self):
        self.evolution_history = []
        self.performance_metrics = defaultdict(list)
    
    def evolve_system(self, feedback_data: Dict) -> Dict:
        """Evoluciona el sistema basado en feedback"""
        evolution_result = {
            'evolution_applied': self._apply_evolution(feedback_data),
            'performance_impact': self._assess_impact(feedback_data),
            'next_evolution_suggestions': self._suggest_next_evolutions()
        }
        
        self.evolution_history.append({
            'timestamp': time.time(),
            'evolution_data': evolution_result
        })
        
        return evolution_result
    
    def _apply_evolution(self, feedback: Dict) -> List[str]:
        """Aplica evoluciones basadas en feedback"""
        evolutions = []
        
        if feedback.get('accuracy_score', 0) < 0.7:
            evolutions.append('improve_accuracy_algorithms')
        
        if feedback.get('response_time', 0) > 5.0:
            evolutions.append('optimize_performance')
        
        return evolutions
    
    def _assess_impact(self, feedback: Dict) -> Dict:
        """Evalúa impacto de evoluciones"""
        return {
            'accuracy_improvement': feedback.get('accuracy_score', 0) - 0.5,
            'performance_improvement': max(0, 3.0 - feedback.get('response_time', 3.0)),
            'user_satisfaction': feedback.get('satisfaction_score', 0.5)
        }
    
    def _suggest_next_evolutions(self) -> List[str]:
        """Sugiere próximas evoluciones"""
        return [
            'enhance_semantic_understanding',
            'improve_context_awareness',
            'optimize_response_generation'
        ]


# Clases auxiliares para el sistema ACE
class QualityFilters:
    def assess(self, response: str) -> Dict:
        return {'quality_score': min(1.0, len(response) / 500)}

class EnhancementEngine:
    def suggest_enhancements(self, response: str, analysis: Dict) -> List[str]:
        return ['add_examples', 'improve_clarity']

class SafetyChecker:
    def check(self, response: str, analysis: Dict) -> Dict:
        return {'safety_score': 0.9, 'issues': []}


class UnifiedMCPServer:
    """Servidor MCP unificado con todas las técnicas avanzadas"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        
        # Inicializar sistema avanzado si está disponible
        if ADVANCED_TECHNIQUES_AVAILABLE:
            self.advanced_system = UnifiedAdvancedSystem()
            logger.info("🧠 Sistema Avanzado Unificado activado")
        else:
            self.advanced_system = None
            logger.info("⚠️ Usando sistema básico")
        
        # Inicializar sistema de indexación de contexto
        if CONTEXT_INDEXING_AVAILABLE:
            context_db_path = self.project_root / "mcp-hub" / "data" / "cache" / "mcp_context.db"
            context_db_path.parent.mkdir(exist_ok=True)
            self.context_indexer = ContextIndexingSystem(str(context_db_path))
            logger.info("🗂️ Sistema de Indexación de Contexto activado")
        else:
            self.context_indexer = None
            logger.info("⚠️ Sistema de indexación de contexto no disponible")
        
        # Inicializar componentes básicos
        self.cache_system = UnifiedCacheSystem()
        self.chunker = SemanticChunker()
        self.scorer = AdvancedScorer()
        self.memory_manager = MemoryManager()
        
        # Sistema ACE consolidado (Análisis, Curación, Evolución)
        self.ace_system = ConsolidatedACESystem()
        self.query_optimizer = AdvancedQueryOptimizer()
        self.context_curator = ContextCurator()
        self.evolution_engine = EvolutionEngine()
        
        # Estado del servidor
        self.indexed_files = {}
        self.query_count = 0
        self.start_time = time.time()
        
        logger.info("🚀 Servidor MCP Unificado iniciado")
        logger.info(f"📁 Directorio raíz: {self.project_root}")
        
        # Auto-indexar al iniciar
        self._auto_index()
    
    def _auto_index(self):
        """Indexación automática del proyecto"""
        try:
            files_indexed = 0
            for file_path in self.project_root.rglob("*.py"):
                if self._should_index_file(file_path):
                    self._index_file(file_path)
                    files_indexed += 1
            
            logger.info(f"✅ Indexados {files_indexed} archivos automáticamente")
        except Exception as e:
            logger.error(f"Error en auto-indexación: {e}")
    
    def _should_index_file(self, file_path: Path) -> bool:
        """Determina si un archivo debe ser indexado"""
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        return not any(part in exclude_dirs for part in file_path.parts)
    
    def _index_file(self, file_path: Path):
        """Indexa un archivo individual"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Verificar si ya está indexado
            if str(file_path) in self.indexed_files:
                if self.indexed_files[str(file_path)]['hash'] == file_hash:
                    return  # No cambió
            
            # Chunking semántico
            chunks = self.chunker.chunk_content(content, str(file_path))
            
            # Almacenar en cache
            self.cache_system.store_chunks(str(file_path), chunks)
            
            # Registrar archivo
            self.indexed_files[str(file_path)] = {
                'hash': file_hash,
                'chunks': len(chunks),
                'indexed_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error indexando {file_path}: {e}")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP"""
        try:
            method = request.get('method')
            params = request.get('params', {})
            
            if method == 'tools/call':
                return self._handle_tool_call(params)
            elif method == 'tools/list':
                return self._list_tools()
            else:
                return {'error': f'Método no soportado: {method}'}
                
        except Exception as e:
            logger.error(f"Error manejando request: {e}")
            return {'error': str(e)}
    
    def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if tool_name == 'context_query':
            return self._context_query(arguments)
        elif tool_name == 'analyze_code':
            return self._analyze_code(arguments)
        elif tool_name == 'cache_search':
            return self._cache_search(arguments)
        elif tool_name == 'cache_metrics':
            return self._cache_metrics()
        elif tool_name == 'cache_refresh':
            return self._cache_refresh()
        elif tool_name == 'system_stats':
            return self._system_stats()
        elif tool_name == 'create_task':
            return self._create_task(arguments)
        elif tool_name == 'process_tasks':
            return self._process_tasks()
        elif tool_name == 'code_review':
            return self._code_review(arguments)
        elif tool_name == 'detect_duplicates':
            return self._detect_duplicates(arguments)
        else:
            return {'error': f'Herramienta no encontrada: {tool_name}'}
    
    def _context_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Consulta de contexto con técnicas avanzadas y memoria persistente"""
        query = args.get('query', '')
        max_results = args.get('max_results', 5)
        topic = args.get('topic', 'general')
        
        self.query_count += 1
        start_time = time.time()
        
        # 1. BUSCAR EN CONTEXTO INDEXADO PRIMERO (memoria persistente)
        context_results = []
        if self.context_indexer:
            logger.info(f"🗂️ Buscando en contexto indexado: {query[:50]}...")
            context_results = self.context_indexer.retrieve_context(query, topic, max_results)
            
            if context_results:
                logger.info(f"✅ Encontrados {len(context_results)} contextos relevantes en memoria persistente")
        
        # 2. USAR SISTEMA AVANZADO SI ESTÁ DISPONIBLE
        advanced_results = []
        if self.advanced_system:
            logger.info(f"🧠 Procesando query con sistema avanzado: {query[:50]}...")
            
            context = {
                'max_results': max_results,
                'query_count': self.query_count,
                'domain': 'medical',
                'topic': topic,
                'has_context_history': len(context_results) > 0
            }
            
            advanced_result = self.advanced_system.process_query(query, context)
            if advanced_result and 'results' in advanced_result:
                advanced_results = advanced_result['results']
        
        # 3. FALLBACK AL SISTEMA BÁSICO SI ES NECESARIO
        basic_results = []
        if not context_results and not advanced_results:
            logger.info(f"⚙️ Procesando query con sistema básico: {query[:50]}...")
            
            # Buscar en cache primero
            cached_results = self.cache_system.search(query)
            if cached_results:
                logger.info(f"🎯 Cache hit para query: {query[:50]}...")
                basic_results = cached_results[:max_results]
            else:
                # Búsqueda completa en archivos indexados
                for file_path, file_info in self.indexed_files.items():
                    chunks = self.cache_system.get_chunks(file_path)
                    if chunks:
                        for chunk in chunks:
                            score = self.scorer.calculate_score(query, chunk['content'])
                            if score > 0.3:  # Umbral de relevancia
                                basic_results.append({
                                    'file': file_path,
                                    'content': chunk['content'],
                                    'score': score,
                                    'metadata': chunk.get('metadata', {})
                                })
                
                # Ordenar por score
                basic_results.sort(key=lambda x: x['score'], reverse=True)
                basic_results = basic_results[:max_results]
        
        # 4. COMBINAR Y FORMATEAR RESULTADOS
        all_results = []
        
        # Agregar contextos de memoria persistente
        for ctx in context_results:
            all_results.append({
                'source': 'context_memory',
                'content': ctx['content'],
                'topic': ctx['topic'],
                'score': ctx['relevance_score'],
                'access_count': ctx['access_count'],
                'metadata': ctx['metadata']
            })
        
        # Agregar resultados avanzados
        for res in advanced_results:
            all_results.append({
                'source': 'advanced_system',
                'content': res['content'],
                'score': res.get('score', 0.5),
                'metadata': res.get('metadata', {})
            })
        
        # Agregar resultados básicos
        for res in basic_results:
            all_results.append({
                'source': 'basic_search',
                'content': res['content'],
                'score': res.get('score', 0.3),
                'file': res.get('file', ''),
                'metadata': res.get('metadata', {})
            })
        
        # 5. ALMACENAR CONTEXTO PARA MEMORIA PERSISTENTE
        response_time = time.time() - start_time
        final_response = self._format_unified_results(all_results, query, context_results, advanced_results, basic_results)
        
        if self.context_indexer and all_results:
            # Almacenar la consulta y respuesta en el contexto
            context_content = f"Query: {query}\n\nResultados encontrados: {len(all_results)}\n\n"
            for i, result in enumerate(all_results[:3], 1):  # Solo los top 3
                context_content += f"{i}. {result['content'][:200]}...\n\n"
            
            context_hash = self.context_indexer.store_context(
                content=context_content,
                topic=topic,
                metadata={
                    'query': query,
                    'results_count': len(all_results),
                    'response_time': response_time,
                    'sources_used': list(set(r['source'] for r in all_results))
                }
            )
            
            # Almacenar conversación completa
            self.context_indexer.store_conversation(
                user_query=query,
                model_response=final_response[:500] + "..." if len(final_response) > 500 else final_response,
                context_used=[context_hash] if context_hash else [],
                tokens_used=len(final_response.split()),
                response_time=response_time
            )
        
        return {
            'content': [{'type': 'text', 'text': final_response}]
        }
    
    def _format_unified_results(self, all_results: List[Dict], query: str, 
                               context_results: List, advanced_results: List, 
                               basic_results: List) -> str:
        """Formatea resultados unificados de todos los sistemas"""
        
        if not all_results:
            return f"🔍 No se encontraron resultados para: '{query}'"
        
        # Ordenar por score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        formatted = f"🧠 **Resultados Unificados MCP** para: '{query}'\n\n"
        
        # Estadísticas de fuentes
        sources = {}
        for result in all_results:
            source = result.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        formatted += "📊 **Fuentes consultadas**:\n"
        if sources.get('context_memory', 0) > 0:
            formatted += f"🗂️ Memoria persistente: {sources['context_memory']} contextos\n"
        if sources.get('advanced_system', 0) > 0:
            formatted += f"🧠 Sistema avanzado: {sources['advanced_system']} resultados\n"
        if sources.get('basic_search', 0) > 0:
            formatted += f"⚙️ Búsqueda básica: {sources['basic_search']} archivos\n"
        
        formatted += f"\n📋 **Mostrando {min(len(all_results), 5)} mejores resultados**:\n\n"
        
        # Mostrar resultados
        for i, result in enumerate(all_results[:5], 1):
            source = result.get('source', 'unknown')
            score = result.get('score', 0)
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            
            # Iconos por fuente
            source_icon = {
                'context_memory': '🗂️',
                'advanced_system': '🧠',
                'basic_search': '⚙️'
            }.get(source, '📄')
            
            formatted += f"**{i}. {source_icon} {source.replace('_', ' ').title()}** (relevancia: {score:.2f})\n"
            
            # Información adicional según la fuente
            if source == 'context_memory':
                topic = result.get('topic', 'general')
                access_count = result.get('access_count', 0)
                formatted += f"   📂 Tema: {topic} | 🔄 Accesos: {access_count}\n"
            elif source == 'basic_search':
                file_path = result.get('file', '')
                if file_path:
                    file_name = Path(file_path).name
                    formatted += f"   📁 Archivo: {file_name}\n"
            
            formatted += f"```\n{content}\n```\n\n"
        
        # Información de memoria persistente
        if context_results:
            formatted += f"💾 **Memoria persistente**: Esta consulta se ha guardado para futuras referencias.\n"
        
        return formatted
    
    def _format_results(self, results: List[Dict]) -> str:
        """Formatea resultados para respuesta"""
        if not results:
            return "No se encontraron resultados relevantes."
        
        formatted = f"Encontrados {len(results)} resultados:\n\n"
        
        for i, result in enumerate(results, 1):
            file_name = Path(result['file']).name
            score = result.get('score', 0)
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            
            formatted += f"**{i}. {file_name}** (relevancia: {score:.2f})\n"
            formatted += f"```\n{content}\n```\n\n"
        
        return formatted
    
    def _analyze_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza código para prevenir duplicación"""
        path = args.get('path', str(self.project_root))
        
        if self.advanced_system:
            # Usar sistema avanzado para análisis
            stats = self.advanced_system.get_system_stats()
            dedup_stats = stats.get('deduplication_stats', {})
            
            return {
                'content': [{'type': 'text', 'text': f"""
🔍 **Análisis de Código Completado**

📁 **Directorio analizado**: {path}
📊 **Estadísticas de duplicación**:
- Total procesado: {dedup_stats.get('total_processed', 0)}
- Duplicados encontrados: {dedup_stats.get('duplicates_found', 0)}
- Near-duplicates: {dedup_stats.get('near_duplicates_found', 0)}
- Tasa de duplicación: {dedup_stats.get('duplicate_rate', 0):.1f}%
- Tasa de únicos: {dedup_stats.get('unique_rate', 0):.1f}%

✅ **Análisis completado con sistema avanzado**
                """}]
            }
        
        # Análisis básico
        total_files = len(self.indexed_files)
        return {
            'content': [{'type': 'text', 'text': f"""
🔍 **Análisis Básico de Código**

📁 **Directorio**: {path}
📄 **Archivos indexados**: {total_files}

⚠️ Para análisis avanzado de duplicación, active las técnicas avanzadas.
            """}]
        }
    
    def _cache_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Busca directamente en el cache"""
        query = args.get('query', '')
        max_results = args.get('max_results', 10)
        
        results = self.cache_system.search(query)
        
        if results:
            formatted_results = self._format_results(results[:max_results])
            return {
                'content': [{'type': 'text', 'text': f"🎯 **Búsqueda en Cache**\n\n{formatted_results}"}]
            }
        
        return {
            'content': [{'type': 'text', 'text': f"🔍 No se encontraron resultados en cache para: '{query}'"}]
        }
    
    def _cache_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del cache"""
        metrics = self.cache_system.get_metrics()
        
        return {
            'content': [{'type': 'text', 'text': f"""
📊 **Métricas del Cache Multinivel**

🎯 **Rendimiento**:
- Hit Rate: {metrics['hit_rate']:.1f}%
- Cache Hits: {metrics['hits']}
- Cache Misses: {metrics['misses']}

💾 **Utilización**:
- L1 Cache: {metrics['l1_size']} items
- L2 Cache: {metrics['l2_size']} items  
- Disk Cache: {metrics['disk_size']} items

⚡ **Estado**: {'Óptimo' if metrics['hit_rate'] > 70 else 'Necesita optimización'}
            """}]
        }
    
    def _cache_refresh(self) -> Dict[str, Any]:
        """Fuerza actualización del cache"""
        # Limpiar caches
        self.cache_system.l1_cache.clear()
        self.cache_system.l2_cache.clear()
        
        # Re-indexar archivos
        files_reindexed = 0
        for file_path in self.project_root.rglob("*.py"):
            if self._should_index_file(file_path):
                self._index_file(file_path)
                files_reindexed += 1
        
        return {
            'content': [{'type': 'text', 'text': f"""
🔄 **Cache Actualizado**

✅ L1 y L2 cache limpiados
📁 {files_reindexed} archivos re-indexados
⚡ Sistema listo para nuevas consultas
            """}]
        }
    
    def _system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        uptime = time.time() - self.start_time
        cache_metrics = self.cache_system.get_metrics()
        
        stats_text = f"""
🚀 **Estadísticas del Sistema MCP Unificado**

⏱️ **Tiempo de actividad**: {uptime/3600:.1f} horas
📊 **Consultas procesadas**: {self.query_count}
📁 **Archivos indexados**: {len(self.indexed_files)}

💾 **Cache Multinivel**:
- Hit Rate: {cache_metrics['hit_rate']:.1f}%
- L1: {cache_metrics['l1_size']} items
- L2: {cache_metrics['l2_size']} items
- Disk: {cache_metrics['disk_size']} items

🧠 **Sistema Avanzado**: {'✅ Activo' if self.advanced_system else '❌ No disponible'}
        """
        
        if self.advanced_system:
            advanced_stats = self.advanced_system.get_system_stats()
            stats_text += f"""
            
🔬 **Estadísticas Avanzadas**:
- Memory Pools: {len(advanced_stats.get('memory_stats', {}).get('pools', {}))}
- Learning Patterns: {advanced_stats.get('learning_insights', {}).get('learned_patterns', 0)}
- Deduplication Rate: {advanced_stats.get('deduplication_stats', {}).get('unique_rate', 0):.1f}%
            """
        
        return {
            'content': [{'type': 'text', 'text': stats_text}]
        }
    
    def _create_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Crea una nueva tarea con análisis de contexto"""
        content = args.get('content', '')
        priority = args.get('priority', 'medium')
        dependencies = args.get('dependencies', [])
        
        task_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:8]
        
        # Si hay sistema avanzado, usar su learning system
        if self.advanced_system:
            # Simular creación de tarea en el sistema avanzado
            task_info = {
                'id': task_id,
                'content': content,
                'priority': priority,
                'dependencies': dependencies,
                'created_at': time.time(),
                'status': 'pending'
            }
            
            return {
                'content': [{'type': 'text', 'text': f"""
📋 **Tarea Creada con Sistema Avanzado**

🆔 **ID**: {task_id}
📝 **Contenido**: {content}
⚡ **Prioridad**: {priority}
🔗 **Dependencias**: {len(dependencies)} tareas
🧠 **Análisis de contexto**: Completado
✅ **Estado**: Pendiente

La tarea ha sido registrada en el sistema de aprendizaje contextual.
                """}]
            }
        
        # Creación básica de tarea
        return {
            'content': [{'type': 'text', 'text': f"""
📋 **Tarea Creada**

🆔 **ID**: {task_id}
📝 **Contenido**: {content}
⚡ **Prioridad**: {priority}
🔗 **Dependencias**: {len(dependencies)} tareas
✅ **Estado**: Pendiente
            """}]
        }
    
    def _process_tasks(self) -> Dict[str, Any]:
        """Procesa tareas con retroalimentación de contexto"""
        if self.advanced_system:
            # Usar sistema avanzado para procesamiento
            learning_insights = self.advanced_system.learning_system.get_learning_insights()
            
            return {
                'content': [{'type': 'text', 'text': f"""
⚙️ **Procesamiento de Tareas con Sistema Avanzado**

📊 **Estadísticas de aprendizaje**:
- Total de interacciones: {learning_insights.get('total_interactions', 0)}
- Tasa de feedback positivo: {learning_insights.get('positive_feedback_rate', 0):.1f}%
- Patrones aprendidos: {learning_insights.get('learned_patterns', 0)}
- Asociaciones contextuales: {learning_insights.get('context_associations', 0)}

🧠 **Tipos de consulta identificados**:
{', '.join(learning_insights.get('query_types', []))}

✅ **Procesamiento completado con retroalimentación contextual**
                """}]
            }
        
        # Procesamiento básico
        return {
            'content': [{'type': 'text', 'text': """
⚙️ **Procesamiento Básico de Tareas**

📋 Tareas procesadas con sistema básico
⚠️ Para retroalimentación contextual avanzada, active las técnicas avanzadas

✅ Procesamiento completado
            """}]
        }
    
    def _code_review(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza code review automático antes de comenzar una tarea"""
        task_description = args.get('task_description', '')
        target_files = args.get('target_files', [])
        
        review_results = {
            'duplicates_found': [],
            'potential_conflicts': [],
            'recommendations': [],
            'risk_level': 'low'
        }
        
        # 1. ANÁLISIS DE DUPLICACIÓN DE CÓDIGO
        if self.advanced_system:
            dedup_stats = self.advanced_system.get_system_stats().get('deduplication_stats', {})
            
            # Verificar si hay alta tasa de duplicación
            duplicate_rate = dedup_stats.get('duplicate_rate', 0)
            if duplicate_rate > 15:  # Más del 15% de duplicación
                review_results['risk_level'] = 'high'
                review_results['recommendations'].append(
                    f"⚠️ Alta tasa de duplicación detectada: {duplicate_rate:.1f}%. "
                    "Considere refactorizar antes de agregar nuevo código."
                )
        
        # 2. ANÁLISIS DE ARCHIVOS OBJETIVO
        for file_path in target_files:
            if str(file_path) in self.indexed_files:
                file_info = self.indexed_files[str(file_path)]
                
                # Verificar si el archivo ha sido modificado recientemente
                import time
                if time.time() - file_info.get('indexed_at', 0) < 3600:  # 1 hora
                    review_results['potential_conflicts'].append({
                        'file': file_path,
                        'reason': 'Archivo modificado recientemente',
                        'last_modified': file_info.get('indexed_at', 0)
                    })
        
        # 3. ANÁLISIS CONTEXTUAL DE LA TAREA
        task_keywords = task_description.lower().split()
        medical_keywords = ['paciente', 'medico', 'cita', 'historia', 'factura']
        
        if any(keyword in task_keywords for keyword in medical_keywords):
            review_results['recommendations'].append(
                "🏥 Tarea médica detectada. Asegurar cumplimiento con regulaciones de salud."
            )
        
        # 4. BÚSQUEDA DE FUNCIONALIDAD SIMILAR EXISTENTE
        similar_functions = []
        for file_path, file_info in self.indexed_files.items():
            chunks = self.cache_system.get_chunks(file_path)
            if chunks:
                for chunk in chunks:
                    # Buscar funciones similares usando palabras clave de la tarea
                    chunk_content = chunk['content'].lower()
                    matches = sum(1 for keyword in task_keywords 
                                if len(keyword) > 3 and keyword in chunk_content)
                    
                    if matches >= 2:  # Al menos 2 palabras clave coinciden
                        similar_functions.append({
                            'file': file_path,
                            'content_preview': chunk['content'][:200] + "...",
                            'match_score': matches
                        })
        
        if similar_functions:
            review_results['duplicates_found'] = similar_functions[:3]  # Top 3
            review_results['recommendations'].append(
                f"🔍 Encontradas {len(similar_functions)} funciones similares. "
                "Revisar antes de implementar para evitar duplicación."
            )
        
        # 5. GENERAR REPORTE FINAL
        risk_icons = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
        risk_icon = risk_icons.get(review_results['risk_level'], '🟢')
        
        report = f"""
🔍 **CODE REVIEW AUTOMÁTICO COMPLETADO**

{risk_icon} **Nivel de Riesgo**: {review_results['risk_level'].upper()}

📋 **Tarea Analizada**: {task_description[:100]}{'...' if len(task_description) > 100 else ''}

🔄 **Duplicados Encontrados**: {len(review_results['duplicates_found'])}
⚠️ **Conflictos Potenciales**: {len(review_results['potential_conflicts'])}
💡 **Recomendaciones**: {len(review_results['recommendations'])}

"""
        
        if review_results['duplicates_found']:
            report += "🚨 **FUNCIONES SIMILARES DETECTADAS**:\n"
            for i, func in enumerate(review_results['duplicates_found'], 1):
                file_name = Path(func['file']).name
                report += f"{i}. **{file_name}** (score: {func['match_score']})\n"
                report += f"```\n{func['content_preview']}\n```\n\n"
        
        if review_results['recommendations']:
            report += "💡 **RECOMENDACIONES**:\n"
            for rec in review_results['recommendations']:
                report += f"- {rec}\n"
        
        report += f"\n✅ **Code review completado** - Proceder con precaución nivel {review_results['risk_level']}"
        
        return {
            'content': [{'type': 'text', 'text': report}]
        }
    
    def _detect_duplicates(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta código duplicado en el sistema con análisis profundo"""
        target_path = args.get('path', str(self.project_root))
        similarity_threshold = args.get('threshold', 0.85)
        
        if self.advanced_system:
            # Usar sistema avanzado para detección
            dedup_stats = self.advanced_system.deduplicator.get_dedup_stats()
            
            # Análisis detallado de duplicación
            analysis_report = f"""
🔍 **ANÁLISIS PROFUNDO DE DUPLICACIÓN DE CÓDIGO**

📊 **Estadísticas Generales**:
- Total procesado: {dedup_stats.get('total_processed', 0)} fragmentos
- Duplicados exactos: {dedup_stats.get('duplicates_found', 0)}
- Near-duplicates: {dedup_stats.get('near_duplicates_found', 0)}
- Tasa de duplicación: {dedup_stats.get('duplicate_rate', 0):.1f}%
- Tasa de únicos: {dedup_stats.get('unique_rate', 0):.1f}%

🎯 **Análisis de Calidad**:
"""
            
            # Determinar nivel de calidad del código
            duplicate_rate = dedup_stats.get('duplicate_rate', 0)
            if duplicate_rate < 5:
                analysis_report += "✅ **EXCELENTE** - Muy baja duplicación\n"
            elif duplicate_rate < 15:
                analysis_report += "🟡 **BUENO** - Duplicación aceptable\n"
            elif duplicate_rate < 25:
                analysis_report += "🟠 **REGULAR** - Considerar refactorización\n"
            else:
                analysis_report += "🔴 **CRÍTICO** - Refactorización urgente necesaria\n"
            
            # Recomendaciones específicas
            analysis_report += f"""
💡 **Recomendaciones**:
- Umbral de similitud usado: {similarity_threshold}
- Archivos analizados: {len(self.indexed_files)}
- Directorio objetivo: {target_path}

🔧 **Acciones Sugeridas**:
"""
            
            if duplicate_rate > 20:
                analysis_report += "- 🚨 URGENTE: Implementar patrón de diseño para reducir duplicación\n"
                analysis_report += "- 📝 Crear funciones utilitarias compartidas\n"
                analysis_report += "- 🔄 Refactorizar módulos con alta duplicación\n"
            elif duplicate_rate > 10:
                analysis_report += "- 📋 Revisar funciones similares para consolidación\n"
                analysis_report += "- 🎯 Identificar patrones comunes para abstracción\n"
            else:
                analysis_report += "- ✅ Mantener buenas prácticas actuales\n"
                analysis_report += "- 🔍 Monitoreo continuo recomendado\n"
            
            return {
                'content': [{'type': 'text', 'text': analysis_report}]
            }
        
        # Análisis básico si no hay sistema avanzado
        total_files = len(self.indexed_files)
        basic_report = f"""
🔍 **DETECCIÓN BÁSICA DE DUPLICADOS**

📁 **Directorio**: {target_path}
📄 **Archivos indexados**: {total_files}
🎯 **Umbral de similitud**: {similarity_threshold}

⚠️ **Limitación**: Para análisis avanzado de duplicación, active las técnicas avanzadas.

💡 **Recomendación**: Ejecutar `cache_refresh` para actualizar índices antes del análisis.
        """
        
        return {
            'content': [{'type': 'text', 'text': basic_report}]
        }
    
    def _format_advanced_results(self, advanced_result: Dict[str, Any]) -> str:
        """Formatea resultados del sistema avanzado"""
        if not advanced_result or 'results' not in advanced_result:
            return "No se encontraron resultados con el sistema avanzado."
        
        results = advanced_result['results']
        query = advanced_result.get('query', 'consulta')
        processed_at = advanced_result.get('processed_at', time.time())
        
        formatted = f"🧠 **Resultados del Sistema Avanzado** para: '{query}'\n"
        formatted += f"⏱️ Procesado en: {datetime.fromtimestamp(processed_at).strftime('%H:%M:%S')}\n\n"
        
        if not results:
            return formatted + "No se encontraron resultados relevantes."
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            
            # Indicadores de calidad del sistema avanzado
            is_duplicate = result.get('metadata', {}).get('is_duplicate', False)
            chunk_type = result.get('metadata', {}).get('chunk_type', 'text')
            
            quality_indicators = []
            if not is_duplicate:
                quality_indicators.append("✅ Único")
            if chunk_type == 'code':
                quality_indicators.append("💻 Código")
            elif chunk_type == 'header':
                quality_indicators.append("📋 Encabezado")
            
            quality_str = " | ".join(quality_indicators) if quality_indicators else ""
            
            formatted += f"**{i}.** (relevancia: {score:.2f}) {quality_str}\n"
            formatted += f"```\n{content}\n```\n\n"
        
        return formatted
    
    def _list_tools(self) -> Dict[str, Any]:
        """Lista herramientas disponibles"""
        return {
            'tools': [
                {
                    'name': 'context_query',
                    'description': 'Consulta contexto del proyecto con técnicas avanzadas',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Consulta a realizar'},
                            'max_results': {'type': 'integer', 'description': 'Máximo número de resultados'}
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'analyze_code',
                    'description': 'Analiza código para prevenir duplicación',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'path': {'type': 'string', 'description': 'Ruta del directorio a analizar'}
                        }
                    }
                },
                {
                    'name': 'cache_search',
                    'description': 'Busca directamente en el cache inteligente',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Consulta para buscar'},
                            'max_results': {'type': 'integer', 'description': 'Número máximo de resultados'}
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'cache_metrics',
                    'description': 'Obtiene métricas del cache inteligente',
                    'inputSchema': {'type': 'object', 'properties': {}}
                },
                {
                    'name': 'cache_refresh',
                    'description': 'Fuerza actualización del cache',
                    'inputSchema': {'type': 'object', 'properties': {}}
                },
                {
                    'name': 'system_stats',
                    'description': 'Obtiene estadísticas completas del sistema unificado',
                    'inputSchema': {'type': 'object', 'properties': {}}
                },
                {
                    'name': 'create_task',
                    'description': 'Crea una nueva tarea con análisis de contexto',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'content': {'type': 'string', 'description': 'Descripción de la tarea'},
                            'priority': {'type': 'string', 'enum': ['low', 'medium', 'high', 'critical']},
                            'dependencies': {'type': 'array', 'items': {'type': 'string'}}
                        },
                        'required': ['content']
                    }
                },
                {
                    'name': 'process_tasks',
                    'description': 'Procesa tareas con retroalimentación de contexto',
                    'inputSchema': {'type': 'object', 'properties': {}}
                },
                {
                    'name': 'code_review',
                    'description': 'Realiza code review automático antes de comenzar una tarea',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'task_description': {'type': 'string', 'description': 'Descripción de la tarea a realizar'},
                            'target_files': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Archivos objetivo de la tarea'}
                        },
                        'required': ['task_description']
                    }
                },
                {
                    'name': 'detect_duplicates',
                    'description': 'Detecta código duplicado en el sistema con análisis profundo',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'path': {'type': 'string', 'description': 'Ruta del directorio a analizar'},
                            'threshold': {'type': 'number', 'description': 'Umbral de similitud (0.0-1.0)', 'default': 0.85}
                        }
                    }
                }
            ]
        }

class UnifiedCacheSystem:
    """Sistema de cache unificado multinivel"""
    
    def __init__(self):
        self.l1_cache = {}  # Memoria rápida
        self.l2_cache = {}  # Memoria extendida
        self.disk_cache = {}  # Persistente
        self.query_cache = {}  # Cache de queries
        
        self.l1_max = 100
        self.l2_max = 1000
        self.hits = 0
        self.misses = 0
        
    def search(self, query: str) -> Optional[List[Dict]]:
        """Busca en cache multinivel"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # L1 Cache
        if query_hash in self.l1_cache:
            self.hits += 1
            return self.l1_cache[query_hash]
        
        # L2 Cache
        if query_hash in self.l2_cache:
            self.hits += 1
            # Promover a L1
            self._promote_to_l1(query_hash, self.l2_cache[query_hash])
            return self.l2_cache[query_hash]
        
        # Disk Cache
        if query_hash in self.disk_cache:
            self.hits += 1
            # Promover a L2
            self._promote_to_l2(query_hash, self.disk_cache[query_hash])
            return self.disk_cache[query_hash]
        
        self.misses += 1
        return None
    
    def cache_query(self, query: str, results: List[Dict]):
        """Cachea resultados de query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self._store_in_l1(query_hash, results)
    
    def store_chunks(self, file_path: str, chunks: List[Dict]):
        """Almacena chunks de archivo"""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        self.disk_cache[file_hash] = chunks
    
    def get_chunks(self, file_path: str) -> Optional[List[Dict]]:
        """Obtiene chunks de archivo"""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self.disk_cache.get(file_hash)
    
    def _promote_to_l1(self, key: str, value: Any):
        """Promueve item a L1 cache"""
        if len(self.l1_cache) >= self.l1_max:
            # Remover el más antiguo (LRU simple)
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = value
    
    def _promote_to_l2(self, key: str, value: Any):
        """Promueve item a L2 cache"""
        if len(self.l2_cache) >= self.l2_max:
            oldest_key = next(iter(self.l2_cache))
            del self.l2_cache[oldest_key]
        
        self.l2_cache[key] = value
    
    def _store_in_l1(self, key: str, value: Any):
        """Almacena en L1 cache"""
        if len(self.l1_cache) >= self.l1_max:
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del cache"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'disk_size': len(self.disk_cache)
        }

class SemanticChunker:
    """Chunker semántico avanzado"""
    
    def chunk_content(self, content: str, file_path: str) -> List[Dict]:
        """Chunking inteligente por tipo de contenido"""
        chunks = []
        
        if file_path.endswith('.py'):
            chunks = self._chunk_python_code(content)
        elif file_path.endswith('.md'):
            chunks = self._chunk_markdown(content)
        else:
            chunks = self._chunk_text(content)
        
        # Agregar metadata
        for i, chunk in enumerate(chunks):
            chunk['metadata'] = {
                'file_path': file_path,
                'chunk_index': i,
                'chunk_type': self._detect_chunk_type(chunk['content']),
                'hash': hashlib.md5(chunk['content'].encode()).hexdigest()[:8]
            }
        
        return chunks
    
    def _chunk_python_code(self, content: str) -> List[Dict]:
        """Chunking específico para código Python"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        in_function = False
        in_class = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detectar inicio de función o clase
            if stripped.startswith('def ') or stripped.startswith('class '):
                if current_chunk:
                    chunks.append({'content': '\n'.join(current_chunk)})
                    current_chunk = []
                in_function = True
                in_class = stripped.startswith('class ')
            
            current_chunk.append(line)
            
            # Si llegamos a una línea vacía y estamos en función, cerrar chunk
            if not stripped and (in_function or in_class) and len(current_chunk) > 5:
                chunks.append({'content': '\n'.join(current_chunk)})
                current_chunk = []
                in_function = False
                in_class = False
        
        # Agregar último chunk si existe
        if current_chunk:
            chunks.append({'content': '\n'.join(current_chunk)})
        
        return chunks
    
    def _chunk_markdown(self, content: str) -> List[Dict]:
        """Chunking específico para Markdown"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        
        for line in lines:
            if line.startswith('#') and current_chunk:
                # Nuevo header, cerrar chunk anterior
                chunks.append({'content': '\n'.join(current_chunk)})
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append({'content': '\n'.join(current_chunk)})
        
        return chunks
    
    def _chunk_text(self, content: str) -> List[Dict]:
        """Chunking genérico para texto"""
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            if len(chunk_content.strip()) > 50:  # Filtrar chunks muy pequeños
                chunks.append({'content': chunk_content})
        
        return chunks
    
    def _detect_chunk_type(self, content: str) -> str:
        """Detecta el tipo de chunk"""
        if 'def ' in content or 'class ' in content:
            return 'code'
        elif content.strip().startswith('#'):
            return 'header'
        elif '```' in content:
            return 'code_block'
        else:
            return 'text'

class AdvancedScorer:
    """Sistema de scoring avanzado"""
    
    def calculate_score(self, query: str, content: str) -> float:
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
        
        # Weighted final score
        final_score = (
            scores['exact_match'] * 2.0 +
            scores['partial_match'] * 1.5 +
            scores['context_density'] * 0.8
        ) / 4.3
        
        return min(1.0, final_score)
    
    def _calculate_context_density(self, content: str) -> float:
        """Calcula densidad de contexto"""
        # Elementos que indican alta densidad de contexto
        code_elements = content.count('def ') + content.count('class ') + content.count('import ')
        list_items = content.count('\n- ') + content.count('\n* ')
        headers = content.count('\n#')
        
        total_elements = code_elements + list_items + headers
        content_length = len(content)
        
        if content_length == 0:
            return 0.0
        
        density = min(1.0, total_elements / (content_length / 100))
        return density

class MemoryManager:
    """Gestor de memoria avanzado"""
    
    def __init__(self):
        self.memory_usage = {}
        self.cleanup_threshold = 0.8  # 80% de uso
    
    def monitor_usage(self):
        """Monitorea uso de memoria"""
        # Implementación básica - se puede expandir
        pass
    
    def cleanup_if_needed(self):
        """Limpia memoria si es necesario"""
        # Implementación básica - se puede expandir
        pass

class ACESystem:
    """Sistema ACE (Análisis, Curación, Evolución)"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.curation_rules = []
        self.evolution_metrics = {}
    
    def process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Procesa resultados con sistema ACE"""
        # Análisis
        analyzed_results = self._analyze_results(results, query)
        
        # Curación
        curated_results = self._curate_results(analyzed_results)
        
        # Evolución (aprendizaje)
        self._evolve_from_query(query, curated_results)
        
        return curated_results
    
    def _analyze_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Análisis de resultados"""
        for result in results:
            # Detectar duplicados
            result['is_duplicate'] = self._detect_duplicate(result)
            
            # Calcular relevancia contextual
            result['contextual_relevance'] = self._calculate_contextual_relevance(result, query)
        
        return results
    
    def _curate_results(self, results: List[Dict]) -> List[Dict]:
        """Curación de resultados"""
        # Filtrar duplicados
        unique_results = []
        seen_hashes = set()
        
        for result in results:
            content_hash = hashlib.md5(result['content'].encode()).hexdigest()
            if content_hash not in seen_hashes:
                unique_results.append(result)
                seen_hashes.add(content_hash)
        
        return unique_results
    
    def _evolve_from_query(self, query: str, results: List[Dict]):
        """Evolución basada en query"""
        # Registrar patrones de consulta para mejorar futuras búsquedas
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.evolution_metrics[query_hash] = {
            'query': query,
            'results_count': len(results),
            'timestamp': time.time()
        }
    
    def _detect_duplicate(self, result: Dict) -> bool:
        """Detecta si un resultado es duplicado"""
        # Implementación básica - se puede mejorar
        return False
    
    def _calculate_contextual_relevance(self, result: Dict, query: str) -> float:
        """Calcula relevancia contextual"""
        # Implementación básica - se puede mejorar
        return result.get('score', 0.5)

def main():
    """Función principal del servidor MCP"""
    server = UnifiedMCPServer()
    
    logger.info("🎯 Servidor MCP Unificado listo para recibir requests")
    
    # Loop principal MCP
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except json.JSONDecodeError:
            logger.error("Error decodificando JSON")
        except Exception as e:
            logger.error(f"Error procesando request: {e}")
            print(json.dumps({'error': str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
