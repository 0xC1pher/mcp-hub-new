"""
Query Expansion Automática - Sistema de expansión y reformulación de queries
Implementa técnicas avanzadas de expansión semántica, reformulación contextual y optimización de queries
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict, Counter
import hashlib


class ExpansionStrategy(Enum):
    SEMANTIC = "semantic"           # Expansión semántica basada en embeddings
    SYNONYMS = "synonyms"          # Expansión por sinónimos
    CONTEXTUAL = "contextual"      # Expansión contextual
    STATISTICAL = "statistical"    # Expansión estadística (co-ocurrencia)
    HIERARCHICAL = "hierarchical"  # Expansión jerárquica (hiperónimos/hipónimos)
    DOMAIN_SPECIFIC = "domain_specific"  # Expansión específica de dominio


class QueryType(Enum):
    FACTUAL = "factual"           # Preguntas factuales
    PROCEDURAL = "procedural"     # Preguntas de "cómo hacer"
    CONCEPTUAL = "conceptual"     # Preguntas conceptuales
    COMPARATIVE = "comparative"   # Preguntas comparativas
    EXPLORATORY = "exploratory"   # Exploración abierta


@dataclass
class ExpandedTerm:
    original_term: str
    expanded_term: str
    expansion_type: ExpansionStrategy
    confidence: float
    weight: float
    context: Optional[str] = None


@dataclass
class QueryExpansion:
    original_query: str
    expanded_queries: List[str]
    expanded_terms: List[ExpandedTerm]
    query_type: QueryType
    expansion_strategies: List[ExpansionStrategy]
    confidence_score: float
    processing_time: float


class SemanticExpander:
    """Expansor semántico basado en embeddings y similitud"""

    def __init__(self):
        # Simulamos un diccionario de embeddings para términos comunes
        self.term_embeddings = self._build_semantic_dictionary()
        self.similarity_threshold = 0.7

    def _build_semantic_dictionary(self) -> Dict[str, np.ndarray]:
        """Construye diccionario semántico (simulado)"""
        terms = {
            # Términos técnicos
            'algoritmo': ['método', 'procedimiento', 'proceso', 'técnica'],
            'programa': ['software', 'aplicación', 'código', 'sistema'],
            'datos': ['información', 'registros', 'contenido', 'elementos'],
            'modelo': ['representación', 'esquema', 'patrón', 'estructura'],
            'optimización': ['mejora', 'perfeccionamiento', 'refinamiento'],
            'análisis': ['examen', 'estudio', 'investigación', 'evaluación'],

            # Términos de ML/AI
            'machine learning': ['aprendizaje automático', 'ml', 'aprendizaje de máquina'],
            'inteligencia artificial': ['ia', 'ai', 'sistemas inteligentes'],
            'red neuronal': ['neural network', 'nn', 'redes neuronales'],
            'deep learning': ['aprendizaje profundo', 'dl', 'redes profundas'],

            # Términos de programación
            'función': ['method', 'procedimiento', 'rutina', 'subrutina'],
            'variable': ['parámetro', 'atributo', 'campo', 'propiedad'],
            'clase': ['objeto', 'tipo', 'estructura', 'entidad'],
            'bucle': ['loop', 'iteración', 'ciclo', 'repetición'],
        }

        embeddings = {}
        for term, related in terms.items():
            # Simular embedding basado en hash del término
            np.random.seed(hash(term) % 2**32)
            embeddings[term] = np.random.normal(0, 1, 100).astype(np.float32)

            # Embeddings para términos relacionados (similares pero con ruido)
            for related_term in related:
                np.random.seed(hash(related_term) % 2**32)
                base_embedding = embeddings[term] + np.random.normal(0, 0.3, 100)
                embeddings[related_term] = base_embedding.astype(np.float32)

        return embeddings

    def expand_term(self, term: str, max_expansions: int = 3) -> List[ExpandedTerm]:
        """Expande un término usando similitud semántica"""
        term_lower = term.lower()
        expansions = []

        if term_lower not in self.term_embeddings:
            return expansions

        term_embedding = self.term_embeddings[term_lower]

        # Calcular similitudes con todos los términos
        similarities = []
        for candidate_term, candidate_embedding in self.term_embeddings.items():
            if candidate_term != term_lower:
                similarity = self._cosine_similarity(term_embedding, candidate_embedding)
                if similarity >= self.similarity_threshold:
                    similarities.append((candidate_term, similarity))

        # Ordenar por similitud y tomar los mejores
        similarities.sort(key=lambda x: x[1], reverse=True)

        for candidate_term, similarity in similarities[:max_expansions]:
            expansion = ExpandedTerm(
                original_term=term,
                expanded_term=candidate_term,
                expansion_type=ExpansionStrategy.SEMANTIC,
                confidence=similarity,
                weight=similarity * 0.8,  # Peso ligeramente reducido
                context=f"semantic_similarity_{similarity:.3f}"
            )
            expansions.append(expansion)

        return expansions

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)


class StatisticalExpander:
    """Expansor estadístico basado en co-ocurrencia y frecuencias"""

    def __init__(self):
        self.term_cooccurrence = self._build_cooccurrence_matrix()
        self.term_frequencies = self._build_frequency_dictionary()
        self.min_cooccurrence_score = 0.1

    def _build_cooccurrence_matrix(self) -> Dict[str, Dict[str, float]]:
        """Construye matriz de co-ocurrencia (simulada)"""
        # Simulamos co-ocurrencias basadas en dominios temáticos
        cooccurrences = {
            'python': {
                'programación': 0.8, 'código': 0.7, 'script': 0.6,
                'desarrollo': 0.5, 'software': 0.4
            },
            'machine learning': {
                'algoritmo': 0.9, 'modelo': 0.8, 'predicción': 0.7,
                'datos': 0.6, 'entrenamiento': 0.8
            },
            'base de datos': {
                'sql': 0.9, 'consulta': 0.8, 'tabla': 0.7,
                'registro': 0.6, 'índice': 0.5
            },
            'web': {
                'html': 0.8, 'css': 0.7, 'javascript': 0.6,
                'navegador': 0.5, 'servidor': 0.7
            }
        }

        return cooccurrences

    def _build_frequency_dictionary(self) -> Dict[str, float]:
        """Construye diccionario de frecuencias de términos"""
        return {
            'python': 0.15, 'javascript': 0.12, 'java': 0.10,
            'machine learning': 0.08, 'algoritmo': 0.06, 'datos': 0.14,
            'web': 0.11, 'html': 0.09, 'css': 0.07, 'sql': 0.08
        }

    def expand_term(self, term: str, max_expansions: int = 4) -> List[ExpandedTerm]:
        """Expande término usando co-ocurrencia estadística"""
        term_lower = term.lower()
        expansions = []

        if term_lower in self.term_cooccurrence:
            cooccurrent_terms = self.term_cooccurrence[term_lower]

            # Ordenar por score de co-ocurrencia
            sorted_terms = sorted(cooccurrent_terms.items(),
                                key=lambda x: x[1], reverse=True)

            for cooccurrent_term, score in sorted_terms[:max_expansions]:
                if score >= self.min_cooccurrence_score:
                    # Ajustar peso por frecuencia del término
                    frequency_boost = self.term_frequencies.get(cooccurrent_term, 0.01)
                    adjusted_weight = score * (1 + frequency_boost)

                    expansion = ExpandedTerm(
                        original_term=term,
                        expanded_term=cooccurrent_term,
                        expansion_type=ExpansionStrategy.STATISTICAL,
                        confidence=score,
                        weight=min(adjusted_weight, 1.0),
                        context=f"cooccurrence_{score:.3f}_freq_{frequency_boost:.3f}"
                    )
                    expansions.append(expansion)

        return expansions


class ContextualExpander:
    """Expansor contextual que considera el contexto de la query"""

    def __init__(self):
        self.context_patterns = self._build_context_patterns()
        self.domain_terms = self._build_domain_terms()

    def _build_context_patterns(self) -> Dict[str, List[str]]:
        """Patrones contextuales para diferentes tipos de queries"""
        return {
            'how_to': ['tutorial', 'guía', 'pasos', 'instrucciones', 'proceso'],
            'what_is': ['definición', 'concepto', 'explicación', 'significado'],
            'compare': ['diferencia', 'versus', 'comparación', 'ventajas', 'desventajas'],
            'best': ['recomendación', 'óptimo', 'mejor', 'eficiente', 'popular'],
            'error': ['solución', 'fix', 'resolver', 'problema', 'debug'],
            'example': ['ejemplo', 'muestra', 'demo', 'caso', 'ilustración']
        }

    def _build_domain_terms(self) -> Dict[str, List[str]]:
        """Términos específicos por dominio"""
        return {
            'programming': [
                'código', 'función', 'variable', 'clase', 'método', 'biblioteca',
                'framework', 'api', 'debugging', 'testing'
            ],
            'machine_learning': [
                'entrenamiento', 'modelo', 'predicción', 'clasificación', 'regresión',
                'features', 'dataset', 'accuracy', 'overfitting'
            ],
            'web_development': [
                'frontend', 'backend', 'responsive', 'seo', 'deployment',
                'hosting', 'domain', 'ssl', 'cdn'
            ],
            'database': [
                'query', 'join', 'index', 'normalization', 'transaction',
                'backup', 'replication', 'sharding'
            ]
        }

    def expand_query(self, query: str, detected_domain: str = None) -> List[ExpandedTerm]:
        """Expande query basándose en contexto y dominio"""
        query_lower = query.lower()
        expansions = []

        # 1. Detectar patrones contextuales
        for pattern, expansion_terms in self.context_patterns.items():
            if self._matches_pattern(query_lower, pattern):
                for exp_term in expansion_terms:
                    expansion = ExpandedTerm(
                        original_term=query,
                        expanded_term=exp_term,
                        expansion_type=ExpansionStrategy.CONTEXTUAL,
                        confidence=0.7,
                        weight=0.6,
                        context=f"pattern_{pattern}"
                    )
                    expansions.append(expansion)

        # 2. Agregar términos de dominio específico
        if detected_domain and detected_domain in self.domain_terms:
            domain_terms = self.domain_terms[detected_domain]
            for domain_term in domain_terms[:3]:  # Máximo 3 términos de dominio
                expansion = ExpandedTerm(
                    original_term=query,
                    expanded_term=domain_term,
                    expansion_type=ExpansionStrategy.DOMAIN_SPECIFIC,
                    confidence=0.6,
                    weight=0.5,
                    context=f"domain_{detected_domain}"
                )
                expansions.append(expansion)

        return expansions

    def _matches_pattern(self, query: str, pattern: str) -> bool:
        """Verifica si la query coincide con un patrón contextual"""
        pattern_checks = {
            'how_to': any(phrase in query for phrase in ['cómo', 'como', 'how to', 'how do']),
            'what_is': any(phrase in query for phrase in ['qué es', 'que es', 'what is', 'define']),
            'compare': any(phrase in query for phrase in ['vs', 'versus', 'comparar', 'diferencia']),
            'best': any(phrase in query for phrase in ['mejor', 'best', 'recomend', 'óptimo']),
            'error': any(phrase in query for phrase in ['error', 'problema', 'fix', 'solución']),
            'example': any(phrase in query for phrase in ['ejemplo', 'example', 'muestra', 'demo'])
        }

        return pattern_checks.get(pattern, False)


class QueryTypeClassifier:
    """Clasificador de tipos de queries"""

    def __init__(self):
        self.type_patterns = self._build_type_patterns()

    def _build_type_patterns(self) -> Dict[QueryType, List[str]]:
        """Patrones para clasificar tipos de queries"""
        return {
            QueryType.FACTUAL: [
                r'\bqué es\b', r'\bwhat is\b', r'\bdefine\b', r'\bcuándo\b',
                r'\bdónde\b', r'\bwhen\b', r'\bwhere\b'
            ],
            QueryType.PROCEDURAL: [
                r'\bcómo\b', r'\bhow to\b', r'\bpasos\b', r'\btutorial\b',
                r'\bguía\b', r'\bprocess\b'
            ],
            QueryType.CONCEPTUAL: [
                r'\bpor qué\b', r'\bwhy\b', r'\bexplicar\b', r'\bconcepto\b',
                r'\bfundamento\b', r'\bprincip\w*\b'
            ],
            QueryType.COMPARATIVE: [
                r'\bvs\b', r'\bversus\b', r'\bcompar\w*\b', r'\bdiferenci\w*\b',
                r'\bmejor que\b', r'\bbetter than\b'
            ],
            QueryType.EXPLORATORY: [
                r'\btodo sobre\b', r'\babout\b', r'\binformación\b', r'\bresumen\b',
                r'\boverview\b'
            ]
        }

    def classify_query(self, query: str) -> QueryType:
        """Clasifica una query en su tipo correspondiente"""
        query_lower = query.lower()

        # Contar matches por tipo
        type_scores = {}
        for query_type, patterns in self.type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score

        # Devolver el tipo con mayor score
        if max(type_scores.values()) > 0:
            return max(type_scores.items(), key=lambda x: x[1])[0]

        # Default: factual
        return QueryType.FACTUAL


class AutoQueryExpander:
    """Sistema principal de expansión automática de queries"""

    def __init__(self):
        self.semantic_expander = SemanticExpander()
        self.statistical_expander = StatisticalExpander()
        self.contextual_expander = ContextualExpander()
        self.query_classifier = QueryTypeClassifier()

        # Configuración de estrategias por tipo de query
        self.strategy_weights = {
            QueryType.FACTUAL: {
                ExpansionStrategy.SEMANTIC: 0.4,
                ExpansionStrategy.STATISTICAL: 0.3,
                ExpansionStrategy.CONTEXTUAL: 0.3
            },
            QueryType.PROCEDURAL: {
                ExpansionStrategy.CONTEXTUAL: 0.5,
                ExpansionStrategy.SEMANTIC: 0.3,
                ExpansionStrategy.STATISTICAL: 0.2
            },
            QueryType.CONCEPTUAL: {
                ExpansionStrategy.SEMANTIC: 0.5,
                ExpansionStrategy.CONTEXTUAL: 0.3,
                ExpansionStrategy.STATISTICAL: 0.2
            },
            QueryType.COMPARATIVE: {
                ExpansionStrategy.CONTEXTUAL: 0.4,
                ExpansionStrategy.SEMANTIC: 0.4,
                ExpansionStrategy.STATISTICAL: 0.2
            },
            QueryType.EXPLORATORY: {
                ExpansionStrategy.STATISTICAL: 0.4,
                ExpansionStrategy.SEMANTIC: 0.3,
                ExpansionStrategy.CONTEXTUAL: 0.3
            }
        }

    def expand_query(self,
                    query: str,
                    max_expansions: int = 10,
                    strategies: List[ExpansionStrategy] = None,
                    domain_hint: str = None) -> QueryExpansion:
        """
        Expande una query automáticamente usando múltiples estrategias
        """
        import time
        start_time = time.time()

        # 1. Clasificar tipo de query
        query_type = self.query_classifier.classify_query(query)

        # 2. Determinar estrategias a usar
        if strategies is None:
            strategies = list(self.strategy_weights[query_type].keys())

        # 3. Extraer términos clave de la query
        key_terms = self._extract_key_terms(query)

        # 4. Generar expansiones por estrategia
        all_expansions = []

        for strategy in strategies:
            if strategy == ExpansionStrategy.SEMANTIC:
                for term in key_terms:
                    expansions = self.semantic_expander.expand_term(term, max_expansions // len(key_terms))
                    all_expansions.extend(expansions)

            elif strategy == ExpansionStrategy.STATISTICAL:
                for term in key_terms:
                    expansions = self.statistical_expander.expand_term(term, max_expansions // len(key_terms))
                    all_expansions.extend(expansions)

            elif strategy in [ExpansionStrategy.CONTEXTUAL, ExpansionStrategy.DOMAIN_SPECIFIC]:
                expansions = self.contextual_expander.expand_query(query, domain_hint)
                all_expansions.extend(expansions)

        # 5. Filtrar y rankear expansiones
        filtered_expansions = self._filter_and_rank_expansions(
            all_expansions, query_type, max_expansions
        )

        # 6. Generar queries expandidas
        expanded_queries = self._generate_expanded_queries(
            query, filtered_expansions
        )

        # 7. Calcular score de confianza
        confidence_score = self._calculate_confidence_score(
            filtered_expansions, query_type
        )

        processing_time = time.time() - start_time

        return QueryExpansion(
            original_query=query,
            expanded_queries=expanded_queries,
            expanded_terms=filtered_expansions,
            query_type=query_type,
            expansion_strategies=strategies,
            confidence_score=confidence_score,
            processing_time=processing_time
        )

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extrae términos clave de la query"""
        # Remover stop words comunes
        stop_words = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no',
            'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al',
            'the', 'is', 'at', 'of', 'on', 'and', 'or', 'but', 'in', 'with',
            'to', 'for', 'as', 'by', 'from', 'up', 'about', 'into', 'through'
        }

        # Limpiar y tokenizar
        words = re.findall(r'\b\w{3,}\b', query.lower())
        key_terms = [word for word in words if word not in stop_words]

        # Detectar frases compuestas importantes
        compound_phrases = self._detect_compound_phrases(query)
        key_terms.extend(compound_phrases)

        return list(set(key_terms))  # Remover duplicados

    def _detect_compound_phrases(self, query: str) -> List[str]:
        """Detecta frases compuestas importantes"""
        compound_patterns = [
            r'machine learning', r'artificial intelligence', r'deep learning',
            r'data science', r'computer vision', r'natural language',
            r'base de datos', r'inteligencia artificial', r'aprendizaje automático'
        ]

        phrases = []
        query_lower = query.lower()

        for pattern in compound_patterns:
            if re.search(pattern, query_lower):
                phrases.append(pattern)

        return phrases

    def _filter_and_rank_expansions(self,
                                   expansions: List[ExpandedTerm],
                                   query_type: QueryType,
                                   max_expansions: int) -> List[ExpandedTerm]:
        """Filtra y rankea las expansiones generadas"""

        # 1. Remover duplicados
        unique_expansions = {}
        for expansion in expansions:
            key = expansion.expanded_term.lower()
            if key not in unique_expansions or expansion.confidence > unique_expansions[key].confidence:
                unique_expansions[key] = expansion

        expansions = list(unique_expansions.values())

        # 2. Aplicar pesos por tipo de query
        strategy_weights = self.strategy_weights[query_type]

        for expansion in expansions:
            strategy_weight = strategy_weights.get(expansion.expansion_type, 0.1)
            expansion.weight *= strategy_weight

        # 3. Filtrar por umbral de confianza mínima
        min_confidence = 0.3
        expansions = [exp for exp in expansions if exp.confidence >= min_confidence]

        # 4. Rankear por weight combinado
        expansions.sort(key=lambda x: x.weight, reverse=True)

        # 5. Tomar top expansions
        return expansions[:max_expansions]

    def _generate_expanded_queries(self,
                                  original_query: str,
                                  expansions: List[ExpandedTerm]) -> List[str]:
        """Genera queries expandidas combinando términos originales y expandidos"""
        expanded_queries = []

        # 1. Query original + términos expandidos individuales
        for expansion in expansions[:5]:  # Top 5 expansiones
            if expansion.weight > 0.5:
                expanded_query = f"{original_query} {expansion.expanded_term}"
                expanded_queries.append(expanded_query)

        # 2. Query reformulada reemplazando términos
        for expansion in expansions[:3]:  # Top 3 para reemplazo
            if expansion.confidence > 0.7:
                reformulated = original_query.replace(
                    expansion.original_term,
                    expansion.expanded_term
                )
                if reformulated != original_query:
                    expanded_queries.append(reformulated)

        # 3. Query con múltiples expansiones
        if len(expansions) >= 3:
            top_terms = [exp.expanded_term for exp in expansions[:3]]
            multi_expanded = f"{original_query} {' '.join(top_terms)}"
            expanded_queries.append(multi_expanded)

        # Remover duplicados preservando orden
        seen = set()
        unique_queries = []
        for query in expanded_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)

        return unique_queries

    def _calculate_confidence_score(self,
                                   expansions: List[ExpandedTerm],
                                   query_type: QueryType) -> float:
        """Calcula score de confianza para las expansiones generadas"""
        if not expansions:
            return 0.0

        # Factores de confianza
        avg_confidence = np.mean([exp.confidence for exp in expansions])
        avg_weight = np.mean([exp.weight for exp in expansions])
        expansion_diversity = len(set(exp.expansion_type for exp in expansions)) / len(ExpansionStrategy)
        expansion_count_factor = min(len(expansions) / 10.0, 1.0)

        # Combinar factores
        confidence_score = (
            avg_confidence * 0.4 +
            avg_weight * 0.3 +
            expansion_diversity * 0.2 +
            expansion_count_factor * 0.1
        )

        return min(confidence_score, 1.0)

    def get_expansion_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de expansión"""
        return {
            'semantic_terms': len(self.semantic_expander.term_embeddings),
            'cooccurrence_terms': len(self.statistical_expander.term_cooccurrence),
            'context_patterns': len(self.contextual_expander.context_patterns),
            'domain_categories': len(self.contextual_expander.domain_terms),
            'supported_strategies': [s.value for s in ExpansionStrategy],
            'supported_query_types': [t.value for t in QueryType]
        }


# Funciones de utilidad
def expand_query(query: str,
                max_expansions: int = 8,
                strategies: List[str] = None,
                domain: str = None) -> QueryExpansion:
    """
    Función de conveniencia para expansión de queries
    """
    expander = AutoQueryExpander()

    strategy_enums = None
    if strategies:
        strategy_enums = [ExpansionStrategy(s) for s in strategies]

    return expander.expand_query(query, max_expansions, strategy_enums, domain)


if __name__ == "__main__":
    # Ejemplo de uso completo
    print("🔍 Auto Query Expansion - Demo")
    print("=" * 50)

    expander = AutoQueryExpander()

    # Queries de ejemplo
    test_queries = [
        "¿Cómo funciona machine learning?",
        "Mejor algoritmo para clasificación",
        "Diferencia entre Python y JavaScript",
        "¿Qué es una base de datos?",
        "Tutorial de programación web"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")

        try:
            expansion = expander.expand_query(query, max_expansions=6)

            print(f"Tipo de query: {expansion.query_type.value}")
            print(f"Estrategias usadas: {[s.value for s in expansion.expansion_strategies]}")
            print(f"Confianza: {expansion.confidence_score:.3f}")
            print(f"Tiempo: {expansion.processing_time:.3f}s")

            print("\nTérminos expandidos:")
            for j, term in enumerate(expansion.expanded_terms, 1):
                print(f"  {j}. '{term.expanded_term}' <- '{term.original_term}'")
                print(f"     Estrategia: {term.expansion_type.value}, Confianza: {term.confidence:.3f}")

            print("\nQueries expandidas:")
            for j, expanded_query in enumerate(expansion.expanded_queries, 1):
                print(f"  {j}. {expanded_query}")

        except Exception as e:
            print(f"Error expandiendo query: {e}")

    # Estadísticas del sistema
    print(f"\n📊 Estadísticas del Sistema:")
    stats = expander.get_expansion_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print(f"\n🎉 Query Expansion Demo Completed!")
