"""
Spec-Driven Development Module para MCP
Implementa desarrollo basado en especificaciones para estructurar contexto.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

class SpecParser:
    """Parser para identificar y extraer especificaciones de documentos"""

    def __init__(self):
        # Patrones para identificar tipos de specs
        self.spec_patterns = {
            'user_stories': [
                r'##+\s*(?:User\s+Stories?|Historias\s+de\s+Usuario|Stories)',
                r'As\s+(?:a|an)\s+(.+),\s+I\s+want\s+(.+),\s+so\s+that\s+(.+)',
                r'Como\s+(.+),\s+quiero\s+(.+),\s+para\s+(.+)'
            ],
            'functional_requirements': [
                r'##+\s*(?:Functional\s+Requirements?|Requisitos\s+Funcionales?|FR)',
                r'(?:FR|RF)\d+:\s*(.+)',
                r'Requerimiento\s+Funcional:?\s*(.+)'
            ],
            'non_functional_requirements': [
                r'##+\s*(?:Non-Functional\s+Requirements?|Requisitos\s+No\s+Funcionales?|NFR)',
                r'(?:NFR|RNF)\d+:\s*(.+)',
                r'Requerimiento\s+No\s+Funcional:?\s*(.+)'
            ],
            'api_specifications': [
                r'##+\s*(?:API\s+Specs?|Especificaciones\s+API|API\s+Endpoints?)',
                r'```(?:http|rest|api)\s*\n(.*?)\n```',
                r'POST|GET|PUT|DELETE|PATCH\s+/.+'
            ],
            'technical_specs': [
                r'##+\s*(?:Technical\s+Specs?|Especificaciones\s+Técnicas?|Tech\s+Specs)',
                r'Arquitectura:?\s*(.+)',
                r'Tecnologías:?\s*(.+)',
                r'Framework:?\s*(.+)'
            ],
            'acceptance_criteria': [
                r'##+\s*(?:Acceptance\s+Criteria|Criterios\s+de\s+Aceptación|AC)',
                r'(?:Given|Cuando|Dado)\s+(.+),\s+(?:When|Entonces|Then)\s+(.+)',
                r'Criterio\s+de\s+Aceptación:?\s*(.+)'
            ],
            'business_rules': [
                r'##+\s*(?:Business\s+Rules?|Reglas\s+de\s+Negocio|BR)',
                r'Regla\s+de\s+Negocio:?\s*(.+)',
                r'(?:BR|RN)\d+:\s*(.+)'
            ]
        }

    def parse_document(self, content: str, filename: str) -> Dict[str, List[Dict]]:
        """Parsea un documento y extrae todas las specs"""
        specs = {}

        for spec_type, patterns in self.spec_patterns.items():
            specs[spec_type] = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for match in matches:
                    if isinstance(match, tuple):
                        # Para patrones con grupos
                        spec_content = ' '.join(match).strip()
                    else:
                        spec_content = match.strip()

                    if spec_content:
                        specs[spec_type].append({
                            'content': spec_content,
                            'source_file': filename,
                            'line_number': self._estimate_line_number(content, spec_content),
                            'type': spec_type,
                            'confidence': self._calculate_confidence(spec_content, pattern)
                        })

        return specs

    def _estimate_line_number(self, content: str, spec_content: str) -> int:
        """Estima el número de línea donde aparece la spec"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if spec_content[:50] in line:
                return i + 1
        return 0

    def _calculate_confidence(self, content: str, pattern: str) -> float:
        """Calcula confianza en que es una spec válida"""
        # Confianza basada en longitud y estructura
        if len(content) < 10:
            return 0.3
        if len(content) > 50:
            return 0.9
        return 0.6

    def extract_spec_metadata(self, specs: Dict[str, List[Dict]]) -> Dict[str, any]:
        """Extrae metadata general del conjunto de specs"""
        total_specs = sum(len(spec_list) for spec_list in specs.values())

        metadata = {
            'total_specs': total_specs,
            'specs_by_type': {k: len(v) for k, v in specs.items()},
            'files_covered': len(set(
                spec['source_file']
                for spec_list in specs.values()
                for spec in spec_list
            )),
            'avg_confidence': sum(
                spec['confidence']
                for spec_list in specs.values()
                for spec in spec_list
            ) / total_specs if total_specs > 0 else 0
        }

        return metadata

class SpecIndexer:
    """Indexa specs para búsqueda eficiente"""

    def __init__(self):
        self.spec_index = {}
        self.keyword_to_specs = {}

    def index_specs(self, all_specs: Dict[str, Dict[str, List[Dict]]]):
        """Indexa todas las specs de todos los documentos"""
        self.spec_index = all_specs

        # Crear índice invertido de keywords a specs
        for filename, file_specs in all_specs.items():
            for spec_type, specs in file_specs.items():
                for spec in specs:
                    keywords = self._extract_keywords(spec['content'])
                    for keyword in keywords:
                        if keyword not in self.keyword_to_specs:
                            self.keyword_to_specs[keyword] = []
                        self.keyword_to_specs[keyword].append({
                            'spec_type': spec_type,
                            'content': spec['content'],
                            'filename': filename,
                            'confidence': spec['confidence']
                        })

    def _extract_keywords(self, content: str) -> List[str]:
        """Extrae keywords de una spec"""
        # Palabras clave técnicas comunes
        tech_keywords = [
            'api', 'endpoint', 'database', 'authentication', 'authorization',
            'user', 'admin', 'login', 'password', 'security', 'encryption',
            'frontend', 'backend', 'mobile', 'web', 'api', 'rest', 'graphql',
            'python', 'django', 'react', 'javascript', 'typescript',
            'mysql', 'postgresql', 'mongodb', 'redis',
            'aws', 'docker', 'kubernetes', 'ci', 'cd', 'git',
            'testing', 'unit', 'integration', 'e2e',
            'performance', 'scalability', 'security', 'monitoring'
        ]

        words = re.findall(r'\b\w+\b', content.lower())
        keywords = [word for word in words if word in tech_keywords or len(word) > 4]

        return list(set(keywords))  # Remover duplicados

    def search_specs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Busca specs relevantes para una query"""
        query_keywords = self._extract_keywords(query.lower())

        if not query_keywords:
            # Fallback: buscar en todo el contenido
            return self._search_fallback(query, max_results)

        # Encontrar specs que contengan las keywords
        candidate_specs = []
        for keyword in query_keywords:
            if keyword in self.keyword_to_specs:
                candidate_specs.extend(self.keyword_to_specs[keyword])

        # Rankear por relevancia
        ranked_specs = self._rank_specs(candidate_specs, query_keywords)

        return ranked_specs[:max_results]

    def _search_fallback(self, query: str, max_results: int) -> List[Dict]:
        """Búsqueda fallback cuando no hay keywords específicas"""
        results = []
        query_lower = query.lower()

        for filename, file_specs in self.spec_index.items():
            for spec_type, specs in file_specs.items():
                for spec in specs:
                    if query_lower in spec['content'].lower():
                        results.append({
                            'spec_type': spec_type,
                            'content': spec['content'],
                            'filename': filename,
                            'confidence': spec['confidence'],
                            'relevance_score': 0.7  # Score por defecto para fallback
                        })

        # Ordenar por confianza
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:max_results]

    def _rank_specs(self, specs: List[Dict], query_keywords: List[str]) -> List[Dict]:
        """Rankea specs por relevancia"""
        ranked = []

        for spec in specs:
            # Calcular score basado en:
            # - Número de keywords matching
            # - Confianza de la spec
            # - Tipo de spec (algunos son más relevantes)
            spec_keywords = self._extract_keywords(spec['content'])
            matching_keywords = len(set(query_keywords) & set(spec_keywords))

            type_boost = {
                'user_stories': 1.2,
                'functional_requirements': 1.1,
                'api_specifications': 1.3,
                'technical_specs': 1.0,
                'acceptance_criteria': 0.9,
                'business_rules': 0.8,
                'non_functional_requirements': 0.7
            }

            relevance_score = (
                matching_keywords * 0.4 +
                spec['confidence'] * 0.4 +
                type_boost.get(spec['spec_type'], 1.0) * 0.2
            )

            spec_copy = spec.copy()
            spec_copy['relevance_score'] = min(relevance_score, 1.0)  # Clamp a 1.0
            ranked.append(spec_copy)

        # Ordenar por score
        ranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        return ranked

    def get_spec_summary(self) -> Dict[str, any]:
        """Obtiene resumen de specs indexadas"""
        total_specs = sum(
            len(specs) for file_specs in self.spec_index.values()
            for specs in file_specs.values()
        )

        specs_by_type = {}
        for file_specs in self.spec_index.values():
            for spec_type, specs in file_specs.items():
                specs_by_type[spec_type] = specs_by_type.get(spec_type, 0) + len(specs)

        return {
            'total_specs': total_specs,
            'specs_by_type': specs_by_type,
            'indexed_files': len(self.spec_index),
            'unique_keywords': len(self.keyword_to_specs)
        }
