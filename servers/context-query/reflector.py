"""
Reflector Module para ACE (Agentic Context Engineering)
Analiza feedback para generar insights sobre fallos en consultas.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Reflector:
    """Clase que analiza feedback para identificar patrones de fallo"""

    def __init__(self, feedback_file):
        self.feedback_file = Path(feedback_file)

    def analyze_feedback(self):
        """Analiza todo el feedback y genera insights"""
        feedback_list = self._load_feedback()

        if not feedback_list:
            return {"insights": [], "summary": "No hay feedback disponible"}

        # Separar feedback útil y no útil
        helpful = [f for f in feedback_list if f.get('helpful', False)]
        unhelpful = [f for f in feedback_list if not f.get('helpful', False)]

        insights = []

        # Análisis de queries no útiles
        if unhelpful:
            unhelpful_insights = self._analyze_unhelpful_queries(unhelpful)
            insights.extend(unhelpful_insights)

        # Análisis de sugerencias
        suggestion_insights = self._analyze_suggestions(feedback_list)
        insights.extend(suggestion_insights)

        # Estadísticas generales
        total_queries = len(feedback_list)
        helpful_rate = len(helpful) / total_queries if total_queries > 0 else 0

        summary = {
            "total_feedback": total_queries,
            "helpful_rate": helpful_rate,
            "unhelpful_count": len(unhelpful),
            "insights_count": len(insights)
        }

        return {
            "insights": insights,
            "summary": summary
        }

    def _analyze_unhelpful_queries(self, unhelpful_feedback):
        """Analiza queries que no fueron útiles"""
        insights = []

        # Extraer palabras clave comunes de queries no útiles
        query_words = []
        for fb in unhelpful_feedback:
            query = fb.get('query', '').lower()
            words = re.findall(r'\b\w+\b', query)
            query_words.extend(words)

        # Contar palabras frecuentes
        word_counts = Counter(query_words)
        common_words = [word for word, count in word_counts.most_common(10) if count > 1]

        if common_words:
            insights.append({
                "type": "missing_keywords",
                "description": f"Queries no útiles contienen palabras comunes: {', '.join(common_words[:5])}",
                "suggested_action": "Agregar estas palabras al índice keyword-to-sections.json",
                "affected_queries": len(unhelpful_feedback),
                "keywords": common_words
            })

        # Agrupar por temas similares
        themes = self._group_by_themes(unhelpful_feedback)
        for theme, queries in themes.items():
            if len(queries) > 1:
                insights.append({
                    "type": "thematic_failure",
                    "description": f"Múltiples queries sobre '{theme}' fallaron",
                    "suggested_action": "Revisar sección relacionada o agregar nueva sección",
                    "affected_queries": len(queries),
                    "theme": theme,
                    "examples": queries[:3]
                })

        return insights

    def _analyze_suggestions(self, feedback_list):
        """Analiza sugerencias de usuarios"""
        insights = []

        suggestions = [fb.get('suggestion', '').strip() for fb in feedback_list if fb.get('suggestion', '').strip()]

        if not suggestions:
            return insights

        # Agrupar sugerencias similares
        suggestion_themes = defaultdict(list)
        for sugg in suggestions:
            # Simplificar: usar primeras palabras como tema
            words = sugg.lower().split()[:3]
            theme = ' '.join(words)
            suggestion_themes[theme].append(sugg)

        for theme, sugg_list in suggestion_themes.items():
            if len(sugg_list) > 1:
                insights.append({
                    "type": "recurring_suggestion",
                    "description": f"Sugerencia recurrente: '{theme}'",
                    "suggested_action": "Implementar la mejora sugerida",
                    "frequency": len(sugg_list),
                    "examples": sugg_list[:3]
                })

        return insights

    def _group_by_themes(self, feedback_list):
        """Agrupa queries por temas similares"""
        themes = defaultdict(list)

        for fb in feedback_list:
            query = fb.get('query', '').lower()
            # Tema simple: primera palabra significativa
            words = re.findall(r'\b\w{4,}\b', query)  # palabras de 4+ letras
            if words:
                theme = words[0]
                themes[theme].append(query)

        return themes

    def get_recent_insights(self, hours=24):
        """Obtiene insights de feedback reciente"""
        import time
        cutoff = time.time() - (hours * 3600)

        feedback_list = self._load_feedback()
        recent_feedback = [f for f in feedback_list if f.get('timestamp', 0) > cutoff]

        if not recent_feedback:
            return {"insights": [], "summary": "No hay feedback reciente"}

        # Análisis rápido de feedback reciente
        unhelpful_recent = [f for f in recent_feedback if not f.get('helpful', False)]

        insights = []
        if unhelpful_recent:
            queries = [f['query'] for f in unhelpful_recent]
            insights.append({
                "type": "recent_failures",
                "description": f"{len(unhelpful_recent)} queries fallaron en las últimas {hours} horas",
                "queries": queries[:5]
            })

        return {
            "insights": insights,
            "summary": f"Analizado {len(recent_feedback)} feedback reciente"
        }

    def _load_feedback(self):
        """Carga feedback desde archivo"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
