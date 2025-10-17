"""
Curator Module para ACE (Agentic Context Engineering)
Integra insights del Reflector en actualizaciones incrementales del contexto.
"""

import json
import time
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class Curator:
    """Clase que aplica actualizaciones incrementales basadas en insights"""

    def __init__(self, index_file, guidelines_file, feedback_file):
        self.index_file = Path(index_file)
        self.guidelines_file = Path(guidelines_file)
        self.feedback_file = Path(feedback_file)

    def apply_insights(self, insights):
        """Aplica insights para actualizar contexto"""
        updates = []

        for insight in insights:
            if insight['type'] == 'missing_keywords':
                update = self._add_missing_keywords(insight)
                if update:
                    updates.append(update)

            elif insight['type'] == 'thematic_failure':
                update = self._handle_thematic_failure(insight)
                if update:
                    updates.append(update)

            elif insight['type'] == 'recurring_suggestion':
                update = self._implement_suggestion(insight)
                if update:
                    updates.append(update)

        # Aplicar todas las actualizaciones
        if updates:
            self._apply_updates(updates)
            logger.info(f"Aplicadas {len(updates)} actualizaciones incrementales")

        return updates

    def _add_missing_keywords(self, insight):
        """Agrega keywords faltantes al índice"""
        keywords = insight.get('keywords', [])
        if not keywords:
            return None

        # Cargar índice actual
        index_data = self._load_index()

        # Sugerir sección más probable para estos keywords
        # Por simplicidad, agregar a una sección general o pedir manual
        # Aquí, podríamos usar lógica para mapear a secciones existentes

        # Para demo, agregar a 'constraints' si contiene palabras de seguridad
        security_words = ['seguridad', 'auth', 'login', 'permisos', 'vulnerabilidad']
        target_section = 'constraints' if any(kw in security_words for kw in keywords) else 'tech_architecture'

        # Agregar keywords
        for keyword in keywords[:3]:  # Limitar a 3 por insight
            if keyword not in index_data:
                index_data[keyword] = [target_section]
            elif target_section not in index_data[keyword]:
                index_data[keyword].append(target_section)

        # Guardar índice actualizado
        self._save_index(index_data)

        return {
            "type": "keyword_addition",
            "keywords_added": keywords[:3],
            "target_section": target_section,
            "timestamp": time.time()
        }

    def _handle_thematic_failure(self, insight):
        """Maneja fallos temáticos agregando bullets o secciones"""
        theme = insight.get('theme', '')
        examples = insight.get('examples', [])

        if not theme or not examples:
            return None

        # Crear un bullet nuevo para el tema
        bullet = {
            "id": f"bullet_{int(time.time())}_{theme.replace(' ', '_')}",
            "content": f"Información adicional sobre {theme}. Basado en consultas fallidas: {', '.join(examples[:2])}",
            "helpful_count": 0,
            "harmful_count": len(examples),
            "theme": theme,
            "source": "curator_insight",
            "timestamp": time.time()
        }

        # Para ahora, solo loggear; en futura versión, integrar en guidelines
        logger.info(f"Nuevo bullet sugerido para tema '{theme}': {bullet['content']}")

        # Podríamos guardar en un archivo separado de bullets
        bullets_file = self.index_file.parent / "context_bullets.json"
        bullets = self._load_bullets(bullets_file)
        bullets.append(bullet)
        self._save_bullets(bullets_file, bullets)

        return {
            "type": "bullet_creation",
            "theme": theme,
            "bullet_id": bullet["id"],
            "timestamp": time.time()
        }

    def _implement_suggestion(self, insight):
        """Implementa sugerencias recurrentes"""
        description = insight.get('description', '')
        frequency = insight.get('frequency', 0)

        if frequency < 2:
            return None

        # Para sugerencias, crear una nota de mejora
        improvement_note = {
            "type": "pending_improvement",
            "description": description,
            "frequency": frequency,
            "status": "pending",
            "timestamp": time.time()
        }

        # Guardar en archivo de mejoras
        improvements_file = self.index_file.parent / "pending_improvements.json"
        improvements = self._load_improvements(improvements_file)
        improvements.append(improvement_note)
        self._save_improvements(improvements_file, improvements)

        return {
            "type": "improvement_notation",
            "description": description,
            "frequency": frequency,
            "timestamp": time.time()
        }

    def refine_bullets(self):
        """Refina bullets existentes: de-duplica, actualiza counters"""
        bullets_file = self.index_file.parent / "context_bullets.json"
        bullets = self._load_bullets(bullets_file)

        if not bullets:
            return []

        # Actualizar counters basado en feedback reciente
        feedback_list = self._load_feedback()
        recent_feedback = [f for f in feedback_list if time.time() - f.get('timestamp', 0) < 86400]  # Últimas 24h

        for bullet in bullets:
            # Lógica simple: si query relacionada fue útil, incrementar helpful
            # Esto es placeholder; en realidad necesitaríamos matching más sofisticado
            pass

        # De-duplicación simple por contenido similar
        unique_bullets = self._deduplicate_bullets(bullets)

        if len(unique_bullets) != len(bullets):
            self._save_bullets(bullets_file, unique_bullets)
            logger.info(f"De-duplicados {len(bullets) - len(unique_bullets)} bullets")

        return unique_bullets

    def _deduplicate_bullets(self, bullets):
        """De-duplica bullets por contenido similar"""
        seen = set()
        unique = []

        for bullet in bullets:
            content = bullet.get('content', '').lower()[:100]  # Primeros 100 chars
            if content not in seen:
                seen.add(content)
                unique.append(bullet)

        return unique

    def _apply_updates(self, updates):
        """Aplica múltiples updates de manera atómica"""
        # Por ahora, solo loggear; en producción, backup y rollback
        for update in updates:
            logger.info(f"Update aplicado: {update['type']} - {update.get('description', 'N/A')}")

    def _load_index(self):
        """Carga índice"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self, index_data):
        """Guarda índice"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def _load_feedback(self):
        """Carga feedback"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _load_bullets(self, bullets_file):
        """Carga bullets"""
        if bullets_file.exists():
            with open(bullets_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_bullets(self, bullets_file, bullets):
        """Guarda bullets"""
        with open(bullets_file, 'w', encoding='utf-8') as f:
            json.dump(bullets, f, ensure_ascii=False, indent=2)

    def _load_improvements(self, improvements_file):
        """Carga mejoras pendientes"""
        if improvements_file.exists():
            with open(improvements_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_improvements(self, improvements_file, improvements):
        """Guarda mejoras pendientes"""
        with open(improvements_file, 'w', encoding='utf-8') as f:
            json.dump(improvements, f, ensure_ascii=False, indent=2)
