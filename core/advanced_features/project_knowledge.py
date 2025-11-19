"""Utility helpers to build knowledge sessions from project artifacts.

These helpers keep the demo portable by standardizing how feature documents are
loaded and how interactive context is collected from the user.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_CONTEXT_QUESTIONS: Iterable[tuple[str, str]] = (
    ("project_name", "Nombre del proyecto"),
    ("primary_goal", "Objetivo principal o problema a resolver"),
    ("key_constraints", "Restricciones clave (tecnológicas, de tiempo, legales)"),
    ("target_users", "Usuarios o stakeholders principales"),
    ("success_metrics", "Cómo sabremos que el proyecto fue exitoso"),
)


class ProjectKnowledgeManager:
    """Carga documentos base y solicita contexto adicional al usuario."""

    def __init__(
        self,
        base_dir: Path,
        knowledge_files: Optional[List[str]] = None,
        context_questions: Iterable[tuple[str, str]] = DEFAULT_CONTEXT_QUESTIONS,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.knowledge_files = knowledge_files or ["feature.md"]
        self.context_questions = tuple(context_questions)

    def load_documents(self) -> List[Dict[str, Any]]:
        """Carga documentos declarados en `knowledge_files`.

        Returns:
            Lista de diccionarios compatibles con el orquestador.
        """

        documents: List[Dict[str, Any]] = []
        for rel_path in self.knowledge_files:
            path = Path(rel_path)
            if not path.is_absolute():
                path = self.base_dir / rel_path

            try:
                content = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"⚠️  No se encontró el archivo de conocimiento: {path}")
                continue

            doc_id = path.stem.replace(" ", "_")
            documents.append(
                {
                    "id": doc_id or "knowledge_doc",
                    "content": content,
                    "path": str(path),
                    "type": "markdown" if path.suffix.lower() == ".md" else "text",
                    "domain": "project_requirements",
                    "complexity": self._estimate_complexity(content),
                }
            )

        return documents

    def gather_project_context(self) -> Dict[str, Any]:
        """Solicita información contextual al usuario."""

        print("\n🧭 Configuración rápida del proyecto (deja en blanco si no aplica):")
        context: Dict[str, Any] = {}

        for key, prompt in self.context_questions:
            answer = self._safe_input(f"   {prompt}: ")
            if answer:
                context[key] = answer

        if context:
            context["summary"] = self._build_context_summary(context)

        return context

    def collect_user_queries(self, default_queries: Optional[List[str]] = None) -> List[str]:
        """Pide al usuario las consultas que desea resolver."""

        print("\n🗣️  Define las consultas que quieres responder (Enter para terminar):")
        queries: List[str] = []
        idx = 1
        while True:
            query = self._safe_input(f"   Consulta {idx}: ")
            if not query:
                break
            queries.append(query)
            idx += 1

        if queries:
            return queries
        return list(default_queries or ["Resumen de las reglas clave del feature.md"])

    def build_context_payload(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Genera payload compacto listo para pasar al orquestador."""

        if not project_context:
            return {}

        payload = {"project_context": project_context}
        if "summary" in project_context:
            payload["context_summary"] = project_context["summary"]
        return payload

    @staticmethod
    def _estimate_complexity(content: str) -> float:
        """Calcula una heurística simple de complejidad."""

        length = len(content)
        # Normalizar entre 0.1 y 0.9 para evitar extremos
        normalized = max(0.1, min(0.9, length / 5000))
        return round(normalized, 2)

    @staticmethod
    def _build_context_summary(context: Dict[str, Any]) -> str:
        pairs = [f"{key}: {value}" for key, value in context.items() if key != "summary"]
        return " | ".join(pairs)

    @staticmethod
    def _safe_input(prompt: str) -> str:
        try:
            return input(prompt).strip()
        except EOFError:
            print("   (sin entrada, se usará valor por defecto)")
            return ""
