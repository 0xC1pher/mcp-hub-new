#!/usr/bin/env python3
"""
Script de validación para verificar sincronización entre project-guidelines.md y keyword-to-sections.json
Versión: 1.0.0
"""

import json
import re
import sys
import pathlib
from typing import Set, Dict, List

def log(message: str, level: str = "INFO") -> None:
    """Función de logging simple"""
    emoji = {
        "INFO": "📝",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }
    print(f"{emoji.get(level, '📝')} {message}")

def load_files(base_path: pathlib.Path) -> tuple:
    """Carga los archivos necesarios para validación"""
    md_file = base_path / "servers" / "context-query" / "context" / "project-guidelines.md"
    json_file = base_path / "servers" / "context-query" / "index" / "keyword-to-sections.json"

    try:
        # Cargar guidelines
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Cargar índice
        with open(json_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        return md_content, index_data

    except FileNotFoundError as e:
        log(f"Archivo no encontrado: {e.filename}", "ERROR")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log(f"Error en JSON del índice: {e}", "ERROR")
        sys.exit(1)

def extract_sections_from_md(md_content: str) -> Set[str]:
    """Extrae los SECTION_ID del archivo markdown"""
    pattern = r"<!-- SECTION_ID: ([a-z0-9_]+) -->"
    sections = set(re.findall(pattern, md_content))

    log(f"Encontradas {len(sections)} secciones en el markdown: {', '.join(sorted(sections))}")
    return sections

def extract_sections_from_index(index_data: Dict) -> Set[str]:
    """Extrae todas las secciones referenciadas en el índice"""
    sections = set()

    for keyword, section_list in index_data.items():
        if isinstance(section_list, list):
            sections.update(section_list)
        elif isinstance(section_list, str):
            sections.add(section_list)

    log(f"Encontradas {len(sections)} secciones en el índice: {', '.join(sorted(sections))}")
    return sections

def validate_index_structure(index_data: Dict) -> List[str]:
    """Valida la estructura básica del índice"""
    errors = []

    if not isinstance(index_data, dict):
        errors.append("El índice debe ser un objeto JSON")
        return errors

    for keyword, sections in index_data.items():
        # Validar que la keyword sea string
        if not isinstance(keyword, str):
            errors.append(f"Keyword '{keyword}' debe ser string")

        # Validar formato de keyword (solo minúsculas, letras, números, guiones)
        if not re.match(r'^[a-z0-9_]+$', keyword):
            errors.append(f"Keyword '{keyword}' debe contener solo minúsculas, números y guiones bajos")

        # Validar que sections sea lista o string
        if not isinstance(sections, (list, str)):
            errors.append(f"Secciones para keyword '{keyword}' deben ser lista o string")
            continue

        # Convertir a lista para validación uniforme
        section_list = [sections] if isinstance(sections, str) else sections

        # Validar cada sección
        for section in section_list:
            if not isinstance(section, str):
                errors.append(f"Sección '{section}' para keyword '{keyword}' debe ser string")
                continue

            # Validar formato de sección (snake_case)
            if not re.match(r'^[a-z0-9_]+$', section):
                errors.append(f"Sección '{section}' debe estar en snake_case")

    return errors

def main():
    """Función principal de validación"""
    log("🔍 Iniciando validación del índice MCP...")

    # Obtener ruta base
    script_path = pathlib.Path(__file__).resolve()
    base_path = script_path.parent.parent

    # Cargar archivos
    md_content, index_data = load_files(base_path)

    # Validar estructura del índice
    structure_errors = validate_index_structure(index_data)
    if structure_errors:
        for error in structure_errors:
            log(error, "ERROR")
        log("Falló validación de estructura del índice", "ERROR")
        sys.exit(1)

    # Extraer secciones de ambos archivos
    md_sections = extract_sections_from_md(md_content)
    index_sections = extract_sections_from_index(index_data)

    # Verificar sincronización
    missing_in_index = md_sections - index_sections
    extra_in_index = index_sections - md_sections

    if missing_in_index:
        log(f"Secciones en MD pero no en índice: {', '.join(sorted(missing_in_index))}", "WARNING")

    if extra_in_index:
        log(f"Secciones en índice pero no en MD: {', '.join(sorted(extra_in_index))}", "WARNING")

    # Validación final
    if missing_in_index or extra_in_index:
        log("⚠️  Desincronización detectada entre archivos", "WARNING")
        log("Recomendación: Actualiza el índice para que refleje exactamente las secciones del markdown", "WARNING")

        # En modo estricto, fallar si hay desincronización
        if "--strict" in sys.argv:
            sys.exit(1)
    else:
        log("✅ Índice y secciones sincronizados correctamente", "SUCCESS")

    # Estadísticas
    total_keywords = len(index_data)
    total_sections = len(index_sections)
    avg_sections_per_keyword = total_sections / total_keywords if total_keywords > 0 else 0

    log("📊 Estadísticas del índice:")
    log(f"   - Keywords únicos: {total_keywords}")
    log(f"   - Secciones únicas: {total_sections}")
    log(f"   - Promedio de secciones por keyword: {avg_sections_per_keyword:.1f}")

if __name__ == "__main__":
    main()
