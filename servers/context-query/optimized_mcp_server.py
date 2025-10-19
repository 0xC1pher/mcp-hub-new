#!/usr/bin/env python3
"""
Servidor MCP Context Query para SoftMedic - Versión Optimizada Completa
Implementa TODAS las estrategias avanzadas de OPTIMIZATION-STRATEGIES.md:
- Token Budgeting Inteligente
- Chunking Semántico Avanzado
- Cache Multinivel (L1/L2/Disk)
- Query Optimization con expansión semántica
- Rate Limiting Adaptativo
- Resource Monitoring y Performance Metrics
- Fuzzy Search y Relevance Scoring
- Arquitectura modular y escalable
"""

import json
import sys
import logging
import re
import time
import threading
import psutil
import os
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TokenBudgetManager:
    """Gestión inteligente de presupuesto de tokens"""
    
    def __init__(self, max_tokens: int = 4000, reserved_tokens: int = 500):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens

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
            'access_count': section.get('access_count', 0)
        }
        
        weights = {
            'relevance_score': 0.4,
            'recency': 0.2,
            'context_density': 0.3,
            'access_count': 0.1
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

class SemanticChunker:
    """Chunking semántico avanzado"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_content(self, content: str) -> List[Dict]:
        """Divide contenido en chunks semánticos"""
        if not content:
            return []
        
        # Dividir por secciones primero
        sections = self._extract_sections(content)
        chunks = []
        
        for section_id, section_content in sections.items():
            section_chunks = self._chunk_section(section_id, section_content)
            chunks.extend(section_chunks)
        
        return chunks

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extrae secciones del contenido"""
        sections = {}
        pattern = r'<!-- SECTION_ID:\s*([^>]+)\s*-->(.*?)(?=<!-- SECTION_ID:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for section_id, section_content in matches:
            sections[section_id.strip()] = section_content.strip()
        
        return sections

    def _chunk_section(self, section_id: str, content: str) -> List[Dict]:
        """Divide una sección en chunks"""
        if len(content) <= self.chunk_size:
            return [{
                'id': f"{section_id}_chunk_0",
                'section_id': section_id,
                'content': content,
                'start_pos': 0,
                'end_pos': len(content)
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Ajustar límite para evitar cortar palabras
            if end < len(content):
                # Buscar último espacio o salto de línea
                last_space = content.rfind(' ', start, end)
                last_newline = content.rfind('\n', start, end)
                if last_space > start or last_newline > start:
                    end = max(last_space, last_newline)
            
            chunk_content = content[start:end]
            chunks.append({
                'id': f"{section_id}_chunk_{chunk_index}",
                'section_id': section_id,
                'content': chunk_content,
                'start_pos': start,
                'end_pos': end
            })
            
            start = max(start + self.chunk_size - self.overlap, end)
            chunk_index += 1
        
        return chunks

class MultiLevelCache:
    """Cache multinivel (L1/L2/Disk)"""
    
    def __init__(self, l1_size: int = 100, l2_size: int = 1000):
        self.l1_cache = {}  # Cache rápido en memoria
        self.l2_cache = {}   # Cache medio en memoria
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.access_counts = defaultdict(int)
        self.last_access = defaultdict(float)
        
        # Cache en disco
        self.disk_cache_dir = Path("cache")
        self.disk_cache_dir.mkdir(exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        current_time = time.time()
        
        # L1 Cache (más rápido)
        if key in self.l1_cache:
            self.access_counts[key] += 1
            self.last_access[key] = current_time
            return self.l1_cache[key]
        
        # L2 Cache
        if key in self.l2_cache:
            self.access_counts[key] += 1
            self.last_access[key] = current_time
            # Promover a L1
            self._promote_to_l1(key)
            return self.l2_cache[key]
        
        # Disk Cache
        disk_value = self._get_from_disk(key)
        if disk_value is not None:
            self.access_counts[key] += 1
            self.last_access[key] = current_time
            # Promover a L2
            self._promote_to_l2(key, disk_value)
            return disk_value
        
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Establece valor en cache"""
        current_time = time.time()
        
        # Guardar en L1
        self.l1_cache[key] = value
        self.access_counts[key] += 1
        self.last_access[key] = current_time
        
        # Guardar en disco
        self._save_to_disk(key, value, ttl)
        
        # Limpiar caches si están llenos
        self._cleanup_caches()

    def _promote_to_l1(self, key: str) -> None:
        """Promueve elemento de L2 a L1"""
        if len(self.l1_cache) < self.l1_size:
            self.l1_cache[key] = self.l2_cache[key]
        else:
            # Reemplazar elemento menos usado
            lru_key = min(self.l1_cache.keys(), key=lambda k: self.last_access[k])
            del self.l1_cache[lru_key]
            self.l1_cache[key] = self.l2_cache[key]

    def _promote_to_l2(self, key: str, value: Any) -> None:
        """Promueve elemento de disco a L2"""
        if len(self.l2_cache) < self.l2_size:
            self.l2_cache[key] = value
        else:
            # Reemplazar elemento menos usado
            lru_key = min(self.l2_cache.keys(), key=lambda k: self.last_access[k])
            del self.l2_cache[lru_key]
            self.l2_cache[key] = value

    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache en disco"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('expires', 0) > time.time():
                        return data.get('value')
                    else:
                        cache_file.unlink()  # Eliminar archivo expirado
        except Exception as e:
            logger.error(f"Error leyendo cache de disco: {e}")
        return None

    def _save_to_disk(self, key: str, value: Any, ttl: int) -> None:
        """Guarda valor en cache de disco"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.json"
            data = {
                'value': value,
                'expires': time.time() + ttl
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error guardando cache en disco: {e}")

    def _cleanup_caches(self) -> None:
        """Limpia caches cuando están llenos"""
        if len(self.l1_cache) > self.l1_size:
            # Eliminar elementos menos usados
            sorted_items = sorted(self.l1_cache.items(), key=lambda x: self.last_access[x[0]])
            for key, _ in sorted_items[:len(self.l1_cache) - self.l1_size]:
                del self.l1_cache[key]

class QueryOptimizer:
    """Optimización de consultas con expansión semántica"""
    
    def __init__(self, cache=None):
        self.cache = cache  # Inyección de dependencia del cache
        # Sinónimos específicos del dominio médico y SoftMedic
        self.synonyms = {
            # Términos médicos
            'paciente': ['enfermo', 'usuario', 'cliente', 'persona', 'individuo'],
            'medico': ['doctor', 'profesional', 'especialista', 'facultativo', 'galeno'],
            'cita': ['consulta', 'turno', 'appointment', 'visita', 'sesión'],
            'historia_clinica': ['historial médico', 'expediente', 'ficha médica', 'registro clínico'],
            'diagnostico': ['diagnóstico', 'evaluación', 'valoración', 'análisis clínico'],
            'tratamiento': ['terapia', 'medicación', 'prescripción', 'plan terapéutico'],
            
            # Términos técnicos
            'codigo': ['código', 'programa', 'desarrollo', 'implementación', 'script'],
            'arquitectura': ['estructura', 'diseño', 'organización', 'sistema', 'framework'],
            'base_datos': ['database', 'bd', 'almacenamiento', 'persistencia', 'repositorio'],
            'api': ['endpoint', 'servicio', 'interfaz', 'rest', 'webservice'],
            'modelo': ['entidad', 'clase', 'objeto', 'estructura de datos'],
            'vista': ['template', 'plantilla', 'interfaz', 'ui', 'frontend'],
            
            # Términos de negocio
            'negocio': ['modelo', 'comercial', 'empresa', 'ventas', 'business'],
            'facturacion': ['billing', 'cobro', 'pago', 'transacción', 'monetización'],
            'consultorio': ['clínica', 'centro médico', 'hospital', 'ambulatorio'],
            'escalabilidad': ['crecimiento', 'expansión', 'scaling', 'ampliación'],
            
            # Términos técnicos avanzados
            'tecnologia': ['tecnología', 'stack', 'herramientas', 'plataforma', 'tech'],
            'seguridad': ['protección', 'privacidad', 'confidencialidad', 'encriptación'],
            'performance': ['rendimiento', 'velocidad', 'eficiencia', 'optimización'],
            'cache': ['caché', 'almacenamiento temporal', 'buffer', 'memoria'],
            'migracion': ['migración', 'actualización', 'upgrade', 'transición']
        }
        
        # Términos relacionados por contexto
        self.related_terms = {
            # Ecosistema Django
            'django': ['python', 'web', 'framework', 'backend', 'orm', 'mvc'],
            'postgresql': ['base de datos', 'sql', 'datos', 'almacenamiento', 'postgres'],
            'tailwind': ['css', 'estilos', 'diseño', 'frontend', 'ui'],
            'gunicorn': ['servidor', 'wsgi', 'deployment', 'producción'],
            
            # Módulos SoftMedic
            'pacientes': ['médico', 'consulta', 'historia clínica', 'cita', 'registro'],
            'citas': ['agenda', 'calendario', 'programación', 'horarios', 'turnos'],
            'facturacion': ['pago', 'cobro', 'dinero', 'transacción', 'finanzas'],
            'historia_clinica': ['diagnóstico', 'tratamiento', 'evolución', 'antecedentes'],
            'almacen': ['inventario', 'productos', 'stock', 'medicamentos', 'insumos'],
            'ecografias': ['imágenes', 'estudios', 'diagnóstico por imágenes', 'ultrasonido'],
            'estadisticas': ['métricas', 'reportes', 'análisis', 'dashboard', 'kpi'],
            'finanzas': ['contabilidad', 'ingresos', 'egresos', 'balance', 'flujo de caja'],
            
            # Arquitectura y patrones
            'mvc': ['modelo', 'vista', 'controlador', 'separación de responsabilidades'],
            'repository': ['patrón', 'abstracción', 'datos', 'persistencia'],
            'observer': ['eventos', 'notificaciones', 'signals', 'listeners'],
            'strategy': ['algoritmos', 'comportamiento', 'polimorfismo'],
            
            # Seguridad médica
            'hipaa': ['privacidad', 'datos médicos', 'confidencialidad', 'compliance'],
            'auditoria': ['logs', 'trazabilidad', 'seguimiento', 'registro de cambios'],
            'permisos': ['roles', 'autorización', 'acceso', 'privilegios'],
            
            # Performance y escalabilidad
            'cache': ['redis', 'memcached', 'optimización', 'velocidad'],
            'indexing': ['índices', 'búsqueda', 'consultas', 'performance'],
            'concurrencia': ['threads', 'procesos', 'paralelismo', 'async'],
            
            # UI/UX médico
            'responsive': ['móvil', 'tablet', 'adaptativo', 'multi-dispositivo'],
            'accesibilidad': ['a11y', 'usabilidad', 'inclusión', 'wcag'],
            'workflow': ['flujo de trabajo', 'proceso', 'procedimiento', 'protocolo']
        }

    def expand_query(self, query: str, user_context: Optional[Dict] = None) -> List[str]:
        """Expande consulta con sinónimos, términos relacionados y contexto de usuario"""
        query_lower = query.lower()
        expanded_terms = [query_lower]
        
        # Agregar sinónimos con coincidencia parcial
        for term, synonyms in self.synonyms.items():
            if term in query_lower or any(syn in query_lower for syn in synonyms):
                expanded_terms.extend(synonyms)
                expanded_terms.append(term)
        
        # Agregar términos relacionados con coincidencia parcial
        for term, related in self.related_terms.items():
            if term in query_lower or any(rel in query_lower for rel in related):
                expanded_terms.extend(related)
                expanded_terms.append(term)
        
        # Contexto de usuario (si está disponible)
        if user_context:
            # Agregar términos basados en el rol del usuario
            user_role = user_context.get('role', '')
            if user_role == 'medico':
                expanded_terms.extend(['diagnóstico', 'tratamiento', 'paciente', 'historia clínica'])
            elif user_role == 'secretaria':
                expanded_terms.extend(['cita', 'agenda', 'facturación', 'registro'])
            elif user_role == 'administrador':
                expanded_terms.extend(['configuración', 'usuarios', 'permisos', 'sistema'])
            
            # Agregar términos basados en módulos frecuentes del usuario
            frequent_modules = user_context.get('frequent_modules', [])
            for module in frequent_modules:
                if module in self.related_terms:
                    expanded_terms.extend(self.related_terms[module])
        
        # Expansión contextual inteligente
        expanded_terms = self._add_contextual_terms(query_lower, expanded_terms)
        
        # Eliminar duplicados y términos muy cortos
        unique_terms = list(set(term for term in expanded_terms if len(term) > 2))
        
        # Ordenar por relevancia (términos originales primero)
        original_terms = [term for term in unique_terms if term in query_lower]
        other_terms = [term for term in unique_terms if term not in query_lower]
        
        return original_terms + other_terms
    
    def _add_contextual_terms(self, query: str, current_terms: List[str]) -> List[str]:
        """Añade términos contextuales basados en patrones comunes"""
        contextual_terms = current_terms.copy()
        
        # Patrones médicos comunes
        medical_patterns = {
            'crear': ['nuevo', 'registro', 'alta', 'ingreso'],
            'editar': ['modificar', 'actualizar', 'cambiar', 'corregir'],
            'eliminar': ['borrar', 'quitar', 'remover', 'suprimir'],
            'buscar': ['encontrar', 'localizar', 'filtrar', 'consultar'],
            'listar': ['mostrar', 'ver', 'visualizar', 'enumerar'],
            'reportar': ['informe', 'reporte', 'estadística', 'análisis']
        }
        
        for pattern, terms in medical_patterns.items():
            if pattern in query:
                contextual_terms.extend(terms)
        
        # Patrones temporales
        if any(word in query for word in ['hoy', 'ayer', 'mañana', 'semana', 'mes']):
            contextual_terms.extend(['fecha', 'calendario', 'programación', 'agenda'])
        
        # Patrones de urgencia
        if any(word in query for word in ['urgente', 'emergencia', 'prioritario', 'inmediato']):
            contextual_terms.extend(['prioridad', 'crítico', 'importante', 'rápido'])
        
        return contextual_terms

    def extract_context_terms(self, query: str, document_content: str = "") -> Dict[str, List[str]]:
        """Extrae términos de contexto para análisis semántico avanzado"""
        query_lower = query.lower()
        
        context_terms = {
            'technical_terms': [],
            'medical_terms': [],
            'business_terms': [],
            'variable_names': [],
            'module_names': [],
            'action_terms': [],
            'temporal_terms': [],
            'priority_terms': []
        }
        
        # Patrones para términos técnicos
        technical_patterns = [
            r'\b(django|postgresql|python|api|rest|orm|mvc)\b',
            r'\b(class|function|method|model|view|template)\b',
            r'\b(database|query|migration|index|foreign_key)\b',
            r'\b(cache|session|authentication|authorization)\b'
        ]
        
        # Patrones para términos médicos
        medical_patterns = [
            r'\b(paciente|médico|doctor|cita|consulta)\b',
            r'\b(historia_clínica|diagnóstico|tratamiento|medicamento)\b',
            r'\b(ecografía|radiografía|laboratorio|análisis)\b',
            r'\b(síntoma|enfermedad|patología|terapia)\b'
        ]
        
        # Patrones para términos de negocio
        business_patterns = [
            r'\b(facturación|pago|cobro|precio|tarifa)\b',
            r'\b(consultorio|clínica|hospital|centro_médico)\b',
            r'\b(licencia|suscripción|saas|escalabilidad)\b',
            r'\b(ingresos|egresos|balance|rentabilidad)\b'
        ]
        
        # Patrones para nombres de variables/funciones (snake_case, camelCase)
        variable_patterns = [
            r'\b[a-z_][a-z0-9_]*\b',  # snake_case
            r'\b[a-z][a-zA-Z0-9]*\b'   # camelCase
        ]
        
        # Patrones para módulos de SoftMedic
        module_patterns = [
            r'\b(pacientes|medicos|citas|historia_clinica)\b',
            r'\b(facturacion|almacen|ecografias|estadisticas)\b',
            r'\b(finanzas|promociones|puntos|correos)\b'
        ]
        
        # Patrones para acciones
        action_patterns = [
            r'\b(crear|editar|eliminar|buscar|listar)\b',
            r'\b(guardar|actualizar|modificar|borrar)\b',
            r'\b(generar|procesar|calcular|validar)\b'
        ]
        
        # Patrones temporales
        temporal_patterns = [
            r'\b(hoy|ayer|mañana|semana|mes|año)\b',
            r'\b(fecha|hora|tiempo|calendario|agenda)\b',
            r'\b(programar|agendar|citar|turno)\b'
        ]
        
        # Patrones de prioridad
        priority_patterns = [
            r'\b(urgente|emergencia|prioritario|crítico)\b',
            r'\b(importante|inmediato|rápido|lento)\b'
        ]
        
        # Extraer términos usando patrones
        text_to_analyze = f"{query_lower} {document_content.lower()}"
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['technical_terms'].extend(matches)
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['medical_terms'].extend(matches)
        
        for pattern in business_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['business_terms'].extend(matches)
        
        for pattern in module_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['module_names'].extend(matches)
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['action_terms'].extend(matches)
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['temporal_terms'].extend(matches)
        
        for pattern in priority_patterns:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            context_terms['priority_terms'].extend(matches)
        
        # Extraer nombres de variables/funciones solo de la query (más específico)
        for pattern in variable_patterns:
            matches = re.findall(pattern, query_lower)
            # Filtrar palabras comunes y muy cortas
            filtered_matches = [
                match for match in matches 
                if len(match) > 3 and match not in ['para', 'como', 'este', 'esta', 'donde', 'cuando']
            ]
            context_terms['variable_names'].extend(filtered_matches)
        
        # Eliminar duplicados y limpiar
        for category in context_terms:
            context_terms[category] = list(set(context_terms[category]))
            # Remover términos vacíos
            context_terms[category] = [term for term in context_terms[category] if term.strip()]
        
        # Añadir términos derivados basados en el contexto
        context_terms = self._add_derived_terms(context_terms)
        
        return context_terms
    
    def _add_derived_terms(self, context_terms: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Añade términos derivados basados en el contexto extraído"""
        
        # Si hay términos médicos, añadir términos relacionados
        if context_terms['medical_terms']:
            context_terms['medical_terms'].extend([
                'expediente', 'ficha', 'antecedentes', 'evolución',
                'prescripción', 'receta', 'dosis', 'posología'
            ])
        
        # Si hay términos técnicos, añadir términos de desarrollo
        if context_terms['technical_terms']:
            context_terms['technical_terms'].extend([
                'deployment', 'testing', 'debugging', 'refactoring',
                'optimization', 'scalability', 'performance', 'security'
            ])
        
        # Si hay términos de negocio, añadir términos financieros
        if context_terms['business_terms']:
            context_terms['business_terms'].extend([
                'roi', 'revenue', 'profit', 'cost', 'budget',
                'investment', 'growth', 'market', 'customer'
            ])
        
        # Si hay acciones, añadir verbos relacionados
        if context_terms['action_terms']:
            context_terms['action_terms'].extend([
                'implementar', 'desarrollar', 'configurar', 'instalar',
                'ejecutar', 'monitorear', 'analizar', 'optimizar'
            ])
        
        return context_terms

    def check_semantic_cache(self, query: str, similarity_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """Verifica el caché semántico para consultas similares"""
        query_normalized = self.normalize_query(query)
        
        # Buscar en caché por coincidencia exacta primero
        exact_match = self.cache.get(f"query:{query_normalized}")
        if exact_match and not self._is_cache_expired(exact_match):
            return exact_match
        
        # Buscar coincidencias semánticas
        for cached_key in self.cache.l1_cache.keys():
            if cached_key.startswith("query:"):
                cached_query = cached_key[6:]  # Remover prefijo "query:"
                similarity = self._calculate_semantic_similarity(query_normalized, cached_query)
                
                if similarity >= similarity_threshold:
                    cached_result = self.cache.get(cached_key)
                    if cached_result and not self._is_cache_expired(cached_result):
                        # Actualizar estadísticas de hit semántico
                        cached_result['semantic_hit'] = True
                        cached_result['similarity_score'] = similarity
                        return cached_result
        
        return None
    
    def _calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """Calcula similitud semántica entre dos consultas"""
        # Tokenizar y normalizar
        tokens1 = set(self._extract_keywords(query1))
        tokens2 = set(self._extract_keywords(query2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Similitud de Jaccard básica
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Expandir con sinónimos para mejor comparación
        expanded_tokens1 = self._expand_tokens_with_synonyms(tokens1)
        expanded_tokens2 = self._expand_tokens_with_synonyms(tokens2)
        
        # Similitud expandida
        expanded_intersection = len(expanded_tokens1.intersection(expanded_tokens2))
        expanded_union = len(expanded_tokens1.union(expanded_tokens2))
        expanded_similarity = expanded_intersection / expanded_union if expanded_union > 0 else 0.0
        
        # Combinar ambas métricas (70% Jaccard, 30% expandida)
        final_similarity = (jaccard_similarity * 0.7) + (expanded_similarity * 0.3)
        
        return final_similarity
    
    def _expand_tokens_with_synonyms(self, tokens: Set[str]) -> Set[str]:
        """Expande tokens con sinónimos para mejor comparación semántica"""
        expanded = set(tokens)
        
        for token in tokens:
            if token in self.synonyms:
                expanded.update(self.synonyms[token])
            
            # Buscar token como sinónimo en otros grupos
            for main_term, synonyms in self.synonyms.items():
                if token in synonyms:
                    expanded.add(main_term)
                    expanded.update(synonyms)
        
        return expanded
    
    def _is_cache_expired(self, cached_item: Dict[str, Any]) -> bool:
        """Verifica si un elemento del caché ha expirado"""
        if 'timestamp' not in cached_item or 'ttl' not in cached_item:
            return True
        
        current_time = time.time()
        expiry_time = cached_item['timestamp'] + cached_item['ttl']
        
        return current_time > expiry_time
    
    def normalize_query(self, query: str) -> str:
        """Normaliza consulta para consistencia"""
        # Convertir a minúsculas
        normalized = query.lower().strip()
        
        # Remover espacios múltiples
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remover caracteres especiales excepto espacios y guiones
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        # Ordenar palabras para consistencia (opcional, puede afectar el contexto)
        # words = normalized.split()
        # normalized = ' '.join(sorted(words))
        
        return normalized

    def calculate_term_weights(self, terms: List[str], context_terms: Dict[str, List[str]], 
                             original_query: str) -> Dict[str, float]:
        """Calcula pesos inteligentes para términos basado en origen y contexto"""
        weights = {}
        
        # Extraer términos originales de la consulta
        original_terms = set(self._extract_keywords(original_query.lower()))
        
        for term in terms:
            weight = 1.0  # Peso base
            
            # Factor 1: Origen del término
            if term in original_terms:
                weight *= 2.0  # Términos originales tienen mayor peso
            
            # Factor 2: Longitud del término (términos más largos son más específicos)
            if len(term) > 6:
                weight *= 1.3
            elif len(term) > 4:
                weight *= 1.1
            elif len(term) <= 2:
                weight *= 0.7  # Términos muy cortos tienen menor peso
            
            # Factor 3: Categoría del contexto
            for category, category_terms in context_terms.items():
                if term in category_terms:
                    if category == 'medical_terms':
                        weight *= 1.5  # Términos médicos son importantes
                    elif category == 'technical_terms':
                        weight *= 1.3  # Términos técnicos son relevantes
                    elif category == 'module_names':
                        weight *= 1.4  # Módulos específicos son importantes
                    elif category == 'priority_terms':
                        weight *= 1.6  # Términos de prioridad son críticos
                    elif category == 'action_terms':
                        weight *= 1.2  # Acciones son relevantes
                    elif category == 'business_terms':
                        weight *= 1.1  # Términos de negocio son útiles
                    break
            
            # Factor 4: Frecuencia en sinónimos (términos con muchos sinónimos son más generales)
            if term in self.synonyms:
                synonym_count = len(self.synonyms[term])
                if synonym_count > 5:
                    weight *= 0.9  # Términos muy generales tienen menor peso
                elif synonym_count > 2:
                    weight *= 1.1  # Términos con algunos sinónimos son útiles
            
            # Factor 5: Términos relacionados (términos con muchas relaciones son centrales)
            if term in self.related_terms:
                related_count = len(self.related_terms[term])
                if related_count > 8:
                    weight *= 1.2  # Términos muy conectados son importantes
                elif related_count > 4:
                    weight *= 1.1
            
            # Normalizar peso (evitar pesos excesivamente altos)
            weight = min(weight, 3.0)
            weights[term] = round(weight, 2)
        
        return weights

    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Optimiza consulta para mejor búsqueda con caché y expansión completa"""
        # Generar ID único para seguimiento
        query_id = f"query_{int(time.time() * 1000)}_{hash(query) % 10000}"
        
        # Verificar caché semántico primero
        cached_result = self.check_semantic_cache(query)
        if cached_result:
            cached_result['query_id'] = query_id
            cached_result['cache_hit'] = True
            return cached_result
        
        # Procesar consulta completa
        expanded_terms = self.expand_query(query)
        context_terms = self.extract_context_terms(query)
        keywords = self._extract_keywords(query)
        query_type = self._classify_query(query)
        
        # Calcular pesos de términos
        all_terms = list(set(expanded_terms + keywords))
        term_weights = self.calculate_term_weights(all_terms, context_terms, query)
        
        result = {
            'query_id': query_id,
            'original_query': query,
            'normalized_query': self.normalize_query(query),
            'expanded_terms': expanded_terms,
            'context_terms': context_terms,
            'keywords': keywords,
            'query_type': query_type,
            'term_weights': term_weights,
            'cache_hit': False,
            'timestamp': time.time(),
            'ttl': 3600  # 1 hora de TTL
        }
        
        # Guardar en caché
        cache_key = f"query:{self.normalize_query(query)}"
        # Aquí necesitaríamos acceso al cache, que debería ser inyectado
        # self.cache.set(cache_key, result, ttl=3600)
        
        return result

    def _extract_keywords(self, query: str) -> List[str]:
        """Extrae palabras clave de la consulta"""
        # Remover palabras comunes
        stop_words = {'el', 'la', 'de', 'del', 'en', 'con', 'para', 'por', 'como', 'qué', 'cómo', 'cuál', 'cuáles'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _classify_query(self, query: str) -> str:
        """Clasifica el tipo de consulta con categorías específicas del dominio médico"""
        query_lower = query.lower()
        
        # Categorías específicas de SoftMedic
        medical_keywords = ['paciente', 'médico', 'doctor', 'cita', 'consulta', 'historia clínica', 
                           'diagnóstico', 'tratamiento', 'ecografía', 'facturación médica']
        
        business_keywords = ['negocio', 'modelo', 'comercial', 'ventas', 'ingresos', 'licencias',
                           'saas', 'escalabilidad', 'mercado', 'clínica', 'hospital', 'consultorio']
        
        architecture_keywords = ['arquitectura', 'estructura', 'diseño', 'sistema', 'modular',
                               'django', 'postgresql', 'mvc', 'repository', 'observer', 'strategy']
        
        coding_keywords = ['código', 'programa', 'desarrollo', 'implementación', 'función',
                          'clase', 'método', 'api', 'endpoint', 'modelo', 'vista', 'template']
        
        technology_keywords = ['tecnología', 'stack', 'herramientas', 'framework', 'base de datos',
                             'servidor', 'gunicorn', 'nginx', 'docker', 'rest', 'tailwind']
        
        security_keywords = ['seguridad', 'autenticación', 'autorización', 'permisos', 'roles',
                           'encriptación', 'datos sensibles', 'hipaa', 'privacidad', 'auditoría']
        
        data_keywords = ['datos', 'base de datos', 'modelo', 'migración', 'query', 'sql',
                        'postgresql', 'sqlite', 'backup', 'respaldo', 'índice']
        
        ui_ux_keywords = ['interfaz', 'usuario', 'diseño', 'ui', 'ux', 'template', 'css',
                         'responsive', 'móvil', 'accesibilidad', 'usabilidad']
        
        performance_keywords = ['rendimiento', 'performance', 'optimización', 'velocidad',
                              'cache', 'memoria', 'cpu', 'escalabilidad', 'concurrencia']
        
        # Clasificación por prioridad (más específico primero)
        if any(word in query_lower for word in medical_keywords):
            return 'medical'
        elif any(word in query_lower for word in security_keywords):
            return 'security'
        elif any(word in query_lower for word in data_keywords):
            return 'data'
        elif any(word in query_lower for word in performance_keywords):
            return 'performance'
        elif any(word in query_lower for word in ui_ux_keywords):
            return 'ui_ux'
        elif any(word in query_lower for word in coding_keywords):
            return 'coding'
        elif any(word in query_lower for word in architecture_keywords):
            return 'architecture'
        elif any(word in query_lower for word in business_keywords):
            return 'business'
        elif any(word in query_lower for word in technology_keywords):
            return 'technology'
        else:
            return 'general'

class RateLimiter:
    """Rate limiting adaptativo"""
    
    def __init__(self, max_requests_per_second: int = 10, max_requests_per_minute: int = 100):
        self.max_rps = max_requests_per_second
        self.max_rpm = max_requests_per_minute
        self.requests = deque()
        self.penalties = defaultdict(int)
        self.last_penalty_reset = time.time()

    def is_allowed(self, client_id: str = "default") -> bool:
        """Verifica si la request está permitida"""
        current_time = time.time()
        
        # Resetear penalizaciones cada hora
        if current_time - self.last_penalty_reset > 3600:
            self.penalties.clear()
            self.last_penalty_reset = current_time
        
        # Verificar penalización
        if self.penalties[client_id] > 0:
            self.penalties[client_id] -= 1
            return False
        
        # Limpiar requests antiguas
        while self.requests and self.requests[0] < current_time - 60:
            self.requests.popleft()
        
        # Verificar límites
        recent_requests = [req_time for req_time in self.requests if req_time > current_time - 1]
        if len(recent_requests) >= self.max_rps:
            self.penalties[client_id] = 10  # Penalizar por 10 requests
            return False
        
        if len(self.requests) >= self.max_rpm:
            self.penalties[client_id] = 30  # Penalizar por 30 requests
            return False
        
        # Registrar request
        self.requests.append(current_time)
        return True

class ResourceMonitor:
    """Monitoreo de recursos del sistema"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'cache_hits': deque(maxlen=100),
            'cache_misses': deque(maxlen=100)
        }
        self.start_time = time.time()
        self.request_count = 0

    def record_metrics(self, response_time: float, cache_hit: bool = False):
        """Registra métricas de performance"""
        self.request_count += 1
        
        # CPU y memoria
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage'].append(memory_percent)
        self.metrics['response_times'].append(response_time)
        
        if cache_hit:
            self.metrics['cache_hits'].append(1)
        else:
            self.metrics['cache_misses'].append(1)

    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas actuales"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime': uptime,
            'request_count': self.request_count,
            'cpu_avg_percent': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'memory_avg_percent': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'response_time_avg': sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0,
            'cache_hit_rate': len(self.metrics['cache_hits']) / (len(self.metrics['cache_hits']) + len(self.metrics['cache_misses'])) if (self.metrics['cache_hits'] or self.metrics['cache_misses']) else 0
        }

class FuzzySearch:
    """Búsqueda fuzzy con n-gramas"""
    
    def __init__(self):
        self.index = {}
        self.ngram_size = 3

    def build_index(self, documents: List[Dict]):
        """Construye índice de búsqueda fuzzy"""
        self.index = {}
        
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_id = doc.get('id', '')
            
            # Generar n-gramas
            ngrams = self._generate_ngrams(content)
            
            for ngram in ngrams:
                if ngram not in self.index:
                    self.index[ngram] = []
                self.index[ngram].append(doc_id)

    def search(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Búsqueda fuzzy con scoring"""
        query_lower = query.lower()
        query_ngrams = self._generate_ngrams(query_lower)
        
        doc_scores = defaultdict(float)
        
        for ngram in query_ngrams:
            if ngram in self.index:
                for doc_id in self.index[ngram]:
                    doc_scores[doc_id] += 1
        
        # Normalizar scores
        results = []
        for doc_id, score in doc_scores.items():
            normalized_score = score / len(query_ngrams)
            if normalized_score >= threshold:
                results.append((doc_id, normalized_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _generate_ngrams(self, text: str) -> List[str]:
        """Genera n-gramas del texto"""
        words = re.findall(r'\b\w+\b', text)
        ngrams = []
        
        for word in words:
            if len(word) >= self.ngram_size:
                for i in range(len(word) - self.ngram_size + 1):
                    ngrams.append(word[i:i + self.ngram_size])
        
        return ngrams

class RelevanceScorer:
    """Scorer de relevancia multifactor"""
    
    def __init__(self):
        self.weights = {
            'exact_match': 1.0,
            'partial_match': 0.7,
            'semantic_match': 0.5,
            'context_density': 0.3,
            'recency': 0.2
        }

    def score_document(self, doc: Dict, query: str, query_optimized: Dict) -> float:
        """Calcula score de relevancia para un documento"""
        content = doc.get('content', '').lower()
        query_lower = query.lower()
        
        scores = {}
        
        # Exact match
        scores['exact_match'] = 1.0 if query_lower in content else 0.0
        
        # Partial match
        query_words = query_optimized.get('keywords', [])
        partial_matches = sum(1 for word in query_words if word in content)
        scores['partial_match'] = partial_matches / len(query_words) if query_words else 0.0
        
        # Semantic match
        expanded_terms = query_optimized.get('expanded_terms', [])
        semantic_matches = sum(1 for term in expanded_terms if term in content)
        scores['semantic_match'] = semantic_matches / len(expanded_terms) if expanded_terms else 0.0
        
        # Context density
        scores['context_density'] = self._calculate_context_density(content)
        
        # Recency
        scores['recency'] = doc.get('recency_score', 0.5)
        
        # Calcular score final
        final_score = sum(score * self.weights[factor] for factor, score in scores.items())
        
        return min(1.0, final_score)

    def _calculate_context_density(self, content: str) -> float:
        """Calcula densidad de contexto"""
        if not content:
            return 0.0
        
        # Métricas de densidad
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

class OptimizedMCPContextServer:
    """Servidor MCP Context Query con TODAS las optimizaciones"""
    
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent
        self.project_root = self.base_path.parent.parent.parent  # Ir al directorio raíz del proyecto SoftMedic
        
        # Archivos de documentación a cargar
        self.documentation_files = [
            self.base_path / "context" / "project-guidelines.md",  # Archivo original
            self.project_root / "README.md",  # README principal
            self.project_root / "docs" / "modules" / "pacientes.md",  # Documentación de pacientes
            self.project_root / "finanzas" / "DOCUMENTACION_FINANZAS.md",  # Documentación de finanzas
            self.project_root / "MIGRATION_REPORT.md",  # Reporte de migración
            self.project_root / "mcp-hub" / "README.md",  # README del MCP
            self.project_root / "mcp-hub" / "CONFIGURACION_COMPLETADA.md",  # Configuración MCP
            self.project_root / "mcp-hub" / "IMPLEMENTACION_COMPLETA.md",  # Estado de implementación y técnicas
            self.project_root / "mcp-hub" / "new-requerimientos.md",  # Requerimientos del sistema (siempre accesible)
            self.project_root / "Master" / "MCP_Complete_Technical_Documentation.md",  # Documentación técnica
        ]
        
        self.index_file = self.base_path / "index" / "keyword-to-sections.json"
        
        # Inicializar componentes de optimización
        self.token_budget = TokenBudgetManager()
        self.semantic_chunker = SemanticChunker()
        self.cache = MultiLevelCache()
        self.query_optimizer = QueryOptimizer(cache=self.cache)
        self.rate_limiter = RateLimiter()
        self.resource_monitor = ResourceMonitor()
        self.fuzzy_search = FuzzySearch()
        self.relevance_scorer = RelevanceScorer()
        
        # Cache de archivos
        self.files_cache = {}
        self.cache_timestamp = 0
        self.cache_ttl = 30
        self.last_docs_mtime = 0
        
        logger.info("Servidor MCP Context Query Optimizado iniciado con documentación completa")

    def _load_files(self):
        """Carga múltiples archivos de documentación con cache multinivel"""
        current_time = time.time()
        # Detectar cambios en archivos de documentación (mtime máximo)
        current_mtime = 0
        try:
            for doc_file in self.documentation_files:
                if doc_file.exists():
                    mtime = doc_file.stat().st_mtime
                    if mtime > current_mtime:
                        current_mtime = mtime
        except Exception as e:
            logger.warning(f"Error obteniendo mtime de documentación: {e}")
        
        # Verificar cache L1/L2/Disk
        cached_data = self.cache.get('project_files')
        # Usar cache solo si TTL no ha expirado y no hay cambios en archivos
        if (
            cached_data 
            and (current_time - self.cache_timestamp) < self.cache_ttl 
            and current_mtime <= self.last_docs_mtime
        ):
            return cached_data
        
        try:
            # Cargar todos los archivos de documentación
            all_content = ""
            loaded_files = []
            
            for doc_file in self.documentation_files:
                if doc_file.exists():
                    try:
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Agregar metadatos del archivo
                            file_header = f"\n\n<!-- FILE: {doc_file.name} -->\n<!-- PATH: {doc_file} -->\n\n"
                            all_content += file_header + content + "\n\n"
                            loaded_files.append(str(doc_file))
                            logger.info(f"Cargado: {doc_file.name} ({len(content)} chars)")
                    except Exception as e:
                        logger.warning(f"Error cargando {doc_file}: {e}")
                else:
                    logger.warning(f"Archivo no encontrado: {doc_file}")
            
            # Cargar índice existente
            index_data = {}
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            
            # Procesar con chunking semántico
            chunks = self.semantic_chunker.chunk_content(all_content)
            
            # Construir índice de búsqueda fuzzy
            if chunks:
                self.fuzzy_search.build_index(chunks)
            
            # Generar índice automático de palabras clave
            auto_index = self._generate_keyword_index(all_content, chunks)
            
            # Combinar índices
            combined_index = {**index_data, **auto_index}
            
            data = {
                'content': all_content,
                'index': combined_index,
                'chunks': chunks,
                'loaded_files': loaded_files,
                'last_updated': current_time,
                'total_chars': len(all_content),
                'total_chunks': len(chunks)
            }
            
            # Cachear datos procesados
            self.cache.set('project_files', data, ttl=600)  # 10 minutos
            self.files_cache = data
            self.cache_timestamp = current_time
            self.last_docs_mtime = current_mtime
            
            logger.info(f"Documentación cargada: {len(loaded_files)} archivos, {len(all_content)} chars, {len(chunks)} chunks")
            return data
            
        except Exception as e:
            logger.error(f"Error cargando archivos: {e}")
            return {'content': '', 'index': {}, 'chunks': [], 'loaded_files': []}
    
    def _generate_keyword_index(self, content: str, chunks: List[Dict]) -> Dict[str, List[str]]:
        """Genera índice automático de palabras clave basado en el contenido"""
        keyword_index = defaultdict(list)
        
        # Palabras clave importantes del dominio médico y técnico
        important_keywords = [
            # Módulos del sistema
            'pacientes', 'medicos', 'citas', 'facturacion', 'finanzas', 'almacen',
            'ecografias', 'estadisticas', 'dashboard', 'historia_clinica',
            
            # Tecnologías
            'django', 'python', 'postgresql', 'sqlite', 'tailwind', 'html',
            'css', 'javascript', 'gunicorn', 'nginx',
            
            # Conceptos médicos
            'consulta', 'diagnostico', 'tratamiento', 'medicamento', 'receta',
            'historial', 'sintoma', 'enfermedad', 'especialidad',
            
            # Conceptos de negocio
            'factura', 'pago', 'precio', 'descuento', 'promocion', 'punto',
            'cliente', 'proveedor', 'inventario', 'stock',
            
            # Conceptos técnicos
            'modelo', 'vista', 'template', 'form', 'url', 'migration',
            'admin', 'signal', 'manager', 'queryset', 'api', 'rest'
        ]
        
        content_lower = content.lower()
        
        for keyword in important_keywords:
            if keyword in content_lower:
                # Encontrar chunks que contienen esta palabra clave
                for i, chunk in enumerate(chunks):
                    if keyword in chunk.get('content', '').lower():
                        keyword_index[keyword].append(f"chunk_{i}")
        
        return dict(keyword_index)

    def _find_relevant_sections(self, query: str, query_optimized: Dict, chunks: List[Dict], index: Dict[str, List[str]]) -> List[Dict]:
        """Encuentra secciones relevantes con optimizaciones"""
        relevant_chunks = []
        
        # 1. Búsqueda en índice tradicional
        query_words = query_optimized.get('keywords', [])
        for word in query_words:
            if word in index:
                for section_id in index[word]:
                    section_chunks = [chunk for chunk in chunks if chunk.get('section_id') == section_id]
                    relevant_chunks.extend(section_chunks)
        
        # 2. Búsqueda fuzzy
        fuzzy_results = self.fuzzy_search.search(query)
        for chunk_id, score in fuzzy_results:
            chunk = next((c for c in chunks if c.get('id') == chunk_id), None)
            if chunk:
                chunk['fuzzy_score'] = score
                if chunk not in relevant_chunks:
                    relevant_chunks.append(chunk)
        
        # 3. Calcular relevancia para cada chunk
        for chunk in relevant_chunks:
            chunk['relevance'] = self.relevance_scorer.score_document(chunk, query, query_optimized)
            chunk['last_updated'] = time.time()
            chunk['access_count'] = chunk.get('access_count', 0) + 1
        
        # 4. Ordenar por relevancia y aplicar token budgeting
        relevant_chunks = self.token_budget.allocate_tokens(relevant_chunks)
        
        return relevant_chunks[:2]  # Máximo 2 secciones

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja inicialización del servidor MCP"""
        logger.info("Inicializando servidor MCP Optimizado")
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "softmedic-context-optimized",
                "version": "2.0.0-optimized"
            }
        }

    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista las herramientas disponibles"""
        logger.info("Listando herramientas disponibles")
        return {
            "tools": [
                {
                    "name": "context_query",
                    "description": "Obtiene fragmentos relevantes del contexto del proyecto basado en una consulta semántica optimizada.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Consulta sobre el contexto del proyecto"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }

    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas con optimizaciones"""
        start_time = time.time()
        
        # Rate limiting
        if not self.rate_limiter.is_allowed():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Rate limit excedido. Por favor, espera un momento antes de hacer otra consulta."
                    }
                ],
                "isError": True
            }
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Llamada a herramienta optimizada: {tool_name}")
        
        if tool_name == "context_query":
            query = arguments.get("query", "")
            logger.info(f"Procesando consulta optimizada: '{query}'")
            
            try:
                # Optimizar consulta
                query_optimized = self.query_optimizer.optimize_query(query)
                
                # Cargar archivos con cache
                files_data = self._load_files()
                chunks = files_data.get('chunks', [])
                index = files_data.get('index', {})
                
                # Encontrar secciones relevantes con optimizaciones
                relevant_chunks = self._find_relevant_sections(query, query_optimized, chunks, index)
                
                # Construir respuesta optimizada
                result_parts = []
                for chunk in relevant_chunks:
                    section_id = chunk.get('section_id', 'unknown')
                    content = chunk.get('content', '')
                    relevance = chunk.get('relevance', 0)
                    
                    result_parts.append(f"**{section_id.replace('_', ' ').title()}** (Relevancia: {relevance:.2f})\n\n{content}")
                
                if result_parts:
                    result = "\n\n---\n\n".join(result_parts)
                else:
                    available_sections = list(set(chunk.get('section_id', '') for chunk in chunks))
                    result = f"No se encontró información relevante para la consulta: '{query}'. Las secciones disponibles son: {', '.join(available_sections)}"
                
                # Registrar métricas
                response_time = time.time() - start_time
                self.resource_monitor.record_metrics(response_time, cache_hit=True)
                
                logger.info(f"Respuesta optimizada generada: {len(result)} caracteres en {response_time:.3f}s")
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error procesando consulta optimizada: {e}")
                response_time = time.time() - start_time
                self.resource_monitor.record_metrics(response_time, cache_hit=False)
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error procesando la consulta: {str(e)}"
                        }
                    ],
                    "isError": True
                }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Herramienta desconocida: {tool_name}"
                }
            ],
            "isError": True
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP con optimizaciones"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.info(f"Procesando request optimizado: {method}")
        
        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            else:
                result = {"error": f"Método no soportado: {method}"}
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error manejando request optimizado: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    def run(self):
        """Ejecuta el servidor MCP optimizado"""
        logger.info("Iniciando servidor MCP Context Query Optimizado...")
        logger.info("Optimizaciones activas:")
        logger.info("  • Token Budgeting Inteligente")
        logger.info("  • Chunking Semántico Avanzado")
        logger.info("  • Cache Multinivel (L1/L2/Disk)")
        logger.info("  • Query Optimization con expansión semántica")
        logger.info("  • Rate Limiting Adaptativo")
        logger.info("  • Resource Monitoring")
        logger.info("  • Fuzzy Search y Relevance Scoring")
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error decodificando JSON: {e}")
                except Exception as e:
                    logger.error(f"Error procesando línea: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Servidor optimizado detenido por usuario")
        except Exception as e:
            logger.error(f"Error en servidor optimizado: {e}")

if __name__ == "__main__":
    server = OptimizedMCPContextServer()
    server.run()
