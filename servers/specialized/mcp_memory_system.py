#!/usr/bin/env python3
"""
MCP Memory System - Servidor especializado en gestión avanzada de memoria y tokens
Migra TokenBudgetManager, QueryOptimizer y AdvancedScorer del optimized_mcp_server.py
"""

import json
import sys
import logging
import re
import math
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mcp-memory-system')

class TokenBudgetManager:
    """Gestión inteligente de presupuesto de tokens"""
    
    def __init__(self, max_tokens: int = 4000, reserved_tokens: int = 500):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens
        self.lock = threading.RLock()
        
        logger.info(f"✅ Token Budget Manager - Max:{max_tokens}, Available:{self.available_tokens}")

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
        with self.lock:
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


class QueryOptimizer:
    """Optimización de consultas con expansión semántica médica"""
    
    def __init__(self):
        # Sinónimos específicos del dominio médico y técnico
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
            'vista': ['template', 'plantilla', 'interfaz', 'ui', 'frontend']
        }
        
        # Términos relacionados por contexto
        self.related_terms = {
            # Ecosistema Django
            'django': ['python', 'web', 'framework', 'backend', 'orm', 'mvc'],
            'postgresql': ['base de datos', 'sql', 'datos', 'almacenamiento', 'postgres'],
            'tailwind': ['css', 'estilos', 'diseño', 'frontend', 'ui'],
            
            # Módulos médicos
            'pacientes': ['médico', 'consulta', 'historia clínica', 'cita', 'registro'],
            'citas': ['agenda', 'calendario', 'programación', 'horarios', 'turnos'],
            'facturacion': ['pago', 'cobro', 'dinero', 'transacción', 'finanzas'],
            'historia_clinica': ['diagnóstico', 'tratamiento', 'evolución', 'antecedentes']
        }
        
        logger.info("✅ Query Optimizer inicializado con expansión semántica médica")

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
            user_role = user_context.get('role', '')
            if user_role == 'medico':
                expanded_terms.extend(['diagnóstico', 'tratamiento', 'paciente', 'historia clínica'])
            elif user_role == 'secretaria':
                expanded_terms.extend(['cita', 'agenda', 'facturación', 'registro'])
            elif user_role == 'administrador':
                expanded_terms.extend(['configuración', 'usuarios', 'permisos', 'sistema'])
        
        # Eliminar duplicados y términos muy cortos
        unique_terms = list(set(term for term in expanded_terms if len(term) > 2))
        
        # Ordenar por relevancia (términos originales primero)
        original_terms = [term for term in unique_terms if term in query_lower]
        other_terms = [term for term in unique_terms if term not in query_lower]
        
        return original_terms + other_terms


class AdvancedScorer:
    """Sistema avanzado de scoring y relevancia"""
    
    def __init__(self):
        self.scoring_weights = {
            'exact_match': 1.0,
            'partial_match': 0.7,
            'semantic_match': 0.5,
            'context_match': 0.3,
            'frequency_bonus': 0.2
        }
        
        logger.info("✅ Advanced Scorer inicializado")

    def calculate_score(self, query: str, content: str, metadata: Dict = None) -> float:
        """Calcula score de relevancia avanzado"""
        if not query or not content:
            return 0.0
        
        query_lower = query.lower()
        content_lower = content.lower()
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        scores = {}
        
        # Exact match score
        scores['exact_match'] = 1.0 if query_lower in content_lower else 0.0
        
        # Partial match score
        word_matches = len(query_words.intersection(content_words))
        scores['partial_match'] = word_matches / len(query_words) if query_words else 0.0
        
        # Semantic match score (basado en sinónimos)
        semantic_score = self._calculate_semantic_score(query_words, content_words)
        scores['semantic_match'] = semantic_score
        
        # Context match score (basado en metadata)
        context_score = self._calculate_context_score(query, metadata or {})
        scores['context_match'] = context_score
        
        # Frequency bonus
        frequency_score = self._calculate_frequency_score(query_words, content_lower)
        scores['frequency_bonus'] = frequency_score
        
        # Score final ponderado
        final_score = sum(score * self.scoring_weights[key] for key, score in scores.items())
        
        # Normalizar entre 0 y 1
        return min(1.0, final_score)

    def _calculate_semantic_score(self, query_words: Set[str], content_words: Set[str]) -> float:
        """Calcula score semántico basado en sinónimos"""
        # Simplificado: buscar palabras relacionadas
        medical_terms = {'paciente', 'medico', 'cita', 'historia', 'diagnostico', 'tratamiento'}
        tech_terms = {'codigo', 'api', 'base', 'datos', 'sistema', 'framework'}
        
        query_medical = len(query_words.intersection(medical_terms))
        content_medical = len(content_words.intersection(medical_terms))
        
        query_tech = len(query_words.intersection(tech_terms))
        content_tech = len(content_words.intersection(tech_terms))
        
        semantic_matches = 0
        if query_medical > 0 and content_medical > 0:
            semantic_matches += min(query_medical, content_medical)
        if query_tech > 0 and content_tech > 0:
            semantic_matches += min(query_tech, content_tech)
        
        return min(1.0, semantic_matches / max(1, len(query_words)))

    def _calculate_context_score(self, query: str, metadata: Dict) -> float:
        """Calcula score basado en contexto/metadata"""
        context_score = 0.0
        
        # Score basado en tipo de archivo
        file_type = metadata.get('file_type', '')
        if 'python' in query.lower() and file_type == 'py':
            context_score += 0.3
        elif 'config' in query.lower() and file_type in ['json', 'yaml']:
            context_score += 0.3
        
        # Score basado en módulo
        module = metadata.get('module', '')
        if any(word in module.lower() for word in query.lower().split()):
            context_score += 0.4
        
        # Score basado en recencia
        last_modified = metadata.get('last_modified', 0)
        if last_modified > 0:
            days_old = (time.time() - last_modified) / (24 * 3600)
            recency_score = max(0, 1 - (days_old / 30))  # Decae en 30 días
            context_score += recency_score * 0.3
        
        return min(1.0, context_score)

    def _calculate_frequency_score(self, query_words: Set[str], content: str) -> float:
        """Calcula score basado en frecuencia de términos"""
        total_frequency = 0
        for word in query_words:
            frequency = content.count(word)
            # Logaritmo para evitar que frecuencias muy altas dominen
            total_frequency += math.log(1 + frequency)
        
        # Normalizar por número de palabras de query
        return min(1.0, total_frequency / max(1, len(query_words)))


class MCPMemoryServer:
    """Servidor MCP especializado en gestión de memoria y tokens"""
    
    def __init__(self):
        self.token_manager = TokenBudgetManager()
        self.query_optimizer = QueryOptimizer()
        self.scorer = AdvancedScorer()
        
        # Estadísticas
        self.query_count = 0
        self.optimization_stats = defaultdict(int)
        
        logger.info("🚀 MCP Memory Server iniciado")

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja requests MCP"""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        try:
            if method == 'initialize':
                result = self._handle_initialize(params)
            elif method == 'tools/list':
                result = self._handle_tools_list(params)
            elif method == 'tools/call':
                result = self._handle_tools_call(params)
            else:
                result = {'error': f'Método no soportado: {method}'}
            
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error manejando request: {e}")
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'error': {
                    'code': -32603,
                    'message': str(e)
                }
            }

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja inicialización"""
        return {
            'protocolVersion': '2024-11-05',
            'capabilities': {
                'tools': {
                    'listChanged': True
                }
            },
            'serverInfo': {
                'name': 'mcp-memory-system',
                'version': '1.0.0'
            }
        }

    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista herramientas disponibles"""
        return {
            'tools': [
                {
                    'name': 'allocate_tokens',
                    'description': 'Asigna tokens disponibles a secciones priorizadas',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'sections': {'type': 'array', 'description': 'Secciones para priorizar'},
                            'max_tokens': {'type': 'integer', 'description': 'Máximo tokens', 'default': 4000}
                        },
                        'required': ['sections']
                    }
                },
                {
                    'name': 'expand_query',
                    'description': 'Expande consulta con sinónimos y términos relacionados',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Query a expandir'},
                            'user_context': {'type': 'object', 'description': 'Contexto del usuario'}
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'calculate_relevance',
                    'description': 'Calcula score de relevancia avanzado',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Query de búsqueda'},
                            'content': {'type': 'string', 'description': 'Contenido a evaluar'},
                            'metadata': {'type': 'object', 'description': 'Metadata adicional'}
                        },
                        'required': ['query', 'content']
                    }
                },
                {
                    'name': 'memory_stats',
                    'description': 'Obtiene estadísticas del sistema de memoria',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            ]
        }

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        try:
            if tool_name == 'allocate_tokens':
                return self._allocate_tokens(arguments)
            elif tool_name == 'expand_query':
                return self._expand_query(arguments)
            elif tool_name == 'calculate_relevance':
                return self._calculate_relevance(arguments)
            elif tool_name == 'memory_stats':
                return self._memory_stats(arguments)
            else:
                return {
                    'content': [{'type': 'text', 'text': f'Herramienta desconocida: {tool_name}'}],
                    'isError': True
                }
                
        except Exception as e:
            logger.error(f"Error ejecutando herramienta {tool_name}: {e}")
            return {
                'content': [{'type': 'text', 'text': f'Error ejecutando {tool_name}: {str(e)}'}],
                'isError': True
            }

    def _allocate_tokens(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Asigna tokens a secciones"""
        sections = args.get('sections', [])
        max_tokens = args.get('max_tokens', 4000)
        
        # Actualizar configuración si es necesario
        if max_tokens != self.token_manager.max_tokens:
            self.token_manager = TokenBudgetManager(max_tokens)
        
        allocated_sections = self.token_manager.allocate_tokens(sections)
        
        total_tokens_used = sum(
            self.token_manager.estimate_tokens(section.get('content', '')) 
            for section in allocated_sections
        )
        
        response = f'''🧠 **Asignación de Tokens Completada**

📊 **Estadísticas**:
- Secciones evaluadas: {len(sections)}
- Secciones seleccionadas: {len(allocated_sections)}
- Tokens utilizados: {total_tokens_used}/{max_tokens}
- Tokens disponibles: {max_tokens - total_tokens_used}

⚡ **Eficiencia**: {(total_tokens_used/max_tokens)*100:.1f}% del presupuesto utilizado

**Secciones priorizadas**:
'''
        
        for i, section in enumerate(allocated_sections[:3], 1):
            priority = self.token_manager.calculate_priority(section)
            tokens = self.token_manager.estimate_tokens(section.get('content', ''))
            response += f"{i}. Prioridad: {priority:.2f}, Tokens: {tokens}\n"
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }

    def _expand_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Expande query con sinónimos"""
        query = args.get('query', '')
        user_context = args.get('user_context', {})
        
        self.query_count += 1
        expanded_terms = self.query_optimizer.expand_query(query, user_context)
        
        response = f'''🔍 **Expansión de Query Completada**

**Query original**: {query}
**Términos expandidos**: {len(expanded_terms)}

**Términos principales**:
'''
        
        for i, term in enumerate(expanded_terms[:10], 1):
            response += f"{i}. {term}\n"
        
        if len(expanded_terms) > 10:
            response += f"... y {len(expanded_terms) - 10} términos más\n"
        
        response += f"\n✅ **Optimización semántica aplicada** para dominio médico"
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }

    def _calculate_relevance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula relevancia"""
        query = args.get('query', '')
        content = args.get('content', '')
        metadata = args.get('metadata', {})
        
        score = self.scorer.calculate_score(query, content, metadata)
        
        # Clasificar relevancia
        if score > 0.8:
            relevance_level = "🟢 MUY ALTA"
        elif score > 0.6:
            relevance_level = "🟡 ALTA"
        elif score > 0.4:
            relevance_level = "🟠 MEDIA"
        elif score > 0.2:
            relevance_level = "🔴 BAJA"
        else:
            relevance_level = "⚫ MUY BAJA"
        
        response = f'''📊 **Cálculo de Relevancia Completado**

**Query**: {query[:50]}{'...' if len(query) > 50 else ''}
**Score**: {score:.3f}
**Nivel**: {relevance_level}

**Factores evaluados**:
- Coincidencia exacta
- Coincidencia parcial  
- Coincidencia semántica
- Contexto y metadata
- Frecuencia de términos

✅ **Análisis completado con scoring avanzado**
'''
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }

    def _memory_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        response = f'''📊 **Estadísticas del Sistema de Memoria**

🧠 **Token Budget Manager**:
- Tokens máximos: {self.token_manager.max_tokens}
- Tokens reservados: {self.token_manager.reserved_tokens}
- Tokens disponibles: {self.token_manager.available_tokens}

🔍 **Query Optimizer**:
- Sinónimos médicos: {len(self.query_optimizer.synonyms)}
- Términos relacionados: {len(self.query_optimizer.related_terms)}
- Queries procesadas: {self.query_count}

📈 **Advanced Scorer**:
- Pesos de scoring: {len(self.scorer.scoring_weights)} factores
- Algoritmo: Scoring ponderado multinivel

🎯 **Estado**: 🟢 Óptimo - Todos los componentes activos
'''
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }

    def run(self):
        """Ejecuta el servidor MCP"""
        logger.info("🚀 Iniciando MCP Memory Server...")
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = self.handle_request(request)
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error decodificando JSON: {e}")
                except Exception as e:
                    logger.error(f"Error procesando línea: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Servidor detenido por usuario")
        except Exception as e:
            logger.error(f"Error en servidor: {e}")


if __name__ == "__main__":
    server = MCPMemoryServer()
    server.run()
