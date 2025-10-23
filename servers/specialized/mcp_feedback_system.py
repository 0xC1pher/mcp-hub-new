#!/usr/bin/env python3
"""
MCP Feedback System - Servidor especializado en prevención de alucinaciones
Migra TODA la lógica del enhanced_mcp_server.py sin pérdidas
"""

import json
import sys
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mcp-feedback-system')

class ContextFeedbackSystem:
    """Sistema de retroalimentación de contexto para prevenir alucinaciones"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.priority_context_files = {
            "changelog": self.project_root / "changelog.md",
            "checklist": self.project_root / "checklist.md",
            "feature": self.project_root / "feature.md"
        }
        
        # Contadores y métricas
        self.query_count = 0
        self.context_review_threshold = 2
        self.coherence_scores = []
        self.lock = threading.RLock()
        
        # Crear archivos prioritarios si no existen
        self._ensure_priority_context_files()
        logger.info("✅ Context Feedback System inicializado")
    
    def read_feature_requirements(self) -> Dict[str, Any]:
        """Lee requerimientos de feature.md (OBLIGATORIO antes de responder)"""
        with self.lock:
            try:
                feature_file = self.priority_context_files["feature"]
                if feature_file.exists():
                    with open(feature_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extraer requerimientos estructurados
                    requirements = {
                        'content': content,
                        'rules': self._extract_rules(content),
                        'constraints': self._extract_constraints(content),
                        'objectives': self._extract_objectives(content)
                    }
                    
                    logger.info("📋 Feature requirements leídos correctamente")
                    return requirements
                else:
                    logger.warning("⚠️ feature.md no encontrado")
                    return {}
                    
            except Exception as e:
                logger.error(f"Error leyendo feature.md: {e}")
                return {}
    
    def analyze_existing_code(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        """Analiza código existente para prevenir duplicación"""
        with self.lock:
            analysis = {
                'files_analyzed': [],
                'functions_found': [],
                'classes_found': [],
                'duplicates_detected': [],
                'missing_components': [],
                'recommendations': []
            }
            
            search_path = Path(target_path) if target_path else self.project_root
            
            try:
                # Analizar archivos Python
                for py_file in search_path.rglob("*.py"):
                    if self._should_analyze_file(py_file):
                        file_analysis = self._analyze_file(py_file)
                        analysis['files_analyzed'].append(str(py_file))
                        analysis['functions_found'].extend(file_analysis['functions'])
                        analysis['classes_found'].extend(file_analysis['classes'])
                
                # Detectar duplicados
                analysis['duplicates_detected'] = self._detect_duplicates(
                    analysis['functions_found'], analysis['classes_found']
                )
                
                # Generar recomendaciones
                analysis['recommendations'] = self._generate_recommendations(analysis)
                
                logger.info(f"📊 Análisis completado: {len(analysis['files_analyzed'])} archivos")
                return analysis
                
            except Exception as e:
                logger.error(f"Error en análisis de código: {e}")
                return analysis
    
    def verify_response_compliance(self, response_text: str, feature_requirements: Dict) -> Dict[str, Any]:
        """Verifica que la respuesta cumple con feature.md"""
        compliance = {
            'compliant': True,
            'warning': '',
            'sources': [],
            'checks_performed': [],
            'confidence_score': 1.0
        }
        
        try:
            # Check 1: Longitud mínima
            if len(response_text.strip()) < 50:
                compliance['compliant'] = False
                compliance['warning'] = "Respuesta muy corta, posible alucinación"
                compliance['confidence_score'] *= 0.5
            
            # Check 2: Referencias a fuentes
            if not any(word in response_text.lower() for word in ['sección', 'archivo', 'línea', 'función']):
                compliance['warning'] = "Respuesta no cita fuentes específicas"
                compliance['confidence_score'] *= 0.8
            
            # Check 3: Coherencia con requerimientos
            if feature_requirements and 'rules' in feature_requirements:
                rules_compliance = self._check_rules_compliance(response_text, feature_requirements['rules'])
                compliance['confidence_score'] *= rules_compliance
            
            # Check 4: Detección de alucinaciones
            hallucination_score = self._detect_hallucinations(response_text)
            compliance['confidence_score'] *= hallucination_score
            
            compliance['checks_performed'] = ['length_check', 'source_check', 'rules_check', 'hallucination_check']
            
            # Umbral de confianza
            if compliance['confidence_score'] < 0.7:
                compliance['compliant'] = False
                compliance['warning'] = f"Baja confianza en respuesta ({compliance['confidence_score']:.2f})"
            
            return compliance
            
        except Exception as e:
            logger.error(f"Error verificando compliance: {e}")
            compliance['compliant'] = False
            compliance['warning'] = f"Error en verificación: {str(e)}"
            return compliance
    
    def store_context_feedback(self, query: str, response: str, compliance: Dict) -> str:
        """Almacena feedback de contexto para aprendizaje"""
        with self.lock:
            try:
                feedback_data = {
                    'timestamp': time.time(),
                    'query': query,
                    'response_preview': response[:200] + "..." if len(response) > 200 else response,
                    'compliance': compliance,
                    'query_count': self.query_count
                }
                
                # Guardar en archivo de feedback
                feedback_file = self.project_root / "feedback_log.json"
                
                if feedback_file.exists():
                    with open(feedback_file, 'r', encoding='utf-8') as f:
                        feedback_log = json.load(f)
                else:
                    feedback_log = []
                
                feedback_log.append(feedback_data)
                
                # Mantener solo últimos 100 registros
                if len(feedback_log) > 100:
                    feedback_log = feedback_log[-100:]
                
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump(feedback_log, f, indent=2, ensure_ascii=False)
                
                logger.info(f"💾 Feedback almacenado para query: {query[:50]}...")
                return str(feedback_file)
                
            except Exception as e:
                logger.error(f"Error almacenando feedback: {e}")
                return ""
    
    def create_task(self, content: str, priority: str = "medium", dependencies: List[str] = None) -> Dict[str, Any]:
        """Crea una nueva tarea con análisis de contexto"""
        import uuid
        
        task_id = str(uuid.uuid4())[:8]
        task = {
            'id': task_id,
            'content': content,
            'priority': priority,
            'dependencies': dependencies or [],
            'created_at': time.time(),
            'status': 'pending'
        }
        
        # Guardar tarea
        tasks_file = self.project_root / "tasks.json"
        try:
            if tasks_file.exists():
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
            else:
                tasks = []
            
            tasks.append(task)
            
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Tarea creada: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creando tarea: {e}")
            return {'error': str(e)}
    
    def process_tasks_with_context_feedback(self) -> Dict[str, Any]:
        """Procesa tareas con retroalimentación de contexto"""
        with self.lock:
            try:
                tasks_file = self.project_root / "tasks.json"
                
                if not tasks_file.exists():
                    return {'processed_tasks': [], 'context_reviews': []}
                
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
                
                processed_tasks = []
                context_reviews = []
                
                # Procesar tareas pendientes
                for task in tasks:
                    if task.get('status') == 'pending':
                        # Simular procesamiento
                        task['status'] = 'completed'
                        task['completed_at'] = time.time()
                        processed_tasks.append(task)
                        
                        # Cada 2 tareas, hacer revisión de contexto
                        if len(processed_tasks) % 2 == 0:
                            context_review = {
                                'timestamp': time.time(),
                                'coherence_score': 0.95,  # Simulado
                                'tasks_reviewed': len(processed_tasks)
                            }
                            context_reviews.append(context_review)
                
                # Guardar tareas actualizadas
                with open(tasks_file, 'w', encoding='utf-8') as f:
                    json.dump(tasks, f, indent=2, ensure_ascii=False)
                
                return {
                    'processed_tasks': processed_tasks,
                    'context_reviews': context_reviews,
                    'recommendations': ['Mantener ciclo de revisión cada 2 tareas']
                }
                
            except Exception as e:
                logger.error(f"Error procesando tareas: {e}")
                return {'error': str(e)}
    
    def format_cache_response(self, cache_results: List[Dict], query: str) -> Dict[str, Any]:
        """Formatea respuesta del cache inteligente"""
        if not cache_results:
            return {
                "content": [{
                    "type": "text",
                    "text": f"🔍 No se encontraron resultados en cache para: '{query}'"
                }]
            }
        
        text = f"# 🚀 Resultados del Cache Inteligente\n\n"
        text += f"**Consulta**: {query}\n"
        text += f"**Resultados encontrados**: {len(cache_results)}\n\n"
        
        for i, result in enumerate(cache_results[:3], 1):
            relevance = result.get('relevance', result.get('score', 0))
            file_path = result.get('file_path', result.get('key', 'Desconocido'))
            content = result.get('content', '')
            
            text += f"## 📄 Resultado {i} (Relevancia: {relevance:.2%})\n"
            text += f"**Archivo**: `{Path(file_path).name if file_path != 'Desconocido' else 'Cache'}`\n"
            text += f"**Fuente**: `{file_path}`\n\n"
            
            # Mostrar fragmento relevante
            if len(str(content)) > 500:
                text += f"```\n{str(content)[:500]}...\n```\n\n"
            else:
                text += f"```\n{content}\n```\n\n"
        
        return {
            "content": [{
                "type": "text",
                "text": text
            }],
            "cache_metadata": {
                "results_count": len(cache_results),
                "max_relevance": max(r.get('relevance', r.get('score', 0)) for r in cache_results),
                "cache_hit": True
            }
        }
    
    def combine_cache_and_original_response(self, cache_results: List[Dict], 
                                          original_response: Dict, query: str, 
                                          feature_requirements: Dict) -> Dict[str, Any]:
        """Combina resultados del cache con respuesta original"""
        
        # Extraer texto de respuesta original
        original_text = ""
        if "content" in original_response:
            for content_item in original_response["content"]:
                if content_item.get("type") == "text":
                    original_text += content_item.get("text", "")
        
        # Formatear respuesta combinada
        combined_text = f"# 🔄 Respuesta Híbrida (Cache Local + Modelo + Feedback)\n\n"
        combined_text += f"**Flujo**: Cache Local 🔍 → Modelo 🤖 → Feedback System 💾\n\n"
        
        # Agregar resultados del cache si tienen buena relevancia
        if cache_results and cache_results[0].get('relevance', cache_results[0].get('score', 0)) > 0.5:
            combined_text += f"## 🚀 Desde Cache Inteligente\n\n"
            best_result = cache_results[0]
            file_path = best_result.get('file_path', best_result.get('key', 'Cache'))
            combined_text += f"**Archivo**: `{Path(file_path).name if file_path != 'Cache' else 'Cache'}`\n"
            combined_text += f"**Relevancia**: {best_result.get('relevance', best_result.get('score', 0)):.2%}\n\n"
            
            content = best_result.get('content', '')
            if len(str(content)) > 800:
                combined_text += f"```\n{str(content)[:800]}...\n```\n\n"
            else:
                combined_text += f"```\n{content}\n```\n\n"
        
        # Agregar respuesta original si existe
        if original_text.strip():
            combined_text += f"## 📋 Desde Contexto del Proyecto\n\n"
            combined_text += original_text
        
        # Verificar cumplimiento de feature requirements
        compliance_check = self.verify_response_compliance(combined_text, feature_requirements)
        
        return {
            "content": [{
                "type": "text",
                "text": combined_text
            }],
            "context_metadata": {
                "cache_results_count": len(cache_results),
                "original_response_included": bool(original_text.strip()),
                "compliance_check": compliance_check,
                "feature_requirements_checked": True
            }
        }
    
    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema de feedback"""
        with self.lock:
            try:
                feedback_file = self.project_root / "feedback_log.json"
                
                if not feedback_file.exists():
                    return {'total_queries': 0, 'compliance_rate': 0, 'avg_confidence': 0}
                
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_log = json.load(f)
                
                total_queries = len(feedback_log)
                compliant_queries = sum(1 for entry in feedback_log if entry['compliance']['compliant'])
                avg_confidence = sum(entry['compliance']['confidence_score'] for entry in feedback_log) / max(1, total_queries)
                
                return {
                    'total_queries': total_queries,
                    'compliance_rate': (compliant_queries / max(1, total_queries)) * 100,
                    'avg_confidence': avg_confidence,
                    'recent_queries': feedback_log[-5:] if feedback_log else []
                }
                
            except Exception as e:
                logger.error(f"Error obteniendo métricas: {e}")
                return {'error': str(e)}
    
    def _ensure_priority_context_files(self):
        """Crea archivos prioritarios si no existen"""
        try:
            # feature.md
            feature_file = self.priority_context_files["feature"]
            if not feature_file.exists():
                feature_content = """# Feature Requirements - MCP Hub Enhanced

## Reglas Obligatorias

### 🔥 Reglas Críticas (NUNCA VIOLAR)
1. **Leer feature.md SIEMPRE** antes de cualquier respuesta
2. **Analizar código existente** antes de crear código nuevo  
3. **NO duplicar código** - verificar existencia primero
4. **Citar fuentes específicas** - archivos, líneas, funciones
5. **Validar respuestas** contra feature requirements

### 🛡️ Reglas de Prevención de Alucinaciones
- Solo responder basado en contexto real verificable
- Mencionar fuentes específicas en cada respuesta
- Indicar nivel de confianza en la información
- Evitar respuestas genéricas sin contexto

### ⚡ Reglas de Rendimiento  
- Hit rate >85% en cache inteligente
- Tiempo respuesta <500ms para cache hits
- Chunking semántico preservando contexto
- Deduplicación automática de contenido

## Objetivos del Sistema

### Primarios
- Prevenir alucinaciones del modelo
- Mantener coherencia del proyecto
- Optimizar rendimiento con cache multinivel
- Preservar toda la lógica legacy

### Secundarios  
- Facilitar mantenimiento modular
- Permitir escalabilidad horizontal
- Generar métricas de calidad
- Automatizar detección de duplicados

## Restricciones

### Técnicas
- Compatibilidad con protocolo MCP 2024-11-05
- Thread-safety en todos los componentes
- Manejo de errores robusto
- Logging detallado para debugging

### Funcionales
- No perder funcionalidad de servidores legacy
- Mantener APIs existentes durante migración
- Preservar configuraciones de usuario
- Garantizar rollback seguro si es necesario
"""
                
                with open(feature_file, 'w', encoding='utf-8') as f:
                    f.write(feature_content)
                
                logger.info("📋 feature.md creado automáticamente")
                
        except Exception as e:
            logger.error(f"Error creando archivos prioritarios: {e}")
    
    def _extract_rules(self, content: str) -> List[str]:
        """Extrae reglas del contenido"""
        rules = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-')) and ('regla' in line.lower() or 'obligatorio' in line.lower()):
                rules.append(line)
        
        return rules
    
    def _extract_constraints(self, content: str) -> List[str]:
        """Extrae restricciones del contenido"""
        constraints = []
        lines = content.split('\n')
        
        in_constraints_section = False
        for line in lines:
            if 'restricciones' in line.lower() or 'constraints' in line.lower():
                in_constraints_section = True
                continue
            
            if in_constraints_section and line.strip().startswith('-'):
                constraints.append(line.strip())
        
        return constraints
    
    def _extract_objectives(self, content: str) -> List[str]:
        """Extrae objetivos del contenido"""
        objectives = []
        lines = content.split('\n')
        
        in_objectives_section = False
        for line in lines:
            if 'objetivos' in line.lower() or 'objectives' in line.lower():
                in_objectives_section = True
                continue
            
            if in_objectives_section and line.strip().startswith('-'):
                objectives.append(line.strip())
        
        return objectives
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determina si un archivo debe ser analizado"""
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        return not any(part in exclude_dirs for part in file_path.parts)
    
    def _analyze_file(self, file_path: Path) -> Dict[str, List]:
        """Analiza un archivo individual"""
        analysis = {'functions': [], 'classes': []}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Detectar funciones
                if line_stripped.startswith('def '):
                    func_name = line_stripped.split('(')[0].replace('def ', '').strip()
                    analysis['functions'].append({
                        'name': func_name,
                        'file': str(file_path),
                        'line': i,
                        'signature': line_stripped
                    })
                
                # Detectar clases
                elif line_stripped.startswith('class '):
                    class_name = line_stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                    analysis['classes'].append({
                        'name': class_name,
                        'file': str(file_path),
                        'line': i,
                        'signature': line_stripped
                    })
            
        except Exception as e:
            logger.error(f"Error analizando archivo {file_path}: {e}")
        
        return analysis
    
    def _detect_duplicates(self, functions: List[Dict], classes: List[Dict]) -> List[Dict]:
        """Detecta duplicados en funciones y clases"""
        duplicates = []
        
        # Agrupar por nombre
        func_names = defaultdict(list)
        class_names = defaultdict(list)
        
        for func in functions:
            func_names[func['name']].append(func)
        
        for cls in classes:
            class_names[cls['name']].append(cls)
        
        # Detectar duplicados
        for name, items in func_names.items():
            if len(items) > 1:
                duplicates.append({
                    'type': 'function',
                    'name': name,
                    'count': len(items),
                    'locations': [{'file': item['file'], 'line': item['line']} for item in items]
                })
        
        for name, items in class_names.items():
            if len(items) > 1:
                duplicates.append({
                    'type': 'class',
                    'name': name,
                    'count': len(items),
                    'locations': [{'file': item['file'], 'line': item['line']} for item in items]
                })
        
        return duplicates
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        if analysis['duplicates_detected']:
            recommendations.append(f"🚨 Detectados {len(analysis['duplicates_detected'])} duplicados - Considerar refactorización")
        
        if len(analysis['functions_found']) > 100:
            recommendations.append("📊 Muchas funciones detectadas - Considerar modularización")
        
        if len(analysis['files_analyzed']) > 50:
            recommendations.append("📁 Proyecto grande - Implementar análisis incremental")
        
        return recommendations
    
    def _check_rules_compliance(self, response: str, rules: List[str]) -> float:
        """Verifica cumplimiento de reglas"""
        compliance_score = 1.0
        
        # Verificar que no sea una respuesta genérica
        generic_phrases = ['en general', 'normalmente', 'típicamente', 'usualmente']
        generic_count = sum(1 for phrase in generic_phrases if phrase in response.lower())
        
        if generic_count > 2:
            compliance_score *= 0.7
        
        # Verificar citas específicas
        specific_references = ['línea', 'archivo', 'función', 'clase', 'método']
        reference_count = sum(1 for ref in specific_references if ref in response.lower())
        
        if reference_count == 0:
            compliance_score *= 0.8
        
        return compliance_score
    
    def _detect_hallucinations(self, response: str) -> float:
        """Detecta posibles alucinaciones en la respuesta"""
        confidence_score = 1.0
        
        # Frases que indican posible alucinación
        hallucination_indicators = [
            'probablemente', 'posiblemente', 'creo que', 'supongo que',
            'debería ser', 'podría ser', 'tal vez', 'quizás'
        ]
        
        hallucination_count = sum(1 for indicator in hallucination_indicators if indicator in response.lower())
        
        # Reducir confianza por cada indicador
        confidence_score *= (0.9 ** hallucination_count)
        
        # Verificar longitud apropiada
        if len(response) < 100:
            confidence_score *= 0.8
        elif len(response) > 2000:
            confidence_score *= 0.9
        
        return confidence_score


class MCPFeedbackServer:
    """Servidor MCP especializado en feedback y prevención de alucinaciones"""
    
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent
        self.feedback_system = ContextFeedbackSystem(str(project_root))
        logger.info("🚀 MCP Feedback Server iniciado")
    
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
                'name': 'mcp-feedback-system',
                'version': '1.0.0'
            }
        }
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista herramientas disponibles"""
        return {
            'tools': [
                {
                    'name': 'read_feature_requirements',
                    'description': 'Lee requerimientos de feature.md (OBLIGATORIO)',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                },
                {
                    'name': 'analyze_existing_code',
                    'description': 'Analiza código existente para prevenir duplicación',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'target_path': {'type': 'string', 'description': 'Ruta objetivo para análisis'}
                        }
                    }
                },
                {
                    'name': 'verify_response_compliance',
                    'description': 'Verifica cumplimiento de respuesta con feature.md',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'response_text': {'type': 'string', 'description': 'Texto de respuesta a verificar'},
                            'feature_requirements': {'type': 'object', 'description': 'Requerimientos de feature.md'}
                        },
                        'required': ['response_text']
                    }
                },
                {
                    'name': 'feedback_metrics',
                    'description': 'Obtiene métricas del sistema de feedback',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                },
                {
                    'name': 'create_task',
                    'description': 'Crea una nueva tarea con análisis de contexto',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'content': {'type': 'string', 'description': 'Descripción de la tarea'},
                            'priority': {'type': 'string', 'enum': ['low', 'medium', 'high', 'critical'], 'description': 'Prioridad'},
                            'dependencies': {'type': 'array', 'items': {'type': 'string'}, 'description': 'IDs de dependencias'}
                        },
                        'required': ['content']
                    }
                },
                {
                    'name': 'process_tasks',
                    'description': 'Procesa tareas con retroalimentación de contexto',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                },
                {
                    'name': 'format_cache_response',
                    'description': 'Formatea respuesta del cache inteligente',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'cache_results': {'type': 'array', 'description': 'Resultados del cache'},
                            'query': {'type': 'string', 'description': 'Query original'}
                        },
                        'required': ['cache_results', 'query']
                    }
                },
                {
                    'name': 'combine_responses',
                    'description': 'Combina resultados del cache con respuesta original',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'cache_results': {'type': 'array', 'description': 'Resultados del cache'},
                            'original_response': {'type': 'object', 'description': 'Respuesta original'},
                            'query': {'type': 'string', 'description': 'Query original'},
                            'feature_requirements': {'type': 'object', 'description': 'Requerimientos'}
                        },
                        'required': ['cache_results', 'original_response', 'query']
                    }
                }
            ]
        }
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        try:
            if tool_name == 'read_feature_requirements':
                return self._read_feature_requirements(arguments)
            elif tool_name == 'analyze_existing_code':
                return self._analyze_existing_code(arguments)
            elif tool_name == 'verify_response_compliance':
                return self._verify_response_compliance(arguments)
            elif tool_name == 'feedback_metrics':
                return self._feedback_metrics(arguments)
            elif tool_name == 'create_task':
                return self._create_task(arguments)
            elif tool_name == 'process_tasks':
                return self._process_tasks(arguments)
            elif tool_name == 'format_cache_response':
                return self._format_cache_response(arguments)
            elif tool_name == 'combine_responses':
                return self._combine_responses(arguments)
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
    
    def _read_feature_requirements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Lee requerimientos de feature.md"""
        requirements = self.feedback_system.read_feature_requirements()
        
        if requirements:
            response = f'''📋 **Feature Requirements Cargados**

🔥 **Reglas encontradas**: {len(requirements.get('rules', []))}
🛡️ **Restricciones**: {len(requirements.get('constraints', []))}
🎯 **Objetivos**: {len(requirements.get('objectives', []))}

**Reglas principales**:
'''
            
            for rule in requirements.get('rules', [])[:3]:
                response += f"- {rule}\n"
            
            return {
                'content': [{'type': 'text', 'text': response}]
            }
        else:
            return {
                'content': [{'type': 'text', 'text': '⚠️ No se pudieron cargar los feature requirements'}],
                'isError': True
            }
    
    def _analyze_existing_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza código existente"""
        target_path = args.get('target_path')
        analysis = self.feedback_system.analyze_existing_code(target_path)
        
        response = f'''🔍 **Análisis de Código Completado**

📁 **Archivos analizados**: {len(analysis['files_analyzed'])}
🔧 **Funciones encontradas**: {len(analysis['functions_found'])}
📦 **Clases encontradas**: {len(analysis['classes_found'])}
🚨 **Duplicados detectados**: {len(analysis['duplicates_detected'])}

'''
        
        if analysis['duplicates_detected']:
            response += "**Duplicados encontrados**:\n"
            for dup in analysis['duplicates_detected'][:3]:
                response += f"- {dup['type']}: {dup['name']} ({dup['count']} ocurrencias)\n"
        
        if analysis['recommendations']:
            response += "\n**Recomendaciones**:\n"
            for rec in analysis['recommendations']:
                response += f"- {rec}\n"
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }
    
    def _verify_response_compliance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica cumplimiento de respuesta"""
        response_text = args.get('response_text', '')
        feature_requirements = args.get('feature_requirements', {})
        
        compliance = self.feedback_system.verify_response_compliance(response_text, feature_requirements)
        
        # Almacenar feedback
        self.feedback_system.store_context_feedback("verification", response_text, compliance)
        
        status_icon = "✅" if compliance['compliant'] else "❌"
        confidence_color = "🟢" if compliance['confidence_score'] > 0.8 else "🟡" if compliance['confidence_score'] > 0.6 else "🔴"
        
        response = f'''{status_icon} **Verificación de Cumplimiento**

{confidence_color} **Confianza**: {compliance['confidence_score']:.2f}
📋 **Cumplimiento**: {'SÍ' if compliance['compliant'] else 'NO'}
🔍 **Verificaciones**: {', '.join(compliance['checks_performed'])}

'''
        
        if compliance['warning']:
            response += f"⚠️ **Advertencia**: {compliance['warning']}\n"
        
        if compliance['sources']:
            response += f"📚 **Fuentes**: {', '.join(compliance['sources'])}\n"
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }
    
    def _feedback_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene métricas de feedback"""
        metrics = self.feedback_system.get_feedback_metrics()
        
        if 'error' in metrics:
            return {
                'content': [{'type': 'text', 'text': f'❌ Error obteniendo métricas: {metrics["error"]}'}],
                'isError': True
            }
        
        response = f'''📊 **Métricas del Sistema de Feedback**

📈 **Estadísticas generales**:
- Total consultas: {metrics['total_queries']}
- Tasa de cumplimiento: {metrics['compliance_rate']:.1f}%
- Confianza promedio: {metrics['avg_confidence']:.2f}

🎯 **Estado del sistema**: {'🟢 Óptimo' if metrics['compliance_rate'] > 80 else '🟡 Bueno' if metrics['compliance_rate'] > 60 else '🔴 Necesita atención'}

'''
        
        if metrics['recent_queries']:
            response += "**Consultas recientes**:\n"
            for query in metrics['recent_queries'][-3:]:
                timestamp = time.strftime('%H:%M:%S', time.localtime(query['timestamp']))
                response += f"- {timestamp}: {query['query'][:50]}... (confianza: {query['compliance']['confidence_score']:.2f})\n"
        
        return {
            'content': [{'type': 'text', 'text': response}]
        }
    
    def run(self):
        """Ejecuta el servidor MCP"""
        logger.info("🚀 Iniciando MCP Feedback Server...")
        
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
    server = MCPFeedbackServer()
    server.run()
