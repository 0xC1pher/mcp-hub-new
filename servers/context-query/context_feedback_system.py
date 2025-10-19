#!/usr/bin/env python3
"""
Context Feedback Loop System para MCP Hub
Sistema que previene alucinaciones y mantiene coherencia del proyecto
mediante retroalimentación continua del contexto.

REGLAS PRIORITARIAS:
1. Antes de crear código, debe revisar el código existente
2. Después de analizar el proyecto debe poder crear las funciones/módulos que falten
3. No puede duplicar código
4. Debe tomar 2 tareas, luego regresar al contexto por 1 tarea
5. Siempre debe leer feature.md antes de dar respuestas
"""

import json
import time
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CONTEXT_REVIEW = "context_review"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Task:
    id: str
    content: str
    status: TaskStatus
    priority: TaskPriority
    created_at: float
    dependencies: List[str]
    context_data: Dict[str, Any]
    code_review_required: bool = True
    
class ContextFeedbackSystem:
    """Sistema de retroalimentación de contexto que previene alucinaciones"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.context_file = self.project_root / "context" / "project-guidelines.md"
        self.feature_file = self.project_root / "feature.md"
        self.feedback_file = self.project_root / "context_feedback.json"
        self.task_queue = deque()
        self.completed_tasks = []
        self.context_cache = {}
        self.code_analysis_cache = {}
        self.task_counter = 0
        self.context_review_counter = 0
        
        # Cargar estado previo si existe
        self._load_state()
        
    def _load_state(self):
        """Carga el estado previo del sistema"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.task_counter = data.get('task_counter', 0)
                    self.context_review_counter = data.get('context_review_counter', 0)
                    
                    # Reconstruir tareas
                    for task_data in data.get('tasks', []):
                        task = Task(
                            id=task_data['id'],
                            content=task_data['content'],
                            status=TaskStatus(task_data['status']),
                            priority=TaskPriority(task_data['priority']),
                            created_at=task_data['created_at'],
                            dependencies=task_data['dependencies'],
                            context_data=task_data['context_data'],
                            code_review_required=task_data.get('code_review_required', True)
                        )
                        
                        if task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                            self.task_queue.append(task)
                        else:
                            self.completed_tasks.append(task)
                            
            except Exception as e:
                logger.warning(f"Error cargando estado previo: {e}")
    
    def _save_state(self):
        """Guarda el estado actual del sistema"""
        try:
            all_tasks = list(self.task_queue) + self.completed_tasks
            data = {
                'task_counter': self.task_counter,
                'context_review_counter': self.context_review_counter,
                'tasks': [asdict(task) for task in all_tasks],
                'last_updated': time.time()
            }
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
    
    def read_feature_requirements(self) -> Dict[str, Any]:
        """
        REGLA PRIORITARIA: Siempre leer feature.md antes de dar respuestas
        """
        if not self.feature_file.exists():
            logger.warning("feature.md no encontrado, creando archivo base")
            self._create_default_feature_file()
        
        try:
            with open(self.feature_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parsear requerimientos del feature.md
            requirements = self._parse_feature_requirements(content)
            
            # Cachear para evitar lecturas repetitivas
            self.context_cache['feature_requirements'] = {
                'content': content,
                'requirements': requirements,
                'last_read': time.time()
            }
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error leyendo feature.md: {e}")
            return {}
    
    def _create_default_feature_file(self):
        """Crea un archivo feature.md base si no existe"""
        default_content = """# Feature Requirements - MCP Hub Context System

## Objetivos Principales
- Prevenir alucinaciones del modelo
- Mantener coherencia del proyecto
- Implementar retroalimentación continua
- Gestionar tareas de forma estructurada

## Reglas Prioritarias
1. **Revisar código existente**: Antes de crear código nuevo, analizar el existente
2. **Análisis antes de creación**: Después de analizar, crear funciones/módulos faltantes
3. **No duplicar código**: Verificar existencia antes de crear
4. **Gestión de tareas**: 2 tareas -> contexto -> 1 tarea -> contexto
5. **Leer feature.md**: Siempre consultar antes de responder

## Funcionalidades Requeridas
- [ ] Sistema de análisis de código existente
- [ ] Cache de contexto inteligente
- [ ] Gestión de cola de tareas
- [ ] Retroalimentación automática
- [ ] Prevención de duplicación

## Métricas de Éxito
- Reducción de alucinaciones > 80%
- Coherencia del código > 95%
- Tiempo de respuesta < 500ms
- Precisión de análisis > 90%
"""
        
        try:
            with open(self.feature_file, 'w', encoding='utf-8') as f:
                f.write(default_content)
            logger.info("Archivo feature.md creado con contenido base")
        except Exception as e:
            logger.error(f"Error creando feature.md: {e}")
    
    def _parse_feature_requirements(self, content: str) -> Dict[str, Any]:
        """Parsea los requerimientos del feature.md"""
        requirements = {
            'objectives': [],
            'priority_rules': [],
            'required_features': [],
            'success_metrics': []
        }
        
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith('## Objetivos'):
                current_section = 'objectives'
            elif line.startswith('## Reglas'):
                current_section = 'priority_rules'
            elif line.startswith('## Funcionalidades'):
                current_section = 'required_features'
            elif line.startswith('## Métricas'):
                current_section = 'success_metrics'
            elif line.startswith('- ') and current_section:
                requirements[current_section].append(line[2:])
        
        return requirements
    
    def analyze_existing_code(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        """
        REGLA PRIORITARIA: Revisar código existente antes de crear nuevo
        """
        if target_path:
            analysis_path = Path(target_path)
        else:
            analysis_path = self.project_root
        
        # Verificar cache
        cache_key = str(analysis_path)
        if cache_key in self.code_analysis_cache:
            cached = self.code_analysis_cache[cache_key]
            if time.time() - cached['timestamp'] < 300:  # 5 minutos
                return cached['analysis']
        
        analysis = {
            'files_analyzed': [],
            'functions_found': [],
            'classes_found': [],
            'imports_found': [],
            'duplicates_detected': [],
            'missing_components': [],
            'recommendations': []
        }
        
        try:
            # Analizar archivos Python
            for py_file in analysis_path.rglob('*.py'):
                if py_file.name.startswith('.') or 'test' in py_file.name:
                    continue
                    
                file_analysis = self._analyze_python_file(py_file)
                analysis['files_analyzed'].append(str(py_file))
                analysis['functions_found'].extend(file_analysis['functions'])
                analysis['classes_found'].extend(file_analysis['classes'])
                analysis['imports_found'].extend(file_analysis['imports'])
            
            # Detectar duplicados
            analysis['duplicates_detected'] = self._detect_duplicates(analysis)
            
            # Identificar componentes faltantes
            analysis['missing_components'] = self._identify_missing_components(analysis)
            
            # Generar recomendaciones
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            # Cachear resultado
            self.code_analysis_cache[cache_key] = {
                'analysis': analysis,
                'timestamp': time.time()
            }
            
            logger.info(f"Análisis de código completado: {len(analysis['files_analyzed'])} archivos")
            
        except Exception as e:
            logger.error(f"Error analizando código: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_python_file(self, file_path: Path) -> Dict[str, List]:
        """Analiza un archivo Python específico"""
        analysis = {
            'functions': [],
            'classes': [],
            'imports': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraer funciones
            functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
            analysis['functions'] = [{'name': func, 'file': str(file_path)} for func in functions]
            
            # Extraer clases
            classes = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:', content)
            analysis['classes'] = [{'name': cls, 'file': str(file_path)} for cls in classes]
            
            # Extraer imports
            imports = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+([\w.,\s]+)', content)
            analysis['imports'] = [imp.strip() for imp_line in imports for imp in imp_line.split(',')]
            
        except Exception as e:
            logger.warning(f"Error analizando {file_path}: {e}")
        
        return analysis
    
    def _detect_duplicates(self, analysis: Dict) -> List[Dict]:
        """Detecta código duplicado"""
        duplicates = []
        
        # Detectar funciones duplicadas
        function_names = [f['name'] for f in analysis['functions_found']]
        for name in set(function_names):
            occurrences = [f for f in analysis['functions_found'] if f['name'] == name]
            if len(occurrences) > 1:
                duplicates.append({
                    'type': 'function',
                    'name': name,
                    'occurrences': occurrences,
                    'count': len(occurrences)
                })
        
        # Detectar clases duplicadas
        class_names = [c['name'] for c in analysis['classes_found']]
        for name in set(class_names):
            occurrences = [c for c in analysis['classes_found'] if c['name'] == name]
            if len(occurrences) > 1:
                duplicates.append({
                    'type': 'class',
                    'name': name,
                    'occurrences': occurrences,
                    'count': len(occurrences)
                })
        
        return duplicates
    
    def _identify_missing_components(self, analysis: Dict) -> List[Dict]:
        """Identifica componentes que faltan según feature.md"""
        missing = []
        
        # Leer requerimientos
        requirements = self.read_feature_requirements()
        
        # Verificar funcionalidades requeridas
        for feature in requirements.get('required_features', []):
            if 'Sistema de análisis' in feature and not any('analyze' in f['name'] for f in analysis['functions_found']):
                missing.append({
                    'type': 'function',
                    'description': 'Sistema de análisis de código',
                    'suggested_name': 'analyze_code_structure'
                })
            
            if 'Cache de contexto' in feature and not any('cache' in f['name'].lower() for f in analysis['functions_found']):
                missing.append({
                    'type': 'class',
                    'description': 'Sistema de cache inteligente',
                    'suggested_name': 'IntelligentCache'
                })
        
        return missing
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        if analysis['duplicates_detected']:
            recommendations.append(f"Eliminar {len(analysis['duplicates_detected'])} duplicados detectados")
        
        if analysis['missing_components']:
            recommendations.append(f"Implementar {len(analysis['missing_components'])} componentes faltantes")
        
        if len(analysis['files_analyzed']) > 10:
            recommendations.append("Considerar refactorización para mejorar organización")
        
        return recommendations
    
    def create_task(self, content: str, priority: TaskPriority = TaskPriority.MEDIUM, 
                   dependencies: List[str] = None, context_data: Dict = None) -> Task:
        """Crea una nueva tarea con análisis de código previo"""
        
        # REGLA: Analizar código existente antes de crear tarea
        code_analysis = self.analyze_existing_code()
        
        task = Task(
            id=f"task_{self.task_counter:04d}",
            content=content,
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=time.time(),
            dependencies=dependencies or [],
            context_data=context_data or {},
            code_review_required=True
        )
        
        # Agregar análisis de código al contexto de la tarea
        task.context_data['code_analysis'] = code_analysis
        task.context_data['feature_requirements'] = self.read_feature_requirements()
        
        self.task_queue.append(task)
        self.task_counter += 1
        
        logger.info(f"Tarea creada: {task.id} - {content[:50]}...")
        self._save_state()
        
        return task
    
    def process_tasks_with_context_feedback(self) -> Dict[str, Any]:
        """
        REGLA PRIORITARIA: 2 tareas -> contexto -> 1 tarea -> contexto
        """
        results = {
            'processed_tasks': [],
            'context_reviews': [],
            'recommendations': []
        }
        
        tasks_processed = 0
        max_tasks_before_context = 2
        
        while self.task_queue and tasks_processed < 10:  # Límite de seguridad
            
            # Procesar hasta 2 tareas
            batch_results = []
            for _ in range(min(max_tasks_before_context, len(self.task_queue))):
                if not self.task_queue:
                    break
                    
                task = self.task_queue.popleft()
                result = self._process_single_task(task)
                batch_results.append(result)
                tasks_processed += 1
            
            results['processed_tasks'].extend(batch_results)
            
            # RETROALIMENTACIÓN DE CONTEXTO después de 2 tareas
            if batch_results:
                context_review = self._perform_context_review(batch_results)
                results['context_reviews'].append(context_review)
                
                # Ajustar siguiente procesamiento basado en contexto
                if context_review.get('complexity_high', False):
                    max_tasks_before_context = 1  # Reducir a 1 tarea si hay alta complejidad
                else:
                    max_tasks_before_context = 2  # Mantener 2 tareas
        
        # Generar recomendaciones finales
        results['recommendations'] = self._generate_final_recommendations(results)
        
        self._save_state()
        return results
    
    def _process_single_task(self, task: Task) -> Dict[str, Any]:
        """Procesa una tarea individual con verificaciones de contexto"""
        
        result = {
            'task_id': task.id,
            'content': task.content,
            'status': 'processing',
            'start_time': time.time(),
            'context_checks': []
        }
        
        try:
            # 1. Verificar dependencias
            if not self._check_dependencies(task):
                result['status'] = 'blocked'
                result['reason'] = 'Dependencias no completadas'
                return result
            
            # 2. Revisar código existente si es necesario
            if task.code_review_required:
                code_review = self.analyze_existing_code()
                result['context_checks'].append({
                    'type': 'code_review',
                    'duplicates_found': len(code_review.get('duplicates_detected', [])),
                    'missing_components': len(code_review.get('missing_components', []))
                })
            
            # 3. Verificar feature requirements
            feature_check = self._verify_feature_compliance(task)
            result['context_checks'].append(feature_check)
            
            # 4. Procesar la tarea
            task.status = TaskStatus.IN_PROGRESS
            processing_result = self._execute_task_logic(task)
            result.update(processing_result)
            
            # 5. Marcar como completada
            task.status = TaskStatus.COMPLETED
            self.completed_tasks.append(task)
            
            result['status'] = 'completed'
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Error procesando tarea {task.id}: {e}")
        
        return result
    
    def _check_dependencies(self, task: Task) -> bool:
        """Verifica que las dependencias de la tarea estén completadas"""
        if not task.dependencies:
            return True
        
        completed_task_ids = {t.id for t in self.completed_tasks}
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)
    
    def _verify_feature_compliance(self, task: Task) -> Dict[str, Any]:
        """Verifica que la tarea cumpla con los requerimientos de feature.md"""
        
        # Leer requerimientos actuales
        requirements = self.read_feature_requirements()
        
        compliance_check = {
            'type': 'feature_compliance',
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Verificar reglas prioritarias
        priority_rules = requirements.get('priority_rules', [])
        
        for rule in priority_rules:
            if 'revisar código existente' in rule.lower() and task.code_review_required:
                if 'code_analysis' not in task.context_data:
                    compliance_check['violations'].append("Falta análisis de código existente")
                    compliance_check['compliant'] = False
            
            if 'no duplicar código' in rule.lower():
                duplicates = task.context_data.get('code_analysis', {}).get('duplicates_detected', [])
                if duplicates:
                    compliance_check['violations'].append(f"Detectados {len(duplicates)} duplicados")
                    compliance_check['compliant'] = False
        
        return compliance_check
    
    def _execute_task_logic(self, task: Task) -> Dict[str, Any]:
        """Ejecuta la lógica específica de la tarea"""
        
        # Simulación de procesamiento de tarea
        # En implementación real, aquí iría la lógica específica
        
        execution_result = {
            'actions_taken': [],
            'files_modified': [],
            'functions_created': [],
            'tests_added': []
        }
        
        # Ejemplo de lógica basada en el contenido de la tarea
        if 'crear función' in task.content.lower():
            execution_result['actions_taken'].append('Función creada')
            execution_result['functions_created'].append('nueva_funcion')
        
        if 'refactorizar' in task.content.lower():
            execution_result['actions_taken'].append('Código refactorizado')
            execution_result['files_modified'].append('archivo_refactorizado.py')
        
        return execution_result
    
    def _perform_context_review(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Realiza revisión de contexto después de procesar tareas"""
        
        self.context_review_counter += 1
        
        review = {
            'review_id': f"context_review_{self.context_review_counter:04d}",
            'timestamp': time.time(),
            'tasks_reviewed': len(batch_results),
            'complexity_high': False,
            'issues_found': [],
            'recommendations': []
        }
        
        # Analizar complejidad de las tareas procesadas
        total_duration = sum(r.get('duration', 0) for r in batch_results)
        error_count = sum(1 for r in batch_results if r.get('status') == 'error')
        
        if total_duration > 10 or error_count > 0:
            review['complexity_high'] = True
            review['issues_found'].append('Alta complejidad o errores detectados')
        
        # Verificar coherencia del contexto
        context_coherence = self._check_context_coherence(batch_results)
        review.update(context_coherence)
        
        # Generar recomendaciones
        if review['complexity_high']:
            review['recommendations'].append('Reducir tareas por lote a 1')
        
        if review['issues_found']:
            review['recommendations'].append('Revisar implementación de tareas problemáticas')
        
        logger.info(f"Revisión de contexto completada: {review['review_id']}")
        
        return review
    
    def _check_context_coherence(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Verifica la coherencia del contexto entre tareas"""
        
        coherence = {
            'coherent': True,
            'coherence_score': 1.0,
            'inconsistencies': []
        }
        
        # Verificar consistencia en archivos modificados
        all_files = []
        for result in batch_results:
            all_files.extend(result.get('files_modified', []))
        
        # Detectar modificaciones conflictivas
        file_counts = defaultdict(int)
        for file in all_files:
            file_counts[file] += 1
        
        for file, count in file_counts.items():
            if count > 1:
                coherence['inconsistencies'].append(f"Archivo {file} modificado {count} veces")
                coherence['coherent'] = False
        
        # Calcular score de coherencia
        if coherence['inconsistencies']:
            coherence['coherence_score'] = max(0.0, 1.0 - len(coherence['inconsistencies']) * 0.2)
        
        return coherence
    
    def _generate_final_recommendations(self, results: Dict) -> List[str]:
        """Genera recomendaciones finales basadas en todos los resultados"""
        
        recommendations = []
        
        # Analizar resultados generales
        total_tasks = len(results['processed_tasks'])
        error_tasks = sum(1 for t in results['processed_tasks'] if t.get('status') == 'error')
        
        if error_tasks > 0:
            recommendations.append(f"Revisar {error_tasks} tareas con errores")
        
        # Analizar revisiones de contexto
        context_reviews = results['context_reviews']
        high_complexity_reviews = sum(1 for r in context_reviews if r.get('complexity_high', False))
        
        if high_complexity_reviews > len(context_reviews) / 2:
            recommendations.append("Considerar dividir tareas más complejas")
        
        # Recomendaciones de coherencia
        avg_coherence = sum(r.get('coherence_score', 1.0) for r in context_reviews) / max(1, len(context_reviews))
        if avg_coherence < 0.8:
            recommendations.append("Mejorar coherencia entre tareas relacionadas")
        
        return recommendations
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema"""
        
        return {
            'task_queue_size': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'context_reviews_performed': self.context_review_counter,
            'cache_status': {
                'context_cache_size': len(self.context_cache),
                'code_analysis_cache_size': len(self.code_analysis_cache)
            },
            'last_feature_read': self.context_cache.get('feature_requirements', {}).get('last_read'),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Evalúa la salud general del sistema"""
        
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Verificar archivos esenciales
        if not self.feature_file.exists():
            health['issues'].append('feature.md no encontrado')
            health['status'] = 'warning'
        
        if not self.context_file.exists():
            health['issues'].append('project-guidelines.md no encontrado')
            health['status'] = 'warning'
        
        # Verificar carga de trabajo
        if len(self.task_queue) > 50:
            health['issues'].append('Cola de tareas muy grande')
            health['recommendations'].append('Procesar tareas pendientes')
        
        return health

# Función principal para uso directo
def main():
    """Función principal para testing del sistema"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar sistema
    project_root = Path(__file__).parent
    feedback_system = ContextFeedbackSystem(str(project_root))
    
    # Crear tareas de ejemplo
    feedback_system.create_task(
        "Implementar sistema de cache inteligente",
        TaskPriority.HIGH,
        context_data={'module': 'cache_system'}
    )
    
    feedback_system.create_task(
        "Refactorizar análisis de código existente",
        TaskPriority.MEDIUM,
        dependencies=['task_0000']
    )
    
    # Procesar tareas con retroalimentación
    results = feedback_system.process_tasks_with_context_feedback()
    
    # Mostrar resultados
    print("=== RESULTADOS DEL PROCESAMIENTO ===")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    
    # Mostrar estado del sistema
    status = feedback_system.get_system_status()
    print("\n=== ESTADO DEL SISTEMA ===")
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
