#!/usr/bin/env python3
"""
Tests para el Context Feedback System
Valida que el sistema previene alucinaciones y mantiene coherencia
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from context_feedback_system import (
    ContextFeedbackSystem, 
    Task, 
    TaskStatus, 
    TaskPriority
)

class TestContextFeedbackSystem(unittest.TestCase):
    """Tests para el sistema de retroalimentación de contexto"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Crear estructura de directorios
        (self.project_root / "context").mkdir(exist_ok=True)
        
        # Crear archivos de prueba
        self._create_test_files()
        
        # Inicializar sistema
        self.feedback_system = ContextFeedbackSystem(str(self.project_root))
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_files(self):
        """Crea archivos de prueba"""
        
        # Crear feature.md de prueba
        feature_content = """# Feature Requirements Test

## Objetivos Principales
- Prevenir alucinaciones
- Mantener coherencia

## Reglas Prioritarias
1. Revisar código existente antes de crear nuevo
2. No duplicar código

## Funcionalidades Requeridas
- [ ] Sistema de análisis de código
- [ ] Cache de contexto inteligente
"""
        
        with open(self.project_root / "feature.md", 'w', encoding='utf-8') as f:
            f.write(feature_content)
        
        # Crear project-guidelines.md de prueba
        guidelines_content = """<!-- SECTION_ID: test_section -->
# Test Section
Contenido de prueba para testing
<!-- SECTION_ID: test_section -->"""
        
        with open(self.project_root / "context" / "project-guidelines.md", 'w', encoding='utf-8') as f:
            f.write(guidelines_content)
        
        # Crear archivo Python de prueba
        python_content = """
def test_function():
    '''Función de prueba'''
    return "test"

class TestClass:
    '''Clase de prueba'''
    def test_method(self):
        return "method"
"""
        
        with open(self.project_root / "test_module.py", 'w', encoding='utf-8') as f:
            f.write(python_content)
    
    def test_read_feature_requirements(self):
        """Test: Lectura de feature.md"""
        
        requirements = self.feedback_system.read_feature_requirements()
        
        # Verificar que se leyeron los requerimientos
        self.assertIn('objectives', requirements)
        self.assertIn('priority_rules', requirements)
        self.assertIn('required_features', requirements)
        
        # Verificar contenido específico
        self.assertIn('Prevenir alucinaciones', str(requirements['objectives']))
        self.assertIn('Revisar código existente', str(requirements['priority_rules']))
    
    def test_analyze_existing_code(self):
        """Test: Análisis de código existente"""
        
        analysis = self.feedback_system.analyze_existing_code()
        
        # Verificar estructura del análisis
        self.assertIn('files_analyzed', analysis)
        self.assertIn('functions_found', analysis)
        self.assertIn('classes_found', analysis)
        self.assertIn('duplicates_detected', analysis)
        
        # Verificar que encontró el archivo de prueba
        files_analyzed = [str(f) for f in analysis['files_analyzed']]
        self.assertTrue(any('test_module.py' in f for f in files_analyzed))
        
        # Verificar que encontró funciones y clases
        function_names = [f['name'] for f in analysis['functions_found']]
        self.assertIn('test_function', function_names)
        
        class_names = [c['name'] for c in analysis['classes_found']]
        self.assertIn('TestClass', class_names)
    
    def test_create_task_with_code_analysis(self):
        """Test: Creación de tarea con análisis de código"""
        
        task = self.feedback_system.create_task(
            "Crear nueva función de prueba",
            TaskPriority.HIGH
        )
        
        # Verificar que la tarea se creó correctamente
        self.assertIsInstance(task, Task)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.priority, TaskPriority.HIGH)
        
        # Verificar que se incluyó análisis de código
        self.assertIn('code_analysis', task.context_data)
        self.assertIn('feature_requirements', task.context_data)
        
        # Verificar que está en la cola
        self.assertIn(task, self.feedback_system.task_queue)
    
    def test_duplicate_detection(self):
        """Test: Detección de código duplicado"""
        
        # Crear archivo con función duplicada
        duplicate_content = """
def test_function():
    '''Función duplicada'''
    return "duplicate"
"""
        
        with open(self.project_root / "duplicate_module.py", 'w', encoding='utf-8') as f:
            f.write(duplicate_content)
        
        analysis = self.feedback_system.analyze_existing_code()
        
        # Verificar que detectó el duplicado
        duplicates = analysis['duplicates_detected']
        self.assertTrue(len(duplicates) > 0)
        
        # Verificar que el duplicado es test_function
        duplicate_names = [d['name'] for d in duplicates]
        self.assertIn('test_function', duplicate_names)
    
    def test_task_processing_with_feedback(self):
        """Test: Procesamiento de tareas con retroalimentación"""
        
        # Crear varias tareas
        task1 = self.feedback_system.create_task("Tarea 1", TaskPriority.HIGH)
        task2 = self.feedback_system.create_task("Tarea 2", TaskPriority.MEDIUM)
        task3 = self.feedback_system.create_task("Tarea 3", TaskPriority.LOW)
        
        # Procesar tareas
        results = self.feedback_system.process_tasks_with_context_feedback()
        
        # Verificar estructura de resultados
        self.assertIn('processed_tasks', results)
        self.assertIn('context_reviews', results)
        self.assertIn('recommendations', results)
        
        # Verificar que se procesaron tareas
        self.assertTrue(len(results['processed_tasks']) > 0)
        
        # Verificar que se realizaron revisiones de contexto
        self.assertTrue(len(results['context_reviews']) > 0)
    
    def test_dependency_checking(self):
        """Test: Verificación de dependencias"""
        
        # Crear tarea sin dependencias
        task1 = self.feedback_system.create_task("Tarea base", TaskPriority.HIGH)
        
        # Crear tarea con dependencia
        task2 = self.feedback_system.create_task(
            "Tarea dependiente", 
            TaskPriority.MEDIUM,
            dependencies=[task1.id]
        )
        
        # Verificar que task2 tiene dependencias
        self.assertEqual(task2.dependencies, [task1.id])
        
        # Verificar verificación de dependencias
        self.assertTrue(self.feedback_system._check_dependencies(task1))
        self.assertFalse(self.feedback_system._check_dependencies(task2))
        
        # Completar task1 y verificar nuevamente
        task1.status = TaskStatus.COMPLETED
        self.feedback_system.completed_tasks.append(task1)
        self.assertTrue(self.feedback_system._check_dependencies(task2))
    
    def test_feature_compliance_verification(self):
        """Test: Verificación de cumplimiento de feature.md"""
        
        # Crear tarea con análisis de código
        task = self.feedback_system.create_task("Test compliance", TaskPriority.HIGH)
        
        # Verificar cumplimiento
        compliance = self.feedback_system._verify_feature_compliance(task)
        
        # Verificar estructura
        self.assertIn('type', compliance)
        self.assertIn('compliant', compliance)
        self.assertIn('violations', compliance)
        
        # Debería ser compliant porque tiene code_analysis
        self.assertTrue(compliance['compliant'])
    
    def test_context_coherence_checking(self):
        """Test: Verificación de coherencia de contexto"""
        
        # Crear resultados de prueba
        batch_results = [
            {
                'task_id': 'task_001',
                'files_modified': ['file1.py'],
                'duration': 1.0,
                'status': 'completed'
            },
            {
                'task_id': 'task_002', 
                'files_modified': ['file1.py', 'file2.py'],
                'duration': 2.0,
                'status': 'completed'
            }
        ]
        
        coherence = self.feedback_system._check_context_coherence(batch_results)
        
        # Verificar estructura
        self.assertIn('coherent', coherence)
        self.assertIn('coherence_score', coherence)
        self.assertIn('inconsistencies', coherence)
        
        # Debería detectar inconsistencia (file1.py modificado 2 veces)
        self.assertFalse(coherence['coherent'])
        self.assertTrue(len(coherence['inconsistencies']) > 0)
    
    def test_system_state_persistence(self):
        """Test: Persistencia del estado del sistema"""
        
        # Crear tarea
        task = self.feedback_system.create_task("Test persistence", TaskPriority.HIGH)
        
        # Verificar que se guardó el estado
        self.assertTrue(self.feedback_system.feedback_file.exists())
        
        # Crear nuevo sistema y verificar que cargó el estado
        new_system = ContextFeedbackSystem(str(self.project_root))
        
        # Verificar que la tarea se cargó
        self.assertEqual(len(new_system.task_queue), 1)
        loaded_task = new_system.task_queue[0]
        self.assertEqual(loaded_task.content, "Test persistence")
    
    def test_cache_functionality(self):
        """Test: Funcionalidad de cache"""
        
        # Primera llamada - debería cachear
        analysis1 = self.feedback_system.analyze_existing_code()
        
        # Verificar que se cacheó
        cache_key = str(self.project_root)
        self.assertIn(cache_key, self.feedback_system.code_analysis_cache)
        
        # Segunda llamada - debería usar cache
        analysis2 = self.feedback_system.analyze_existing_code()
        
        # Deberían ser idénticos
        self.assertEqual(analysis1, analysis2)
    
    def test_system_health_assessment(self):
        """Test: Evaluación de salud del sistema"""
        
        health = self.feedback_system._assess_system_health()
        
        # Verificar estructura
        self.assertIn('status', health)
        self.assertIn('issues', health)
        self.assertIn('recommendations', health)
        
        # Con archivos de prueba, debería estar saludable
        self.assertEqual(health['status'], 'healthy')
    
    def test_missing_components_identification(self):
        """Test: Identificación de componentes faltantes"""
        
        analysis = {
            'functions_found': [{'name': 'existing_func', 'file': 'test.py'}],
            'classes_found': []
        }
        
        missing = self.feedback_system._identify_missing_components(analysis)
        
        # Debería identificar componentes faltantes basado en feature.md
        self.assertIsInstance(missing, list)
        
        # Verificar que identifica cache faltante
        missing_types = [m['type'] for m in missing]
        self.assertIn('class', missing_types)

class TestEnhancedMCPIntegration(unittest.TestCase):
    """Tests para la integración con Enhanced MCP Server"""
    
    def setUp(self):
        """Configuración para tests de integración"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Crear archivos necesarios
        (self.project_root / "context").mkdir(exist_ok=True)
        
        # Crear feature.md mínimo
        with open(self.project_root / "feature.md", 'w') as f:
            f.write("# Test Feature\n## Objetivos\n- Test")
    
    def tearDown(self):
        """Limpieza"""
        shutil.rmtree(self.temp_dir)
    
    @patch('enhanced_mcp_server.ContextFeedbackSystem')
    def test_enhanced_server_initialization(self, mock_feedback_system):
        """Test: Inicialización del servidor mejorado"""
        
        # Mock del sistema de feedback
        mock_instance = MagicMock()
        mock_feedback_system.return_value = mock_instance
        
        # Importar y crear servidor (con mock)
        from enhanced_mcp_server import EnhancedMCPServer
        server = EnhancedMCPServer()
        
        # Verificar que se inicializó el sistema de feedback
        mock_feedback_system.assert_called_once()
        self.assertIsNotNone(server.feedback_system)
    
    def test_tool_list_enhancement(self):
        """Test: Lista de herramientas mejorada"""
        
        # Crear servidor con directorio temporal
        with patch('enhanced_mcp_server.ContextFeedbackSystem'):
            from enhanced_mcp_server import EnhancedMCPServer
            server = EnhancedMCPServer()
            
            # Obtener lista de herramientas
            tools_response = server.handle_tools_list({})
            tools = tools_response.get('tools', [])
            
            # Verificar que incluye nuevas herramientas
            tool_names = [tool['name'] for tool in tools]
            self.assertIn('analyze_code', tool_names)
            self.assertIn('create_task', tool_names)
            self.assertIn('process_tasks', tool_names)

def run_all_tests():
    """Ejecuta todos los tests"""
    
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Crear suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestContextFeedbackSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedMCPIntegration))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Mostrar resumen
    print(f"\n{'='*50}")
    print(f"RESUMEN DE TESTS")
    print(f"{'='*50}")
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Éxito: {result.wasSuccessful()}")
    
    if result.errors:
        print(f"\nERRORES:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    if result.failures:
        print(f"\nFALLOS:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
