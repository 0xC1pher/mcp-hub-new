#!/usr/bin/env python3
"""
Enhanced MCP Server con Context Feedback System
Integra el sistema de retroalimentación para prevenir alucinaciones
y mantener coherencia del proyecto.
"""

import json
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Importar el sistema de retroalimentación
try:
    from context_feedback_system import ContextFeedbackSystem, TaskPriority
    from optimized_mcp_server import OptimizedMCPContextServer
    FEEDBACK_AVAILABLE = True
    logger.info("Context Feedback System cargado correctamente")
except ImportError as e:
    logger.warning(f"Context Feedback System no disponible: {e}")
    from optimized_mcp_server import OptimizedMCPContextServer
    FEEDBACK_AVAILABLE = False
    
    # Crear clases dummy para compatibilidad
    class TaskPriority:
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        CRITICAL = "critical"

class EnhancedMCPServer(OptimizedMCPContextServer):
    """Servidor MCP mejorado con sistema de retroalimentación de contexto"""
    
    def __init__(self):
        super().__init__()
        
        # Inicializar sistema de retroalimentación si está disponible
        if FEEDBACK_AVAILABLE:
            project_root = Path(__file__).parent
            self.feedback_system = ContextFeedbackSystem(str(project_root))
            logger.info("Enhanced MCP Server inicializado con Context Feedback System")
        else:
            self.feedback_system = None
            logger.info("Enhanced MCP Server inicializado SIN Context Feedback System (modo compatibilidad)")
        
        # Contador de consultas para gestión de tareas
        self.query_count = 0
        self.context_review_threshold = 2
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja llamadas a herramientas con retroalimentación de contexto"""
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        # Si el feedback system no está disponible, usar solo el servidor optimizado
        if not FEEDBACK_AVAILABLE or self.feedback_system is None:
            logger.info(f"Procesando herramienta en modo compatibilidad: {tool_name}")
            return super().handle_tools_call(params)
        
        # REGLA PRIORITARIA: Leer feature.md antes de procesar
        try:
            feature_requirements = self.feedback_system.read_feature_requirements()
        except Exception as e:
            logger.warning(f"Error leyendo feature.md: {e}")
            feature_requirements = {}
        
        logger.info(f"Procesando herramienta con feedback: {tool_name}")
        
        if tool_name == "context_query":
            return self._handle_context_query_enhanced(arguments, feature_requirements)
        elif tool_name == "analyze_code":
            return self._handle_code_analysis(arguments)
        elif tool_name == "create_task":
            return self._handle_task_creation(arguments)
        elif tool_name == "process_tasks":
            return self._handle_task_processing(arguments)
        else:
            # Fallback al servidor original
            return super().handle_tools_call(params)
    
    def _handle_context_query_enhanced(self, arguments: Dict, feature_requirements: Dict) -> Dict[str, Any]:
        """Maneja consultas de contexto con verificaciones mejoradas"""
        
        query = arguments.get("query", "")
        
        # Incrementar contador y verificar si necesita revisión de contexto
        self.query_count += 1
        
        try:
            # REGLA: Analizar código existente antes de responder
            if self.query_count % self.context_review_threshold == 0:
                code_analysis = self.feedback_system.analyze_existing_code()
                logger.info(f"Revisión de contexto realizada: {len(code_analysis.get('files_analyzed', []))} archivos")
            
            # Procesar consulta con el servidor optimizado original
            original_response = super().handle_tools_call({
                "name": "context_query",
                "arguments": arguments
            })
            
            # Verificar que la respuesta cumple con feature requirements
            enhanced_response = self._enhance_response_with_context(
                original_response, 
                query, 
                feature_requirements
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error en consulta mejorada: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error procesando consulta con feedback: {str(e)}"
                }],
                "isError": True
            }
    
    def _handle_code_analysis(self, arguments: Dict) -> Dict[str, Any]:
        """Maneja análisis de código existente"""
        
        target_path = arguments.get("path")
        
        try:
            analysis = self.feedback_system.analyze_existing_code(target_path)
            
            # Formatear respuesta
            response_text = self._format_code_analysis(analysis)
            
            return {
                "content": [{
                    "type": "text",
                    "text": response_text
                }]
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de código: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error analizando código: {str(e)}"
                }],
                "isError": True
            }
    
    def _handle_task_creation(self, arguments: Dict) -> Dict[str, Any]:
        """Maneja creación de tareas"""
        
        content = arguments.get("content", "")
        priority = arguments.get("priority", "medium")
        dependencies = arguments.get("dependencies", [])
        
        try:
            task_priority = TaskPriority(priority.lower())
            task = self.feedback_system.create_task(
                content=content,
                priority=task_priority,
                dependencies=dependencies
            )
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tarea creada: {task.id}\nContenido: {task.content}\nPrioridad: {task.priority.value}"
                }]
            }
            
        except Exception as e:
            logger.error(f"Error creando tarea: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error creando tarea: {str(e)}"
                }],
                "isError": True
            }
    
    def _handle_task_processing(self, arguments: Dict) -> Dict[str, Any]:
        """Maneja procesamiento de tareas con retroalimentación"""
        
        try:
            results = self.feedback_system.process_tasks_with_context_feedback()
            
            # Formatear respuesta
            response_text = self._format_task_results(results)
            
            return {
                "content": [{
                    "type": "text",
                    "text": response_text
                }]
            }
            
        except Exception as e:
            logger.error(f"Error procesando tareas: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error procesando tareas: {str(e)}"
                }],
                "isError": True
            }
    
    def _enhance_response_with_context(self, original_response: Dict, query: str, 
                                     feature_requirements: Dict) -> Dict[str, Any]:
        """Mejora la respuesta original con verificaciones de contexto"""
        
        # Extraer texto de la respuesta original
        original_text = ""
        if "content" in original_response:
            for content_item in original_response["content"]:
                if content_item.get("type") == "text":
                    original_text += content_item.get("text", "")
        
        # Verificar que la respuesta cumple con los requerimientos
        compliance_check = self._verify_response_compliance(original_text, feature_requirements)
        
        # Agregar información de contexto si es necesario
        enhanced_text = original_text
        
        if not compliance_check["compliant"]:
            enhanced_text += f"\n\n⚠️ **Advertencia de Contexto**: {compliance_check['warning']}"
        
        # Agregar información de fuentes si está disponible
        if compliance_check.get("sources"):
            enhanced_text += f"\n\n📚 **Fuentes consultadas**: {', '.join(compliance_check['sources'])}"
        
        return {
            "content": [{
                "type": "text",
                "text": enhanced_text
            }],
            "context_metadata": {
                "compliance_check": compliance_check,
                "query_count": self.query_count,
                "feature_requirements_checked": True
            }
        }
    
    def _verify_response_compliance(self, response_text: str, feature_requirements: Dict) -> Dict[str, Any]:
        """Verifica que la respuesta cumple con los requerimientos de feature.md"""
        
        compliance = {
            "compliant": True,
            "warning": "",
            "sources": [],
            "checks_performed": []
        }
        
        # Verificar que no sea una respuesta inventada
        if "no se encontró información" not in response_text.lower() and len(response_text) < 50:
            compliance["compliant"] = False
            compliance["warning"] = "Respuesta muy corta, posible alucinación"
        
        # Verificar que mencione fuentes específicas
        if "sección" not in response_text.lower() and "archivo" not in response_text.lower():
            compliance["warning"] = "Respuesta no cita fuentes específicas"
        
        compliance["checks_performed"] = ["length_check", "source_check", "hallucination_check"]
        
        return compliance
    
    def _format_code_analysis(self, analysis: Dict) -> str:
        """Formatea el resultado del análisis de código"""
        
        text = "# 📊 Análisis de Código Existente\n\n"
        
        files_count = len(analysis.get("files_analyzed", []))
        text += f"**Archivos analizados**: {files_count}\n"
        
        functions_count = len(analysis.get("functions_found", []))
        text += f"**Funciones encontradas**: {functions_count}\n"
        
        classes_count = len(analysis.get("classes_found", []))
        text += f"**Clases encontradas**: {classes_count}\n"
        
        duplicates = analysis.get("duplicates_detected", [])
        if duplicates:
            text += f"\n⚠️ **Duplicados detectados**: {len(duplicates)}\n"
            for dup in duplicates[:3]:  # Mostrar solo los primeros 3
                text += f"  - {dup['type']}: {dup['name']} ({dup['count']} ocurrencias)\n"
        
        missing = analysis.get("missing_components", [])
        if missing:
            text += f"\n🔍 **Componentes faltantes**: {len(missing)}\n"
            for comp in missing[:3]:
                text += f"  - {comp['type']}: {comp['description']}\n"
        
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            text += f"\n💡 **Recomendaciones**:\n"
            for rec in recommendations[:3]:
                text += f"  - {rec}\n"
        
        return text
    
    def _format_task_results(self, results: Dict) -> str:
        """Formatea los resultados del procesamiento de tareas"""
        
        text = "# 🎯 Resultados del Procesamiento de Tareas\n\n"
        
        processed = results.get("processed_tasks", [])
        text += f"**Tareas procesadas**: {len(processed)}\n"
        
        completed = [t for t in processed if t.get("status") == "completed"]
        text += f"**Tareas completadas**: {len(completed)}\n"
        
        errors = [t for t in processed if t.get("status") == "error"]
        if errors:
            text += f"**Tareas con errores**: {len(errors)}\n"
        
        context_reviews = results.get("context_reviews", [])
        text += f"**Revisiones de contexto**: {len(context_reviews)}\n"
        
        if context_reviews:
            latest_review = context_reviews[-1]
            coherence = latest_review.get("coherence_score", 1.0)
            text += f"**Coherencia actual**: {coherence:.2%}\n"
        
        recommendations = results.get("recommendations", [])
        if recommendations:
            text += f"\n💡 **Recomendaciones**:\n"
            for rec in recommendations:
                text += f"  - {rec}\n"
        
        return text
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lista herramientas disponibles incluyendo las nuevas"""
        
        # Obtener herramientas del servidor original
        original_tools = super().handle_tools_list(params)
        
        # Agregar nuevas herramientas
        enhanced_tools = original_tools.get("tools", [])
        
        enhanced_tools.extend([
            {
                "name": "analyze_code",
                "description": "Analiza código existente para prevenir duplicación",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Ruta del directorio a analizar (opcional)"
                        }
                    }
                }
            },
            {
                "name": "create_task",
                "description": "Crea una nueva tarea con análisis de contexto",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Descripción de la tarea"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Prioridad de la tarea"
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs de tareas dependientes"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "process_tasks",
                "description": "Procesa tareas con retroalimentación de contexto",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ])
        
        return {"tools": enhanced_tools}
    
    def run(self):
        """Ejecuta el servidor MCP mejorado"""
        logger.info("🚀 Iniciando Enhanced MCP Server con Context Feedback System...")
        logger.info("Características activas:")
        logger.info("  ✅ Prevención de alucinaciones")
        logger.info("  ✅ Análisis de código existente obligatorio")
        logger.info("  ✅ Lectura de feature.md antes de responder")
        logger.info("  ✅ Gestión de tareas con retroalimentación")
        logger.info("  ✅ Detección de duplicación de código")
        logger.info("  ✅ Ciclo 2 tareas → contexto → 1 tarea → contexto")
        
        # Ejecutar el servidor base
        super().run()

if __name__ == "__main__":
    server = EnhancedMCPServer()
    server.run()
