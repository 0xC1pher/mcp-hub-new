"""
Python AST Parser - Extracts code structure without executing code
Parses Python files to extract functions, classes, and their metadata
"""

import ast
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PythonASTParser:
    """
    Parses Python source code to extract structural information
    Uses Python's built-in ast module for safe parsing
    """
    
    def __init__(self):
        """Initialize the AST parser"""
        self.current_file = None
    
    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a Python file and extract structure
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary with parsed structure or None on error
        """
        try:
            self.current_file = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code, filename=str(file_path))
            
            result = {
                'file_path': str(file_path),
                'functions': [],
                'classes': [],
                'imports': [],
                'module_docstring': ast.get_docstring(tree)
            }
            
            # Extract top-level elements
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function(node)
                    if func_info:
                        result['functions'].append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class(node)
                    if class_info:
                        result['classes'].append(class_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._extract_import(node)
                    if import_info:
                        result['imports'].extend(import_info)
            
            logger.info(
                f"Parsed {file_path}: {len(result['functions'])} functions, "
                f"{len(result['classes'])} classes"
            )
            
            return result
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_function(self, node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Extract function metadata"""
        try:
            # Get function signature
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                arg_annotation = None
                if arg.annotation:
                    arg_annotation = ast.unparse(arg.annotation)
                args.append({'name': arg_name, 'annotation': arg_annotation})
            
            # Get return annotation
            return_annotation = None
            if node.returns:
                return_annotation = ast.unparse(node.returns)
            
            # Build signature
            arg_strings = []
            for arg in args:
                if arg['annotation']:
                    arg_strings.append(f"{arg['name']}: {arg['annotation']}")
                else:
                    arg_strings.append(arg['name'])
            
            signature = f"def {node.name}({', '.join(arg_strings)})"
            if return_annotation:
                signature += f" -> {return_annotation}"
            
            return {
                'name': node.name,
                'signature': signature,
                'line_start': node.lineno,
                'line_end': node.end_lineno or node.lineno,
                'docstring': ast.get_docstring(node),
                'args': args,
                'return_type': return_annotation,
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'decorators': [ast.unparse(d) for d in node.decorator_list]
            }
            
        except Exception as e:
            logger.warning(f"Error extracting function {node.name}: {e}")
            return None
    
    def _extract_class(self, node: ast.ClassDef) -> Optional[Dict[str, Any]]:
        """Extract class metadata"""
        try:
            # Get base classes
            bases = []
            for base in node.bases:
                bases.append(ast.unparse(base))
            
            # Get methods
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = self._extract_function(item)
                    if method_info:
                        methods.append(method_info)
            
            # Get class attributes
            attributes = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attr_name = item.target.id
                    attr_type = ast.unparse(item.annotation) if item.annotation else None
                    attributes.append({'name': attr_name, 'type': attr_type})
            
            return{
                'name': node.name,
                'line_start': node.lineno,
                'line_end': node.end_lineno or node.lineno,
                'docstring': ast.get_docstring(node),
                'bases': bases,
                'methods': methods,
                'attributes': attributes,
                'decorators': [ast.unparse(d) for d in node.decorator_list]
            }
            
        except Exception as e:
            logger.warning(f"Error extracting class {node.name}: {e}")
            return None
    
    def _extract_import(self, node: ast.Import | ast.ImportFrom) -> List[Dict[str, Any]]:
        """Extract import information"""
        imports = []
        
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'name': alias.asname or alias.name,
                        'is_from': False
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'is_from': True
                    })
            
        except Exception as e:
            logger.warning(f"Error extracting import: {e}")
        
        return imports
    
    def get_function_calls(self, file_path: str, function_name: str) -> List[str]:
        """
        Find all function calls made by a specific function
        
        Args:
            file_path: Path to Python file
            function_name: Name of function to analyze
            
        Returns:
            List of called function names
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Find the function
            target_function = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    target_function = node
                    break
            
            if not target_function:
                return []
            
            # Find calls within the function
            calls = []
            for node in ast.walk(target_function):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
            
            return list(set(calls))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error getting function calls: {e}")
            return []
