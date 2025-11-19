"""
TOON Serializer - Token-Oriented Object Notation for MCP v6
Optimizes data serialization for LLM context with 60-70% token savings
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TOONSerializer:
    """
    Serializer for TOON (Token-Oriented Object Notation) format
    
    TOON is optimized for LLM consumption with:
    - 60-70% token reduction vs JSON
    - Tabular format for uniform arrays
    - Explicit metadata (array lengths, field declarations)
    - Indentation-based nesting
    """
    
    @staticmethod
    def encode_session_history(history: List[Dict], max_turns: int = 10) -> str:
        """
        Encode session history to TOON format
        
        Args:
            history: List of session turns
            max_turns: Maximum turns to include
            
        Returns:
            TOON formatted string
        """
        if not history:
            return "No session history available"
        
        # Take last N turns
        recent = history[-max_turns:] if len(history) > max_turns else history
        
        # Standard fields for session turns
        fields = ["turn", "role", "content", "entities"]
        
        # Build rows
        rows = []
        for item in recent:
            turn = item.get("turn_id", item.get("turn", "?"))
            role = item.get("role", "unknown")
            content = str(item.get("content", "")).replace(",", ";")  # Escape commas
            entities = "|".join(item.get("entities_mentioned", item.get("entities", [])))
            
            rows.append(f"{turn},{role},{content},{entities}")
        
        # Build TOON
        header = f"session_history[{len(rows)}]{{{','.join(fields)}}}:"
        return header + "\n" + "\n".join(rows)
    
    @staticmethod
    def encode_code_entities(entities: List[Dict], max_results: int = 20) -> str:
        """
        Encode code entities (functions/classes) to TOON format
        
        Args:
            entities: List of entity dicts
            max_results: Maximum entities to include
            
        Returns:
            TOON formatted string
        """
        if not entities:
            return "No code entities found"
        
        # Take top N results
        top = entities[:max_results]
        
        # Fields for code entities
        fields = ["name", "type", "file", "line_start", "line_end", "signature"]
        
        # Build rows
        rows = []
        for entity in top:
            name = entity.get("name", entity.get("full_name", "unknown"))
            entity_type = entity.get("type", "unknown")
            file_path = entity.get("file_path", "")
            filename = file_path.split("/")[-1].split("\\")[-1] if file_path else "?"
            
            line_range = entity.get("line_range", [0, 0])
            line_start = line_range[0] if len(line_range) > 0 else 0
            line_end = line_range[1] if len(line_range) > 1 else line_start
            
            signature = entity.get("signature", "N/A").replace(",", ";")
            
            rows.append(f"{name},{entity_type},{filename},{line_start},{line_end},{signature}")
        
        # Build TOON
        header = f"code_entities[{len(rows)}]{{{','.join(fields)}}}:"
        return header + "\n" + "\n".join(rows)
    
    @staticmethod
    def encode_dependencies(deps: Dict[str, List[str]], max_deps: int = 30) -> str:
        """
        Encode dependency graph to TOON format
        
        Args:
            deps: Dict mapping entity -> list of dependencies
            max_deps: Maximum dependencies to include
            
        Returns:
            TOON formatted string
        """
        if not deps:
            return "No dependencies tracked"
        
        # Flatten to rows
        rows = []
        for entity, calls in deps.items():
            for call in calls:
                rows.append(f"{entity},{call},calls")
                if len(rows) >= max_deps:
                    break
            if len(rows) >= max_deps:
                break
        
        if not rows:
            return "No dependencies tracked"
        
        # Build TOON
        header = f"dependencies[{len(rows)}]{{from,to,type}}:"
        return header + "\n" + "\n".join(rows)
    
    @staticmethod
    def encode_entity_mentions(mentions: List[Dict], max_mentions: int = 15) -> str:
        """
        Encode entity mentions to TOON format
        
        Args:
            mentions: List of entity mention dicts
            max_mentions: Maximum mentions to include
            
        Returns:
            TOON formatted string
        """
        if not mentions:
            return "No entity mentions"
        
        # Take most recent
        recent = mentions[-max_mentions:] if len(mentions) > max_mentions else mentions
        
        # Fields
        fields = ["entity", "type", "turn", "context"]
        
        # Build rows
        rows = []
        for mention in recent:
            entity = mention.get("entity_name", "?")
            entity_type = mention.get("entity_type", "?")
            turn = mention.get("turn_id", "?")
            context = mention.get("context", "")[:50].replace(",", ";")  # Truncate and escape
            
            rows.append(f"{entity},{entity_type},{turn},{context}")
        
        # Build TOON
        header = f"entity_mentions[{len(rows)}]{{{','.join(fields)}}}:"
        return header + "\n" + "\n".join(rows)
    
    @staticmethod
    def encode_nested_data(data: Dict[str, Any], indent: int = 0) -> str:
        """
        Encode nested dictionary to TOON format with indentation
        
        Args:
            data: Dictionary to encode
            indent: Current indentation level
            
        Returns:
            TOON formatted string
        """
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(TOONSerializer.encode_nested_data(value, indent + 1))
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # Array of objects - use tabular format
                    lines.append(f"{prefix}{key}:")
                    lines.append(TOONSerializer._encode_object_array(value, indent + 1))
                else:
                    # Simple array
                    list_str = "|".join(str(v) for v in value)
                    lines.append(f"{prefix}{key}: {list_str}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _encode_object_array(objects: List[Dict], indent: int = 0) -> str:
        """Helper to encode array of objects in tabular TOON format"""
        if not objects:
            return ""
        
        prefix = "  " * indent
        fields = list(objects[0].keys())
        field_str = ",".join(fields)
        
        rows = []
        for obj in objects:
            values = [str(obj.get(f, "")).replace(",", ";") for f in fields]
            rows.append(prefix + ",".join(values))
        
        header = f"{prefix}[{len(objects)}]{{{field_str}}}:"
        return header + "\n" + "\n".join(rows)
    
    @staticmethod
    def build_llm_context(
        query: str,
        session_history: Optional[List[Dict]] = None,
        code_entities: Optional[List[Dict]] = None,
        dependencies: Optional[Dict[str, List[str]]] = None,
        entity_mentions: Optional[List[Dict]] = None,
        vector_results: Optional[List[Dict]] = None
    ) -> str:
        """
        Build complete LLM context using TOON format
        
        This is the main method for v6 to construct prompts
        
        Args:
            query: User query
            session_history: Session turns
            code_entities: Code index results
            dependencies: Dependency graph
            entity_mentions: Entity mention history
            vector_results: Vector search results
            
        Returns:
            Complete context string optimized for LLM
        """
        sections = []
        
        # Query (always included)
        sections.append(f"<query>\n{query}\n</query>")
        
        # Session history
        if session_history:
            history_toon = TOONSerializer.encode_session_history(session_history)
            sections.append(f"<session_context>\n{history_toon}\n</session_context>")
        
        # Code entities
        if code_entities:
            entities_toon = TOONSerializer.encode_code_entities(code_entities)
            sections.append(f"<code_index>\n{entities_toon}\n</code_index>")
        
        # Dependencies
        if dependencies:
            deps_toon = TOONSerializer.encode_dependencies(dependencies)
            sections.append(f"<dependencies>\n{deps_toon}\n</dependencies>")
        
        # Entity mentions
        if entity_mentions:
            mentions_toon = TOONSerializer.encode_entity_mentions(entity_mentions)
            sections.append(f"<entity_mentions>\n{mentions_toon}\n</entity_mentions>")
        
        # Vector search results (keep as brief XML)
        if vector_results:
            vector_section = "<vector_results>"
            for i, result in enumerate(vector_results[:5], 1):
                score = result.get("score", 0)
                source = result.get("source", "unknown")
                content = result.get("content", "")[:100]
                vector_section += f"\n  [{i}] score={score:.2f} src={source}\n      {content}..."
            vector_section += "\n</vector_results>"
            sections.append(vector_section)
        
        return "\n\n".join(sections)
    
    @staticmethod
    def decode_toon(toon_str: str) -> List[Dict]:
        """
        Decode TOON format back to Python structures
        
        Args:
            toon_str: TOON formatted string
            
        Returns:
            List of dictionaries
        """
        lines = toon_str.strip().split("\n")
        
        if not lines:
            return []
        
        # Parse header
        header = lines[0]
        if "[" not in header or "{" not in header:
            return []
        
        # Extract field names
        try:
            fields_part = header.split("{")[1].split("}")[0]
            fields = [f.strip() for f in fields_part.split(",")]
        except IndexError:
            return []
        
        # Parse data rows
        results = []
        for line in lines[1:]:
            if not line.strip():
                continue
            
            values = line.split(",")
            if len(values) != len(fields):
                continue
            
            row_dict = {fields[i]: values[i].strip() for i in range(len(fields))}
            results.append(row_dict)
        
        return results
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 chars for English)
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    @staticmethod
    def compare_formats(data: List[Dict], name: str = "data") -> Dict[str, Any]:
        """
        Compare TOON vs JSON token usage
        
        Args:
            data: Data to compare
            name: Name for the comparison
            
        Returns:
            Comparison statistics
        """
        import json
        
        # JSON format
        json_str = json.dumps(data, indent=2)
        json_tokens = TOONSerializer.estimate_tokens(json_str)
        
        # TOON format (assuming tabular data)
        if data and isinstance(data[0], dict):
            fields = list(data[0].keys())
            field_str = ",".join(fields)
            rows = []
            for item in data:
                values = [str(item.get(f, "")).replace(",", ";") for f in fields]
                rows.append(",".join(values))
            
            toon_str = f"[{len(data)}]{{{field_str}}}:\n" + "\n".join(rows)
            toon_tokens = TOONSerializer.estimate_tokens(toon_str)
        else:
            toon_tokens = json_tokens
            toon_str = json_str
        
        savings = ((json_tokens - toon_tokens) / json_tokens * 100) if json_tokens > 0 else 0
        
        return {
            'name': name,
            'json_tokens': json_tokens,
            'toon_tokens': toon_tokens,
            'savings_percent': round(savings, 1),
            'savings_absolute': json_tokens - toon_tokens,
            'json_size': len(json_str),
            'toon_size': len(toon_str)
        }
