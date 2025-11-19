"""
Unit Tests for TOONSerializer
Tests token-optimized serialization for MCP v6
"""

import unittest
from core.shared.toon_serializer import TOONSerializer


class TestTOONSessionHistory(unittest.TestCase):
    """Test session history encoding"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_history = [
            {
                "turn_id": 1,
                "role": "user",
                "content": "Show me payment functions",
                "entities": ["process_payment", "validate_card"]
            },
            {
                "turn_id": 2,
                "role": "assistant",
                "content": "Found 3 payment functions",
                "entities": []
            },
            {
                "turn_id": 3,
                "role": "user",
                "content": "Explain process_payment",
                "entities": ["process_payment"]
            }
        ]
    
    def test_encode_basic(self):
        """Test basic session encoding"""
        result = TOONSerializer.encode_session_history(self.sample_history)
        
        # Check header
        self.assertIn("session_history[3]", result)
        self.assertIn("{turn,role,content,entities}", result)
        
        # Check data rows
        self.assertIn("1,user,Show me payment functions", result)
        self.assertIn("2,assistant,Found 3 payment functions", result)
    
    def test_encode_empty(self):
        """Test encoding empty history"""
        result = TOONSerializer.encode_session_history([])
        self.assertEqual(result, "No session history available")
    
    def test_encode_max_turns(self):
        """Test max_turns parameter"""
        long_history = [{"turn_id": i, "role": "user", "content": f"Turn {i}", "entities": []} 
                       for i in range(20)]
        
        result = TOONSerializer.encode_session_history(long_history, max_turns=5)
        
        # Should only have last 5 turns
        self.assertIn("session_history[5]", result)
        self.assertIn("Turn 19", result)  # Last turn
        self.assertNotIn("Turn 10", result)  # Earlier turn
    
    def test_comma_escaping(self):
        """Test that commas in content are escaped"""
        history = [{
            "turn_id": 1,
            "role": "user",
            "content": "This has a comma, see?",
            "entities": []
        }]
        
        result = TOONSerializer.encode_session_history(history)
        
        # Commas should be escaped to semicolons
        self.assertIn("This has a comma; see?", result)
    
    def test_entity_joining(self):
        """Test entity array joining"""
        result = TOONSerializer.encode_session_history(self.sample_history)
        
        # Entities should be pipe-separated
        self.assertIn("process_payment|validate_card", result)


class TestTOONCodeEntities(unittest.TestCase):
    """Test code entity encoding"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_entities = [
            {
                "name": "process_payment",
                "type": "function",
                "file_path": "/path/to/payment/processor.py",
                "line_range": [45, 78],
                "signature": "def process_payment(amount, card)"
            },
            {
                "name": "PaymentProcessor",
                "type": "class",
                "file_path": "/path/to/payment/processor.py",
                "line_range": [10, 120],
                "signature": "class PaymentProcessor"
            }
        ]
    
    def test_encode_basic(self):
        """Test basic entity encoding"""
        result = TOONSerializer.encode_code_entities(self.sample_entities)
        
        # Check header
        self.assertIn("code_entities[2]", result)
        self.assertIn("{name,type,file,line_start,line_end,signature}", result)
        
        # Check data
        self.assertIn("process_payment,function,processor.py,45,78", result)
        self.assertIn("PaymentProcessor,class,processor.py,10,120", result)
    
    def test_encode_empty(self):
        """Test encoding empty entities"""
        result = TOONSerializer.encode_code_entities([])
        self.assertEqual(result, "No code entities found")
    
    def test_max_results(self):
        """Test max_results limit"""
        many_entities = [
            {"name": f"func_{i}", "type": "function", "file_path": "test.py",
             "line_range": [i, i+10], "signature": f"def func_{i}()"}
            for i in range(50)
        ]
        
        result = TOONSerializer.encode_code_entities(many_entities, max_results=10)
        
        # Should only have 10 entities
        self.assertIn("code_entities[10]", result)
        self.assertIn("func_0", result)
        self.assertNotIn("func_20", result)
    
    def test_filename_extraction(self):
        """Test filename extraction from full path"""
        result = TOONSerializer.encode_code_entities(self.sample_entities)
        
        # Should extract just "processor.py" from full path
        self.assertIn("processor.py", result)
        self.assertNotIn("/path/to/payment", result)
    
    def test_signature_comma_escaping(self):
        """Test comma escaping in signatures"""
        entities = [{
            "name": "test_func",
            "type": "function",
            "file_path": "test.py",
            "line_range": [1, 5],
            "signature": "def test_func(arg1, arg2, arg3)"
        }]
        
        result = TOONSerializer.encode_code_entities(entities)
        
        # Commas in signature should be escaped
        self.assertIn("def test_func(arg1; arg2; arg3)", result)


class TestTOONDependencies(unittest.TestCase):
    """Test dependency graph encoding"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_deps = {
            "process_payment": ["validate_card", "log_transaction", "send_receipt"],
            "validate_card": ["check_luhn"],
            "send_receipt": ["email_service"]
        }
    
    def test_encode_basic(self):
        """Test basic dependency encoding"""
        result = TOONSerializer.encode_dependencies(self.sample_deps)
        
        # Check header
        self.assertIn("dependencies", result)
        self.assertIn("{from,to,type}", result)
        
        # Check relationships
        self.assertIn("process_payment,validate_card,calls", result)
        self.assertIn("validate_card,check_luhn,calls", result)
    
    def test_encode_empty(self):
        """Test encoding empty dependencies"""
        result = TOONSerializer.encode_dependencies({})
        self.assertEqual(result, "No dependencies tracked")
    
    def test_max_deps_limit(self):
        """Test max_deps parameter"""
        large_deps = {f"func_{i}": [f"dep_{j}" for j in range(10)] 
                     for i in range(20)}
        
        result = TOONSerializer.encode_dependencies(large_deps, max_deps=15)
        
        # Should limit to 15 dependency relationships
        lines = result.split("\n")
        # Header + 15 data lines = 16 total
        self.assertLessEqual(len(lines), 16)


class TestTOONContextBuilder(unittest.TestCase):
    """Test complete LLM context building"""
    
    def test_build_complete_context(self):
        """Test building complete context with all components"""
        history = [
            {"turn_id": 1, "role": "user", "content": "Test query", "entities": []}
        ]
        
        entities = [
            {"name": "test_func", "type": "function", "file_path": "test.py",
             "line_range": [1, 10], "signature": "def test_func()"}
        ]
        
        deps = {"test_func": ["helper"]}
        
        mentions = [
            {"entity_name": "test_func", "entity_type": "function", 
             "turn_id": 1, "context": "mentioned in query"}
        ]
        
        context = TOONSerializer.build_llm_context(
            query="Show me test functions",
            session_history=history,
            code_entities=entities,
            dependencies=deps,
            entity_mentions=mentions
        )
        
        # Check all sections present
        self.assertIn("<query>", context)
        self.assertIn("Show me test functions", context)
        self.assertIn("<session_context>", context)
        self.assertIn("<code_index>", context)
        self.assertIn("<dependencies>", context)
        self.assertIn("<entity_mentions>", context)
    
    def test_build_minimal_context(self):
        """Test building context with only query"""
        context = TOONSerializer.build_llm_context(
            query="Simple query"
        )
        
        # Should have query only
        self.assertIn("<query>", context)
        self.assertIn("Simple query", context)
        
        # Should not have other sections
        self.assertNotIn("<session_context>", context)
        self.assertNotIn("<code_index>", context)


class TestTOONDecode(unittest.TestCase):
    """Test TOON decoding"""
    
    def test_decode_basic(self):
        """Test decoding TOON back to Python"""
        toon_str = """session_history[2]{turn,role,content}:
1,user,Test message
2,assistant,Response"""
        
        result = TOONSerializer.decode_toon(toon_str)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["turn"], "1")
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["content"], "Response")
    
    def test_decode_empty(self):
        """Test decoding empty string"""
        result = TOONSerializer.decode_toon("")
        self.assertEqual(result, [])


class TestTOONComparison(unittest.TestCase):
    """Test JSON vs TOON comparison"""
    
    def test_compare_formats(self):
        """Test format comparison utility"""
        data = [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"}
        ]
        
        comparison = TOONSerializer.compare_formats(data, "test_data")
        
        # Check result structure
        self.assertIn("name", comparison)
        self.assertIn("json_tokens", comparison)
        self.assertIn("toon_tokens", comparison)
        self.assertIn("savings_percent", comparison)
        
        # TOON should save tokens
        self.assertGreater(comparison["json_tokens"], comparison["toon_tokens"])
        self.assertGreater(comparison["savings_percent"], 0)
    
    def test_token_estimation(self):
        """Test token estimation"""
        text = "This is a test string with about twenty tokens"
        tokens = TOONSerializer.estimate_tokens(text)
        
        # Rough estimate: 1 token ≈ 4 chars
        expected = len(text) // 4
        self.assertEqual(tokens, expected)


if __name__ == "__main__":
    unittest.main()
