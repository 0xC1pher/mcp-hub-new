from optimized_mcp_server import QueryOptimizer

opt = QueryOptimizer()
result = opt.normalize_query("¿Cómo CREAR un Paciente?")
print(f"Resultado: '{result}'")
print(f"Palabras: {result.split()}")
