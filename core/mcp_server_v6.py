"""
MCP Server v6.0.0 - The Complete System
Integrates Session Memory, Code Indexing, Context Resolution, and TOON Optimization.
"""

import sys
import json
import logging
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging to stderr (CRITICAL for MCP)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("MCP-v6")

# Force binary mode for stdout/stdin on Windows
if sys.platform == "win32":
    import msvcrt
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)

# Path setup
current_dir = Path(__file__).resolve().parent
mcp_hub_root = current_dir.parent
sys.path.insert(0, str(mcp_hub_root))
sys.path.insert(0, str(current_dir))

# --- v5 Imports (Base Storage) ---
from storage import MP4Storage, VectorEngine

# --- v6 Imports (New Capabilities) ---
from memory.session_manager import SessionManager
from indexing.code_indexer import CodeIndexer
from context.query_resolver import ContextualQueryResolver
from shared.toon_serializer import TOONSerializer

class MCPServerV6:
    def __init__(self):
        logger.info("Initializing MCP Server v6.0.0...")
        
        # 1. Load Config
        self.config_path = mcp_hub_root / "config" / "v6_config.json"
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # 2. Initialize v5 Storage (MP4 + Vectors)
        self.storage = MP4Storage(str(mcp_hub_root / self.config['storage']['mp4_path']))
        self.vector_engine = VectorEngine(self.config)
        self._init_vector_index()
        
        # 3. Initialize v6 Components
        # Session Manager (Memory)
        self.session_manager = SessionManager(
            storage_dir=mcp_hub_root / "data" / "sessions"
        )
        
        # Code Indexer (Structure)
        self.code_indexer = CodeIndexer(
            storage_path=mcp_hub_root / "data" / "code_index" / "index.json"
        )
        
        # Context Resolver (Understanding)
        self.resolver = ContextualQueryResolver(
            self.session_manager,
            self.code_indexer
        )
        
        logger.info("✅ MCP v6 System Ready")

    def _init_vector_index(self):
        """Load existing vector index from MP4"""
        if self.storage.load_snapshot():
            hnsw_offset, hnsw_size = self.storage.get_hnsw_blob_offset()
            if hnsw_size > 0:
                with open(self.storage.mp4_path, 'rb') as f:
                    f.seek(hnsw_offset)
                    hnsw_data = f.read(hnsw_size)
                    self.vector_engine.load_index_from_bytes(hnsw_data, len(self.storage.chunks))
            logger.info("Vector index loaded")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request"""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "mcp-hub-v6",
                        "version": "6.0.0"
                    }
                }
            }
            
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "get_context",
                            "description": "Retrieve context with v6 intelligence (Session + Code + Vectors)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "session_id": {"type": "string"},
                                    "use_toon": {"type": "boolean", "default": True}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "manage_session",
                            "description": "Manage development sessions",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["create", "get_history", "add_turn"]},
                                    "session_id": {"type": "string"},
                                    "content": {"type": "string"},
                                    "role": {"type": "string"}
                                },
                                "required": ["action"]
                            }
                        },
                        {
                            "name": "index_codebase",
                            "description": "Index project code structure",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"}
                                },
                                "required": ["path"]
                            }
                        }
                    ]
                }
            }
            
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            
            try:
                result = await self._handle_tool_call(name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result
                }
            except Exception as e:
                logger.error(f"Tool error: {e}", exc_info=True)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

    async def _handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool calls to v6 components"""
        
        # --- Tool: get_context (The Brain) ---
        if name == "get_context":
            query = args["query"]
            session_id = args.get("session_id")
            use_toon = args.get("use_toon", True)
            
            # 1. Resolve Context (What is "that function"?)
            resolved_query = await self.resolver.resolve_query(query, session_id)
            
            # 2. Get Code Entities
            code_results = await self.code_indexer.search(resolved_query.expanded_query)
            
            # 3. Get Vector Results (Docs/Knowledge)
            vector_results = self.vector_engine.search(resolved_query.expanded_query)
            
            # 4. Get Session History
            history = []
            if session_id:
                session = await self.session_manager.get_session(session_id)
                if session:
                    history = await session.get_items()
            
            # 5. Format Output (TOON vs JSON)
            if use_toon:
                # TOON Optimization (60-70% savings)
                context = TOONSerializer.build_llm_context(
                    query=resolved_query.expanded_query,
                    session_history=history,
                    code_entities=code_results,
                    dependencies=[], # Can fetch deps if needed
                    entity_mentions=[]
                )
                return {"content": [{"type": "text", "text": context}], "format": "toon"}
            else:
                # Legacy JSON
                return {
                    "content": [{
                        "type": "text", 
                        "text": json.dumps({
                            "query": resolved_query.expanded_query,
                            "code": code_results,
                            "vectors": vector_results,
                            "history": history
                        }, indent=2)
                    }]
                }

        # --- Tool: manage_session ---
        elif name == "manage_session":
            action = args["action"]
            
            if action == "create":
                session = await self.session_manager.create_session()
                return {"content": [{"type": "text", "text": session.session_id}]}
                
            elif action == "add_turn":
                sid = args["session_id"]
                await self.session_manager.add_turn(
                    sid, args["role"], args["content"]
                )
                return {"content": [{"type": "text", "text": "Turn added"}]}
                
            elif action == "get_history":
                sid = args["session_id"]
                session = await self.session_manager.get_session(sid)
                history = await session.get_items()
                return {"content": [{"type": "text", "text": TOONSerializer.encode_session_history(history)}]}

        # --- Tool: index_codebase ---
        elif name == "index_codebase":
            path = args["path"]
            stats = await self.code_indexer.index_directory(path)
            return {"content": [{"type": "text", "text": json.dumps(stats)}]}

        raise ValueError(f"Unknown tool: {name}")

async def main():
    server = MCPServerV6()
    
    # Main Loop
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            request = json.loads(line)
            response = await server.process_request(request)
            
            # Send response to stdout (JSON only)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Loop error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
