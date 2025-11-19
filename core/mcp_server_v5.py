"""
MCP Server v5 - Memory and Context Only
Pure retrieval system with MP4-based vector storage
No business logic - only context retrieval
"""

import json
import sys
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path setup
current_dir = Path(__file__).resolve().parent
mcp_hub_root = current_dir.parent
sys.path.insert(0, str(mcp_hub_root))
sys.path.insert(0, str(current_dir))

# Import storage components
from storage import MP4Storage, VirtualChunk, VectorEngine

# Import advanced features (preserved from v4)
try:
    from advanced_features.dynamic_chunking import DynamicChunker
    from advanced_features.query_expansion import QueryExpander
    from advanced_features.confidence_calibration import ConfidenceCalibrator
    ADVANCED_AVAILABLE = True
    logger.info("Advanced features loaded")
except ImportError as e:
    logger.warning(f"Advanced features not available: {e}")
    ADVANCED_AVAILABLE = False


class MCPServerV5:
    """
    MCP Server v5 - Memory and Context Only
    
    Principles:
    1. Single source of truth: model.md, checklist.md, changelog.md
    2. Retrieval-only: No reasoning, no invention
    3. Confidence threshold: Abstain if score < threshold
    4. Provenance mandatory: Every response includes source metadata
    5. Auditable: All queries logged
    """
    
    def __init__(self, config_path: str = None):
        """Initialize MCP Server v5"""
        
        logger.info("="*80)
        logger.info("MCP Server v5 - Memory and Context Only")
        logger.info("="*80)
        
        # Load configuration
        if config_path is None:
            config_path = mcp_hub_root / "config" / "v5_config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize components
        self.storage = MP4Storage(str(mcp_hub_root / self.config['storage']['mp4_path']))
        self.vector_engine = VectorEngine(self.config)
        
        # Advanced features
        if ADVANCED_AVAILABLE:
            self.chunker = DynamicChunker(self.config['chunking'])
            self.query_expander = QueryExpander()
            self.confidence_calibrator = ConfidenceCalibrator(
                self.config['anti_hallucination']['confidence_thresholds']
            )
        else:
            self.chunker = None
            self.query_expander = None
            self.confidence_calibrator = None
        
        # State
        self.query_count = 0
        self.start_time = time.time()
        self.audit_log = []
        
        # Load existing snapshot or build new one
        self._initialize_index()
        
        logger.info("="*80)
        logger.info("MCP Server v5 ready")
        logger.info("="*80)
    
    def _initialize_index(self):
        """Initialize or load vector index"""
        if self.storage.load_snapshot():
            logger.info("Loaded existing snapshot")
            # Load HNSW index from MP4
            hnsw_offset, hnsw_size = self.storage.get_hnsw_blob_offset()
            if hnsw_size > 0:
                with open(self.storage.mp4_path, 'rb') as f:
                    f.seek(hnsw_offset)
                    hnsw_data = f.read(hnsw_size)
                    self.vector_engine.load_index_from_bytes(hnsw_data, len(self.storage.chunks))
                logger.info("HNSW index loaded from MP4")
        else:
            logger.info("No existing snapshot - building new index")
            self._build_initial_index()
    
    def _build_initial_index(self):
        """Build initial index from source MD files"""
        logger.info("Building index from source files")
        
        source_files = self.config['sources']['allowed_files']
        base_path = Path(self.config['sources']['base_path'])
        
        all_chunks = []
        all_texts = []
        
        for filename in source_files:
            file_path = base_path / filename
            if not file_path.exists():
                logger.warning(f"Source file not found: {file_path}")
                continue
            
            logger.info(f"Processing {filename}")
            chunks = self._chunk_file(str(file_path))
            all_chunks.extend(chunks)
            all_texts.extend([chunk.get_text() for chunk in chunks])
        
        if not all_chunks:
            logger.warning("No chunks created - no source files found")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        vectors = self.vector_engine.embed_batch(all_texts)
        
        # Create HNSW index
        self.vector_engine.create_index(len(all_chunks))
        chunk_ids = [chunk.chunk_id for chunk in all_chunks]
        self.vector_engine.add_vectors(vectors, chunk_ids)
        
        # Assign vector offsets
        current_offset = 0
        for i, chunk in enumerate(all_chunks):
            chunk.vector_offset = current_offset
            chunk.vector_size = self.config['embedding']['dimension']
            chunk.text_hash = chunk.compute_hash()
            current_offset += chunk.vector_size * 2  # float16 = 2 bytes per dimension
        
        # Serialize vectors
        vectors_blob = vectors.tobytes()
        
        # Serialize HNSW index
        hnsw_blob = self.vector_engine.serialize_index()
        
        # Create MP4 snapshot
        metadata = {
            'version': self.config['version'],
            'embedding_model': self.config['embedding']['model'],
            'vector_dimension': self.config['embedding']['dimension'],
            'created_at': datetime.now().isoformat()
        }
        
        snapshot_hash = self.storage.create_snapshot(
            all_chunks, vectors_blob, hnsw_blob, metadata
        )
        
        logger.info(f"Index built and saved: {snapshot_hash}")
    
    def _chunk_file(self, file_path: str) -> List[VirtualChunk]:
        """Chunk a single MD file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        chunks = []
        
        if self.chunker and ADVANCED_AVAILABLE:
            # Use advanced chunking
            chunk_ranges = self.chunker.chunk_text(content)
            for i, (start, end, section) in enumerate(chunk_ranges):
                chunk = VirtualChunk(
                    chunk_id=f"{Path(file_path).stem}_{i}",
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    vector_offset=0,  # Will be set later
                    vector_size=self.config['embedding']['dimension'],
                    section=section,
                    summary=""
                )
                chunks.append(chunk)
        else:
            # Simple chunking by token count
            max_tokens = self.config['chunking']['max_tokens']
            overlap = self.config['chunking']['overlap_percent'] / 100
            
            current_start = 0
            chunk_idx = 0
            
            while current_start < len(lines):
                # Estimate tokens (rough: ~4 chars per token)
                current_tokens = 0
                current_end = current_start
                
                while current_end < len(lines) and current_tokens < max_tokens:
                    current_tokens += len(lines[current_end]) / 4
                    current_end += 1
                
                if current_end > current_start:
                    chunk = VirtualChunk(
                        chunk_id=f"{Path(file_path).stem}_{chunk_idx}",
                        file_path=file_path,
                        start_line=current_start,
                        end_line=min(current_end - 1, len(lines) - 1),
                        vector_offset=0,
                        vector_size=self.config['embedding']['dimension'],
                        section="",
                        summary=""
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Move forward with overlap
                current_start = current_end - int((current_end - current_start) * overlap)
                if current_start >= len(lines):
                    break
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    
    # MCP Protocol Handlers
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol request"""
        try:
            method = request.get('method')
            params = request.get('params', {})
            request_id = request.get('id')
            
            if method == 'initialize':
                result = self._handle_initialize(params)
            elif method == 'tools/list':
                result = self._handle_tools_list()
            elif method == 'tools/call':
                result = self._handle_tools_call(params)
            else:
                result = {'error': f'Unsupported method: {method}'}
            
            return {'jsonrpc': '2.0', 'id': request_id, 'result': result}
            
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'error': {'code': -32603, 'message': str(e)}
            }
    
    def _handle_initialize(self, params: Dict) -> Dict:
        """Handle initialize request"""
        return {
            'protocolVersion': self.config['protocol_version'],
            'capabilities': {'tools': {'listChanged': True}},
            'serverInfo': {
                'name': self.config['server_info']['name'],
                'version': self.config['version'],
                'description': self.config['server_info']['description']
            }
        }
    
    def _handle_tools_list(self) -> Dict:
        """List available tools"""
        return {
            'tools': [
                {
                    'name': 'get_context',
                    'description': 'Retrieve context from memory with provenance',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Query text'},
                            'top_k': {'type': 'integer', 'default': 5},
                            'min_score': {'type': 'number', 'default': 0.75}
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'validate_response',
                    'description': 'Validate response against evidence',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'candidate_text': {'type': 'string'},
                            'evidence_ids': {'type': 'array', 'items': {'type': 'string'}}
                        },
                        'required': ['candidate_text', 'evidence_ids']
                    }
                },
                {
                    'name': 'index_status',
                    'description': 'Get index status and statistics',
                    'inputSchema': {'type': 'object'}
                }
            ]
        }
    
    def _handle_tools_call(self, params: Dict) -> Dict:
        """Execute tool"""
        tool = params.get('name')
        args = params.get('arguments', {})
        
        try:
            if tool == 'get_context':
                return self._get_context(args)
            elif tool == 'validate_response':
                return self._validate_response(args)
            elif tool == 'index_status':
                return self._index_status()
            else:
                return {
                    'content': [{'type': 'text', 'text': f'Unknown tool: {tool}'}],
                    'isError': True
                }
        except Exception as e:
            logger.error(f"Error in tool {tool}: {e}", exc_info=True)
            return {
                'content': [{'type': 'text', 'text': f'Error: {str(e)}'}],
                'isError': True
            }
    
    def _get_context(self, args: Dict) -> Dict:
        """Retrieve context with provenance"""
        query = args.get('query', '')
        top_k = args.get('top_k', self.config['retrieval']['top_k'])
        min_score = args.get('min_score', self.config['retrieval']['min_score'])
        
        start_time = time.time()
        self.query_count += 1
        
        logger.info(f"Query #{self.query_count}: {query[:100]}")
        
        # Generate query embedding
        query_vector = self.vector_engine.embed_text(query)
        
        # Search
        chunk_ids, scores = self.vector_engine.search(query_vector, top_k)
        
        # Filter by min_score
        results = []
        for chunk_id, score in zip(chunk_ids, scores):
            if score < min_score:
                continue
            
            # Find chunk
            chunk = next((c for c in self.storage.chunks if c.chunk_id == chunk_id), None)
            if chunk:
                results.append({
                    'chunk_id': chunk_id,
                    'file': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'text': chunk.get_text(),
                    'score': score,
                    'section': chunk.section
                })
        
        elapsed = time.time() - start_time
        
        # Check if we should abstain
        if not results or (results and results[0]['score'] < min_score):
            response_text = "No sufficient information found in memory for this query."
            abstained = True
        else:
            response_text = self._format_context_response(query, results)
            abstained = False
        
        # Audit logging
        if self.config['logging']['audit_queries']:
            self._log_query(query, results, abstained, elapsed)
        
        return {
            'content': [{'type': 'text', 'text': response_text}],
            'metadata': {
                'query': query,
                'results_count': len(results),
                'abstained': abstained,
                'time_ms': round(elapsed * 1000, 2),
                'snapshot_hash': self.storage.metadata.get('snapshot_hash', ''),
                'provenance': [
                    {
                        'chunk_id': r['chunk_id'],
                        'file': r['file'],
                        'lines': f"{r['start_line']}-{r['end_line']}",
                        'score': round(r['score'], 3)
                    }
                    for r in results
                ]
            }
        }
    
    def _format_context_response(self, query: str, results: List[Dict]) -> str:
        """Format context retrieval response"""
        text = f"Context retrieval for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            text += f"[{i}] {Path(result['file']).name} (lines {result['start_line']}-{result['end_line']}, score: {result['score']:.3f})\n"
            text += f"{result['text']}\n\n"
        
        return text
    
    def _validate_response(self, args: Dict) -> Dict:
        """Validate response against evidence -- placeholder for now"""
        candidate = args.get('candidate_text', '')
        evidence_ids = args.get('evidence_ids', [])
        
        # Simple validation: check if evidence exists
        found_evidence = []
        for eid in evidence_ids:
            chunk = next((c for c in self.storage.chunks if c.chunk_id == eid), None)
            if chunk:
                found_evidence.append(chunk)
        
        return {
            'content': [{
                'type': 'text',
                'text': f"Validated {len(found_evidence)}/{len(evidence_ids)} evidence chunks"
            }]
        }
    
    def _index_status(self) -> Dict:
        """Get index status"""
        stats = self.vector_engine.get_stats()
        
        text = f"MCP Server v5 - Index Status\n\n"
        text += f"Version: {self.config['version']}\n"
        text += f"Snapshot: {self.storage.metadata.get('snapshot_hash', 'N/A')[:16]}...\n"
        text += f"Chunks: {len(self.storage.chunks)}\n"
        text += f"Vectors: {stats.get('num_vectors', 0)}\n"
        text += f"Model: {stats.get('model', 'N/A')}\n"
        text += f"Queries: {self.query_count}\n"
        text += f"Uptime: {(time.time() - self.start_time)/60:.1f} minutes\n"
        
        return {
            'content': [{'type': 'text', 'text': text}]
        }
    
    def _log_query(self, query: str, results: List[Dict], abstained: bool, elapsed: float):
        """Log query to audit log"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results_count': len(results),
            'abstained': abstained,
            'elapsed_ms': round(elapsed * 1000, 2),
            'top_score': results[0]['score'] if results else 0.0
        }
        
        self.audit_log.append(entry)
        
        # Write to file if configured
        audit_path = self.config['logging'].get('audit_path')
        if audit_path:
            audit_file = Path(audit_path)
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')


def main():
    """Entry point for MCP stdio protocol"""
    try:
        server = MCPServerV5()
        
        logger.info("Server ready - waiting for requests on stdin")
        
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            
            try:
                request = json.loads(line.strip())
                response = server.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
