# MCP Hub v5 - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Architecture](#architecture)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [Development](#development)

---

## Introduction

MCP Hub v5 is a pure memory and context retrieval server implementing the Model Context Protocol (MCP). It operates under strict principles:

- **Memory Only**: No business logic, only context retrieval
- **Single Source of Truth**: Only indexes `model.md`, `checklist.md`, and `changelog.md`
- **Retrieval-First**: Returns evidence from source documents, does not generate content
- **Provenance Mandatory**: Every response includes source metadata
- **Anti-Hallucination**: Built-in confidence thresholding and abstention mechanisms

### Key Features

- MP4-based vector storage for portability and efficiency
- HNSW (Hierarchical Navigable Small World) indexing for fast similarity search
- Virtual chunking system to avoid text duplication
- Semantic chunking with overlap for context preservation
- Query expansion and confidence calibration
- Full audit logging of all queries

---

## Installation

### Prerequisites

- Python 3.8 or higher (Python 3.11 recommended)
- pip package manager

### Steps

1. Navigate to the project directory:

```bash
cd path/to/mcp-hub
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `numpy`: Numerical operations
- `hnswlib`: Vector similarity search
- `pymp4`: MP4 file manipulation
- `tiktoken`: Token counting
- `sentence-transformers`: Text embeddings
- Additional dependencies for caching and compression

3. Verify installation:

```bash
python core/mcp_server_v5.py --help
```

---

## Configuration

### Configuration File

The server uses `config/v5_config.json` for all settings.

#### Key Configuration Sections

**Sources**
```json
"sources": {
  "allowed_files": [
    "model.md",
    "checklist.md",
    "changelog.md"
  ],
  "base_path": "."
}
```

**Embedding Settings**
```json
"embedding": {
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimension": 384,
  "normalize": true,
  "dtype": "float16"
}
```

**Retrieval Parameters**
```json
"retrieval": {
  "top_k": 8,
  "rerank_top": 3,
  "min_score": 0.75
}
```

**Anti-Hallucination Thresholds**
```json
"anti_hallucination": {
  "confidence_thresholds": {
    "factual": 0.78,
    "procedural": 0.72,
    "conceptual": 0.65,
    "temporal": 0.85
  }
}
```

---

## Architecture

### Components

#### 1. MP4 Storage (`core/storage/mp4_storage.py`)

Handles vector storage using MP4 container format.

**Structure:**
- `ftyp`: File type box (identifies as MCP v5 format)
- `moov/udta/mcpi`: Index metadata (JSON)
- `mdat`: Vector embeddings and HNSW index

**Key Classes:**
- `VirtualChunk`: References text in source files without duplication
- `MP4Storage`: Manages MP4 read/write operations

#### 2. Vector Engine (`core/storage/vector_engine.py`)

Manages embeddings and HNSW-based search.

**Features:**
- Text-to-vector embedding using sentence-transformers
- HNSW indexing for fast approximate nearest neighbor search
- Batch processing for efficiency
- Serialization/deserialization for MP4 storage

#### 3. MCP Server (`core/mcp_server_v5.py`)

Main server implementing MCP protocol.

**Responsibilities:**
- Handle JSON-RPC requests via stdin/stdout
- Coordinate chunking, embedding, and retrieval
- Enforce anti-hallucination policies
- Audit logging

### Data Flow

```
1. Initialization:
   MD Files -> Chunking -> Embeddings -> HNSW Index -> MP4 Storage

2. Query Processing:
   Query -> Embedding -> HNSW Search -> Score Filtering -> Provenance -> Response
```

---

## Usage

### Starting the Server

**Standard Mode:**
```bash
python core/mcp_server_v5.py
```

The server will:
1. Load configuration from `config/v5_config.json`
2. Load existing MP4 snapshot or build new index
3. Listen for JSON-RPC requests on stdin

### Initial Indexing

On first run, the server will automatically:
1. Read `model.md`, `checklist.md`, `changelog.md`
2. Chunk content semantically
3. Generate embeddings
4. Build HNSW index
5. Save to `data/context_vectors.mp4`

This process may take 1-5 minutes depending on file size.

### Querying the Server

The server uses JSON-RPC 2.0 over stdio.

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_context",
    "arguments": {
      "query": "How is the payment flow handled?",
      "top_k": 5,
      "min_score": 0.75
    }
  }
}
```

**Example Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Context retrieval for: How is the payment flow handled?\n\n[1] model.md (lines 45-67, score: 0.892)\nThe payment flow is handled through...\n"
      }
    ],
    "metadata": {
      "query": "How is the payment flow handled?",
      "results_count": 3,
      "abstained": false,
      "time_ms": 23.45,
      "provenance": [
        {
          "chunk_id": "model_3",
          "file": "model.md",
          "lines": "45-67",
          "score": 0.892
        }
      ]
    }
  }
}
```

---

## API Reference

### Tools

#### `get_context`

Retrieve context from memory with full provenance.

**Input Schema:**
```json
{
  "query": "string (required)",
  "top_k": "integer (default: 5)",
  "min_score": "number (default: 0.75)"
}
```

**Returns:**
- `content`: Array of text results
- `metadata`: Query metadata including provenance

**Behavior:**
- If no results meet `min_score` threshold, returns abstention message
- All results include source file, line numbers, and similarity score
- Results are ranked by similarity score

#### `validate_response`

Validate a candidate response against evidence chunks.

**Input Schema:**
```json
{
  "candidate_text": "string (required)",
  "evidence_ids": "array of strings (required)"
}
```

**Returns:**
Validation report indicating which evidence chunks were found.

#### `index_status`

Get current index status and statistics.

**Input Schema:**
```json
{}
```

**Returns:**
- Snapshot hash
- Number of chunks
- Number of vectors
- Embedding model name
- Query count
- Server uptime

---

## Advanced Features

### Dynamic Chunking

The server uses semantic-aware chunking that:
- Preserves paragraph and section boundaries
- Maintains configurable overlap (default 25%)
- Adapts chunk size based on content structure

Configure in `config/v5_config.json`:
```json
"chunking": {
  "min_tokens": 150,
  "max_tokens": 450,
  "overlap_percent": 25,
  "preserve_structure": true
}
```

### Query Expansion

Automatically reformulates queries to improve recall by:
- Generating lexical variations
- Adding technical synonyms
- Expanding abbreviations

This feature is enabled automatically when advanced_features are available.

### Confidence Calibration

Different confidence thresholds for different query types:

- **Factual** (0.78): Concrete facts and data
- **Procedural** (0.72): How-to and process descriptions
- **Conceptual** (0.65): Abstract concepts
- **Temporal** (0.85): Time-sensitive information from changelog

### Audit Logging

All queries are logged to `logs/audit.jsonl` (if configured).

**Log Entry Format:**
```json
{
  "timestamp": "2025-11-18T21:45:00",
  "query": "payment flow",
  "results_count": 3,
  "abstained": false,
  "elapsed_ms": 23.45,
  "top_score": 0.892
}
```

---

## Troubleshooting

### Common Issues

#### Issue: "MP4 file not found"

**Solution:** The server will create a new index on first run. Ensure source files exist in the base path.

#### Issue: "Index loading failed"

**Solution:** Delete `data/context_vectors.mp4` to force reindexing.

#### Issue: "No results returned"

**Possible causes:**
- Query too specific or using different terminology
- `min_score` threshold too high
- Source files not indexed

**Solution:** 
- Lower `min_score` in request
- Check that source files contain relevant content
- Review chunking configuration

#### Issue: "Embedding model download fails"

**Solution:**
- Ensure internet connection
- Check firewall settings
- Manually download model: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"`

### Logging

Increase log verbosity by editing `core/mcp_server_v5.py`:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

Logs will show:
- Index building progress
- Query processing details
- Vector search operations
- Cache hits/misses

---

## Development

### Project Structure

```
mcp-hub/
├── core/
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── mp4_storage.py
│   │   └── vector_engine.py
│   ├── advanced_features/
│   │   ├── dynamic_chunking.py
│   │   ├── query_expansion.py
│   │   └── confidence_calibration.py
│   └── mcp_server_v5.py
├── config/
│   └── v5_config.json
├── data/
│   └── context_vectors.mp4
├── logs/
│   └── audit.jsonl
├── requirements.txt
├── MANUAL.md
└── README.md
```

### Adding New Source Files

Edit `config/v5_config.json`:

```json
"sources": {
  "allowed_files": [
    "model.md",
    "checklist.md",
    "changelog.md",
    "new_file.md"
  ]
}
```

Then restart the server or trigger reindexing.

### Customizing Embedding Model

To use a different model:

1. Edit `config/v5_config.json`:
```json
"embedding": {
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "dimension": 384
}
```

2. Delete existing index
3. Restart server

### Extending the API

To add new tools:

1. Add tool definition in `_handle_tools_list()`
2. Implement handler in `_handle_tools_call()`
3. Add business logic in new method

Example:
```python
def _my_new_tool(self, args: Dict) -> Dict:
    # Implementation
    return {
        'content': [{'type': 'text', 'text': 'Result'}]
    }
```

---

## Version History

**v5.0.0** (Current)
- MP4-based vector storage
- Complete rewrite focusing on memory-only architecture
- Enhanced anti-hallucination measures
- Full provenance tracking
- Removed all business logic

---

## License

This project is part of the MCP Hub ecosystem. Refer to the main repository for licensing information.

---

## Support

For issues, questions, or contributions:
1. Check this manual first
2. Review logs in `logs/` directory
3. Check configuration in `config/v5_config.json`
4. Consult the implementation plan and requirements documents
