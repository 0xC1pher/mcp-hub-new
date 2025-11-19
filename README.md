# MCP Hub v6.0.0

Production-ready Model Context Protocol (MCP) server implementing advanced memory management, code indexing, and token optimization for AI coding assistants.

![Version](https://img.shields.io/badge/version-6.0.0-blue)
![Status](https://img.shields.io/badge/status-production-green)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TOON](https://img.shields.io/badge/TOON-60%25%20savings-orange)

## Overview

MCP Hub v6 transforms how AI agents interact with your codebase by providing persistent memory, structural understanding, and efficient context retrieval. It bridges the gap between stateless LLMs and complex project histories.

**Key Innovation:** TOON (Token-Oriented Object Notation) format reduces token usage by 60-70% compared to JSON, significantly lowering API costs while improving model comprehension.

## Architecture & Technologies

### v6 Intelligence Layer (New)
*   **TOON Optimization:** Custom serialization format optimized for LLMs.
*   **Session Memory Manager:** Maintains persistent conversation history across IDE restarts.
*   **Code Structure Indexing:** Parses and indexes codebase structure (AST-based) to resolve ambiguous references.
*   **Contextual Query Resolution:** Disambiguates user queries by combining session history with code index data.

### v5 Storage Layer (Retained Core)
*   **MP4 Vector Storage:** High-performance, zero-dependency vector storage using the MP4 container format.
*   **HNSW Vector Search:** Fast approximate nearest neighbor search (20-50ms latency).
*   **Anti-Hallucination:** Strict confidence thresholds and mandatory provenance tracking.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure your MCP client (Cursor, Windsurf, etc.) to use the server.

## Quick Start

```bash
# Start server (automatically builds index)
python core/mcp_server_v6.py
```

The server will automatically:
1.  Index source files (`model.md`, `checklist.md`, etc.)
2.  Build vector index with HNSW
3.  Initialize session storage
4.  Listen for JSON-RPC requests

## Project Structure

```
mcp-hub/
├── core/
│   ├── mcp_server_v6.py          # Main server (Unified)
│   ├── storage/                  # v5 Storage Engine (MP4 + HNSW)
│   ├── memory/                   # v6 Session Management
│   ├── indexing/                 # v6 Code Structure Indexing
│   ├── context/                  # v6 Context Resolution
│   └── shared/                   # TOON Serializer
├── config/
│   └── v5_config.json            # Configuration
├── docs/                         # Documentation
└── data/                         # Runtime data (Vectors, Sessions)
```

## Tools Available

*   `get_context(query, session_id)`: Retrieves relevant context using semantic search and session history, formatted in TOON.
*   `manage_session(action, ...)`: Creates or updates development sessions.
*   `index_codebase(path)`: Scans and indexes the project structure.

## License

Proprietary. Internal use only.