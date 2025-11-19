# Model Documentation

## Overview

This document contains the system model and architecture information for the MCP Hub v5 project.

## Architecture

### Core Principles

The MCP Hub v5 operates under the following principles:

1. Memory Only - No business logic
2. Single Source of Truth - Only indexes specified markdown files
3. Retrieval-First - Returns evidence from source documents
4. Full Provenance - Every response includes metadata

### System Components

#### Storage Layer

The storage layer uses MP4 containers for vector data:
- HNSW index for fast similarity search
- Virtual chunks to avoid text duplication
- Memory-mapped I/O for efficiency

#### Retrieval Layer

The retrieval layer implements semantic search:
- Sentence transformer embeddings
- Configurable confidence thresholds
- Query expansion capabilities
- Dynamic chunking with overlap

#### Anti-Hallucination Layer

The anti-hallucination layer enforces strict policies:
- Abstention when confidence is below threshold
- Mandatory source attribution
- Line number tracking
- Comprehensive audit logging

## Data Flow

1. User sends query via JSON-RPC
2. Query is converted to embedding vector
3. HNSW index returns similar chunks
4. Results are filtered by confidence threshold
5. Response is formatted with full provenance
6. Query is logged to audit trail

## Configuration

The system is configured via `config/v5_config.json`:

### Key Parameters

- `sources.allowed_files`: List of markdown files to index
- `embedding.model`: Sentence transformer model name
- `retrieval.min_score`: Minimum confidence threshold
- `chunking.max_tokens`: Maximum tokens per chunk

## Performance Characteristics

- Index build time: ~30 seconds for 50MB of text
- Query latency: 20-50ms average
- Storage efficiency: 96% reduction vs traditional methods
- Memory footprint: 100-200MB runtime

## Extension Points

The system can be extended in several ways:

### Custom Chunking

Implement new chunking strategies by extending the DynamicChunker class.

### Custom Embeddings

Replace the default sentence transformer with a custom model via configuration.

### Custom Retrieval

Add new retrieval methods by modifying the VectorEngine class.
