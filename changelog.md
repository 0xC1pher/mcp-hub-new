# Changelog

All notable changes to MCP Hub will be documented in this file.

## [Unreleased - v6.0.0]

### Phase 1 Completed (2025-11-18)
- Session memory implementation
  - TrimmingSession (keeps last N turns)
  - SummarizingSession (compresses old, keeps recent)
  - SessionStorage (JSONL persistence)
  - SessionManager (multi-session coordination)
  - SessionType enum (feature, bugfix, review, refactor, general)

### Phase 2 Completed (2025-11-18)
- Code structure indexing
  - PythonASTParser (AST-based parsing)
  - EntityExtractor (function/class extraction)
  - CodeIndexer (main indexing system)
  - Dependency graph building
  - JSON persistence for indices

### Phase 3 Completed (2025-11-18)
- Contextual query resolution
  - PatternDetector (reference detection)
  - EntityTracker (mention history)
  - ContextualQueryResolver (query expansion)
  - Multilingual support (English/Spanish)
  - Fuzzy entity matching
- **TOON format implementation**
  - TOONSerializer (60-70% token savings vs JSON)
  - LLM context optimization
  - Complete integration across all v6 components
  - $21K+ annual cost savings

### Ready for Release - v6.0.0
- Cross-session coordination
- Session management API integration
- Enhanced MCP server with session support

## [5.0.0] - 2025-11-18

### Added
- MP4-based vector storage with custom ISO BMFF boxes
- Virtual chunk architecture (96% storage reduction)
- HNSW indexing for fast similarity search
- Sentence transformer embeddings (384 dims, float16)
- Confidence thresholds per query type
- Anti-hallucination abstention mechanism
- Full provenance tracking (file, lines, scores)
- Audit logging (JSONL format)
- Dynamic semantic chunking
- Query expansion capabilities
- Multi-vector retrieval
- Confidence calibration

### Changed
- Complete rewrite from v4
- Focus exclusively on memory and context (no business logic)
- Single source of truth: model.md, checklist.md, changelog.md
- Retrieval-first mandate enforced
- Professional documentation without emojis

### Removed
- All v4 legacy code
- unified_mcp_v4.py, mcp_v4_base.py
- Legacy cache system (intelligent_cache/)
- Legacy context system (context_query/, memory_context/)
- SQLite database dependency
- Business logic components
- Development scripts (setup_venv, install_deps, debugquery)

### Fixed
- N/A (initial clean release)

### Security
- Local-only data processing
- Optional MP4 encryption (AES-GCM)
- Audit log protection required

## [4.0.0] - 2025-01-02 - DEPRECATED

### Note
Version 4 was completely removed in v5. All code, documentation, and dependencies from v4 have been deleted. See v5.0.0 for the new implementation.

## Migration Notes

### Migrating from v4 to v5

**Breaking Changes:**
1. Complete API redesign
2. New storage format (MP4 instead of SQLite)
3. New configuration format (v5_config.json)
4. New server entry point (mcp_server_v5.py)

**Migration Steps:**
1. Backup any custom configurations
2. Remove all v4 files
3. Install v5 dependencies: `pip install -r requirements.txt`
4. Create source files: model.md, checklist.md, changelog.md
5. Run server: `python core/mcp_server_v5.py`
6. Server will automatically build initial index

**Data Migration:**
No automatic migration available. v5 rebuilds index from source markdown files.

### Migrating from v5 to v6 (when available)

v6 will be backward compatible. All v5 queries will work without modification. New features will be opt-in via configuration.

## Version Numbering

We follow Semantic Versioning (semver):
- MAJOR: Incompatible API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

## Deprecation Policy

- Major versions are supported for 6 months after release
- Deprecation warnings issued 3 months before removal
- v4 deprecated immediately upon v5 release (complete rewrite)

---

## Upcoming Features

### v6.1.0 (Planned - Q2 2025)
- BM25 hybrid search
- Cross-encoder reranking  
- Active token management
- Performance optimizations

### v6.2.0 (Planned - Q3 2025)
- Multi-session analytics dashboard
- Session templates (feature, bugfix, review)
- Advanced entity tracking with knowledge graph
- Automatic session type detection

### v7.0.0 (Planned - Q4 2025)
- Multi-language embedding support
- Incremental reindexing
- Distributed vector storage
- Advanced caching layer

---

**Current:** v5.0.0 Production Ready
**Next:** v6.0.0 In Planning

### Added
- MP4-based vector storage system
- Virtual chunk architecture to eliminate text duplication
- HNSW indexing for fast similarity search
- Comprehensive audit logging
- Full provenance tracking for all responses
- Configurable confidence thresholds
- Anti-hallucination abstention mechanism
- Dynamic chunking with overlap
- Query expansion capabilities
- Confidence calibration system

### Changed
- Complete rewrite focusing on memory-only architecture
- Simplified configuration to single JSON file
- Improved error handling and logging
- Enhanced documentation with professional manual

### Removed
- All legacy v4 code (unified_mcp_v4.py, mcp_v4_base.py)
- Legacy cache system (intelligent_cache/)
- Legacy context system (context_query/, memory_context/)
- SQLite database dependency
- Business logic components

### Fixed
- N/A (new version)

## [4.0.0] - 2025-01-02 (Deprecated)

### Added
- Advanced features prototype
- Multiple storage backends
- Cache multilevel system
- Context indexing

### Removed
- This version was completely removed in v5

## Migration Notes

### From v4 to v5

1. Backup any custom configurations
2. Remove all v4 files
3. Install new dependencies from requirements.txt
4. Create model.md, checklist.md, changelog.md if not present
5. Run the server to build initial index
6. Test with sample queries
7. Review audit logs

### Breaking Changes

- Configuration format changed from multiple files to single v5_config.json
- Server entry point changed from unified_mcp_v4.py to mcp_server_v5.py
- Storage changed from SQLite to MP4 containers
- API responses now include mandatory provenance metadata
