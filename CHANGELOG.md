# Changelog

All notable changes to the Cognitive Meeting Intelligence project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and documentation
- Core data models (Memory, Meeting, MemoryConnection)
- SQLite database schema with 5 tables
- ONNX embedding infrastructure scaffolding
- 400D vector composition system (384D semantic + 16D cognitive)
- Dimension extractors (temporal, emotional, social, causal, evolutionary)
- Qdrant vector store with 3-tier memory system
- Memory extraction pipeline
- FastAPI skeleton with health, ingest, and search endpoints
- Comprehensive test structure
- Development tooling (Makefile, Docker Compose)

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Outdated documentation files that conflicted with IMPLEMENTATION_GUIDE.md

### Fixed
- Created missing src/models/entities.py file

### Security
- Nothing yet

## [0.1.0] - TBD (First Release)

Initial release will include:
- Complete Week 1 MVP implementation
- Basic memory extraction and storage
- Simple vector search
- API documentation

---

## Task References

Tasks completed in this version:
- PROJECT-001: Git repository initialization
- PROJECT-002: Essential file creation (partial)
- Initial skeleton creation per IMPLEMENTATION_GUIDE.md Days 1-7