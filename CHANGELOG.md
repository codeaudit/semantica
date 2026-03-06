# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- **Multi-Founder LLM Extraction & Reasoner Inference Fix** (PR #354 by @KaifAhmad1):
  - Fixed `_parse_relation_result` in `methods.py` — unmatched subjects/objects now produce a synthetic `UNKNOWN` entity instead of silently dropping the relation; all LLM-returned co-founders are preserved
  - Rewrote `_match_pattern` in `reasoner.py` — splits pattern on `?var` placeholders first, then escapes only the literal segments; pre-bound variables resolve to exact literals, repeated variables use backreferences, non-greedy `.+?` prevents over-consumption of literal separators
  - Added `tests/reasoning/test_reasoner.py` with 4 tests covering multi-word value inference, pre-bound variables, binding conflicts, and single-word regression
  - Added `tests/semantic_extract/test_relation_extractor.py` with 6 tests covering all-founders returned, synthetic entity creation, matched entity integrity, predicate/confidence preservation, empty response, and malformed entries

- **Incremental/Delta Processing Feature** (PR #349 by @ZohaibHassan16, reviewed and fixed by @KaifAhmad1):
  - Native delta computation between graph snapshots using SPARQL queries
  - Delta-aware pipeline execution with `delta_mode` configuration for processing only changed data
  - Version snapshot management with graph URI tracking and metadata storage
  - Snapshot retention policies with automatic cleanup via `prune_versions()` method
  - Integration with pipeline execution engine for incremental workflows
  - Significant performance improvements: processes only changes instead of full datasets
  - Cost optimization: dramatically reduces compute and storage requirements for large-scale operations
  - Production-ready for near real-time pipelines and frequent deployment scenarios
  - Bug fixes: corrected SPARQL variable order, fixed class references, resolved duplicate dictionary keys
  - Comprehensive test coverage including delta mode integration tests
  - Complete documentation with usage examples and API references
  - Essential for enterprise-grade, large-scale semantic infrastructure

## [0.0.5] - 2025-11-26

### Changed
- Configured Trusted Publishing for secure automated PyPI deployments

## [0.0.4] - 2025-11-26

### Changed
- Fixed PyPI deployment issues from v0.0.3

## [0.0.3] - 2025-11-25

### Changed
- Simplified CI/CD workflows - removed failing tests and strict linting
- Combined release and PyPI publishing into single workflow
- Simplified security scanning to weekly pip-audit only
- Streamlined GitHub Actions configuration

### Added
- Comprehensive issue templates (Bug, Feature, Documentation, Support, Grant/Partnership)
- Updated pull request template with clear guidelines
- Community support documentation (SUPPORT.md)
- Funding and sponsorship configuration (FUNDING.yml)
- GitHub configuration README for maintainers
- 10+ new domain-specific cookbook examples (Finance, Healthcare, Cybersecurity, etc.)

### Removed
- Redundant scripts folder (8 shell/PowerShell scripts)
- Unnecessary automation workflows (label-issues, mark-answered)
- Excessive issue templates

## [0.0.2] - 2025-11-25

### Changed
- Updated README with streamlined content and better examples
- Added more notebooks to cookbook
- Improved documentation structure

## [0.0.1] - 2024-01-XX

### Added
- Core framework architecture
- Universal data ingestion (50+ file formats)
- Semantic intelligence engine (NER, relation extraction, event detection)
- Knowledge graph construction with entity resolution
- 6-stage ontology generation pipeline
- GraphRAG engine for hybrid retrieval
- Multi-agent system infrastructure
- Production-ready quality assurance modules
- Comprehensive documentation with MkDocs
- Cookbook with interactive tutorials
- Support for multiple vector stores (Weaviate, Qdrant, FAISS)
- Support for multiple graph databases (Neo4j, NetworkX, RDFLib)
- Temporal knowledge graph support
- Conflict detection and resolution
- Deduplication and entity merging
- Schema template enforcement
- Seed data management
- Multi-format export (RDF, JSON-LD, CSV, GraphML)
- Visualization tools
- Pipeline orchestration
- Streaming support (Kafka, RabbitMQ, Kinesis)
- Context engineering for AI agents
- Reasoning and inference engine

### Documentation
- Getting started guide
- API reference for all modules
- Concepts and architecture documentation
- Use case examples
- Cookbook tutorials
- Community projects showcase

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

## Migration Guides

When breaking changes are introduced, migration guides will be provided in the release notes and documentation.

---

For detailed release notes, see [GitHub Releases](https://github.com/Hawksight-AI/semantica/releases).

