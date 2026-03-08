# System Design

## Goals
- Keep a consistent, research-grade repository layout.
- Separate source code, execution scripts, data assets, and results.
- Preserve benchmark comparability and output traceability.

## Top-Level Structure
- `src/`: reusable Python packages.
- `scripts/`: executable entrypoints for setup and evaluation.
- `data/`: datasets, chunks, and vector store artifacts.
- `results/`: versioned benchmark outputs.
- `docs/`: architecture and operational guides.
- `config/`: configuration files and future environment presets.
- `tests/`: automated validation.

## Source Modules
- `src/rag/simple.py`: semantic-only retriever baseline.
- `src/rag/hybrid.py`: BM25 + semantic ensemble retriever.
- `src/rag/hyde.py`: HyDE retrieval with hypothetical document generation.
- `src/rag/rewriter.py`: multi-query rewrite retrieval flow.
- `src/evaluation/ragas_evaluator.py`: central orchestration for RAGAS evaluations.

## Execution Entry Points
- `scripts/create_embeddings.py`: generate Chroma vectors from chunks.
- `scripts/test_retrieval.py`: smoke-test retrieval against persisted vectors.
- `scripts/view_embeddings.py`: inspect Chroma collection contents.
- `scripts/run_evaluation.py`: canonical benchmark CLI.

## Compatibility
- `results/ragas_evaluator.py` remains as a wrapper that forwards to `src.evaluation.ragas_evaluator.main`.

## Naming Conventions
- Folders and files in English.
- Python modules in `snake_case`.
- Classes in `PascalCase`.
- Keep result filenames timestamped for reproducibility.
