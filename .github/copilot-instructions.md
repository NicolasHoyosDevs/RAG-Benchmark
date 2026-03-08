# Project Guidelines

## Build and Test
- Create a Python environment and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Create `.env` from `.env.example` and set `OPENAI_API_KEY`.
- Initialize vector DB before running any RAG module: `python scripts/create_embeddings.py`.
- Quick retrieval smoke test: `python scripts/test_retrieval.py`.
- Run interactive RAG modules directly when needed:
  - `python src/rag/simple.py`
  - `python src/rag/hybrid.py`
  - `python src/rag/hyde.py`
  - `python src/rag/rewriter.py`
- Run benchmark/evaluation:
  - `python scripts/run_evaluation.py simple|hybrid|hyde|rewriter`
  - `python scripts/run_evaluation.py multi-model <rag_type>`
  - `python scripts/run_evaluation.py all-models-all-rags`

## Architecture
- `data/` contains preprocessing outputs, chunk JSON, and persistent Chroma DB.
- Four RAG implementations live in separate modules:
  - `src/rag/simple.py`
  - `src/rag/hybrid.py`
  - `src/rag/hyde.py`
  - `src/rag/rewriter.py`
- `src/evaluation/ragas_evaluator.py` is the central orchestrator for RAGAS evaluation and imports each module's `query_for_evaluation` function.
- Module-level initialization (LLMs/retrievers/vector stores) is used across the codebase; keep startup side effects predictable.

## Code Style
- Keep Python code simple and explicit, following existing patterns: type hints, docstrings, and `pathlib.Path` for paths.
- Prefer absolute paths built from `Path(__file__).resolve()` to avoid working-directory bugs.
- Preserve returned data contracts used by evaluation code; avoid ad hoc field renaming.

## Conventions
- Domain is obstetric/pregnancy medical Q&A. Prompts and evaluation questions are domain-specific.
- Final answers must be in Spanish. Existing prompts enforce this; do not remove this behavior unless requested.
- Shared retrieval defaults are important for comparability: Chroma collection `guia_embarazo_parto`, embeddings model `text-embedding-3-small`, typical retrieval `k=5`.
- `query_for_evaluation(...)` should return a dict with at least `question`, `answer`, `contexts`, and `metadata` keys.
- When adding a new RAG variant, mirror current module structure and wire it into `src/evaluation/ragas_evaluator.py` selection logic.

## Pitfalls
- Missing `OPENAI_API_KEY` fails fast during imports.
- `Hybrid RAG` depends on both ChromaDB and `data/chunks/chunks_final.json` (for BM25 corpus).
- Evaluations can trigger many OpenAI calls and generate cost; prefer targeted runs while iterating.
- Repository includes generated artifacts (`data/embeddings/chroma_db/`, `results/*.json`); avoid unnecessary regeneration in routine edits.

## Key References
- `README.md`
- `scripts/create_embeddings.py`
- `src/rag/simple.py`
- `src/evaluation/ragas_evaluator.py`