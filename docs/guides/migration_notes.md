# Migration Notes: Legacy Layout to src-based Layout

## Completed Moves
- `Data/` -> `data/`
- `Simple_Semantic_RAG/simple_semantic_rag.py` -> `src/rag/simple.py`
- `Hybrid_RAG/hybrid_langchain_bm25.py` -> `src/rag/hybrid.py`
- `HyDE_RAG/hyde_rag.py` -> `src/rag/hyde.py`
- `Query_Rewriter_RAG/main_rewriter.py` -> `src/rag/rewriter.py`
- `results/ragas_evaluator.py` logic -> `src/evaluation/ragas_evaluator.py`
- Embedding utilities -> `scripts/`

## New Commands
- Create embeddings: `python3 scripts/create_embeddings.py`
- Retrieval smoke test: `python3 scripts/test_retrieval.py`
- Evaluate a RAG: `python3 scripts/run_evaluation.py simple`
- Multi-model evaluation: `python3 scripts/run_evaluation.py multi-model hybrid`

## Notes
- Historical JSON results remain under `results/`.
- Evaluation contract (`query_for_evaluation`) is preserved across all RAG variants.
- Current evaluator still supports the legacy command through `results/ragas_evaluator.py` wrapper.
