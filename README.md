[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NicolasHoyosDevs/RAG-Benchmark)

# RAG Benchmark System

A comprehensive benchmarking framework for evaluating Retrieval-Augmented Generation (RAG) systems using RAGAS metrics. This project implements and compares multiple RAG architectures including Simple Semantic RAG, Hybrid RAG (BM25 + Semantic), HyDE RAG, and Query Rewriter RAG.

## Overview

This project provides a complete pipeline for:
- **Data Processing**: Text chunking, embedding creation, and vector storage
- **RAG Implementation**: Four different RAG architectures
- **Evaluation**: Comprehensive evaluation using RAGAS framework
- **Comparison**: Automated comparison across different models and architectures
- **Visualization**: Results analysis and reporting

## Project Structure

```
RAG-Benchmark/
├── Data/
│   ├── raw/                     # Raw documents
│   ├── processed/               # Processed documents
│   ├── chunks/                  # Text chunks (JSON)
│   ├── embeddings/              # Embedding creation and storage
│   │   ├── create_embeddings.py # Create embeddings script
│   │   ├── test_retrieval.py    # Test retrieval functionality
│   │   ├── view_embeddings.py   # View embedding data
│   │   └── chroma_db/          # ChromaDB vector database
│   └── parsed_docs/            # Parsed document files
├── Simple_Semantic_RAG/        # Simple semantic search RAG
│   └── simple_semantic_rag.py
├── Hybrid_RAG/                 # Hybrid BM25 + Semantic RAG
│   └── hybrid_langchain_bm25.py
├── HyDE_RAG/                   # Hypothetical Document Embeddings RAG
│   └── hyde_rag.py
├── Query_Rewriter_RAG/         # Query rewriting RAG
│   └── main_rewriter.py
├── results/                    # Evaluation results and analysis
│   ├── ragas_evaluator.py      # Main evaluation script
│   ├── utils.py                # Utility functions
│   ├── ragas_analysis/         # Analysis tools and reports
│   └── [JSON files]            # Evaluation results
├── benchmark_ragas.py          # Benchmark script
├── test_comparison.py          # Comparison testing
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### 1. Clone and Setup

```bash
git clone https://github.com/NicolasHoyosDevs/RAG-Benchmark.git
cd RAG-Benchmark
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Prepare Data

#### Option A: Use Existing Data
The project includes pre-processed data. Skip to step 5.

#### Option B: Process New Data
If you want to process your own documents:

1. Place your documents in `Data/raw/`
2. Run the preprocessing pipeline (if available)
3. Create chunks and embeddings

### 5. Create Embeddings

```bash
cd Data/embeddings
python create_embeddings.py
```

This will:
- Load text chunks from `Data/chunks/chunks_final.json`
- Create embeddings using OpenAI's text-embedding-3-small
- Store them in ChromaDB at `Data/embeddings/chroma_db/`

## RAG Architectures

### 1. Simple Semantic RAG
- Uses semantic similarity search
- Direct retrieval from vector database
- Fast and straightforward approach

### 2. Hybrid RAG (BM25 + Semantic)
- Combines BM25 keyword search with semantic search
- Better retrieval accuracy for diverse queries
- Balances precision and recall

### 3. HyDE RAG (Hypothetical Document Embeddings)
- Generates hypothetical documents for the query
- Uses embeddings of hypothetical content for retrieval
- Effective for complex or abstract queries

### 4. Query Rewriter RAG
- Rewrites queries in multiple ways
- Performs multiple retrievals with different query formulations
- Improves results for ambiguous queries

## Evaluation with RAGAS

The system uses RAGAS (Retrieval-Augmented Generation Assessment) for comprehensive evaluation:

### Metrics Evaluated:
- **Faithfulness**: How well the response matches the retrieved context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: Precision of retrieved context
- **Context Recall**: Recall of retrieved context

## Available Commands

### Individual RAG Evaluation

Evaluate each RAG system individually:

```bash
# Simple Semantic RAG
python results/ragas_evaluator.py simple

# Hybrid RAG
python results/ragas_evaluator.py hybrid

# HyDE RAG
python results/ragas_evaluator.py hyde

# Query Rewriter RAG
python results/ragas_evaluator.py rewriter
```

### Multi-Model Evaluation

Evaluate a specific RAG with multiple models:

```bash
# Evaluate Hybrid RAG with all models
python results/ragas_evaluator.py multi-model hybrid

# Evaluate Simple RAG with all models
python results/ragas_evaluator.py multi-model simple
```

### Comprehensive Evaluation

Evaluate all RAGs with all models in a single run:

```bash
python results/ragas_evaluator.py all-models-all-rags
```

This command will:
- Test all 4 RAG architectures
- Use all 4 GPT models (gpt-3.5-turbo, gpt-4o, gpt-4o-mini, gpt-4)
- Generate 16 evaluation runs
- Create a consolidated JSON file with all results

### Other Commands

```bash
# Run benchmark script
python benchmark_ragas.py

# Run comparison tests
python test_comparison.py

# View embedding data
cd Data/embeddings
python view_embeddings.py

# Test retrieval functionality
cd Data/embeddings
python test_retrieval.py
```

## Results and Analysis

### Output Files
Results are saved in the `results/` directory as JSON files:
- `ragas_evaluation_[type]_[timestamp].json` - Individual evaluations
- `ragas_comprehensive_all_rags_all_models_[timestamp].json` - Complete evaluation

### JSON Output Structure

#### Individual RAG Evaluation JSON Structure
```json
{
  "metadata": {
    "rag_type": "hybrid",
    "model_used": "gpt-4o",
    "timestamp": "20250830_181136",
    "total_questions": 5,
    "evaluation_duration": "45.2s"
  },
  "rag_results": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.92,
    "context_recall": 0.76
  },
  "question_by_question": [
    {
      "question": "What are the main stages of pregnancy?",
      "ground_truth": "Pregnancy is divided into three trimesters...",
      "answer": "Pregnancy consists of three main trimesters...",
      "contexts": ["Pregnancy is divided into...", "First trimester includes..."],
      "faithfulness": 0.88,
      "answer_relevancy": 0.82,
      "context_precision": 0.95,
      "context_recall": 0.79
    }
  ]
}
```

#### Comprehensive All-Models-All-RAGs JSON Structure
```json
{
  "metadata": {
    "evaluation_type": "all-models-all-rags",
    "timestamp": "20250830_181136",
    "total_evaluations": 16,
    "total_questions": 5,
    "models_tested": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4"],
    "rags_tested": ["simple", "hybrid", "hyde", "rewriter"]
  },
  "summary": {
    "best_performing_rag": "hybrid",
    "best_performing_model": "gpt-4",
    "highest_faithfulness": 0.89,
    "highest_answer_relevancy": 0.84
  },
  "rag_results": {
    "simple": {
      "gpt-3.5-turbo": {
        "faithfulness": 0.78,
        "answer_relevancy": 0.72,
        "context_precision": 0.85,
        "context_recall": 0.69
      },
      "gpt-4o": {
        "faithfulness": 0.82,
        "answer_relevancy": 0.76,
        "context_precision": 0.88,
        "context_recall": 0.73
      }
    },
    "hybrid": {
      "gpt-3.5-turbo": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.79,
        "context_precision": 0.91,
        "context_recall": 0.75
      },
      "gpt-4o": {
        "faithfulness": 0.89,
        "answer_relevancy": 0.84,
        "context_precision": 0.94,
        "context_recall": 0.81
      }
    },
    "hyde": {
      "gpt-3.5-turbo": {
        "faithfulness": 0.81,
        "answer_relevancy": 0.77,
        "context_precision": 0.87,
        "context_recall": 0.71
      }
    },
    "rewriter": {
      "gpt-3.5-turbo": {
        "faithfulness": 0.83,
        "answer_relevancy": 0.78,
        "context_precision": 0.89,
        "context_recall": 0.74
      }
    }
  },
  "detailed_results": {
    "simple_gpt-3.5-turbo": {
      "metadata": {
        "rag_type": "simple",
        "model_used": "gpt-3.5-turbo",
        "timestamp": "20250830_181136"
      },
      "question_by_question": [...]
    }
  }
}
```

#### Key JSON Fields Explained

- **`metadata`**: Contains evaluation information (RAG type, model used, timestamp, etc.)
- **`rag_results`**: Aggregated metrics for the evaluation
- **`question_by_question`**: Detailed results for each test question including:
  - Original question
  - Ground truth answer
  - Generated answer
  - Retrieved contexts
  - Individual metric scores
- **`summary`**: Overview of best performers (in comprehensive evaluations)
- **`detailed_results`**: Complete breakdown by RAG-model combination

#### Metrics Description
- **Faithfulness** (0-1): How well the answer matches the retrieved context
- **Answer Relevancy** (0-1): How relevant the answer is to the question
- **Context Precision** (0-1): Precision of the retrieved context chunks
- **Context Recall** (0-1): How well the context covers the ground truth

### Analysis Tools
Use the analysis tools in `results/ragas_analysis/`:
- View detailed metrics
- Compare performance across RAGs and models
- Generate reports and visualizations

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration
The system supports these OpenAI models:
- gpt-3.5-turbo
- gpt-4o
- gpt-4o-mini
- gpt-4

Models are automatically switched during multi-model evaluations.

## Customization

### Adding New Documents
1. Place documents in `Data/raw/`
2. Process them into chunks
3. Update `Data/chunks/chunks_final.json`
4. Re-run embedding creation

### Modifying RAG Parameters
Edit the respective RAG files:
- `Simple_Semantic_RAG/simple_semantic_rag.py`
- `Hybrid_RAG/hybrid_langchain_bm25.py`
- `HyDE_RAG/hyde_rag.py`
- `Query_Rewriter_RAG/main_rewriter.py`

### Custom Evaluation Metrics
Modify `results/ragas_evaluator.py` to add custom metrics or evaluation logic.

## Documentation

### Data Processing
- Documents are chunked and stored as JSON
- Embeddings are created using OpenAI's embedding models
- Vector database uses ChromaDB for efficient similarity search

### RAG Implementation Details
Each RAG architecture is implemented as a separate module with:
- Document ingestion
- Query processing
- Retrieval logic
- Response generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Support

- **Issues**: Create an issue on GitHub
- **Documentation**: This README and inline code comments
- **Community**: Check existing issues and discussions

## Example Usage

```python
# Example: Evaluate Hybrid RAG
from results.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator()
results = evaluator.evaluate_rag("hybrid", "gpt-4o")
print(f"Faithfulness: {results['faithfulness']}")
print(f"Answer Relevancy: {results['answer_relevancy']}")
```

For more advanced usage, see the individual RAG implementation files and the evaluation script.
