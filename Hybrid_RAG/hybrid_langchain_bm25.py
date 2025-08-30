"""
Simplified Hybrid RAG using LangChain's EnsembleRetriever.

This script implements a Hybrid RAG pipeline combining lexical search (BM25)
and semantic search (ChromaDB) using LangChain's EnsembleRetriever.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Environment and Path Configuration ---

# Load environment variables from .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in the .env file")

# Define paths
script_dir = Path(__file__).resolve().parent
chroma_db_dir = script_dir.parent / "Data" / "embeddings" / "chroma_db"
collection_name = "guia_embarazo_parto"
chunks_file = script_dir.parent / "Data" / "chunks" / "chunks_final.json"

# --- Document Loading ---


def load_documents() -> List[Document]:
    """Loads chunks from the JSON file and converts them to LangChain Documents."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    return [
        Document(page_content=d['content'], metadata=d)
        for d in chunks_data
    ]


documents = load_documents()

# --- Model and Retriever Configuration ---

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 1. Lexical Retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 2. Semantic Retriever (Chroma)
vectorstore = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Ensemble Retriever
ensemble_weight_bm25 = 0.2
ensemble_weight_semantic = 0.8
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[ensemble_weight_bm25, ensemble_weight_semantic]
)


# --- Prompt Templates ---

# Prompt for generating the final answer
qa_template = """
You are an expert in maternal health and pregnancy. Analyze the following medical context and answer the question accurately and in detail.

INSTRUCTIONS:
- Use ONLY the information provided in the context.
- If the information is sufficient, provide a detailed answer.
- If there is not enough information, state that clearly.
- Remember that you are a medical specialist answering queries about pregnancy and childbirth.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

DETAILED MEDICAL ANSWER:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)


# --- Core Functions ---

def format_docs(docs: List[Document]) -> str:
    """
    Formats the retrieved documents to be included in the final prompt.

    Args:
        docs (list): A list of retrieved LangChain Document objects.

    Returns:
        str: A formatted string containing the content of the documents.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page_number', 'N/A')

        formatted_doc = f"""--- Document {i+1} ---
Source: {source}, Page: {page}
Content: {doc.page_content}"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


def process_hybrid_query(query: str) -> Dict[str, Any]:
    """
    Processes a query using the hybrid RAG pipeline.

    Args:
        query (str): The user's question.

    Returns:
        Dict[str, Any]: A dictionary with the final answer, contexts, and detailed metrics.
    """
    # 1. Retrieve similar documents using the ensemble retriever
    retrieved_docs = ensemble_retriever.invoke(query)

    # 2. Format context
    formatted_context = format_docs(retrieved_docs)

    # 3. Generate final answer
    with get_openai_callback() as cb_answer:
        response = llm.invoke(qa_prompt.format_messages(
            context=formatted_context,
            question=query
        ))

    # 4. Return response and all metrics
    return {
        'answer': response.content,
        'contexts': [doc.page_content for doc in retrieved_docs],
        'retrieved_documents': retrieved_docs,
        'metrics': {
            'input_tokens': cb_answer.prompt_tokens,
            'output_tokens': cb_answer.completion_tokens,
            'cost': cb_answer.total_cost
        }
    }


def query_for_evaluation(question: str) -> dict:
    """
    A wrapper function for RAG evaluation frameworks like Ragas.

    This function processes a question and returns a dictionary structured for
    easy integration with evaluation tools, preserving the original output format.

    Args:
        question (str): The question to process.

    Returns:
        dict: A dictionary containing the question, answer, contexts, source_documents, and metadata.
    """
    start_time = time.time()
    result = process_hybrid_query(question)
    end_time = time.time()
    execution_time = end_time - start_time

    input_tokens = result["metrics"]["input_tokens"]
    output_tokens = result["metrics"]["output_tokens"]

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "source_documents": result["retrieved_documents"],
        "metadata": {
            "num_contexts": len(result["contexts"]),
            "retrieval_method": "hybrid_bm25_semantic",
            "ensemble_weights": [ensemble_weight_bm25, ensemble_weight_semantic],
            "llm_model": "gpt-4o",
            "embedding_model": "text-embedding-3-small",
            "execution_time": execution_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": result["metrics"]["cost"],
            "tokens_used": input_tokens + output_tokens,
        }
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    print("\n=== Hybrid RAG (LangChain BM25 + Semantic) ===")
    print("This system uses a hybrid search to retrieve relevant documents and generate an answer.")
    print(f"Documents loaded: {len(documents)}")
    try:
        print(f"Vector store documents: {vectorstore._collection.count()}")
    except Exception as e:
        print(f"Could not retrieve vector store document count: {e}")
    print("\nType your question or 'exit' to finish.")

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        result = process_hybrid_query(query)
        end_time = time.time()

        print("\n" + "="*50)
        print("ANSWER:")
        print(result['answer'])
        print("\n" + "="*50)

        # Display detailed metrics
        print("\n DETAILED METRICS:")
        print(f"     Total time: {end_time - start_time:.2f} seconds")
        print(
            f"   - Input Tokens (prompt): {result['metrics']['input_tokens']}")
        print(
            f"   - Output Tokens (answer): {result['metrics']['output_tokens']}")
        print(f"   - Total Cost (USD): ${result['metrics']['cost']:.6f}")

    print("\nSystem finished.")
