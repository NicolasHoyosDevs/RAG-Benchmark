"""
RAG with Multi-Query Rewriting for enhanced retrieval.

This script implements a RAG pipeline that uses a multi-query rewriting
strategy to improve document retrieval. It generates several variations of the
user's question, retrieves documents for each variation, and then synthesizes
an answer based on the combined, re-ranked results.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document

# --- Environment and Path Configuration ---

# Load environment variables from .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in the .env file")

# --- ChromaDB Configuration ---
DB_DIRECTORY = Path(__file__).resolve().parent.parent / \
    "Data" / "embeddings" / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"

# --- Model and Vector Store Configuration ---

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load ChromaDB vector store
try:
    vectorstore = Chroma(
        persist_directory=str(DB_DIRECTORY),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    vectorstore = None

# Configure the base retriever
base_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 8, "score_threshold": 0.05}
)

# --- Query Rewriting Configuration ---

REPHRASE_TEMPLATE_1 = """Rewrite this question to be a standalone, specific query about pregnancy and childbirth.

Original question: {question}

Instructions:
- Maintain the medical/obstetric context if relevant.
- Be specific and clear in medical terms.
- Focus on pregnancy, childbirth, prenatal care, or maternal health.
- Ensure the question is complete and self-contained.

Standalone question:"""

REPHRASE_TEMPLATE_2 = """Rephrase this question about pregnancy and childbirth using synonyms and alternative medical terms.

Original question: {question}

Instructions:
- Use precise medical terminology.
- Include synonyms and alternative terms.
- Maintain the meaning but change the wording.
- Focus on clinical and obstetric aspects.

Rephrased question:"""

REPHRASE_TEMPLATE_3 = """Expand this question to include related aspects and additional context about pregnancy and childbirth.

Base question: {question}

Instructions:
- Expand the question to include related aspects.
- Add context about complications, prevention, or care.
- Include possible variations or special cases.
- Keep the focus on maternal and perinatal health.

Expanded question:"""

REPHRASE_PROMPTS = [
    PromptTemplate.from_template(REPHRASE_TEMPLATE_1),
    PromptTemplate.from_template(REPHRASE_TEMPLATE_2),
    PromptTemplate.from_template(REPHRASE_TEMPLATE_3)
]

llm_rewriter = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
llm_answer = ChatOpenAI(model_name="gpt-4o", temperature=0)


# --- Final Answer Prompt ---

qa_template = """
You are a medical specialist expert in pregnancy and childbirth. Analyze ONLY the provided context and answer the question accurately and directly.

STRICT RULES:
1. Answer the question DIRECTLY using ONLY the information from the context.
2. PRIORITIZE Documents 1 and 2 as they have the highest relevance to the question.
3. DO NOT add external knowledge that is not in the context.
4. If there are specific data (dosages, numbers, protocols), include them exactly as they appear.

PRIORITY ORDER:
- Documents 1-2: HIGH relevance - use as the main source.
- Documents 3-4: MEDIUM relevance - use as a supplement.
- Documents 5+: LOW relevance - use only if necessary.

RESPONSE FORMAT:
1. Direct answer to the question (based mainly on Documents 1-2).
2. Specific details from the context.
3. Relevant related information (if applicable).
4. Limitations of the available information (only if any).

IMPORTANT: This information is for educational purposes and does not replace professional medical consultation.

Context from medical guides (ordered by relevance):
{context}

Question: {question}

Medical answer based on the context:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)


# --- Core Functions ---

def format_docs(docs: List[Document]) -> str:
    """Formats documents for the context, indicating relevance."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        relevance = 'High' if i < 2 else 'Medium' if i < 4 else 'Low'
        formatted_doc = f"--- DOCUMENT {i+1} (Relevance: {relevance}) ---\n"
        formatted_doc += f"Source: {source}\n"
        formatted_doc += f"Content: {doc.page_content}"
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)


def process_rewriter_query(question: str, max_final_docs: int = 8) -> Dict[str, Any]:
    """
    Processes a query using the multi-query rewriting RAG pipeline.

    Args:
        question (str): The user's question.
        max_final_docs (int): The maximum number of documents to return.

    Returns:
        Dict[str, Any]: A dictionary with the final answer, contexts, and detailed metrics.
    """
    # 1. Generate rewritten queries and track metrics
    rewritten_queries = []
    rewrite_input_tokens, rewrite_output_tokens, rewrite_cost = 0, 0, 0

    for prompt in REPHRASE_PROMPTS:
        with get_openai_callback() as cb:
            rewritten_query = (prompt | llm_rewriter | StrOutputParser()).invoke(
                {"question": question}).strip()
            rewritten_queries.append(rewritten_query)
            rewrite_input_tokens += cb.prompt_tokens
            rewrite_output_tokens += cb.completion_tokens
            rewrite_cost += cb.total_cost

    # 2. Retrieve documents for each rewritten query
    all_docs_with_scores = []
    doc_ids_seen = set()

    for i, query in enumerate(rewritten_queries, 1):
        results = vectorstore.similarity_search_with_score(query, k=5)
        for doc, distance in results:
            similarity = max(0.0, 1.0 - distance)
            # Use a slice of content as a unique ID
            doc_id = doc.page_content[:100]
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                # Penalize queries from later, more speculative prompts
                query_weight = 1.0 - (i - 1) * 0.05
                all_docs_with_scores.append((doc, similarity * query_weight))

    # 3. Re-rank and select the best documents
    all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    retrieved_docs = [doc for doc, _ in all_docs_with_scores[:max_final_docs]]

    # 4. Format context and generate final answer
    formatted_context = format_docs(retrieved_docs)
    with get_openai_callback() as cb_answer:
        answer = (qa_prompt | llm_answer | StrOutputParser()).invoke({
            "context": formatted_context,
            "question": question
        })

    # 5. Consolidate and return all information
    total_input = rewrite_input_tokens + cb_answer.prompt_tokens
    total_output = rewrite_output_tokens + cb_answer.completion_tokens
    total_cost = rewrite_cost + cb_answer.total_cost

    return {
        'answer': answer,
        'contexts': [doc.page_content for doc in retrieved_docs],
        'retrieved_documents': retrieved_docs,
        'rewritten_queries': rewritten_queries,
        'metrics': {
            'rewrite_input_tokens': rewrite_input_tokens,
            'rewrite_output_tokens': rewrite_output_tokens,
            'rewrite_cost': rewrite_cost,
            'answer_input_tokens': cb_answer.prompt_tokens,
            'answer_output_tokens': cb_answer.completion_tokens,
            'answer_cost': cb_answer.total_cost,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_cost': total_cost,
        }
    }


def query_for_evaluation(question: str) -> dict:
    """
    A wrapper function for RAG evaluation frameworks like Ragas.

    This function processes a question and returns a dictionary structured for
    easy integration with evaluation tools, preserving the original output format.
    """
    start_time = time.time()
    result = process_rewriter_query(question)
    end_time = time.time()
    execution_time = end_time - start_time

    total_input = result['metrics']['total_input_tokens']
    total_output = result['metrics']['total_output_tokens']

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "source_documents": result["retrieved_documents"],
        "metadata": {
            "num_contexts": len(result["contexts"]),
            "retrieval_method": "multi_query_rewrite",
            "rewrite_count": len(REPHRASE_PROMPTS),
            "llm_model": "gpt-4o",
            "rewriter_model": "gpt-3.5-turbo",
            "execution_time": execution_time,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_cost": result['metrics']['total_cost'],
            "tokens_used": total_input + total_output,
        }
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    print("\n=== RAG with Multi-Query Rewriter ===")
    print("This system rewrites your question in multiple ways to improve document retrieval.")
    if vectorstore:
        print(
            f"Connected to ChromaDB with {vectorstore._collection.count()} documents.")
    print("\nType your question or 'exit' to finish.")

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        result = process_rewriter_query(query)
        end_time = time.time()

        print("\n" + "="*50)
        print("REWRITTEN QUERIES:")
        for i, rq in enumerate(result['rewritten_queries']):
            print(f"  {i+1}. {rq}")

        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print(result['answer'])
        print("\n" + "="*50)

        # Display detailed metrics
        print("\n DETAILED METRICS:")
        print(f"     Total time: {end_time - start_time:.2f} seconds")

        print("\n QUERY REWRITING:")
        print(
            f"   - Input Tokens: {result['metrics']['rewrite_input_tokens']}")
        print(
            f"   - Output Tokens: {result['metrics']['rewrite_output_tokens']}")
        print(f"   - Cost: ${result['metrics']['rewrite_cost']:.6f}")

        print("\n FINAL ANSWER GENERATION:")
        print(f"   - Input Tokens: {result['metrics']['answer_input_tokens']}")
        print(
            f"   - Output Tokens: {result['metrics']['answer_output_tokens']}")
        print(f"   - Cost: ${result['metrics']['answer_cost']:.6f}")

        print("\n TOTALS:")
        print(
            f"   - Total Input Tokens: {result['metrics']['total_input_tokens']}")
        print(
            f"   - Total Output Tokens: {result['metrics']['total_output_tokens']}")
        print(f"   - Total Cost (USD): ${result['metrics']['total_cost']:.6f}")

    print("\nSystem finished.")
