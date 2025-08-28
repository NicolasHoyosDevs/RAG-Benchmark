"""Hybrid RAG usando BM25 de LangChain para la parte lexical.

Este archivo reemplaza la implementación manual de BM25 por el BM25Retriever
de LangChain, pero mantiene:
 - Búsqueda semántica vía Chroma + OpenAI embeddings
 - Lógica de fusión (lineal y RRF)
 - Normalización de scores y trazabilidad de cada documento


Nota: BM25Retriever actualmente no expone directamente los scores en los
Document que retorna. Para poder integrar con la fusión necesitamos acceder
al objeto interno `bm25` (BM25Okapi) y calcular los scores completos, luego
ordenar y recortar al top-k.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
load_dotenv(dotenv_path='../.env')

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")


class HybridRAGResearchLCBM25:
    """Versión híbrida que usa BM25Retriever de LangChain.

    Mantiene los métodos de fusión y la estructura de retorno de documentos
    para poder ser intercambiable con la versión previa.
    """

    def __init__(self):
        # Rutas y nombres
        self.chroma_db_dir = Path("../Data/embeddings/chroma_db")
        self.collection_name = "guia_embarazo_parto"
        self.chunks_file = Path("../Data/chunks/chunks_final.json")

        # Modelos
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Vector store (semántico)
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_db_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        # Cargar documentos base
        self.documents: List[Dict] = self._load_documents()

        # Construir lista de Document para BM25Retriever
        self._build_bm25_retriever()

        print("✅ Sistema híbrido (LangChain BM25) inicializado:")
        print(f"   📄 Documentos cargados: {len(self.documents)}")

    # ----------------------------- Carga ----------------------------- #
    def _load_documents(self) -> List[Dict]:
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_bm25_retriever(self):
        """Construye el BM25Retriever de LangChain.

        Guardamos la referencia a los Document en el mismo orden que el JSON
        para poder mapear índices (enumerate) a self.documents.
        """
        lc_docs: List[Document] = []
        for d in self.documents:
            lc_docs.append(Document(page_content=d['content'], metadata=d))

        # Crear retriever
        self.bm25_retriever: BM25Retriever = BM25Retriever.from_documents(
            lc_docs)

        self.bm25_obj = getattr(self.bm25_retriever, 'vectorizer')

    # ----------------------------- Búsquedas ----------------------------- #
    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        semantic_results: List[Tuple[int, float]] = []

        for doc, distance in results:
            similarity = 1 / (1 + distance)
            page_num = doc.metadata.get('page_number')
            chunk_idx = doc.metadata.get('chunk_index')

            # Mapear al índice del JSON original
            mapped_index = None
            # Optimización: metemos un dict de lookup si se vuelve costoso; por ahora lineal
            for i, d in enumerate(self.documents):
                if d.get('page_number') == page_num and d.get('chunk_index') == chunk_idx:
                    mapped_index = i
                    break
            if mapped_index is not None:
                semantic_results.append((mapped_index, similarity))
        return semantic_results

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Obtiene scores BM25 desde el retriever de LangChain.

        Si tenemos acceso al objeto BM25 interno, extraemos los scores.
        Si no, usamos ranking ordinal basado en el orden de documentos devueltos.
        """
        if self.bm25_obj and hasattr(self.bm25_obj, 'get_scores'):
            # Método 1: Acceso directo a scores
            if hasattr(self.bm25_retriever, '_tokenizer'):
                tokens_query = self.bm25_retriever._tokenizer(query)
            else:
                tokens_query = query.lower().split()

            scores_all = self.bm25_obj.get_scores(tokens_query)
            indexed_scores = list(enumerate(scores_all))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            return indexed_scores[:k]
        else:
            # Método 2: Usar retriever normal y asignar scores ordinales
            print("   📝 Usando BM25 ordinal (sin scores exactos)")
            docs = self.bm25_retriever.get_relevant_documents(query)
            results = []

            for rank, doc in enumerate(docs[:k]):
                # Mapear documento a índice original
                doc_idx = None
                for i, orig_doc in enumerate(self.documents):
                    if orig_doc['content'] == doc.page_content:
                        doc_idx = i
                        break

                if doc_idx is not None:
                    # Score ordinal: máximo para el primero, decrece linealmente
                    score = max(0, k - rank) / k
                    results.append((doc_idx, score))

            return results

    # ----------------------------- Fusión ----------------------------- #
    def _normalize_scores(self, scores: List[Tuple[int, float]]):
        if not scores:
            return {}
        max_score = max(s for _, s in scores)
        if max_score == 0:
            return {i: 0.0 for i, _ in scores}
        return {i: s / max_score for i, s in scores}

    def _linear_fusion(self, semantic_results: List[Tuple[int, float]], bm25_results: List[Tuple[int, float]], alpha: float):
        sem_norm = self._normalize_scores(semantic_results)
        bm25_norm = self._normalize_scores(bm25_results)
        all_ids = set(sem_norm) | set(bm25_norm)
        return {
            i: alpha * sem_norm.get(i, 0.0) +
            (1 - alpha) * bm25_norm.get(i, 0.0)
            for i in all_ids
        }

    def _rrf_fusion(self, semantic_results: List[Tuple[int, float]], bm25_results: List[Tuple[int, float]], alpha: float, k_rrf: int = 60):
        combined = defaultdict(float)
        for rank, (idx, _) in enumerate(semantic_results):
            combined[idx] += alpha * (1 / (k_rrf + rank + 1))
        for rank, (idx, _) in enumerate(bm25_results):
            combined[idx] += (1 - alpha) * (1 / (k_rrf + rank + 1))
        return dict(combined)

    # ----------------------------- Híbrido ----------------------------- #
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7, fusion_method: str = "linear") -> List[Document]:

        print(f"🔍 Consultando: '{query}'")
        print(f"   Parámetros: k={k}, alpha={alpha}, fusion={fusion_method}")

        semantic_results = self.semantic_search(query, k=k*2)
        bm25_results = self.bm25_search(query, k=k*2)

        if fusion_method == "rrf":
            combined = self._rrf_fusion(semantic_results, bm25_results, alpha)
        else:
            combined = self._linear_fusion(
                semantic_results, bm25_results, alpha)

        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        final_docs: List[Document] = []
        for idx, final_score in top:
            if idx < len(self.documents):
                base = self.documents[idx]
                sem_score = next(
                    (s for i, s in semantic_results if i == idx), 0.0)
                bm25_score = next(
                    (s for i, s in bm25_results if i == idx), 0.0)
                final_docs.append(
                    Document(
                        page_content=base['content'],
                        metadata={
                            **base,
                            'hybrid_score': final_score,
                            'semantic_score': sem_score,
                            'bm25_score': bm25_score,
                            'fusion_method': fusion_method,
                            'alpha': alpha,
                            'bm25_impl': 'langchain'
                        }
                    )
                )
        return final_docs

    # (Opcional) Diagnóstico limitado
    def diagnostic_search(self, query: str):
        print(f"\n=== Diagnóstico: {query} ===")
        sem = self.semantic_search(query, k=5)
        bm = self.bm25_search(query, k=5)
        print("Semántico (idx, score):", sem[:3])
        print("BM25 (idx, score):", bm[:3])


# ----------------------------- Cadena RAG (opcional) ----------------------------- #
_instance = HybridRAGResearchLCBM25()


def lc_bm25_retriever(query: str) -> List[Document]:
    return _instance.hybrid_search(query, k=5, alpha=0.5, fusion_method="linear")


template = """
Eres un experto en salud materna y embarazo. Analiza el siguiente contexto médico y responde la pregunta de manera precisa y detallada.

INSTRUCCIONES:
- Usa ÚNICAMENTE la información proporcionada en el contexto
- Si encuentras información relevante, proporciona una respuesta detallada con recomendaciones específicas
- Si no hay información suficiente, di claramente qué información falta
- Enfócate en la evidencia médica y recomendaciones cuando estén disponibles

CONTEXTO MÉDICO:
{context}

PREGUNTA: {question}

RESPUESTA DETALLADA:
"""

_prompt = ChatPromptTemplate.from_template(template)


def _format_docs(docs: List[Document]):
    formatted_docs = []
    for i, doc in enumerate(docs):
        hybrid_score = doc.metadata.get('hybrid_score', 0.0)
        semantic_score = doc.metadata.get('semantic_score', 0.0)
        bm25_score = doc.metadata.get('bm25_score', 0.0)
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page_number', 'N/A')

        formatted_doc = f"""--- Documento {i+1} ---
Puntuación Híbrida: {hybrid_score:.4f} (Semántica: {semantic_score:.4f}, BM25: {bm25_score:.4f})
Fuente: {source}, Página: {page}
Contenido: {doc.page_content}"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


def _context_fn(q: str) -> str:
    return _format_docs(lc_bm25_retriever(q))


rag_chain_lc_bm25 = (
    # type: ignore
    {"context": RunnableLambda(_context_fn), "question": RunnablePassthrough()}
    | _prompt
    | _instance.llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print("=== RAG Híbrido (LangChain BM25) ===")
    print("Opciones:")
    print("- 'diagnostic [consulta]': Diagnóstico detallado")
    print("- 'semantic [consulta]': Solo búsqueda semántica")
    print("- 'salir': Terminar")

    while (query := input("\nPregunta: ")) != "salir":
        print("\n" + "="*50)

        if query.startswith("diagnostic "):
            actual_query = query[11:]
            _instance.diagnostic_search(actual_query)
        elif query.startswith("semantic "):
            actual_query = query[9:]
            results = _instance.semantic_search(actual_query, k=3)
            print(f"Resultados semánticos para: '{actual_query}'")
            for i, (idx, score) in enumerate(results):
                if idx < len(_instance.documents):
                    doc = _instance.documents[idx]
                    print(
                        f"{i+1}. Score: {score:.4f} - {doc['content'][:150]}...")
        else:
            # 1. Recuperar documentos
            retrieved_docs = lc_bm25_retriever(query)

            # 2. Mostrar los documentos y sus scores
            print("DOCUMENTOS RECUPERADOS:")
            formatted_context = _format_docs(retrieved_docs)
            print(formatted_context)
            print("\n" + "-"*20 + "\n")

            # 3. Generar respuesta usando los documentos recuperados
            chain_with_context = (
                _prompt
                | _instance.llm
                | StrOutputParser()
            )
            answer = chain_with_context.invoke({
                "context": formatted_context,
                "question": query
            })

            print("RESPUESTA:")
            print(answer)

        print("="*50)


def query_for_evaluation(question: str) -> dict:
    """
    Función específica para evaluación con RAGAS del RAG Híbrido.
    Retorna estructura completa: pregunta, respuesta, contextos y metadatos.
    
    Args:
        question (str): La pregunta a procesar
        
    Returns:
        dict: Estructura con question, answer, contexts, source_documents y metadata
    """
    print(f"🔍 Evaluando (Hybrid BM25+Semántico): {question}")
    
    # 1. Obtener documentos usando búsqueda híbrida
    # Usar parámetros balanceados para evaluación
    retrieved_docs = _instance.hybrid_search(
        question, 
        k=5,                    # Top 5 documentos
        alpha=0.5,              # 70% peso semántico, 30% BM25 
        fusion_method="linear"   # Fusión lineal (más estable que RRF)
    )
    
    # 2. Formatear contextos para la respuesta
    formatted_context = _format_docs(retrieved_docs)
    
    # 3. Generar respuesta usando el LLM con el prompt médico
    response = _instance.llm.invoke(_prompt.format_messages(
        context=formatted_context, 
        question=question
    ))
    
    # 4. Preparar lista de contextos (solo contenido textual para RAGAS)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    # 5. Extraer información de scores para metadatos
    hybrid_scores = [doc.metadata.get('hybrid_score', 0.0) for doc in retrieved_docs]
    semantic_scores = [doc.metadata.get('semantic_score', 0.0) for doc in retrieved_docs]
    bm25_scores = [doc.metadata.get('bm25_score', 0.0) for doc in retrieved_docs]
    
    # 6. Retornar estructura completa para RAGAS
    return {
        "question": question,
        "answer": response.content,
        "contexts": contexts,  # Lista de strings con el contenido de los documentos
        "source_documents": retrieved_docs,  # Documentos completos con metadata híbrido
        "metadata": {
            "num_contexts": len(contexts),
            "retrieval_method": "hybrid_bm25_semantic",
            "fusion_method": "linear",
            "alpha": 0.7,
            "llm_model": "gpt-4o",
            "semantic_model": "text-embedding-3-small",
            "bm25_impl": "langchain",
            "avg_hybrid_score": sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0.0,
            "avg_semantic_score": sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0,
            "avg_bm25_score": sum(bm25_scores) / len(bm25_scores) if bm25_scores else 0.0,
            "score_distribution": {
                "hybrid": hybrid_scores,
                "semantic": semantic_scores, 
                "bm25": bm25_scores
            }
        }
    }
