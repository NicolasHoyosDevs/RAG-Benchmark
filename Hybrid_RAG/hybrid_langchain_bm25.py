"""
Hybrid RAG simplificado usando EnsembleRetriever de LangChain.

"""

import os
import json
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
load_dotenv(dotenv_path='../.env')

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")


class HybridRAG:
    """Versión híbrida simplificada que usa EnsembleRetriever de LangChain."""

    def __init__(self):
        # Rutas y nombres
        script_dir = Path(__file__).resolve().parent
        self.chroma_db_dir = script_dir.parent / "Data" / "embeddings" / "chroma_db"
        self.collection_name = "guia_embarazo_parto"
        self.chunks_file = script_dir.parent / "Data" / "chunks" / "chunks_final.json"

        # Modelos
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Cargar documentos base
        self.documents: List[Document] = self._load_documents()

        # 1. Retriever lexical (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 5

        # 2. Retriever semántico (Chroma)
        vectorstore = Chroma(
            persist_directory=str(self.chroma_db_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        self.semantic_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5})

        # 3. Ensamble de retrievers
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.semantic_retriever],
            weights=[0.5, 0.5]  # Ponderación equitativa
        )

        # print("✅ Sistema híbrido simplificado inicializado.")
        print(f"   📄 Documentos cargados: {len(self.documents)}")

    def _load_documents(self) -> List[Document]:
        """Carga los chunks desde el JSON y los convierte a Documentos de LangChain."""
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        return [
            Document(page_content=d['content'], metadata=d)
            for d in chunks_data
        ]

    def search(self, query: str) -> List[Document]:
        """Realiza una búsqueda híbrida usando el EnsembleRetriever."""
        print(f"Consultando: '{query}'")
        return self.ensemble_retriever.invoke(query)


# ----------------------------- Cadena RAG ----------------------------- #
_instance = HybridRAG()

template = """
Eres un experto en salud materna y embarazo. Analiza el siguiente contexto médico y responde la pregunta de manera precisa y detallada.

INSTRUCCIONES:
- Usa ÚNICAMENTE la información proporcionada en el contexto.
- Si la información es suficiente, proporciona una respuesta detallada.
- Si no hay información suficiente, indícalo claramente.

CONTEXTO MÉDICO:
{context}

PREGUNTA: {question}

RESPUESTA DETALLADA:
"""

_prompt = ChatPromptTemplate.from_template(template)


def _format_docs(docs: List[Document]) -> str:
    """Formatea los documentos para ser incluidos en el prompt."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page_number', 'N/A')

        formatted_doc = f"""--- Documento {i+1} ---
Fuente: {source}, Página: {page}
Contenido: {doc.page_content}"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


# Definición de la cadena RAG
rag_chain = (
    {"context": RunnableLambda(_instance.search) |
     _format_docs, "question": RunnablePassthrough()}
    | _prompt
    | _instance.llm
    | StrOutputParser()
)


if __name__ == "__main__":
    # print("\n=== RAG Híbrido Simplificado ===")
    print("Escribe tu pregunta o 'salir' para terminar.")

    # while (query := input("\nPregunta: ")) != "salir":
    #     print("\n" + "="*50)

    #     # Generar y mostrar la respuesta
    #     answer = rag_chain.invoke(query)

    #     print("RESPUESTA:")
    #     print(answer)

    #     print("="*50)

    query = input("\nPregunta: ")

    start_time = time.time()
    with get_openai_callback() as cb:
        answer = rag_chain.invoke(query)

    end_time = time.time()

    print("\n" + "="*50)
    print("RESPUESTA:")
    print(answer)
    print("\n" + "="*50)
    print("\nEstadísticas de la consulta:")
    print(f"   - Tiempo total: {end_time - start_time:.2f} segundos")
    print(f"   - Tokens de entrada (prompt): {cb.prompt_tokens}")
    print(f"   - Tokens de salida (respuesta): {cb.completion_tokens}")
    print(f"   - Costo total (USD): ${cb.total_cost:.6f}")
