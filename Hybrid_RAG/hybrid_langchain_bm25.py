"""
Hybrid RAG simplificado usando EnsembleRetriever de LangChain.
"""

import os
import json
import time
from pathlib import Path

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

# Definir rutas y configuraciones
script_dir = Path(__file__).resolve().parent
chroma_db_dir = script_dir.parent / "Data" / "embeddings" / "chroma_db"
collection_name = "guia_embarazo_parto"
chunks_file = script_dir.parent / "Data" / "chunks" / "chunks_final.json"

# Cargar documentos


def load_documents():
    """Carga los chunks desde el JSON y los convierte a Documentos de LangChain."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    return [
        Document(page_content=d['content'], metadata=d)
        for d in chunks_data
    ]


documents = load_documents()
print(f"   Documentos cargados: {len(documents)}")

# Configurar modelos y retrievers
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 1. Retriever lexical (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 2. Retriever semántico (Chroma)
vectorstore = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.2, 0.8]  # Ponderación equitativa
)

# Función de búsqueda


def search(query):
    """Realiza una búsqueda híbrida usando el EnsembleRetriever."""
    print(f"Consultando: '{query}'")
    return ensemble_retriever.invoke(query)


# Plantilla para el prompt
template = """
Eres un experto en salud materna y embarazo. Analiza el siguiente contexto médico y responde la pregunta de manera precisa y detallada.

INSTRUCCIONES:
- Usa ÚNICAMENTE la información proporcionada en el contexto.
- Si la información es suficiente, proporciona una respuesta detallada.
- Si no hay información suficiente, indícalo claramente.
- Recuerda que eres un especialista médico respondiendo consultas sobre embarazo y parto.

CONTEXTO MÉDICO:
{context}

PREGUNTA: {question}

RESPUESTA MÉDICA DETALLADA:
"""

prompt = ChatPromptTemplate.from_template(template)

# Formateo de documentos


def format_docs(docs):
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
    {"context": RunnableLambda(search) | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ejecución principal
if __name__ == "__main__":
    print("Escribe tu pregunta o 'salir' para terminar.")

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
