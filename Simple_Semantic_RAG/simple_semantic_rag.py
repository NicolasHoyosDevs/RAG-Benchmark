"""
Simple Semantic RAG using only semantic search with ChromaDB.

"""

import os
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")

# Definir rutas y configuraciones
script_dir = Path(__file__).resolve().parent
chroma_db_dir = script_dir.parent / "Data" / "embeddings" / "chroma_db"
collection_name = "guia_embarazo_parto"

# Configurar modelos
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Cargar ChromaDB
vectorstore = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
)

# Configurar retriever semántico
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

print("✅ Sistema RAG semántico simple inicializado.")
print(f"   📄 Documentos en base de datos: {vectorstore._collection.count()}")

# Función simple de búsqueda


def search(query: str) -> List[Document]:
    """Realiza una búsqueda semántica usando ChromaDB."""
    print(f"Consultando: '{query}'")
    return retriever.invoke(query)


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

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs: List[Document]) -> str:
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


if __name__ == "__main__":
    print("\n=== RAG Semántico Simple ===")
    print("Escribe tu pregunta o 'salir' para terminar.")

    query = input("\nPregunta: ")

    if query.lower() != "salir":
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
    else:
        print("Sistema finalizado.")
