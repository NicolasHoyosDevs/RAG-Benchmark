"""
HyDE RAG - Hypothetical Document Embeddings for medical Q&A.

HyDE generates a hypothetical document based on the query, then uses that
document to perform semantic search instead of the original query.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

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
# Más creativo para HyDE
llm_hyde = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
llm_answer = ChatOpenAI(model_name="gpt-4o", temperature=0)

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

print("Sistema RAG HyDE inicializado.")
print(f"   Documentos en base de datos: {vectorstore._collection.count()}")

# Prompt para generar documentos hipotéticos
hyde_prompt = ChatPromptTemplate.from_template("""
Eres un experto médico escribiendo una sección detallada de una guía médica sobre embarazo y parto.

Basándote en esta pregunta: {question}

Escribe un documento médico detallado y completo que respondería perfectamente a esta pregunta.
El documento debe incluir:
- Información médica precisa sobre el tema
- Detalles clínicos relevantes
- Recomendaciones médicas apropiadas
- Consideraciones importantes para la salud materna
- Información práctica y consejos

Escribe el documento como si fuera parte de una guía médica oficial sobre embarazo y parto.
Sé específico, detallado y usa terminología médica apropiada.

DOCUMENTO HIPOTÉTICO:
""")


def generate_hypothetical_document(query: str) -> Dict[str, Any]:
    """Genera un documento hipotético basado en la pregunta y retorna métricas."""
    with get_openai_callback() as cb:
        response = (hyde_prompt | llm_hyde | StrOutputParser()
                    ).invoke({"question": query})

    hypothetical_doc = response.strip()

    return {
        'document': hypothetical_doc,
        'input_tokens': cb.prompt_tokens,
        'output_tokens': cb.completion_tokens,
        'cost': cb.total_cost
    }


def search_with_hyde(query: str) -> tuple:
    """Realiza búsqueda usando HyDE: genera documento hipotético y lo usa para buscar."""
    print(f"Pregunta original: '{query}'")

    # 1. Generar documento hipotético con métricas
    print("Generando documento hipotético...")
    hyde_result = generate_hypothetical_document(query)

    hypothetical_doc = hyde_result['document']
    print(
        f"Documento hipotético generado ({len(hypothetical_doc)} caracteres)")
    print(f"   - Tokens prompt: {hyde_result['input_tokens']}")
    print(f"   - Tokens respuesta: {hyde_result['output_tokens']}")
    print(f"   - Costo generación: ${hyde_result['cost']:.6f}")

    # Mostrar preview del documento hipotético
    print(f"\n--- DOCUMENTO HIPOTÉTICO (primeros 300 caracteres) ---")
    print(hypothetical_doc[:300] +
          "..." if len(hypothetical_doc) > 300 else hypothetical_doc)
    print("--- FIN DOCUMENTO HIPOTÉTICO ---\n")

    # 2. Usar el documento hipotético para búsqueda semántica
    print("Buscando documentos similares al documento hipotético...")
    retrieved_docs = retriever.invoke(hypothetical_doc)

    print(f"Documentos encontrados: {len(retrieved_docs)}")

    # Retornar documentos y métricas de HyDE
    return retrieved_docs, hyde_result


# ----------------------------- Cadena RAG ----------------------------- #

# Template para respuesta final
qa_template = """
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

qa_prompt = ChatPromptTemplate.from_template(qa_template)


def format_docs(docs_and_hyde) -> str:
    """Formatea los documentos para ser incluidos en el prompt."""
    docs, hyde_result = docs_and_hyde

    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page_number', 'N/A')

        formatted_doc = f"""--- Documento {i+1} ---
Fuente: {source}, Página: {page}
Contenido: {doc.page_content}"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


# Función principal que maneja todo el proceso HyDE
def process_hyde_query(query: str):
    """Procesa una consulta usando HyDE y retorna respuesta con métricas completas."""
    print(f"Pregunta original: '{query}'")

    # 1. Generar documento hipotético con métricas
    print("Generando documento hipotético...")
    hyde_result = generate_hypothetical_document(query)

    hypothetical_doc = hyde_result['document']
    print(
        f"Documento hipotético generado ({len(hypothetical_doc)} caracteres)")
    print(f"   - Tokens prompt: {hyde_result['input_tokens']}")
    print(f"   - Tokens respuesta: {hyde_result['output_tokens']}")
    print(f"   - Costo generación: ${hyde_result['cost']:.6f}")

    # Mostrar preview del documento hipotético
    print(f"\n--- DOCUMENTO HIPOTÉTICO (primeros 300 caracteres) ---")
    print(hypothetical_doc[:300] +
          "..." if len(hypothetical_doc) > 300 else hypothetical_doc)
    print("--- FIN DOCUMENTO HIPOTÉTICO ---\n")

    # 2. Buscar documentos similares
    print("Buscando documentos similares al documento hipotético...")
    retrieved_docs = retriever.invoke(hypothetical_doc)
    print(f"Documentos encontrados: {len(retrieved_docs)}")

    # 3. Formatear contexto
    formatted_context = format_docs((retrieved_docs, hyde_result))

    # 4. Generar respuesta final con métricas
    with get_openai_callback() as cb_answer:
        response = llm_answer.invoke(qa_prompt.format_messages(
            context=formatted_context,
            question=query
        ))

    # Retornar respuesta y todas las métricas
    return {
        'answer': response.content,
        # Add contexts for RAGAS
        'contexts': [doc.page_content for doc in retrieved_docs],
        'hyde_metrics': hyde_result,
        'answer_metrics': {
            'input_tokens': cb_answer.prompt_tokens,
            'output_tokens': cb_answer.completion_tokens,
            'cost': cb_answer.total_cost
        },
        'total_cost': hyde_result['cost'] + cb_answer.total_cost,
        'total_input_tokens': hyde_result['input_tokens'] + cb_answer.prompt_tokens,
        'total_output_tokens': hyde_result['output_tokens'] + cb_answer.completion_tokens
    }


if __name__ == "__main__":
    print("\n=== RAG con HyDE (Hypothetical Document Embeddings) ===")
    print("Este sistema genera un documento hipotético basado en tu pregunta")
    print("y luego busca documentos similares a ese documento hipotético.")
    print("\nEscribe tu pregunta o 'salir' para terminar.")

    query = input("\nPregunta: ")

    if query.lower() != "salir":
        start_time = time.time()
        result = process_hyde_query(query)
        end_time = time.time()

        print("\n" + "="*50)
        print("RESPUESTA:")
        print(result['answer'])
        print("\n" + "="*50)

        # Mostrar métricas detalladas
        print("\n MÉTRICAS DETALLADAS:")
        print(f"     Tiempo total: {end_time - start_time:.2f} segundos")

        print(f"\n GENERACIÓN DOCUMENTO HIPOTÉTICO:")
        print(
            f"   - Tokens entrada (prompt): {result['hyde_metrics']['input_tokens']}")
        print(
            f"   - Tokens salida (documento): {result['hyde_metrics']['output_tokens']}")
        print(f"   - Costo: ${result['hyde_metrics']['cost']:.6f}")

        print(f"\n GENERACIÓN RESPUESTA FINAL:")
        print(
            f"   - Tokens entrada (prompt): {result['answer_metrics']['input_tokens']}")
        print(
            f"   - Tokens salida (respuesta): {result['answer_metrics']['output_tokens']}")
        print(f"   - Costo: ${result['answer_metrics']['cost']:.6f}")

        print(f"\n TOTALES:")
        print(f"   - Tokens entrada total: {result['total_input_tokens']}")
        print(f"   - Tokens salida total: {result['total_output_tokens']}")
        print(f"   - Costo total (USD): ${result['total_cost']:.6f}")

    else:
        print("Sistema finalizado.")
