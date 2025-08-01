import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List

# Cargar variables de entorno
load_dotenv(dotenv_path='../.env')

# --- Configuración ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "FinancialDocs"
# 0.0 = BM25, 1.0 = Vectorial. 0.7 da más peso a la búsqueda vectorial.
ALPHA_HYBRID = 0.7

# --- Conexión a Weaviate Cloud ---
auth_credentials = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth_credentials
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# Función para realizar búsqueda híbrida usando la API nativa de Weaviate


def hybrid_search(query: str, k: int = 5, alpha: float = 0.7) -> List[Document]:
    """
    Realiza una búsqueda híbrida usando la API nativa de Weaviate

    Args:
        query: La consulta de búsqueda
        k: Número de documentos a devolver
        alpha: Peso híbrido (0.0 = solo BM25, 1.0 = solo vectorial)

    Returns:
        Lista de documentos de LangChain
    """
    try:
        # Obtener la colección
        collection = client.collections.get(COLLECTION_NAME)

        # Generar embedding para la consulta usando OpenAI
        query_vector = embeddings.embed_query(query)

        # Realizar búsqueda híbrida con vector pre-calculado (SIN filtro)
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,  # Usar el vector que generamos
            limit=k,
            alpha=alpha
        )

        print(
            f"Búsqueda híbrida exitosa: {len(response.objects)} documentos encontrados")

        # Convertir resultados a documentos de LangChain
        documents = []
        for i, obj in enumerate(response.objects):
            # Crear el documento con el contenido y metadatos
            # Usar el índice como score alternativo si no hay score real
            # Score decreciente basado en posición
            fallback_score = (k - i) / k

            doc = Document(
                page_content=obj.properties.get('content', ''),
                metadata={
                    'source': obj.properties.get('source', 'N/A'),
                    'title': obj.properties.get('title', 'N/A'),
                    '_additional': {
                        'score': obj.metadata.score if obj.metadata and hasattr(obj.metadata, 'score') and obj.metadata.score is not None else fallback_score,
                        'position': i + 1  # Posición en los resultados
                    }
                }
            )
            documents.append(doc)

        return documents

    except Exception as e:
        print(f"Error en búsqueda híbrida: {e}")
        print("Intentando búsqueda híbrida sin vector externo...")

        # Fallback 1: Híbrido sin vector externo (dejando que Weaviate genere el vector)
        try:
            collection = client.collections.get(COLLECTION_NAME)
            response = collection.query.hybrid(
                query=query,
                limit=k,
                alpha=alpha
            )

            documents = []
            for i, obj in enumerate(response.objects):
                fallback_score = (k - i) / k
                doc = Document(
                    page_content=obj.properties.get('content', ''),
                    metadata={
                        'source': obj.properties.get('source', 'N/A'),
                        'title': obj.properties.get('title', 'N/A'),
                        '_additional': {
                            'score': obj.metadata.score if obj.metadata and hasattr(obj.metadata, 'score') and obj.metadata.score is not None else fallback_score
                        }
                    }
                )
                documents.append(doc)
            print("Búsqueda híbrida sin vector externo exitosa")
            return documents

        except Exception as e1:
            print(f"Error en búsqueda híbrida sin vector: {e1}")

            # Fallback 2: Solo vectorial con nuestro vector
            try:
                query_vector = embeddings.embed_query(query)
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=k
                )

                documents = []
                for obj in response.objects:
                    doc = Document(
                        page_content=obj.properties.get('content', ''),
                        metadata={
                            'source': obj.properties.get('source', 'N/A'),
                            'title': obj.properties.get('title', 'N/A'),
                            '_additional': {
                                'distance': obj.metadata.distance if obj.metadata and hasattr(obj.metadata, 'distance') else 0.0
                            }
                        }
                    )
                    documents.append(doc)
                print("Usando búsqueda vectorial pura como fallback")
                return documents

            except Exception as e2:
                print(f"Error en búsqueda vectorial: {e2}")

                # Fallback 3: Solo BM25 como último recurso
                try:
                    response = collection.query.bm25(
                        query=query,
                        limit=k
                    )

                    documents = []
                    for obj in response.objects:
                        doc = Document(
                            page_content=obj.properties.get('content', ''),
                            metadata={
                                'source': obj.properties.get('source', 'N/A'),
                                'title': obj.properties.get('title', 'N/A'),
                                '_additional': {
                                    'score': obj.metadata.score if obj.metadata and hasattr(obj.metadata, 'score') else 0.0
                                }
                            }
                        )
                        documents.append(doc)
                    print("Usando búsqueda BM25 como último fallback")
                    return documents

                except Exception as e3:
                    print(f"Error en todos los métodos de búsqueda: {e3}")
                    return []

# Función que actúa como retriever para usar en la cadena


def retriever_function(query: str) -> List[Document]:
    return hybrid_search(query, k=5, alpha=ALPHA_HYBRID)


# --- Creación del RAG Chain (usando LCEL) ---
template = """
Eres un analista financiero experto. Analiza el siguiente contexto financiero y responde la pregunta de manera precisa y detallada.

INSTRUCCIONES:
- Usa ÚNICAMENTE la información proporcionada en el contexto
- Si encuentras información relevante, proporciona una respuesta detallada con cifras específicas
- Si no hay información suficiente, di claramente qué información falta
- Enfócate en los datos numéricos y tendencias cuando estén disponibles

CONTEXTO FINANCIERO:
{context}

PREGUNTA: {question}

RESPUESTA DETALLADA:
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)


def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Weaviate hybrid search devuelve metadatos adicionales como 'score'
        score_info = doc.metadata.get('_additional', {})
        score = score_info.get('score', 0.0)
        position = score_info.get('position', i + 1)

        # Manejar el caso donde score puede ser None
        if score is None:
            score = 0.0
        source = doc.metadata.get('source', 'N/A')

        formatted_doc = f"--- Documento {position} (Relevancia: {score:.4f}, Fuente: {source}) ---\n{doc.page_content}"
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)


# Función que combina la búsqueda y el formateo
def get_context(query: str) -> str:
    """Obtiene el contexto formateado para una consulta"""
    docs = retriever_function(query)
    return format_docs(docs)


rag_chain = (
    {"context": RunnableLambda(get_context), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Ejecución ---
if __name__ == "__main__":
    try:
        print("--- Sistema RAG Híbrido (Vector + BM25) ---")
        print(f"Parámetro Alpha (peso híbrido): {ALPHA_HYBRID}")

        while (query := input("\nIntroduce tu pregunta (o 'salir' para terminar): ")) != "salir":
            # Recuperar documentos para depuración
            print("\nRecuperando documentos...")
            retrieved_docs = retriever_function(query)

            print(f"\nDocumentos recuperados ({len(retrieved_docs)}):")
            for i, doc in enumerate(retrieved_docs):
                score_info = doc.metadata.get('_additional', {})
                score = score_info.get('score', 0.0)
                # Manejar el caso donde score puede ser None
                if score is None:
                    score = 0.0
                print(
                    f"  {i+1}. Score: {score:.4f} | Fuente: {doc.metadata.get('source')}")
                print(f"     Contenido: {doc.page_content[:150]}...")

            # Generar respuesta
            print("\nGenerando respuesta...")
            answer = rag_chain.invoke(query)
            print("\nRespuesta del RAG Híbrido:")
            print(answer)

    finally:
        # Cerrar la conexión a Weaviate
        if client.is_connected():
            client.close()
            print("Conexión a Weaviate cerrada.")
