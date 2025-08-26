import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
from langchain.retrievers import RePhraseQueryRetriever
from langchain_core.documents import Document

# Cargar configuración (como antes...)
load_dotenv(dotenv_path='../.env')
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "FinancialDocs"

# Conexión a Weaviate (como antes...)
auth_credentials = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth_credentials
)

# Configuración de componentes (como antes...)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# --- Retriever Personalizado para Weaviate v4 ---
class WeaviateRetriever:
    def __init__(self, client, collection_name: str, k: int = 5):
        self.client = client
        self.collection_name = collection_name
        self.k = k
        # Inicializar embeddings para búsqueda vectorial
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    
    def invoke(self, query: str) -> List[Document]:
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Búsqueda VECTORIAL en lugar de BM25
            query_vector = self.embeddings.embed_query(query)
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=self.k
            )
            
            documents = []
            for obj in response.objects:
                doc = Document(
                    page_content=obj.properties.get('content', ''),
                    metadata={
                        'source': obj.properties.get('source', 'N/A'),
                        'title': obj.properties.get('title', 'N/A'),
                        'distance': obj.metadata.distance if obj.metadata else 0.0
                    }
                )
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"Error en búsqueda vectorial: {e}")
            return []

# --- Configuración del Retriever Base ---
base_retriever = WeaviateRetriever(client, COLLECTION_NAME, k=5)

# ✅ NUEVA IMPLEMENTACIÓN

class ChatHistoryManager:
    """Maneja el historial de conversación de forma eficiente"""
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
    
    def add_exchange(self, human_msg: str, ai_msg: str):
        self.history.append({"human": human_msg, "ai": ai_msg})
        # Mantener solo los últimos N intercambios
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_formatted_history(self, max_turns: int = 5) -> str:
        if not self.history:
            return ""
        
        recent = self.history[-max_turns:]
        formatted = []
        for exchange in recent:
            formatted.append(f"Humano: {exchange['human']}")
            formatted.append(f"AI: {exchange['ai']}")
        return "\n".join(formatted)
    
    def clear(self):
        self.history = []

# Configuración del rewriter
REPHRASE_TEMPLATE = """Dada la siguiente conversación y una pregunta de seguimiento, 
reescribe la pregunta de seguimiento para que sea una pregunta independiente y específica.

Historial de chat:
{chat_history}

Pregunta de seguimiento: {question}

Instrucciones:
- Si la pregunta hace referencia a algo mencionado antes, inclúyelo explícitamente
- Mantén el contexto financiero si es relevante
- Sé específico y claro

Pregunta independiente:"""

REPHRASE_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
llm_rewriter = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

def contextual_retriever(inputs: dict) -> List:
    """Retriever que usa historial para reescribir consultas"""
    question = inputs["question"]
    chat_history = inputs.get("chat_history", "")
    
    # SIEMPRE reescribir (incluso sin historial)
    llm_chain_rewriter = LLMChain(llm=llm_rewriter, prompt=REPHRASE_PROMPT)
    rewritten_result = llm_chain_rewriter.invoke({
        "chat_history": chat_history if chat_history.strip() else "Sin historial previo.",
        "question": question
    })
    rewritten_query = rewritten_result["text"].strip()
    print(f"🔄 Query original: '{question}'")
    print(f"🔄 Query reescrita: '{rewritten_query}'")
    
    # Buscar documentos con VECTORIAL
    docs = base_retriever.invoke(rewritten_query)
    print(f"📄 Documentos encontrados: {len(docs)}")
    
    # Debug: Mostrar qué documentos se encontraron
    if docs:
        for i, doc in enumerate(docs[:2]):
            print(f"📄 Doc {i+1}: {doc.page_content[:100]}...")
    else:
        print("❌ No se encontraron documentos relevantes")
    
    return docs

def format_docs(docs):
    """Formatea documentos para el contexto"""
    return "\n\n".join(
        f"--- Fuente: {doc.metadata.get('source', 'N/A')} ---\n{doc.page_content}" 
        for doc in docs
    )

# Template para respuesta final
qa_template = """
Eres un analista financiero experto. Analiza el contexto proporcionado y responde la pregunta de la manera más útil posible.

INSTRUCCIONES DE RESPUESTA:
1. SIEMPRE intenta responder con la información disponible, aunque sea parcial
2. Si la información es limitada, di qué SÍ puedes concluir basándote en los datos
3. Extrae insights, patrones o datos relevantes del contexto
4. Si preguntan sobre una empresa en general, usa información financiera para describirla
5. Incluye números específicos, fechas, y detalles cuando estén disponibles
6. Si falta información específica, menciona qué datos relacionados SÍ tienes

FORMATO DE RESPUESTA:
- Respuesta directa con la información disponible
- Datos específicos y cifras cuando sea posible
- Si hay limitaciones, menciónalas AL FINAL, no al principio
- Ofrece información relacionada que pueda ser útil

Contexto financiero:
{context}

Pregunta: {question}

Respuesta analítica:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# ✅ PIPELINE FINAL OPTIMIZADO
def create_conversational_rag():
    return (
        {
            "context": RunnableLambda(contextual_retriever) | RunnableLambda(format_docs),
            "question": lambda x: x["question"]
        }
        | qa_prompt
        | qa_llm
        | StrOutputParser()
    )

# Ejecución principal
if __name__ == "__main__":
    try:
        print("🤖 --- Sistema RAG Conversacional con Query Rewriter ---")
        
        # Inicializar componentes
        history_manager = ChatHistoryManager(max_history=10)
        rag_chain = create_conversational_rag()
        
        while (query := input("\n💬 Tu pregunta (o 'salir'): ")) != "salir":
            # Preparar inputs
            formatted_history = history_manager.get_formatted_history(max_turns=5)
            
            inputs = {
                "question": query,
                "chat_history": formatted_history
            }
            
            print("\n🔍 Procesando...")
            
            # Generar respuesta
            response = rag_chain.invoke(inputs)
            
            print(f"\n🤖 Respuesta:")
            print(response)
            
            # Actualizar historial
            history_manager.add_exchange(query, response)
            
            # Mostrar historial actual (opcional, para debug)
            if formatted_history:
                print(f"\n📚 Historial actual: {len(history_manager.history)} intercambios")
    
    finally:
        if client.is_connected():
            client.close()
            print("\n🔌 Conexión cerrada.")