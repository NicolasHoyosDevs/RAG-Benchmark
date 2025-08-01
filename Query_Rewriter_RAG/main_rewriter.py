import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import RePhraseQueryRetriever

# Cargar variables de entorno
load_dotenv(dotenv_path='../.env')

# --- Configuración ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "FinancialDocs"

# --- Conexión a Weaviate Cloud ---
auth_credentials = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth_credentials
)

# --- Configuración del Retriever Base (solo vectorial) ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=OPENAI_API_KEY)

vectorstore = Weaviate(client, COLLECTION_NAME, "content",
                       attributes=["title", "source"])
base_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={'k': 5})

# --- Configuración del Query Rewriter ---
REPHRASE_TEMPLATE = """Dada la siguiente conversación y una pregunta de seguimiento, 
reescribe la pregunta de seguimiento para que sea una pregunta independiente.

Historial de chat:
{chat_history}
Pregunta de seguimiento: {question}
Pregunta independiente:"""

REPHRASE_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
llm_rewriter = ChatOpenAI(model_name="gpt-3.5-turbo",
                          temperature=0, api_key=OPENAI_API_KEY)

# Creamos el retriever que reescribe la consulta
# Este retriever internamente llamará al llm_rewriter y luego usará la nueva query en el base_retriever
rewriter_retriever = RePhraseQueryRetriever.from_llm(
    retriever=base_retriever, llm=llm_rewriter, prompt=REPHRASE_PROMPT
)

# --- Creación del RAG Chain (usando LCEL) ---
qa_template = """
Responde la pregunta basándote únicamente en el siguiente contexto.
Si la respuesta no está en el contexto, di que no tienes la información.

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)


def format_docs(docs):
    return "\n\n".join(f"--- Fuente: {doc.metadata.get('source', 'N/A')} ---\n{doc.page_content}" for doc in docs)


rag_chain = (
    {"context": rewriter_retriever | format_docs, "question": RunnablePassthrough()}
    | qa_prompt
    | qa_llm
    | StrOutputParser()
)

# --- Ejecución ---
if __name__ == "__main__":
    try:
        print("--- Sistema RAG con Query Rewriter ---")

        chat_history = []

        while (query := input("\nIntroduce tu pregunta (o 'salir' para terminar): ")) != "salir":
            # Simular reescritura para depuración
            llm_chain_rewriter = LLMChain(
                llm=llm_rewriter, prompt=REPHRASE_PROMPT)
            rewritten_query = llm_chain_rewriter.invoke(
                {"chat_history": chat_history, "question": query})

            print(f"\nQuery Original: '{query}'")
            print(f"Query Reescrita: '{rewritten_query['text']}'")

            # Generar respuesta
            print("\nGenerando respuesta...")
            # Pasamos el historial de chat para que el rewriter funcione correctamente
            result = rag_chain.invoke(
                {"chat_history": chat_history, "question": query})

            print("\nRespuesta del RAG con Rewriter:")
            print(result)

            # Actualizar historial
            chat_history.append(f"Humano: {query}")
            chat_history.append(f"AI: {result}")

    finally:
        # Cerrar la conexión a Weaviate
        if client.is_connected():
            client.close()
            print("Conexión a Weaviate cerrada.")
