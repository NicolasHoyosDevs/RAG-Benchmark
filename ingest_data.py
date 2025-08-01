import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey
from weaviate.classes.data import DataObject
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Cargar variables de entorno
load_dotenv()

# --- Configuración ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNKS_FILE_PATH = "Data/chunks/chunks_20250710_142931.json"
COLLECTION_NAME = "FinancialDocs"

# --- Conexión a Weaviate Cloud (v4) ---
auth_credentials = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth_credentials
)

# --- Definición del Esquema (Collection) ---


def create_weaviate_collection():
    if client.collections.exists(COLLECTION_NAME):
        print(
            f"La colección '{COLLECTION_NAME}' ya existe. Saltando creación.")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            wvc.Property(name="title", data_type=wvc.DataType.TEXT),
            wvc.Property(name="source", data_type=wvc.DataType.TEXT),
        ],
        vector_config=wvc.Configure.Vectorizer.none()
    )
    print(f"Colección '{COLLECTION_NAME}' creada exitosamente.")

# --- Carga y Procesamiento de Chunks ---


def load_chunks_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for chunk in data['chunks']:
        source_metadata = chunk.get("source_metadata", {})
        doc = {
            "content": chunk.get("content", ""),
            "title": f"{source_metadata.get('company', '')} {source_metadata.get('year', '')} {source_metadata.get('quarter', '')}".strip(),
            "source": source_metadata.get("filename", "N/A")
        }
        docs.append(doc)
    return docs

# --- Ingesta de Datos ---


def ingest_data():
    print("Iniciando la ingesta de datos...")

    try:
        # 1. Crear colección si no existe
        create_weaviate_collection()

        # 2. Cargar chunks desde el archivo JSON
        documents_to_ingest = load_chunks_from_file(CHUNKS_FILE_PATH)
        print(
            f"Se cargaron {len(documents_to_ingest)} chunks para la ingesta.")

        # 3. Inicializar embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=OPENAI_API_KEY)

        print(
            "Iniciando el proceso de embedding e ingesta en Weaviate. Esto puede tardar...")

        # 4. Obtener la colección
        collection = client.collections.get(COLLECTION_NAME)

        # 5. Procesar documentos en lotes
        batch_size = 100
        for i in tqdm(range(0, len(documents_to_ingest), batch_size), desc="Procesando lotes"):
            batch_docs = documents_to_ingest[i:i + batch_size]

            # Crear embeddings para el lote
            batch_texts = [doc['content'] for doc in batch_docs]
            batch_embeddings = embeddings.embed_documents(batch_texts)

            # Preparar objetos para Weaviate usando DataObject
            objects_to_insert = []
            for j, doc in enumerate(batch_docs):
                data_object = DataObject(
                    properties={
                        "content": doc['content'],
                        "title": doc['title'],
                        "source": doc['source']
                    },
                    vector=batch_embeddings[j]
                )
                objects_to_insert.append(data_object)

            # Insertar lote en Weaviate
            collection.data.insert_many(objects_to_insert)

        print("¡Ingesta de datos completada!")

    finally:
        # Cerrar la conexión a Weaviate
        if client.is_connected():
            client.close()
            print("Conexión a Weaviate cerrada.")


if __name__ == "__main__":
    ingest_data()
