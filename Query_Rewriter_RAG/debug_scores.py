#!/usr/bin/env python3
"""
Script de debug para analizar los scores reales de ChromaDB
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar configuración
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuración de ChromaDB
SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIRECTORY = SCRIPT_DIR.parent / "Data" / "embeddings" / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"

def debug_chromadb_scores():
    """Analiza los scores reales que retorna ChromaDB"""
    print("🔍 Analizando scores de ChromaDB...")
    
    # Inicializar embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=OPENAI_API_KEY
    )
    
    # Cargar base de datos
    try:
        db = Chroma(
            persist_directory=str(DB_DIRECTORY),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print(f"✅ ChromaDB cargada: {db._collection.count()} documentos")
    except Exception as e:
        print(f"❌ Error al cargar ChromaDB: {e}")
        return
    
    # Queries de prueba
    test_queries = [
        "¿Qué cuidados necesita una embarazada?",
        "dolor de cabeza durante el embarazo",
        "control prenatal rutinario"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Búsqueda con scores
        try:
            results = db.similarity_search_with_score(query, k=10)
            print(f"📄 Documentos encontrados: {len(results)}")
            
            if results:
                print("📊 Scores originales (distancia):")
                for i, (doc, score) in enumerate(results[:5]):
                    similarity_v1 = 1.0 - score  # Conversión original
                    similarity_v2 = max(0, 1.0 - score)  # Con límite mínimo
                    print(f"  Doc {i+1}: distancia={score:.4f} | sim_v1={similarity_v1:.4f} | sim_v2={similarity_v2:.4f}")
                    print(f"          Content: {doc.page_content[:100]}...")
                
                # Estadísticas
                scores = [score for _, score in results]
                print(f"\n📈 Estadísticas de distancias:")
                print(f"   Min: {min(scores):.4f}")
                print(f"   Max: {max(scores):.4f}")
                print(f"   Promedio: {sum(scores)/len(scores):.4f}")
            else:
                print("❌ No se encontraron documentos")
                
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")

if __name__ == "__main__":
    debug_chromadb_scores()