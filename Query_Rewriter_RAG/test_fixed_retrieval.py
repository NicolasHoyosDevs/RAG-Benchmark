#!/usr/bin/env python3
"""
Test rápido para verificar que el retrieval funciona después de la corrección
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar configuración
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Importar después de cargar dotenv
from main_rewriter import contextual_retriever, base_retriever

def test_retrieval():
    """Prueba básica del retrieval corregido"""
    print("🧪 Probando retrieval corregido...")
    print(f"📊 Configuración actual:")
    print(f"   - k: {base_retriever.k}")
    print(f"   - score_threshold: {base_retriever.score_threshold}")
    
    # Query de prueba
    test_query = "¿Qué cuidados necesita una embarazada?"
    print(f"\n🔍 Query de prueba: '{test_query}'")
    
    try:
        # Probar retrieval
        results = contextual_retriever(test_query, max_final_docs=5)
        
        print(f"\n✅ Retrieval exitoso!")
        print(f"📄 Documentos retornados: {len(results)}")
        
        if results:
            print(f"\n📋 Muestra de documentos:")
            for i, doc in enumerate(results[:2]):
                print(f"   Doc {i+1}: {doc.page_content[:150]}...")
        else:
            print("❌ No se retornaron documentos")
            
    except Exception as e:
        print(f"❌ Error en retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()