"""
Script de prueba para la función query_for_evaluation
"""

import sys
from pathlib import Path

# Agregar el directorio del proyecto al path para importar
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importar la función de evaluación
from Query_Rewriter_RAG.main_rewriter import query_for_evaluation

def test_evaluation_function():
    """Prueba la función de evaluación con una pregunta de ejemplo"""
    
    print("🧪 === PRUEBA DE FUNCIÓN DE EVALUACIÓN ===")
    print("Probando query_for_evaluation con una pregunta de embarazo...")
    
    # Pregunta de prueba
    test_question = "¿Qué es la preeclampsia y cuáles son sus síntomas?"
    
    try:
        # Llamar a la función de evaluación
        result = query_for_evaluation(test_question)
        
        # Verificar estructura del resultado
        print("\n✅ Función ejecutada exitosamente!")
        print(f"📋 Estructura del resultado:")
        print(f"   - question: {type(result.get('question'))} ({'✓' if result.get('question') else '✗'})")
        print(f"   - answer: {type(result.get('answer'))} ({'✓' if result.get('answer') else '✗'})")
        print(f"   - contexts: {type(result.get('contexts'))} con {len(result.get('contexts', []))} elementos")
        print(f"   - source_documents: {type(result.get('source_documents'))} con {len(result.get('source_documents', []))} elementos")
        print(f"   - metadata: {type(result.get('metadata'))} ({'✓' if result.get('metadata') else '✗'})")
        
        # Mostrar contenido de ejemplo
        print(f"\n📄 Contenido de ejemplo:")
        print(f"   Pregunta: {result['question']}")
        print(f"   Respuesta (primeros 200 chars): {result['answer'][:200]}...")
        print(f"   Número de contextos: {len(result['contexts'])}")
        print(f"   Metadatos: {result['metadata']}")
        
        # Verificar que contexts es una lista de strings
        if result.get('contexts'):
            print(f"   Primer contexto (primeros 100 chars): {result['contexts'][0][:100]}...")
        
        print("\n🎉 ¡Función lista para usar con RAGAS!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error al ejecutar la función: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation_function()
    if success:
        print("\n✅ Prueba completada exitosamente")
    else:
        print("\n❌ Prueba falló")