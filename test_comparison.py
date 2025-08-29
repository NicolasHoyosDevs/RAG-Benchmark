#!/usr/bin/env python3
"""
Script de prueba para generar archivo de comparación consolidado
"""

import sys
from pathlib import Path

# Agregar el directorio results al path
sys.path.append(str(Path(__file__).parent / "results"))

from ragas_evaluator import save_comparison_results

def test_comparison():
    """Probar la función de comparación con datos simulados"""
    
    # Simular resultados vacíos (solo para probar la función)
    mock_results = {
        "simple": None,
        "hybrid": None, 
        "hyde": None,
        "rewriter": None
    }
    
    print("🧪 Probando función de comparación...")
    
    try:
        # Llamar a la función de comparación
        save_comparison_results(mock_results)
        print("✅ Función ejecutada correctamente")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comparison()
