"""
Script principal para ejecutar benchmarks de RAG
"""

import asyncio
import json
import yaml
from pathlib import Path
import sys

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.rags.graph_rag import GraphRAG
from src.rags.rewrite_rag import RewriteRAG
from src.rags.hybrid_rag import HybridRAG
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.metrics import RAGMetrics


async def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Carga la configuración desde archivo YAML"""
    config_file = Path(__file__).parent / config_path
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_test_dataset(dataset_path: str) -> list:
    """Carga el dataset de prueba"""
    # Dataset de ejemplo - en un caso real, cargarías desde archivo
    return [
        {
            "question": "¿Qué es la inteligencia artificial?",
            "answer": "La inteligencia artificial es una rama de la ciencia de la computación que se encarga de crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
            "context": "La IA incluye machine learning, deep learning, procesamiento de lenguaje natural y visión por computadora."
        },
        {
            "question": "¿Cuáles son los beneficios del machine learning?",
            "answer": "El machine learning permite automatizar decisiones, encontrar patrones en datos complejos y mejorar el rendimiento con la experiencia.",
            "context": "ML se usa en recomendaciones, detección de fraude, diagnósticos médicos y vehículos autónomos."
        },
        {
            "question": "¿Qué es un transformer en deep learning?",
            "answer": "Un transformer es una arquitectura de red neuronal que usa mecanismos de atención para procesar secuencias de datos de manera paralela.",
            "context": "Los transformers son la base de modelos como GPT, BERT y T5, y han revolucionado el procesamiento de lenguaje natural."
        }
    ]


async def setup_rag_systems(config: dict) -> list:
    """Configura e inicializa todos los sistemas RAG"""
    systems = []
    
    # Graph RAG
    graph_rag = GraphRAG(config)
    await graph_rag.initialize()
    systems.append(graph_rag)
    
    # Rewrite RAG
    rewrite_rag = RewriteRAG(config)
    await rewrite_rag.initialize()
    systems.append(rewrite_rag)
    
    # Hybrid RAG
    hybrid_rag = HybridRAG(config)
    await hybrid_rag.initialize()
    systems.append(hybrid_rag)
    
    return systems


async def add_sample_documents(rag_systems: list):
    """Añade documentos de muestra a todos los sistemas RAG"""
    sample_docs = [
        """La inteligencia artificial (IA) es una rama de la ciencia de la computación 
        que se encarga de crear sistemas capaces de realizar tareas que normalmente 
        requieren inteligencia humana, como el reconocimiento de patrones, la toma 
        de decisiones y el aprendizaje.""",
        
        """El machine learning es una subdisciplina de la inteligencia artificial 
        que permite a las computadoras aprender y mejorar automáticamente a través 
        de la experiencia sin ser programadas explícitamente para cada tarea específica.""",
        
        """Los transformers son una arquitectura de red neuronal introducida en 2017 
        que utiliza mecanismos de atención para procesar secuencias de datos. Han 
        revolucionado el campo del procesamiento de lenguaje natural.""",
        
        """El procesamiento de lenguaje natural (NLP) es una rama de la IA que se 
        centra en la interacción entre computadoras y lenguaje humano, incluyendo 
        tareas como traducción, análisis de sentimientos y generación de texto."""
    ]
    
    for rag_system in rag_systems:
        await rag_system.add_documents(sample_docs)
        print(f"Documentos añadidos a {rag_system.name}")


async def main():
    """Función principal del benchmark"""
    print("🚀 Iniciando RAG Benchmark")
    print("=" * 50)
    
    # Cargar configuración
    config = await load_config()
    print("✅ Configuración cargada")
    
    # Configurar sistemas RAG
    print("🔧 Configurando sistemas RAG...")
    rag_systems = await setup_rag_systems(config)
    print(f"✅ {len(rag_systems)} sistemas RAG configurados")
    
    # Añadir documentos de muestra
    print("📚 Añadiendo documentos de muestra...")
    await add_sample_documents(rag_systems)
    print("✅ Documentos añadidos")
    
    # Cargar dataset de prueba
    test_dataset = load_test_dataset("../Data/queries/test_queries.json")
    print(f"✅ Dataset de prueba cargado: {len(test_dataset)} consultas")
    
    # Configurar evaluador
    metrics = RAGMetrics()
    evaluator = RAGEvaluator(metrics)
    
    # Ejecutar evaluación
    print("🧪 Iniciando evaluación...")
    results = await evaluator.evaluate_dataset(rag_systems, test_dataset)
    
    # Guardar resultados
    results_path = Path(__file__).parent.parent / "results" / "benchmark_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    evaluator.save_results(results, str(results_path))
    
    # Mostrar resumen
    print("\\n📊 RESUMEN DE RESULTADOS:")
    print("=" * 50)
    
    for system_name, summary in results["system_comparison"].items():
        if system_name != "best_performers":
            print(f"\\n{system_name}:")
            print(f"  BLEU Score: {summary.get('avg_bleu_score', 0):.3f}")
            print(f"  ROUGE-L: {summary.get('avg_rouge_l', 0):.3f}")
            print(f"  BERT Score F1: {summary.get('avg_bert_score_f1', 0):.3f}")
            print(f"  Tiempo promedio: {summary.get('avg_response_time', 0):.3f}s")
    
    print(f"\\n🏆 MEJORES SISTEMAS POR MÉTRICA:")
    best_performers = results["system_comparison"].get("best_performers", {})
    for metric, info in best_performers.items():
        print(f"  {metric}: {info['system']} ({info['score']:.3f})")
    
    # Limpiar recursos
    for rag_system in rag_systems:
        await rag_system.cleanup()
    
    print("\\n✅ Benchmark completado exitosamente!")
    print(f"📁 Resultados guardados en: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())