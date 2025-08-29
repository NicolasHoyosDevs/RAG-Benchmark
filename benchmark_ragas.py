"""
Benchmark script using RAGAS to evaluate 4 RAG strategies.
Evaluates: faithfulness, answer_relevancy, context_precision, context_recall.
Also tracks time, tokens, and cost.
"""

from Simple_Semantic_RAG.simple_semantic_rag import SimpleSemanticRAG, rag_chain as simple_rag_chain
from HyDE_RAG.hyde_rag import process_hyde_query
from Query_Rewriter_RAG.main_rewriter import query_for_evaluation as query_rewriter_eval
from Hybrid_RAG.hybrid_langchain_bm25 import HybridRAG, rag_chain as hybrid_rag_chain
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Load environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Import RAG modules

# Questions and ground truths
QUESTIONS_DATA = [
    {
        "query": "¿En qué momento y quién debe reevaluar el riesgo clínico de una paciente con embarazo de curso normal?",
        "ground_truth": "El Ginecobstetra en la semana 28 - 30 y semana 34 – 36."
    },
    {
        "query": "¿Cuál es la semana ideal para iniciar los controles prenatales?",
        "ground_truth": "Se recomienda realizar el primer control prenatal en el primer trimestre, idealmente antes de la semana 10 de gestación."
    },
    {
        "query": "¿Cuándo se considera inicio tardío de controles prenatales?",
        "ground_truth": "Después de la semana 16 – 18."
    },
    {
        "query": "¿Cuál es la cantidad ideal de controles prenatales?",
        "ground_truth": "Se recomienda un programa de diez citas. Para una mujer multípara con un embarazo de curso normal se recomienda un programa de siete citas."
    },
    {
        "query": "¿Con cuál herramienta realizo la valoración de riesgo psicosocial en los controles prenatales?",
        "ground_truth": "Se recomienda que las gestantes de bajo riesgo reciban en el momento de la inscripción al control prenatal, y luego en cada trimestre, una valoración de riesgo psicosocial."
    },
    {
        "query": "¿Cuándo realizar la valoración de riesgo psicosocial en el seguimiento de una materna?",
        "ground_truth": "Se recomienda evaluar el riesgo biológico y psicosocial a todas las gestantes mediante la escala de Herrera & Hurtado."
    },
    {
        "query": "¿Cada cuánto se realiza el tamizaje para depresión posparto durante el embarazo?",
        "ground_truth": "Se recomienda que, en el primer control prenatal, en la semana 28 de gestación y en la consulta de puerperio se identifique el riesgo de depresión postparto."
    },
    {
        "query": "¿Cuál es la probabilidad de parto vaginal luego de una cesárea?",
        "ground_truth": "La probabilidad de parto vaginal es de 74% luego de una cesárea sin otros riesgos asociados."
    },
    {
        "query": "¿Cuáles son las metas de ganancia de peso en las mujeres gestantes?",
        "ground_truth": "Se recomienda registrar el Índice de Masa Corporal (IMC) de la gestante en la cita de inscripción al control prenatal (alrededor de la semana 10) y con base en este establecer las metas de ganancia de peso durante la gestación de acuerdo a los siguientes parámetros:\n• IMC < 20 kg/m² = ganancia entre 12 a 18 Kg\n• IMC entre 20 y 24,9 kg/m² = ganancia entre 10 a 13 Kg\n• IMC entre 25 y 29,9 kg/m² = ganancia entre 7 a 10 Kg\n• IMC > 30 kg/m² = ganancia entre 6 a 7 Kg"
    },
    {
        "query": "¿Cuál es el tratamiento para las náuseas y vómito del embarazo?",
        "ground_truth": "A juicio del médico tratante, las intervenciones recomendadas para la reducción de la náusea y el vómito incluyen el jengibre, los antihistamínicos y la vitamina B6."
    }
]

# Initialize RAG instances
hybrid_rag = HybridRAG()
simple_rag = SimpleSemanticRAG()


def evaluate_hybrid_rag(query: str) -> Dict[str, Any]:
    """Evaluate Hybrid RAG and return structured data."""
    try:
        start_time = time.time()

        # Get retrieved docs for contexts
        retrieved_docs = hybrid_rag.search(query)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generate answer
        answer = hybrid_rag_chain.invoke(query)

        end_time = time.time()
        total_time = end_time - start_time

        # Note: Hybrid RAG doesn't track tokens/cost in this version, so set to 0
        # You can modify hybrid_langchain_bm25.py to track them if needed
        return {
            "success": True,
            "answer": answer,
            "contexts": contexts,
            "performance": {
                "tiempo_total": total_time,
                "tokens_entrada": 0,  # Not tracked
                "tokens_salida": 0,   # Not tracked
                "costo_total": 0.0    # Not tracked
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "performance": {
                "tiempo_total": 0,
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "costo_total": 0
            }
        }


def evaluate_simple_semantic_rag(query: str) -> Dict[str, Any]:
    """Evaluate Simple Semantic RAG and return structured data."""
    try:
        start_time = time.time()

        # Get retrieved docs for contexts
        retrieved_docs = simple_rag.search(query)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generate answer
        answer = simple_rag_chain.invoke(query)

        end_time = time.time()
        total_time = end_time - start_time

        # Note: Simple RAG doesn't track tokens/cost in this version, so set to 0
        return {
            "success": True,
            "answer": answer,
            "contexts": contexts,
            "performance": {
                "tiempo_total": total_time,
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "costo_total": 0.0
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "performance": {
                "tiempo_total": 0,
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "costo_total": 0
            }
        }


def evaluate_query_rewriter_rag(query: str) -> Dict[str, Any]:
    """Evaluate Query Rewriter RAG and return structured data."""
    try:
        result = query_rewriter_eval(query)
        return {
            "success": True,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "performance": {
                "tiempo_total": 0,  # Not tracked in this function, can be added
                # Approximate split
                "tokens_entrada": result["metadata"]["tokens_used"] // 2,
                "tokens_salida": result["metadata"]["tokens_used"] // 2,
                "costo_total": result["metadata"]["total_cost"]
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "performance": {
                "tiempo_total": 0,
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "costo_total": 0
            }
        }


def evaluate_hyde_rag(query: str) -> Dict[str, Any]:
    """Evaluate HyDE RAG and return structured data."""
    try:
        start_time = time.time()
        result = process_hyde_query(query)
        end_time = time.time()
        total_time = end_time - start_time

        # Get contexts from the result
        contexts = result.get("contexts", [])

        return {
            "success": True,
            "answer": result["answer"],
            "contexts": contexts,
            "performance": {
                "tiempo_total": total_time,
                "tokens_entrada": result["total_input_tokens"],
                "tokens_salida": result["total_output_tokens"],
                "costo_total": result["total_cost"]
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "performance": {
                "tiempo_total": 0,
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "costo_total": 0
            }
        }


def run_ragas_evaluation(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run RAGAS evaluation on the dataset."""
    dataset = Dataset.from_list(data)
    metrics = [faithfulness, answer_relevancy,
               context_precision, context_recall]

    try:
        results = evaluate(dataset, metrics)

        def safe_float(value):
            """Safely convert value to float, handling lists/arrays."""
            if isinstance(value, (list, tuple)) and len(value) > 0:
                # If it's a list/array, take the first element or mean
                if hasattr(value, 'mean'):
                    return float(value.mean())
                else:
                    return float(value[0])
            elif hasattr(value, 'item'):
                # If it's a numpy scalar
                return float(value.item())
            else:
                return float(value)

        return {
            "faithfulness": safe_float(results["faithfulness"]),
            "answer_relevancy": safe_float(results["answer_relevancy"]),
            "context_precision": safe_float(results["context_precision"]),
            "context_recall": safe_float(results["context_recall"])
        }
    except Exception as e:
        print(f"Error in RAGAS evaluation: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }


def main():
    """Main benchmark function."""
    results = {}

    for i, q_data in enumerate(QUESTIONS_DATA, 1):
        query = q_data["query"]
        ground_truth = q_data["ground_truth"]
        q_key = f"q{i}"

        print(f"Evaluating question {i}: {query[:50]}...")

        results[q_key] = {
            "query": query,
            "ground_truth": ground_truth,
            "strategies": {}
        }

        # Evaluate each strategy
        strategies = {
            "Hybrid_RAG": evaluate_hybrid_rag,
            "Query_Rewriter_RAG": evaluate_query_rewriter_rag,
            "HyDE_RAG": evaluate_hyde_rag,
            "Simple_Semantic_RAG": evaluate_simple_semantic_rag
        }

        for strategy_name, eval_func in strategies.items():
            print(f"  Running {strategy_name}...")
            strategy_result = eval_func(query)

            # Prepare data for RAGAS if successful
            if strategy_result["success"] and strategy_result["contexts"]:
                # Validate and clean contexts
                contexts = strategy_result["contexts"]
                if isinstance(contexts, list) and len(contexts) > 0:
                    # Ensure all contexts are strings and not empty
                    valid_contexts = [str(ctx).strip()
                                      for ctx in contexts if str(ctx).strip()]
                    if valid_contexts:
                        ragas_data = [{
                            "question": query,
                            "answer": str(strategy_result["answer"]).strip(),
                            "contexts": valid_contexts,
                            "ground_truth": ground_truth
                        }]
                        print(
                            f"    Evaluating with {len(valid_contexts)} contexts")
                        metrics = run_ragas_evaluation(ragas_data)
                    else:
                        print(
                            f"    No valid contexts found for {strategy_name}")
                        metrics = {
                            "faithfulness": 0.0,
                            "answer_relevancy": 0.0,
                            "context_precision": 0.0,
                            "context_recall": 0.0
                        }
                else:
                    print(
                        f"    Invalid contexts format for {strategy_name}: {type(contexts)}")
                    metrics = {
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "context_precision": 0.0,
                        "context_recall": 0.0
                    }
            else:
                print(
                    f"    Strategy {strategy_name} failed or has no contexts")
                metrics = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0
                }

            results[q_key]["strategies"][strategy_name] = {
                "success": strategy_result["success"],
                "answer": strategy_result.get("answer", ""),
                "contexts": strategy_result.get("contexts", []),
                "performance": strategy_result["performance"],
                "metrics": metrics
            }

            if not strategy_result["success"]:
                results[q_key]["strategies"][strategy_name]["error"] = strategy_result.get(
                    "error", "")

    # Save results to JSON
    output_path = Path(__file__).resolve().parent / \
        "results" / "benchmark_ragas_results.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
