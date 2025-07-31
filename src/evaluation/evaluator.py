"""
Sistema de evaluación para comparar diferentes implementaciones de RAG
"""

import time
from typing import List, Dict, Any, Optional
from ..common.base_rag import BaseRAG, RAGResponse
from .metrics import RAGMetrics


class RAGEvaluator:
    """
    Evaluador para comparar diferentes implementaciones de RAG
    """
    
    def __init__(self, metrics: Optional[RAGMetrics] = None):
        self.metrics = metrics or RAGMetrics()
        self.results = []
    
    async def evaluate_single(self, rag_system: BaseRAG, 
                            question: str, ground_truth: str,
                            context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evalúa una sola consulta para un sistema RAG
        """
        # Medir tiempo de respuesta
        start_time = time.time()
        response = await rag_system.query(question)
        response_time = time.time() - start_time
        
        # Calcular métricas
        metrics_result = await self.metrics.calculate_all_metrics(
            prediction=response.answer,
            reference=ground_truth,
            context=context,
            sources=response.sources
        )
        
        # Compilar resultado
        result = {
            "rag_system": rag_system.name,
            "question": question,
            "predicted_answer": response.answer,
            "ground_truth": ground_truth,
            "response_time": response_time,
            "confidence": response.confidence,
            "sources_count": len(response.sources),
            "metadata": response.metadata,
            **metrics_result
        }
        
        return result
    
    async def evaluate_dataset(self, rag_systems: List[BaseRAG], 
                             test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evalúa múltiples sistemas RAG en un dataset completo
        """
        all_results = []
        
        for rag_system in rag_systems:
            system_results = []
            
            for test_case in test_dataset:
                question = test_case["question"]
                ground_truth = test_case["answer"]
                context = test_case.get("context")
                
                result = await self.evaluate_single(
                    rag_system, question, ground_truth, context
                )
                system_results.append(result)
                all_results.append(result)
            
            # Calcular métricas promedio por sistema
            system_summary = self._calculate_system_summary(system_results)
            print(f"Resultados para {rag_system.name}:")
            self._print_summary(system_summary)
        
        # Calcular comparación entre sistemas
        comparison = self._compare_systems(all_results, rag_systems)
        
        return {
            "individual_results": all_results,
            "system_comparison": comparison,
            "dataset_size": len(test_dataset),
            "systems_evaluated": [rag.name for rag in rag_systems]
        }
    
    def _calculate_system_summary(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calcula métricas promedio para un sistema
        """
        if not results:
            return {}
        
        summary = {}
        numeric_fields = [
            "response_time", "confidence", "sources_count",
            "bleu_score", "rouge_1", "rouge_2", "rouge_l",
            "bert_score_f1", "faithfulness", "relevance"
        ]
        
        for field in numeric_fields:
            values = [r.get(field, 0) for r in results if r.get(field) is not None]
            if values:
                summary[f"avg_{field}"] = sum(values) / len(values)
                summary[f"min_{field}"] = min(values)
                summary[f"max_{field}"] = max(values)
        
        return summary
    
    def _compare_systems(self, all_results: List[Dict[str, Any]], 
                        rag_systems: List[BaseRAG]) -> Dict[str, Any]:
        """
        Compara el rendimiento entre sistemas
        """
        comparison = {}
        
        for rag_system in rag_systems:
            system_results = [r for r in all_results if r["rag_system"] == rag_system.name]
            summary = self._calculate_system_summary(system_results)
            comparison[rag_system.name] = summary
        
        # Determinar el mejor sistema por métrica
        best_systems = {}
        metrics_to_compare = ["avg_bleu_score", "avg_rouge_l", "avg_bert_score_f1", 
                             "avg_faithfulness", "avg_relevance"]
        
        for metric in metrics_to_compare:
            best_score = -1
            best_system = None
            
            for system_name, summary in comparison.items():
                score = summary.get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_system = system_name
            
            if best_system:
                best_systems[metric] = {
                    "system": best_system,
                    "score": best_score
                }
        
        comparison["best_performers"] = best_systems
        return comparison
    
    def _print_summary(self, summary: Dict[str, float]) -> None:
        """
        Imprime un resumen de métricas
        """
        print(f"  Tiempo promedio de respuesta: {summary.get('avg_response_time', 0):.3f}s")
        print(f"  BLEU Score promedio: {summary.get('avg_bleu_score', 0):.3f}")
        print(f"  ROUGE-L promedio: {summary.get('avg_rouge_l', 0):.3f}")
        print(f"  BERT Score F1 promedio: {summary.get('avg_bert_score_f1', 0):.3f}")
        print(f"  Faithfulness promedio: {summary.get('avg_faithfulness', 0):.3f}")
        print(f"  Relevance promedio: {summary.get('avg_relevance', 0):.3f}")
        print(f"  Confianza promedio: {summary.get('avg_confidence', 0):.3f}")
        print()
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Guarda los resultados de evaluación
        """
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados guardados en: {output_path}")