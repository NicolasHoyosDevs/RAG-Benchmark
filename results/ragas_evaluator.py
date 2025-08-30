"""
RAGAS Evaluator for RAG Systems
Professional evaluation using RAGAS fundamental metrics
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Project configuration
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# RAG systems imports
from Query_Rewriter_RAG.main_rewriter import query_for_evaluation as rewriter_query_for_evaluation
from Hybrid_RAG.hybrid_langchain_bm25 import query_for_evaluation as hybrid_query_for_evaluation
from HyDE_RAG.hyde_rag import query_for_evaluation as hyde_query_for_evaluation
from Simple_Semantic_RAG.simple_semantic_rag import query_for_evaluation as simple_query_for_evaluation

# Obstetric and pregnancy specific dataset
DATA_GT = [
    {
        "question": "¿En qué momento y quien debe reevaluar el riesgo clínico de una paciente con embarazo de curso normal?",
        "ground_truth": "El Ginecobstetra en la semana 28 - 30 y semana 34 – 36."
    },
    {
        "question": "Cual es la semana ideal para iniciar los controles prenatales",
        "ground_truth": "Se recomienda realizar el primer control prenatal en el primer trimestre, idealmente antes de la semana 10 de gestación."
    },
    {
        "question": "¿Cuándo se considera inicio tardío de controles prenatales?",
        "ground_truth": "después de la semana 16 – 18"
    },
    {
        "question": "¿Cuál es la cantidad ideal de controles prenatales?",
        "ground_truth": "se recomienda un programa de diez citas. Para una mujer multípara con un embarazo de curso normal se recomienda un programa de siete citas"
    },
    {
        "question": "¿Con cuál herramienta realizo la valoración de riesgo psico social en los controles prenatales?",
        "ground_truth": "Se recomienda que las gestantes de bajo riesgo reciban en el momento de la inscripción al control prenatal, y luego en cada trimestre, una valoración de riesgo psicosocial"
    },
    {
        "question": "¿Cuándo realizar la valoración de riesgo psico social en el seguimiento de una materna?",
        "ground_truth": "Se recomienda evaluar el riesgo biológico y psicosocial a todas las gestantes mediante la escala de Herrera & Hurtado"
    },
    {
        "question": "¿Cada cuanto se realiza el tamizaje para depresión posparto durante el embarazo?",
        "ground_truth": "Se recomienda que, en el primer control prenatal, en la semana 28 de gestación y en la consulta de puerperio se identifique el riesgo de depresión postparto"
    },
    {
        "question": "¿Cuál es la probabilidad de parto vaginal luego de una cesárea?",
        "ground_truth": "La probabilidad de parto vaginal es de 74% luego de una cesárea sin otros riesgos asociados."
    },
    {
        "question": "¿Cuáles son las metas de ganancia de peso en las mujeres gestantes?",
        "ground_truth": "Se recomienda registrar el Índice de Masa Corporal (IMC) de la gestante en la cita de inscripción al control prenatal (alrededor de la semana 10) y con base en este establecer las metas de ganancia de peso durante la gestación de acuerdo a los siguientes parámetros: • IMC < 20 kg/m2 = ganancia entre 12 a 18 Kg • IMC entre 20 y 24,9 kg/m2 = ganancia entre 10 a 13 Kg • IMC entre 25 y 29,9 kg/m2 = ganancia entre 7 a 10 Kg • IMC > 30 kg/m2 = ganancia entre 6 a 7 Kg"
    },
    {
        "question": "Cual es el tratamiento para las nauseas y vomito del embarazo",
        "ground_truth": "A juicio del médico tratante, las intervenciones recomendadas para la reducción de la náusea y el vómito incluyen el jengibre, los antihistamínicos y la vitamina B6"
    },
]


class RAGASEvaluator:
    """Professional RAGAS evaluator for RAG systems"""
    
    def __init__(self, rag_type: str = "rewriter", debug: bool = False):
        """
        Initialize RAGAS evaluator
        
        Args:
            rag_type (str): RAG type to evaluate ("rewriter", "hybrid", "hyde", or "simple")
            debug (bool): Enable debug output
        """
        self.metrics = [
            faithfulness,        # Answer faithfulness to context
            answer_relevancy,    # Answer relevance to question
            context_precision,   # Precision of retrieved contexts
            context_recall       # Recall of necessary information
        ]
        self.results_dir = Path(__file__).parent
        self.debug = debug
        
        # Configure evaluation function based on RAG type
        if rag_type.lower() == "hybrid":
            self.query_function = hybrid_query_for_evaluation
            self.rag_name = "Hybrid RAG (BM25 + Semantic)"
            self.rag_type = "hybrid"
        elif rag_type.lower() == "rewriter":
            self.query_function = rewriter_query_for_evaluation
            self.rag_name = "Rewriter RAG (Multi-Query)"
            self.rag_type = "rewriter"
        elif rag_type.lower() == "hyde":
            self.query_function = hyde_query_for_evaluation
            self.rag_name = "HyDE RAG (Hypothetical Documents)"
            self.rag_type = "hyde"
        elif rag_type.lower() == "simple":
            self.query_function = simple_query_for_evaluation
            self.rag_name = "Simple Semantic RAG"
            self.rag_type = "simple"
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}. Use 'rewriter', 'hybrid', 'hyde', or 'simple'")
        
        print(f"RAGAS Evaluator configured for: {self.rag_name}")
        
    def load_test_queries(self, use_obstetric_dataset: bool = True) -> List[Dict]:
        """
        Load test queries from obstetric dataset
        """
        return DATA_GT
        
    
    def prepare_dataset(self, test_queries: List[Dict]) -> Dataset:
        """
        Prepare RAGAS dataset format using the configured RAG system
        
        Args:
            test_queries: List of test queries
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        # Capture performance metadata
        self.performance_metadata = []
        
        print(f"Processing {len(test_queries)} queries with {self.rag_name}")
        
        for i, query_data in enumerate(test_queries, 1):
            question = query_data["question"]
            ground_truth = query_data.get("ground_truth", query_data.get("answer", ""))
            
            try:
                # Use configured evaluation function
                rag_result = self.query_function(question)
                
                # Extract data for RAGAS
                questions.append(question)
                answers.append(rag_result["answer"])
                contexts.append(rag_result["contexts"])
                ground_truths.append(ground_truth)
                
                # Debug: Print first few queries to verify data (only if debug enabled)
                if self.debug and i <= 2:
                    print(f"DEBUG - Query {i}:")
                    print(f"  Question: {question[:100]}...")
                    print(f"  Answer: {rag_result['answer'][:100]}...")
                    print(f"  Contexts count: {len(rag_result['contexts'])}")
                    print(f"  Ground truth: {ground_truth[:100]}...")
                
                # Save performance metadata
                metadata = rag_result.get("metadata", {})
                performance_data = {
                    "question": question,
                    "execution_time": metadata.get("execution_time", 0.0),
                    "input_tokens": metadata.get("input_tokens", 0),
                    "output_tokens": metadata.get("output_tokens", 0),
                    "total_cost": metadata.get("total_cost", 0.0)
                }
                self.performance_metadata.append(performance_data)
                
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Create RAGAS dataset
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        
        print(f"Dataset prepared: {len(dataset)} queries processed")
        return dataset
    
    def evaluate_rag(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate RAG using RAGAS metrics
        
        Args:
            dataset: Prepared dataset with questions, answers, contexts and ground truth
            
        Returns:
            Dict with evaluation results
        """
        print("Starting RAGAS evaluation...")
        
        # Store original dataset for reference in save_results
        self.original_dataset = dataset
        
        try:
            # Execute evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
            )
            
            print("Evaluation completed")
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def display_results(self, results):
        """Display evaluation results in a clean format"""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        
        # Extract metrics
        scores = {}
        for metric in self.metrics:
            metric_name = metric.name
            try:
                if hasattr(results, metric_name):
                    score = getattr(results, metric_name)
                elif hasattr(results, 'to_pandas'):
                    df = results.to_pandas()
                    if metric_name in df.columns:
                        score = df[metric_name].mean()
                    else:
                        continue
                else:
                    score = results[metric_name]
                
                scores[metric_name] = float(score)
                print(f"{metric_name.replace('_', ' ').title()}: {score:.3f}")
            except (KeyError, AttributeError, TypeError) as e:
                if self.debug:
                    print(f"Could not extract metric {metric_name}: {e}")
                continue
        
        if not scores:
            print("Could not extract metrics")
            return
        
        # Overall evaluation
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            print(f"\nAverage Score: {avg_score:.3f}")
            
            if avg_score >= 0.8:
                print("Performance: Excellent")
            elif avg_score >= 0.6:
                print("Performance: Good")
            elif avg_score >= 0.4:
                print("Performance: Needs improvement")
            else:
                print("Performance: Significant improvements needed")
    
    def save_results(self, results, filename: str = None):
        """Save evaluation results to JSON file with detailed per-question results"""
        if not results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_evaluation_{self.rag_type}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Extract aggregated metrics from results object
        aggregated_results = {}
        detailed_results = []
        
        try:
            # Get aggregated metrics
            for metric in self.metrics:
                metric_name = metric.name
                if hasattr(results, metric_name):
                    aggregated_results[metric_name] = float(getattr(results, metric_name))
                elif hasattr(results, 'to_pandas'):
                    df = results.to_pandas()
                    if metric_name in df.columns:
                        aggregated_results[metric_name] = float(df[metric_name].mean())
            
            # Get detailed per-question results
            if hasattr(results, 'to_pandas'):
                df = results.to_pandas()
                
                # Debug: Print column names and data to understand structure (only if debug enabled)
                if self.debug:
                    print(f"DEBUG - DataFrame columns: {df.columns.tolist()}")
                    print(f"DEBUG - DataFrame shape: {df.shape}")
                    if len(df) > 0:
                        first_row = df.iloc[0]
                        print(f"DEBUG - First row sample: question={str(first_row.get('question', ''))[:50]}...")
                    
                    # Debug original dataset if available
                    if hasattr(self, 'original_dataset') and self.original_dataset is not None:
                        original_df = self.original_dataset.to_pandas()
                        print(f"DEBUG - Original dataset columns: {original_df.columns.tolist()}")
                
                for i, row in df.iterrows():
                    # Get data from original dataset if available, otherwise from results
                    if hasattr(self, 'original_dataset') and self.original_dataset is not None and i < len(self.original_dataset):
                        orig_df = self.original_dataset.to_pandas()
                        orig_row = orig_df.iloc[i]
                        question_text = str(orig_row['question']) if 'question' in orig_df.columns else ''
                        ground_truth_text = str(orig_row['ground_truth']) if 'ground_truth' in orig_df.columns else ''
                        answer_text = str(orig_row['answer']) if 'answer' in orig_df.columns else ''
                        contexts_data = orig_row['contexts'] if 'contexts' in orig_df.columns else []
                    else:
                        # Fallback to results dataframe
                        question_text = str(row['question']) if 'question' in df.columns else ''
                        ground_truth_text = str(row['ground_truth']) if 'ground_truth' in df.columns else ''
                        answer_text = str(row['answer']) if 'answer' in df.columns else ''
                        contexts_data = row['contexts'] if 'contexts' in df.columns else []
                    
                    # Debug contexts data for first few entries (only if debug enabled)
                    if self.debug and i < 2:
                        print(f"DEBUG - Question {i+1} contexts:")
                        print(f"  Type: {type(contexts_data)}")
                        if isinstance(contexts_data, list):
                            print(f"  Length: {len(contexts_data)}")
                        elif hasattr(contexts_data, '__len__'):
                            print(f"  Length: {len(contexts_data)}")
                    
                    # Improved contexts count calculation
                    if isinstance(contexts_data, list):
                        contexts_count = len(contexts_data)
                    elif hasattr(contexts_data, '__len__'):
                        contexts_count = len(contexts_data)
                    elif contexts_data is not None:
                        contexts_count = 1
                    else:
                        contexts_count = 0
                    
                    question_result = {
                        "question_id": i + 1,
                        "question": question_text,
                        "ground_truth": ground_truth_text,
                        "answer": answer_text,
                        "contexts_count": contexts_count,
                        "metrics": {
                            "faithfulness": float(row['faithfulness']) if 'faithfulness' in df.columns else 0.0,
                            "answer_relevancy": float(row['answer_relevancy']) if 'answer_relevancy' in df.columns else 0.0,
                            "context_precision": float(row['context_precision']) if 'context_precision' in df.columns else 0.0,
                            "context_recall": float(row['context_recall']) if 'context_recall' in df.columns else 0.0
                        }
                    }
                    
                    # Add performance metadata if available
                    if hasattr(self, 'performance_metadata') and i < len(self.performance_metadata):
                        perf_data = self.performance_metadata[i]
                        question_result["performance"] = {
                            "execution_time": perf_data.get('execution_time', 0.0),
                            "input_tokens": perf_data.get('input_tokens', 0),
                            "output_tokens": perf_data.get('output_tokens', 0),
                            "total_cost": perf_data.get('total_cost', 0.0)
                        }
                    else:
                        # Add empty performance data
                        question_result["performance"] = {
                            "execution_time": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_cost": 0.0
                        }
                    
                    detailed_results.append(question_result)
                    
        except Exception as e:
            print(f"Error extracting metrics for saving: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            aggregated_results = {"error": str(e)}
        
        # Calculate additional aggregated statistics
        overall_stats = {}
        if detailed_results:
            # Average performance metrics
            avg_execution_time = sum(q.get('performance', {}).get('execution_time', 0) for q in detailed_results) / len(detailed_results)
            total_input_tokens = sum(q.get('performance', {}).get('input_tokens', 0) for q in detailed_results)
            total_output_tokens = sum(q.get('performance', {}).get('output_tokens', 0) for q in detailed_results)
            total_cost = sum(q.get('performance', {}).get('total_cost', 0) for q in detailed_results)
            
            overall_stats = {
                "average_execution_time": round(avg_execution_time, 3),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost": round(total_cost, 6),
                "average_cost_per_question": round(total_cost / len(detailed_results), 6) if detailed_results else 0,
                "overall_average_score": round(sum(aggregated_results.values()) / len(aggregated_results), 3) if aggregated_results and len(aggregated_results) > 0 else 0
            }
        
        # Prepare comprehensive data for saving
        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "rag_system": {
                    "name": self.rag_name,
                    "type": self.rag_type
                },
                "dataset_size": len(detailed_results),
                "metrics_used": [m.name for m in self.metrics]
            },
            "aggregated_results": aggregated_results,
            "overall_statistics": overall_stats,
            "detailed_results": detailed_results
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
            
        return filepath
    
    def run_evaluation(self):
        """
        Execute complete evaluation with obstetric dataset
        """
        print("Starting RAGAS evaluation")
        print(f"System: {self.rag_name}")
        print(f"Dataset: Obstetric queries ({len(DATA_GT)} questions)")
        print("="*60)
        
        # 1. Load test queries
        test_queries = self.load_test_queries(use_obstetric_dataset=True)
        if not test_queries:
            print("Cannot load test queries")
            return
        
        # 2. Prepare dataset
        dataset = self.prepare_dataset(test_queries)
        if len(dataset) == 0:
            print("Could not prepare dataset")
            return
        
        # 3. Evaluate
        results = self.evaluate_rag(dataset)
        if not results:
            print("Evaluation failed")
            return
        
        # 4. Display results
        self.display_results(results)
        
        # 5. Save results
        self.save_results(results)
        
        print(f"\nEvaluation completed - {len(dataset)} queries processed")
        return results


def evaluate_rewriter_rag(export_analysis: bool = False, debug: bool = False):
    """Evaluate Rewriter RAG specifically"""
    evaluator = RAGASEvaluator(rag_type="rewriter", debug=debug)
    results = evaluator.run_evaluation()
    
    if export_analysis:
        try:
            from utils import export_ragas_analysis
            
            performance_metadata = getattr(evaluator, 'performance_metadata', None)
            export_files = export_ragas_analysis(results, "rewriter_rag", performance_metadata=performance_metadata)
            print("\nDetailed analysis exported:")
            for file_type, file_path in export_files.items():
                print(f"  {file_type}: {file_path.name}")
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def evaluate_hybrid_rag(export_analysis: bool = False, debug: bool = False):
    """Evaluate Hybrid RAG specifically"""
    evaluator = RAGASEvaluator(rag_type="hybrid", debug=debug)
    results = evaluator.run_evaluation()
    
    if export_analysis:
        try:
            from utils import export_ragas_analysis
            
            performance_metadata = getattr(evaluator, 'performance_metadata', None)
            export_files = export_ragas_analysis(results, "hybrid_rag", performance_metadata=performance_metadata)
            print("\nDetailed analysis exported:")
            for file_type, file_path in export_files.items():
                print(f"  {file_type}: {file_path.name}")
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def evaluate_hyde_rag(export_analysis: bool = False, debug: bool = False):
    """Evaluate HyDE RAG specifically"""
    evaluator = RAGASEvaluator(rag_type="hyde", debug=debug)
    results = evaluator.run_evaluation()
    
    if export_analysis:
        try:
            from utils import export_ragas_analysis
            
            performance_metadata = getattr(evaluator, 'performance_metadata', None)
            export_files = export_ragas_analysis(results, "hyde_rag", performance_metadata=performance_metadata)
            print("\nDetailed analysis exported:")
            for file_type, file_path in export_files.items():
                print(f"  {file_type}: {file_path.name}")
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def evaluate_simple_rag(export_analysis: bool = False, debug: bool = False):
    """Evaluate Simple Semantic RAG specifically"""
    evaluator = RAGASEvaluator(rag_type="simple", debug=debug)
    results = evaluator.run_evaluation()
    
    if export_analysis:
        try:
            from utils import export_ragas_analysis
            
            performance_metadata = getattr(evaluator, 'performance_metadata', None)
            export_files = export_ragas_analysis(results, "simple_rag", performance_metadata=performance_metadata)
            print("\nDetailed analysis exported:")
            for file_type, file_path in export_files.items():
                print(f"  {file_type}: {file_path.name}")
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def evaluate_both_rags(export_analysis: bool = False, debug: bool = False):
    """Evaluate both original RAG systems sequentially (rewriter and hybrid)"""
    print("Evaluating both original RAG systems")
    print("="*70)
    
    # Evaluate Rewriter RAG
    print("\n" + "="*25 + " REWRITER RAG " + "="*25)
    rewriter_results = evaluate_rewriter_rag(export_analysis=export_analysis, debug=debug)
    
    print("\n" + "="*70)
    print("Pause between evaluations...")
    import time
    time.sleep(2)
    
    # Evaluate Hybrid RAG
    print("\n" + "="*26 + " HYBRID RAG " + "="*26)
    hybrid_results = evaluate_hybrid_rag(export_analysis=export_analysis, debug=debug)
    
    print("\n" + "="*70)
    print("Complete evaluation of both systems finished")
    if export_analysis:
        print("Detailed analysis exported for both systems")
    else:
        print("Check generated JSON files to compare results")
        print("For detailed analysis export use: --export")
    
    return {
        "rewriter": rewriter_results,
        "hybrid": hybrid_results
    }


def evaluate_all_rags(export_analysis: bool = False, debug: bool = False):
    """Evaluate all 4 RAG systems sequentially"""
    print("Evaluating all 4 RAG systems")
    print("="*80)
    
    results = {}
    
    # Evaluate Simple Semantic RAG
    print("\n" + "="*25 + " SIMPLE SEMANTIC RAG " + "="*25)
    results["simple"] = evaluate_simple_rag(export_analysis=export_analysis, debug=debug)
    
    print("\n" + "="*80)
    print("Pause between evaluations...")
    import time
    time.sleep(2)
    
    # Evaluate HyDE RAG
    print("\n" + "="*30 + " HYDE RAG " + "="*30)
    results["hyde"] = evaluate_hyde_rag(export_analysis=export_analysis, debug=debug)
    
    print("\n" + "="*80)
    print("Pause between evaluations...")
    time.sleep(2)
    
    # Evaluate Rewriter RAG
    print("\n" + "="*27 + " REWRITER RAG " + "="*27)
    results["rewriter"] = evaluate_rewriter_rag(export_analysis=export_analysis, debug=debug)
    
    print("\n" + "="*80)
    print("Pause between evaluations...")
    time.sleep(2)
    
    # Evaluate Hybrid RAG
    print("\n" + "="*28 + " HYBRID RAG " + "="*28)
    results["hybrid"] = evaluate_hybrid_rag(export_analysis=export_analysis, debug=debug)
    
    print("\n" + "="*80)
    print("Complete evaluation of all 4 RAG systems finished")
    if export_analysis:
        print("Detailed analysis exported for all systems")
    else:
        print("Check generated JSON files to compare results")
        print("For detailed analysis export use: --export")
    
    # Save comparison results
    save_comparison_results(results)
    return results


def save_comparison_results(all_results: Dict[str, Any]):
    """Save comparison results of all RAGs to a comprehensive JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(__file__).parent / f"ragas_comparison_all_rags_{timestamp}.json"
    
    # Extract aggregated data for each RAG
    comparison_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "comprehensive_rag_comparison",
            "dataset_size": 10,
            "rags_evaluated": list(all_results.keys())
        },
        "summary": {},
        "detailed_comparison": {},
        "question_by_question": []
    }
    
    # Initialize question-by-question structure
    for i in range(10):  # 10 questions in our dataset
        question_data = {
            "question_id": i + 1,
            "question": DATA_GT[i]["question"],
            "ground_truth": DATA_GT[i]["ground_truth"],
            "rag_results": {}
        }
        comparison_data["question_by_question"].append(question_data)
    
    # Process each RAG's results - now we need to get the detailed data from saved JSON files
    for rag_type, rag_results in all_results.items():
        if not rag_results:
            continue
            
        try:
            # Try to find the most recent JSON file for this RAG type
            results_dir = Path(__file__).parent
            pattern = f"ragas_evaluation_{rag_type}_*.json"
            json_files = list(results_dir.glob(pattern))
            
            if not json_files:
                print(f"Warning: No JSON file found for {rag_type}")
                continue
            
            # Get the most recent file
            json_files.sort(key=lambda x: x.stem.split('_')[-2:], reverse=True)
            latest_json_file = json_files[0]
            
            # Load the detailed JSON data
            with open(latest_json_file, 'r', encoding='utf-8') as f:
                detailed_data = json.load(f)
            
            # Extract aggregated metrics from the JSON
            aggregated_metrics = detailed_data.get('aggregated_results', {})
            
            # Add overall average if not present
            if aggregated_metrics and 'overall_average' not in aggregated_metrics:
                metric_values = [v for v in aggregated_metrics.values() if isinstance(v, (int, float))]
                if metric_values:
                    overall_avg = sum(metric_values) / len(metric_values)
                    aggregated_metrics['overall_average'] = round(overall_avg, 3)
            
            # Extract performance metrics from overall_statistics
            overall_stats = detailed_data.get('overall_statistics', {})
            performance_metrics = {
                'avg_execution_time': round(overall_stats.get('average_execution_time', 0), 3),
                'total_input_tokens': overall_stats.get('total_input_tokens', 0),
                'total_output_tokens': overall_stats.get('total_output_tokens', 0),
                'total_tokens': overall_stats.get('total_input_tokens', 0) + overall_stats.get('total_output_tokens', 0),
                'total_cost': round(overall_stats.get('total_cost', 0), 6),
                'avg_cost_per_question': round(overall_stats.get('average_cost_per_question', 0), 6),
                'cost_per_token': round(overall_stats.get('total_cost', 0) / max(1, overall_stats.get('total_input_tokens', 0) + overall_stats.get('total_output_tokens', 0)), 8)
            }
            
            # Extract detailed results from the JSON
            detailed_results = detailed_data.get('detailed_results', [])
            
            # Process question-by-question results
            for i, question_result in enumerate(detailed_results):
                if i < len(comparison_data["question_by_question"]):
                    # Extract performance data
                    performance = question_result.get('performance', {})
                    
                    rag_result = {
                        "answer": question_result.get('answer', ''),
                        "contexts_count": question_result.get('contexts_count', 0),
                        "metrics": question_result.get('metrics', {}),
                        "input_tokens": performance.get('input_tokens', 0),
                        "output_tokens": performance.get('output_tokens', 0),
                        "cost": performance.get('total_cost', 0.0),
                        "execution_time": performance.get('execution_time', 0.0)
                    }
                    
                    # Calculate question-level average if not present
                    if 'average_score' not in rag_result["metrics"]:
                        metric_values = [v for v in rag_result["metrics"].values() if isinstance(v, (int, float))]
                        if metric_values:
                            q_avg = sum(metric_values) / len(metric_values)
                            rag_result["metrics"]["average_score"] = round(q_avg, 3)
                    
                    comparison_data["question_by_question"][i]["rag_results"][rag_type] = rag_result
            
            # Add to summary
            comparison_data["summary"][rag_type] = {
                "rag_name": get_rag_name(rag_type),
                "metrics": aggregated_metrics,
                "performance": performance_metrics
            }
            
        except Exception as e:
            print(f"Error processing {rag_type} results: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate best performing RAG for each metric
    if comparison_data["summary"]:
        best_performers = {}
        
        # RAGAS metrics (higher is better)
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'overall_average']:
            best_score = 0
            best_rag = None
            for rag_type, data in comparison_data["summary"].items():
                score = data["metrics"].get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_rag = rag_type
            
            if best_rag:
                best_performers[metric] = {
                    "rag_type": best_rag,
                    "rag_name": get_rag_name(best_rag),
                    "score": round(best_score, 3)
                }
        
        # Performance metrics (lower is better for time and cost)
        performance_metrics_to_check = ['avg_execution_time', 'total_cost', 'total_tokens']
        for metric in performance_metrics_to_check:
            best_score = float('inf')
            best_rag = None
            for rag_type, data in comparison_data["summary"].items():
                score = data["performance"].get(metric, float('inf'))
                if score < best_score:
                    best_score = score
                    best_rag = rag_type
            
            if best_rag:
                best_performers[f"best_{metric}"] = {
                    "rag_type": best_rag,
                    "rag_name": get_rag_name(best_rag),
                    "score": round(best_score, 6) if 'cost' in metric else round(best_score, 3)
                }
        
        comparison_data["best_performers"] = best_performers
    
    # Save comparison file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        print(f"\nComparison results saved to: {filepath}")
        print(f"This file contains:")
        print(f"  - Aggregated metrics for all {len(all_results)} RAGs")
        print(f"  - Question-by-question comparison")
        print(f"  - Best performer analysis")
    except Exception as e:
        print(f"Error saving comparison results: {e}")


def get_rag_name(rag_type: str) -> str:
    """Get descriptive name for RAG type"""
    names = {
        "simple": "Simple Semantic RAG",
        "hyde": "HyDE RAG (Hypothetical Documents)",
        "rewriter": "Rewriter RAG (Multi-Query)",
        "hybrid": "Hybrid RAG (BM25 + Semantic)"
    }
    return names.get(rag_type, rag_type)


def main():
    """Main function to execute evaluation"""
    import sys
    
    # Detect flags
    export_analysis = "--export" in sys.argv or "-e" in sys.argv
    debug = "--debug" in sys.argv or "-d" in sys.argv
    
    # Clean arguments from flags
    args = [arg for arg in sys.argv[1:] if arg not in ["--export", "-e", "--debug", "-d"]]
    
    if len(args) > 0:
        rag_type = args[0].lower()
        if rag_type == "rewriter":
            return evaluate_rewriter_rag(export_analysis=export_analysis, debug=debug)
        elif rag_type == "hybrid":
            return evaluate_hybrid_rag(export_analysis=export_analysis, debug=debug)
        elif rag_type == "hyde":
            return evaluate_hyde_rag(export_analysis=export_analysis, debug=debug)
        elif rag_type == "simple":
            return evaluate_simple_rag(export_analysis=export_analysis, debug=debug)
        elif rag_type == "both":
            return evaluate_both_rags(export_analysis=export_analysis, debug=debug)
        elif rag_type == "all":
            return evaluate_all_rags(export_analysis=export_analysis, debug=debug)
        else:
            print("Invalid RAG type. Use: 'rewriter', 'hybrid', 'hyde', 'simple', 'both', or 'all'")
            return
    
    # Default: show usage
    print("RAGAS Evaluator - Professional RAG Evaluation")
    print("Available RAG types:")
    print("  - rewriter: Multi-Query Rewriter RAG")
    print("  - hybrid: Hybrid RAG (BM25 + Semantic)")
    print("  - hyde: HyDE RAG (Hypothetical Documents)")
    print("  - simple: Simple Semantic RAG")
    print("  - both: Evaluate original two RAGs (rewriter + hybrid)")
    print("  - all: Evaluate all 4 RAG systems")
    print("\nUsage: python ragas_evaluator.py [type] [--export] [--debug]")
    print("Examples:")
    print("  python ragas_evaluator.py simple")
    print("  python ragas_evaluator.py all --export")
    print("  python ragas_evaluator.py hyde --debug")
    return evaluate_rewriter_rag(export_analysis=export_analysis, debug=debug)


if __name__ == "__main__":
    main()
