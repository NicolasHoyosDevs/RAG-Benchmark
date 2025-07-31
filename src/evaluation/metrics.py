"""
Métricas de evaluación para sistemas RAG
"""

from typing import List, Dict, Any, Optional
import asyncio


class RAGMetrics:
    """
    Clase para calcular diferentes métricas de evaluación para sistemas RAG
    """
    
    def __init__(self):
        self.available_metrics = [
            "bleu", "rouge", "bert_score", "faithfulness", "relevance"
        ]
    
    async def calculate_all_metrics(self, prediction: str, reference: str,
                                  context: Optional[str] = None,
                                  sources: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Calcula todas las métricas disponibles
        """
        metrics = {}
        
        # Métricas de similaridad textual
        metrics.update(await self.calculate_bleu(prediction, reference))
        metrics.update(await self.calculate_rouge(prediction, reference))
        metrics.update(await self.calculate_bert_score(prediction, reference))
        
        # Métricas específicas de RAG
        if context or sources:
            metrics.update(await self.calculate_faithfulness(prediction, context, sources))
        
        metrics.update(await self.calculate_relevance(prediction, reference))
        
        return metrics
    
    async def calculate_bleu(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calcula BLEU score
        """
        try:
            from sacrebleu import sentence_bleu
            
            # Tokenizar
            pred_tokens = prediction.lower().split()
            ref_tokens = [reference.lower().split()]
            
            # Calcular BLEU
            bleu_score = sentence_bleu(prediction, [reference]).score / 100.0
            
            return {"bleu_score": bleu_score}
        
        except ImportError:
            return {"bleu_score": 0.0}
    
    async def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calcula ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, prediction)
            
            return {
                "rouge_1": scores['rouge1'].fmeasure,
                "rouge_2": scores['rouge2'].fmeasure,
                "rouge_l": scores['rougeL'].fmeasure
            }
        
        except ImportError:
            return {
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0
            }
    
    async def calculate_bert_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calcula BERTScore
        """
        try:
            from bert_score import score
            
            # Calcular BERTScore
            P, R, F1 = score([prediction], [reference], lang="es", verbose=False)
            
            return {
                "bert_score_precision": P.item(),
                "bert_score_recall": R.item(),
                "bert_score_f1": F1.item()
            }
        
        except ImportError:
            return {
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0
            }
    
    async def calculate_faithfulness(self, prediction: str, 
                                   context: Optional[str] = None,
                                   sources: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Calcula faithfulness - qué tan fiel es la respuesta al contexto proporcionado
        """
        if not context and not sources:
            return {"faithfulness": 0.0}
        
        # Combinar contexto de fuentes si no hay contexto directo
        if not context and sources:
            context_parts = []
            for source in sources:
                if 'content' in source:
                    context_parts.append(source['content'])
            context = " ".join(context_parts)
        
        if not context:
            return {"faithfulness": 0.0}
        
        # Métrica simple: overlap de palabras entre predicción y contexto
        pred_words = set(prediction.lower().split())
        context_words = set(context.lower().split())
        
        if not pred_words:
            return {"faithfulness": 0.0}
        
        overlap = len(pred_words.intersection(context_words))
        faithfulness = overlap / len(pred_words)
        
        return {"faithfulness": min(faithfulness, 1.0)}
    
    async def calculate_relevance(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calcula relevancia - qué tan relevante es la respuesta a la pregunta
        """
        # Métrica simple basada en overlap semántico
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        if not pred_words or not ref_words:
            return {"relevance": 0.0}
        
        # Jaccard similarity
        intersection = len(pred_words.intersection(ref_words))
        union = len(pred_words.union(ref_words))
        
        relevance = intersection / union if union > 0 else 0.0
        
        return {"relevance": relevance}
    
    async def calculate_custom_metric(self, prediction: str, reference: str,
                                    metric_name: str, metric_function) -> Dict[str, float]:
        """
        Permite añadir métricas personalizadas
        """
        try:
            score = await metric_function(prediction, reference)
            return {metric_name: score}
        except Exception as e:
            print(f"Error calculando métrica personalizada {metric_name}: {e}")
            return {metric_name: 0.0}