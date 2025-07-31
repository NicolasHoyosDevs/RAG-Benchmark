"""
Hybrid RAG Implementation
RAG híbrido que combina las fortalezas de Graph RAG y Rewrite RAG
"""

from typing import List, Dict, Any, Optional
from ...common.base_rag import BaseRAG, RAGResponse
from ..graph_rag.graph_rag import GraphRAG
from ..rewrite_rag.rewrite_rag import RewriteRAG


class HybridRAG(BaseRAG):
    """
    Implementación híbrida que combina Graph RAG y Rewrite RAG.
    Fusiona resultados de ambos enfoques para obtener mejor rendimiento.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.graph_rag = GraphRAG(config)
        self.rewrite_rag = RewriteRAG(config)
        self.fusion_method = config.get('hybrid_rag', {}).get('fusion_method', 'weighted')
        self.weights = config.get('hybrid_rag', {}).get('weights', {
            'graph_score': 0.4,
            'vector_score': 0.6
        })
        
    async def initialize(self) -> None:
        """Inicializa ambos sistemas RAG"""
        await self.graph_rag.initialize()
        await self.rewrite_rag.initialize()
        self.is_initialized = True
    
    async def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Añade documentos a ambos sistemas RAG
        """
        # Añadir a Graph RAG
        await self.graph_rag.add_documents(documents, metadata)
        
        # Añadir a Rewrite RAG
        await self.rewrite_rag.add_documents(documents, metadata)
    
    async def query(self, question: str, **kwargs) -> RAGResponse:
        """
        Procesa una consulta usando ambos enfoques y fusiona los resultados
        """
        # Obtener respuestas de ambos sistemas
        graph_response = await self.graph_rag.query(question, **kwargs)
        rewrite_response = await self.rewrite_rag.query(question, **kwargs)
        
        # Fusionar resultados
        fused_answer = await self._fuse_answers(
            question, graph_response, rewrite_response
        )
        
        # Combinar fuentes
        combined_sources = self._combine_sources(
            graph_response.sources, rewrite_response.sources
        )
        
        # Calcular confianza combinada
        combined_confidence = self._calculate_combined_confidence(
            graph_response.confidence, rewrite_response.confidence
        )
        
        return RAGResponse(
            answer=fused_answer,
            sources=combined_sources,
            confidence=combined_confidence,
            metadata={
                "fusion_method": self.fusion_method,
                "graph_metadata": graph_response.metadata,
                "rewrite_metadata": rewrite_response.metadata,
                "weights": self.weights
            }
        )
    
    async def _fuse_answers(self, question: str, graph_response: RAGResponse, 
                          rewrite_response: RAGResponse) -> str:
        """
        Fusiona las respuestas de ambos sistemas RAG
        """
        if self.fusion_method == "weighted":
            return await self._weighted_fusion(question, graph_response, rewrite_response)
        elif self.fusion_method == "rrf":
            return await self._reciprocal_rank_fusion(question, graph_response, rewrite_response)
        elif self.fusion_method == "linear_combination":
            return await self._linear_combination(question, graph_response, rewrite_response)
        else:
            # Por defecto, usar fusión ponderada
            return await self._weighted_fusion(question, graph_response, rewrite_response)
    
    async def _weighted_fusion(self, question: str, graph_response: RAGResponse, 
                             rewrite_response: RAGResponse) -> str:
        """
        Fusión ponderada basada en los pesos configurados
        """
        graph_weight = self.weights.get('graph_score', 0.4)
        vector_weight = self.weights.get('vector_score', 0.6)
        
        # TODO: Implementar fusión inteligente usando LLM
        # Por ahora, combinar las respuestas de forma simple
        if graph_response.confidence * graph_weight > rewrite_response.confidence * vector_weight:
            primary_answer = graph_response.answer
            secondary_answer = rewrite_response.answer
        else:
            primary_answer = rewrite_response.answer
            secondary_answer = graph_response.answer
        
        # Generar respuesta fusionada
        fused_answer = f"{primary_answer}\n\nInformación adicional: {secondary_answer}"
        
        return fused_answer
    
    async def _reciprocal_rank_fusion(self, question: str, graph_response: RAGResponse, 
                                    rewrite_response: RAGResponse) -> str:
        """
        Fusión usando Reciprocal Rank Fusion (RRF)
        """
        # TODO: Implementar RRF para combinar resultados
        # Por ahora, usar fusión simple
        return await self._weighted_fusion(question, graph_response, rewrite_response)
    
    async def _linear_combination(self, question: str, graph_response: RAGResponse, 
                                rewrite_response: RAGResponse) -> str:
        """
        Combinación lineal de las respuestas
        """
        # TODO: Implementar combinación lineal inteligente
        # Por ahora, usar fusión simple
        return await self._weighted_fusion(question, graph_response, rewrite_response)
    
    def _combine_sources(self, graph_sources: List[Dict[str, Any]], 
                        rewrite_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combina las fuentes de ambos sistemas
        """
        combined_sources = []
        
        # Añadir fuentes del grafo
        for source in graph_sources:
            source['origin'] = 'graph_rag'
            combined_sources.append(source)
        
        # Añadir fuentes del rewrite
        for source in rewrite_sources:
            source['origin'] = 'rewrite_rag'
            combined_sources.append(source)
        
        # Ordenar por confianza/score
        combined_sources.sort(
            key=lambda x: x.get('confidence', x.get('score', 0)), 
            reverse=True
        )
        
        return combined_sources
    
    def _calculate_combined_confidence(self, graph_confidence: float, 
                                     rewrite_confidence: float) -> float:
        """
        Calcula la confianza combinada
        """
        graph_weight = self.weights.get('graph_score', 0.4)
        vector_weight = self.weights.get('vector_score', 0.6)
        
        return (graph_confidence * graph_weight + 
                rewrite_confidence * vector_weight)
    
    async def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene documentos relevantes de ambos sistemas y los fusiona
        """
        # Obtener documentos de ambos sistemas
        graph_docs = await self.graph_rag.get_relevant_documents(query, top_k)
        rewrite_docs = await self.rewrite_rag.get_relevant_documents(query, top_k)
        
        # Combinar y eliminar duplicados
        all_docs = graph_docs + rewrite_docs
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.get('content', ''))
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                doc['origin'] = 'graph_rag' if doc in graph_docs else 'rewrite_rag'
                unique_docs.append(doc)
        
        # Ordenar por score y retornar top-k
        unique_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_docs[:top_k]
    
    async def cleanup(self) -> None:
        """Limpia recursos de ambos sistemas"""
        await self.graph_rag.cleanup()
        await self.rewrite_rag.cleanup()
        await super().cleanup()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas combinadas"""
        base_stats = super().get_stats()
        base_stats.update({
            "graph_rag_stats": self.graph_rag.get_stats(),
            "rewrite_rag_stats": self.rewrite_rag.get_stats(),
            "fusion_method": self.fusion_method,
            "weights": self.weights
        })
        return base_stats