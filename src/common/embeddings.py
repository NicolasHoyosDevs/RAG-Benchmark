"""
Servicio de embeddings para sistemas RAG
"""

from typing import List, Dict, Any
import numpy as np


class EmbeddingService:
    """
    Servicio para generar y gestionar embeddings
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('embeddings', {}).get('model', 'text-embedding-ada-002')
        self.dimension = config.get('embeddings', {}).get('dimension', 1536)
        self.batch_size = config.get('embeddings', {}).get('batch_size', 100)
        self.client = None
        
    async def initialize(self) -> None:
        """Inicializa el servicio de embeddings"""
        if 'openai' in self.model_name.lower() or 'ada' in self.model_name.lower():
            await self._initialize_openai()
        else:
            await self._initialize_local()
    
    async def _initialize_openai(self) -> None:
        """Inicializa cliente OpenAI"""
        try:
            from openai import AsyncOpenAI
            import os
            
            self.client = AsyncOpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        except ImportError:
            raise ImportError("Instala openai: pip install openai")
    
    async def _initialize_local(self) -> None:
        """Inicializa modelo local"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.client = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError("Instala sentence-transformers: pip install sentence-transformers")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Genera embedding para un texto
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos
        """
        if 'openai' in self.model_name.lower() or 'ada' in self.model_name.lower():
            return await self._embed_texts_openai(texts)
        else:
            return await self._embed_texts_local(texts)
    
    async def _embed_texts_openai(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings usando OpenAI"""
        all_embeddings = []
        
        # Procesar en lotes
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _embed_texts_local(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings usando modelo local"""
        embeddings = self.client.encode(texts, batch_size=self.batch_size)
        return embeddings.tolist()
    
    def calculate_similarity(self, embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """
        Calcula similaridad coseno entre dos embeddings
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalizar vectores
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calcular similaridad coseno
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]],
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encuentra los embeddings más similares a la consulta
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Ordenar por similaridad descendente
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]