"""
Clase base abstracta para todas las implementaciones de RAG
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class RAGResponse(BaseModel):
    """Modelo para respuestas de RAG"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = {}


class BaseRAG(ABC):
    """Clase base para todas las implementaciones de RAG"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Inicializa el sistema RAG"""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Añade documentos al sistema RAG"""
        pass
    
    @abstractmethod
    async def query(self, question: str, **kwargs) -> RAGResponse:
        """Procesa una consulta y retorna una respuesta"""
        pass
    
    @abstractmethod
    async def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Obtiene documentos relevantes para una consulta"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema RAG"""
        return {
            "name": self.name,
            "initialized": self.is_initialized,
            "config": self.config
        }
    
    async def cleanup(self) -> None:
        """Limpia recursos del sistema RAG"""
        self.is_initialized = False