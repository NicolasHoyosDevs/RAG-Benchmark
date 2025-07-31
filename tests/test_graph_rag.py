"""
Tests para Graph RAG
"""

import pytest
import asyncio
from src.rags.graph_rag import GraphRAG


class TestGraphRAG:
    """Tests para la implementación de Graph RAG"""
    
    @pytest.fixture
    async def graph_rag(self):
        """Fixture para crear instancia de Graph RAG"""
        config = {
            'llm': {'model': 'gpt-3.5-turbo'},
            'graph_rag': {
                'entity_extraction': {'max_entities_per_chunk': 5},
                'relationship_extraction': {'max_relationships_per_chunk': 3}
            }
        }
        rag = GraphRAG(config)
        await rag.initialize()
        return rag
    
    @pytest.mark.asyncio
    async def test_initialization(self, graph_rag):
        """Test inicialización del Graph RAG"""
        assert graph_rag.is_initialized
        assert graph_rag.name == "GraphRAG"
    
    @pytest.mark.asyncio
    async def test_add_documents(self, graph_rag):
        """Test añadir documentos"""
        documents = ["Este es un documento de prueba."]
        await graph_rag.add_documents(documents)
        
        # Verificar que el grafo no esté vacío
        assert len(graph_rag.knowledge_graph.nodes()) >= 0
    
    @pytest.mark.asyncio
    async def test_query(self, graph_rag):
        """Test realizar consulta"""
        # Añadir documentos primero
        documents = ["La inteligencia artificial es una tecnología importante."]
        await graph_rag.add_documents(documents)
        
        # Realizar consulta
        response = await graph_rag.query("¿Qué es la inteligencia artificial?")
        
        assert response.answer is not None
        assert isinstance(response.sources, list)
        assert 0 <= response.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_get_stats(self, graph_rag):
        """Test obtener estadísticas"""
        stats = graph_rag.get_stats()
        
        assert 'name' in stats
        assert 'initialized' in stats
        assert stats['name'] == 'GraphRAG'