"""
Gestor de almacenes vectoriales para sistemas RAG
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path


class VectorStoreManager:
    """
    Gestor para diferentes tipos de almacenes vectoriales
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.store_type = config.get('vector_store_type', 'chroma')
        self.store = None
        
    async def initialize(self, store_type: str = None) -> None:
        """Inicializa el almacén vectorial"""
        if store_type:
            self.store_type = store_type
            
        if self.store_type == 'chroma':
            await self._initialize_chroma()
        elif self.store_type == 'faiss':
            await self._initialize_faiss()
        elif self.store_type == 'pinecone':
            await self._initialize_pinecone()
        else:
            raise ValueError(f"Tipo de almacén vectorial no soportado: {self.store_type}")
    
    async def _initialize_chroma(self) -> None:
        """Inicializa ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            chroma_config = self.config.get('vector_stores', {}).get('chroma', {})
            persist_directory = chroma_config.get('persist_directory', './Data/embeddings/chroma')
            collection_name = chroma_config.get('collection_name', 'rag_documents')
            
            # Crear directorio si no existe
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Inicializar cliente
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Obtener o crear colección
            try:
                self.store = self.client.get_collection(name=collection_name)
            except:
                self.store = self.client.create_collection(name=collection_name)
                
        except ImportError:
            raise ImportError("Instala chromadb: pip install chromadb")
    
    async def _initialize_faiss(self) -> None:
        """Inicializa FAISS"""
        try:
            import faiss
            import pickle
            
            faiss_config = self.config.get('vector_stores', {}).get('faiss', {})
            self.index_path = faiss_config.get('index_path', './Data/embeddings/faiss_index')
            self.dimension = self.config.get('embeddings', {}).get('dimension', 1536)
            
            # Crear directorio si no existe
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Cargar índice existente o crear nuevo
            index_file = f"{self.index_path}.index"
            metadata_file = f"{self.index_path}_metadata.pkl"
            
            if Path(index_file).exists():
                self.store = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.store = faiss.IndexFlatIP(self.dimension)  # Inner product index
                self.metadata = []
                
        except ImportError:
            raise ImportError("Instala faiss: pip install faiss-cpu")
    
    async def _initialize_pinecone(self) -> None:
        """Inicializa Pinecone"""
        try:
            import pinecone
            import os
            
            pinecone_config = self.config.get('vector_stores', {}).get('pinecone', {})
            
            # Inicializar Pinecone
            pinecone.init(
                api_key=os.getenv('PINECONE_API_KEY'),
                environment=pinecone_config.get('environment', 'us-west1-gcp')
            )
            
            index_name = pinecone_config.get('index_name', 'rag-benchmark')
            
            # Conectar a índice existente o crear nuevo
            if index_name in pinecone.list_indexes():
                self.store = pinecone.Index(index_name)
            else:
                pinecone.create_index(
                    name=index_name,
                    dimension=self.config.get('embeddings', {}).get('dimension', 1536),
                    metric='cosine'
                )
                self.store = pinecone.Index(index_name)
                
        except ImportError:
            raise ImportError("Instala pinecone: pip install pinecone-client")
    
    async def add_documents(self, documents: List[str], 
                          embeddings: List[List[float]],
                          metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Añade documentos al almacén vectorial"""
        if self.store_type == 'chroma':
            await self._add_documents_chroma(documents, embeddings, metadata)
        elif self.store_type == 'faiss':
            await self._add_documents_faiss(documents, embeddings, metadata)
        elif self.store_type == 'pinecone':
            await self._add_documents_pinecone(documents, embeddings, metadata)
    
    async def _add_documents_chroma(self, documents: List[str], 
                                  embeddings: List[List[float]],
                                  metadata: Optional[List[Dict[str, Any]]]) -> None:
        """Añade documentos a ChromaDB"""
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadata is None:
            metadata = [{"source": f"document_{i}"} for i in range(len(documents))]
        
        self.store.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    async def _add_documents_faiss(self, documents: List[str], 
                                 embeddings: List[List[float]],
                                 metadata: Optional[List[Dict[str, Any]]]) -> None:
        """Añade documentos a FAISS"""
        import numpy as np
        import pickle
        
        # Normalizar embeddings para usar con IndexFlatIP
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        # Añadir al índice
        self.store.add(embeddings_array)
        
        # Guardar metadata
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if metadata else {}
            doc_metadata['document'] = doc
            doc_metadata['id'] = len(self.metadata)
            self.metadata.append(doc_metadata)
        
        # Persistir índice y metadata
        faiss.write_index(self.store, f"{self.index_path}.index")
        with open(f"{self.index_path}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    async def _add_documents_pinecone(self, documents: List[str], 
                                    embeddings: List[List[float]],
                                    metadata: Optional[List[Dict[str, Any]]]) -> None:
        """Añade documentos a Pinecone"""
        vectors = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_metadata = metadata[i] if metadata else {}
            doc_metadata['document'] = doc
            
            vectors.append({
                'id': f"doc_{i}",
                'values': embedding,
                'metadata': doc_metadata
            })
        
        # Upsert en lotes
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.store.upsert(vectors=batch)
    
    async def search(self, query_embedding: List[float], 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca documentos similares"""
        if self.store_type == 'chroma':
            return await self._search_chroma(query_embedding, top_k)
        elif self.store_type == 'faiss':
            return await self._search_faiss(query_embedding, top_k)
        elif self.store_type == 'pinecone':
            return await self._search_pinecone(query_embedding, top_k)
    
    async def _search_chroma(self, query_embedding: List[float], 
                           top_k: int) -> List[Dict[str, Any]]:
        """Busca en ChromaDB"""
        results = self.store.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],  # Convertir distancia a score
                'id': results['ids'][0][i]
            })
        
        return documents
    
    async def _search_faiss(self, query_embedding: List[float], 
                          top_k: int) -> List[Dict[str, Any]]:
        """Busca en FAISS"""
        import numpy as np
        
        # Normalizar query
        query_array = np.array([query_embedding]).astype('float32')
        import faiss
        faiss.normalize_L2(query_array)
        
        # Buscar
        scores, indices = self.store.search(query_array, top_k)
        
        documents = []
        for i in range(len(indices[0])):
            if indices[0][i] != -1:  # -1 indica que no se encontró resultado
                metadata = self.metadata[indices[0][i]]
                documents.append({
                    'content': metadata['document'],
                    'metadata': metadata,
                    'score': float(scores[0][i]),
                    'id': metadata['id']
                })
        
        return documents
    
    async def _search_pinecone(self, query_embedding: List[float], 
                             top_k: int) -> List[Dict[str, Any]]:
        """Busca en Pinecone"""
        results = self.store.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        documents = []
        for match in results['matches']:
            documents.append({
                'content': match['metadata']['document'],
                'metadata': match['metadata'],
                'score': match['score'],
                'id': match['id']
            })
        
        return documents