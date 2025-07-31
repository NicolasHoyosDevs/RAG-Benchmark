"""
Procesador de documentos para sistemas RAG
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import re


class DocumentProcessor:
    """
    Procesador para cargar, limpiar y dividir documentos
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('data', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('data', {}).get('chunk_overlap', 200)
        self.supported_formats = config.get('data', {}).get('supported_formats', 
                                                           ['.txt', '.pdf', '.docx', '.md'])
    
    async def load_document(self, file_path: str) -> str:
        """Carga un documento desde archivo"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Formato no soportado: {path.suffix}")
        
        if path.suffix.lower() == '.txt' or path.suffix.lower() == '.md':
            return await self._load_text_file(path)
        elif path.suffix.lower() == '.pdf':
            return await self._load_pdf_file(path)
        elif path.suffix.lower() == '.docx':
            return await self._load_docx_file(path)
        else:
            raise ValueError(f"Formato no implementado: {path.suffix}")
    
    async def _load_text_file(self, path: Path) -> str:
        """Carga archivo de texto plano"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def _load_pdf_file(self, path: Path) -> str:
        """Carga archivo PDF"""
        try:
            import PyPDF2
            
            text = ""
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return text
        except ImportError:
            raise ImportError("Instala PyPDF2: pip install PyPDF2")
    
    async def _load_docx_file(self, path: Path) -> str:
        """Carga archivo DOCX"""
        try:
            from docx import Document
            
            doc = Document(path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
        except ImportError:
            raise ImportError("Instala python-docx: pip install python-docx")
    
    async def load_documents_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Carga todos los documentos de un directorio"""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory_path}")
        
        documents = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    content = await self.load_document(str(file_path))
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': str(file_path),
                            'filename': file_path.name,
                            'extension': file_path.suffix,
                            'size': file_path.stat().st_size
                        }
                    })
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")
        
        return documents
    
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        # Eliminar caracteres especiales y normalizar espacios
        text = re.sub(r'\\s+', ' ', text)  # Múltiples espacios a uno
        text = re.sub(r'\\n+', '\\n', text)  # Múltiples saltos de línea a uno
        text = text.strip()
        
        # Eliminar caracteres de control
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\\n\\t')
        
        return text
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Divide el texto en chunks con overlap"""
        if metadata is None:
            metadata = {}
        
        # Limpiar texto
        clean_text = self.clean_text(text)
        
        # Dividir en chunks
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(clean_text):
            # Determinar fin del chunk
            end = start + self.chunk_size
            
            # Si no es el último chunk, buscar un punto de corte natural
            if end < len(clean_text):
                # Buscar punto, salto de línea o espacio más cercano
                for delimiter in ['.\\n', '\\n', '. ', ' ']:
                    last_delimiter = clean_text.rfind(delimiter, start, end)
                    if last_delimiter != -1:
                        end = last_delimiter + len(delimiter)
                        break
            
            # Extraer chunk
            chunk_text = clean_text[start:end].strip()
            
            if chunk_text:  # Solo añadir chunks no vacíos
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': end,
                    'chunk_size': len(chunk_text)
                })
                
                chunks.append({
                    'content': chunk_text,
                    'metadata': chunk_metadata
                })
                
                chunk_id += 1
            
            # Calcular siguiente posición con overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    async def process_documents(self, documents: List[str], 
                              metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Procesa una lista de documentos y los divide en chunks"""
        all_chunks = []
        
        for i, document in enumerate(documents):
            doc_metadata = metadata_list[i] if metadata_list else {'document_id': i}
            
            # Dividir documento en chunks
            chunks = self.chunk_text(document, doc_metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_document_stats(self, documents: List[str]) -> Dict[str, Any]:
        """Obtiene estadísticas de los documentos"""
        total_chars = sum(len(doc) for doc in documents)
        total_words = sum(len(doc.split()) for doc in documents)
        
        # Estimar número de chunks
        estimated_chunks = 0
        for doc in documents:
            doc_chunks = len(self.chunk_text(doc))
            estimated_chunks += doc_chunks
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_document_length': total_chars / len(documents) if documents else 0,
            'estimated_chunks': estimated_chunks,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }