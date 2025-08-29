import os
import time
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from typing import List, Dict, Any
# Cargar configuración
# Construir ruta absoluta al archivo .env en el directorio raíz
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verificar que la API key se haya cargado
if not OPENAI_API_KEY:
    print(f"Error: OPENAI_API_KEY not found in {ENV_PATH}")
    print("Please verify that the .env file exists and contains: OPENAI_API_KEY=your_key")
    exit(1)

# Configuración de ChromaDB (basada en los archivos existentes)
SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIRECTORY = SCRIPT_DIR.parent / "Data" / "embeddings" / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"

# --- Sistema de Tracking Simplificado ---


class SimpleTracker:
    """Sistema simple para rastrear tokens y costos de cada operacion"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia las estadisticas"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_embedding_tokens = 0
        self.total_cost = 0.0
        self.operations = []

    def add_operation(self, name: str, input_tokens: int, output_tokens: int, cost: float):
        """Agrega una operacion al tracking"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        operation = {
            "name": name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
        self.operations.append(operation)

    def add_embedding_operation(self, name: str, tokens: int, cost: float):
        """Agrega una operacion de embedding al tracking usando datos reales de OpenAI callback"""
        self.total_embedding_tokens += tokens
        self.total_cost += cost
        
        operation = {
            "name": name,
            "embedding_tokens": tokens,
            "cost": cost
        }
        self.operations.append(operation)

    def show_summary(self):
        """Muestra resumen total"""
        print(f"\nToken Usage Summary:")
        print(f"  Input tokens: {self.total_input_tokens}")
        print(f"  Output tokens: {self.total_output_tokens}")
        print(f"  Embedding tokens: {self.total_embedding_tokens}")
        print(f"  Total cost: ${self.total_cost:.6f}")


# Inicializar tracker global
tracker = SimpleTracker()

# --- Configuración del Retriever Base ---
base_retriever = None  # Se inicializará después de la clase ChromaRetriever

# --- Retriever ChromaDB ---


class ChromaRetriever:
    def __init__(self, db_directory: str, collection_name: str, k: int = 5, score_threshold: float = 0.05):
        self.db_directory = db_directory
        self.collection_name = collection_name
        self.k = k
        self.score_threshold = score_threshold  # Nuevo: threshold de similaridad

        # Inicializar embeddings (mismo modelo que create_embeddings.py)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )

        # Cargar base de datos
        self._load_db()

    def _load_db(self):
        """Carga la base de datos ChromaDB"""
        try:
            self.db = Chroma(
                persist_directory=str(self.db_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"Database loaded: {self.db._collection.count()} documents")
        except Exception as e:
            print(f"Error loading database: {e}")
            self.db = None

    def invoke(self, query: str) -> List[tuple]:
        """Realiza busqueda por similitud con scores"""
        if not self.db:
            print("Database not available")
            return []

        try:
            # Busqueda por similitud con scores en ChromaDB
            results_with_scores = self.db.similarity_search_with_score(query, k=self.k)

            # Convertir distancias a similaridades y aplicar filtrado
            filtered_results = []
            for doc, distance in results_with_scores:
                similarity = max(0.0, 1.0 - distance)

                if similarity >= self.score_threshold:
                    # Ensure doc is a proper Document object
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        filtered_results.append((doc, similarity))
                    else:
                        # If not a proper Document, create one
                        from langchain_core.documents import Document
                        if isinstance(doc, dict):
                            content = doc.get('page_content', doc.get('content', str(doc)))
                            metadata = doc.get('metadata', {})
                            proper_doc = Document(page_content=content, metadata=metadata)
                            filtered_results.append((proper_doc, similarity))

            return filtered_results

        except Exception as e:
            print(f"Database search error: {e}")
            return []


# --- Configuración del Retriever Base ---
base_retriever = ChromaRetriever(
    db_directory=DB_DIRECTORY,
    collection_name=COLLECTION_NAME,
    k=8,  # Aumentar k para tener más candidatos
    score_threshold=0.05  # Threshold muy permisivo (>5% similaridad)
)


# Configuración del rewriter con múltiples enfoques
REPHRASE_TEMPLATE_1 = """Reescribe esta pregunta para que sea una consulta independiente y específica sobre embarazo y parto.

Pregunta original: {question}

Instrucciones:
- Mantén el contexto médico/obstétrico si es relevante
- Sé específico y claro en términos médicos
- Enfócate en embarazo, parto, control prenatal, o salud materna
- Asegúrate de que la pregunta sea completa y autocontenida

Pregunta independiente:"""

REPHRASE_TEMPLATE_2 = """Reformula esta pregunta sobre embarazo y parto usando sinónimos y términos médicos alternativos.

Pregunta original: {question}

Instrucciones:
- Usa terminología médica precisa
- Incluye sinónimos y términos alternativos
- Mantén el significado pero cambia la formulación
- Enfócate en aspectos clínicos y obstétricos

Pregunta reformulada:"""

REPHRASE_TEMPLATE_3 = """Amplía esta pregunta para incluir aspectos relacionados y contexto adicional sobre embarazo y parto.

Pregunta base: {question}

Instrucciones:
- Amplía la pregunta para incluir aspectos relacionados
- Agrega contexto sobre complicaciones, prevención o cuidados
- Incluye posibles variaciones o casos especiales
- Mantén el foco en salud materna y perinatal

Pregunta ampliada:"""

# Crear los prompts para cada variación
REPHRASE_PROMPTS = [
    PromptTemplate.from_template(REPHRASE_TEMPLATE_1),
    PromptTemplate.from_template(REPHRASE_TEMPLATE_2),
    PromptTemplate.from_template(REPHRASE_TEMPLATE_3)
]

llm_rewriter = ChatOpenAI(model_name="gpt-3.5-turbo",
                          temperature=0.3, api_key=OPENAI_API_KEY)


def contextual_retriever(question: str, max_final_docs: int = 8) -> List:
    """Retriever que reescribe consultas multiples veces para mejor recuperacion con filtrado por relevancia"""
    
    # Generar multiples reescrituras de la consulta
    rewritten_queries = []

    for i, prompt in enumerate(REPHRASE_PROMPTS, 1):
        # Usar callback individual para cada reescritura
        with get_openai_callback() as cb:
            rewritten_query = (prompt | llm_rewriter | StrOutputParser()).invoke(
                {"question": question}).strip()

        rewritten_queries.append(rewritten_query)

        # Registrar la operacion de reescritura
        tracker.add_operation(
            f"Reescritura {i}",
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_cost
        )

    print(f"Generated {len(rewritten_queries)} query variations:")
    for i, query in enumerate(rewritten_queries, 1):
        print(f"  {i}. {query}")
    print()

    # Buscar documentos con cada query reescrita
    all_docs_with_scores = []
    doc_ids_seen = set()  # Para evitar duplicados

    for i, query in enumerate(rewritten_queries, 1):
        # Usar callback para capturar tokens reales de embedding
        with get_openai_callback() as cb:
            docs_with_scores = base_retriever.invoke(query)
        
        # Registrar solo tokens reales capturados por OpenAI callback
        if cb.total_tokens > 0:  # Para embeddings los tokens están en total_tokens
            tracker.add_embedding_operation(f"Embedding query {i}", cb.total_tokens, cb.total_cost)

        # Filtrar duplicados y agregar peso por posicion de la query
        for doc, score in docs_with_scores:
            # Usar los primeros 100 caracteres como ID unico
            doc_id = doc.page_content[:100]
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)

                # Penalizar ligeramente documentos de queries posteriores
                query_weight = 1.0 - (i - 1) * 0.05
                weighted_score = score * query_weight

                all_docs_with_scores.append(
                    (doc, weighted_score, f"query_{i}"))

    # Ordenar por score de relevancia (mayor a menor)
    all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Seleccionar los top-k mas relevantes
    if len(all_docs_with_scores) > max_final_docs:
        selected_docs = all_docs_with_scores[:max_final_docs]
    else:
        selected_docs = all_docs_with_scores

    # Extraer solo los documentos (sin scores)
    final_docs = [doc for doc, score, source in selected_docs]

    print(f"Retrieved {len(final_docs)} documents")

    return final_docs


def format_docs(docs):
    """Formatea documentos para el contexto"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Ensure we have a proper Document object
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'
            content = doc.page_content
        else:
            # Fallback for other formats
            source = 'N/A'
            content = str(doc)

        formatted_doc = f"=== DOCUMENTO {i+1} (Relevancia: {'Alta' if i < 2 else 'Media' if i < 4 else 'Baja'}) ===\n"
        formatted_doc += f"Fuente: {source}\n"
        formatted_doc += f"Contenido: {content}"
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


# Template para respuesta final (adaptado al dominio de embarazo)
qa_template = """
Eres un especialista medico experto en embarazo y parto. Analiza UNICAMENTE el contexto proporcionado y responde la pregunta de manera precisa y directa.

REGLAS ESTRICTAS:
1. Responde DIRECTAMENTE la pregunta usando SOLO la informacion del contexto
2. PRIORIZA los DOCUMENTOS 1 y 2 que tienen la mayor relevancia para la pregunta
3. NO añadidas conocimiento externo que no este en el contexto
4. Si hay datos especificos (dosis, numeros, protocolos), incluyelos exactamente como aparecen

ORDEN DE PRIORIDAD:
- Documentos 1-2: ALTA relevancia - usalos como fuente principal
- Documentos 3-4: MEDIA relevancia - usalos como complemento
- Documentos 5+: BAJA relevancia - usalos solo si necesario

FORMATO DE RESPUESTA:
1. Respuesta directa a la pregunta (basandote principalmente en Documentos 1-2)
2. Detalles especificos del contexto
3. Informacion relacionada relevante (si aplica)
4. Limitaciones de la informacion disponible (solo si las hay)

IMPORTANTE: Esta informacion es para fines educativos y no reemplaza la consulta medica profesional.

Contexto de guias medicas (ordenado por relevancia):
{context}

Pregunta: {question}

Respuesta medica basada en el contexto:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)


def create_conversational_rag():
    """Crear la cadena RAG conversacional con tracking detallado"""
    def tracked_qa_chain(inputs):
        """Wrapper que registra la respuesta final"""
        context = inputs["context"]
        question = inputs["question"]

        # Usar callback para la respuesta final
        with get_openai_callback() as cb:
            response = (qa_prompt | qa_llm | StrOutputParser()).invoke({
                "context": context,
                "question": question
            })

        # Registrar la operacion de respuesta
        tracker.add_operation(
            "Respuesta final",
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_cost
        )

        return response

    return (
        {
            "context": RunnableLambda(lambda x: contextual_retriever(x["question"])) | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(tracked_qa_chain)
    )


def query_for_evaluation(question: str) -> dict:
    """
    Funcion especifica para evaluacion con RAGAS.
    Retorna estructura completa: pregunta, respuesta, contextos y metadatos.

    Args:
        question (str): La pregunta a procesar

    Returns:
        dict: Estructura con question, answer, contexts, source_documents y metadata
    """
    print(f"Evaluating: {question[:60]}...")
    
    # Reiniciar tracker para esta evaluación específica
    tracker.reset()
    
    # Medir tiempo de ejecución
    import time
    start_time = time.time()

    # Tracking completo de embeddings + LLM
    with get_openai_callback() as cb:
        # 1. Obtener contextos usando la reescritura multiple (incluye embeddings)
        retrieved_docs = contextual_retriever(question)

        # 2. Formatear contextos para la respuesta
        formatted_context = format_docs(retrieved_docs)

        # 3. Generar respuesta usando el LLM
        response = qa_llm.invoke(qa_prompt.format_messages(
            context=formatted_context,
            question=question
        ))

    # Registrar la operacion de evaluacion
    tracker.add_operation(
        "Evaluacion RAGAS",
        cb.prompt_tokens,
        cb.completion_tokens,
        cb.total_cost
    )
    
    # Calcular tiempo total
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Evaluation completed in {execution_time:.2f}s")

    # 4. Preparar lista de contextos (contenido de documentos)
    contexts = [doc.page_content for doc in retrieved_docs]

    # 5. Retornar estructura completa para RAGAS
    return {
        "question": question,
        "answer": response.content,
        "contexts": contexts,  # Lista de strings con el contenido de los documentos
        "source_documents": retrieved_docs,  # Documentos completos con metadata
        "metadata": {
            "num_contexts": len(contexts),
            "retrieval_method": "multi_query_rewrite",
            "rewrite_count": 3,
            "llm_model": "gpt-4o",
            "rewriter_model": "gpt-3.5-turbo",
            
            # Métricas de rendimiento para export
            "execution_time": execution_time,
            "input_tokens": tracker.total_input_tokens,
            "output_tokens": tracker.total_output_tokens,
            # "embedding_tokens": 0,  # Comentado temporalmente
            "total_cost": tracker.total_cost,
            
            # Métricas legacy (mantener compatibilidad)
            "tokens_used": tracker.total_input_tokens + tracker.total_output_tokens,
        }
    }


# Ejecución principal
if __name__ == "__main__":
    try:
        print("Multi-Query RAG System for Medical Consultation")
        print("Pregnancy and Childbirth Knowledge Base")
        print("-" * 50)

        # Inicializar componentes
        rag_chain = create_conversational_rag()

        print("\nType 'exit' to end session")

        while (query := input("\nQuestion: ")) != "exit":
            print("Processing...")

            # Preparar inputs
            inputs = {"question": query}

            # Medir tiempo total del proceso
            start_time = time.time()
            response = rag_chain.invoke(inputs)
            end_time = time.time()

            print(f"\nResponse:")
            print(response)

            print(f"\nExecution time: {end_time - start_time:.2f}s")
            tracker.show_summary()

    finally:
        print("\nSession ended.")
        if tracker.total_cost > 0:
            tracker.show_summary()
