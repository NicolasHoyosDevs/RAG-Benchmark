
import os
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any
# Cargar configuración
# Construir ruta absoluta al archivo .env en el directorio raíz
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verificar que la API key se haya cargado
if not OPENAI_API_KEY:
    print(f"Error: No se pudo cargar OPENAI_API_KEY desde {ENV_PATH}")
    print(" Verifica que el archivo .env existe y contiene: OPENAI_API_KEY=tu_clave")
    exit(1)
else:
    print(f"API Key cargada correctamente desde: {ENV_PATH}")

# Configuración de ChromaDB (basada en los archivos existentes)
SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIRECTORY = SCRIPT_DIR.parent / "Data" / "embeddings" / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"

# --- Token Counter y Cost Calculator ---
class TokenUsageTracker:
    """Clase para rastrear el uso de tokens y calcular costos aproximados"""
    
    def __init__(self):
        self.reset_session()
        
        # Precios por 1K tokens (aproximados, actualiza según OpenAI)
        self.prices = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "text-embedding-3-small": {"input": 0.00002, "output": 0}
        }
        
        # Inicializar tokenizers
        try:
            self.tokenizer_gpt35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.tokenizer_gpt4 = tiktoken.encoding_for_model("gpt-4o")
            self.tokenizer_embedding = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception as e:
            print(f"⚠️ Warning: No se pudo inicializar tokenizers: {e}")
            self.tokenizer_gpt35 = None
            self.tokenizer_gpt4 = None
            self.tokenizer_embedding = None
    
    def reset_session(self):
        """Reinicia las estadísticas de la sesión"""
        self.session_stats = {
            "rewriter_queries": 0,
            "rewriter_input_tokens": 0,
            "rewriter_output_tokens": 0,
            "qa_input_tokens": 0,
            "qa_output_tokens": 0,
            "embedding_tokens": 0,
            "total_cost": 0.0,
            "start_time": datetime.now()
        }
    
    def count_tokens(self, text: str, model: str) -> int:
        """Cuenta tokens para un texto y modelo específico"""
        if not text:
            return 0
            
        try:
            if model in ["gpt-3.5-turbo"] and self.tokenizer_gpt35:
                return len(self.tokenizer_gpt35.encode(text))
            elif model in ["gpt-4o"] and self.tokenizer_gpt4:
                return len(self.tokenizer_gpt4.encode(text))
            elif model in ["text-embedding-3-small"] and self.tokenizer_embedding:
                return len(self.tokenizer_embedding.encode(text))
            else:
                # Estimación aproximada: ~4 caracteres por token
                return len(text) // 4
        except Exception:
            # Fallback: estimación aproximada
            return len(text) // 4
    
    def log_rewriter_usage(self, input_text: str, output_text: str):
        """Registra el uso de tokens del rewriter"""
        input_tokens = self.count_tokens(input_text, "gpt-3.5-turbo")
        output_tokens = self.count_tokens(output_text, "gpt-3.5-turbo")
        
        self.session_stats["rewriter_queries"] += 1
        self.session_stats["rewriter_input_tokens"] += input_tokens
        self.session_stats["rewriter_output_tokens"] += output_tokens
        
        # Calcular costo
        input_cost = (input_tokens / 1000) * self.prices["gpt-3.5-turbo"]["input"]
        output_cost = (output_tokens / 1000) * self.prices["gpt-3.5-turbo"]["output"]
        self.session_stats["total_cost"] += input_cost + output_cost
        
        return input_tokens, output_tokens
    
    def log_qa_usage(self, input_text: str, output_text: str):
        """Registra el uso de tokens del QA"""
        input_tokens = self.count_tokens(input_text, "gpt-4o")
        output_tokens = self.count_tokens(output_text, "gpt-4o")
        
        self.session_stats["qa_input_tokens"] += input_tokens
        self.session_stats["qa_output_tokens"] += output_tokens
        
        # Calcular costo
        input_cost = (input_tokens / 1000) * self.prices["gpt-4o"]["input"]
        output_cost = (output_tokens / 1000) * self.prices["gpt-4o"]["output"]
        self.session_stats["total_cost"] += input_cost + output_cost
        
        return input_tokens, output_tokens
    
    def log_embedding_usage(self, text: str):
        """Registra el uso de tokens de embeddings"""
        tokens = self.count_tokens(text, "text-embedding-3-small")
        self.session_stats["embedding_tokens"] += tokens
        
        # Calcular costo
        cost = (tokens / 1000) * self.prices["text-embedding-3-small"]["input"]
        self.session_stats["total_cost"] += cost
        
        return tokens
    
    def get_session_summary(self) -> str:
        """Retorna un resumen de la sesión actual"""
        duration = datetime.now() - self.session_stats["start_time"]
        total_tokens = (
            self.session_stats["rewriter_input_tokens"] + 
            self.session_stats["rewriter_output_tokens"] +
            self.session_stats["qa_input_tokens"] + 
            self.session_stats["qa_output_tokens"] +
            self.session_stats["embedding_tokens"]
        )
        
        return f"""
📊 === RESUMEN DE TOKENS DE LA SESIÓN ===
⏱️ Duración: {str(duration).split('.')[0]}
🔄 Consultas reescritas: {self.session_stats["rewriter_queries"]}

🤖 GPT-3.5-turbo (Rewriter):
   📥 Input tokens: {self.session_stats["rewriter_input_tokens"]:,}
   📤 Output tokens: {self.session_stats["rewriter_output_tokens"]:,}

🧠 GPT-4o (QA):
   📥 Input tokens: {self.session_stats["qa_input_tokens"]:,}
   📤 Output tokens: {self.session_stats["qa_output_tokens"]:,}

🔍 Embeddings:
   📄 Tokens: {self.session_stats["embedding_tokens"]:,}

💰 Costo total aproximado: ${self.session_stats["total_cost"]:.4f}
🎯 Total tokens usados: {total_tokens:,}
"""

# Inicializar tracker global
token_tracker = TokenUsageTracker()

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
            print(f"✅ ChromaDB cargada: {self.db._collection.count()} documentos")
        except Exception as e:
            print(f"❌ Error al cargar ChromaDB: {e}")
            self.db = None
    
    def invoke(self, query: str) -> List[tuple]:
        """Realiza búsqueda por similitud con scores"""
        if not self.db:
            print("❌ Base de datos no disponible")
            return []
        
        try:
            # Búsqueda por similitud con scores en ChromaDB
            results_with_scores = self.db.similarity_search_with_score(query, k=self.k)
            
            # Convertir distancias a similaridades y aplicar filtrado suave
            filtered_results = []
            for doc, distance in results_with_scores:
                # ChromaDB usa distancia (menor = más similar), convertir a similaridad
                similarity = max(0.0, 1.0 - distance)  # Asegurar que no sea negativo
                
                # Filtrado muy permisivo - solo eliminar documentos realmente irrelevantes
                if similarity >= self.score_threshold:
                    filtered_results.append((doc, similarity))
            
            print(f"📄 Documentos encontrados: {len(results_with_scores)}, filtrados por score: {len(filtered_results)}")
            
            # Mostrar scores para debug (distancia -> similaridad)
            if results_with_scores:
                print(f"   📊 Distancias originales: {[f'{dist:.3f}' for _, dist in results_with_scores[:3]]}")
            if filtered_results:
                print(f"   📊 Similaridades convertidas: {[f'{sim:.3f}' for _, sim in filtered_results[:3]]}")
            
            return filtered_results
            
        except Exception as e:
            print(f"❌ Error en búsqueda ChromaDB: {e}")
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

llm_rewriter = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

def contextual_retriever(question: str, max_final_docs: int = 8) -> List:
    """Retriever que reescribe consultas múltiples veces para mejor recuperación con filtrado por relevancia"""
    print(f"🔄 Query original: '{question}'")
    
    # Generar múltiples reescrituras de la consulta
    rewritten_queries = []
    
    for i, prompt in enumerate(REPHRASE_PROMPTS, 1):
        llm_chain_rewriter = LLMChain(llm=llm_rewriter, prompt=prompt)
        
        # Preparar input para token tracking
        input_text = prompt.format(question=question)
        
        rewritten_result = llm_chain_rewriter.invoke({
            "question": question
        })
        rewritten_query = rewritten_result["text"].strip()
        rewritten_queries.append(rewritten_query)
        
        # Registrar uso de tokens
        input_tokens, output_tokens = token_tracker.log_rewriter_usage(input_text, rewritten_query)
        print(f"🔄 Query reescrita {i}: '{rewritten_query}'")
        print(f"   💰 Tokens: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
    
    # Buscar documentos con cada query reescrita
    all_docs_with_scores = []
    doc_ids_seen = set()  # Para evitar duplicados
    
    for i, query in enumerate(rewritten_queries, 1):
        print(f"\n🔍 Buscando con query {i}...")
        
        # Registrar tokens de embedding
        embedding_tokens = token_tracker.log_embedding_usage(query)
        print(f"   🔍 Embedding tokens: {embedding_tokens}")
        
        docs_with_scores = base_retriever.invoke(query)
        
        # Filtrar duplicados y agregar peso por posición de la query
        for doc, score in docs_with_scores:
            # Usar los primeros 100 caracteres como ID único
            doc_id = doc.page_content[:100]
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                
                # Penalizar ligeramente documentos de queries posteriores
                query_weight = 1.0 - (i - 1) * 0.05  # Primera query: 1.0, segunda: 0.95, tercera: 0.90
                weighted_score = score * query_weight
                
                all_docs_with_scores.append((doc, weighted_score, f"query_{i}"))
        
        print(f"📄 Documentos únicos encontrados con query {i}: {len(docs_with_scores)}")
    
    # Ordenar por score de relevancia (mayor a menor)
    all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Seleccionar los top-k más relevantes
    if len(all_docs_with_scores) > max_final_docs:
        selected_docs = all_docs_with_scores[:max_final_docs]
        print(f"🎯 Seleccionando top {max_final_docs} documentos más relevantes")
        
        # Mostrar scores de los seleccionados
        print(f"   📊 Top scores: {[f'{score:.3f}({source})' for _, score, source in selected_docs[:5]]}")
    else:
        selected_docs = all_docs_with_scores
        print(f"✅ Usando todos los {len(selected_docs)} documentos encontrados")
    
    # Extraer solo los documentos (sin scores)
    final_docs = [doc for doc, score, source in selected_docs]
    
    print(f"\n📄 Total documentos finales: {len(final_docs)}")
    
    # Debug: Mostrar qué documentos se seleccionaron
    if final_docs:
        for i, doc in enumerate(final_docs[:3]):
            print(f"📄 Doc {i+1}: {doc.page_content[:100]}...")
    else:
        print("❌ No se encontraron documentos relevantes")
    
    return final_docs

def format_docs(docs):
    """Formatea documentos para el contexto"""
    return "\n\n".join(
        f"=== DOCUMENTO {i+1} (Relevancia: {'Alta' if i < 2 else 'Media' if i < 4 else 'Baja'}) ===\n"
        f"Fuente: {doc.metadata.get('source', 'N/A')}\n"
        f"Contenido: {doc.page_content}" 
        for i, doc in enumerate(docs)
    )

# Template para respuesta final (adaptado al dominio de embarazo)
qa_template = """
Eres un especialista médico experto en embarazo y parto. Analiza ÚNICAMENTE el contexto proporcionado y responde la pregunta de manera precisa y directa.

REGLAS ESTRICTAS:
1. Responde DIRECTAMENTE la pregunta usando SOLO la información del contexto
2. PRIORIZA los DOCUMENTOS 1 y 2 que tienen la mayor relevancia para la pregunta
3. NO añadas conocimiento externo que no esté en el contexto
4. Si hay datos específicos (dosis, números, protocolos), inclúyelos exactamente como aparecen

ORDEN DE PRIORIDAD:
- Documentos 1-2: ALTA relevancia - úsalos como fuente principal
- Documentos 3-4: MEDIA relevancia - úsalos como complemento
- Documentos 5+: BAJA relevancia - úsalos solo si necesario

FORMATO DE RESPUESTA:
1. Respuesta directa a la pregunta (basándote principalmente en Documentos 1-2)
2. Detalles específicos del contexto
3. Información relacionada relevante (si aplica)
4. Limitaciones de la información disponible (solo si las hay)

IMPORTANTE: Esta información es para fines educativos y no reemplaza la consulta médica profesional.

Contexto de guías médicas (ordenado por relevancia):
{context}

Pregunta: {question}

Respuesta médica basada en el contexto:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

def create_conversational_rag():
    def tracked_qa_chain(inputs):
        """Wrapper que rastrear tokens del QA"""
        context = inputs["context"]
        question = inputs["question"]
        
        # Preparar el prompt completo para contar tokens
        full_prompt = qa_prompt.format(context=context, question=question)
        
        # Generar respuesta
        response = qa_llm.invoke(qa_prompt.format_messages(context=context, question=question))
        response_text = response.content
        
        # Registrar uso de tokens
        input_tokens, output_tokens = token_tracker.log_qa_usage(full_prompt, response_text)
        print(f"\n💰 QA Tokens: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
        
        return response_text
    
    return (
        {
            "context": RunnableLambda(lambda x: contextual_retriever(x["question"])) | RunnableLambda(format_docs),
            "question": lambda x: x["question"]
        }
        | RunnableLambda(tracked_qa_chain)
    )

def query_for_evaluation(question: str) -> dict:
    """
    Función específica para evaluación con RAGAS.
    Retorna estructura completa: pregunta, respuesta, contextos y metadatos.
    
    Args:
        question (str): La pregunta a procesar
        
    Returns:
        dict: Estructura con question, answer, contexts, source_documents y metadata
    """
    print(f"🔍 Evaluando: {question}")
    
    # 1. Obtener contextos usando la reescritura múltiple
    retrieved_docs = contextual_retriever(question)
    
    # 2. Formatear contextos para la respuesta
    formatted_context = format_docs(retrieved_docs)
    
    # 3. Generar respuesta usando el LLM
    response = qa_llm.invoke(qa_prompt.format_messages(
        context=formatted_context, 
        question=question
    ))
    
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
            "rewriter_model": "gpt-3.5-turbo"
        }
    }

# Ejecución principal
if __name__ == "__main__":
    try:
        print("🤖 --- Sistema RAG Conversacional con Multi-Query Rewriter para Embarazo y Parto ---")
        print("    ✨ Características: Reescritura múltiple de consultas (3 variaciones) para mejor búsqueda")
        
        # Inicializar componentes
        rag_chain = create_conversational_rag()
        
        print("\n💡 Comandos especiales:")
        print("   - 'salir': Terminar sesión")
        print("   - 'stats': Ver estadísticas de tokens actuales")
        print("   - 'reset': Reiniciar contador de tokens")
        
        while (query := input("\n💬 Tu pregunta (o 'salir', 'stats', 'reset'): ")) != "salir":
            # Comandos especiales
            if query.lower() == "stats":
                print(token_tracker.get_session_summary())
                continue
            elif query.lower() == "reset":
                token_tracker.reset_session()
                print("🔄 Contador de tokens reiniciado")
                continue
            
            # Preparar inputs (simplificado sin historial)
            inputs = {
                "question": query
            }
            
            print("\n🔍 Procesando...")
            
            # Generar respuesta
            response = rag_chain.invoke(inputs)
            
            print(f"\n🤖 Respuesta:")
            print(response)
            
            # Mostrar resumen rápido de tokens después de cada consulta
            current_cost = token_tracker.session_stats["total_cost"]
            total_tokens = (
                token_tracker.session_stats["rewriter_input_tokens"] + 
                token_tracker.session_stats["rewriter_output_tokens"] +
                token_tracker.session_stats["qa_input_tokens"] + 
                token_tracker.session_stats["qa_output_tokens"] +
                token_tracker.session_stats["embedding_tokens"]
            )
            print(f"💰 Sesión actual: {total_tokens:,} tokens | ${current_cost:.4f}")
    
    finally:
        print("\n🔌 Sistema finalizado correctamente.")
        print(token_tracker.get_session_summary())