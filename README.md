# RAG Benchmark - Comparación de Implementaciones RAG

Este proyecto implementa y evalúa tres tipos diferentes de sistemas RAG (Retrieval-Augmented Generation):

## 🎯 Tipos de RAG Implementados

### 1. **Graph RAG**
- Extrae entidades y relaciones de los documentos
- Construye un grafo de conocimiento
- Responde consultas navegando el grafo
- Ideal para preguntas que requieren múltiples saltos de razonamiento

### 2. **Rewrite RAG**
- Reescribe consultas de múltiples formas
- Mejora la recuperación con diferentes versiones de la pregunta
- Fusiona resultados de múltiples búsquedas
- Ideal para consultas ambiguas o complejas

### 3. **Hybrid RAG**
- Combina Graph RAG y Rewrite RAG
- Fusiona resultados usando diferentes estrategias
- Aprovecha las fortalezas de ambos enfoques
- Ideal para casos de uso diversos

## 🏗️ Estructura del Proyecto

```
RAG-Benchmark/
├── src/
│   ├── rags/                    # Implementaciones RAG
│   │   ├── graph_rag/          # Graph RAG
│   │   ├── rewrite_rag/        # Rewrite RAG
│   │   └── hybrid_rag/         # Hybrid RAG
│   ├── common/                  # Utilidades compartidas
│   ├── evaluation/              # Sistema de evaluación
│   └── utils/                   # Utilidades generales
├── Data/
│   ├── raw/                     # Documentos originales
│   ├── processed/               # Documentos procesados
│   ├── embeddings/              # Embeddings vectoriales
│   ├── graphs/                  # Datos de grafos
│   └── queries/                 # Consultas de prueba
├── config/                      # Configuraciones
├── tests/                       # Tests unitarios
├── scripts/                     # Scripts de ejecución
├── docs/                        # Documentación
└── results/                     # Resultados de evaluación
```

## 🚀 Configuración Inicial

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tus API keys
```

### 3. Configurar Neo4j (para Graph RAG)

```bash
# Docker
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# O instalar localmente
# https://neo4j.com/download/
```

## 🧪 Ejecutar Benchmark

### Ejecución Completa

```bash
cd scripts
python run_benchmark.py
```

### Ejecución Individual

```python
import asyncio
from src.rags.graph_rag import GraphRAG

async def test_graph_rag():
    config = {...}  # Tu configuración
    rag = GraphRAG(config)
    await rag.initialize()
    
    # Añadir documentos
    await rag.add_documents(["Documento de ejemplo..."])
    
    # Hacer consulta
    response = await rag.query("¿Qué dice el documento?")
    print(response.answer)

asyncio.run(test_graph_rag())
```

## 📊 Métricas de Evaluación

El sistema evalúa automáticamente:

- **BLEU Score**: Precisión de n-gramas
- **ROUGE Scores**: Recall de n-gramas y secuencias
- **BERT Score**: Similaridad semántica
- **Faithfulness**: Fidelidad al contexto
- **Relevance**: Relevancia de la respuesta
- **Tiempo de Respuesta**: Performance

## ⚙️ Configuración

Edita `config/config.yaml` para personalizar:

```yaml
# Configuración de LLM
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.1

# Configuración específica para Graph RAG
graph_rag:
  entity_extraction:
    model: "gpt-3.5-turbo"
    max_entities_per_chunk: 10

# Configuración para Rewrite RAG
rewrite_rag:
  query_rewrite:
    num_rewrites: 3
    temperature: 0.3

# Configuración para Hybrid RAG
hybrid_rag:
  weights:
    graph_score: 0.4
    vector_score: 0.6
  fusion_method: "weighted"
```

## 🧰 Uso Programático

### Ejemplo Básico

```python
from src.rags import GraphRAG, RewriteRAG, HybridRAG
import asyncio

async def main():
    # Configuración
    config = load_config("config/config.yaml")
    
    # Inicializar RAG
    rag = HybridRAG(config)
    await rag.initialize()
    
    # Añadir documentos
    documents = ["Tu contenido aquí..."]
    await rag.add_documents(documents)
    
    # Hacer consulta
    response = await rag.query("¿Cuál es la respuesta?")
    
    print(f"Respuesta: {response.answer}")
    print(f"Confianza: {response.confidence}")
    print(f"Fuentes: {len(response.sources)}")

asyncio.run(main())
```

### Evaluación Personalizada

```python
from src.evaluation import RAGEvaluator, RAGMetrics

# Configurar evaluador
metrics = RAGMetrics()
evaluator = RAGEvaluator(metrics)

# Evaluar consulta individual
result = await evaluator.evaluate_single(
    rag_system=rag,
    question="¿Qué es la IA?",
    ground_truth="La IA es...",
    context="Contexto relevante..."
)
```

## 📈 Resultados

Los resultados se guardan en `results/` con:

- Métricas individuales por consulta
- Promedios por sistema RAG
- Comparación entre sistemas
- Mejores performers por métrica

## 🔧 Extensibilidad

### Añadir Nueva Métrica

```python
from src.evaluation.metrics import RAGMetrics

class CustomMetrics(RAGMetrics):
    async def calculate_custom_metric(self, prediction, reference):
        # Tu lógica aquí
        return {"custom_score": score}
```

### Crear Nuevo Tipo de RAG

```python
from src.common.base_rag import BaseRAG, RAGResponse

class MyCustomRAG(BaseRAG):
    async def initialize(self):
        # Tu inicialización
        pass
    
    async def query(self, question):
        # Tu lógica de consulta
        return RAGResponse(answer="...", sources=[], confidence=0.8)
```

## 📚 Documentación Adicional

- [Guía de Graph RAG](docs/graph_rag.md)
- [Guía de Rewrite RAG](docs/rewrite_rag.md)
- [Guía de Hybrid RAG](docs/hybrid_rag.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature
3. Añade tests para nuevas funcionalidades
4. Envía un pull request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

- Issues: GitHub Issues
- Documentación: `/docs`
- Email: [tu-email@ejemplo.com]