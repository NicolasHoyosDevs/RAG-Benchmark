"""
Utilidades para análisis y exportación de resultados RAG con métricas RAGAS
Función completa para exportar análisis de evaluación con ground truth y scores
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import webbrowser

def export_ragas_analysis(evaluation_results, rag_name: str = "rag_system", open_browser: bool = True, performance_metadata: List[Dict] = None) -> Dict[str, Path]:
    """
    Exporta análisis completo de evaluación RAGAS con ground truth y métricas
    
    Args:
        evaluation_results: Resultados de evaluación RAGAS. Puede ser:
            - Dataset de RAGAS (retornado por evaluate())
            - Lista de diccionarios con estructura específica
        rag_name: Nombre del sistema RAG (para nombres de archivos)
        open_browser: Si abrir automáticamente el dashboard HTML
        
    Returns:
        Dict con rutas de todos los archivos generados
        
    Example:
        >>> results = [{"question": "test", "ground_truth": "expected", "answer": "actual", ...}, ...]
        >>> files = export_ragas_analysis(results, "rewriter_rag")
        >>> print(files['html'])  # Ruta al dashboard HTML
    """
    print(f"📊 Exportando análisis RAGAS completo para {rag_name}...")
    
    # === DETECTAR Y CONVERTIR FORMATO DE DATOS ===
    if hasattr(evaluation_results, 'to_pandas'):
        # Es un Dataset de RAGAS - convertir a lista de diccionarios
        print("🔄 Detectado Dataset de RAGAS, convirtiendo al formato de exportación...")
        try:
            df = evaluation_results.to_pandas()
            converted_results = []
            
            print(f"📋 Columnas disponibles en Dataset RAGAS: {df.columns.tolist()}")
            
            for i, row in df.iterrows():
                # Intentar extraer datos de diferentes posibles nombres de columnas
                question = row.get("question", "") or row.get("user_input", "") or row.get("query", "")
                ground_truth = row.get("ground_truth", "") or row.get("ground_truths", "") or row.get("reference", "")
                answer = row.get("answer", "") or row.get("response", "") or row.get("generated_answer", "")
                contexts = row.get("contexts", []) or row.get("retrieved_contexts", []) or []
                
                # Si contexts es string, convertir a lista
                if isinstance(contexts, str):
                    contexts = [contexts]
                
                result_item = {
                    "question": str(question),
                    "ground_truth": str(ground_truth), 
                    "answer": str(answer),
                    "contexts": list(contexts) if contexts else [],
                    
                    # Métricas RAGAS
                    "faithfulness": float(row.get("faithfulness", 0.0)),
                    "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
                    "context_precision": float(row.get("context_precision", 0.0)),
                    "context_recall": float(row.get("context_recall", 0.0)),
                    
                    # Placeholder para metadatos que se añadirán después
                    "metadata": {}
                }
                converted_results.append(result_item)
            
            evaluation_results = converted_results
            print(f"✅ Convertidos {len(evaluation_results)} resultados RAGAS para exportación")
            
            # Debug: mostrar una muestra para verificar la conversión
            if evaluation_results:
                sample = evaluation_results[0]
                print(f"🔍 Muestra de datos convertidos:")
                print(f"   - Question: '{sample['question'][:50]}...' ({len(sample['question'])} chars)")
                print(f"   - Answer: '{sample['answer'][:50]}...' ({len(sample['answer'])} chars)")
                print(f"   - Contexts: {len(sample['contexts'])} elementos")
            
        except Exception as e:
            print(f"❌ Error convirtiendo Dataset RAGAS: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    elif isinstance(evaluation_results, list):
        # Ya es una lista de diccionarios - usar directamente
        print(f"✅ Detectada lista de diccionarios ({len(evaluation_results)} elementos)")
    
    else:
        print(f"❌ Formato de datos no soportado: {type(evaluation_results)}")
        return {}
    
    # === COMBINAR CON METADATOS DE RENDIMIENTO ===
    if performance_metadata and len(performance_metadata) == len(evaluation_results):
        print(f"🔗 Combinando con metadatos de rendimiento ({len(performance_metadata)} elementos)...")
        for i, (result, perf_data) in enumerate(zip(evaluation_results, performance_metadata)):
            if result.get("question") == perf_data.get("question"):
                # Combinar metadatos de rendimiento
                result["metadata"] = {
                    "execution_time": perf_data.get("execution_time", 0.0),
                    "input_tokens": perf_data.get("input_tokens", 0),
                    "output_tokens": perf_data.get("output_tokens", 0),
                    "embedding_tokens": perf_data.get("embedding_tokens", 0),
                    "total_cost": perf_data.get("total_cost", 0.0)
                }
            else:
                print(f"⚠️ Discrepancia en pregunta {i+1}, usando metadatos vacíos")
                result["metadata"] = {
                    "execution_time": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "embedding_tokens": 0,
                    "total_cost": 0.0
                }
        print(f"✅ Metadatos de rendimiento combinados exitosamente")
    elif performance_metadata:
        print(f"⚠️ Número de metadatos ({len(performance_metadata)}) no coincide con resultados ({len(evaluation_results)})")
    else:
        print(f"ℹ️ No se proporcionaron metadatos de rendimiento, usando valores por defecto")
    
    # Crear directorio de salida
    output_dir = Path(__file__).parent / "ragas_analysis"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === PROCESAR DATOS RAGAS ===
    processed_data = []
    low_performance_data = []
    
    for i, result in enumerate(evaluation_results):
        # Extraer datos básicos
        question = result.get('question', '')
        ground_truth = result.get('ground_truth', '')
        answer = result.get('answer', '')
        contexts = result.get('contexts', [])
        
        # Extraer métricas RAGAS
        faithfulness = result.get('faithfulness', 0.0)
        answer_relevancy = result.get('answer_relevancy', 0.0)
        context_precision = result.get('context_precision', 0.0)
        context_recall = result.get('context_recall', 0.0)
        
        # Calcular métricas derivadas
        average_score = (faithfulness + answer_relevancy + context_precision + context_recall) / 4
        answer_length = len(answer)
        ground_truth_length = len(ground_truth)
        num_contexts = len(contexts)
        
        # Extraer métricas de rendimiento del sistema (si están disponibles)
        metadata = result.get('metadata', {})
        execution_time = metadata.get('execution_time', 0.0)
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        embedding_tokens = metadata.get('embedding_tokens', 0)
        total_cost = metadata.get('total_cost', 0.0)
        total_tokens = input_tokens + output_tokens + embedding_tokens
        
        # Fila procesada para exportación
        row = {
            'id': i + 1,
            'question': question[:150] + "..." if len(question) > 150 else question,
            'ground_truth': ground_truth[:150] + "..." if len(ground_truth) > 150 else ground_truth,
            'llm_answer': answer[:150] + "..." if len(answer) > 150 else answer,
            'question_length': len(question),
            'ground_truth_length': ground_truth_length,
            'answer_length': answer_length,
            'num_contexts': num_contexts,
            
            # Métricas RAGAS (redondeadas)
            'faithfulness': round(faithfulness, 3),
            'answer_relevancy': round(answer_relevancy, 3),
            'context_precision': round(context_precision, 3),
            'context_recall': round(context_recall, 3),
            'average_score': round(average_score, 3),
            
            # Métricas de rendimiento del sistema
            'execution_time_seconds': round(execution_time, 3),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'embedding_tokens': embedding_tokens,
            'total_tokens': total_tokens,
            'total_cost_usd': round(total_cost, 6),
            'cost_per_token': round(total_cost / max(1, total_tokens), 8) if total_tokens > 0 else 0.0,
            
            # Comparación de longitudes
            'answer_vs_truth_ratio': round(answer_length / max(1, ground_truth_length), 2),
        }
        
        processed_data.append(row)
        
        # Si tiene bajo rendimiento RAGAS, agregarlo a la lista de bajo rendimiento
        if average_score < 0.6:  # Solo basado en métricas RAGAS
            low_performance_data.append(row)
    
    # === EXPORTAR ARCHIVOS ===
    files = {}
    
    # 1. CSV Principal - Todas las evaluaciones
    csv_file = output_dir / f"{rag_name}_ragas_evaluation_{timestamp}.csv"
    _export_to_csv(processed_data, csv_file)
    files['csv'] = csv_file
    
    # 2. CSV Bajo Rendimiento - Para análisis de problemas
    if low_performance_data:
        performance_file = output_dir / f"{rag_name}_low_performance_{timestamp}.csv"
        _export_to_csv(low_performance_data, performance_file)
        files['low_performance_csv'] = performance_file
    
    # 3. JSON Completo - Con datos originales RAGAS
    json_file = output_dir / f"{rag_name}_ragas_complete_{timestamp}.json"
    _export_ragas_json(evaluation_results, json_file, rag_name, timestamp, processed_data)
    files['json'] = json_file
    
    # 4. Dashboard HTML - Para visualización RAGAS
    html_file = output_dir / f"{rag_name}_ragas_dashboard_{timestamp}.html"
    _create_ragas_dashboard(processed_data, html_file, rag_name)
    files['html'] = html_file
    
    # === IMPRIMIR RESUMEN RAGAS ===
    _print_ragas_summary(processed_data, low_performance_data)
    
    # === ABRIR EN NAVEGADOR ===
    if open_browser:
        _open_in_browser(html_file)
    
    return files


def _detect_performance_issues(faithfulness: float, answer_relevancy: float, 
                             context_precision: float, context_recall: float,
                             answer_length: int, num_contexts: int) -> List[str]:
    """Detecta problemas de rendimiento basados en métricas RAGAS"""
    issues = []
    
    # Thresholds para detectar problemas
    LOW_THRESHOLD = 0.6
    VERY_LOW_THRESHOLD = 0.4
    
    # Problemas por métrica específica
    if faithfulness < VERY_LOW_THRESHOLD:
        issues.append("Faithfulness muy bajo")
    elif faithfulness < LOW_THRESHOLD:
        issues.append("Faithfulness bajo")
    
    if answer_relevancy < VERY_LOW_THRESHOLD:
        issues.append("Relevancia muy baja")
    elif answer_relevancy < LOW_THRESHOLD:
        issues.append("Relevancia baja")
        
    if context_precision < VERY_LOW_THRESHOLD:
        issues.append("Precisión contexto muy baja")
    elif context_precision < LOW_THRESHOLD:
        issues.append("Precisión contexto baja")
        
    if context_recall < VERY_LOW_THRESHOLD:
        issues.append("Recall contexto muy bajo")
    elif context_recall < LOW_THRESHOLD:
        issues.append("Recall contexto bajo")
    
    # Problemas generales
    if num_contexts < 2:
        issues.append("Muy pocos contextos")
    
    if answer_length < 50:
        issues.append("Respuesta muy corta")
    elif answer_length > 2000:
        issues.append("Respuesta muy larga")
    
    # Problema de rendimiento general
    average = (faithfulness + answer_relevancy + context_precision + context_recall) / 4
    if average < 0.5:
        issues.append("Rendimiento general muy bajo")
    
    return issues


def _classify_quality(average_score: float) -> str:
    """Clasifica la calidad basada en el score promedio"""
    if average_score >= 0.9:
        return "Excelente"
    elif average_score >= 0.8:
        return "Muy Bueno"
    elif average_score >= 0.7:
        return "Bueno"
    elif average_score >= 0.6:
        return "Regular"
    elif average_score >= 0.5:
        return "Malo"
    else:
        return "Muy Malo"


def _find_weakest_metric(faithfulness: float, answer_relevancy: float, 
                        context_precision: float, context_recall: float) -> str:
    """Encuentra la métrica con el score más bajo"""
    metrics = {
        'Faithfulness': faithfulness,
        'Answer Relevancy': answer_relevancy,
        'Context Precision': context_precision,
        'Context Recall': context_recall
    }
    return min(metrics, key=metrics.get)


def _find_strongest_metric(faithfulness: float, answer_relevancy: float, 
                          context_precision: float, context_recall: float) -> str:
    """Encuentra la métrica con el score más alto"""
    metrics = {
        'Faithfulness': faithfulness,
        'Answer Relevancy': answer_relevancy,
        'Context Precision': context_precision,
        'Context Recall': context_recall
    }
    return max(metrics, key=metrics.get)


def _export_to_csv(data: List[Dict], csv_file: Path):
    """Exporta datos a CSV"""
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        if data:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def _export_ragas_json(evaluation_results: List[Dict], json_file: Path, rag_name: str, 
                      timestamp: str, processed_data: List[Dict]):
    """Exporta datos completos RAGAS a JSON"""
    
    # Calcular estadísticas generales (RAGAS + Rendimiento)
    avg_scores = {}
    system_metrics = {}
    if processed_data:
        total = len(processed_data)
        avg_scores = {
            'faithfulness': sum(row['faithfulness'] for row in processed_data) / total,
            'answer_relevancy': sum(row['answer_relevancy'] for row in processed_data) / total,
            'context_precision': sum(row['context_precision'] for row in processed_data) / total,
            'context_recall': sum(row['context_recall'] for row in processed_data) / total,
            'overall_average': sum(row['average_score'] for row in processed_data) / total
        }
        
        system_metrics = {
            'avg_execution_time_seconds': sum(row['execution_time_seconds'] for row in processed_data) / total,
            'total_input_tokens': sum(row['input_tokens'] for row in processed_data),
            'total_output_tokens': sum(row['output_tokens'] for row in processed_data),
            'total_embedding_tokens': sum(row['embedding_tokens'] for row in processed_data),
            'total_cost_usd': sum(row['total_cost_usd'] for row in processed_data),
            'avg_cost_per_question': sum(row['total_cost_usd'] for row in processed_data) / total
        }
    
    export_data = {
        "metadata": {
            "rag_system": rag_name,
            "export_timestamp": timestamp,
            "evaluation_type": "RAGAS_with_Performance",
            "total_questions": len(evaluation_results),
            "low_performance_count": len([r for r in processed_data if r['average_score'] < 0.6]),
            "ragas_scores": avg_scores,
            "system_performance": system_metrics
        },
        "evaluation_results": evaluation_results,
        "processed_analysis": processed_data
    }
    
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(export_data, file, indent=2, ensure_ascii=False, default=str)


def _create_ragas_dashboard(processed_data: List[Dict], html_file: Path, rag_name: str):
    """Crea dashboard HTML específico para métricas RAGAS"""
    
    # Calcular estadísticas RAGAS y rendimiento
    total = len(processed_data)
    if total == 0:
        return
        
    low_performance_count = sum(1 for row in processed_data if row['average_score'] < 0.6)
    
    # Métricas RAGAS
    avg_faithfulness = sum(row['faithfulness'] for row in processed_data) / total
    avg_relevancy = sum(row['answer_relevancy'] for row in processed_data) / total
    avg_precision = sum(row['context_precision'] for row in processed_data) / total
    avg_recall = sum(row['context_recall'] for row in processed_data) / total
    avg_overall = sum(row['average_score'] for row in processed_data) / total
    
    # Métricas de rendimiento del sistema
    avg_execution_time = sum(row['execution_time_seconds'] for row in processed_data) / total
    total_input_tokens = sum(row['input_tokens'] for row in processed_data)
    total_output_tokens = sum(row['output_tokens'] for row in processed_data)
    total_embedding_tokens = sum(row['embedding_tokens'] for row in processed_data)
    total_all_tokens = sum(row['total_tokens'] for row in processed_data)
    total_cost = sum(row['total_cost_usd'] for row in processed_data)
    avg_cost_per_question = total_cost / total if total > 0 else 0
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAGAS Evaluation - {rag_name}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #666; margin-bottom: 5px; }}
        .low-performance {{ border-left-color: #dc3545; }}
        .low-performance .metric-value {{ color: #dc3545; }}
        .medium-performance {{ border-left-color: #ffc107; }}
        .medium-performance .metric-value {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }}
        th {{ background: #4CAF50; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f8f9fa; }}
        .excellent {{ background-color: #e8f5e8; }}
        .very-good {{ background-color: #f0f8f0; }}
        .good {{ background-color: #fff8e1; }}
        .regular {{ background-color: #fff3e0; }}
        .bad {{ background-color: #ffebee; }}
        .very-bad {{ background-color: #ffcdd2; }}
        .filter-container {{ margin: 20px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
        .filter-container input, .filter-container select {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
        .search-input {{ width: 300px; }}
        .table-container {{ max-height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .score-cell {{ font-weight: bold; }}
        .score-high {{ color: #4CAF50; }}
        .score-medium {{ color: #ff9800; }}
        .score-low {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 RAGAS Evaluation Dashboard</h1>
            <h2>{rag_name.upper()}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Total Questions</div>
                <div class="metric-value">{total}</div>
            </div>
            <div class="metric-box {'low-performance' if avg_faithfulness < 0.6 else 'medium-performance' if avg_faithfulness < 0.7 else ''}">
                <div class="metric-label">Avg Faithfulness</div>
                <div class="metric-value">{avg_faithfulness:.3f}</div>
            </div>
            <div class="metric-box {'low-performance' if avg_relevancy < 0.6 else 'medium-performance' if avg_relevancy < 0.7 else ''}">
                <div class="metric-label">Avg Answer Relevancy</div>
                <div class="metric-value">{avg_relevancy:.3f}</div>
            </div>
            <div class="metric-box {'low-performance' if avg_precision < 0.6 else 'medium-performance' if avg_precision < 0.7 else ''}">
                <div class="metric-label">Avg Context Precision</div>
                <div class="metric-value">{avg_precision:.3f}</div>
            </div>
            <div class="metric-box {'low-performance' if avg_recall < 0.6 else 'medium-performance' if avg_recall < 0.7 else ''}">
                <div class="metric-label">Avg Context Recall</div>
                <div class="metric-value">{avg_recall:.3f}</div>
            </div>
            <div class="metric-box {'low-performance' if avg_overall < 0.6 else 'medium-performance' if avg_overall < 0.7 else ''}">
                <div class="metric-label">Overall Average</div>
                <div class="metric-value">{avg_overall:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg Execution Time</div>
                <div class="metric-value">{avg_execution_time:.2f}s</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Input Tokens</div>
                <div class="metric-value">{total_input_tokens:,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Output Tokens</div>
                <div class="metric-value">{total_output_tokens:,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Embedding Tokens</div>
                <div class="metric-value">{total_embedding_tokens:,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Cost</div>
                <div class="metric-value">${total_cost:.4f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg Cost per Question</div>
                <div class="metric-value">${avg_cost_per_question:.4f}</div>
            </div>
            <div class="metric-box {'low-performance' if low_performance_count > total/2 else 'medium-performance' if low_performance_count > total/4 else ''}">
                <div class="metric-label">Low RAGAS Performance</div>
                <div class="metric-value">{low_performance_count} ({low_performance_count/total*100:.1f}%)</div>
            </div>
        </div>
        
        <div class="filter-container">
            <input type="text" id="searchInput" class="search-input" placeholder="🔍 Search questions, answers..." onkeyup="searchTable()">
            <select id="performanceFilter" onchange="filterByPerformance()">
                <option value="all">All Performance Levels</option>
                <option value="high">High RAGAS (≥0.7)</option>
                <option value="medium">Medium RAGAS (0.6-0.7)</option>
                <option value="low">Low RAGAS (<0.6)</option>
            </select>
            <select id="costFilter" onchange="filterByCost()">
                <option value="all">All Cost Levels</option>
                <option value="high-cost">High Cost (>avg)</option>
                <option value="low-cost">Low Cost (≤avg)</option>
            </select>
        </div>
        
        <div class="table-container">
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Question</th>
                        <th>Ground Truth</th>
                        <th>LLM Answer</th>
                        <th>Faithfulness</th>
                        <th>Relevancy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Avg Score</th>
                        <th>Exec Time (s)</th>
                        <th>Input Tokens</th>
                        <th>Output Tokens</th>
                        <th>Embedding Tokens</th>
                        <th>Cost (USD)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Agregar filas de datos
    for row in processed_data:
        # Determinar clases de color para scores RAGAS
        def get_score_class(score):
            if score >= 0.7:
                return "score-high"
            elif score >= 0.6:
                return "score-medium"
            else:
                return "score-low"
        
        # Determinar clase para rendimiento general
        performance_class = get_score_class(row['average_score']).replace('score-', '')
        
        # Determinar clase para costo (comparando con promedio)
        cost_class = "high-cost" if row['total_cost_usd'] > avg_cost_per_question else "low-cost"
        
        html_content += f"""
                    <tr class="{performance_class}" data-performance="{performance_class}" data-cost="{cost_class}" data-avg-score="{row['average_score']}">
                        <td>{row['id']}</td>
                        <td title="{row['question']}">{row['question']}</td>
                        <td title="{row['ground_truth']}">{row['ground_truth']}</td>
                        <td title="{row['llm_answer']}">{row['llm_answer']}</td>
                        <td class="score-cell {get_score_class(row['faithfulness'])}">{row['faithfulness']}</td>
                        <td class="score-cell {get_score_class(row['answer_relevancy'])}">{row['answer_relevancy']}</td>
                        <td class="score-cell {get_score_class(row['context_precision'])}">{row['context_precision']}</td>
                        <td class="score-cell {get_score_class(row['context_recall'])}">{row['context_recall']}</td>
                        <td class="score-cell {get_score_class(row['average_score'])}"><strong>{row['average_score']}</strong></td>
                        <td>{row['execution_time_seconds']}</td>
                        <td>{row['input_tokens']:,}</td>
                        <td>{row['output_tokens']:,}</td>
                        <td>{row['embedding_tokens']:,}</td>
                        <td>${row['total_cost_usd']:.6f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        function searchTable() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('resultsTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {
                const text = rows[i].textContent.toLowerCase();
                rows[i].style.display = text.includes(filter) ? '' : 'none';
            }
        }
        
        function filterByPerformance() {
            const select = document.getElementById('performanceFilter');
            const filter = select.value;
            const table = document.getElementById('resultsTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {
                const row = rows[i];
                const avgScore = parseFloat(row.getAttribute('data-avg-score'));
                
                let show = false;
                if (filter === 'all') {
                    show = true;
                } else if (filter === 'high' && avgScore >= 0.7) {
                    show = true;
                } else if (filter === 'medium' && avgScore >= 0.6 && avgScore < 0.7) {
                    show = true;
                } else if (filter === 'low' && avgScore < 0.6) {
                    show = true;
                }
                
                row.style.display = show ? '' : 'none';
            }
        }
        
        function filterByCost() {
            const select = document.getElementById('costFilter');
            const filter = select.value;
            const table = document.getElementById('resultsTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {
                const row = rows[i];
                const costClass = row.getAttribute('data-cost');
                
                let show = false;
                if (filter === 'all') {
                    show = true;
                } else if (filter === 'high-cost' && costClass === 'high-cost') {
                    show = true;
                } else if (filter === 'low-cost' && costClass === 'low-cost') {
                    show = true;
                }
                
                row.style.display = show ? '' : 'none';
            }
        }
    </script>
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(html_content)


def _print_ragas_summary(processed_data: List[Dict], low_performance_data: List[Dict]):
    """Imprime resumen de evaluación RAGAS en consola"""
    total = len(processed_data)
    low_performance_count = len(low_performance_data)
    
    print(f"\n📊 === RESUMEN EVALUACIÓN RAGAS ===")
    print(f"🔍 Total de preguntas evaluadas: {total}")
    print(f"❌ Preguntas con bajo rendimiento RAGAS (<0.6): {low_performance_count} ({low_performance_count/total*100:.1f}%)")
    print(f"✅ Preguntas con buen rendimiento RAGAS (≥0.6): {total - low_performance_count} ({(total-low_performance_count)/total*100:.1f}%)")
    
    if processed_data:
        # Métricas RAGAS
        avg_faithfulness = sum(row['faithfulness'] for row in processed_data) / total
        avg_relevancy = sum(row['answer_relevancy'] for row in processed_data) / total
        avg_precision = sum(row['context_precision'] for row in processed_data) / total
        avg_recall = sum(row['context_recall'] for row in processed_data) / total
        avg_overall = sum(row['average_score'] for row in processed_data) / total
        
        print(f"\n📈 Métricas RAGAS Promedio:")
        print(f"   🔍 Faithfulness: {avg_faithfulness:.3f}")
        print(f"   🎯 Answer Relevancy: {avg_relevancy:.3f}")
        print(f"   📊 Context Precision: {avg_precision:.3f}")
        print(f"   📋 Context Recall: {avg_recall:.3f}")
        print(f"   ⭐ Overall Average: {avg_overall:.3f}")
        
        # Métricas de rendimiento del sistema
        avg_execution_time = sum(row['execution_time_seconds'] for row in processed_data) / total
        total_input_tokens = sum(row['input_tokens'] for row in processed_data)
        total_output_tokens = sum(row['output_tokens'] for row in processed_data)
        total_embedding_tokens = sum(row['embedding_tokens'] for row in processed_data)
        total_cost = sum(row['total_cost_usd'] for row in processed_data)
        
        print(f"\nSystem Performance Metrics:")
        print(f"  Average time per question: {avg_execution_time:.2f}s")
        print(f"  Total input tokens: {total_input_tokens:,}")
        print(f"  Total output tokens: {total_output_tokens:,}")
        print(f"  Total embedding tokens: {total_embedding_tokens:,}")
        print(f"  Total tokens used: {total_input_tokens + total_output_tokens + total_embedding_tokens:,}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Average cost per question: ${total_cost/total:.4f}")
    
    print(f"\n📁 Archivos generados en: results/ragas_analysis/")


def _open_in_browser(html_file: Path):
    """Intenta abrir el dashboard HTML en el navegador"""
    try:
        webbrowser.open(f"file://{html_file.absolute()}")
        print(f"🌐 Dashboard RAGAS abierto en el navegador")
    except Exception as e:
        print(f"⚠️ No se pudo abrir automáticamente: {e}")
        print(f"   Abre manualmente: {html_file}")


# Mantener función original para compatibilidad
def export_rag_analysis(results: List[Dict], rag_name: str = "rag_system", open_browser: bool = True) -> Dict[str, Path]:
    """
    Función de compatibilidad - redirige a export_ragas_analysis
    """
    return export_ragas_analysis(results, rag_name, open_browser)