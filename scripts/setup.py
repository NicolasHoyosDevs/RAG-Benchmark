"""
Script de configuración inicial para RAG Benchmark
"""

import os
import sys
import asyncio
from pathlib import Path

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))


async def setup_environment():
    """Configura el entorno inicial"""
    print("🔧 Configurando entorno RAG Benchmark...")
    
    # Crear directorios necesarios
    directories = [
        "Data/raw",
        "Data/processed", 
        "Data/embeddings",
        "Data/graphs",
        "results",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent.parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")
    
    # Crear archivos .gitkeep para directorios vacíos
    gitkeep_dirs = ["Data/raw", "Data/processed", "Data/embeddings", "Data/graphs", "results"]
    for directory in gitkeep_dirs:
        gitkeep_path = Path(__file__).parent.parent / directory / ".gitkeep"
        gitkeep_path.touch()
    
    print("✅ Directorios configurados")


def check_dependencies():
    """Verifica dependencias opcionales"""
    print("🔍 Verificando dependencias...")
    
    optional_deps = {
        'openai': 'pip install openai',
        'chromadb': 'pip install chromadb', 
        'faiss': 'pip install faiss-cpu',
        'neo4j': 'pip install neo4j',
        'sentence_transformers': 'pip install sentence-transformers'
    }
    
    missing_deps = []
    
    for dep, install_cmd in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - {install_cmd}")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\\n⚠️  Dependencias faltantes: {', '.join(missing_deps)}")
        print("Instala las que necesites según tu configuración.")
    else:
        print("✅ Todas las dependencias opcionales están instaladas")


def setup_config_files():
    """Configura archivos de configuración"""
    print("📝 Configurando archivos...")
    
    env_example = Path(__file__).parent.parent / ".env.example"
    env_file = Path(__file__).parent.parent / ".env"
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Archivo .env creado desde .env.example")
        print("⚠️  Recuerda configurar tus API keys en .env")
    
    print("✅ Configuración completada")


def show_next_steps():
    """Muestra los siguientes pasos"""
    print("\\n🎉 ¡Configuración inicial completada!")
    print("\\n📋 Siguientes pasos:")
    print("1. Configura tus API keys en .env")
    print("2. Si usas Graph RAG, configura Neo4j")
    print("3. Añade tus documentos a Data/raw/")
    print("4. Ejecuta: python scripts/run_benchmark.py")
    print("\\n📚 Consulta README.md para más información")


async def main():
    """Función principal de configuración"""
    print("🚀 RAG Benchmark - Configuración Inicial")
    print("=" * 50)
    
    await setup_environment()
    check_dependencies()
    setup_config_files()
    show_next_steps()


if __name__ == "__main__":
    asyncio.run(main())