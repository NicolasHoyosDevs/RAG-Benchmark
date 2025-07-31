"""
Cargador de configuración para RAG Benchmark
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """
    Cargador de configuración que maneja archivos YAML y variables de entorno
    """
    
    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """
        Carga configuración desde archivo YAML y variables de entorno
        """
        if config_path is None:
            # Buscar config.yaml en el directorio del proyecto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        config = {}
        
        # Cargar desde archivo YAML
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        # Sobrescribir con variables de entorno
        config = ConfigLoader._override_with_env_vars(config)
        
        return config
    
    @staticmethod
    def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sobrescribe configuración con variables de entorno
        """
        env_mappings = {
            'OPENAI_API_KEY': ['llm', 'api_key'],
            'NEO4J_URI': ['graph_rag', 'neo4j', 'uri'],
            'NEO4J_USERNAME': ['graph_rag', 'neo4j', 'username'],
            'NEO4J_PASSWORD': ['graph_rag', 'neo4j', 'password'],
            'PINECONE_API_KEY': ['vector_stores', 'pinecone', 'api_key'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                ConfigLoader._set_nested_config(config, config_path, value)
        
        return config
    
    @staticmethod
    def _set_nested_config(config: Dict[str, Any], path: list, value: Any):
        """
        Establece valor en configuración anidada
        """
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Valida que la configuración tenga los campos requeridos
        """
        required_fields = [
            ['llm', 'model'],
            ['embeddings', 'model'],
            ['data', 'chunk_size']
        ]
        
        for field_path in required_fields:
            if not ConfigLoader._has_nested_field(config, field_path):
                print(f"Campo requerido faltante: {'.'.join(field_path)}")
                return False
        
        return True
    
    @staticmethod
    def _has_nested_field(config: Dict[str, Any], path: list) -> bool:
        """
        Verifica si existe un campo anidado en la configuración
        """
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        return True