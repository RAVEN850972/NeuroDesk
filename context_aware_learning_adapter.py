from neurograph.integration.base import IComponentAdapter, ProcessingRequest
from typing import Dict, Any, List
from neurograph.core.logging import get_logger

class ContextAwareLearningAdapter(IComponentAdapter):
    """Адаптер для контекстно-зависимого обучения графа."""
    
    def __init__(self):
        self.adapter_name = "context_aware_learning"
        self.logger = get_logger("context_aware_learning")
    
    def adapt(self, source_data: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """Адаптирует данные для обучения с учетом контекста."""
        if target_format != "graph_updates":
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
        
        context = source_data.get("context", {})
        nlp_data = source_data.get("nlp_data", {})
        
        # Извлекаем предметную область из контекста
        domain = context.get("domain", "general")
        
        # Усиливаем веса для релевантных узлов и связей
        graph_updates = {
            "nodes_to_add": [],
            "edges_to_add": [],
            "context_weight": 1.0  # Базовый вес
        }
        
        # Если указана предметная область, увеличиваем вес связей
        if domain != "general":
            graph_updates["context_weight"] = 1.5
        
        # Обрабатываем сущности из NLP
        for entity in nlp_data.get("entities", []):
            node = {
                "id": entity["text"],
                "type": entity["entity_type"],
                "weight": graph_updates["context_weight"],
                "domain": domain,
                "confidence": entity.get("confidence", 0.5)
            }
            graph_updates["nodes_to_add"].append(node)
        
        # Обрабатываем связи
        for relation in nlp_data.get("relations", []):
            edge = {
                "source": relation["subject"]["text"],
                "target": relation["object"]["text"],
                "type": relation["predicate"],
                "weight": graph_updates["context_weight"] * relation.get("confidence", 0.5),
                "domain": domain
            }
            graph_updates["edges_to_add"].append(edge)
        
        self.logger.info(f"Сформировано {len(graph_updates['nodes_to_add'])} узлов и "
                        f"{len(graph_updates['edges_to_add'])} связей для домена {domain}")
        
        return graph_updates
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы адаптера."""
        return {
            "input": ["nlp_data"],
            "output": ["graph_updates"]
        }