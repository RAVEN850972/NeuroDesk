from neurograph.core import Component
from neurograph.core.logging import get_logger
from typing import Dict, Any, List
from neurograph.integration.base import IntegrationConfig, ProcessingResponse

class FeedbackLearner(Component):
    """Компонент для обучения на основе обратной связи."""
    
    def __init__(self):
        super().__init__("feedback_learner")
        self.logger = get_logger("feedback_learner")
    
    def apply_feedback(self, semgraph: Any, response: ProcessingResponse, feedback: float) -> Dict[str, Any]:
        """Корректировка графа на основе обратной связи."""
        metrics = {
            "edges_adjusted": 0,
            "nodes_adjusted": 0,
            "feedback_value": feedback
        }
        
        try:
            # Получаем данные поиска в графе
            graph_search = response.structured_data.get("graph_search", {})
            found_nodes = graph_search.get("found_nodes", [])
            activated_edges = graph_search.get("activated_edges", [])
            
            # Положительный фидбек (1.0) усиливает связи, отрицательный (-1.0) ослабляет
            weight_adjustment = 0.2 if feedback > 0 else -0.2
            
            # Корректируем веса узлов
            for node_id in found_nodes:
                current_weight = semgraph.get_node_weight(node_id) or 1.0
                new_weight = max(0.1, min(2.0, current_weight + weight_adjustment))
                semgraph.set_node_weight(node_id, new_weight)
                metrics["nodes_adjusted"] += 1
            
            # Корректируем веса связей
            for edge in activated_edges:
                current_weight = semgraph.get_edge_weight(edge["source"], edge["target"], edge["type"]) or 1.0
                new_weight = max(0.1, min(2.0, current_weight + weight_adjustment))
                semgraph.set_edge_weight(edge["source"], edge["target"], edge["type"], new_weight)
                metrics["edges_adjusted"] += 1
            
            self.logger.info(f"Применен фидбек {feedback}: {metrics}")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Ошибка применения фидбека: {e}", exc_info=True)
            metrics["error"] = str(e)
            return metrics