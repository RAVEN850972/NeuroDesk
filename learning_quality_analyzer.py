from neurograph.core import Component
from neurograph.core.logging import get_logger
from typing import Dict, Any, List
from neurograph.integration.base import ProcessingResponse

class LearningQualityAnalyzer(Component):
    """Компонент для анализа качества обучения."""
    
    def __init__(self):
        super().__init__("learning_quality_analyzer")
        self.logger = get_logger("learning_quality_analyzer")
        self.learning_stats = []
    
    def analyze(self, response: ProcessingResponse, input_text: str) -> Dict[str, Any]:
        """Анализ качества обучения."""
        metrics = {
            "nodes_added": 0,
            "edges_added": 0,
            "relevance_score": 0.0,
            "text_length": len(input_text),
            "unique_entities": 0,
            "timestamp": time.time()
        }
        
        try:
            learning_data = response.structured_data.get("learning", {})
            metrics["nodes_added"] = learning_data.get("nodes_added", 0)
            metrics["edges_added"] = learning_data.get("edges_added", 0)
            
            # Оценка релевантности (упрощенно: на основе количества сущностей и их уверенности)
            nlp_data = response.structured_data.get("nlp", {})
            entities = nlp_data.get("entities", [])
            metrics["unique_entities"] = len(set(e["text"] for e in entities))
            
            if entities:
                avg_confidence = sum(e.get("confidence", 0.5) for e in entities) / len(entities)
                metrics["relevance_score"] = min(1.0, metrics["nodes_added"] / 10.0 + avg_confidence * 0.5)
            
            self.learning_stats.append(metrics)
            self.logger.info(f"Анализ качества обучения: {metrics}")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Ошибка анализа качества: {e}", exc_info=True)
            metrics["error"] = str(e)
            return metrics
    
    def get_quality_report(self, max_entries: int = 10) -> Dict[str, Any]:
        """Получение отчета о качестве обучения."""
        recent_stats = self.learning_stats[-max_entries:] if self.learning_stats else []
        
        if not recent_stats:
            return {"message": "Нет данных об обучении"}
        
        avg_metrics = {
            "avg_nodes_added": sum(s["nodes_added"] for s in recent_stats) / len(recent_stats),
            "avg_edges_added": sum(s["edges_added"] for s in recent_stats) / len(recent_stats),
            "avg_relevance_score": sum(s["relevance_score"] for s in recent_stats) / len(recent_stats),
            "avg_unique_entities": sum(s["unique_entities"] for s in recent_stats) / len(recent_stats),
            "total_sessions": len(recent_stats)
        }
        
        return {
            "recent_stats": recent_stats,
            "averages": avg_metrics,
            "total_learned": len(self.learning_stats)
        }