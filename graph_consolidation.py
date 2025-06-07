from neurograph.core import Component
from neurograph.core.logging import get_logger
from typing import Dict, Any, List
import time

class GraphConsolidator(Component):
    """Компонент для консолидации графа знаний."""
    
    def __init__(self):
        super().__init__("graph_consolidator")
        self.logger = get_logger("graph_consolidator")
        self.min_confidence = 0.3
        self.similarity_threshold = 0.85
    
    def consolidate(self, semgraph: Any) -> Dict[str, Any]:
        """Консолидация графа."""
        start_time = time.time()
        metrics = {
            "nodes_merged": 0,
            "edges_removed": 0,
            "low_confidence_removed": 0,
            "processing_time": 0.0
        }
        
        try:
            # Поиск дублирующихся узлов
            nodes = semgraph.get_all_nodes()
            node_pairs = [(n1, n2) for i, n1 in enumerate(nodes) for n2 in nodes[i+1:]
                          if self._is_similar(n1["id"], n2["id"])]
            
            for node1, node2 in node_pairs:
                # Объединяем узлы
                merged_id = node1["id"]
                semgraph.merge_nodes(node1["id"], node2["id"], merged_id)
                metrics["nodes_merged"] += 1
                self.logger.info(f"Объединены узлы: {node1['id']} и {node2['id']} -> {merged_id}")
            
            # Удаление низкокачественных узлов и связей
            for node in nodes:
                if node.get("confidence", 1.0) < self.min_confidence:
                    semgraph.remove_node(node["id"])
                    metrics["low_confidence_removed"] += 1
            
            edges = semgraph.get_all_edges()
            for edge in edges:
                if edge.get("weight", 1.0) < self.min_confidence:
                    semgraph.remove_edge(edge["source"], edge["target"], edge["type"])
                    metrics["edges_removed"] += 1
            
            metrics["processing_time"] = time.time() - start_time
            self.logger.info(f"Консолидация завершена: {metrics}")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Ошибка консолидации: {e}", exc_info=True)
            metrics["error"] = str(e)
            return metrics
    
    def _is_similar(self, id1: str, id2: str) -> bool:
        """Проверка схожести узлов (упрощенная версия)."""
        # Здесь можно использовать более сложные методы (например, косинусное сходство)
        return id1.lower() == id2.lower() or \
               (len(id1) > 3 and len(id2) > 3 and
                max(len(id1), len(id2)) / min(len(id1), len(id2)) < 1.2 and
                id1.lower() in id2.lower() or id2.lower() in id1.lower())