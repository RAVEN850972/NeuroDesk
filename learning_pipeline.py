from neurograph.integration.pipelines import LearningPipeline
from neurograph.integration.base import ProcessingRequest, ProcessingResponse, IComponentProvider
from neurograph.core.logging import get_logger
import time

class ContextAwareLearningPipeline(LearningPipeline):
    """Конвейер обучения с учетом контекста."""
    
    def __init__(self):
        super().__init__()
        self.pipeline_name = "context_aware_learning"
        self.logger = get_logger("context_aware_learning_pipeline")
    
    def process(self, request: ProcessingRequest, provider: IComponentProvider) -> ProcessingResponse:
        """Обработка запроса с учетом контекста."""
        start_time = time.time()
        
        try:
            # Валидация запроса
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_error_response(request, error_msg)
            
            components_used = []
            structured_data = {"learning": {}}
            
            # NLP-обработка
            nlp_result = {}
            if provider.is_component_available("nlp"):
                nlp = provider.get_component("nlp")
                nlp_result = nlp.process_text(request.content)
                components_used.append("nlp")
                structured_data["nlp"] = nlp_result
            
            # Контекстно-зависимая адаптация
            if provider.is_component_available("context_aware_learning"):
                adapter = provider.get_component("context_aware_learning")
                graph_updates = adapter.adapt(
                    {"nlp_data": nlp_result, "context": request.context},
                    "graph_updates"
                )
                components_used.append("context_aware_learning")
                
                # Добавление в граф
                if provider.is_component_available("semgraph"):
                    semgraph = provider.get_component("semgraph")
                    nodes_added = 0
                    edges_added = 0
                    
                    for node in graph_updates["nodes_to_add"]:
                        semgraph.add_node(node["id"], node["type"], node["weight"], node["domain"])
                        nodes_added += 1
                    
                    for edge in graph_updates["edges_to_add"]:
                        semgraph.add_edge(edge["source"], edge["target"], edge["type"], edge["weight"])
                        edges_added += 1
                    
                    components_used.append("semgraph")
                    structured_data["learning"]["nodes_added"] = nodes_added
                    structured_data["learning"]["edges_added"] = edges_added
            
            # Сохранение в память
            if provider.is_component_available("memory"):
                memory = provider.get_component("memory")
                memory.store_item(request.content, metadata={
                    "content_type": "learned_content",
                    "domain": request.context.get("domain", "general")
                })
                components_used.append("memory")
            
            # Анализ качества обучения
            if provider.is_component_available("learning_quality_analyzer"):
                analyzer = provider.get_component("learning_quality_analyzer")
                quality_metrics = analyzer.analyze(
                    ProcessingResponse(
                        request_id=request.request_id,
                        success=True,
                        structured_data=structured_data,
                        components_used=components_used
                    ),
                    request.content
                )
                structured_data["learning_quality"] = quality_metrics
                components_used.append("learning_quality_analyzer")
            
            processing_time = time.time() - start_time
            return ProcessingResponse(
                request_id=request.request_id,
                success=True,
                primary_response="Обучение завершено успешно",
                structured_data=structured_data,
                processing_time=processing_time,
                components_used=components_used
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_error_response(request, str(e))