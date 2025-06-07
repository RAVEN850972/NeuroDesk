from neurograph.integration.pipelines import QueryProcessingPipeline
from neurograph.integration.base import ProcessingRequest, ProcessingResponse, IComponentProvider
from neurograph.core.logging import get_logger
from openai import OpenAI
import time
import os
from typing import Dict, Any, Optional

class HybridQueryPipeline(QueryProcessingPipeline):
    """Гибридный конвейер с fallback на OpenAI и автообучением."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        super().__init__()
        self.pipeline_name = "hybrid_query"
        self.logger = get_logger("hybrid_query_pipeline")
        
        # Инициализация OpenAI клиента
        api_key = "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"
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            self.logger.warning("OpenAI API ключ не найден. Fallback недоступен.")
        
        # Пороги для определения качества ответа
        self.min_confidence_threshold = 0.4
        self.min_response_length = 20
        
        # Статистика использования
        self.stats = {
            "graph_responses": 0,
            "openai_fallbacks": 0,
            "learning_sessions": 0,
            "total_queries": 0
        }
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка запроса с гибридной логикой."""
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            # Валидация запроса
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_error_response(request, error_msg)
            
            # Шаг 1: Попытка получить ответ из графа знаний
            graph_response = self._query_graph(request, provider)
            
            # Проверяем качество ответа из графа
            if self._is_response_sufficient(graph_response):
                self.stats["graph_responses"] += 1
                self.logger.info(f"Ответ получен из графа знаний (confidence: {graph_response.confidence:.2f})")
                return graph_response
            
            # Шаг 2: Fallback на OpenAI API
            if self.openai_client:
                openai_response = self._query_openai(request, provider)
                if openai_response.success:
                    self.stats["openai_fallbacks"] += 1
                    
                    # Шаг 3: Обучение графа на ответе OpenAI
                    learning_success = self._learn_from_openai_response(
                        request, openai_response, provider
                    )
                    
                    if learning_success:
                        self.stats["learning_sessions"] += 1
                        self.logger.info("Граф обучен на ответе OpenAI")
                    
                    # Помечаем ответ как полученный от OpenAI
                    openai_response.metadata["source"] = "openai_fallback"
                    openai_response.metadata["graph_learning"] = learning_success
                    
                    return openai_response
            
            # Шаг 4: Если OpenAI недоступен, возвращаем лучшее, что есть
            if graph_response.success:
                graph_response.metadata["source"] = "graph_low_confidence"
                graph_response.warnings.append("Ответ получен из графа с низкой уверенностью")
                return graph_response
            
            # Шаг 5: Последний резерв - заглушка
            return self._create_fallback_response(request)
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Критическая ошибка в гибридном конвейере: {e}", exc_info=True)
            return self._create_error_response(request, f"Системная ошибка: {str(e)}")
    
    def _query_graph(self, request: ProcessingRequest, 
                    provider: IComponentProvider) -> ProcessingResponse:
        """Запрос к графу знаний."""
        try:
            # Используем базовую логику родительского класса
            return super().process(request, provider)
        except Exception as e:
            self.logger.error(f"Ошибка запроса к графу: {e}")
            return ProcessingResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"Ошибка графа: {str(e)}"
            )
    
    def _query_openai(self, request: ProcessingRequest, 
                     provider: IComponentProvider) -> ProcessingResponse:
        """Запрос к OpenAI API."""
        start_time = time.time()
        
        try:
            # Формируем контекст для OpenAI
            context_prompt = self._build_openai_context(request, provider)
            
            # Делаем запрос к OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Используем более экономичную модель
                messages=[
                    {
                        "role": "system", 
                        "content": "Ты - умный ассистент. Отвечай кратко, точно и по существу. "
                                  "Если не знаешь ответа, честно скажи об этом."
                    },
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                timeout=30
            )
            
            openai_answer = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            # Формируем ответ в формате системы
            return ProcessingResponse(
                request_id=request.request_id,
                success=True,
                primary_response=openai_answer,
                processing_time=processing_time,
                components_used=["openai_api"],
                confidence=0.8,  # Считаем OpenAI достаточно надежным
                metadata={
                    "source": "openai",
                    "model_used": "gpt-4o-mini",
                    "tokens_used": response.usage.total_tokens if response.usage else None
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка OpenAI API: {e}")
            return ProcessingResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"OpenAI API недоступен: {str(e)}",
                processing_time=processing_time
            )
    
    def _build_openai_context(self, request: ProcessingRequest, 
                             provider: IComponentProvider) -> str:
        """Построение контекста для OpenAI на основе доступной информации."""
        context_parts = [request.content]
        
        # Добавляем контекст пользователя, если есть
        if request.context:
            user_context = request.context.get("user_context", {})
            if user_context.get("expertise_areas"):
                context_parts.append(
                    f"Контекст пользователя: эксперт в областях {', '.join(user_context['expertise_areas'][:3])}"
                )
            
            domain = request.context.get("domain")
            if domain and domain != "general":
                context_parts.append(f"Предметная область: {domain}")
        
        # Пытаемся получить релевантную информацию из памяти
        if provider.is_component_available("memory"):
            try:
                memory = provider.get_component("memory")
                recent_items = memory.search_by_similarity(request.content, max_results=3)
                if recent_items:
                    relevant_context = []
                    for item in recent_items:
                        if len(item.content) < 200:  # Избегаем слишком длинных контекстов
                            relevant_context.append(item.content[:100])
                    
                    if relevant_context:
                        context_parts.append(
                            f"Релевантная информация из памяти системы: {'; '.join(relevant_context)}"
                        )
            except Exception as e:
                self.logger.warning(f"Не удалось получить контекст из памяти: {e}")
        
        return "\n\n".join(context_parts)
    
    def _learn_from_openai_response(self, request: ProcessingRequest, 
                                   openai_response: ProcessingResponse,
                                   provider: IComponentProvider) -> bool:
        """Обучение графа на основе ответа OpenAI."""
        try:
            # Формируем обучающий контент
            learning_content = self._create_learning_content(request, openai_response)
            
            # Создаем запрос на обучение
            learning_request = ProcessingRequest(
                content=learning_content,
                request_type="context_aware_learning",
                context={
                    "source": "openai_fallback",
                    "original_query": request.content,
                    "confidence": openai_response.confidence,
                    "domain": request.context.get("domain", "general") if request.context else "general"
                },
                enable_nlp=True,
                enable_graph_reasoning=True,
                enable_memory_search=False  # Не ищем при обучении
            )
            
            # Выполняем обучение через соответствующий конвейер
            if provider.is_component_available("context_aware_learning"):
                from learning_pipeline import ContextAwareLearningPipeline
                learning_pipeline = ContextAwareLearningPipeline()
                learning_response = learning_pipeline.process(learning_request, provider)
                
                if learning_response.success:
                    nodes_added = learning_response.structured_data.get("learning", {}).get("nodes_added", 0)
                    edges_added = learning_response.structured_data.get("learning", {}).get("edges_added", 0)
                    
                    self.logger.info(
                        f"Обучение успешно: добавлено {nodes_added} узлов, {edges_added} связей"
                    )
                    return True
                else:
                    self.logger.warning(f"Обучение не удалось: {learning_response.error_message}")
                    return False
            else:
                # Fallback на базовое обучение
                return self._basic_learning(learning_content, provider)
                
        except Exception as e:
            self.logger.error(f"Ошибка обучения на ответе OpenAI: {e}", exc_info=True)
            return False
    
    def _create_learning_content(self, request: ProcessingRequest, 
                               openai_response: ProcessingResponse) -> str:
        """Создание контента для обучения."""
        # Формируем связь вопрос-ответ для обучения
        learning_patterns = [
            f"Вопрос: {request.content}",
            f"Ответ: {openai_response.primary_response}",
        ]
        
        # Добавляем контекст, если есть
        if request.context:
            domain = request.context.get("domain")
            if domain and domain != "general":
                learning_patterns.append(f"Предметная область: {domain}")
        
        return "\n".join(learning_patterns)
    
    def _basic_learning(self, content: str, provider: IComponentProvider) -> bool:
        """Базовое обучение без продвинутых адаптеров."""
        try:
            # Сохраняем в память
            if provider.is_component_available("memory"):
                memory = provider.get_component("memory")
                memory.store_item(content, metadata={
                    "content_type": "openai_learning",
                    "source": "openai_fallback"
                })
            
            # Простое добавление в граф через NLP
            if provider.is_component_available("nlp") and provider.is_component_available("semgraph"):
                nlp = provider.get_component("nlp")
                semgraph = provider.get_component("semgraph")
                
                nlp_result = nlp.process_text(content)
                
                # Добавляем сущности как узлы
                for entity in nlp_result.entities[:5]:  # Ограничиваем количество
                    semgraph.add_node(
                        entity.text, 
                        entity.entity_type,
                        weight=1.2,  # Немного повышенный вес для данных от OpenAI
                        source="openai"
                    )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Ошибка базового обучения: {e}")
            return False
    
    def _is_response_sufficient(self, response: ProcessingResponse) -> bool:
        """Проверка достаточности ответа из графа."""
        if not response.success:
            return False
        
        # Проверяем уверенность
        if response.confidence < self.min_confidence_threshold:
            return False
        
        # Проверяем длину ответа
        if len(response.primary_response.strip()) < self.min_response_length:
            return False
        
        # Проверяем, что ответ не является заглушкой
        placeholder_phrases = [
            "не найдено информации",
            "недостаточно данных",
            "не удалось найти",
            "информация отсутствует"
        ]
        
        response_lower = response.primary_response.lower()
        if any(phrase in response_lower for phrase in placeholder_phrases):
            return False
        
        return True
    
    def _create_fallback_response(self, request: ProcessingRequest) -> ProcessingResponse:
        """Создание резервного ответа."""
        return ProcessingResponse(
            request_id=request.request_id,
            success=True,
            primary_response="Извините, я пока не могу ответить на этот вопрос. "
                           "Попробуйте переформулировать или задать более конкретный вопрос.",
            confidence=0.1,
            components_used=["fallback"],
            metadata={"source": "system_fallback"},
            warnings=["Ответ не найден ни в графе знаний, ни через внешние API"]
        )
    
    def _create_error_response(self, request: ProcessingRequest, 
                              error_message: str) -> ProcessingResponse:
        """Создание ответа с ошибкой."""
        return ProcessingResponse(
            request_id=request.request_id,
            success=False,
            error_message=error_message,
            components_used=[],
            processing_time=0.0
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Получение статистики работы конвейера."""
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "graph_success_rate": self.stats["graph_responses"] / total,
            "openai_fallback_rate": self.stats["openai_fallbacks"] / total,
            "learning_success_rate": (
                self.stats["learning_sessions"] / max(1, self.stats["openai_fallbacks"])
            )
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Информация о конвейере."""
        return {
            "name": self.pipeline_name,
            "type": "hybrid_query",
            "openai_available": self.openai_client is not None,
            "min_confidence_threshold": self.min_confidence_threshold,
            "statistics": self.get_pipeline_stats(),
            "capabilities": [
                "graph_knowledge_search",
                "openai_fallback",
                "automatic_learning",
                "quality_assessment"
            ]
        }