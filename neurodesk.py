import os
import json
import uuid
import random
import copy
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from neurograph.core.logging import get_logger
import time
import pickle

from openai import OpenAI
from neurograph.integration import NeuroGraphEngine
from neurograph.integration.base import IntegrationConfig, ProcessingResponse, ProcessingRequest
from context_aware_learning_adapter import ContextAwareLearningAdapter
from graph_consolidation import GraphConsolidator
from feedback_learning import FeedbackLearner
from learning_quality_analyzer import LearningQualityAnalyzer

# Импорт нашего нового гибридного конвейера
from hybrid_query_pipeline import HybridQueryPipeline

# Ключ OpenAI должен быть в переменной окружения
SESSION_DIR = Path("./sessions")
SESSION_DIR.mkdir(exist_ok=True)

class PipelineTuner:
    """Генетический тюнер параметров конвейеров."""
    
    def __init__(self, engine: NeuroGraphEngine):
        self.engine = engine
        self.logger = get_logger("PipelineTuner")
        base_config = engine.config.to_dict()
        
        self.best_config = self._safe_copy(base_config)
        self.best_score = 0.0
        self.mutation_rate = 0.1
        self.iterations_without_improvement = 0
        
        # Параметры для мутации гибридного конвейера
        self.hybrid_params = {
            "min_confidence_threshold": 0.4,
            "min_response_length": 20,
            "learning_enabled": True
        }
    
    def _safe_copy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Безопасное копирование конфигурации."""
        safe_config = {}
        for key, value in config.items():
            try:
                pickle.dumps(value)
                safe_config[key] = copy.deepcopy(value)
            except (pickle.PicklingError, TypeError, AttributeError):
                self.logger.warning(f"Skipping non-serializable config key: {key}")
                safe_config[key] = None
        return safe_config
    
    def mutate_hybrid_params(self) -> Dict[str, Any]:
        """Мутация параметров гибридного конвейера."""
        mutated = self.hybrid_params.copy()
        
        if random.random() < self.mutation_rate:
            # Мутируем порог уверенности
            mutated["min_confidence_threshold"] = max(0.1, min(0.8, 
                self.hybrid_params["min_confidence_threshold"] + random.uniform(-0.1, 0.1)
            ))
        
        if random.random() < self.mutation_rate:
            # Мутируем минимальную длину ответа
            mutated["min_response_length"] = max(10, min(100,
                self.hybrid_params["min_response_length"] + random.randint(-5, 5)
            ))
        
        return mutated
    
    def evaluate(self, response: ProcessingResponse, feedback: Optional[int] = None) -> float:
        """Оценка качества ответа."""
        if not response.success:
            return 0.0
        
        score = response.confidence or 0.0
        
        # Бонус за использование графа знаний
        if response.metadata.get("source") == "graph":
            score += 0.2
        elif response.metadata.get("source") == "openai_fallback":
            # OpenAI получает базовый балл, но меньше чем граф
            score += 0.1
        
        # Учитываем обратную связь пользователя
        if feedback is not None:
            score += feedback * 0.3
        
        # Учитываем успешность обучения
        if response.metadata.get("graph_learning"):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def update(self, config: Dict[str, Any], score: float):
        """Обновление лучшей конфигурации."""
        if score > self.best_score:
            self.best_score = score
            self.best_config = self._safe_copy(config)
            self.iterations_without_improvement = 0
            self.logger.info(f"Новая лучшая конфигурация: score={score:.3f}")
        else:
            self.iterations_without_improvement += 1
        
        # Адаптивная мутация
        if self.iterations_without_improvement > 10:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
        elif self.iterations_without_improvement == 0:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)

class NeuroDesk:
    """Персональный AI-ассистент с гибридной архитектурой."""

    def __init__(self, user_id: str = "anon", config_path: str = None, 
                 openai_api_key: str = None):
        self.user_id = user_id
        self.session_id = f"{user_id}_{int(time.time())}"
        self.logger = get_logger("NeuroDesk")
        
        # Инициализация конфигурации
        self.config = self._load_config(config_path)
        self.engine = NeuroGraphEngine()
        self.engine.initialize(self.config)
        
        # Инициализация гибридного конвейера
        self.hybrid_pipeline = HybridQueryPipeline(openai_api_key)
        
        # Тюнер параметров
        self.tuner = PipelineTuner(self.engine)
        
        # Регистрация компонентов
        self._register_components()
        
        # Состояние сессии
        self._load_session_state()
        self._start_background_tasks()
        
        self.logger.info(f"NeuroDesk инициализирован для пользователя {user_id}")
    
    def _register_components(self):
        """Регистрация всех необходимых компонентов."""
        provider = self.engine.provider
        
        provider.register_lazy_component(
            "context_aware_learning", 
            lambda: ContextAwareLearningAdapter()
        )
        provider.register_lazy_component(
            "graph_consolidator", 
            lambda: GraphConsolidator()
        )
        provider.register_lazy_component(
            "feedback_learner", 
            lambda: FeedbackLearner()
        )
        provider.register_lazy_component(
            "learning_quality_analyzer", 
            lambda: LearningQualityAnalyzer()
        )
        
        # Регистрируем гибридный конвейер как компонент
        provider.register_component("hybrid_pipeline", self.hybrid_pipeline)
    
    def _load_config(self, config_path: str = None) -> IntegrationConfig:
        """Загрузка конфигурации."""
        from neurograph.integration.config import IntegrationConfigManager
        manager = IntegrationConfigManager()

        if config_path and os.path.exists(config_path):
            return manager.load_config(config_path)
        
        # Создаем конфигурацию по умолчанию
        config_data = manager.create_template_config("default")
        config_data["components"]["memory"]["params"] = {
            "stm_capacity": 100,
            "ltm_capacity": 5000
        }
        config_data["components"]["semgraph"] = {
            "type": "persistent",
            "params": {
                "file_path": f"graph_{self.user_id}.json",
                "auto_save_interval": 300.0
            }
        }
        return IntegrationConfig(**config_data)
    
    def _load_session_state(self):
        """Загрузка состояния сессии."""
        session_file = SESSION_DIR / f"{self.user_id}_session.json"
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    self.session_id = session_data.get("session_id", self.session_id)
                    self.logger.info(f"Загружена сессия: {self.session_id}")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить сессию: {e}")
    
    def _save_session_state(self):
        """Сохранение состояния сессии."""
        try:
            # Сохраняем ID сессии
            session_file = SESSION_DIR / f"{self.user_id}_session.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump({"session_id": self.session_id}, f)
            
            # Сохраняем состояние компонентов
            self.logger.info("💾 Сохраняем состояние системы")
            for comp_name in ["memory", "semgraph"]:
                if self.engine.provider.is_component_available(comp_name):
                    comp = self.engine.provider.get_component(comp_name)
                    if hasattr(comp, "save"):
                        try:
                            comp.save()
                            self.logger.debug(f"Сохранен компонент: {comp_name}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Ошибка сохранения {comp_name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния: {e}")
    
    def _start_background_tasks(self):
        """Запуск фоновых задач."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Запускаем фоновую консолидацию
            asyncio.create_task(self._background_consolidation())
        except Exception as e:
            self.logger.warning(f"Не удалось запустить фоновые задачи: {e}")
    
    async def _background_consolidation(self):
        """Фоновая консолидация графа."""
        while True:
            try:
                await asyncio.sleep(3600)  # Каждый час
                
                if (self.engine.provider.is_component_available("graph_consolidator") and 
                    self.engine.provider.is_component_available("semgraph")):
                    
                    consolidator = self.engine.provider.get_component("graph_consolidator")
                    semgraph = self.engine.provider.get_component("semgraph")
                    metrics = consolidator.consolidate(semgraph)
                    self.logger.info(f"🔄 Фоновая консолидация: {metrics}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка фоновой консолидации: {e}")
    
    def ask(self, prompt: str, feedback: Optional[int] = None, 
            domain: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Основной метод для вопросов с гибридной логикой.
        
        Args:
            prompt: Вопрос пользователя
            feedback: Обратная связь (-1, 0, 1)
            domain: Предметная область
            **kwargs: Дополнительные параметры
        
        Returns:
            Словарь с ответом и метаданными
        """
        try:
            # Мутируем параметры для тюнинга
            mutated_params = self.tuner.mutate_hybrid_params()
            
            # Обновляем параметры гибридного конвейера
            self.hybrid_pipeline.min_confidence_threshold = mutated_params["min_confidence_threshold"]
            self.hybrid_pipeline.min_response_length = mutated_params["min_response_length"]
            
            # Формируем запрос
            request = ProcessingRequest(
                content=prompt,
                request_type="hybrid_query",
                session_id=self.session_id,
                context={
                    "user_id": self.user_id,
                    "domain": domain or "general",
                    "user_context": self._get_user_context()
                },
                explanation_level="detailed",
                **kwargs
            )
            
            # Выполняем запрос через гибридный конвейер
            response = self.hybrid_pipeline.process(request, self.engine.provider)
            
            # Применяем обратную связь, если есть
            if feedback is not None:
                self._apply_feedback(response, feedback)
            
            # Оцениваем и обновляем тюнер
            score = self.tuner.evaluate(response, feedback)
            self.tuner.update(mutated_params, score)
            
            # Сохраняем состояние
            self._save_session_state()
            
            # Формируем результат
            result = {
                "answer": response.primary_response,
                "success": response.success,
                "confidence": response.confidence,
                "source": response.metadata.get("source", "unknown"),
                "processing_time": response.processing_time,
                "components_used": response.components_used,
                "explanation": response.explanation,
                "warnings": response.warnings
            }
            
            # Добавляем специфичную информацию для гибридного режима
            if response.metadata.get("source") == "openai_fallback":
                result["learned_from_openai"] = response.metadata.get("graph_learning", False)
                result["tokens_used"] = response.metadata.get("tokens_used")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка в ask(): {e}", exc_info=True)
            return {
                "answer": f"⚠️ Произошла ошибка при обработке запроса: {str(e)}",
                "success": False,
                "confidence": 0.0,
                "source": "error",
                "error": str(e)
            }
    
    def learn(self, text: str, domain: str = "general", 
              context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Явное обучение системы.
        
        Args:
            text: Текст для изучения
            domain: Предметная область
            context: Дополнительный контекст
            **kwargs: Дополнительные параметры
        
        Returns:
            Результат обучения
        """
        try:
            learning_context = {
                "domain": domain,
                "user_id": self.user_id,
                "learning_type": "explicit",
                **(context or {})
            }
            
            response = self.engine.learn(
                text,
                session_id=self.session_id,
                context=learning_context,
                request_type="context_aware_learning",
                **kwargs
            )
            
            self._save_session_state()
            
            return {
                "success": response.success,
                "message": response.primary_response,
                "nodes_added": response.structured_data.get("learning", {}).get("nodes_added", 0),
                "edges_added": response.structured_data.get("learning", {}).get("edges_added", 0),
                "processing_time": response.processing_time,
                "error": response.error_message if not response.success else None
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка в learn(): {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Ошибка обучения: {str(e)}",
                "error": str(e)
            }
    
    def _get_user_context(self) -> Dict[str, Any]:
        """Получение контекста пользователя."""
        context = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "expertise_areas": [],
            "preferences": {}
        }
        
        # Пытаемся получить информацию из памяти
        try:
            if self.engine.provider.is_component_available("memory"):
                memory = self.engine.provider.get_component("memory")
                recent_items = memory.get_recent_items(hours=24.0)
                
                # Извлекаем области экспертизы из недавней активности
                domains = set()
                for item in recent_items[:20]:  # Последние 20 элементов
                    item_domain = item.metadata.get("domain")
                    if item_domain and item_domain != "general":
                        domains.add(item_domain)
                
                context["expertise_areas"] = list(domains)[:5]  # Топ-5
                
        except Exception as e:
            self.logger.warning(f"Не удалось получить контекст пользователя: {e}")
        
        return context
    
    def _apply_feedback(self, response: ProcessingResponse, feedback: int):
        """Применение обратной связи пользователя."""
        try:
            if (self.engine.provider.is_component_available("feedback_learner") and 
                self.engine.provider.is_component_available("semgraph")):
                
                feedback_learner = self.engine.provider.get_component("feedback_learner")
                semgraph = self.engine.provider.get_component("semgraph")
                
                # Нормализуем фидбек к диапазону [-1, 1]
                normalized_feedback = max(-1.0, min(1.0, feedback / 5.0))
                
                metrics = feedback_learner.apply_feedback(semgraph, response, normalized_feedback)
                self.logger.info(f"🔄 Применен фидбек {feedback}: {metrics}")
                
        except Exception as e:
            self.logger.error(f"Ошибка применения фидбека: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Получение статистики системы."""
        stats = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "uptime": time.time(),
            "components_status": {},
            "pipeline_stats": {},
            "learning_quality": {},
            "tuner_stats": {
                "best_score": self.tuner.best_score,
                "mutation_rate": self.tuner.mutation_rate,
                "iterations_without_improvement": self.tuner.iterations_without_improvement
            }
        }
        
        # Статус компонентов
        for comp_name in ["memory", "semgraph", "nlp", "feedback_learner"]:
            if self.engine.provider.is_component_available(comp_name):
                try:
                    comp = self.engine.provider.get_component(comp_name)
                    if hasattr(comp, "get_statistics"):
                        stats["components_status"][comp_name] = comp.get_statistics()
                    else:
                        stats["components_status"][comp_name] = {"status": "available"}
                except Exception as e:
                    stats["components_status"][comp_name] = {"status": "error", "error": str(e)}
            else:
                stats["components_status"][comp_name] = {"status": "unavailable"}
        
        # Статистика гибридного конвейера
        try:
            stats["pipeline_stats"] = self.hybrid_pipeline.get_pipeline_stats()
        except Exception as e:
            stats["pipeline_stats"] = {"error": str(e)}
        
        # Качество обучения
        try:
            if self.engine.provider.is_component_available("learning_quality_analyzer"):
                analyzer = self.engine.provider.get_component("learning_quality_analyzer")
                stats["learning_quality"] = analyzer.get_quality_report()
        except Exception as e:
            stats["learning_quality"] = {"error": str(e)}
        
        return stats
    
    def get_learning_quality_report(self, max_entries: int = 10) -> Dict[str, Any]:
        """Получение отчета о качестве обучения."""
        try:
            if self.engine.provider.is_component_available("learning_quality_analyzer"):
                analyzer = self.engine.provider.get_component("learning_quality_analyzer")
                return analyzer.get_quality_report(max_entries)
            return {"message": "Анализатор качества обучения недоступен"}
        except Exception as e:
            self.logger.error(f"Ошибка получения отчета: {e}", exc_info=True)
            return {"error": str(e)}
    
    def reset_session(self) -> bool:
        """Сброс текущей сессии."""
        try:
            # Сохраняем текущее состояние
            self._save_session_state()
            
            # Создаем новую сессию
            self.session_id = f"{self.user_id}_{int(time.time())}"
            
            # Сбрасываем статистику конвейера
            self.hybrid_pipeline.stats = {
                "graph_responses": 0,
                "openai_fallbacks": 0,
                "learning_sessions": 0,
                "total_queries": 0
            }
            
            # Сбрасываем тюнер
            self.tuner.best_score = 0.0
            self.tuner.iterations_without_improvement = 0
            self.tuner.mutation_rate = 0.1
            
            self.logger.info(f"Создана новая сессия: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сброса сессии: {e}")
            return False
    
    def configure_hybrid_pipeline(self, min_confidence: float = None, 
                                 min_response_length: int = None) -> bool:
        """Настройка параметров гибридного конвейера."""
        try:
            if min_confidence is not None:
                self.hybrid_pipeline.min_confidence_threshold = max(0.1, min(0.9, min_confidence))
            
            if min_response_length is not None:
                self.hybrid_pipeline.min_response_length = max(5, min(200, min_response_length))
            
            self.logger.info(f"Обновлены параметры конвейера: "
                           f"confidence={self.hybrid_pipeline.min_confidence_threshold}, "
                           f"length={self.hybrid_pipeline.min_response_length}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки конвейера: {e}")
            return False
    
    def shutdown(self):
        """Корректное завершение работы."""
        try:
            self.logger.info("🚪 Завершение работы NeuroDesk...")
            
            # Сохраняем состояние
            self._save_session_state()
            
            # Завершаем движок
            self.engine.shutdown()
            
            # Финальная статистика
            final_stats = self.get_system_stats()
            pipeline_stats = final_stats.get("pipeline_stats", {})
            
            if pipeline_stats:
                self.logger.info(
                    f"📊 Финальная статистика: "
                    f"граф={pipeline_stats.get('graph_responses', 0)}, "
                    f"OpenAI={pipeline_stats.get('openai_fallbacks', 0)}, "
                    f"обучение={pipeline_stats.get('learning_sessions', 0)}"
                )
            
            self.logger.info("✅ NeuroDesk завершен корректно")
            
        except Exception as e:
            self.logger.error(f"Ошибка при завершении: {e}")

# Удобные функции для создания экземпляров
def create_neurodesk(user_id: str = "default", 
                    config_path: str = None,
                    openai_api_key: str = None) -> NeuroDesk:
    """
    Создание экземпляра NeuroDesk с гибридной архитектурой.
    
    Args:
        user_id: Идентификатор пользователя
        config_path: Путь к файлу конфигурации
        openai_api_key: API ключ OpenAI (или переменная окружения OPENAI_API_KEY)
    
    Returns:
        Настроенный экземпляр NeuroDesk
    """
    return NeuroDesk(user_id=user_id, config_path=config_path, openai_api_key=openai_api_key)

def create_lightweight_neurodesk(user_id: str = "default", 
                                openai_api_key: str = None) -> NeuroDesk:
    """
    Создание легковесного экземпляра NeuroDesk.
    
    Args:
        user_id: Идентификатор пользователя  
        openai_api_key: API ключ OpenAI
    
    Returns:
        Легковесный экземпляр NeuroDesk
    """
    # Можно добавить специальную легковесную конфигурацию
    return NeuroDesk(user_id=user_id, openai_api_key=openai_api_key)