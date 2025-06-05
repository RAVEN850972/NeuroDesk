import os
import json
import uuid
import random
import copy
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from neurograph.core.logging import get_logger
import time
import pickle

from openai import OpenAI
from neurograph.integration import NeuroGraphEngine
from neurograph.integration.base import IntegrationConfig, ProcessingResponse
from context_aware_learning_adapter import ContextAwareLearningAdapter
from graph_consolidation import GraphConsolidator
from feedback_learning import FeedbackLearner
from learning_quality_analyzer import LearningQualityAnalyzer

# Ключ OpenAI должен быть в переменной окружения
SESSION_DIR = Path("./sessions")
SESSION_DIR.mkdir(exist_ok=True)

# ----------------------- ГЕНЕТИЧЕСКИЙ ТЮНЕР -----------------------

class PipelineTuner:
    def __init__(self, engine: NeuroGraphEngine):
        self.engine = engine
        self.logger = get_logger("PipelineTuner")
        base_config = engine.config.to_dict()  # Преобразуем конфигурацию в словарь
        
        # Используем выборочное копирование, исключая несериализуемые объекты
        self.best_config = self._safe_copy(base_config)
        self.best_score = 0.0
        self.mutation_rate = 0.1
        self.iterations_without_improvement = 0
    
    def _safe_copy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Безопасное копирование конфигурации, исключая несериализуемые объекты."""
        safe_config = {}
        for key, value in config.items():
            try:
                # Проверяем, можно ли сериализовать объект
                pickle.dumps(value)
                safe_config[key] = copy.deepcopy(value)
            except (pickle.PicklingError, TypeError, AttributeError):
                self.logger.warning(f"Skipping non-serializable config key: {key}")
                # Можно задать значение по умолчанию или пропустить
                safe_config[key] = None
        return safe_config
    
    def mutate(self) -> dict[str, Any]:
        config_copy = self._safe_copy(self.best_config)
        for key in config_copy.get("mutations", {}):
            if random.random() < self.mutation_rate:
                value = config_copy["mutations"][key]
                if isinstance(value, bool):
                    config_copy["mutations"][key] = not value
                elif isinstance(value, (int, float)):
                    config_copy["mutations"][value] += random.uniform(-0.1 * value, 0.1 * value)
        return config_copy

    def evaluate(self, response: any, feedback: int = None) -> float:
        if not response.success:
            return 0.0
        score = response.confidence or 0.0
        if feedback is not None:
            score += feedback * 0.2
        return min(1.0, max(0.0, score))

    def update(self, config: dict[str, any], score: float):
        if score > self.best_score:
            self.best_score = score
            self.best_config = self._safe_copy(config)
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        if self.iterations_without_improvement > 10:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
        elif self.iterations_without_improvement == 0:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)

# ----------------------- ГЛАВНЫЙ КЛАСС NEURODESK -----------------------

class NeuroDesk:
    """Персональный AI-ассистент с гибридной архитектурой, автотюнингом и фидбеком."""

    def __init__(self, user_id: str = "anon", config_path: str = None):
        self.user_id = user_id
        self.session_id = f"{user_id}_{int(time.time())}"
        self.logger = get_logger("NeuroDesk")
        self.config = IntegrationConfig.load_config(config_path) if config_path else IntegrationConfig()
        self.engine = NeuroGraphEngine(self.config)
        self.tuner = PipelineTuner(self.engine)
        
        # Регистрация компонентов
        self.engine.provider.register_lazy_component("context_aware_learning", lambda: ContextAwareLearningAdapter())
        self.engine.provider.register_lazy_component("graph_consolidator", lambda: GraphConsolidator())
        self.engine.provider.register_lazy_component("feedback_learner", lambda: FeedbackLearner())
        self.engine.provider.register_lazy_component("learning_quality_analyzer", lambda: LearningQualityAnalyzer())
        
        self._load_session_state()
        self._start_background_consolidation()
    
    async def _background_consolidation(self):
        """Фоновая задача консолидации графа."""
        while True:
            try:
                if self.engine.provider.is_component_available("graph_consolidator") and \
                   self.engine.provider.is_component_available("semgraph"):
                    consolidator = self.engine.provider.get_component("graph_consolidator")  # Исправленное имя
                    semgraph = self.engine.provider.get_component("semgraph")
                    metrics = consolidator.consolidate(semgraph)
                    self.logger.info(f"Результаты консолидации: {metrics}")
                else:
                    self.logger.warning("Компоненты graph_consolidator или semgraph недоступны")
            except Exception as e:
                self.logger.error(f"Ошибка фоновой консолидации: {e}", exc_info=True)
            await asyncio.sleep(3600)  # Консолидация каждый час
    
    def _start_background_consolidation(self):
        """Запуск фоновой задачи."""
        asyncio.create_task(self._background_consolidation())

    def _load_or_create_session(self) -> str:
        session_file = SESSION_DIR / f"{self.user_id}.json"
        if session_file.exists():
            with open(session_file, "r") as f:
                return json.load(f).get("session_id", str(uuid.uuid4()))
        else:
            session_id = str(uuid.uuid4())
            with open(session_file, "w") as f:
                json.dump({"session_id": session_id}, f)
            return session_id

    def _load_config(self, config_path: str = None) -> IntegrationConfig:
        from neurograph.integration.config import IntegrationConfigManager
        manager = IntegrationConfigManager()

        if config_path and os.path.exists(config_path):
            return manager.load_config(config_path)
        else:
            config = manager.create_template_config("default")
            config["components"]["memory"]["params"] = {
                "stm_capacity": 100,
                "ltm_capacity": 5000
            }
            config["components"]["semgraph"] = {
                "type": "persistent",
                "params": {
                    "file_path": f"graph_{self.user_id}.json",
                    "auto_save_interval": 300.0
                }
            }
            return IntegrationConfig(**config)

    def _save_session_state(self):
        self.logger.info("📂 Сохраняем состояние графа и памяти")
        if hasattr(self.engine, "provider"):
            for comp_name in ["memory", "semgraph"]:
                if self.engine.provider.is_component_available(comp_name):
                    comp = self.engine.provider.get_component(comp_name)
                    if hasattr(comp, "save"):
                        try:
                            comp.save()
                        except Exception as e:
                            self.logger.warning(f"⚠️ Ошибка сохранения {comp_name}: {e}")

    def ask(self, prompt: str, explanation_level="detailed", min_confidence=0.4, feedback: int = None, **kwargs) -> str:
        try:
            mutated_cfg = self.tuner.mutate()
            self.engine.shutdown()
            self.engine.initialize(IntegrationConfig(**mutated_cfg))

            response = self.engine.query(prompt, session_id=self.session_id,
                                        explanation_level=explanation_level, **kwargs)

            score = self.tuner.evaluate(response, feedback=feedback)
            self.tuner.update(mutated_cfg, score)

            # Применяем фидбек, если он есть
            if feedback is not None and self.engine.provider.is_component_available("feedback_learner") and \
            self.engine.provider.is_component_available("semgraph"):
                feedback_learner = self.engine.provider.get_component("feedback_learner")
                semgraph = self.engine.provider.get_component("semgraph")
                feedback_metrics = feedback_learner.apply_feedback(semgraph, response, feedback)
                self.logger.info(f"Фидбек обработан: {feedback_metrics}")

            self._save_session_state()

            if response.success and response.primary_response.strip() and response.confidence >= min_confidence:
                return self._format_response(response)

            fallback = self._ask_openai(prompt)
            self.learn(fallback)
            return f"🤖 (ответ нейросети): {fallback}"

        except Exception as e:
            self.logger.error(f"Ошибка запроса: {e}", exc_info=True)
            return "⚠️ Ошибка при обработке."

    def learn(self, text: str, explanation_level="basic", context: Dict[str, Any] = None, **kwargs) -> str:
        try:
            if context is None:
                context = {"domain": "general"}
            
            response = self.engine.learn(
                text,
                session_id=self.session_id,
                explanation_level=explanation_level,
                context=context,
                request_type="context_aware_learning",
                **kwargs
            )
            self._save_session_state()
            return self._format_response(response)
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}", exc_info=True)
            return "⚠️ Не удалось обучиться на этом фрагменте."

    def reply(self, message: str, **kwargs) -> str:
        try:
            response = self.engine.process_text(message, session_id=self.session_id, **kwargs)
            self._save_session_state()
            return self._format_response(response)
        except Exception as e:
            self.logger.error(f"Ошибка обработки текста: {e}", exc_info=True)
            return "⚠️ Не удалось обработать сообщение."

    def shutdown(self):
        self._save_session_state()
        self.engine.shutdown()
        self.logger.info("🚩 Сессия завершена.")

    def _format_response(self, response: ProcessingResponse) -> str:
        if not response.success:
            return f"⚠️ Ошибка: {response.error_message or 'неизвестная'}"

        result = f"🧠 {response.primary_response.strip()}"
        if response.explanation:
            result += "\n\n💡 Объяснение:\n" + "\n".join(f"- {e}" for e in response.explanation[:3])
        return result

    def _ask_openai(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI Error: {e}"
        
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
