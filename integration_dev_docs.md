# NeuroGraph Integration Module - Developer Documentation

## Оглавление

1. [Введение](#введение)
2. [Быстрый старт](#быстрый-старт)
3. [Архитектура модуля](#архитектура-модуля)
4. [Основные интерфейсы](#основные-интерфейсы)
5. [Движок NeuroGraph](#движок-neurograph)
6. [Провайдер компонентов](#провайдер-компонентов)
7. [Конвейеры обработки](#конвейеры-обработки)
8. [Система адаптеров](#система-адаптеров)
9. [Конфигурация системы](#конфигурация-системы)
10. [Мониторинг и диагностика](#мониторинг-и-диагностика)
11. [Расширение системы](#расширение-системы)
12. [Примеры использования](#примеры-использования)
13. [Лучшие практики](#лучшие-практики)
14. [FAQ](#faq)

---

## Введение

Integration Module является центральным компонентом системы NeuroGraph, обеспечивающим координацию всех остальных модулей. Он предоставляет единый интерфейс для работы с нейросимволической системой и управляет жизненным циклом всех компонентов.

### Основные задачи модуля

- **Оркестрация**: координация работы всех модулей системы
- **Маршрутизация**: направление запросов к соответствующим конвейерам обработки
- **Интеграция**: обеспечение взаимодействия между разнородными компонентами
- **Мониторинг**: отслеживание состояния и производительности системы
- **Конфигурация**: управление настройками всей системы

### Ключевые преимущества

- **Модульность**: легко добавлять новые компоненты без изменения существующих
- **Гибкость**: настройка системы под различные сценарии использования
- **Масштабируемость**: поддержка различных конфигураций от легковесных до высокопроизводительных
- **Надежность**: встроенная обработка ошибок и система мониторинга

---

## Быстрый старт

### Установка

```bash
pip install neurograph-integration
```

### Первый пример

```python
from neurograph.integration import create_default_engine

# Создание движка с настройками по умолчанию
engine = create_default_engine()

# Простой запрос
response = engine.process_text("Python - это язык программирования")
print(response.primary_response)

# Запрос к системе знаний
response = engine.query("Что такое Python?")
print(response.primary_response)

# Завершение работы
engine.shutdown()
```

### Интерактивный режим

```bash
python -m neurograph.integration --interactive
```

### Запуск демонстраций

```bash
# Базовые примеры
python -m neurograph.integration --demo basic

# Продвинутые примеры
python -m neurograph.integration --demo advanced

# Все демонстрации
python -m neurograph.integration --demo all
```

---

## Архитектура модуля

### Слоистая структура

```
┌─────────────────────────────────────────┐
│         Application Layer                │
│    (Пользовательские приложения)         │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Integration Layer                │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Engine    │  │   Pipelines     │   │
│  └─────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Provider   │  │    Adapters     │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Component Layer                  │
│  NLP │ SemGraph │ Memory │ Processor    │
└─────────────────────────────────────────┘
```

### Основные компоненты

1. **NeuroGraphEngine** - центральный движок системы
2. **ComponentProvider** - поставщик компонентов с ленивой инициализацией
3. **Pipelines** - конвейеры обработки различных типов запросов
4. **Adapters** - адаптеры для интеграции между модулями
5. **Configuration** - система конфигурации
6. **Monitoring** - система мониторинга и диагностики

---

## Основные интерфейсы

### INeuroGraphEngine

Главный интерфейс движка системы.

```python
from abc import ABC, abstractmethod
from neurograph.integration.base import ProcessingRequest, ProcessingResponse, IntegrationConfig

class INeuroGraphEngine(ABC):
    """Главный интерфейс движка NeuroGraph."""
    
    @abstractmethod
    def initialize(self, config: IntegrationConfig) -> bool:
        """Инициализация движка с указанной конфигурацией."""
        pass
    
    @abstractmethod
    def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Обработка структурированного запроса."""
        pass
    
    @abstractmethod
    def process_text(self, text: str, **kwargs) -> ProcessingResponse:
        """Упрощенная обработка текста."""
        pass
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> ProcessingResponse:
        """Выполнение запроса к системе знаний."""
        pass
    
    @abstractmethod
    def learn(self, content: str, **kwargs) -> ProcessingResponse:
        """Обучение системы на новом контенте."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Получение состояния здоровья системы."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Корректное завершение работы."""
        pass
```

### IComponentProvider

Интерфейс поставщика компонентов.

```python
class IComponentProvider(ABC):
    """Интерфейс провайдера компонентов системы."""
    
    @abstractmethod
    def get_component(self, component_type: str, **kwargs) -> Any:
        """Получение компонента по типу."""
        pass
    
    @abstractmethod
    def register_component(self, component_type: str, component: Any) -> None:
        """Регистрация компонента."""
        pass
    
    @abstractmethod
    def is_component_available(self, component_type: str) -> bool:
        """Проверка доступности компонента."""
        pass
    
    @abstractmethod
    def get_component_health(self, component_type: str) -> Dict[str, Any]:
        """Получение состояния здоровья компонента."""
        pass
```

### IPipeline

Интерфейс конвейера обработки.

```python
class IPipeline(ABC):
    """Интерфейс конвейера обработки."""
    
    @abstractmethod
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка запроса через конвейер."""
        pass
    
    @abstractmethod
    def validate_request(self, request: ProcessingRequest) -> tuple[bool, Optional[str]]:
        """Валидация запроса."""
        pass
    
    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Информация о конвейере."""
        pass
```

### Структуры данных

#### ProcessingRequest

```python
@dataclass
class ProcessingRequest:
    """Запрос на обработку в системе NeuroGraph."""
    
    # Основные данные
    content: str                                    # Текст или запрос
    request_type: str = "query"                     # Тип запроса
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None                # ID сессии пользователя
    
    # Параметры обработки
    mode: ProcessingMode = ProcessingMode.SYNCHRONOUS
    response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL
    max_processing_time: float = 30.0               # Максимальное время обработки
    
    # Контекст и настройки
    context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Компоненты для использования
    enable_nlp: bool = True
    enable_memory_search: bool = True
    enable_graph_reasoning: bool = True
    enable_vector_search: bool = True
    enable_logical_inference: bool = True
    
    # Параметры качества
    confidence_threshold: float = 0.5
    max_results: int = 10
    explanation_level: str = "basic"                # basic, detailed, debug
```

#### ProcessingResponse

```python
@dataclass
class ProcessingResponse:
    """Ответ системы NeuroGraph."""
    
    # Идентификация
    request_id: str
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Результаты
    success: bool = True
    primary_response: str = ""                      # Основной ответ
    structured_data: Dict[str, Any] = field(default_factory=dict)
    
    # Детали обработки
    processing_time: float = 0.0
    components_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Пояснения и контекст
    explanation: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Мета-информация
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
```

---

## Движок NeuroGraph

### Создание движка

#### Использование фабричных методов

```python
from neurograph.integration import (
    create_default_engine,
    create_lightweight_engine,
    create_research_engine
)

# Настройки по умолчанию
engine = create_default_engine()

# Облегченная конфигурация
lightweight_engine = create_lightweight_engine()

# Исследовательская конфигурация
research_engine = create_research_engine()
```

#### Создание с кастомной конфигурацией

```python
from neurograph.integration import IntegrationConfig, NeuroGraphEngine

# Создание конфигурации
config = IntegrationConfig(
    engine_name="my_custom_engine",
    components={
        "semgraph": {
            "type": "memory_efficient",
            "params": {}
        },
        "memory": {
            "params": {
                "stm_capacity": 50,
                "ltm_capacity": 5000
            }
        }
    },
    max_concurrent_requests=5,
    enable_metrics=True
)

# Создание и инициализация движка
engine = NeuroGraphEngine()
if engine.initialize(config):
    print("Движок успешно инициализирован")
else:
    print("Ошибка инициализации")
```

### Основные методы движка

#### process_text() - Обработка текста

```python
# Простая обработка
response = engine.process_text("Искусственный интеллект развивается быстро")

# С дополнительными параметрами
response = engine.process_text(
    "Python используется в машинном обучении",
    response_format=ResponseFormat.STRUCTURED,
    explanation_level="detailed",
    enable_vector_search=True
)

print(f"Ответ: {response.primary_response}")
print(f"Использованные компоненты: {response.components_used}")
print(f"Время обработки: {response.processing_time:.3f}с")
```

#### query() - Запрос к системе знаний

```python
# Простой запрос
response = engine.query("Что такое машинное обучение?")

# Запрос с контекстом
response = engine.query(
    "Какие алгоритмы лучше использовать?",
    context={"domain": "computer_vision", "task": "classification"},
    max_results=5
)

# Анализ результата
if response.success:
    print(f"Ответ: {response.primary_response}")
    
    # Структурированные данные
    if "graph_search" in response.structured_data:
        found_nodes = response.structured_data["graph_search"]["found_nodes"]
        print(f"Найдено в графе: {found_nodes}")
    
    # Объяснения
    for explanation in response.explanation:
        print(f"- {explanation}")
else:
    print(f"Ошибка: {response.error_message}")
```

#### learn() - Обучение системы

```python
# Обучение на новой информации
response = engine.learn(
    "TensorFlow - это библиотека для машинного обучения, разработанная Google"
)

# Проверка результата обучения
if response.success:
    print(f"Система изучила: {response.primary_response}")
    
    # Детали обучения
    learning_data = response.structured_data.get("learning", {})
    rules_created = learning_data.get("rules_created", {}).get("count", 0)
    print(f"Создано правил: {rules_created}")
```

### Работа с конфигурацией

#### Загрузка из файла

```python
from neurograph.integration.config import IntegrationConfigManager

config_manager = IntegrationConfigManager()

# Загрузка из JSON файла
config = config_manager.load_config("my_config.json")
engine = NeuroGraphEngine()
engine.initialize(config)

# Сохранение конфигурации
config_manager.save_config(config, "backup_config.json")
```

#### Создание шаблонов

```python
# Создание шаблона конфигурации
template = config_manager.create_template_config("research")

# Модификация шаблона
template["max_concurrent_requests"] = 50
template["components"]["memory"]["params"]["ltm_capacity"] = 100000

# Создание конфигурации из словаря
config = IntegrationConfig(**template)
```

---

## Провайдер компонентов

### ComponentProvider

Управляет жизненным циклом всех компонентов системы.

#### Базовое использование

```python
from neurograph.integration import ComponentProvider

provider = ComponentProvider()

# Регистрация компонента
from my_module import MyCustomComponent
custom_component = MyCustomComponent()
provider.register_component("my_component", custom_component)

# Получение компонента
component = provider.get_component("semgraph")

# Проверка доступности
if provider.is_component_available("nlp"):
    nlp = provider.get_component("nlp")
```

#### Ленивая инициализация

```python
def create_custom_nlp():
    """Функция создания NLP компонента."""
    from my_nlp import CustomNLPProcessor
    return CustomNLPProcessor(language="ru")

# Регистрация ленивого компонента
provider.register_lazy_component("nlp", create_custom_nlp)

# Компонент будет создан только при первом обращении
nlp = provider.get_component("nlp")  # Здесь происходит инициализация
```

#### Мониторинг здоровья компонентов

```python
# Проверка здоровья конкретного компонента
health = provider.get_component_health("memory")
print(f"Статус: {health['status']}")
print(f"Последняя проверка: {health['last_check']}")

# Проверка всех компонентов
all_status = provider.get_all_components_status()
for component_name, status in all_status.items():
    print(f"{component_name}: {status['status']}")
```

### Создание кастомного провайдера

```python
from neurograph.integration.base import IComponentProvider

class CustomComponentProvider(IComponentProvider):
    def __init__(self):
        self.components = {}
        self.component_factories = {}
    
    def get_component(self, component_type: str, **kwargs):
        if component_type not in self.components:
            if component_type in self.component_factories:
                factory = self.component_factories[component_type]
                self.components[component_type] = factory(**kwargs)
            else:
                raise ValueError(f"Component {component_type} not found")
        
        return self.components[component_type]
    
    def register_component(self, component_type: str, component):
        self.components[component_type] = component
    
    def register_factory(self, component_type: str, factory_func):
        self.component_factories[component_type] = factory_func
    
    def is_component_available(self, component_type: str) -> bool:
        return (component_type in self.components or 
                component_type in self.component_factories)
    
    def get_component_health(self, component_type: str):
        if not self.is_component_available(component_type):
            return {"status": "unknown", "message": "Component not found"}
        
        return {"status": "healthy", "last_check": time.time()}

# Использование кастомного провайдера
custom_provider = CustomComponentProvider()
engine = NeuroGraphEngine(custom_provider)
```

---

## Конвейеры обработки

### Встроенные конвейеры

#### TextProcessingPipeline

Для обработки и изучения произвольного текста.

```python
from neurograph.integration.pipelines import TextProcessingPipeline
from neurograph.integration.base import ProcessingRequest

pipeline = TextProcessingPipeline()

# Создание запроса
request = ProcessingRequest(
    content="Квантовые компьютеры используют принципы квантовой механики",
    request_type="text_processing",
    enable_nlp=True,
    enable_graph_reasoning=True,
    enable_vector_search=True,
    enable_memory_search=True
)

# Обработка через конвейер
response = pipeline.process(request, provider)

# Анализ результата
print(f"Успешность: {response.success}")
print(f"Компоненты: {response.components_used}")

# Структурированные данные
nlp_data = response.structured_data.get("nlp", {})
print(f"Найдено сущностей: {len(nlp_data.get('entities', []))}")

graph_data = response.structured_data.get("graph", {})
print(f"Добавлено узлов в граф: {graph_data.get('nodes_added', 0)}")
```

#### QueryProcessingPipeline

Для обработки пользовательских запросов.

```python
from neurograph.integration.pipelines import QueryProcessingPipeline

pipeline = QueryProcessingPipeline()

request = ProcessingRequest(
    content="Расскажи о квантовых компьютерах",
    request_type="query",
    response_format=ResponseFormat.CONVERSATIONAL,
    explanation_level="detailed"
)

response = pipeline.process(request, provider)

print(f"Ответ: {response.primary_response}")

# Детали поиска
graph_search = response.structured_data.get("graph_search", {})
found_nodes = graph_search.get("found_nodes", [])
print(f"Найдено в графе: {found_nodes}")

# Результаты распространения активации
propagation = response.structured_data.get("propagation", {})
activated = propagation.get("activated_concepts", [])
print(f"Активированные концепты: {[c['concept'] for c in activated[:3]]}")
```

#### LearningPipeline

Для обучения с созданием правил и консолидацией.

```python
from neurograph.integration.pipelines import LearningPipeline

pipeline = LearningPipeline()

request = ProcessingRequest(
    content="Если система имеет квантовые свойства, то она может находиться в суперпозиции состояний",
    request_type="learning"
)

response = pipeline.process(request, provider)

# Результаты обучения
learning_data = response.structured_data.get("learning", {})
rules_created = learning_data.get("rules_created", {}).get("count", 0)
print(f"Создано новых правил: {rules_created}")

consolidation = learning_data.get("memory_consolidation", {})
if consolidation.get("success"):
    consolidated = consolidation.get("consolidated", 0)
    print(f"Консолидировано элементов памяти: {consolidated}")
```

#### InferencePipeline

Для логического вывода и рассуждений.

```python
from neurograph.integration.pipelines import InferencePipeline

pipeline = InferencePipeline()

request = ProcessingRequest(
    content="Если квантовый компьютер находится в суперпозиции, какие выводы можно сделать?",
    request_type="inference",
    explanation_level="detailed"
)

response = pipeline.process(request, provider)

# Результаты вывода
inference = response.structured_data.get("inference", {})
conclusions = inference.get("conclusions", [])

print("Выводы:")
for conclusion in conclusions:
    conf = conclusion["confidence"]
    print(f"- {conclusion['conclusion']} (уверенность: {conf:.2f})")

# Объяснение процесса
explanation_steps = inference.get("explanation_steps", [])
print("\nШаги рассуждения:")
for step in explanation_steps:
    print(f"{step['step']}. {step['reasoning']}")
```

### Создание кастомного конвейера

```python
from neurograph.integration.base import IPipeline, ProcessingRequest, ProcessingResponse
import time

class CustomAnalyticsPipeline(IPipeline):
    """Кастомный конвейер для аналитической обработки."""
    
    def __init__(self):
        self.pipeline_name = "custom_analytics"
        self.processed_count = 0
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка запроса через аналитический конвейер."""
        start_time = time.time()
        
        try:
            # Валидация запроса
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_error_response(request, error_msg)
            
            # Аналитическая обработка
            analytics_result = self._perform_analytics(request, provider)
            
            # Создание ответа
            processing_time = time.time() - start_time
            self.processed_count += 1
            
            return ProcessingResponse(
                request_id=request.request_id,
                success=True,
                primary_response=f"Аналитический отчет готов",
                structured_data={"analytics": analytics_result},
                processing_time=processing_time,
                components_used=["custom_analytics"]
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_error_response(request, str(e))

# ❌ Плохо - игнорирование ошибок
def process_badly(self, request, provider):
    try:
        return main_processing(request, provider)
    except:
        return None  # Теряем информацию об ошибке
```

#### 3. Конфигурация
```python
# ✅ Хорошо - валидация конфигурации
class SafeConfig:
    def __init__(self, config_dict):
        self.max_requests = self._validate_positive_int(
            config_dict.get("max_requests", 10)
        )
        self.timeout = self._validate_positive_float(
            config_dict.get("timeout", 30.0)
        )
    
    def _validate_positive_int(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Expected positive integer, got {value}")
        return value

# ❌ Плохо - отсутствие валидации
class UnsafeConfig:
    def __init__(self, config_dict):
        self.max_requests = config_dict.get("max_requests", 10)  # Может быть None или отрицательным
        self.timeout = config_dict.get("timeout", 30.0)  # Может быть строкой
```

### Производительность

#### 1. Эффективное использование памяти
```python
# ✅ Хорошо - ленивая загрузка
class EfficientProvider(ComponentProvider):
    def get_component(self, component_type: str):
        if component_type not in self._loaded_components:
            self._loaded_components[component_type] = self._create_component(component_type)
        return self._loaded_components[component_type]

# ❌ Плохо - загрузка всех компонентов сразу
class InefficientProvider:
    def __init__(self):
        self.components = {
            "nlp": create_nlp(),      # Загружается всегда
            "graph": create_graph(),  # Даже если не используется
            "memory": create_memory()
        }
```

#### 2. Кеширование
```python
# ✅ Хорошо - интеллектуальное кеширование
from functools import lru_cache
import hashlib

class CachingPipeline(BasePipeline):
    def __init__(self):
        super().__init__("caching_pipeline")
        self._response_cache = {}
    
    def process(self, request, provider):
        # Создаем ключ кеша на основе содержимого
        cache_key = self._create_cache_key(request)
        
        if cache_key in self._response_cache:
            cached_response = self._response_cache[cache_key]
            # Проверяем, не устарел ли кеш
            if time.time() - cached_response["timestamp"] < 300:  # 5 минут
                return cached_response["response"]
        
        # Обрабатываем запрос
        response = super().process(request, provider)
        
        # Кешируем только успешные ответы
        if response.success:
            self._response_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
        
        return response
    
    def _create_cache_key(self, request):
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        return f"{request.request_type}_{content_hash}_{request.confidence_threshold}"

# ❌ Плохо - отсутствие кеширования
class NoCachingPipeline(BasePipeline):
    def process(self, request, provider):
        # Каждый раз обрабатываем заново, даже идентичные запросы
        return super().process(request, provider)
```

#### 3. Пакетная обработка
```python
# ✅ Хорошо - пакетная обработка
class BatchProcessor:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.pending_requests = []
    
    def add_request(self, request):
        self.pending_requests.append(request)
        
        if len(self.pending_requests) >= self.batch_size:
            return self.process_batch()
        
        return None
    
    def process_batch(self):
        if not self.pending_requests:
            return []
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        
        # Обработка всего пакета одновременно
        results = []
        for request in batch:
            result = self.process_single(request)
            results.append(result)
        
        return results

# ❌ Плохо - обработка по одному
class SingleProcessor:
    def process_requests(self, requests):
        results = []
        for request in requests:
            # Каждый запрос обрабатывается независимо
            result = self.process_single(request)
            results.append(result)
        return results
```

### Тестирование

#### 1. Модульное тестирование
```python
import unittest
from unittest.mock import Mock, MagicMock

class TestCustomPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = CustomPipeline()
        self.mock_provider = Mock(spec=IComponentProvider)
    
    def test_successful_processing(self):
        # Настройка мока
        mock_nlp = Mock()
        mock_nlp.process_text.return_value = Mock(entities=[], relations=[])
        self.mock_provider.get_component.return_value = mock_nlp
        self.mock_provider.is_component_available.return_value = True
        
        # Создание тестового запроса
        request = ProcessingRequest(
            content="Test content",
            request_type="test"
        )
        
        # Выполнение
        response = self.pipeline.process(request, self.mock_provider)
        
        # Проверки
        self.assertTrue(response.success)
        self.assertIsNotNone(response.primary_response)
        self.mock_provider.get_component.assert_called_with("nlp")
    
    def test_error_handling(self):
        # Настройка мока для ошибки
        self.mock_provider.get_component.side_effect = Exception("Component error")
        self.mock_provider.is_component_available.return_value = True
        
        request = ProcessingRequest(content="Test content")
        response = self.pipeline.process(request, self.mock_provider)
        
        # Проверка обработки ошибки
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error_message)
```

#### 2. Интеграционное тестирование
```python
class TestIntegrationFlow(unittest.TestCase):
    def setUp(self):
        self.engine = create_lightweight_engine()  # Используем легковесную конфигурацию для тестов
    
    def tearDown(self):
        self.engine.shutdown()
    
    def test_learning_and_query_flow(self):
        # Обучение
        learn_response = self.engine.learn("Python - это язык программирования")
        self.assertTrue(learn_response.success)
        
        # Проверяем, что система изучила информацию
        query_response = self.engine.query("Что такое Python?")
        self.assertTrue(query_response.success)
        self.assertIn("Python", query_response.primary_response)
    
    def test_complex_processing_pipeline(self):
        # Сложный сценарий с несколькими этапами
        
        # 1. Добавляем знания
        self.engine.learn("Машинное обучение использует алгоритмы для анализа данных")
        
        # 2. Запрашиваем информацию
        response = self.engine.query("Расскажи об алгоритмах машинного обучения")
        
        # 3. Проверяем использование компонентов
        self.assertIn("nlp", response.components_used)
        self.assertIn("semgraph", response.components_used)
        
        # 4. Проверяем качество ответа
        self.assertGreater(len(response.primary_response), 20)
        self.assertGreater(response.confidence, 0.0)
```

### Мониторинг и диагностика

#### 1. Структурированное логирование
```python
# ✅ Хорошо - структурированные логи
class StructuredLoggingPipeline(BasePipeline):
    def process(self, request, provider):
        self.logger.info(
            "Pipeline processing started",
            extra={
                "request_id": request.request_id,
                "pipeline": self.pipeline_name,
                "content_length": len(request.content),
                "request_type": request.request_type
            }
        )
        
        start_time = time.time()
        try:
            response = super().process(request, provider)
            
            self.logger.info(
                "Pipeline processing completed",
                extra={
                    "request_id": request.request_id,
                    "success": response.success,
                    "processing_time": time.time() - start_time,
                    "components_used": response.components_used
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Pipeline processing failed",
                extra={
                    "request_id": request.request_id,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                },
                exc_info=True
            )
            raise

# ❌ Плохо - неструктурированные логи
class BadLoggingPipeline(BasePipeline):
    def process(self, request, provider):
        print(f"Processing {request.content}")  # Неинформативно
        
        try:
            response = super().process(request, provider)
            print("Done")  # Нет деталей
            return response
        except Exception as e:
            print(f"Error: {e}")  # Нет контекста
            raise
```

#### 2. Метрики производительности
```python
# ✅ Хорошо - детальные метрики
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.request_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.component_usage = defaultdict(int)
    
    def record_request(self, pipeline_name, processing_time, success, components_used):
        self.request_times[pipeline_name].append(processing_time)
        
        if not success:
            self.error_counts[pipeline_name] += 1
        
        for component in components_used:
            self.component_usage[component] += 1
    
    def get_metrics_summary(self):
        summary = {}
        
        for pipeline, times in self.request_times.items():
            summary[pipeline] = {
                "requests_count": len(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "error_count": self.error_counts[pipeline],
                "error_rate": self.error_counts[pipeline] / len(times)
            }
        
        summary["component_usage"] = dict(self.component_usage)
        return summary

# Интеграция с конвейерами
class MetricsPipeline(BasePipeline):
    def __init__(self, metrics_collector):
        super().__init__("metrics_pipeline")
        self.metrics = metrics_collector
    
    def process(self, request, provider):
        start_time = time.time()
        
        try:
            response = super().process(request, provider)
            processing_time = time.time() - start_time
            
            self.metrics.record_request(
                self.pipeline_name,
                processing_time,
                response.success,
                response.components_used
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.metrics.record_request(
                self.pipeline_name,
                processing_time,
                False,
                []
            )
            
            raise
```

### Безопасность

#### 1. Валидация входных данных
```python
# ✅ Хорошо - тщательная валидация
class SecurePipeline(BasePipeline):
    MAX_CONTENT_LENGTH = 50000
    ALLOWED_REQUEST_TYPES = ["query", "learning", "inference"]
    
    def validate_request(self, request):
        # Проверка типа запроса
        if request.request_type not in self.ALLOWED_REQUEST_TYPES:
            return False, f"Недопустимый тип запроса: {request.request_type}"
        
        # Проверка длины контента
        if len(request.content) > self.MAX_CONTENT_LENGTH:
            return False, f"Контент слишком длинный: {len(request.content)} символов"
        
        # Проверка на вредоносный контент
        if self._contains_malicious_content(request.content):
            return False, "Обнаружен потенциально вредоносный контент"
        
        # Проверка rate limiting
        if not self._check_rate_limit(request.session_id):
            return False, "Превышен лимит запросов"
        
        return True, None
    
    def _contains_malicious_content(self, content):
        # Простые проверки на SQL injection, XSS и т.д.
        malicious_patterns = [
            "DROP TABLE", "SELECT * FROM", "<script>", "javascript:",
            "eval(", "document.cookie", "window.location"
        ]
        
        content_lower = content.lower()
        return any(pattern.lower() in content_lower for pattern in malicious_patterns)
    
    def _check_rate_limit(self, session_id):
        # Реализация rate limiting
        # В реальной системе это было бы более сложно
        return True

# ❌ Плохо - отсутствие валидации
class UnsecurePipeline(BasePipeline):
    def validate_request(self, request):
        return True, None  # Принимаем любой запрос
```

#### 2. Санитизация данных
```python
# ✅ Хорошо - санитизация данных
import re
import html

class DataSanitizer:
    @staticmethod
    def sanitize_text(text):
        if not isinstance(text, str):
            return str(text)
        
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # Экранируем HTML символы
        text = html.escape(text)
        
        # Удаляем потенциально опасные символы
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]"\']', '', text)
        
        # Ограничиваем длину
        if len(text) > 10000:
            text = text[:10000] + "..."
        
        return text.strip()
    
    @staticmethod
    def sanitize_metadata(metadata):
        if not isinstance(metadata, dict):
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            # Санитизируем ключи
            safe_key = re.sub(r'[^\w_]', '', str(key))[:50]
            
            # Санитизируем значения
            if isinstance(value, str):
                safe_value = DataSanitizer.sanitize_text(value)
            elif isinstance(value, (int, float, bool)):
                safe_value = value
            else:
                safe_value = str(value)[:100]
            
            sanitized[safe_key] = safe_value
        
        return sanitized

# Использование в конвейере
class SanitizedPipeline(BasePipeline):
    def process(self, request, provider):
        # Санитизируем входные данные
        safe_request = ProcessingRequest(
            content=DataSanitizer.sanitize_text(request.content),
            request_type=request.request_type,
            context=DataSanitizer.sanitize_metadata(request.context),
            metadata=DataSanitizer.sanitize_metadata(request.metadata)
        )
        
        return super().process(safe_request, provider)
```

---

## FAQ

### Общие вопросы

**Q: Можно ли использовать NeuroGraph Integration без всех модулей?**
A: Да, система спроектирована модульно. Вы можете использовать только необходимые компоненты:

```python
# Минимальная конфигурация только с NLP
config = IntegrationConfig(
    components={
        "nlp": {"params": {"language": "ru"}}
    }
)

engine = NeuroGraphEngine()
engine.initialize(config)

# Система будет работать только с доступными компонентами
response = engine.process_text("Тестовый текст")
```

**Q: Как добавить поддержку нового языка?**
A: Настройте NLP компонент для нужного языка:

```python
config = IntegrationConfig(
    components={
        "nlp": {
            "params": {
                "language": "en",  # или "de", "fr", etc.
                "use_spacy": True
            }
        }
    }
)
```

**Q: Можно ли использовать собственную базу данных для хранения?**
A: Да, создайте кастомный адаптер:

```python
class CustomDBAdapter(IComponentAdapter):
    def __init__(self, db_connection):
        self.db = db_connection
    
    def adapt(self, source_data, target_format):
        # Ваша логика интеграции с БД
        pass
```

### Производительность

**Q: Какие требования к ресурсам у системы?**
A: Зависит от конфигурации:

- **Lightweight**: 512MB RAM, 1 CPU core
- **Default**: 2GB RAM, 2 CPU cores  
- **Research**: 8GB+ RAM, 4+ CPU cores

**Q: Как оптимизировать производительность?**
A: Несколько стратегий:

```python
# 1. Используйте кеширование
config.enable_caching = True
config.cache_ttl = 600

# 2. Ограничьте параллелизм
config.max_concurrent_requests = 5

# 3. Настройте размеры памяти
config.components["memory"]["params"]["stm_capacity"] = 50
```

**Q: Система падает при больших нагрузках. Что делать?**
A: Проверьте конфигурацию и добавьте мониторинг:

```python
# Включите circuit breaker
config.circuit_breaker_threshold = 5

# Добавьте мониторинг
monitor = ComponentMonitor()
monitor.start_monitoring(engine.provider)

# Проверяйте здоровье системы
health = engine.get_health_status()
if health["overall_status"] != "healthy":
    # Принимайте меры
    pass
```

### Разработка

**Q: Как отладить проблемы в конвейере?**
A: Используйте детальное логирование:

```python
# Включите debug режим
config.log_level = "DEBUG"

# Или добавьте собственное логирование
class DebuggingPipeline(BasePipeline):
    def process(self, request, provider):
        self.logger.debug(f"Processing request: {request.request_id}")
        
        try:
            response = super().process(request, provider)
            self.logger.debug(f"Response: {response.success}")
            return response
        except Exception as e:
            self.logger.error(f"Error in pipeline: {e}", exc_info=True)
            raise
```

**Q: Как создать кастомный компонент?**
A: Наследуйтесь от базового класса:

```python
from neurograph.core import Component

class MyComponent(Component):
    def __init__(self):
        super().__init__("my_component")
    
    def my_method(self, data):
        # Ваша логика
        return processed_data
    
    def get_statistics(self):
        return {"operations": self.operation_count}

# Регистрация
provider.register_component("my_component", MyComponent())
```

**Q: Как протестировать интеграцию?**
A: Используйте моки для изоляции:

```python
class TestMyIntegration(unittest.TestCase):
    def test_component_interaction(self):
        # Создаем мок провайдера
        mock_provider = Mock(spec=IComponentProvider)
        
        # Настраиваем моки компонентов
        mock_nlp = Mock()
        mock_nlp.process_text.return_value = Mock(entities=[])
        mock_provider.get_component.return_value = mock_nlp
        
        # Тестируем
        pipeline = MyPipeline()
        response = pipeline.process(test_request, mock_provider)
        
        self.assertTrue(response.success)
```

### Продакшн

**Q: Как развернуть систему в продакшне?**
A: Используйте контейнеризацию:

```dockerfile
FROM python:3.9

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python", "-m", "neurograph.integration", "--config", "production_config.json"]
```

**Q: Как обеспечить высокую доступность?**
A: Используйте несколько экземпляров:

```python
# Конфигурация для продакшна
production_config = IntegrationConfig(
    max_concurrent_requests=50,
    enable_health_checks=True,
    enable_metrics=True,
    circuit_breaker_threshold=10
)

# Load balancer должен проверять health endpoint
health_status = engine.get_health_status()
if health_status["overall_status"] == "healthy":
    # Экземпляр готов принимать трафик
    pass
```

**Q: Как мониторить систему в продакшне?**
A: Интегрируйтесь с системами мониторинга:

```python
# Prometheus метрики
from prometheus_client import Counter, Histogram, generate_latest

request_counter = Counter('neurograph_requests_total', 'Total requests')
request_duration = Histogram('neurograph_request_duration_seconds', 'Request duration')

class MonitoredPipeline(BasePipeline):
    def process(self, request, provider):
        request_counter.inc()
        
        with request_duration.time():
            return super().process(request, provider)

# Endpoint для метрик
def metrics_endpoint():
    return generate_latest()
```

### Безопасность

**Q: Как защитить систему от атак?**
A: Реализуйте многоуровневую защиту:

```python
# 1. Rate limiting
config.rate_limiting = True

# 2. Валидация входных данных
config.enable_input_validation = True
config.max_input_length = 10000

# 3. Санитизация
def sanitize_input(text):
    # Удаляем потенциально опасный контент
    return clean_text

# 4. Мониторинг подозрительной активности
def detect_anomalies(request):
    # Ваша логика детекции
    return is_suspicious
```

**Q: Как обеспечить конфиденциальность данных?**
A: Используйте шифрование и контроль доступа:

```python
# Шифрование чувствительных данных
import cryptography

class EncryptedMemory:
    def __init__(self, encryption_key):
        self.cipher = Fernet(encryption_key)
    
    def store(self, data):
        encrypted_data = self.cipher.encrypt(data.encode())
        return self.storage.store(encrypted_data)

# Контроль доступа
class AccessControlledProvider(ComponentProvider):
    def get_component(self, component_type, user_permissions=None):
        if not self._check_permissions(component_type, user_permissions):
            raise PermissionError(f"Access denied to {component_type}")
        
        return super().get_component(component_type)
```

---

Эта документация должна предоставить достаточно информации для начала разработки с NeuroGraph Integration Module. Для более глубокого понимания рекомендуется изучить примеры кода и экспериментировать с различными конфигурациями.
    
    def validate_request(self, request: ProcessingRequest) -> tuple[bool, Optional[str]]:
        """Валидация запроса."""
        if not request.content:
            return False, "Пустой запрос"
        
        if len(request.content) < 10:
            return False, "Слишком короткий запрос для анализа"
        
        return True, None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Информация о конвейере."""
        return {
            "name": self.pipeline_name,
            "type": "analytics",
            "processed_count": self.processed_count,
            "capabilities": ["text_analysis", "pattern_detection"]
        }
    
    def _perform_analytics(self, request: ProcessingRequest, 
                          provider: IComponentProvider) -> Dict[str, Any]:
        """Выполнение аналитики."""
        text = request.content
        
        # Простая аналитика
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Если доступен NLP компонент, используем его
        entities = []
        if provider.is_component_available("nlp"):
            nlp = provider.get_component("nlp")
            try:
                nlp_result = nlp.process_text(text)
                entities = [entity.text for entity in nlp_result.entities]
            except Exception:
                pass  # Игнорируем ошибки NLP
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "entities_found": entities,
            "complexity_score": word_count / max(1, sentence_count)
        }
    
    def _create_error_response(self, request: ProcessingRequest, 
                              error_msg: str) -> ProcessingResponse:
        """Создание ответа с ошибкой."""
        return ProcessingResponse(
            request_id=request.request_id,
            success=False,
            error_message=error_msg
        )

# Использование кастомного конвейера
custom_pipeline = CustomAnalyticsPipeline()

# Интеграция в движок
engine = NeuroGraphEngine()
# Здесь нужно будет расширить движок для поддержки кастомных конвейеров
```

---

## Система адаптеров

Адаптеры обеспечивают совместимость между различными форматами данных модулей.

### Встроенные адаптеры

#### GraphMemoryAdapter

Интеграция между графом знаний и системой памяти.

```python
from neurograph.integration.adapters import GraphMemoryAdapter

adapter = GraphMemoryAdapter()

# Преобразование данных графа в элементы памяти
graph_data = {
    "nodes": {
        "Python": {"type": "programming_language", "popularity": 0.9},
        "JavaScript": {"type": "programming_language", "popularity": 0.8}
    },
    "edges": [
        ["Python", "JavaScript", "similar_to"],
        ["Python", "machine_learning", "used_for"]
    ]
}

memory_items = adapter.adapt(graph_data, "memory_items")
print(f"Создано элементов памяти: {len(memory_items)}")

for item in memory_items[:2]:
    print(f"- {item['content']} ({item['content_type']})")
```

#### VectorProcessorAdapter

Связывает векторные представления с логическим процессором.

```python
from neurograph.integration.adapters import VectorProcessorAdapter

adapter = VectorProcessorAdapter()

# Векторные данные
vector_data = {
    "vector_keys": ["AI", "ML", "neural_networks"],
    "similarities": [
        {"concept1": "AI", "concept2": "ML", "score": 0.8},
        {"concept1": "ML", "concept2": "neural_networks", "score": 0.9}
    ]
}

# Преобразование в контекст процессора
processing_context = adapter.adapt(vector_data, "processing_context")
context = processing_context["context"]

print("Факты в контексте:")
for fact_key in list(context.facts.keys())[:3]:
    print(f"- {fact_key}")
```

#### NLPGraphAdapter

Преобразует результаты NLP в обновления графа.

```python
from neurograph.integration.adapters import NLPGraphAdapter

adapter = NLPGraphAdapter()

# Результаты NLP
nlp_data = {
    "entities": [
        {"text": "TensorFlow", "entity_type": "PRODUCT", "confidence": 0.9},
        {"text": "Google", "entity_type": "ORG", "confidence": 0.8}
    ],
    "relations": [
        {
            "subject": {"text": "TensorFlow"},
            "predicate": "developed_by",
            "object": {"text": "Google"},
            "confidence": 0.7
        }
    ],
    "processing_time": 0.5
}

# Преобразование в обновления графа
graph_updates = adapter.adapt(nlp_data, "graph_updates")

print(f"Узлов для добавления: {len(graph_updates['nodes_to_add'])}")
print(f"Ребер для добавления: {len(graph_updates['edges_to_add'])}")

for node in graph_updates["nodes_to_add"]:
    print(f"- Узел: {node['id']} ({node['type']})")
```

### Создание кастомного адаптера

```python
from neurograph.integration.base import IComponentAdapter

class DatabaseGraphAdapter(IComponentAdapter):
    """Адаптер для интеграции с внешней базой данных."""
    
    def __init__(self, db_connection):
        self.adapter_name = "database_graph"
        self.db = db_connection
    
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных между БД и графом."""
        if target_format == "graph_structure":
            return self._db_to_graph(source_data)
        elif target_format == "database_records":
            return self._graph_to_db(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _db_to_graph(self, query_result):
        """Преобразование результата запроса к БД в структуру графа."""
        nodes = {}
        edges = []
        
        for record in query_result:
            # Создание узлов из записей БД
            entity_id = record.get("id")
            entity_type = record.get("type", "unknown")
            
            nodes[entity_id] = {
                "type": entity_type,
                "properties": {k: v for k, v in record.items() 
                             if k not in ["id", "type"]},
                "source": "database"
            }
            
            # Создание связей из внешних ключей
            for field, value in record.items():
                if field.endswith("_id") and value:
                    relation_type = field[:-3]  # убираем "_id"
                    edges.append([entity_id, value, relation_type])
        
        return {"nodes": nodes, "edges": edges}
    
    def _graph_to_db(self, graph_data):
        """Преобразование структуры графа в записи БД."""
        db_records = []
        
        for node_id, node_data in graph_data.get("nodes", {}).items():
            record = {
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                **node_data.get("properties", {})
            }
            db_records.append(record)
        
        return db_records
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы адаптера."""
        return {
            "input": ["database_query_result", "graph_data"],
            "output": ["graph_structure", "database_records"]
        }

# Использование кастомного адаптера
import sqlite3

# Подключение к БД
conn = sqlite3.connect("knowledge.db")
conn.row_factory = sqlite3.Row  # Для доступа по именам колонок

# Создание адаптера
db_adapter = DatabaseGraphAdapter(conn)

# Загрузка данных из БД в граф
cursor = conn.execute("SELECT * FROM entities LIMIT 10")
query_result = cursor.fetchall()

graph_structure = db_adapter.adapt(query_result, "graph_structure")
print(f"Загружено узлов: {len(graph_structure['nodes'])}")
```

---

## Конфигурация системы

### IntegrationConfig

Центральная конфигурация всей системы.

```python
from neurograph.integration.base import IntegrationConfig

# Базовая конфигурация
config = IntegrationConfig(
    engine_name="my_app_engine",
    version="1.0.0",
    max_concurrent_requests=20,
    enable_caching=True,
    enable_metrics=True
)

# Настройка компонентов
config.components = {
    "semgraph": {
        "type": "persistent",
        "params": {
            "file_path": "my_graph.json",
            "auto_save_interval": 300.0
        }
    },
    "memory": {
        "params": {
            "stm_capacity": 200,
            "ltm_capacity": 20000,
            "use_semantic_indexing": True
        }
    },
    "nlp": {
        "params": {
            "language": "ru",
            "use_spacy": True
        }
    }
}
```

### IntegrationConfigManager

Управление конфигурациями.

```python
from neurograph.integration.config import IntegrationConfigManager

manager = IntegrationConfigManager()

# Создание шаблона
config_data = manager.create_template_config("production")

# Модификация шаблона
config_data["engine_name"] = "production_system"
config_data["max_concurrent_requests"] = 100
config_data["components"]["memory"]["params"]["ltm_capacity"] = 100000

# Создание конфигурации
config = IntegrationConfig(**config_data)

# Сохранение в файл
manager.save_config(config, "production_config.json")

# Загрузка из файла
loaded_config = manager.load_config("production_config.json")
```

### Конфигурация по типам

#### Lightweight Configuration

```python
lightweight_config = IntegrationConfig(
    engine_name="lightweight_system",
    components={
        "semgraph": {
            "type": "memory_efficient",
            "params": {}
        },
        "contextvec": {
            "type": "static",
            "params": {"vector_size": 100}
        },
        "memory": {
            "params": {
                "stm_capacity": 25,
                "ltm_capacity": 500,
                "use_semantic_indexing": False
            }
        }
    },
    max_concurrent_requests=2,
    enable_caching=False,
    enable_metrics=False
)
```

#### Research Configuration

```python
research_config = IntegrationConfig(
    engine_name="research_system",
    components={
        "semgraph": {
            "type": "persistent",
            "params": {
                "file_path": "research_graph.json",
                "auto_save_interval": 300.0
            }
        },
        "contextvec": {
            "type": "dynamic",
            "params": {"vector_size": 768, "use_indexing": True}
        },
        "memory": {
            "params": {
                "stm_capacity": 300,
                "ltm_capacity": 50000,
                "use_semantic_indexing": True
            }
        },
        "processor": {
            "type": "graph_based",
            "params": {
                "confidence_threshold": 0.3,
                "max_depth": 10,
                "enable_explanations": True
            }
        }
    },
    max_concurrent_requests=20,
    default_timeout=120.0,
    enable_caching=True,
    cache_ttl=600
)
```

### Динамическая конфигурация

```python
# Получение текущей конфигурации
current_config = engine.config

# Модификация настроек во время работы
if hasattr(engine.provider, 'get_component'):
    memory = engine.provider.get_component("memory")
    # Изменение параметров памяти
    memory.configure(stm_capacity=150)

# Обновление конфигурации кеша
if hasattr(engine, 'cache_manager'):
    engine.cache_manager.update_ttl(900)  # 15 минут
```

---

## Мониторинг и диагностика

### ComponentMonitor

Мониторинг состояния компонентов в реальном времени.

```python
from neurograph.integration.utils import ComponentMonitor

# Создание монитора
monitor = ComponentMonitor(check_interval=60.0)

# Настройка порогов алертов
monitor.alert_thresholds = {
    "response_time": 3.0,      # секунды
    "error_rate": 0.05,        # 5%
    "cpu_usage": 70.0,         # 70%
    "memory_usage": 80.0       # 80%
}

# Запуск мониторинга
monitor.start_monitoring(engine.provider)

# Получение отчета
report = monitor.get_monitoring_report()
print(f"Мониторинг активен: {report['monitoring_active']}")

# Получение данных для дашборда
dashboard_data = monitor.get_dashboard_data()
print(f"Общий статус: {dashboard_data['status']['overall']}")
print(f"Запросов в минуту: {dashboard_data['performance']['requests_per_minute']:.1f}")

# Остановка мониторинга
monitor.stop_monitoring()
```

### HealthChecker

Проверка здоровья компонентов.

```python
from neurograph.integration.utils import HealthChecker

checker = HealthChecker()

# Проверка конкретного компонента
if engine.provider.is_component_available("memory"):
    memory = engine.provider.get_component("memory")
    health = checker.check_component_health("memory", memory)
    
    print(f"Статус памяти: {health.status}")
    print(f"Время отклика: {health.response_time:.3f}с")
    if health.details:
        print(f"Размер: {health.details.get('size', 'N/A')}")

# Проверка всех компонентов
all_health = checker.check_all_components(engine.provider)
for component_name, health in all_health.items():
    status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(health.status, "❓")
    print(f"{status_emoji} {component_name}: {health.status}")

# Общее состояние системы
overall_health = checker.get_overall_health()
print(f"\nОбщий статус: {overall_health['status']}")
print(f"Здоровых компонентов: {overall_health['healthy']}")
print(f"Деградированных: {overall_health['degraded']}")
print(f"Неисправных: {overall_health['unhealthy']}")

# Краткая сводка
summary = checker.get_health_summary()
print(f"\n{summary}")
```

### IntegrationMetrics

Сбор и анализ метрик производительности.

```python
from neurograph.integration.utils import IntegrationMetrics

metrics = IntegrationMetrics()

# Регистрация метрик запроса
metrics.record_request("query", 0.5, True)  # тип, время, успех
metrics.record_request("learning", 1.2, True)
metrics.record_request("query", 0.3, False)

# Регистрация метрик компонента
metrics.record_component_metrics("nlp", {
    "operations": 150,
    "errors": 2,
    "total_time": 45.0
})

# Системные метрики
metrics.record_system_metrics()

# Получение сводки
summary = metrics.get_summary()
print(f"Время работы: {summary['uptime_seconds']:.1f} секунд")
print(f"Всего запросов: {summary['requests']['total']}")
print(f"Успешность: {summary['requests']['success_rate']:.1%}")
print(f"Среднее время ответа: {summary['requests']['average_response_time']:.3f}с")

# Детальный отчет
detailed_report = metrics.get_detailed_report()
print("\n" + detailed_report)
```

### Кастомные метрики

```python
class CustomBusinessMetrics:
    """Кастомные бизнес-метрики."""
    
    def __init__(self):
        self.user_satisfaction = []
        self.query_complexity = []
        self.domain_coverage = {}
    
    def record_user_feedback(self, query_id: str, satisfaction: float):
        """Запись обратной связи пользователя."""
        self.user_satisfaction.append({
            "query_id": query_id,
            "satisfaction": satisfaction,
            "timestamp": time.time()
        })
    
    def record_query_complexity(self, query: str, complexity_score: float):
        """Запись сложности запроса."""
        self.query_complexity.append({
            "query": query[:50],  # Первые 50 символов
            "complexity": complexity_score,
            "timestamp": time.time()
        })
    
    def record_domain_usage(self, domain: str):
        """Запись использования предметной области."""
        self.domain_coverage[domain] = self.domain_coverage.get(domain, 0) + 1
    
    def get_business_summary(self) -> Dict[str, Any]:
        """Получение бизнес-сводки."""
        if not self.user_satisfaction:
            avg_satisfaction = 0.0
        else:
            avg_satisfaction = sum(
                item["satisfaction"] for item in self.user_satisfaction
            ) / len(self.user_satisfaction)
        
        if not self.query_complexity:
            avg_complexity = 0.0
        else:
            avg_complexity = sum(
                item["complexity"] for item in self.query_complexity
            ) / len(self.query_complexity)
        
        top_domains = sorted(
            self.domain_coverage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "average_satisfaction": avg_satisfaction,
            "average_complexity": avg_complexity,
            "total_domains": len(self.domain_coverage),
            "top_domains": top_domains,
            "total_feedback": len(self.user_satisfaction)
        }

# Интеграция с основной системой
business_metrics = CustomBusinessMetrics()

# Обработка запроса с метриками
def process_with_metrics(engine, query: str, user_id: str):
    start_time = time.time()
    
    # Оценка сложности запроса
    complexity = len(query.split()) / 10.0  # Простая метрика
    business_metrics.record_query_complexity(query, complexity)
    
    # Обработка
    response = engine.query(query)
    
    # Определение предметной области (упрощенно)
    domain = "general"
    if "программирование" in query.lower():
        domain = "programming"
    elif "медицина" in query.lower():
        domain = "medicine"
    
    business_metrics.record_domain_usage(domain)
    
    return response

# Использование
response = process_with_metrics(engine, "Что такое машинное обучение?", "user123")

# Симуляция обратной связи
business_metrics.record_user_feedback("query123", 4.5)  # 4.5 из 5

# Получение бизнес-отчета
business_summary = business_metrics.get_business_summary()
print(f"Средняя удовлетворенность: {business_summary['average_satisfaction']:.1f}")
print(f"Средняя сложность запросов: {business_summary['average_complexity']:.1f}")
print(f"Топ предметных областей: {business_summary['top_domains']}")
```

---

## Расширение системы

### Создание кастомного модуля

```python
# my_custom_module.py
from neurograph.core import Component
from neurograph.core.logging import get_logger
from typing import Dict, List, Any

class SentimentAnalyzer(Component):
    """Кастомный компонент для анализа тональности."""
    
    def __init__(self):
        super().__init__("sentiment_analyzer")
        self.logger = get_logger("sentiment_analyzer")
        self.positive_words = ["хорошо", "отлично", "прекрасно", "замечательно"]
        self.negative_words = ["плохо", "ужасно", "отвратительно", "кошмар"]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ тональности текста."""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, positive_count / max(1, total_words) * 10)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, negative_count / max(1, total_words) * 10)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "total_words": total_words
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Статистика компонента."""
        return {
            "positive_words_count": len(self.positive_words),
            "negative_words_count": len(self.negative_words),
            "component_status": "active"
        }

# Регистрация кастомного компонента
def create_sentiment_analyzer():
    return SentimentAnalyzer()

provider = ComponentProvider()
provider.register_lazy_component("sentiment", create_sentiment_analyzer)

# Использование
sentiment_analyzer = provider.get_component("sentiment")
result = sentiment_analyzer.analyze_sentiment("Это отличный продукт!")
print(f"Тональность: {result['sentiment']} (уверенность: {result['confidence']:.2f})")
```

### Создание кастомного конвейера с новым модулем

```python
from neurograph.integration.pipelines import BasePipeline

class SentimentAnalysisPipeline(BasePipeline):
    """Конвейер анализа тональности."""
    
    def __init__(self):
        super().__init__("sentiment_analysis")
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка через конвейер анализа тональности."""
        start_time = time.time()
        
        try:
            # Валидация
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_response(request, False, error_message=error_msg)
            
            components_used = []
            structured_data = {}
            
            # 1. Базовый анализ тональности
            if provider.is_component_available("sentiment"):
                sentiment_result = self._analyze_sentiment(request, provider)
                components_used.append("sentiment")
                structured_data["sentiment"] = sentiment_result
            
            # 2. NLP анализ для контекста
            if provider.is_component_available("nlp"):
                nlp_result = self._analyze_with_nlp(request, provider)
                components_used.append("nlp")
                structured_data["nlp"] = nlp_result
            
            # 3. Поиск в памяти похожих текстов
            if provider.is_component_available("memory"):
                memory_result = self._search_similar_sentiments(request, provider)
                components_used.append("memory")
                structured_data["memory"] = memory_result
            
            # 4. Генерация ответа
            primary_response = self._generate_sentiment_response(structured_data)
            
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            return self._create_response(
                request,
                success=True,
                primary_response=primary_response,
                structured_data=structured_data
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            return self._create_response(request, False, error_message=str(e))
    
    def _analyze_sentiment(self, request: ProcessingRequest, 
                          provider: IComponentProvider) -> Dict[str, Any]:
        """Анализ тональности."""
        sentiment_analyzer = provider.get_component("sentiment")
        return sentiment_analyzer.analyze_sentiment(request.content)
    
    def _analyze_with_nlp(self, request: ProcessingRequest,
                         provider: IComponentProvider) -> Dict[str, Any]:
        """NLP анализ для контекста."""
        nlp = provider.get_component("nlp")
        nlp_result = nlp.process_text(request.content)
        
        return {
            "entities": [entity.text for entity in nlp_result.entities],
            "language": nlp_result.language,
            "sentences_count": len(nlp_result.sentences)
        }
    
    def _search_similar_sentiments(self, request: ProcessingRequest,
                                  provider: IComponentProvider) -> Dict[str, Any]:
        """Поиск похожих по тональности текстов в памяти."""
        memory = provider.get_component("memory")
        
        # Получаем недавние элементы памяти
        recent_items = memory.get_recent_items(hours=24.0)
        
        # Фильтруем элементы с анализом тональности
        sentiment_items = [
            item for item in recent_items
            if item.metadata.get("content_type") == "sentiment_analysis"
        ]
        
        return {
            "similar_items_found": len(sentiment_items),
            "total_recent_items": len(recent_items)
        }
    
    def _generate_sentiment_response(self, structured_data: Dict[str, Any]) -> str:
        """Генерация ответа по анализу тональности."""
        sentiment_data = structured_data.get("sentiment", {})
        sentiment = sentiment_data.get("sentiment", "unknown")
        confidence = sentiment_data.get("confidence", 0.0)
        
        nlp_data = structured_data.get("nlp", {})
        entities = nlp_data.get("entities", [])
        
        response_parts = []
        
        # Основной результат
        if sentiment == "positive":
            response_parts.append(f"Обнаружена положительная тональность (уверенность: {confidence:.1%})")
        elif sentiment == "negative":
            response_parts.append(f"Обнаружена отрицательная тональность (уверенность: {confidence:.1%})")
        else:
            response_parts.append(f"Нейтральная тональность (уверенность: {confidence:.1%})")
        
        # Дополнительный контекст
        if entities:
            response_parts.append(f"Обнаружены сущности: {', '.join(entities[:3])}")
        
        memory_data = structured_data.get("memory", {})
        similar_count = memory_data.get("similar_items_found", 0)
        if similar_count > 0:
            response_parts.append(f"Найдено {similar_count} похожих анализов в памяти")
        
        return ". ".join(response_parts) + "."

# Создание и использование кастомного конвейера
sentiment_pipeline = SentimentAnalysisPipeline()

request = ProcessingRequest(
    content="Я очень доволен результатами работы этой системы!",
    request_type="sentiment_analysis"
)

response = sentiment_pipeline.process(request, provider)
print(f"Анализ тональности: {response.primary_response}")

# Детали анализа
sentiment_details = response.structured_data.get("sentiment", {})
print(f"Тональность: {sentiment_details.get('sentiment')}")
print(f"Уверенность: {sentiment_details.get('confidence', 0):.2f}")
```

### Интеграция с внешними API

```python
import requests
import json
from typing import Optional

class ExternalAPIComponent(Component):
    """Компонент для интеграции с внешними API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        super().__init__("external_api")
        self.api_url = api_url
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def call_api(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Вызов внешнего API."""
        url = f"{self.api_url}/{endpoint}"
        
        try:
            response = self.session.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Ошибка API запроса: {e}")
            return {"error": str(e)}
    
    def translate_text(self, text: str, target_language: str = "en") -> Dict[str, Any]:
        """Перевод текста через внешний API."""
        data = {
            "text": text,
            "target_language": target_language
        }
        return self.call_api("translate", data)
    
    def get_embeddings(self, text: str) -> Dict[str, Any]:
        """Получение эмбеддингов через внешний API."""
        data = {"text": text}
        return self.call_api("embeddings", data)

# Адаптер для интеграции внешнего API с системой
class ExternalAPIAdapter(IComponentAdapter):
    """Адаптер для внешнего API."""
    
    def __init__(self):
        self.adapter_name = "external_api"
    
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных внешнего API."""
        if target_format == "vector_representation":
            return self._api_to_vectors(source_data)
        elif target_format == "translation_result":
            return self._api_to_translation(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _api_to_vectors(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразование ответа API в векторный формат."""
        if "error" in api_response:
            return {"vectors": [], "error": api_response["error"]}
        
        embeddings = api_response.get("embeddings", [])
        return {
            "vectors": embeddings,
            "dimension": len(embeddings) if embeddings else 0,
            "source": "external_api"
        }
    
    def _api_to_translation(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразование ответа API перевода."""
        if "error" in api_response:
            return {"translated_text": "", "error": api_response["error"]}
        
        return {
            "translated_text": api_response.get("translated_text", ""),
            "source_language": api_response.get("source_language", "unknown"),
            "target_language": api_response.get("target_language", "unknown"),
            "confidence": api_response.get("confidence", 0.0)
        }
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы."""
        return {
            "input": ["api_response"],
            "output": ["vector_representation", "translation_result"]
        }

# Регистрация и использование
def create_external_api():
    return ExternalAPIComponent(
        api_url="https://api.example.com/v1",
        api_key="your_api_key_here"
    )

provider.register_lazy_component("external_api", create_external_api)

# Создание адаптера
api_adapter = ExternalAPIAdapter()

# Использование в конвейере
class TranslationPipeline(BasePipeline):
    """Конвейер перевода с внешним API."""
    
    def __init__(self):
        super().__init__("translation")
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка перевода."""
        try:
            # Получение компонента внешнего API
            api_component = provider.get_component("external_api")
            
            # Перевод текста
            target_lang = request.context.get("target_language", "en")
            api_response = api_component.translate_text(request.content, target_lang)
            
            # Адаптация результата
            translation_result = api_adapter.adapt(api_response, "translation_result")
            
            if "error" in translation_result:
                return self._create_response(
                    request, False, 
                    error_message=translation_result["error"]
                )
            
            return self._create_response(
                request, True,
                primary_response=translation_result["translated_text"],
                structured_data={"translation": translation_result}
            )
            
        except Exception as e:
            return self._create_response(request, False, error_message=str(e))

# Использование конвейера перевода
translation_pipeline = TranslationPipeline()

request = ProcessingRequest(
    content="Привет, как дела?",
    request_type="translation",
    context={"target_language": "en"}
)

response = translation_pipeline.process(request, provider)
print(f"Перевод: {response.primary_response}")
```

---

## Примеры использования

### Пример 1: Персональный ассистент

```python
class PersonalAssistant:
    """Персональный ИИ-ассистент на базе NeuroGraph."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.engine = create_default_engine()
        self.conversation_history = []
        
        # Настройка персонализации
        self.user_context = {
            "user_id": user_id,
            "preferences": {},
            "interests": [],
            "expertise_areas": []
        }
    
    def chat(self, message: str) -> str:
        """Диалог с пользователем."""
        # Добавляем контекст предыдущих сообщений
        recent_context = self._build_conversation_context()
        
        response = self.engine.query(
            message,
            context={
                **self.user_context,
                "conversation_history": recent_context
            },
            response_format=ResponseFormat.CONVERSATIONAL
        )
        
        # Сохраняем в историю
        self.conversation_history.append({
            "user_message": message,
            "assistant_response": response.primary_response,
            "timestamp": time.time(),
            "components_used": response.components_used
        })
        
        # Обучаем систему на диалоге
        self._learn_from_conversation(message, response)
        
        return response.primary_response
    
    def learn_about_user(self, information: str):
        """Изучение информации о пользователе."""
        learning_response = self.engine.learn(
            f"Пользователь {self.user_id}: {information}",
            context=self.user_context
        )
        
        # Обновляем контекст пользователя
        self._update_user_context(information, learning_response)
    
    def _build_conversation_context(self) -> List[Dict[str, str]]:
        """Построение контекста разговора."""
        # Берем последние 5 сообщений
        recent = self.conversation_history[-5:] if self.conversation_history else []
        return [
            {
                "user": item["user_message"],
                "assistant": item["assistant_response"]
            }
            for item in recent
        ]
    
    def _learn_from_conversation(self, user_message: str, response: ProcessingResponse):
        """Обучение на основе диалога."""
        # Извлекаем темы и интересы из сообщения пользователя
        if response.structured_data.get("nlp"):
            entities = response.structured_data["nlp"].get("entities", [])
            for entity in entities:
                if entity not in self.user_context["interests"]:
                    self.user_context["interests"].append(entity)
    
    def _update_user_context(self, information: str, learning_response: ProcessingResponse):
        """Обновление контекста пользователя."""
        # Анализируем, что узнали о пользователе
        if learning_response.structured_data.get("nlp"):
            nlp_data = learning_response.structured_data["nlp"]
            entities = nlp_data.get("entities", [])
            
            # Обновляем области экспертизы
            for entity in entities:
                if any(keyword in information.lower() for keyword in ["работаю", "изучаю", "специализируюсь"]):
                    if entity not in self.user_context["expertise_areas"]:
                        self.user_context["expertise_areas"].append(entity)
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Получение профиля пользователя."""
        return {
            "user_id": self.user_id,
            "conversation_count": len(self.conversation_history),
            "interests": self.user_context["interests"][:10],  # Топ-10
            "expertise_areas": self.user_context["expertise_areas"],
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def shutdown(self):
        """Завершение работы ассистента."""
        self.engine.shutdown()

# Использование персонального ассистента
assistant = PersonalAssistant("user_123")

# Обучение ассистента информации о пользователе
assistant.learn_about_user("Я работаю Python разработчиком и интересуюсь машинным обучением")
assistant.learn_about_user("Мне нравится читать научную фантастику")

# Диалог
response1 = assistant.chat("Привет! Как дела?")
print(f"Ассистент: {response1}")

response2 = assistant.chat("Посоветуй книгу по машинному обучению")
print(f"Ассистент: {response2}")

response3 = assistant.chat("А что нового в области ИИ?")
print(f"Ассистент: {response3}")

# Профиль пользователя
profile = assistant.get_user_profile()
print(f"\nПрофиль пользователя:")
print(f"Интересы: {profile['interests']}")
print(f"Области экспертизы: {profile['expertise_areas']}")
print(f"Сообщений: {profile['conversation_count']}")

assistant.shutdown()
```

### Пример 2: Система анализа документов

```python
import os
from pathlib import Path
from typing import List

class DocumentAnalysisSystem:
    """Система анализа больших объемов документов."""
    
    def __init__(self):
        self.engine = create_research_engine()  # Используем исследовательскую конфигурацию
        self.processed_documents = []
        self.document_index = {}
    
    def process_document(self, file_path: str, document_type: str = "general") -> Dict[str, Any]:
        """Обработка одного документа."""
        try:
            # Чтение документа
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Обработка через систему
            response = self.engine.learn(
                content,
                context={
                    "document_type": document_type,
                    "source_file": file_path,
                    "processing_mode": "batch"
                }
            )
            
            # Создание записи о документе
            doc_record = {
                "file_path": file_path,
                "document_type": document_type,
                "content_length": len(content),
                "processing_time": response.processing_time,
                "success": response.success,
                "entities_found": len(response.structured_data.get("nlp", {}).get("entities", [])),
                "concepts_added": response.structured_data.get("graph", {}).get("nodes_added", 0),
                "timestamp": time.time()
            }
            
            self.processed_documents.append(doc_record)
            self.document_index[file_path] = doc_record
            
            return doc_record
            
        except Exception as e:
            error_record = {
                "file_path": file_path,
                "error": str(e),
                "success": False,
                "timestamp": time.time()
            }
            self.processed_documents.append(error_record)
            return error_record
    
    def process_directory(self, directory_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        """Обработка всех документов в директории."""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.doc', '.docx']
        
        directory = Path(directory_path)
        files_to_process = []
        
        # Поиск файлов для обработки
        for ext in file_extensions:
            files_to_process.extend(directory.glob(f"*{ext}"))
            files_to_process.extend(directory.glob(f"**/*{ext}"))  # Рекурсивный поиск
        
        results = {
            "total_files": len(files_to_process),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "processing_times": [],
            "errors": []
        }
        
        # Обработка каждого файла
        for file_path in files_to_process:
            print(f"Обработка: {file_path}")
            
            doc_result = self.process_document(str(file_path))
            results["processed"] += 1
            
            if doc_result["success"]:
                results["successful"] += 1
                results["processing_times"].append(doc_result["processing_time"])
            else:
                results["failed"] += 1
                results["errors"].append({
                    "file": str(file_path),
                    "error": doc_result.get("error", "Unknown error")
                })
        
        # Вычисление статистики
        if results["processing_times"]:
            results["avg_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"])
            results["total_processing_time"] = sum(results["processing_times"])
        
        return results
    
    def search_documents(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Поиск информации в обработанных документах."""
        search_response = self.engine.query(
            query,
            context={"search_scope": "processed_documents"},
            max_results=max_results
        )
        
        # Обогащение результатов информацией о документах
        enhanced_results = {
            "query": query,
            "primary_answer": search_response.primary_response,
            "confidence": search_response.confidence,
            "source_documents": [],
            "related_concepts": search_response.related_concepts
        }
        
        # Найдем документы, которые могли повлиять на ответ
        if search_response.structured_data.get("graph_search"):
            found_nodes = search_response.structured_data["graph_search"].get("found_nodes", [])
            
            # Поиск документов, содержащих найденные концепты
            relevant_docs = []
            for doc_record in self.processed_documents:
                if doc_record.get("success"):
                    # Простая эвристика - документ релевантен, если добавил концепты
                    if doc_record.get("concepts_added", 0) > 0:
                        relevant_docs.append({
                            "file_path": doc_record["file_path"],
                            "document_type": doc_record.get("document_type", "unknown"),
                            "concepts_added": doc_record["concepts_added"],
                            "relevance_score": min(1.0, doc_record["concepts_added"] / 10.0)
                        })
            
            # Сортировка по релевантности
            relevant_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
            enhanced_results["source_documents"] = relevant_docs[:5]  # Топ-5
        
        return enhanced_results
    
    def get_analysis_report(self) -> str:
        """Получение отчета об анализе документов."""
        if not self.processed_documents:
            return "Документы не обработаны."
        
        successful_docs = [doc for doc in self.processed_documents if doc.get("success", False)]
        failed_docs = [doc for doc in self.processed_documents if not doc.get("success", False)]
        
        total_entities = sum(doc.get("entities_found", 0) for doc in successful_docs)
        total_concepts = sum(doc.get("concepts_added", 0) for doc in successful_docs)
        total_content_length = sum(doc.get("content_length", 0) for doc in successful_docs)
        
        processing_times = [doc["processing_time"] for doc in successful_docs if "processing_time" in doc]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        report_lines = [
            "=== Отчет об анализе документов ===",
            f"Всего документов: {len(self.processed_documents)}",
            f"Успешно обработано: {len(successful_docs)}",
            f"Ошибки: {len(failed_docs)}",
            "",
            "Статистика обработки:",
            f"  Общий объем текста: {total_content_length:,} символов",
            f"  Найдено сущностей: {total_entities:,}",
            f"  Добавлено концептов: {total_concepts:,}",
            f"  Среднее время обработки: {avg_processing_time:.2f}с",
            "",
            "Типы документов:"
        ]
        
        # Группировка по типам документов
        doc_types = {}
        for doc in successful_docs:
            doc_type = doc.get("document_type", "unknown")
            if doc_type not in doc_types:
                doc_types[doc_type] = {"count": 0, "entities": 0, "concepts": 0}
            
            doc_types[doc_type]["count"] += 1
            doc_types[doc_type]["entities"] += doc.get("entities_found", 0)
            doc_types[doc_type]["concepts"] += doc.get("concepts_added", 0)
        
        for doc_type, stats in doc_types.items():
            report_lines.extend([
                f"  {doc_type}:",
                f"    Документов: {stats['count']}",
                f"    Сущностей: {stats['entities']}",
                f"    Концептов: {stats['concepts']}"
            ])
        
        if failed_docs:
            report_lines.extend([
                "",
                "Ошибки:",
                *[f"  {doc['file_path']}: {doc.get('error', 'Unknown error')}" for doc in failed_docs[:5]]
            ])
            
            if len(failed_docs) > 5:
                report_lines.append(f"  ... и еще {len(failed_docs) - 5} ошибок")
        
        return "\n".join(report_lines)
    
    def shutdown(self):
        """Завершение работы системы."""
        self.engine.shutdown()

# Использование системы анализа документов
analyzer = DocumentAnalysisSystem()

# Обработка одного документа
doc_result = analyzer.process_document("sample_document.txt", "technical")
print(f"Документ обработан: {doc_result['success']}")
print(f"Найдено сущностей: {doc_result['entities_found']}")

# Обработка директории с документами
# directory_results = analyzer.process_directory("./documents", ['.txt', '.md'])
# print(f"Обработано файлов: {directory_results['processed']}")
# print(f"Успешных: {directory_results['successful']}")

# Поиск в обработанных документах
search_results = analyzer.search_documents("машинное обучение")
print(f"Ответ на запрос: {search_results['primary_answer']}")
print(f"Релевантные документы: {len(search_results['source_documents'])}")

# Отчет об анализе
analysis_report = analyzer.get_analysis_report()
print("\n" + analysis_report)

analyzer.shutdown()
```

### Пример 3: Корпоративная база знаний

```python
class CorporateKnowledgeBase:
    """Корпоративная база знаний с ролевым доступом."""
    
    def __init__(self):
        self.engine = create_default_engine()
        self.users = {}
        self.departments = {}
        self.access_log = []
    
    def register_user(self, user_id: str, name: str, department: str, role: str):
        """Регистрация пользователя."""
        self.users[user_id] = {
            "name": name,
            "department": department,
            "role": role,
            "queries_count": 0,
            "last_activity": None,
            "expertise_areas": []
        }
        
        # Добавляем в департамент
        if department not in self.departments:
            self.departments[department] = {"users": [], "knowledge_areas": []}
        
        self.departments[department]["users"].append(user_id)
    
    def add_knowledge(self, content: str, author_id: str, category: str = "general", 
                     access_level: str = "public") -> Dict[str, Any]:
        """Добавление знаний в базу."""
        if author_id not in self.users:
            return {"success": False, "error": "Unknown user"}
        
        user = self.users[author_id]
        
        # Контекст для обучения
        context = {
            "author": user["name"],
            "department": user["department"],
            "category": category,
            "access_level": access_level,
            "contribution_type": "knowledge_addition"
        }
        
        response = self.engine.learn(content, context=context)
        
        # Логирование
        self._log_activity(author_id, "knowledge_addition", {
            "category": category,
            "content_length": len(content),
            "success": response.success
        })
        
        # Обновление экспертизы пользователя
        if response.success and response.structured_data.get("nlp"):
            entities = response.structured_data["nlp"].get("entities", [])
            for entity in entities[:3]:  # Топ-3 сущности
                if entity not in user["expertise_areas"]:
                    user["expertise_areas"].append(entity)
        
        return {
            "success": response.success,
            "knowledge_id": response.response_id,
            "entities_extracted": len(response.structured_data.get("nlp", {}).get("entities", [])),
            "concepts_added": response.structured_data.get("graph", {}).get("nodes_added", 0)
        }
    
    def search_knowledge(self, query: str, user_id: str, department_filter: str = None) -> Dict[str, Any]:
        """Поиск знаний с учетом прав доступа."""
        if user_id not in self.users:
            return {"success": False, "error": "Unknown user"}
        
        user = self.users[user_id]
        user["queries_count"] += 1
        user["last_activity"] = time.time()
        
        # Формирование контекста поиска
        search_context = {
            "user_department": user["department"],
            "user_role": user["role"],
            "user_expertise": user["expertise_areas"][:5]  # Топ-5 областей экспертизы
        }
        
        if department_filter:
            search_context["department_filter"] = department_filter
        
        response = self.engine.query(query, context=search_context)
        
        # Логирование поиска
        self._log_activity(user_id, "knowledge_search", {
            "query": query[:50],  # Первые 50 символов
            "success": response.success,
            "processing_time": response.processing_time
        })
        
        # Обогащение результатов
        enhanced_response = {
            "success": response.success,
            "answer": response.primary_response,
            "confidence": response.confidence,
            "sources": self._identify_knowledge_sources(response),
            "related_experts": self._find_related_experts(response),
            "suggested_contacts": self._suggest_contacts(query, user_id),
            "processing_time": response.processing_time
        }
        
        return enhanced_response
    
    def find_expert(self, topic: str, exclude_user: str = None) -> List[Dict[str, Any]]:
        """Поиск эксперта по теме."""
        experts = []
        
        for user_id, user in self.users.items():
            if exclude_user and user_id == exclude_user:
                continue
            
            # Поиск экспертов по областям знаний
            expertise_match = sum(
                1 for area in user["expertise_areas"]
                if topic.lower() in area.lower() or area.lower() in topic.lower()
            )
            
            if expertise_match > 0:
                experts.append({
                    "user_id": user_id,
                    "name": user["name"],
                    "department": user["department"],
                    "role": user["role"],
                    "expertise_score": expertise_match,
                    "activity_level": user["queries_count"],
                    "expertise_areas": user["expertise_areas"][:3]
                })
        
        # Сортировка по релевантности
        experts.sort(key=lambda x: (x["expertise_score"], x["activity_level"]), reverse=True)
        return experts[:5]  # Топ-5 экспертов
    
    def get_department_analytics(self, department: str) -> Dict[str, Any]:
        """Аналитика по департаменту."""
        if department not in self.departments:
            return {"error": "Department not found"}
        
        dept_users = self.departments[department]["users"]
        dept_user_data = [self.users[uid] for uid in dept_users if uid in self.users]
        
        # Активность департамента
        total_queries = sum(user["queries_count"] for user in dept_user_data)
        active_users = len([user for user in dept_user_data if user["queries_count"] > 0])
        
        # Области экспертизы департамента
        all_expertise = []
        for user in dept_user_data:
            all_expertise.extend(user["expertise_areas"])
        
        expertise_counts = {}
        for area in all_expertise:
            expertise_counts[area] = expertise_counts.get(area, 0) + 1
        
        top_expertise = sorted(expertise_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Последняя активность
        last_activities = [user["last_activity"] for user in dept_user_data if user["last_activity"]]
        last_activity = max(last_activities) if last_activities else None
        
        return {
            "department": department,
            "total_users": len(dept_users),
            "active_users": active_users,
            "total_queries": total_queries,
            "avg_queries_per_user": total_queries / len(dept_users) if dept_users else 0,
            "top_expertise_areas": top_expertise,
            "last_activity": last_activity
        }
    
    def _log_activity(self, user_id: str, activity_type: str, details: Dict[str, Any]):
        """Логирование активности пользователей."""
        log_entry = {
            "user_id": user_id,
            "activity_type": activity_type,
            "details": details,
            "timestamp": time.time()
        }
        
        self.access_log.append(log_entry)
        
        # Ограничиваем размер лога
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]
    
    def _identify_knowledge_sources(self, response: ProcessingResponse) -> List[Dict[str, str]]:
        """Идентификация источников знаний."""
        # Здесь можно анализировать structured_data для определения источников
        sources = []
        
        if response.structured_data.get("memory_search"):
            memory_results = response.structured_data["memory_search"]
            if memory_results.get("relevant_memories"):
                sources.append({
                    "type": "corporate_memory",
                    "description": f"Найдено {len(memory_results['relevant_memories'])} релевантных записей"
                })
        
        if response.structured_data.get("graph_search"):
            graph_results = response.structured_data["graph_search"]
            if graph_results.get("found_nodes"):
                sources.append({
                    "type": "knowledge_graph",
                    "description": f"Найдено {len(graph_results['found_nodes'])} концептов в графе знаний"
                })
        
        return sources
    
    def _find_related_experts(self, response: ProcessingResponse) -> List[Dict[str, str]]:
        """Поиск связанных экспертов."""
        if not response.structured_data.get("nlp"):
            return []
        
        entities = response.structured_data["nlp"].get("entities", [])
        if not entities:
            return []
        
        # Ищем экспертов по первой найденной сущности
        main_topic = entities[0] if entities else ""
        experts = self.find_expert(main_topic)
        
        return [
            {
                "name": expert["name"],
                "department": expert["department"],
                "contact_suggestion": f"Обратитесь к {expert['name']} из {expert['department']}"
            }
            for expert in experts[:3]
        ]
    
    def _suggest_contacts(self, query: str, user_id: str) -> List[str]:
        """Предложение контактов."""
        suggestions = []
        
        # Поиск коллег из того же департамента
        user = self.users[user_id]
        dept_colleagues = [
            uid for uid in self.departments[user["department"]]["users"]
            if uid != user_id and self.users[uid]["queries_count"] > 10
        ]
        
        if dept_colleagues:
            colleague = self.users[dept_colleagues[0]]
            suggestions.append(f"Попробуйте обратиться к {colleague['name']} из вашего департамента")
        
        return suggestions
    
    def get_system_report(self) -> str:
        """Системный отчет."""
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u["queries_count"] > 0])
        total_queries = sum(u["queries_count"] for u in self.users.values())
        
        # Статистика по департаментам
        dept_stats = []
        for dept_name in self.departments:
            dept_analytics = self.get_department_analytics(dept_name)
            dept_stats.append(f"  {dept_name}: {dept_analytics['active_users']}/{dept_analytics['total_users']} активных")
        
        # Топ пользователей
        top_users = sorted(
            [(uid, u) for uid, u in self.users.items()],
            key=lambda x: x[1]["queries_count"],
            reverse=True
        )[:5]
        
        report_lines = [
            "=== Отчет корпоративной базы знаний ===",
            f"Всего пользователей: {total_users}",
            f"Активных пользователей: {active_users}",
            f"Всего запросов: {total_queries}",
            f"Департаментов: {len(self.departments)}",
            "",
            "Статистика по департаментам:",
            *dept_stats,
            "",
            "Топ пользователей по активности:",
            *[f"  {u[1]['name']} ({u[1]['department']}): {u[1]['queries_count']} запросов" 
              for u in top_users]
        ]
        
        return "\n".join(report_lines)
    
    def shutdown(self):
        """Завершение работы системы."""
        self.engine.shutdown()

# Использование корпоративной базы знаний
kb = CorporateKnowledgeBase()

# Регистрация пользователей
kb.register_user("dev001", "Алексей Иванов", "Разработка", "Senior Developer")
kb.register_user("hr001", "Мария Петрова", "HR", "HR Manager")
kb.register_user("qa001", "Дмитрий Сидоров", "QA", "QA Engineer")

# Добавление знаний
kb.add_knowledge(
    "Наша команда использует микросервисную архитектуру с Docker и Kubernetes. "
    "Для CI/CD применяем GitLab CI с автоматическим развертыванием.",
    "dev001",
    "architecture",
    "public"
)

kb.add_knowledge(
    "Процедура онбординга новых сотрудников включает знакомство с командой, "
    "настройку рабочего места и обучение корпоративным процессам.",
    "hr001",
    "hr_processes",
    "internal"
)

# Поиск знаний
search_result = kb.search_knowledge(
    "Как настроить CI/CD для нового проекта?", 
    "qa001"
)

print(f"Ответ: {search_result['answer']}")
print(f"Эксперты: {[e['name'] for e in search_result['related_experts']]}")

# Поиск эксперта
experts = kb.find_expert("архитектура")
print(f"Эксперты по архитектуре: {[e['name'] for e in experts]}")

# Аналитика департамента
dev_analytics = kb.get_department_analytics("Разработка")
print(f"Разработка - активных пользователей: {dev_analytics['active_users']}")

# Системный отчет
system_report = kb.get_system_report()
print("\n" + system_report)

kb.shutdown()
```

---

## Лучшие практики

### Проектирование системы

#### 1. Разделение ответственности
```python
# ✅ Хорошо - четкое разделение
class SearchPipeline(BasePipeline):
    def process(self, request, provider):
        # Только логика поиска
        pass

class LearningPipeline(BasePipeline):
    def process(self, request, provider):
        # Только логика обучения
        pass

# ❌ Плохо - смешанная ответственность
class UniversalPipeline(BasePipeline):
    def process(self, request, provider):
        if request.request_type == "search":
            # логика поиска
        elif request.request_type == "learning":
            # логика обучения
        # ...
```

#### 2. Обработка ошибок
```python
# ✅ Хорошо - graceful degradation
def process_with_fallback(self, request, provider):
    try:
        # Основная логика
        result = main_processing(request, provider)
        return result
    except SpecificError as e:
        self.logger.warning(f"Основная обработка не удалась: {e}")
        # Fallback к простой обработке
        return fallback_processing(request, provider)
    except Exception as e:
        self.logger.error(f"Критическая ошибка: {e}")
        return self._create_error_response(request, str(e))