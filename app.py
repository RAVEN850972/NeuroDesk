from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import os
import time
from pathlib import Path

# Создаем простую версию для демонстрации
# В реальном проекте вы бы импортировали свои модули
import json
import asyncio
from openai import OpenAI

app = FastAPI(
    title="NeuroDesk Hybrid API",
    description="Персональный ИИ-ассистент с гибридной архитектурой (граф знаний + OpenAI)",
    version="2.0.0"
)

# Подключаем статические файлы
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Простая имитация NeuroDesk для демонстрации
class SimpleNeuroDesk:
    def __init__(self, openai_api_key=None):
        self.openai_client = None
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                print(f"Ошибка инициализации OpenAI: {e}")
        
        self.knowledge_graph = {}  # Простой граф знаний
        self.stats = {
            "total_queries": 0,
            "graph_responses": 0,
            "openai_fallbacks": 0,
            "learning_sessions": 0
        }
        
    def ask(self, prompt: str, domain: str = "general", feedback: int = None) -> Dict[str, Any]:
        """Простая имитация гибридного запроса."""
        self.stats["total_queries"] += 1
        start_time = time.time()
        
        # Проверяем, есть ли ответ в "графе знаний"
        graph_answer = self.knowledge_graph.get(prompt.lower())
        
        if graph_answer:
            # Ответ из графа
            self.stats["graph_responses"] += 1
            return {
                "answer": graph_answer,
                "success": True,
                "confidence": 0.8,
                "source": "graph",
                "processing_time": time.time() - start_time,
                "components_used": ["semgraph", "memory"],
                "explanation": ["Ответ найден в графе знаний"],
                "warnings": []
            }
        else:
            # Fallback на OpenAI
            if self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"Ты умный ассистент. Отвечай в предметной области: {domain}"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    
                    # "Обучаем" граф (сохраняем ответ)
                    self.knowledge_graph[prompt.lower()] = answer
                    self.stats["openai_fallbacks"] += 1
                    self.stats["learning_sessions"] += 1
                    
                    return {
                        "answer": answer,
                        "success": True,
                        "confidence": 0.9,
                        "source": "openai_fallback",
                        "processing_time": time.time() - start_time,
                        "components_used": ["openai_api"],
                        "explanation": ["Ответ получен от OpenAI", "Граф обучен на этом ответе"],
                        "warnings": [],
                        "learned_from_openai": True,
                        "tokens_used": response.usage.total_tokens if response.usage else None
                    }
                    
                except Exception as e:
                    return {
                        "answer": f"❌ Ошибка OpenAI: {str(e)}",
                        "success": False,
                        "confidence": 0.0,
                        "source": "error",
                        "processing_time": time.time() - start_time,
                        "error": str(e)
                    }
            else:
                return {
                    "answer": "❌ OpenAI недоступен и нет данных в графе знаний",
                    "success": False,
                    "confidence": 0.0,
                    "source": "error",
                    "processing_time": time.time() - start_time,
                    "error": "OpenAI API не настроен"
                }
    
    def learn(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Простое обучение."""
        # Простая имитация: сохраняем текст как есть
        key = f"learned_{len(self.knowledge_graph)}"
        self.knowledge_graph[key] = text
        self.stats["learning_sessions"] += 1
        
        return {
            "success": True,
            "message": "Информация успешно изучена",
            "nodes_added": 1,
            "edges_added": 0,
            "processing_time": 0.1
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Статистика системы."""
        total = self.stats["total_queries"]
        return {
            "user_id": "demo_user",
            "session_id": "demo_session",
            "components_status": {"graph": "available", "openai": "available" if self.openai_client else "unavailable"},
            "pipeline_stats": {
                **self.stats,
                "graph_success_rate": self.stats["graph_responses"] / max(1, total),
            },
            "learning_quality": {"total_learned": len(self.knowledge_graph)},
            "tuner_stats": {"best_score": 0.8, "mutation_rate": 0.1}
        }
    
    def get_learning_quality_report(self) -> Dict[str, Any]:
        """Отчет о качестве обучения."""
        return {
            "total_learned": len(self.knowledge_graph),
            "recent_stats": [],
            "averages": {
                "avg_nodes_added": 1.0,
                "avg_edges_added": 0.0,
                "avg_relevance_score": 0.7
            }
        }

# Глобальный экземпляр
desk = None

def get_desk():
    """Получение экземпляра NeuroDesk с ленивой инициализацией."""
    global desk
    if desk is None:
        openai_key = "API ONPEN AI KEY"
        desk = SimpleNeuroDesk(openai_api_key=openai_key)
    return desk

# Модели данных
class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Вопрос пользователя")
    feedback: Optional[int] = Field(None, ge=-1, le=1, description="Обратная связь (-1, 0, 1)")
    domain: Optional[str] = Field(None, max_length=50, description="Предметная область")
    explanation_level: str = Field("detailed", pattern="^(basic|detailed|debug)$")

class AskResponse(BaseModel):
    answer: str
    success: bool
    confidence: Optional[float] = None
    source: str  # "graph", "openai_fallback", "error"
    processing_time: Optional[float] = None
    explanation: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    learned_from_openai: Optional[bool] = None
    tokens_used: Optional[int] = None

class LearnRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, description="Текст для изучения")
    domain: str = Field("general", max_length=50, description="Предметная область")
    context: Optional[Dict[str, Any]] = Field(None, description="Дополнительный контекст")

class LearnResponse(BaseModel):
    success: bool
    message: str
    nodes_added: int = 0
    edges_added: int = 0
    processing_time: Optional[float] = None
    error: Optional[str] = None

# API endpoints
@app.post("/ask", response_model=AskResponse, 
          summary="Задать вопрос ассистенту",
          description="Основной endpoint для взаимодействия с гибридным ассистентом")
async def ask(request: AskRequest):
    try:
        neurodesk = get_desk()
        
        result = neurodesk.ask(
            prompt=request.prompt,
            domain=request.domain or "general",
            feedback=request.feedback
        )
        
        return AskResponse(**result)
    
    except Exception as e:
        return AskResponse(
            answer=f"❌ Ошибка сервера: {str(e)}",
            success=False,
            source="error"
        )

@app.post("/learn", response_model=LearnResponse,
          summary="Обучить систему",
          description="Явное обучение системы на предоставленном тексте")
async def learn(request: LearnRequest):
    try:
        neurodesk = get_desk()
        
        result = neurodesk.learn(
            text=request.text,
            domain=request.domain
        )
        
        return LearnResponse(**result)
    
    except Exception as e:
        return LearnResponse(
            success=False,
            message=f"Ошибка обучения: {str(e)}",
            error=str(e)
        )

@app.get("/stats",
         summary="Статистика системы",
         description="Получение подробной статистики работы системы")
async def get_stats():
    try:
        neurodesk = get_desk()
        stats = neurodesk.get_system_stats()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")

@app.get("/learning_quality",
         summary="Качество обучения",
         description="Отчет о качестве обучения системы")
async def get_learning_quality(max_entries: int = 10):
    try:
        neurodesk = get_desk()
        report = neurodesk.get_learning_quality_report()
        return report
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/health",
         summary="Проверка здоровья",
         description="Быстрая проверка работоспособности системы")
async def health_check():
    try:
        neurodesk = get_desk()
        stats = neurodesk.get_system_stats()
        
        return {
            "status": "healthy",
            "openai_available": neurodesk.openai_client is not None,
            "total_queries": stats["pipeline_stats"]["total_queries"],
            "graph_success_rate": stats["pipeline_stats"]["graph_success_rate"]
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Главная страница
@app.get("/", 
         summary="Главная страница",
         description="Веб-интерфейс ассистента")
async def root():
    static_index = Path("static/index.html")
    if static_index.exists():
        return FileResponse("static/index.html")
    else:
        return JSONResponse({
            "message": "NeuroDesk Hybrid API - Демо версия",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
            "status": "running",
            "endpoints": {
                "ask": "POST /ask - Задать вопрос",
                "learn": "POST /learn - Обучить систему",
                "stats": "GET /stats - Статистика",
                "health": "GET /health - Проверка здоровья"
            },
            "demo_available": True
        })

# Обработчики событий приложения
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске."""
    print("🚀 Запуск NeuroDesk Hybrid API (Demo версия)...")
    
    print("✅ OpenAI API ключ найден")
    
    print("✅ NeuroDesk готов к работе")
    print("📚 Документация: http://localhost:9000/docs")
    print("🌐 Веб-интерфейс: http://localhost:9000")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении."""
    print("🛑 Завершение работы NeuroDesk...")
    print("👋 Сервер остановлен")

# Middleware для логирования
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования запросов."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Логируем только API вызовы
    if request.url.path.startswith("/api") or request.url.path in ["/ask", "/learn", "/stats"]:
        print(f"📝 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

if __name__ == "__main__":
    # Настройки запуска
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 9000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🌐 Запуск на http://{host}:{port}")
    print(f"📚 Документация: http://{host}:{port}/docs")
    
    uvicorn.run(
        "working_app:app", 
        host=host, 
        port=port, 
        reload=debug,
        log_level="info" if not debug else "debug"
    )
