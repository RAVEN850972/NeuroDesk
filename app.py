from fastapi import FastAPI, Request
from pydantic import BaseModel
from neurodesk import NeuroDesk
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
desk = NeuroDesk(user_id="test_web")

# Подключаем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

class AskRequest(BaseModel):
    prompt: str
    feedback: int | None = None
    domain: str | None = None  # Новый параметр для контекста

class AskResponse(BaseModel):
    answer: str
    source: str
    confidence: float | None = None
    explanation: list[str] | None = None

class LearningQualityResponse(BaseModel):
    report: dict

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        prompt = request.prompt.strip()
        context = {"domain": request.domain} if request.domain else None
        raw = desk.ask(
            prompt,
            session_id=desk.session_id,
            explanation_level="detailed",
            context=context,
            feedback=request.feedback
        )

        # Разбираем ответ для определения источника
        source = "openai" if "(ответ нейросети)" in raw else "graph"
        explanation = None
        confidence = None

        if source == "graph":
            last_response = desk.engine.get_last_response()  # Предполагаем, что такой метод есть
            if last_response:
                confidence = last_response.confidence
                explanation = last_response.explanation

        return AskResponse(
            answer=raw,
            source=source,
            confidence=confidence,
            explanation=explanation
        )
    except Exception as e:
        return AskResponse(answer=f"\u274c Ошибка: {e}", source="error")

@app.get("/learning_quality", response_model=LearningQualityResponse)
async def get_learning_quality():
    try:
        report = desk.get_learning_quality_report()
        return LearningQualityResponse(report=report)
    except Exception as e:
        return LearningQualityResponse(report={"error": str(e)})

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)