from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Math Routing Agent API",
    description="AI-powered mathematical problem solver with guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MathQuery(BaseModel):
    question: str
    difficulty_level: Optional[str] = "intermediate"
    topic: Optional[str] = None

class MathResponse(BaseModel):
    question: str
    answer: str
    steps: List[str]
    source: str
    confidence: float
    feedback_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: int
    comments: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Math Routing Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "math-routing-agent"}

@app.post("/api/v1/solve", response_model=MathResponse)
async def solve_math_problem(query: MathQuery):
    return JSONResponse(
        content={
            "question": query.question,
            "answer": "Solution will be implemented",
            "steps": ["Step 1: Parse question", "Step 2: Route to appropriate solver"],
            "source": "placeholder",
            "confidence": 0.0
        }
    )

@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    return {"message": "Feedback received", "feedback_id": feedback.feedback_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)