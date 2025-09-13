from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP

# services
from services.main_router import MathRoutingService
from services.feedback import FeedbackService

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Math Routing Agent API (Groq Edition)",
    description="AI-powered mathematical problem solver with Groq LLMs, guardrails and feedback",
    version="2.0.0-groq"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=eval(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

math_router = MathRoutingService()
feedback_service = FeedbackService()

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
    feedback_id: str
    metadata: Optional[dict] = None

class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: int
    comments: Optional[str] = None
    source: Optional[str] = None
    original_response: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "Math Routing Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "math-routing-agent-groq", "version": "2.0-groq"}

@app.post("/api/v1/solve", response_model=MathResponse)
async def solve_math_problem(query: MathQuery):
    """Solve math problem using routing logic with Groq"""
    try:
        logger.info(f"Solving with Groq: {query.question[:50]}...")
        
        result = await math_router.solve_math_problem(
            question=query.question,
            difficulty_level=query.difficulty_level
        )
        
        return MathResponse(**result)
        
    except Exception as e:
        logger.error(f"Solve endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit feedback for continuous improvement"""
    try:
        # Process feedback in background
        background_tasks.add_task(
            feedback_service.submit_feedback,
            feedback.dict()
        )
        
        return {"message": "Feedback received", "feedback_id": feedback.feedback_id}
        
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing error: {str(e)}")

@app.get("/api/v1/analytics")
async def get_analytics():
    """Get system analytics and feedback metrics"""
    try:
        analytics = await feedback_service.get_feedback_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@app.post("/api/v1/init-dataset")
async def initialize_dataset():
    """Initialize sample dataset (for development)"""
    try:
        from .services.knowledge_base import MathDatasetCreator
        creator = MathDatasetCreator()
        count = await creator.create_sample_dataset()
        return {"message": f"Dataset initialized with {count} problems"}
    except Exception as e:
        logger.error(f"Dataset init error: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset initialization error: {str(e)}")

@app.get("/api/v1/kb-stats")
async def get_kb_stats():
    try:
        from .services.knowledge_base import PineconeKnowledgeBase
        kb = PineconeKnowledgeBase()
        stats = await kb.get_stats()
        return stats
    except Exception as e:
        logger.error(f"KB stats error: {e}")
        raise HTTPException(status_code=500, detail=f"KB stats error: {str(e)}")

@app.post("/api/v1/test-groq")
async def test_groq_connection():
    """Test Groq API connection"""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'Groq connection test successful'"}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.0,
            max_tokens=20
        )
        
        return {
            "status": "success",
            "message": completion.choices[0].message.content,
            "model": "openai/gpt-oss-20b"
        }
    except Exception as e:
        logger.error(f"Groq test error: {e}")
        raise HTTPException(status_code=500, detail=f"Groq connection error: {str(e)}")

if __name__ == "__main__":
    mcp = FastApiMCP(app)
    mcp.setup_server()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)