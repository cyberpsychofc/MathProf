from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

class DifficultyLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"

class MathTopic(str, Enum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    TRIGONOMETRY = "trigonometry"

class GuardrailResult(BaseModel):
    passed: bool
    violations: List[str]
    filtered_content: Optional[str] = None

class KnowledgeBaseResult(BaseModel):
    found: bool
    content: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Optional[Dict] = None