from typing import TypedDict, Annotated, List, Optional, Dict
from langgraph.graph import StateGraph, END
import operator

class MathAgentState(TypedDict):
    question: str
    difficulty: str
    topic: Optional[str]
    steps: List[str]
    answer: Optional[str]
    confidence: float
    source: str
    feedback_id: str
    metadata: Dict
    needs_review: bool
    human_input: Optional[str]
    dspy_output: Optional[Dict]
    
    # New HITL-specific fields
    validation_result: Optional[Dict]
    learning_context: Optional[str]
    feedback_history: List[Dict]
    improvement_suggestions: List[str]
    hitl_session_active: bool