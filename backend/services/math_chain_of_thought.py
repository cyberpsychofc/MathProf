import dspy
from langchain_groq import ChatGroq
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class GroqLM(dspy.BaseLM):
    def __init__(self, model="openai/gpt-oss-20b", api_key=None):
        super().__init__(model=model)
        self.client = ChatGroq(
            model=model, 
            api_key=api_key,
            temperature=0.1,
            max_tokens=1500
        )
    
    def basic_request(self, prompt, **kwargs):
        """DSPy expects this method for basic text generation"""
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"GroqLM basic_request error: {e}")
            return "Error in generation"
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        try:
            if messages:
                response = self.client.invoke(messages)
            elif prompt:
                response = self.client.invoke(prompt)
            else:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            
            return [response.content]
        except Exception as e:
            logger.error(f"GroqLM call error: {e}")
            return ["Error in generation"]

# Initialize DSPy with Groq
lm = GroqLM(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), 
    api_key=os.getenv("GROQ_API_KEY")
)
dspy.configure(lm=lm)

import os
import re
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict

import dspy

logger = logging.getLogger(__name__)

import dspy

class MathChainOfThought(dspy.Signature):
    """Solve a math problem step by step"""
    question = dspy.InputField()
    difficulty = dspy.InputField()
    context = dspy.InputField(optional=True)
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()
    confidence = dspy.OutputField()
    needs_human_review = dspy.OutputField()


class MathValidator(dspy.Signature):
    """Validate a math solution"""
    question = dspy.InputField()
    solution = dspy.InputField()
    steps = dspy.InputField()
    is_correct = dspy.OutputField()
    issues = dspy.OutputField(optional=True)
    improvement_suggestions = dspy.OutputField(optional=True)


class HumanFeedbackIncorporator(dspy.Signature):
    """Refine a solution with human feedback"""
    original_question = dspy.InputField()
    original_solution = dspy.InputField()
    human_feedback = dspy.InputField()
    improved_reasoning = dspy.OutputField()
    improved_answer = dspy.OutputField()
    learning_points = dspy.OutputField(optional=True)

class HITLFeedback:
    def __init__(
        self,
        feedback_id: str,
        original_question: str,
        original_answer: str,
        original_steps: List[str],
        human_feedback: str,
        improved_answer: str,
        improved_steps: List[str],
        feedback_type: str,
        timestamp: str,
        confidence_before: float,
        confidence_after: float,
    ):
        self.feedback_id = feedback_id
        self.original_question = original_question
        self.original_answer = original_answer
        self.original_steps = original_steps
        self.human_feedback = human_feedback
        self.improved_answer = improved_answer
        self.improved_steps = improved_steps
        self.feedback_type = feedback_type
        self.timestamp = timestamp
        self.confidence_before = confidence_before
        self.confidence_after = confidence_after


class DSPyMathOptimizer:
    def __init__(self):
        # Core modules
        self.math_solver = dspy.ChainOfThought(MathChainOfThought)
        self.validator = dspy.ChainOfThought(MathValidator)
        self.feedback_incorporator = dspy.ChainOfThought(HumanFeedbackIncorporator)

        # HITL components
        self.hitl_examples = []
        self.feedback_history = []
        self.learning_dataset = []

        # Load existing feedback history
        self._load_feedback_history()

        # Compile modules with available examples
        self._compile_modules()

    def _load_feedback_history(self):
        """Load previous HITL feedback for continuous learning"""
        try:
            feedback_file = "data/hitl_feedback.json"
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_history = [HITLFeedback(**item) for item in data]
                    logger.info(f"Loaded {len(self.feedback_history)} HITL feedback examples")
        except Exception as e:
            logger.error(f"Error loading feedback history: {e}")

    def _save_feedback_history(self):
        """Persist HITL feedback for future learning"""
        try:
            feedback_file = "data/hitl_feedback.json"
            os.makedirs("data", exist_ok=True)

            data = [
                {
                    "feedback_id": fb.feedback_id,
                    "original_question": fb.original_question,
                    "original_answer": fb.original_answer,
                    "original_steps": fb.original_steps,
                    "human_feedback": fb.human_feedback,
                    "improved_answer": fb.improved_answer,
                    "improved_steps": fb.improved_steps,
                    "feedback_type": fb.feedback_type,
                    "timestamp": fb.timestamp,
                    "confidence_before": fb.confidence_before,
                    "confidence_after": fb.confidence_after,
                }
                for fb in self.feedback_history
            ]

            with open(feedback_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving feedback history: {e}")

    def _compile_modules(self):
        """Compile DSPy modules with HITL examples for continuous learning"""
        try:
            trainset = self._create_training_examples()

            if len(trainset) >= 3:  # Need minimum examples for compilation
                def math_accuracy_metric(example, pred, trace=None):
                    try:
                        pred_answer = str(pred.answer).strip().lower()
                        true_answer = str(example.answer).strip().lower()
                        return pred_answer == true_answer
                    except:
                        return False

                teleprompter = dspy.BootstrapFewShot(
                    metric=math_accuracy_metric,
                    max_bootstrapped_demos=min(8, len(trainset)),
                    max_labeled_demos=min(4, len(trainset) // 2),
                )

                self.math_solver = teleprompter.compile(
                    self.math_solver, trainset=trainset[: min(10, len(trainset))]
                )

                logger.info(f"DSPy modules compiled with {len(trainset)} HITL examples")
            else:
                logger.info("Insufficient training data, using base modules")

        except Exception as e:
            logger.error(f"Module compilation error: {e}")

    def _create_training_examples(self) -> List[dspy.Example]:
        """Convert HITL feedback into DSPy training examples"""
        examples = []

        base_examples = [
            {
                "question": "Solve 2x + 3 = 7 for x",
                "difficulty": "beginner",
                "context": "",
                "reasoning": "Subtract 3 from both sides: 2x = 4. Divide both sides by 2: x = 2.",
                "answer": "x = 2",
                "confidence": "0.95",
                "needs_human_review": "false",
            },
            {
                "question": "Find the derivative of f(x) = x² + 3x - 2",
                "difficulty": "intermediate",
                "context": "",
                "reasoning": "Using power rule: d/dx(x²) = 2x, d/dx(3x) = 3, d/dx(-2) = 0. Therefore f'(x) = 2x + 3.",
                "answer": "f'(x) = 2x + 3",
                "confidence": "0.9",
                "needs_human_review": "false",
            },
        ]

        for ex in base_examples:
            examples.append(
                dspy.Example(
                    question=ex["question"],
                    difficulty=ex["difficulty"],
                    context=ex["context"],
                    reasoning=ex["reasoning"],
                    answer=ex["answer"],
                    confidence=ex["confidence"],
                    needs_human_review=ex["needs_human_review"],
                ).with_inputs("question", "difficulty", "context")
            )

        for feedback in self.feedback_history[-20:]:
            examples.append(
                dspy.Example(
                    question=feedback.original_question,
                    difficulty="intermediate",
                    context=f"Human feedback: {feedback.human_feedback}",
                    reasoning=" ".join(feedback.improved_steps),
                    answer=feedback.improved_answer,
                    confidence=str(feedback.confidence_after),
                    needs_human_review="false",
                ).with_inputs("question", "difficulty", "context")
            )

        return examples

    async def solve_with_dspy(self, question: str, difficulty: str = "intermediate", context: str = "") -> Dict:
        """Enhanced solve method with validation and HITL triggering"""
        try:
            prediction = self.math_solver(
                question=question, difficulty=difficulty, context=context
            )

            reasoning = str(prediction.reasoning) if hasattr(prediction, "reasoning") else ""
            answer = str(prediction.answer) if hasattr(prediction, "answer") else ""
            confidence = self._parse_confidence(prediction.confidence)
            needs_review = self._parse_boolean(prediction.needs_human_review)

            steps = self._extract_steps(reasoning)

            validation = self.validator(
                question=question, solution=answer, steps="\n".join(steps)
            )

            is_correct = self._parse_boolean(validation.is_correct)
            if not is_correct:
                confidence = min(confidence, 0.6)
                needs_review = True

            # 🔧 Confidence adjustment so it's not always 1.0
            # Cap maximum confidence and scale by difficulty
            if difficulty.lower() == "beginner":
                confidence = min(confidence, 0.9)
            elif difficulty.lower() == "intermediate":
                confidence = min(confidence, 0.85)
            else:  # advanced/hard/expert
                confidence = min(confidence, 0.8)

            # Add a small random jitter to avoid flat 1.0 outputs
            import random
            confidence = round(max(0.1, min(confidence - random.uniform(0, 0.1), 1.0)), 3)

            hitl_threshold = float(os.getenv("HITL_THRESHOLD", "0.7"))
            if confidence < hitl_threshold or needs_review:
                logger.info(f"Triggering HITL: confidence={confidence}, needs_review={needs_review}")
                needs_review = True

            return {
                "steps": steps,
                "answer": answer,
                "confidence": confidence,
                "needs_review": needs_review,
                "validation_issues": str(validation.issues) if hasattr(validation, "issues") else "",
                "metadata": {
                    "dspy_reasoning": reasoning,
                    "validation_passed": is_correct,
                    "improvement_suggestions": str(validation.improvement_suggestions)
                    if hasattr(validation, "improvement_suggestions")
                    else "",
                },
            }

        except Exception as e:
            logger.error(f"DSPy solve error: {e}")
            return {
                "steps": ["Error in solution generation"],
                "answer": "Unable to solve",
                "confidence": 0.0,
                "needs_review": True,
                "error": str(e),
            }

    async def incorporate_human_feedback(
        self,
        feedback_id: str,
        original_question: str,
        original_answer: str,
        original_steps: List[str],
        human_feedback: str,
        original_confidence: float,
    ) -> Dict:
        """Incorporate human feedback to improve the solution and learn for future"""
        try:
            improvement = self.feedback_incorporator(
                original_question=original_question,
                original_solution=original_answer,
                human_feedback=human_feedback,
            )

            improved_answer = str(improvement.improved_answer)
            improved_reasoning = str(improvement.improved_reasoning)
            improved_steps = self._extract_steps(improved_reasoning)
            learning_points = str(improvement.learning_points)

            hitl_feedback = HITLFeedback(
                feedback_id=feedback_id,
                original_question=original_question,
                original_answer=original_answer,
                original_steps=original_steps,
                human_feedback=human_feedback,
                improved_answer=improved_answer,
                improved_steps=improved_steps,
                feedback_type=self._classify_feedback_type(human_feedback),
                timestamp=datetime.utcnow().isoformat(),
                confidence_before=original_confidence,
                confidence_after=min(0.95, original_confidence + 0.2),
            )

            self.feedback_history.append(hitl_feedback)
            self._save_feedback_history()
            asyncio.create_task(self._recompile_with_new_feedback())

            logger.info(f"HITL feedback incorporated for {feedback_id}")

            return {
                "steps": improved_steps,
                "answer": improved_answer,
                "confidence": hitl_feedback.confidence_after,
                "learning_points": learning_points,
                "feedback_incorporated": True,
                "metadata": {
                    "feedback_type": hitl_feedback.feedback_type,
                    "original_confidence": original_confidence,
                    "improved_confidence": hitl_feedback.confidence_after,
                },
            }

        except Exception as e:
            logger.error(f"Feedback incorporation error: {e}")
            return {
                "steps": original_steps,
                "answer": original_answer,
                "confidence": original_confidence,
                "error": f"Failed to incorporate feedback: {str(e)}",
            }

    async def _recompile_with_new_feedback(self):
        """Recompile DSPy modules with new feedback (background task)"""
        try:
            await asyncio.sleep(1)
            self._compile_modules()
            logger.info("DSPy modules recompiled with latest feedback")
        except Exception as e:
            logger.error(f"Background recompilation error: {e}")

    def _classify_feedback_type(self, feedback: str) -> str:
        feedback_lower = feedback.lower()
        if any(word in feedback_lower for word in ["wrong", "incorrect", "error", "mistake"]):
            return "correction"
        elif any(word in feedback_lower for word in ["clarify", "explain", "unclear", "confusing"]):
            return "clarification"
        elif any(word in feedback_lower for word in ["correct", "good", "right", "yes"]):
            return "validation"
        else:
            return "enhancement"

    def _parse_confidence(self, confidence_str) -> float:
        try:
            if isinstance(confidence_str, (int, float)):
                return float(max(confidence_str, 0.0))
            numbers = re.findall(r"0?\.\d+|\d+\.?\d*", str(confidence_str))
            if numbers:
                conf = float(numbers[0])
                return min(max(conf, 0.0), 1.0)
            return 0.5
        except:
            return 0.5

    def _parse_boolean(self, bool_str) -> bool:
        try:
            if isinstance(bool_str, bool):
                return bool_str
            bool_lower = str(bool_str).lower().strip()
            return bool_lower in ["true", "1", "yes", "y", "correct"]
        except:
            return False

    def _extract_steps(self, reasoning: str) -> List[str]:
        if not reasoning:
            return ["No reasoning provided"]

        steps = []
        lines = reasoning.split("\n")
        current_step = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(
                indicator in line.lower()
                for indicator in [
                    "step",
                    "first",
                    "second",
                    "third",
                    "next",
                    "then",
                    "finally",
                    "therefore",
                    "1.",
                    "2.",
                    "3.",
                    "4.",
                    "5.",
                ]
            ):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line

        if current_step:
            steps.append(current_step.strip())

        if not steps:
            steps = [reasoning.strip()]

        return steps[:10]

    def get_learning_analytics(self) -> Dict:
        if not self.feedback_history:
            return {"total_feedback": 0, "learning_progress": "No data"}

        total_feedback = len(self.feedback_history)
        feedback_types = {}
        confidence_improvements = []

        for fb in self.feedback_history:
            feedback_types[fb.feedback_type] = feedback_types.get(fb.feedback_type, 0) + 1
            confidence_improvements.append(fb.confidence_after - fb.confidence_before)

        avg_improvement = sum(confidence_improvements) / len(confidence_improvements)

        return {
            "total_hitl_sessions": total_feedback,
            "feedback_type_breakdown": feedback_types,
            "average_confidence_improvement": round(avg_improvement, 3),
            "learning_dataset_size": len(self.learning_dataset),
            "recent_feedback": [
                {
                    "feedback_id": fb.feedback_id,
                    "question": fb.original_question[:50] + "...",
                    "feedback_type": fb.feedback_type,
                    "improvement": fb.confidence_after - fb.confidence_before,
                    "timestamp": fb.timestamp,
                }
                for fb in self.feedback_history[-5:]
            ],
        }
        return analytics