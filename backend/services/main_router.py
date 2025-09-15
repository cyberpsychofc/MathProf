import uuid
from typing import Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from services.knowledge_base import PineconeKnowledgeBase
from services.feedback import FeedbackService
from services.math_chain_of_thought import DSPyMathOptimizer
from agents.math_agent import MathAgentState
from .web_search import TavilyWebSearchService
from services.guardrails import GuardrailsService
from utils.generic import convert_numpy_types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from langchain_groq import ChatGroq
import os
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)

class MathRoutingService:
    def __init__(self):
        self.kb = PineconeKnowledgeBase()
        self.web_search = TavilyWebSearchService()
        self.guardrails = GuardrailsService()
        self.feedback_service = FeedbackService()
        self.dspy_optimizer = DSPyMathOptimizer()  # New: DSPy integration
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Keep existing
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
        self.hitl_threshold = float(os.getenv("HITL_THRESHOLD", "0.7"))  # New: Trigger HITL if < this
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.langchain_groq = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
            temperature=0.1
        )
        
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        

    async def solve_math_problem(self, question: str, difficulty_level: str = "intermediate") -> Dict:
        try:
            problem_id = str(uuid.uuid4())
            
            logger.info(f"Processing question: {question[:50]}...")
            guardrail_result = await self.guardrails.check_input(question)
            
            if not guardrail_result['passed']:
                result = {
                    'question': question,
                    'answer': 'I can only help with mathematical problems. Please ask a math-related question.',
                    'steps': ['Question filtered by content guardrails'],
                    'source': 'guardrails',
                    'confidence': 0.0,
                    'feedback_id': problem_id,
                    'violations': guardrail_result['violations']
                }
                return convert_numpy_types(result)
            
            # Step 2: Search knowledge base
            kb_results = await self.kb.search(question, top_k=5)  # Get more candidates for retriever
            
            # Step 2.5: Apply retriever enhancement
            if kb_results:
                enhanced_results = await self._enhance_retrieval_results(
                    question, kb_results, difficulty_level
                )
                
                # Step 3: Decide routing based on enhanced similarity score
                if enhanced_results and enhanced_results[0]['score'] >= self.similarity_threshold:
                    logger.info(f"Using knowledge base - enhanced similarity: {enhanced_results[0]['score']:.3f}")
                    result = await self._create_rag_enhanced_kb_response(
                        enhanced_results, question, problem_id, difficulty_level
                    )
                    return convert_numpy_types(result)
            
            # Step 4: Fall back to web search
            logger.info("Knowledge base match insufficient, using web search")
            web_results = await self.web_search.search_math_problem(question)
            
            if web_results['found'] and web_results['results']:
                result = await self._create_web_response(web_results['results'][0], question, problem_id)
                return convert_numpy_types(result)
            
            # Step 5: Generate solution using Groq LLM as last resort
            logger.info("No web results found, generating LLM solution with Groq")
            result = await self._generate_groq_solution(question, problem_id)
            return convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"Error in math routing: {e}")
            result = {
                'question': question,
                'answer': 'I encountered an error while processing your question. Please try again.',
                'steps': ['An error occurred during processing'],
                'source': 'error',
                'confidence': 0.0,
                'feedback_id': str(uuid.uuid4()),
                'error': str(e)
            }
            return convert_numpy_types(result)

    async def _enhance_retrieval_results(
        self, 
        question: str, 
        kb_results: List[Dict], 
        difficulty_level: str
    ) -> List[Dict]:
        """
        Retriever First Function: Enhance KB search results through re-ranking and filtering
        """
        try:
            logger.info("Enhancing retrieval results with advanced scoring")
            
            if not kb_results:
                return []
            
            enhanced_results = []
            
            # Extract text content for TF-IDF analysis
            question_lower = question.lower()
            result_texts = []
            
            for result in kb_results:
                # Combine problem, solution, and steps for comprehensive matching
                combined_text = f"{result.get('problem', '')} {result.get('solution', '')} {' '.join(result.get('steps', []))}"
                result_texts.append(combined_text)
            
            # Apply TF-IDF based re-ranking if we have multiple results
            if len(result_texts) > 1:
                try:
                    # Create TF-IDF vectors
                    all_texts = [question] + result_texts
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                    
                    # Calculate cosine similarities
                    question_vector = tfidf_matrix[0:1]
                    result_vectors = tfidf_matrix[1:]
                    tfidf_similarities = cosine_similarity(question_vector, result_vectors)[0].tolist()
        
                except Exception as tfidf_error:
                    logger.warning(f"TF-IDF re-ranking failed: {tfidf_error}")
                    tfidf_similarities = [0.5] * len(kb_results)  # Default similarity
            else:
                tfidf_similarities = [0.5]
            
            for i, result in enumerate(kb_results):
                enhanced_result = result.copy()
                
                # Calculate enhanced score combining multiple factors
                original_score = result.get('score', 0.0)
                tfidf_score = tfidf_similarities[i]
                
                # Difficulty matching bonus
                difficulty_bonus = 0.0
                result_difficulty = result.get('difficulty', '').lower()
                if result_difficulty == difficulty_level.lower():
                    difficulty_bonus = 0.1
                elif self._is_difficulty_compatible(result_difficulty, difficulty_level):
                    difficulty_bonus = 0.05
                
                # Topic relevance bonus (check for math keywords)
                topic_bonus = self._calculate_topic_relevance(question_lower, result)
                
                # Length appropriateness (prefer solutions with reasonable step count)
                steps_count = len(result.get('steps', []))
                length_bonus = 0.0
                if 3 <= steps_count <= 8:  # Optimal step range
                    length_bonus = 0.05
                elif steps_count > 0:  # At least has steps
                    length_bonus = 0.02
                
                # Calculate final enhanced score
                enhanced_score = (
                    original_score * 0.5 +  # Original semantic similarity
                    tfidf_score * 0.3 +     # TF-IDF keyword matching
                    difficulty_bonus +       # Difficulty matching
                    topic_bonus +           # Topic relevance
                    length_bonus            # Solution quality indicator
                )
                
                enhanced_result['score'] = float(min(enhanced_score, 1.0))
                enhanced_result['scoring_breakdown'] = {
                    'original_score': float(original_score),
                    'tfidf_score': float(tfidf_similarities[i]),
                    'difficulty_bonus': float(difficulty_bonus),
                    'topic_bonus': float(topic_bonus),
                    'length_bonus': float(length_bonus),
                    'final_score': float(enhanced_result['score'])
                }
                
                enhanced_results.append(enhanced_result)
            
            # Sort by enhanced score
            enhanced_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Remove duplicates based on problem similarity
            filtered_results = self._remove_duplicate_results(enhanced_results)
            
            logger.info(f"Enhanced {len(kb_results)} results to {len(filtered_results)} unique results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in retrieval enhancement: {e}")
            return kb_results  # Return original results if enhancement fails

    def _is_difficulty_compatible(self, result_difficulty: str, target_difficulty: str) -> bool:
        """Check if difficulties are compatible"""
        difficulty_levels = {
            'beginner': 1, 'basic': 1, 'easy': 1,
            'intermediate': 2, 'medium': 2,
            'advanced': 3, 'hard': 3, 'expert': 3
        }
        
        result_level = difficulty_levels.get(result_difficulty, 2)
        target_level = difficulty_levels.get(target_difficulty, 2)
        
        # Compatible if within 1 level
        return abs(result_level - target_level) <= 1

    def _calculate_topic_relevance(self, question: str, result: Dict) -> float:
        """Calculate topic relevance bonus based on math keywords"""
        math_topics = {
            'algebra': ['equation', 'solve', 'variable', 'polynomial', 'linear', 'quadratic'],
            'calculus': ['derivative', 'integral', 'limit', 'differentiate', 'integrate'],
            'geometry': ['triangle', 'circle', 'angle', 'area', 'perimeter', 'volume'],
            'statistics': ['mean', 'median', 'probability', 'distribution', 'variance'],
            'trigonometry': ['sin', 'cos', 'tan', 'sine', 'cosine', 'tangent', 'angle']
        }
        
        topic_bonus = 0.0
        result_topic = result.get('topic', '').lower()
        
        for topic, keywords in math_topics.items():
            if any(keyword in question for keyword in keywords):
                if topic in result_topic:
                    topic_bonus = 0.1  # Strong topic match
                    break
                elif any(keyword in result.get('problem', '').lower() for keyword in keywords):
                    topic_bonus = 0.05  # Partial topic match
        
        return topic_bonus

    def _remove_duplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove near-duplicate results based on problem similarity"""
        if len(results) <= 1:
            return results
        
        unique_results = []
        for result in results:
            is_duplicate = False
            current_problem = result.get('problem', '').lower()
            
            for unique_result in unique_results:
                existing_problem = unique_result.get('problem', '').lower()
                
                # Simple duplicate detection based on problem text similarity
                if len(current_problem) > 0 and len(existing_problem) > 0:
                    # Calculate word overlap
                    current_words = set(current_problem.split())
                    existing_words = set(existing_problem.split())
                    
                    if len(current_words.union(existing_words)) > 0:
                        overlap_ratio = len(current_words.intersection(existing_words)) / len(current_words.union(existing_words))
                        if overlap_ratio > 0.8:  # 80% word overlap considered duplicate
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results

    async def _create_rag_enhanced_kb_response(
        self, 
        enhanced_results: List[Dict], 
        question: str, 
        problem_id: str,
        difficulty_level: str
    ) -> Dict:
        """
        RAG-Enhanced Response Function: Use retrieved KB content as context for LLM enhancement
        """
        try:
            logger.info("Creating RAG-enhanced KB response")
            
            # Get the best result
            best_result = enhanced_results[0]
            
            # Prepare context from multiple results if available
            context_parts = []
            for i, result in enumerate(enhanced_results[:3]):  # Use top 3 results as context
                context_part = f"Reference {i+1}:\n"
                context_part += f"Problem: {result.get('problem', 'N/A')}\n"
                context_part += f"Solution: {result.get('solution', 'N/A')}\n"
                context_part += f"Steps: {' → '.join(result.get('steps', ['N/A']))}\n"
                context_part += f"Topic: {result.get('topic', 'N/A')}\n"
                context_part += f"Difficulty: {result.get('difficulty', 'N/A')}\n"
                context_parts.append(context_part)
            
            context = "\n\n".join(context_parts)
            
            # Create RAG prompt for enhancement
            rag_prompt = f"""
You are a mathematics tutor. Use the following reference solutions to help answer the student's question, but tailor your response to their specific question and learning level.

STUDENT QUESTION: {question}
DIFFICULTY LEVEL: {difficulty_level}

REFERENCE SOLUTIONS:
{context}

INSTRUCTIONS:
1. Answer the student's specific question using insights from the reference solutions
2. Adapt the explanation style to the {difficulty_level} level
3. If the question differs from references, adapt the solution approach accordingly
4. Provide clear, step-by-step reasoning
5. Include mathematical notation where appropriate
6. If references are insufficient, clearly state what additional information might be needed

Please provide:
1. A clear, direct answer
2. Step-by-step solution process
3. Brief explanation of key concepts used

Format your response as a structured solution with clear steps.
"""

            # Generate enhanced response using Groq
            try:
                completion = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert mathematics tutor. Provide clear, accurate, and pedagogically sound mathematical explanations."
                        },
                        {
                            "role": "user",
                            "content": rag_prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                
                enhanced_answer = completion.choices[0].message.content
                
                # Extract steps from the enhanced answer
                enhanced_steps = self._extract_steps_from_response(enhanced_answer)
                
            except Exception as llm_error:
                logger.warning(f"RAG enhancement failed, using original KB response: {llm_error}")
                enhanced_answer = best_result.get('solution', 'Solution not available')
                enhanced_steps = best_result.get('steps', [])
            
            return {
                'question': question,
                'answer': enhanced_answer,
                'steps': enhanced_steps,
                'source': 'knowledge_base_rag_enhanced',
                'confidence': best_result['score'],
                'feedback_id': problem_id,
                'metadata': {
                    'original_problem': best_result.get('problem'),
                    'topic': best_result.get('topic'),
                    'difficulty': best_result.get('difficulty'),
                    'enhancement_method': 'rag',
                    'reference_count': len(enhanced_results),
                    'scoring_breakdown': best_result.get('scoring_breakdown', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG enhancement: {e}")
            # Fallback to original KB response
            return await self._create_kb_response(enhanced_results[0], question, problem_id)

    def _extract_steps_from_response(self, response: str) -> List[str]:
        """Extract step-by-step solution from LLM response"""
        steps = []
        lines = response.split('\n')
        
        current_step = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for step indicators
            if any(indicator in line.lower() for indicator in ['step', '1.', '2.', '3.', '4.', '5.']):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            elif current_step:
                current_step += " " + line
        
        # Add the last step
        if current_step:
            steps.append(current_step.strip())
        
        # If no clear steps found, try to split by sentences
        if not steps and response:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if len(sentences) > 1:
                steps = sentences[:5]  # Take first 5 sentences as steps
        
        return steps[:10]  # Limit to 10 steps maximum

    async def _create_kb_response(self, kb_result: Dict, question: str, problem_id: str) -> Dict:
        """Original KB response function (kept as fallback)"""
        return {
            'question': question,
            'answer': kb_result['solution'],
            'steps': kb_result['steps'],
            'source': 'knowledge_base',
            'confidence': kb_result['score'],
            'feedback_id': problem_id,
            'metadata': {
                'original_problem': kb_result['problem'],
                'topic': kb_result['topic'],
                'difficulty': kb_result['difficulty']
            }
        }
    
    async def _create_web_response(self, web_result: Dict, question: str, problem_id: str) -> Dict:
        """Create response from web search result"""
        # Enhance web content with Groq for better step-by-step solution
        enhanced_solution = await self._enhance_web_solution_with_groq(
            question, web_result['content'], web_result['extracted_steps']
        )
        
        return {
            'question': question,
            'answer': enhanced_solution['answer'],
            'steps': enhanced_solution['steps'],
            'source': 'web_search',
            'confidence': web_result['confidence'],
            'feedback_id': problem_id,
            'metadata': {
                'web_source': web_result['url'],
                'title': web_result['title'],
                'relevance_score': web_result['relevance_score']
            }
        }
    
    async def _enhance_web_solution_with_groq(self, question: str, web_content: str, extracted_steps: List[str]) -> Dict:
        """Use Groq to create clear step-by-step solution from web content"""
        try:
            prompt = f"""
            Based on the following web content about the math problem "{question}", 
            create a clear, step-by-step solution. Use words
            like 'step', 'first', 'then', 'next', 'finally', 'therefore' to separate steps.
            Do not give any context to the web content, just focus on the solution steps involving numbers.
            
            Web Content: {web_content[:1000]}
            
            Extracted Steps: {extracted_steps}
            
            Please provide:
            1. A concise final answer
            2. Clear step-by-step solution (3-7 steps)
            
            Format your response as JSON:
            {{
                "answer": "final answer here",
                "steps": ["step 1", "step 2", ...]
            }}
            
            Important: Respond ONLY with valid JSON, no additional text.
            """
            
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematical expert. Always respond with valid JSON format only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model="openai/gpt-oss-20b",  # Faster model for enhancement
                temperature=0.1,
                max_tokens=800
            )
            
            enhanced = json.loads(completion.choices[0].message.content)
            return enhanced
            
        except Exception as e:
            logger.warning(f"Groq enhancement failed: {e}")
            return {
                'answer': 'Solution available in the reference material',
                'steps': extracted_steps if extracted_steps else ['Check the provided web source for detailed solution']
            }
    
    async def _generate_groq_solution(self, question: str, problem_id: str) -> Dict:
        """Generate solution using Groq when no other sources available"""
        try:
            prompt = f"""
            Solve this math problem step by step: {question}
            
            Requirements:
            1. Provide a clear final answer
            2. Show detailed step-by-step solution
            3. Explain each step clearly
            4. Be educational and assume the student is learning
            
            Format as JSON:
            {{
                "answer": "final answer",
                "steps": ["step 1 with explanation", "step 2 with explanation", ...]
            }}
            
            Important: Respond ONLY with valid JSON, no additional text.
            """
            
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert mathematics tutor. Always provide accurate solutions and respond with valid JSON format only. Use step-by-step reasoning for mathematical problems."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="openai/gpt-oss-20b",  # Use larger model for better math reasoning
                temperature=0.05,  # Very low temperature for mathematical accuracy
                max_tokens=1000
            )
            
            solution = json.loads(completion.choices[0].message.content)
            
            return {
                'question': question,
                'answer': solution['answer'],
                'steps': solution['steps'],
                'source': 'groq_generated',
                'confidence': 0.8,
                'feedback_id': problem_id,
                'metadata': {
                    'model': 'openai/gpt-oss-20b',
                    'provider': 'groq',
                    'note': 'Generated solution using Groq OpenAI OSS 20B - verify for accuracy'
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return await self._fallback_groq_solution(question, problem_id)
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return {
                'question': question,
                'answer': 'I was unable to generate a solution. Please rephrase your question or try again.',
                'steps': ['Solution generation failed'],
                'source': 'error',
                'confidence': 0.0,
                'feedback_id': problem_id,
                'error': str(e)
            }
    
    async def _fallback_groq_solution(self, question: str, problem_id: str) -> Dict:
        """Fallback method for Groq solution generation with simpler prompt"""
        try:
            # Simpler prompt without JSON formatting
            simple_prompt = f"""
            Solve this math problem: {question}
            
            Please provide:
            1. Final answer
            2. Step-by-step solution
            
            Be clear and educational.
            """
            
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematics tutor. Provide clear, step-by-step solutions."
                    },
                    {
                        "role": "user",
                        "content": simple_prompt
                    }
                ],
                model="mixtral-8x7b-32768",  # Alternative model
                temperature=0.1,
                max_tokens=800
            )
            
            response_text = completion.choices[0].message.content
            
            # Parse the response manually
            lines = response_text.strip().split('\n')
            answer = "See solution steps"
            steps = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:
                    if "answer" in line.lower() or "solution" in line.lower():
                        answer = line
                    else:
                        steps.append(line)
            
            # Clean up steps
            steps = [step for step in steps if len(step) > 10][:7]
            
            return {
                'question': question,
                'answer': answer,
                'steps': steps if steps else ["Solution provided above"],
                'source': 'groq_fallback',
                'confidence': 0.7,
                'feedback_id': problem_id,
                'metadata': {
                    'model': 'mixtral-8x7b-32768',
                    'provider': 'groq',
                    'note': 'Fallback solution - manual parsing used'
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback Groq generation failed: {e}")
            return {
                'question': question,
                'answer': 'I apologize, but I was unable to solve this problem. Please try rephrasing your question.',
                'steps': ['Unable to generate solution'],
                'source': 'error',
                'confidence': 0.0,
                'feedback_id': problem_id,
                'error': str(e)
            }
    # Agent State Graph Construction
    def _build_graph(self):
        workflow = StateGraph(MathAgentState)

        # Node 1: Guardrails (reuse your logic)
        async def guardrail_node(state: MathAgentState):
            guardrail_result = await self.guardrails.check_input(state["question"])
            if not guardrail_result['passed']:
                return {
                    **state,
                    "answer": "I can only help with mathematical problems.",
                    "steps": ["Question filtered by content guardrails"],
                    "source": "guardrails",
                    "confidence": 0.0,
                    "needs_review": False,
                    "metadata": {"violations": guardrail_result['violations']}
                }
            if "Passed guardrails check" not in state["steps"]:
                state["steps"].append("Passed guardrails check")
            return state

        # Node 2: KB Retrieval & Enhancement (reuse your _enhance_retrieval_results)
        async def retrieve_node(state: MathAgentState):
            kb_results = await self.kb.search(state["question"], top_k=5)
            if kb_results:
                enhanced = await self._enhance_retrieval_results(state["question"], kb_results, state["difficulty"])
                state["metadata"] = state.get("metadata", {}) | {"kb_results": enhanced}
                if enhanced and enhanced[0]['score'] >= self.similarity_threshold:
                    return await self.rag_node(state)  # Route to RAG if high match
            state["steps"].append("No sufficient KB match")
            return state

        async def rag_node(state: MathAgentState):
            enhanced_results = state["metadata"].get("kb_results", [])
            if not enhanced_results:
                state["steps"].append("No enhanced KB results available")
                return state
            # Reuse your old method (adapt to state)
            problem_id = state["feedback_id"]
            rag_result = await self._create_rag_enhanced_kb_response(
                enhanced_results, state["question"], problem_id, state["difficulty"]
            )
            state["answer"] = rag_result.get("answer", "No RAG answer")
            state["steps"].extend(rag_result.get("steps", []))
            state["source"] = "knowledge_base_rag_enhanced"
            state["confidence"] = enhanced_results[0].get("score", 0.5)
            return state

        # Node 4: Web Search Fallback (reuse _create_web_response)
        async def web_node(state: MathAgentState):
            web_results = await self.web_search.search_math_problem(state["question"])
            if web_results['found'] and web_results['results']:
                problem_id = state["feedback_id"]
                web_result = await self._create_web_response(web_results['results'][0], state["question"], problem_id)
                state["answer"] = web_result.get("answer")
                state["steps"].extend(web_result.get("steps", []))
                state["source"] = "web_search"
                state["confidence"] = 0.75  # Or from metadata
            else:
                state["steps"].append("No web results found")
            return state

        async def generate_node(state: MathAgentState):
            dspy_result = await self.dspy_optimizer.solve_with_dspy(state["question"], state["difficulty"])
            state["dspy_output"] = dspy_result
            state["answer"] = dspy_result["answer"]
            state["steps"].extend(dspy_result["steps"])
            state["source"] = "dspy_groq_generated"
            state["confidence"] = dspy_result["confidence"]
            state["steps"].append("DSPy-optimized generation used")
            return state

        # HITL Node: Pause for review
        async def hitl_node(state: MathAgentState):
            state["feedback_id"] = str(uuid.uuid4())
            state["needs_review"] = True
            # Log partial state to feedback
            partial_data = {
                "feedback_id": state["feedback_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "original_response": {
                    "question": state["question"],
                    "steps": state["steps"],
                    "answer": state["answer"],
                    "confidence": state["confidence"]
                },
                "source": state["source"],
                "comments": "Paused for human review due to low confidence",
                "rating": None
            }
            await self.feedback_service._log_feedback(partial_data)
            logger.info(f"HITL pause for {state['feedback_id']}: confidence {state['confidence']}")
            return state

        # Resume Node: Incorporate human input (refine with DSPy)
        async def resume_node(state: MathAgentState):
            if state["human_input"]:
                # Augment question with human input and re-run DSPy
                augmented_question = f"{state['question']} Human correction: {state['human_input']}"
                dspy_result = await self.dspy_optimizer.solve_with_dspy(augmented_question, state["difficulty"])
                state["answer"] = dspy_result["answer"]
                state["steps"].extend(dspy_result["steps"])
                state["confidence"] = max(state["confidence"], 0.9)  # Boost post-HITL
                state["steps"].append(f"Incorporated human input: {state['human_input']}")
                # Log as positive feedback
                update_data = {"feedback_id": state["feedback_id"], "rating": 4, "comments": state["human_input"]}
                await self.feedback_service.submit_feedback(update_data)
            state["needs_review"] = False
            return state

        # Conditional Routing
        def route_after_retrieve(state: MathAgentState):
            if state["metadata"] and state["metadata"].get("kb_results", [{}])[0].get("score", 0) >= self.similarity_threshold:
                return "rag"
            return "web"

        def route_after_rag_or_web_or_generate(state: MathAgentState):
            if state["confidence"] < self.hitl_threshold:
                return "hitl"
            return "complete"

        # Nodes
        workflow.add_node("guardrail", guardrail_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("rag", rag_node)
        workflow.add_node("web", web_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("hitl", hitl_node)
        workflow.add_node("resume", resume_node)

        # Edges
        workflow.set_entry_point("guardrail")
        workflow.add_edge("guardrail", "retrieve")
        workflow.add_conditional_edges(
            "retrieve", 
            route_after_retrieve, 
            {"rag": "rag", "web": "web"}
        )
        workflow.add_edge("rag", "generate")  # Direct to generate (no route_check)
        workflow.add_edge("web", "generate")
        workflow.add_conditional_edges(
            "generate", 
            route_after_rag_or_web_or_generate, 
            {"hitl": "hitl", "complete": END}  # Rename branch for clarity
        )
        workflow.add_edge("hitl", END)
        workflow.add_edge("resume", END)
        async def route_check(state): return state
        workflow.add_node("route_check", route_check)
        workflow.add_edge("route_check", "generate")

        app = workflow.compile(checkpointer=self.checkpointer)
        return app

    async def solve_math_problem(self, question: str, difficulty_level: str = "intermediate", topic: Optional[str] = None):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # For checkpointing
        initial_state = {
            "question": question,
            "difficulty": difficulty_level,
            "topic": topic,
            "steps": [],
            "confidence": 0.0,
            "needs_review": False,
            "human_input": None,
            "feedback_id": str(uuid.uuid4()),
            "metadata": {}
        }
        final_state = await self.graph.ainvoke(initial_state, config)
        if final_state["needs_review"]:
            return {
                **final_state,
                "status": "paused_for_review"  # For API handling
            }
        return {
            "question": final_state["question"],
            "answer": final_state["answer"] or "No solution generated",
            "steps": final_state["steps"],
            "source": final_state["source"],
            "confidence": final_state["confidence"],
            "feedback_id": final_state["feedback_id"],
            "metadata": final_state["metadata"]
        }

    async def resume_with_input(self, feedback_id: str, human_input: str, config: Optional[dict] = None):
        if not config:
            config = {"configurable": {"thread_id": feedback_id}}
        partial_state = await self._load_partial_state(feedback_id)
        partial_state["human_input"] = human_input
        final_state = await self.graph.ainvoke(partial_state, config)
        return {
            "question": final_state["question"],
            "answer": final_state["answer"],
            "steps": final_state["steps"],
            "source": final_state["source"],
            "confidence": final_state["confidence"],
            "feedback_id": feedback_id,
            "metadata": {**final_state["metadata"], "human_refined": True}
        }

    async def _load_partial_state(self, feedback_id: str) -> MathAgentState:
        partial = await self.feedback_service.load_partial_state(feedback_id)
        if partial:
            return {
                "question": partial.get("question", ""),
                "difficulty": partial.get("difficulty", "intermediate"),
                "steps": partial.get("steps", []),
                "answer": partial.get("answer"),
                "confidence": partial.get("confidence", 0.0),
                "source": partial.get("source", "unknown"),
                "needs_review": True,
                "human_input": None,
                "feedback_id": feedback_id,
                "metadata": partial.get("metadata", {}),
                "dspy_output": None  # Re-generate if needed
            }
        raise ValueError(f"No partial state found for {feedback_id}. Ensure you paused a solve first.")