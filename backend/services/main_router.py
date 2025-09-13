import uuid
from typing import Dict, List, Optional
import logging
from services.knowledge_base import PineconeKnowledgeBase
from .web_search import TavilyWebSearchService
from services.guardrails import GuardrailsService
from groq import Groq
from langchain_groq import ChatGroq
import os
import json

logger = logging.getLogger(__name__)

class MathRoutingService:
    def __init__(self):
        self.kb = PineconeKnowledgeBase()
        self.web_search = TavilyWebSearchService()
        self.guardrails = GuardrailsService()
        
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.langchain_groq = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b", 
            temperature=0.1
        )
        
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
    
    async def solve_math_problem(self, question: str, difficulty_level: str = "intermediate") -> Dict:
        """Main routing logic for solving math problems"""
        try:
            problem_id = str(uuid.uuid4())
            
            logger.info(f"Processing question: {question[:50]}...")
            guardrail_result = await self.guardrails.check_input(question)
            
            if not guardrail_result['passed']:
                return {
                    'question': question,
                    'answer': 'I can only help with mathematical problems. Please ask a math-related question.',
                    'steps': ['Question filtered by content guardrails'],
                    'source': 'guardrails',
                    'confidence': 0.0,
                    'feedback_id': problem_id,
                    'violations': guardrail_result['violations']
                }
            
            # Step 2: Search knowledge base
            kb_results = await self.kb.search(question, top_k=3)
            
            # Step 3: Decide routing based on similarity score
            if kb_results and kb_results[0]['score'] >= self.similarity_threshold:
                logger.info(f"Using knowledge base - similarity: {kb_results[0]['score']:.3f}")
                return await self._create_kb_response(kb_results[0], question, problem_id)
            
            # Step 4: Fall back to web search
            logger.info("Knowledge base match insufficient, using web search")
            web_results = await self.web_search.search_math_problem(question)
            
            if web_results['found'] and web_results['results']:
                return await self._create_web_response(web_results['results'][0], question, problem_id)
            
            # Step 5: Generate solution using Groq LLM as last resort
            logger.info("No web results found, generating LLM solution with Groq")
            return await self._generate_groq_solution(question, problem_id)
            
        except Exception as e:
            logger.error(f"Error in math routing: {e}")
            return {
                'question': question,
                'answer': 'I encountered an error while processing your question. Please try again.',
                'steps': ['An error occurred during processing'],
                'source': 'error',
                'confidence': 0.0,
                'feedback_id': str(uuid.uuid4()),
                'error': str(e)
            }
    
    async def _create_kb_response(self, kb_result: Dict, question: str, problem_id: str) -> Dict:
        """Create response from knowledge base result"""
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
            create a clear, step-by-step solution.
            
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