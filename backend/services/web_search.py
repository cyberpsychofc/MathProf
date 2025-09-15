from tavily import TavilyClient
import os
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class TavilyWebSearchService:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    async def search_math_problem(self, query: str, max_results: int = 5) -> Dict:
        try:
            formatted_query = f"how to solve step by step: {query} mathematics"
            
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                self.executor,
                self._tavily_search,
                formatted_query,
                max_results
            )
            
            formatted_results = await self._process_search_results(search_results, query)
            
            return {
                'found': len(formatted_results) > 0,
                'results': formatted_results,
                'original_query': query,
                'search_query': formatted_query
            }
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {
                'found': False,
                'results': [],
                'error': str(e),
                'original_query': query
            }
    
    def _tavily_search(self, query: str, max_results: int) -> Dict:
        return self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=["khanacademy.org", "mathway.com", "symbolab.com", 
                           "wolframalpha.com", "math.stackexchange.com", 
                           "brilliant.org", "coursera.org", "mit.edu"],
            exclude_domains=["pinterest.com", "instagram.com", "facebook.com"]
        )
    
    async def _process_search_results(self, search_results: Dict, original_query: str) -> List[Dict]:
        processed_results = []
        
        for result in search_results.get('results', []):
            try:
                title = result.get('title', '')
                content = result.get('content', '')
                url = result.get('url', '')
                score = result.get('score', 0.0)
                
                if self._is_math_content(title, content):
                    processed_result = {
                        'title': title,
                        'content': content,
                        'url': url,
                        'relevance_score': score,
                        'source': 'web_search',
                        'extracted_steps': await self._extract_solution_steps(content),
                        'confidence': min(score * 0.8, 0.9)  # Web results slightly less confident
                    }
                    processed_results.append(processed_result)
                    
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return processed_results[:3]
    
    def _is_math_content(self, title: str, content: str) -> bool:
        math_keywords = [
            'solve', 'equation', 'formula', 'calculate', 'step by step',
            'algebra', 'calculus', 'geometry', 'trigonometry', 'statistics',
            'derivative', 'integral', 'theorem', 'proof', 'solution'
        ]
        
        combined_text = (title + ' ' + content).lower()
        return any(keyword in combined_text for keyword in math_keywords)
    
    async def _extract_solution_steps(self, content: str) -> List[str]:
        try:
            sentences = content.split('.')
            steps = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(indicator in sentence.lower() for indicator in 
                       ['step', 'first', 'then', 'next', 'finally', 'therefore']):
                    if len(sentence) > 10 and len(sentence) < 200:
                        steps.append(sentence)
            
            return steps[:5]  # Max 5 steps
            
        except Exception:
            return ["Solution steps available in the linked resource"]