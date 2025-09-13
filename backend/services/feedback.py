import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging
from services.knowledge_base import PineconeKnowledgeBase

logger = logging.getLogger(__name__)

class FeedbackService:
    def __init__(self):
        self.kb = PineconeKnowledgeBase()
        self.feedback_file = "data/feedback_log.json"
        self._ensure_feedback_file()
    
    def _ensure_feedback_file(self):
        """Create feedback log file if it doesn't exist"""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)
    
    async def submit_feedback(self, feedback_data: Dict) -> Dict:
        """Process user feedback and improve the system"""
        try:
            feedback_data['timestamp'] = datetime.utcnow().isoformat()
            feedback_data['processed'] = False
            
            await self._log_feedback(feedback_data)
            
            if feedback_data.get('rating', 5) <= 2:
                await self._process_negative_feedback(feedback_data)
            
            elif feedback_data.get('rating', 1) >= 4:
                await self._process_positive_feedback(feedback_data)
            
            return {
                'status': 'success',
                'message': 'Feedback received and processed',
                'feedback_id': feedback_data.get('feedback_id')
            }
            
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            return {
                'status': 'error',
                'message': 'Failed to process feedback',
                'error': str(e)
            }
    
    async def _log_feedback(self, feedback_data: Dict):
        """Log feedback to file"""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_log = json.load(f)
            
            feedback_log.append(feedback_data)
            
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Feedback logging error: {e}")
    
    async def _process_negative_feedback(self, feedback_data: Dict):
        """Handle negative feedback to improve system"""
        try:
            feedback_id = feedback_data.get('feedback_id')
            comments = feedback_data.get('comments', '')
            rating = feedback_data.get('rating', 1)
            
            logger.warning(f"Negative feedback received for {feedback_id}: {comments}")
            
            # Mark for manual review
            feedback_data['requires_manual_review'] = True
            feedback_data['processed'] = True
            
            # Flag if Groq-generated solution received poor feedback
            if feedback_data.get('source') == 'groq_generated':
                logger.warning(f"Groq solution received low rating: {feedback_id}")
            
        except Exception as e:
            logger.error(f"Negative feedback processing error: {e}")
    
    async def _process_positive_feedback(self, feedback_data: Dict):
        """Use positive feedback to improve knowledge base"""
        try:
            feedback_id = feedback_data.get('feedback_id')
            rating = feedback_data.get('rating', 5)
            
            # If this was a web search or Groq result that worked well,
            # consider adding it to knowledge base
            source = feedback_data.get('source')
            if source in ['web_search', 'groq_generated'] and rating >= 4:
                await self._add_successful_solution_to_kb(feedback_data)
            
            feedback_data['processed'] = True
            logger.info(f"Positive feedback processed for {feedback_id}")
            
        except Exception as e:
            logger.error(f"Positive feedback processing error: {e}")
    
    async def _add_successful_solution_to_kb(self, feedback_data: Dict):
        """Add well-rated solutions to knowledge base"""
        try:
            # Extract solution details from feedback
            original_response = feedback_data.get('original_response', {})
            
            if original_response:
                success = await self.kb.add_problem(
                    problem=original_response.get('question', ''),
                    solution=original_response.get('answer', ''),
                    steps=original_response.get('steps', []),
                    topic='general',  # Could be enhanced with topic detection
                    difficulty='intermediate'
                )
                
                if success:
                    logger.info(f"Added successful solution to knowledge base: {feedback_data.get('feedback_id')}")
                
        except Exception as e:
            logger.error(f"KB addition error: {e}")
    
    async def get_feedback_analytics(self) -> Dict:
        """Get feedback analytics for system improvement"""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_log = json.load(f)
            
            total_feedback = len(feedback_log)
            if total_feedback == 0:
                return {'total': 0, 'average_rating': 0, 'source_breakdown': {}}
            
            # Calculate metrics
            ratings = [f.get('rating', 3) for f in feedback_log]
            avg_rating = sum(ratings) / len(ratings)
            
            # Source breakdown
            source_counts = {}
            for f in feedback_log:
                source = f.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Groq-specific analytics
            groq_feedback = [f for f in feedback_log if f.get('source', '').startswith('groq')]
            groq_avg_rating = 0
            if groq_feedback:
                groq_ratings = [f.get('rating', 3) for f in groq_feedback]
                groq_avg_rating = sum(groq_ratings) / len(groq_ratings)
            
            return {
                'total_feedback': total_feedback,
                'average_rating': round(avg_rating, 2),
                'groq_average_rating': round(groq_avg_rating, 2),
                'source_breakdown': source_counts,
                'recent_feedback': feedback_log[-5:] if total_feedback >= 5 else feedback_log
            }
            
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {'error': str(e)}