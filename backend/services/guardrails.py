from nemoguardrails import LLMRails, RailsConfig
from typing import Dict, Any
import os

class GuardrailsService:
    def __init__(self):
        config_path = 'config'
        self.config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(self.config)
    
    async def check_input(self, user_message: str) -> Dict[str, Any]:
        try:
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}]
            )
            return {
                "passed": True,
                "original_message": user_message,
                "filtered_message": response.get("content", user_message),
                "violations": []
            }
        except Exception as e:
            return {
                "passed": False,
                "original_message": user_message,
                "filtered_message": None,
                "violations": [str(e)]
            }
    
    async def check_output(self, bot_response: str) -> Dict[str, Any]:
        return {
            "passed": True,
            "original_response": bot_response,
            "filtered_response": bot_response,
            "violations": []
        }