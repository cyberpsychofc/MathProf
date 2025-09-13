from datasets import load_dataset, concatenate_datasets
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from services.knowledge_base import PineconeKnowledgeBase

import asyncio
from datasets import load_dataset  # Requires: pip install datasets

class MathDatasetCreator:
    def __init__(self):
        self.kb = PineconeKnowledgeBase()
    
    async def create_JEE_Bench_Math_dataset(self):
        print("Loading JEEBench dataset...")
        dataset = load_dataset("PhysicsWallahAI/JEE-Main-2025-Math", split="test")
        
        math_problems_raw = [row for row in dataset if row['subject'] == 'math']
        print(f"Found {len(math_problems_raw)} math problems in JEEBench.")
        
        sample_problems = []
        for row in math_problems_raw:
            problem_data = {
                "problem": row['question'],
                "solution": row['gold'],  # e.g., 'C' for option C; for Integer types, this would be the numerical answer
                "steps": [
                    f"Type: {row['type']} (e.g., MCQ or Integer)",
                    "Refer to standard JEE solving methods for {row['description']}.",
                    "Detailed step-by-step solution not available in raw dataset; use agent for generation."
                ],
                "topic": "mathematics",  # Fixed for math subset; could infer sub-topic if added to schema
                "difficulty": "advanced"  # JEE Advanced level
            }
            sample_problems.append(problem_data)
        
        print("Adding JEEBench math problems to knowledge base...")
        success_count = 0
        
        for problem_data in sample_problems:
            success = await self.kb.add_problem(**problem_data)
            if success:
                success_count += 1
                print(f"✓ Added: {problem_data['problem'][:50]}...")
            else:
                print(f"✗ Failed: {problem_data['problem'][:50]}...")
        
        print(f"\nDataset creation completed: {success_count}/{len(sample_problems)} problems added")
        return success_count
    
    async def create_jee_2025_math_dataset(self):
        print("Loading JEE 2025 Math dataset...")
        
        jan_data = load_dataset("PhysicsWallahAI/JEE-Main-2025-Math", "jan", split="test")
        apr_data = load_dataset("PhysicsWallahAI/JEE-Main-2025-Math", "apr", split="test")

        dataset = concatenate_datasets([jan_data, apr_data])

        math_problems_raw = [row for row in dataset if row['question_type'] == 1]  # MCQs only
        print(f"Found {len(math_problems_raw)} MCQ math problems in JEE 2025 dataset.")

        sample_problems = []
        for row in math_problems_raw:
            problem_data = {
                "problem": row['question'],
                "solution": row['answer'],  # Final answer (NAT or symbolic form)
                "steps": [
                    f"Type: {'MCQ' if row['question_type'] == 1 else 'Integer'}",
                    "Refer to standard JEE solving methods for this problem.",
                    "Detailed step-by-step solution not available in raw dataset; use agent for generation."
                ],
                "topic": "mathematics",
                "difficulty": "advanced"
            }
            sample_problems.append(problem_data)

        print("Adding JEE 2025 math problems to knowledge base...")
        success_count = 0

        for problem_data in sample_problems:
            success = await self.kb.add_problem(**problem_data)
            if success:
                success_count += 1
                print(f"✓ Added: {problem_data['problem'][:50]}...")
            else:
                print(f"✗ Failed: {problem_data['problem'][:50]}...")

        print(f"\nDataset creation completed: {success_count}/{len(sample_problems)} problems added")
        return success_count
    
async def main():
    creator = MathDatasetCreator()
    await creator.create_jee_2025_math_dataset()

if __name__ == "__main__":
    asyncio.run(main())