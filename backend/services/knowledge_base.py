from pinecone import Pinecone, ServerlessSpec
import os
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from groq import Groq
import hashlib
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
import json


class GroqEmbeddings(Embeddings):
    
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        # We'll use a simple approach since Groq doesn't have dedicated embedding models
        # This is a workaround - in production, consider using a different embedding service
        self.chat_model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-20b",
            temperature=0.0
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        return self._get_text_embedding(text)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        try:
            math_keywords = [
                'equation', 'solve', 'derivative', 'integral', 'algebra', 'calculus',
                'geometry', 'trigonometry', 'quadratic', 'linear', 'polynomial',
                'function', 'graph', 'formula', 'theorem', 'proof', 'variable',
                'coefficient', 'expression', 'factorize', 'simplify', 'calculate'
            ]
            
            features = []
            text_lower = text.lower()
            
            for keyword in math_keywords:
                features.append(1.0 if keyword in text_lower else 0.0)
            
            features.append(len(text) / 1000.0)  # Normalized length
            features.append(len(text.split()) / 100.0)  # Normalized word count
            
            features.append(text.count('=') / 10.0)  # Equation indicators
            features.append(text.count('x') / 20.0)   # Variable indicators
            features.append(text.count('+') / 10.0)   # Operation indicators
            features.append(text.count('-') / 10.0)
            features.append(text.count('*') / 10.0)
            features.append(text.count('/') / 10.0)
            
            # Hash-based features for uniqueness
            text_hash = hashlib.md5(text.encode()).hexdigest()
            for i in range(0, len(text_hash), 2):
                features.append(int(text_hash[i:i+2], 16) / 255.0)
            
            # Pad or truncate to desired dimension (512 for this example)
            target_dim = 1024
            if len(features) < target_dim:
                features.extend([0.0] * (target_dim - len(features)))
            else:
                features = features[:target_dim]
            
            return features
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024

class PineconeKnowledgeBase:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Use custom Groq-based embeddings
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.embeddings = GroqEmbeddings(groq_api_key)
        
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]
            
            if self.index_name not in index_names:
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # Updated dimension for our custom embeddings
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"Index {self.index_name} created successfully")
            else:
                print(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            print(f"Error managing Pinecone index: {e}")
            raise

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar problems in knowledge base"""
        try:
            print(f"Searching for: {query[:50]}...")
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'problem': match['metadata'].get('problem', ''),
                    'solution': match['metadata'].get('solution', ''),
                    'steps': match['metadata'].get('steps', []),
                    'topic': match['metadata'].get('topic', ''),
                    'difficulty': match['metadata'].get('difficulty', '')
                })
            
            print(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []

    async def add_problem(self, problem: str, solution: str,
                         steps: List[str], topic: str, difficulty: str) -> bool:
        """Add a new problem to the knowledge base"""
        try:
            print(f"Adding problem: {problem[:50]}...")
            embedding = self.embeddings.embed_query(problem)
            
            # Create unique ID based on problem content
            problem_hash = hashlib.md5(problem.encode()).hexdigest()
            problem_id = f"math_problem_{problem_hash}"
            
            metadata = {
                'problem': problem,
                'solution': solution,
                'steps': steps if isinstance(steps, list) else [str(steps)],
                'topic': topic,
                'difficulty': difficulty
            }
            
            self.index.upsert(
                vectors=[(problem_id, embedding, metadata)]
            )
            
            print(f"Successfully added problem with ID: {problem_id}")
            return True
            
        except Exception as e:
            print(f"Error adding problem to knowledge base: {e}")
            return False
    
    async def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            print(f"Error getting KB stats: {e}")
            return {'error': str(e)}