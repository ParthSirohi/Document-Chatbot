"""Simple LLM integration - uses FREE models via Groq or Ollama"""
import requests
import json
from typing import Optional
import os

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

class SimpleLLM:
    def __init__(self, use_groq: bool = True, api_key: Optional[str] = None):
        """
        use_groq=True: Use Groq's free API (needs GROQ_API_KEY env var)
        use_groq=False: Use local Ollama (needs Ollama running on localhost:11434)
        """
        self.use_groq = use_groq
        self.api_key = api_key
        
        if use_groq:
            if not api_key:
                import os
                api_key = os.getenv('GROQ_API_KEY')
            self.api_key = api_key
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = "llama-3.3-70b-versatile"  # Free, fast model on Groq
            print(f"[OK] Using Groq API with model: {self.model}")
        else:
            self.base_url = "http://localhost:11434/v1"
            self.model = "mistral"  # Or any model you have in Ollama
            print(f"[OK] Using local Ollama with model: {self.model}")
    
    def answer_query(
        self,
        query: str,
        context: str,
        max_tokens: int = 900,
        chat_history: str = ""
    ) -> str:
        """
        Answer query using LLM with PDF context
        
        Args:
            query: User's question
            context: Relevant text from PDFs
            max_tokens: Max response length
        
        Returns:
            LLM's answer
        """
        
        system_prompt = """You are an expert regulatory analyst assistant.
You must answer ONLY from the supplied document context.
Your goals:
1) Be accurate, specific, and practical.
2) Prefer complete requirement breakdowns over one-line answers.
3) If the question is broad (for example "minimum bank requirement"), return a structured checklist.
4) Cite evidence inline using [Doc X] labels from context.
5) If context is insufficient, clearly state what is missing and ask one focused follow-up question.

Response format:
- First line: direct answer in 1-2 sentences.
- Then sections when relevant:
  - Key Requirements
  - Capital Requirement Details (amount, composition/class, conditions, timelines, exceptions)
  - Practical Interpretation
  - Source Notes
Do not invent facts not present in context."""

        history_block = f"Conversation so far:\n{chat_history}\n\n" if chat_history else ""
        user_message = f"""{history_block}Context from documents:
{context}

Question: {query}

Answer:"""
        
        try:
            if self.use_groq:
                return self._call_groq(system_prompt, user_message, max_tokens)
            else:
                return self._call_ollama(system_prompt, user_message, max_tokens)
        except Exception as e:
            return f"[Error] Could not generate answer: {str(e)}"
    
    def _call_groq(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        """Call Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
    
    def _call_ollama(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        """Call local Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"Ollama error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure it's running on localhost:11434")


if __name__ == "__main__":
    # Test with Ollama
    llm = SimpleLLM(use_groq=False)
    
    context = "Banks require minimum capital of 10% of assets."
    query = "What is the minimum capital requirement?"
    
    answer = llm.answer_query(query, context)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
