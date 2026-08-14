from typing import List, Dict, Any, Optional
from groq import Groq
from src.core.config import settings
import json

# Initialize Groq client
client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are an intelligent document assistant. You will be provided with context chunks from uploaded documents. Answer the user's question using ONLY the provided context. 
If the answer cannot be found in the context, you must reply exactly with: 'I could not find an answer in the provided documents.' 
Do not attempt to guess or use outside knowledge. 
For every claim you make, append a citation in the format [Filename, Page X].

You MUST output your response as a raw JSON object (do not wrap in markdown blocks like ```json).
The JSON object must match this exact schema:
{
  "answer": "Your answer string here, including citations in the text like [Filename, Page X].",
  "citations": [
    {
      "document_name": "filename.pdf",
      "page_number": 1
    }
  ]
}
"""

def build_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved metadata into a readable text block."""
    block = "--- CONTEXT START ---\n"
    for chunk in retrieved_chunks:
        block += f"Source: {chunk.get('filename')} (Page {chunk.get('page_number')})\n"
        block += f"Text: {chunk.get('text')}\n\n"
    block += "--- CONTEXT END ---"
    return block

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Calls Groq API to generate the response based on context and history."""
    
    context_block = build_context_block(retrieved_chunks)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Inject conversational history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Inject current query + context
    user_content = f"{context_block}\n\nUser Question: {query}"
    messages.append({"role": "user", "content": user_content})
    
    # Use LLaMA 3 8B model via Groq for blazing fast inference
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    raw_response = response.choices[0].message.content
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        # Fallback if LLM fails JSON format
        return {
            "answer": raw_response,
            "citations": []
        }
