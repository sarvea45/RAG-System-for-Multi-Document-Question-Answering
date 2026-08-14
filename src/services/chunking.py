from typing import List, Dict, Any
import tiktoken

def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def chunk_document(
    pages_data: List[Dict[str, Any]], 
    document_id: str, 
    filename: str, 
    chunk_size: int = 1000, 
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Splits raw text into manageable chunks while preserving page metadata.
    Uses character-based sliding window.
    """
    chunks = []
    
    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]
        
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Find the last space to avoid cutting words in half if possible
            if end < text_length:
                last_space = chunk_text.rfind(" ")
                if last_space != -1 and last_space > overlap:
                    end = start + last_space
                    chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_num
                }
            })
            
            start = end - overlap

    return chunks
