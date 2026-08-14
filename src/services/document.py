import fitz # PyMuPDF
import docx
import io
from typing import List, Dict, Any

def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts text from PDF, returning a list of dicts with text and page_number."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pages_data.append({
                "page_number": page_num + 1,
                "text": text
            })
            
    return pages_data

def extract_text_from_docx(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts text from DOCX. Approximates pages since docx lacks hard pagination."""
    doc = docx.Document(io.BytesIO(file_bytes))
    
    # We will treat the whole docx as "Page 1" or we can split by arbitrary blocks.
    # For a level 2 project, mapping docx text to page 1 is a common fallback as mentioned in the PRD.
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    if full_text.strip():
        return [{
            "page_number": 1,
            "text": full_text
        }]
    return []

def parse_document(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Routes to the correct parser based on file extension."""
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
