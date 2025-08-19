# app/document_utils.py
import docx
import PyPDF2
from typing import List

async def process_document_and_chunk(file_path: str, file_extension: str) -> List[str]:
    """
    Reads content from a given file and splits it into chunks.

    Args:
        file_path (str): The path to the temporary file.
        file_extension (str): The extension of the file (e.g., '.pdf', '.docx', '.txt').

    Returns:
        List[str]: A list of text chunks.
    """
    text = ""
    if file_extension == '.pdf':
        text = await _read_pdf(file_path)
    elif file_extension == '.docx' or file_extension == '.doc':
        text = await _read_docx(file_path)
    elif file_extension == '.txt':
        text = await _read_txt(file_path)
    
    # Check if we successfully extracted text
    if not text:
        return []

    # Simple chunking strategy: split by a fixed number of characters
    # This is a good starting point. We can make this more sophisticated later.
    chunk_size = 1500
    chunk_overlap = 150
    chunks = []
    
    # We'll use a simple slicing method for chunking.
    # More advanced methods often use libraries like LangChain or LlamaIndex.
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

async def _read_pdf(file_path: str) -> str:
    """
    Reads text content from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

async def _read_docx(file_path: str) -> str:
    """
    Reads text content from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
    return text

async def _read_txt(file_path: str) -> str:
    """
    Reads text content from a TXT file.

    Args:
        file_path (str): The path to the TXT file.

    Returns:
        str: The extracted text.
    """
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading TXT file: {e}")
    return text
