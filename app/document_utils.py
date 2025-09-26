# app/document_utils.py
import docx
import PyPDF2
from typing import List

def process_document(file_path: str, file_extension: str) -> str:
    """
    Reads content from a given file and returns it as a single string.
    This is a synchronous function.

    Args:
        file_path (str): The path to the file.
        file_extension (str): The extension of the file.

    Returns:
        str: The full text extracted from the document.
    """
    text = ""
    if file_extension == '.pdf':
        text = _read_pdf(file_path)
    elif file_extension == '.docx' or file_extension == '.doc':
        text = _read_docx(file_path)
    elif file_extension == '.txt':
        text = _read_txt(file_path)
    
    return text

def _read_pdf(file_path: str) -> str:
    """Reads text content from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

def _read_docx(file_path: str) -> str:
    """Reads text content from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
    return text

def _read_txt(file_path: str) -> str:
    """Reads text content from a TXT file."""
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading TXT file: {e}")
    return text