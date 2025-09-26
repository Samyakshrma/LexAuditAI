# main.py
import shutil
import os
import tempfile
import uuid
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.llm_utils import LLMUtils
from app.document_utils import process_document # Corrected import
from app.rag_pipeline import run_rag_pipeline
from app.vectordb_manager import VectorDBManager

# Initialize FastAPI app
app = FastAPI(title="LexAudit AI API")

# Initialize utilities as singletons
llm_helper = LLMUtils()
db_manager = VectorDBManager(llm_helper=llm_helper)

@app.on_event("startup")
def startup_event():
    """
    This function runs once when the application starts up.
    It initializes the vector database.
    """
    print("Application startup event: Initializing LexAudit AI.")
    db_manager.initialize_db()
    print("LexAudit AI is ready to go!")

@app.post("/check-compliance")
async def check_compliance(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint to upload a legal document, check it for compliance, and
    generate a legal summary.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file was provided.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".pdf", ".docx", ".doc", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Step 1: Extract the full, raw text from the document.
        doc_text = await asyncio.to_thread(process_document, temp_file_path, file_extension)

        if not doc_text:
             raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        # <-- CHANGE: The clause extraction logic is now REMOVED from main.py. -->
        # The rag_pipeline will handle this internally.

        analysis_id = str(uuid.uuid4())

        # Step 2: Pass the ENTIRE document text to the RAG pipeline.
        # <-- CHANGE: The function call is updated to pass the correct argument. -->
        analysis_results = await run_rag_pipeline(
            document_text=doc_text, # Pass the full text string
            llm_helper=llm_helper,
            db_manager=db_manager,
            analysis_id=analysis_id
        )

        return JSONResponse(content={
            "analysis_id": analysis_id,
            "status": "completed",
            "results": analysis_results
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during file processing.")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    return {"message": f"Results for analysis ID {analysis_id} will be available here."}