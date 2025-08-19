# main.py
import shutil
import os
import tempfile
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.llm_utils import LLMUtils
from app.document_utils import process_document_and_chunk
from app.rag_pipeline import run_rag_pipeline
from app.vectordb_manager import VectorDBManager

# Initialize FastAPI app
app = FastAPI(title="LexAudit AI API")

# Initialize utilities as singletons
llm_helper = LLMUtils()
db_manager = VectorDBManager()

@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the application starts up.
    It can be used to load the vector database, connect to external services, etc.
    """
    print("Application startup event: Initializing LexAudit AI.")
    # TODO: In a real-world scenario, you would load your Indian laws dataset here
    # For now, we'll use a placeholder.
    # The VectorDBManager will create a persistent client and load or create a collection
    await db_manager.initialize_db()
    print("LexAudit AI is ready to go!")

@app.post("/check-compliance")
async def check_compliance(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint to upload a legal document, check it for compliance, and
    generate a legal summary.

    Args:
        file (UploadFile): The legal document to be processed.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis ID and status.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file was provided.")

    # Get the file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".pdf", ".docx", ".doc", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, Word, or text document.")
    
    # Save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Process the document to get chunks
        doc_chunks = await process_document_and_chunk(temp_file_path, file_extension)

        # Generate a unique analysis ID
        analysis_id = str(uuid.uuid4())

        # TODO: This will be the main logic for the RAG pipeline.
        # It will be implemented in the next step, but here's the placeholder call.
        analysis_results = await run_rag_pipeline(
            doc_chunks=doc_chunks,
            llm_helper=llm_helper,
            db_manager=db_manager,
            analysis_id=analysis_id
        )

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Return a response with the analysis ID
        return JSONResponse(content={
            "analysis_id": analysis_id,
            "status": "processing",
            "results": analysis_results
        })

    except Exception as e:
        # Clean up the temp file if an error occurs
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during file processing.")


# Endpoint for future features (e.g., getting analysis results, downloading reports)
@app.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    # This will be implemented later to retrieve the stored results
    return {"message": f"Results for analysis ID {analysis_id} will be available here."}
