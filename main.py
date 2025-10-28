import shutil
import os
import tempfile
import uuid
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # <-- THIS LINE IS NOW ADDED
from typing import Dict, Any

from app.llm_utils import LLMUtils
from app.document_utils import process_document # Corrected import
from app.rag_pipeline import run_rag_pipeline
from app.vectordb_manager import VectorDBManager

# Initialize FastAPI app
app = FastAPI(title="LexAudit AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing)
    # For production, you'd want to restrict this to your frontend's domain:
    # allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Allow GET, POST, and OPTIONS
    allow_headers=["*"],  # Allows all headers
)

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
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Step 1: Extract the full, raw text from the document.
        # Run the blocking I/O operation in a separate thread
        doc_text = await asyncio.to_thread(process_document, temp_file_path, file_extension)

        if not doc_text:
             raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        analysis_id = str(uuid.uuid4())

        # Step 2: Pass the ENTIRE document text to the RAG pipeline.
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
        # Log the exception for debugging
        print(f"An unexpected error occurred: {e}")
        # Return a generic 500 error to the client
        raise HTTPException(status_code=500, detail="An internal server error occurred during file processing.")
    
    finally:
        # Ensure the temporary file is always deleted
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Close the uploaded file stream
        if file:
            await file.close()


@app.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """
    Placeholder endpoint to retrieve results by ID.
    In this application, the results are returned directly by /check-compliance.
    """
    # In a real-world async scenario, you might store results in a DB
    # and use this endpoint to poll for them.
    return {"message": f"This endpoint is a placeholder. Results for analysis ID {analysis_id} were returned directly by the POST request."}

