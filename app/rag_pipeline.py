# app/rag_pipeline.py
import asyncio
from typing import List, Dict, Any

from app.llm_utils import LLMUtils
from app.vectordb_manager import VectorDBManager

# A flag to control the LLM calls during development/testing
# You can set this to False if you want to test the pipeline without incurring LLM costs
# In production, this should always be True.
ENABLE_LLM_CALLS = True
# The deployment ID for the LLM that will handle compliance checks
COMPLIANCE_LLM_DEPLOYMENT_ID = "gpt-4.1"
# The deployment ID for the LLM that will handle summarization
SUMMARIZER_LLM_DEPLOYMENT_ID = "gpt-35-turbo"


async def run_rag_pipeline(doc_chunks: List[str], llm_helper: LLMUtils, db_manager: VectorDBManager, analysis_id: str) -> List[Dict[str, Any]]:
    """
    Orchestrates the entire RAG pipeline for a given document.

    For each document chunk, it queries the vector database for relevant laws,
    crafts a prompt with the retrieved context, and calls the LLM for analysis and summary.

    Args:
        doc_chunks (List[str]): The list of text chunks from the user's document.
        llm_helper (LLMUtils): An instance of the LLM utility class.
        db_manager (VectorDBManager): An instance of the vector database manager.
        analysis_id (str): The unique ID for this analysis run.

    Returns:
        List[Dict[str, Any]]: A list of analysis results for each document chunk.
    """
    analysis_results = []
    
    # We will process each chunk in a separate asynchronous task to improve performance
    # This is where FastAPI's async capabilities shine.
    tasks = []
    for chunk in doc_chunks:
        tasks.append(_process_chunk(chunk, llm_helper, db_manager))

    # Run all tasks concurrently and wait for them to complete
    results = await asyncio.gather(*tasks)
    
    # Filter out any None values from failed tasks
    analysis_results = [result for result in results if result is not None]

    return analysis_results


async def _process_chunk(chunk: str, llm_helper: LLMUtils, db_manager: VectorDBManager) -> Dict[str, Any] | None:
    """
    Processes a single document chunk: queries the DB and calls the LLM for analysis.
    """
    try:
        # Step 1: Retrieve relevant laws from the vector database
        relevant_laws = await db_manager.query_relevant_laws(query=chunk, n_results=3)

        # Step 2: Craft a prompt for the LLM with the retrieved context
        context_string = "\n".join([f"Source: {law['metadata']['act_title']} - Section {law['metadata']['section']}\nText: {law['document']}" for law in relevant_laws])
        
        compliance_system_prompt = (
            "You are LexAudit AI, an expert legal AI for Indian law. "
            "Your task is to analyze a clause from a legal document and check it for compliance "
            "with the provided Indian laws. Your analysis must be based *only* on the context given."
            "Cite the specific legal provision for any compliance issues or confirmations."
            "Output your response in a clear, structured JSON format."
        )

        compliance_user_message = (
            f"Here is a legal clause from a document:\n\n---\n{chunk}\n---\n\n"
            f"Here are the relevant legal provisions from the Indian laws dataset:\n\n---\n{context_string}\n---\n\n"
            "Based on this, determine if the clause is non-compliant, risky, or compliant. "
            "Provide a detailed analysis and cite the law provisions that support your conclusion."
        )

        analysis_response = ""
        if ENABLE_LLM_CALLS:
            analysis_response = llm_helper.call_llm(
                deployment_id=COMPLIANCE_LLM_DEPLOYMENT_ID,
                system_prompt=compliance_system_prompt,
                user_message=compliance_user_message,
                max_tokens=2000
            )
        else:
            analysis_response = "Placeholder analysis from RAG pipeline."

        # Step 3: Call LLM for a plain-language summary (if compliance issues are found)
        summary_response = ""
        if "non-compliant" in analysis_response.lower() or "risky" in analysis_response.lower():
            if ENABLE_LLM_CALLS:
                summarizer_system_prompt = "You are a legal summarizer. Simplify the following legal analysis into plain English for a non-lawyer."
                summary_response = llm_helper.call_llm(
                    deployment_id=SUMMARIZER_LLM_DEPLOYMENT_ID,
                    system_prompt=summarizer_system_prompt,
                    user_message=analysis_response,
                    max_tokens=500
                )
            else:
                summary_response = "Placeholder summary from RAG pipeline."

        # Step 4: Combine all information into a structured result
        return {
            "original_chunk": chunk,
            "compliance_analysis": analysis_response,
            "plain_language_summary": summary_response,
            "relevant_laws": relevant_laws
        }

    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None
