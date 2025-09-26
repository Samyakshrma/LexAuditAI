# app/rag_pipeline.py
import asyncio
from typing import List, Dict, Any

from app.llm_utils import LLMUtils
from app.vectordb_manager import VectorDBManager
import json

# --- CONFIGURATION (Consider moving to a config.py file) ---
ENABLE_LLM_CALLS = True
# Use a model with a large context window for parsing the whole document
CLAUSE_EXTRACTION_LLM = "gpt-4o" 
COMPLIANCE_LLM = "gpt-4o" # Or another powerful model like "gpt-4-turbo"
SUMMARIZER_LLM = "gpt-35-turbo"

# <-- CHANGE: The function now accepts the full document text, not pre-made chunks -->
async def run_rag_pipeline(document_text: str, llm_helper: LLMUtils, db_manager: VectorDBManager, analysis_id: str) -> List[Dict[str, Any]]:
    """
    Orchestrates an intelligent RAG pipeline.
    1. Extracts distinct clauses from the full document text using an LLM.
    2. Processes each clause concurrently to check for compliance.
    """
    # --- STEP 1: INTELLIGENTLY EXTRACT CLAUSES FROM THE DOCUMENT ---
    print("Step 1: Extracting clauses from the document using an LLM...")
    
    # <-- NEW: This is the critical clause extraction step -->
    clause_extraction_prompt = (
        "You are a legal document parsing expert. Your task is to analyze the following document text "
        "and extract all distinct clauses, sub-clauses, and numbered or lettered points. "
        "Ignore headings, titles, and definitions unless they contain a direct obligation. "
        "Return the output as a single JSON object with one key: 'clauses', which contains a list of strings. "
        "Each string in the list should be the full, verbatim text of a single clause."
    )

    # Run the synchronous LLM call in a separate thread to avoid blocking
    response_str = await asyncio.to_thread(
        llm_helper.call_llm,
        deployment_id=CLAUSE_EXTRACTION_LLM,
        system_prompt=clause_extraction_prompt,
        user_message=document_text,
        max_tokens=4096 # Allow for a large response
    )

    if not response_str:
        print("Error: Clause extraction failed, LLM returned no response.")
        return []

    parsed_response = llm_helper.parse_llm_json_response(response_str)
    extracted_clauses = parsed_response.get("clauses", [])

    if not extracted_clauses or not isinstance(extracted_clauses, list):
        print("Error: Could not parse clauses from the document.")
        return []
    
    print(f"Successfully extracted {len(extracted_clauses)} clauses. Now processing each one.")
    
    # --- STEP 2: PROCESS EACH EXTRACTED CLAUSE FOR COMPLIANCE ---
    tasks = []
    # <-- CHANGE: We now loop over the intelligently extracted clauses -->
    for clause in extracted_clauses:
        tasks.append(_process_clause(clause, llm_helper, db_manager))

    results = await asyncio.gather(*tasks)
    analysis_results = [result for result in results if result is not None]

    return analysis_results


# <-- CHANGE: Renamed from _process_chunk to _process_clause for clarity -->
async def _process_clause(clause: str, llm_helper: LLMUtils, db_manager: VectorDBManager) -> Dict[str, Any] | None:
    """
    Processes a single legal clause: queries the DB and calls an LLM for analysis.
    """
    try:
        # Run synchronous DB query in a separate thread
        relevant_laws = await asyncio.to_thread(
            db_manager.query_relevant_laws, query_text=clause, n_results=3
        )

        context_string = "\n".join([f"Source: {law['metadata']['act_title']} - Section {law['metadata']['section']}\nText: {law['document']}" for law in relevant_laws])
        
        compliance_system_prompt = (
            "You are LexAudit AI, an expert legal AI for Indian law. "
            "Your task is to analyze a clause from a legal document and check it for compliance "
            "with the provided Indian laws. Your analysis must be based *only* on the context given. "
            "Cite the specific legal provision for any compliance issues. "
            "Output your response in a clear, structured JSON format with three keys: "
            "'status' (string: 'Compliant', 'Non-Compliant', or 'Risky'), "
            "'analysis' (string: your detailed explanation), and "
            "'citation' (string: the specific Act and Section number, e.g., 'The Indian Contract Act, 1872 - Section 10')."
        )
        compliance_user_message = (
            f"Legal Clause to Analyze:\n---\n{clause}\n---\n\n"
            f"Relevant Indian Law Provisions:\n---\n{context_string}\n---\n\n"
            "Perform the compliance analysis based on the provided laws."
        )

        # Run synchronous LLM call in a separate thread
        analysis_response_string = await asyncio.to_thread(
            llm_helper.call_llm,
            deployment_id=COMPLIANCE_LLM,
            system_prompt=compliance_system_prompt,
            user_message=compliance_user_message,
            max_tokens=2000
        ) if ENABLE_LLM_CALLS else '{"status": "Compliant", "analysis": "Placeholder analysis.", "citation": "N/A"}'

        analysis_response = llm_helper.parse_llm_json_response(analysis_response_string)
        
        summary_response = ""
        if analysis_response.get("status", "").lower() in ["non-compliant", "risky"]:
            summarizer_system_prompt = "You are a legal summarizer. Simplify the following legal analysis into plain English for a non-lawyer."
            summary_response = await asyncio.to_thread(
                llm_helper.call_llm,
                deployment_id=SUMMARIZER_LLM,
                system_prompt=summarizer_system_prompt,
                user_message=analysis_response.get("analysis", ""),
                max_tokens=500
            ) if ENABLE_LLM_CALLS else "Placeholder summary."

        return {
            "original_clause": clause, # Changed from original_chunk
            "compliance_analysis": analysis_response,
            "plain_language_summary": summary_response,
            "retrieved_laws": relevant_laws # Renamed for clarity
        }

    except Exception as e:
        print(f"Error processing clause: {e}")
        return None