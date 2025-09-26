import asyncio
from typing import List, Dict, Any

from app.llm_utils import LLMUtils
from app.vectordb_manager import VectorDBManager
import json

# --- CONFIGURATION ---
ENABLE_LLM_CALLS = True
# <-- CHANGE: Removed hardcoded deployment IDs. This is now handled by LLMUtils -->

async def run_rag_pipeline(document_text: str, llm_helper: LLMUtils, db_manager: VectorDBManager, analysis_id: str) -> List[Dict[str, Any]]:
    """
    Orchestrates an intelligent RAG pipeline.
    1. Extracts distinct clauses from the full document text using an LLM.
    2. Processes each clause concurrently to check for compliance.
    """
    print("Step 1: Extracting clauses from the document using an LLM...")
    
    clause_extraction_prompt = (
        "You are a legal document parsing expert. Your task is to analyze the following document text "
        "and extract all distinct clauses, sub-clauses, and numbered or lettered points. "
        "Ignore headings, titles, and definitions unless they contain a direct obligation. "
        "Return the output as a single JSON object with one key: 'clauses', which contains a list of strings. "
        "Each string in the list should be the full, verbatim text of a single clause."
    )

    # Run the synchronous LLM call in a separate thread
    response_str = await asyncio.to_thread(
        llm_helper.call_llm,
        # <-- CHANGE: Removed the 'deployment_id' argument -->
        system_prompt=clause_extraction_prompt,
        user_message=document_text,
        max_tokens=4096
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
    
    tasks = []
    for clause in extracted_clauses:
        tasks.append(_process_clause(clause, llm_helper, db_manager))

    results = await asyncio.gather(*tasks)
    analysis_results = [result for result in results if result is not None]

    return analysis_results


async def _process_clause(clause: str, llm_helper: LLMUtils, db_manager: VectorDBManager) -> Dict[str, Any] | None:
    """
    Processes a single legal clause: queries the DB and calls an LLM for analysis.
    """
    try:
        relevant_laws = await asyncio.to_thread(
            db_manager.query_relevant_laws, query_text=clause, n_results=3
        )

        context_string = "\n".join([f"Source: {law['metadata']['act_title']} - Section {law['metadata']['section']}\nText: {law['document']}" for law in relevant_laws])
        
        compliance_system_prompt = (
            "You are LexAudit AI, an expert legal AI for Indian law. Analyze a clause from a legal document for compliance "
            "with the provided Indian laws. Your analysis must be based *only* on the context given. Cite the specific legal provision for any issues. "
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

        analysis_response_string = await asyncio.to_thread(
            llm_helper.call_llm,
            # <-- CHANGE: Removed the 'deployment_id' argument -->
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
                # <-- CHANGE: Removed the 'deployment_id' argument -->
                system_prompt=summarizer_system_prompt,
                user_message=analysis_response.get("analysis", ""),
                max_tokens=500
            ) if ENABLE_LLM_CALLS else "Placeholder summary."

        return {
            "original_clause": clause,
            "compliance_analysis": analysis_response,
            "plain_language_summary": summary_response,
            "retrieved_laws": relevant_laws
        }

    except Exception as e:
        print(f"Error processing clause: {e}")
        return None