



# LexAudit AI

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=)

LexAudit AI is an intelligent legal tech application designed to analyze legal documents for compliance with a comprehensive database of Indian laws. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide detailed, clause-by-clause analysis, identify potential risks, and generate plain-language summaries of complex legal issues.

## Features

  - **AI-Powered Clause Extraction:** Intelligently parses uploaded documents (PDF, DOCX, TXT) to identify and isolate distinct legal clauses for analysis.
  - **Retrieval-Augmented Generation (RAG):** For each clause, it retrieves relevant sections from a vectorized database of Indian laws to provide contextually accurate compliance checks.
  - **Multi-Model AI Engine:** Utilizes Azure OpenAI's powerful models, with `text-embedding-ada-002` for vectorization and `gpt-4o` for legal reasoning and analysis.
  - **Risk Assessment:** Categorizes each clause as 'Compliant', 'Risky', or 'Non-Compliant' with detailed justifications.
  - **Plain-Language Summaries:** Automatically generates easy-to-understand summaries for non-compliant or risky clauses.
  - **REST API:** Built with FastAPI, providing a simple endpoint for easy integration into other systems.

## Architecture Overview

LexAudit AI is built on a modern, modular architecture designed for accuracy and scalability.

1.  **FastAPI Backend:** Serves as the entry point for handling file uploads and API requests.
2.  **Document Processing:** Extracts raw text from uploaded documents and uses an LLM to intelligently segment the text into semantic clauses.
3.  **Azure OpenAI Service:**
      - **Embedding Client:** Connects to an Azure resource to convert both the law database and user-submitted clauses into vector embeddings using `text-embedding-ada-002`.
      - **Chat Client:** Connects to a separate Azure resource to use a powerful generative model (e.g., `gpt-4o`) for the clause extraction, compliance analysis, and summarization steps.
4.  **ChromaDB Vector Store:** A local, persistent vector database that stores the embeddings of the Indian law dataset, enabling fast and efficient similarity searches.
5.  **RAG Pipeline:** The core logic that orchestrates the workflow:
      - **Clause Extraction:** Parses the uploaded document text into clauses.
      - **Retrieve:** For each clause, queries ChromaDB to find the most relevant legal provisions.
      - **Augment & Generate:** Combines the original clause with the retrieved laws in a detailed prompt and sends it to the chat model for analysis and generation of the compliance report.

## Setup and Installation

### Prerequisites

  - Python 3.9+
  - An active Microsoft Azure subscription with access to the Azure OpenAI Service.
  - Two separate Azure OpenAI resources provisioned: one for an embedding model and one for a chat model.

### 1\. Clone the Repository

```bash
git clone https://github.com/Samyakshrma/LexAuditAI.git
cd LexAuditAI
```

### 2\. Create and Activate a Virtual Environment

**On Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Configure Environment Variables

Create a file named `.env` in the root of the project directory. Copy the contents of `.env.example` (if provided) or use the template below and fill it with your actual credentials from the Azure Portal.

```env
# .env file

# --- Credentials for the CHAT/COMPLETION Model ---
# The BASE endpoint from your Azure resource for GPT-4o
AZURE_OPENAI_CHAT_ENDPOINT=https://your-chat-resource-name.openai.azure.com/
# The API key for your chat resource
AZURE_OPENAI_CHAT_API_KEY=your_api_key_for_the_chat_resource
# The deployment name for your chat model (e.g., gpt-4o) in the chat resource
AZURE_OPENAI_CHAT_DEPLOYMENT_ID=gpt-4o

# --- Credentials for the EMBEDDING Model ---
# The BASE endpoint from your Azure resource for ada-002
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-embedding-resource-name.openai.azure.com/
# The API key for your embedding resource
AZURE_OPENAI_EMBEDDING_API_KEY=your_api_key_for_the_embedding_resource
# The deployment name for your embedding model in the embedding resource
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID=text-embedding-ada-002
```

## Running the Application

### 1\. Start the API Server

```bash
uvicorn main:app --reload
```

The server will be running at `http://127.0.0.1:8000`.

### 2\. First-Time Data Loading (Important\!)

On the very first run, the application will detect that the local vector database (`./chroma_db`) is empty. It will automatically begin a **long, one-time process** to read, vectorize, and store all the laws from the `Law_Dataset`. This can take several hours depending on the dataset size.

Monitor the terminal for progress messages like:

  - `⏳ Processed 10 lines from source file...`
  - `✅ Added batch 1...`

Once this process is complete, subsequent application startups will be almost instantaneous.

## API Usage

The application exposes one main endpoint for compliance checking. You can access the interactive API documentation (Swagger UI) by navigating to `http://127.0.0.1:8000/docs` in your browser.

### Endpoint: `POST /check-compliance`

  - **Description:** Upload a legal document to perform a full compliance analysis.
  - **Request:** `multipart/form-data` with a single file field.
      - `file`: The document to be analyzed (.pdf, .docx, .doc, .txt).
  - **Response:** A JSON object containing the results of the analysis.

**Example cURL Request:**

```bash
curl -X POST "http://127.0.0.1:8000/check-compliance" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/document.docx"
```

## Project Structure

```
.
├── app/                  # Main application source code
│   ├── document_utils.py # File reading and parsing utilities
│   ├── llm_utils.py      # Handles all interactions with Azure OpenAI
│   ├── rag_pipeline.py   # Core RAG workflow orchestration
│   └── vectordb_manager.py # Manages the ChromaDB vector store
├── Law_Dataset/          # Contains the source legal data
│   └── indian_laws_dataset.jsonl
├── .env                  # Environment variables (you must create this)
├── main.py               # FastAPI application entry point
└── requirements.txt      # Python dependencies
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.