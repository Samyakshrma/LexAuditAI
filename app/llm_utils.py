import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
load_dotenv()

class LLMUtils:
    """
    Utility class for making calls to Azure OpenAI services.
    Manages separate clients for chat completions and embeddings.
    """
    def __init__(self):
        try:
            # --- Initialize Chat Client ---
            self.chat_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
                api_version="2024-02-01"
            )
            self.chat_deployment_id = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_ID")
            
            # --- Initialize Embedding Client ---
            self.embedding_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
                api_version="2024-02-01"
            )
            self.embedding_deployment_id = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID")

            print("Azure OpenAI clients for Chat and Embeddings initialized successfully.")

        except Exception as e:
            print(f"Error initializing Azure OpenAI clients: {e}")
            self.chat_client = None
            self.embedding_client = None

    def call_llm(self, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Makes a call to the Azure OpenAI Chat Completion API using the dedicated chat client."""
        if not self.chat_client or not self.chat_deployment_id:
            print("Chat client or deployment ID not initialized. Cannot make API call.")
            return None
        try:
            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure LLM ({self.chat_deployment_id}): {e}")
            return None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generates a vector embedding for a given text using the dedicated embedding client."""
        if not self.embedding_client or not self.embedding_deployment_id:
            print("Embedding client or deployment ID not initialized.")
            return None
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment_id,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def parse_llm_json_response(self, response_text: str) -> Dict[str, Any]:
        """Safely parses an LLM string response into a JSON dictionary."""
        try:
            # Handle cases where the LLM might wrap the JSON in markdown code fences
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3]
            return json.loads(response_text)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"JSON parsing error: {e}. Malformed response from LLM: {response_text}")
            return {"error": "Failed to parse JSON response from LLM.", "raw_response": response_text}