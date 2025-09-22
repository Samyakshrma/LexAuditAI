# app/llm_utils.py
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class LLMUtils:
    """
    Utility class for making calls to an Azure OpenAI LLM.
    """
    def __init__(self):
        """
        Initializes the Azure OpenAI client.
        Requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_API_VERSION
        to be set in your environment variables or .env file.
        """
        try:
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
            print("Azure OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            print("Please ensure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_API_VERSION are set.")
            self.client = None

    def call_llm(self, deployment_id: str, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """
        Makes a call to the Azure OpenAI Chat Completion API.
        ... (rest of your original docstring)
        """
        if not self.client:
            print("LLM client not initialized. Cannot make API call.")
            return None

        try:
            response = self.client.chat.completions.create(
                model=deployment_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure LLM ({deployment_id}): {e}")
            return None

    def parse_llm_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Safely parses an LLM string response into a JSON dictionary.

        Args:
            response_text (str): The string response from the LLM.

        Returns:
            Dict[str, Any]: The parsed JSON dictionary, or an empty dict if parsing fails.
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}. Malformed JSON received from LLM: {response_text}")
            return {}