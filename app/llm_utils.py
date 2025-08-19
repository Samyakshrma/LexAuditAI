# app/llm_utils.py
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

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

    def call_llm(self, deployment_id: str, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 1000):
        """
        Makes a call to the Azure OpenAI Chat Completion API.

        Args:
            deployment_id (str): The name of the deployment in Azure OpenAI Studio.
            system_prompt (str): The system prompt to set the context/role of the AI.
            user_message (str): The user's input/query.
            temperature (float): Controls randomness. Lower means more deterministic.
            max_tokens (int): The maximum number of tokens to generate in the completion.

        Returns:
            str: The generated text response from the LLM, or None if an error occurs.
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
