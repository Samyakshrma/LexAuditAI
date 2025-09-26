import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def test_azure_embedding():
    """
    Tests the Azure OpenAI embedding API call using dedicated embedding credentials.
    """
    print("--- Starting Azure Embedding Test ---")
    
    endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID")
    
    print(f"Embedding Endpoint: {endpoint}")
    print(f"Embedding API Key Loaded: {'Yes' if api_key else 'NO'}")
    print(f"Embedding Deployment ID: {deployment}")

    if not all([endpoint, api_key, deployment]):
        print("\n!!! ERROR: Missing one or more EMBEDDING environment variables.")
        return

    try:
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-02-01")
        test_sentence = "This is a test sentence."
        print(f"Attempting to create embedding for: '{test_sentence}'")
        response = client.embeddings.create(model=deployment, input=test_sentence)
        print(f"✅ SUCCESS! Embedding test passed.")
    except Exception as e:
        print(f"\n!!! ERROR: Embedding API call failed: {e}")
    finally:
        print("--- Embedding Test Complete ---\n")

def test_azure_chat():
    """
    Tests the Azure OpenAI chat completion API call using dedicated chat credentials.
    """
    print("--- Starting Azure Chat Test ---")
    
    endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_ID")
    
    print(f"Chat Endpoint: {endpoint}")
    print(f"Chat API Key Loaded: {'Yes' if api_key else 'NO'}")
    print(f"Chat Deployment ID: {deployment}")

    if not all([endpoint, api_key, deployment]):
        print("\n!!! ERROR: Missing one or more CHAT environment variables.")
        return

    try:
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-02-01")
        print("Attempting to create chat completion...")
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'Hello, World!'"}]
        )
        message = response.choices[0].message.content
        print(f"✅ SUCCESS! Chat test passed. Model responded: '{message}'")
    except Exception as e:
        print(f"\n!!! ERROR: Chat API call failed: {e}")
    finally:
        print("--- Chat Test Complete ---")


if __name__ == "__main__":
    print("Loading .env file...")
    if not load_dotenv():
        print("!!! WARNING: .env file not found.")
    else:
        print(".env file loaded.\n")
        
    test_azure_embedding()
    test_azure_chat()