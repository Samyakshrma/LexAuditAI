import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def test_azure_embedding():
    """
    A self-contained script to test the Azure OpenAI embedding API call
    using dedicated embedding credentials.
    """
    print("--- Starting Azure Embedding Test ---")
    
    # 1. Load environment variables
    print("Loading .env file...")
    if not load_dotenv():
        print("!!! WARNING: .env file not found. Make sure it's in the root directory.")
        return
    
    # 2. Read and print the specific environment variables for the EMBEDDING service
    # <-- CHANGE: Updated to use the new, specific variable names -->
    endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID")
    
    print(f"Embedding Endpoint: {endpoint}")
    print(f"Embedding API Key Loaded: {'Yes' if api_key else 'NO - THIS IS A PROBLEM'}")
    print(f"Embedding Deployment ID: {embedding_deployment}")

    if not all([endpoint, api_key, embedding_deployment]):
        print("\n!!! ERROR: One or more embedding environment variables are missing. Please check your .env file.")
        return

    # 3. Initialize the client
    try:
        print("\nInitializing AzureOpenAI client for embeddings...")
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
        print("Client initialized successfully.")
    except Exception as e:
        print(f"\n!!! ERROR: Failed to initialize client: {e}")
        return

    # 4. Attempt to create an embedding
    try:
        test_sentence = "This is a test sentence to check the embedding service."
        print(f"\nAttempting to create embedding for: '{test_sentence}'")
        
        response = client.embeddings.create(
            model=embedding_deployment,
            input=test_sentence
        )
        
        embedding_vector = response.data[0].embedding
        print(f"\n✅ SUCCESS! Embedding created successfully.")
        print(f"Vector length: {len(embedding_vector)}")
        print(f"First 5 dimensions: {embedding_vector[:5]}")

    except Exception as e:
        print(f"\n!!! ERROR: The API call failed: {e}")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_azure_embedding()