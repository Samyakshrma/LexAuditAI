# app/vectordb_manager.py
import os
import chromadb
from typing import List, Dict, Any
import json


class VectorDBManager:
    """
    Manages the ChromaDB vector database for the LexAudit AI application.
    Handles client initialization, collection management, and similarity search.
    """
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "indian_laws"):
        """
        Initializes the VectorDBManager with a specified database path and collection name.

        Args:
            db_path (str): The local directory where ChromaDB data will be stored.
            collection_name (str): The name of the collection to store legal documents.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    async def initialize_db(self):
        """
        Initializes the persistent ChromaDB client and gets or creates the collection.
        This method should be called on application startup.
        """
        try:
            # Create the ChromaDB client with a persistent directory
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create the collection. This is idempotent.
            # If the collection exists, it will be loaded. Otherwise, it will be created.
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            
            print(f"ChromaDB client initialized and collection '{self.collection_name}' ready.")
            
            # Check if the collection is empty and load data if it is.
            if self.collection.count() == 0:
                print("Collection is empty. Loading Indian laws data...")
                # In a real-world scenario, you would load your data here.
                # For this example, we will add a few placeholder documents.
                await self.load_indian_laws()
                print("Indian laws loaded successfully.")
            else:
                print(f"Collection '{self.collection_name}' already contains {self.collection.count()} documents.")

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None

    async def load_indian_laws(self):
        """
        (Placeholder) Loads the Indian laws data into the ChromaDB collection.
        The user has this data in vector format, but for demonstration, we will
        add a few sample documents.
        """
        # TODO: The user should replace this function with their actual data ingestion logic.
        # This is where you would load your vector data and add it to the collection.
        # Example:
        # with open('path/to/my_laws.json', 'r') as f:
        #     laws_data = json.load(f)
        # self.collection.add(
        #     documents=[item['text'] for item in laws_data],
        #     metadatas=[{'source': item['source']} for item in laws_data],
        #     ids=[item['id'] for item in laws_data]
        # )

        sample_laws = [
            {"id": "law_1", "text": "The Indian Contract Act, 1872, governs all contractual agreements in India.", "source": "Indian Contract Act"},
            {"id": "law_2", "text": "The Information Technology Act, 2000, regulates electronic commerce and data privacy.", "source": "Information Technology Act"},
            {"id": "law_3", "text": "The Companies Act, 2013, governs the incorporation, responsibilities, and winding up of companies.", "source": "Companies Act"},
            {"id": "law_4", "text": "Privacy Policy: A company must clearly state how user data is collected, used, and protected.", "source": "IT Act, 2000 & Data Protection Rules"}
        ]

        self.collection.add(
            documents=[law["text"] for law in sample_laws],
            metadatas=[{"source": law["source"]} for law in sample_laws],
            ids=[law["id"] for law in sample_laws]
        )
        print(f"Added {self.collection.count()} sample documents to the collection.")


    async def query_relevant_laws(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a similarity search to find the most relevant law provisions.

        Args:
            query (str): The text query (a document chunk from the user's file).
            n_results (int): The number of relevant results to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the found documents and their metadata.
        """
        if not self.collection:
            print("ChromaDB collection not available.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['metadatas', 'documents']
            )

            # Format the results into a more usable list of dictionaries
            formatted_results = []
            if results and results['documents']:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    formatted_results.append({
                        "document": doc,
                        "metadata": meta
                    })
            
            return formatted_results

        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []
