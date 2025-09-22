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
                
                # --- THIS IS THE CRITICAL FIX ---
                # Build a robust path that works from any directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, '..', 'Law_Dataset', 'indian_laws_dataset.jsonl')
                await self.load_indian_laws(data_path)
                # ----------------------------------
                
                print("Indian laws loaded successfully.")
            else:
                print(f"Collection '{self.collection_name}' already contains {self.collection.count()} documents.")

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None

    async def load_indian_laws(self, filepath: str):
        """
        Loads Indian laws from a JSONL file into ChromaDB, using batching.
        Each JSONL row should have: act_title, section, law
        """
        try:
            documents, metadatas, ids = [], [], []
            
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    law = json.loads(line.strip())
                    
                    if law.get("law"):
                        law_id = f"{law['act_title'].replace(' ', '_')}_section{law['section']}"
                        
                        documents.append(law["law"])
                        metadatas.append({
                            "act_title": law["act_title"],
                            "section": law["section"]
                        })
                        ids.append(law_id)
            
            # --- THIS IS THE CRITICAL FIX ---
            # Split the data into batches and add to the collection
            batch_size = 5000  # Set a safe batch size below the max limit
            for i in range(0, len(documents), batch_size):
                end_index = i + batch_size
                self.collection.add(
                    documents=documents[i:end_index],
                    metadatas=metadatas[i:end_index],
                    ids=ids[i:end_index]
                )
                print(f"✅ Added batch {i//batch_size + 1} of {len(documents)//batch_size + 1}...")

            print(f"✅ Added {len(documents)} laws from {filepath} into collection '{self.collection_name}'.")

        except Exception as e:
            print(f"❌ Error loading laws from {filepath}: {e}")


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