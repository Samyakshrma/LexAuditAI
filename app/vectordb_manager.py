import os
import json
import chromadb
import tiktoken # <-- ADDED: For accurately counting tokens
from typing import List, Dict, Any
from app.llm_utils import LLMUtils

class VectorDBManager:
    def __init__(self, llm_helper: LLMUtils, db_path: str = "./chroma_db", collection_name: str = "indian_laws_ada"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.llm_helper = llm_helper
        self.client = None
        self.collection = None

    def initialize_db(self):
        """Initializes the persistent ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            collection_metadata = {"hnsw:space": "cosine", "embedding_dim": 1536}
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=collection_metadata
            )
            print(f"ChromaDB client initialized and collection '{self.collection_name}' ready.")
            if self.collection.count() == 0:
                print("Collection is empty. Loading Indian laws data...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, '..', 'Law_Dataset', 'indian_laws_dataset.jsonl')
                self.load_indian_laws(data_path)
                print("Indian laws loaded successfully.")
            else:
                print(f"Collection '{self.collection_name}' already contains {self.collection.count()} documents.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")

    # <-- EDITED: This function now includes token counting and chunking logic -->
    def load_indian_laws(self, filepath: str):
        """Loads and embeds Indian laws, chunking long documents to fit token limits."""
        try:
            documents, metadatas, ids, embeddings = [], [], [], []
            
            # Initialize the tokenizer to count tokens accurately
            encoding = tiktoken.get_encoding("cl100k_base")
            max_tokens = 8190  # Safe limit below the model's max (8192)

            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        law = json.loads(line.strip())
                        law_text = law.get("law")

                        if law_text and isinstance(law_text, str):
                            num_tokens = len(encoding.encode(law_text))

                            # Logic to handle both short and long documents
                            if num_tokens <= max_tokens:
                                # If the text is short enough, process it directly
                                embedding = self.llm_helper.generate_embedding(law_text)
                                if embedding:
                                    law_id = f"{law.get('act_title', 'unknown')}_{law.get('section', 'unknown')}"
                                    documents.append(law_text)
                                    metadatas.append({"act_title": law.get('act_title', 'N/A'), "section": law.get('section', 'N/A')})
                                    ids.append(law_id)
                                    embeddings.append(embedding)
                            else:
                                # If the text is too long, split it into chunks
                                print(f"⚠️  Law text on line {i+1} is too long ({num_tokens} tokens). Chunking it...")
                                chunk_size = 1000  # Size of each chunk in tokens
                                chunk_overlap = 100 # Overlap between chunks in tokens
                                
                                tokens = encoding.encode(law_text)
                                for chunk_num, start_pos in enumerate(range(0, len(tokens), chunk_size - chunk_overlap)):
                                    end_pos = start_pos + chunk_size
                                    chunk_tokens = tokens[start_pos:end_pos]
                                    chunk_text = encoding.decode(chunk_tokens)
                                    
                                    chunk_embedding = self.llm_helper.generate_embedding(chunk_text)
                                    if chunk_embedding:
                                        base_id = f"{law.get('act_title', 'unknown')}_{law.get('section', 'unknown')}"
                                        chunk_id = f"{base_id}_chunk_{chunk_num}"
                                        
                                        documents.append(chunk_text)
                                        metadatas.append({"act_title": law.get('act_title', 'N/A'), "section": f"{law.get('section', 'N/A')} (Part {chunk_num + 1})"})
                                        ids.append(chunk_id)
                                        embeddings.append(chunk_embedding)
                        else:
                            print(f"⚠️  Skipping invalid or empty law text on line {i+1}")
                    except json.JSONDecodeError:
                        print(f"⚠️  Skipping malformed JSON on line {i+1}")
                    
                    if (i + 1) % 50 == 0:
                        print(f"⏳ Processed {i + 1} lines from source file...")

            # Batch add all collected documents and chunks to the database
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_index = i + batch_size
                self.collection.add(
                    embeddings=embeddings[i:end_index],
                    documents=documents[i:end_index],
                    metadatas=metadatas[i:end_index],
                    ids=ids[i:end_index]
                )
                print(f"✅ Added batch {i//batch_size + 1}...")

            print(f"✅ Added a total of {len(documents)} document chunks to the collection.")
        except Exception as e:
            print(f"❌ Error loading laws from {filepath}: {e}")

    def query_relevant_laws(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Performs a similarity search using an embedding for the query text."""
        if not self.collection:
            print("ChromaDB collection not available.")
            return []
        try:
            query_embedding = self.llm_helper.generate_embedding(query_text)
            if not query_embedding:
                print("Could not generate embedding for the query.")
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents']
            )
            formatted_results = []
            if results and results['documents']:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    formatted_results.append({"document": doc, "metadata": meta})
            return formatted_results
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []