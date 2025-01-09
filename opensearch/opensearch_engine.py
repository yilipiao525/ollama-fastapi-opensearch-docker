import os
import PyPDF2
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class OpensearchEngine:
    def __init__(
        self, 
        opensearch_url: str = None,
        index_name: str = "pdf-index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        embedding_dim: int = 384
    ):
        """
        Initialize the PDF Search Engine.

        Args:
            opensearch_url: URL of the OpenSearch instance 
                (defaults to env variable OPENSEARCH_URL or "http://localhost:9200")
            index_name: Name of the OpenSearch index
            model_name: Name of the sentence transformer model
            chunk_size: Size of text chunks for processing
            embedding_dim: Dimension of the embedding vectors
        """
        if not opensearch_url:
            # fallback to environment variable or default if not set
            opensearch_url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        
        self.opensearch_url = opensearch_url
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(model_name)

    def create_index(self) -> None:
        """Create the OpenSearch index with appropriate mappings."""
        index_body = {
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "pdf_id": {"type": "keyword"},
                    "chunk_text": {"type": "text"},
                    "chunk_embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim
                    }
                }
            }
        }

        response = requests.put(
            f"{self.opensearch_url}/{self.index_name}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(index_body)
        )

        # 400/404 means index might already exist
        if response.status_code not in (200, 201):
            print("Index creation response:", response.text)

        response.raise_for_status()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using a Sentence Transformer."""
        return self.model.encode(chunks)

    def index_pdf(self, pdf_path: str) -> None:
        """
        Process and index a PDF file into OpenSearch.
        """
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        chunks = self.chunk_text(pdf_text)
        
        # Generate embeddings
        chunk_embeddings = self.generate_embeddings(chunks)
        
        # Index each chunk
        for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            doc = {
                "pdf_id": pdf_path,
                "chunk_text": chunk_text,
                "chunk_embedding": embedding.tolist()
            }

            response = requests.post(
                f"{self.opensearch_url}/{self.index_name}/_doc",
                headers={"Content-Type": "application/json"},
                data=json.dumps(doc)
            )

            # If there's an error, raise it
            try:
                response.raise_for_status()
            except Exception as e:
                print(f"Error indexing chunk {i}: {e}")
                print("Response body:", response.text)

    def search_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[float, str]]:
        """
        Search for similar text chunks in the OpenSearch index using KNN search.
        
        Args:
            query_text (str): The query text to embed and search.
            top_k (int): Number of results to return.
            
        Returns:
            A list of (score, text_chunk) tuples.
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query_text])[0]

        # KNN query
        query_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "chunk_embedding": {
                        "vector": query_embedding.tolist(),
                        "k": top_k
                    }
                }
            }
        }

        response = requests.post(
            f"{self.opensearch_url}/{self.index_name}/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query_body)
        )
        response.raise_for_status()

        hits = response.json()["hits"]["hits"]
        return [
            (hit["_score"], hit["_source"]["chunk_text"]) 
            for hit in hits
        ]


if __name__ == "__main__":
    """
    Example usage:
    python pdf_search.py  # (assuming this file is named pdf_search.py)
    """
    # Initialize the search engine, reading OPENSEARCH_URL from env if not provided
    search_engine = OpensearchEngine()

    # Create the index (only once, or handle 400 if it already exists)
    search_engine.create_index()

    # Index a PDF
    search_engine.index_pdf("your-file.pdf")

    # Perform a similarity search
    results = search_engine.search_similar("What does the PDF say about data processing?")
    for score, text in results:
        print(f"Score: {score}\nText: {text}\n")

