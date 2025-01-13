from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import os
from opensearch_engine import (
    connect_to_existing_vector_db,
    create_retriever,
    create_chain,
    process_and_create_vector_db
)

# Import necessary components for RAG workflow
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define Pydantic models for request validation
class PromptRequest(BaseModel):
    prompt: str

# Create FastAPI app instance
app = FastAPI(
    title="FastAPI with Ollama and OpenSearch",
    description="API for text generation and PDF search",
    version="1.0.0"
)

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
LLM_SERVER_URL = f"{OLLAMA_URL}/api/generate"
MODEL_NAME = "gemma2:2b"   
EMBEDDING_MODEL="nomic-embed-text" 
system_prompt = "You are a fast food ordering drive thru AI assistant. Your name is Fasta."

@app.post("/generate-answer")
def generate_answer(request: PromptRequest):
    try:
        # First check if Ollama server is running
        try:
            version_response = requests.get(
                f"{OLLAMA_URL}/api/version",
                timeout=5
            )
            version_response.raise_for_status()
            
            # Then check if our model is available
            list_response = requests.get(
                f"{OLLAMA_URL}/api/tags",
                timeout=5
            )
            list_response.raise_for_status()
            
            models = list_response.json().get("models", [])
            if not any(model["name"] == MODEL_NAME for model in models):
                raise HTTPException(
                    status_code=503,
                    detail=f"Required model {MODEL_NAME} is not available"
                )
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama service check failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Ollama service is not healthy: {str(e)}"
            )

        # Make the actual generation request
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "system": system_prompt,
                "prompt": request.prompt,
                "stream": False,
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
            },
            timeout=60
        )
        response.raise_for_status()
        return {"answer": response.json()}

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request to LLM server timed out. The model might be still loading or the request is too complex."
        )
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with LLM server: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with LLM server: {str(e)}"
        )


@app.post("/rag-workflow")
def run_rag_workflow(request: PromptRequest):
    """Run the RAG workflow using direct Ollama API calls"""
    try:
        # Connect to vector database
        vector_db = connect_to_existing_vector_db()
        logging.info("Connected to vector database")

        # Get embeddings for the query using direct HTTP request
        try:
            embed_response = requests.post(
                "http://ollama:11434/api/embed",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": request.prompt
                }
            )
            embed_response.raise_for_status()
            logging.info("Generated embeddings for query")

            # Get relevant documents
            docs = vector_db.similarity_search(request.prompt, k=4)
            context = "\n".join(doc.page_content for doc in docs)
            logging.info("Retrieved relevant documents")

            # Generate the final response
            response = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": f"Answer the question based ONLY on the following context:\n{context}\nQuestion: {request.prompt}",
                    "stream": False
                }
            )
            response.raise_for_status()
            
            return {
                "question": request.prompt,
                "response": response.json()["response"],
                "context": context
            }

        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Ollama API: {str(e)}")
            raise

    except Exception as e:
        logging.error(f"Error in RAG workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in RAG workflow: {str(e)}"
        )


@app.post("/process-pdfs")
async def process_pdfs_api(directory_path: str):
    """
    API endpoint to process PDF documents and create a vector database.
    """
    try:
        # Verify directory exists
        if not os.path.exists(directory_path):
            logging.error(f"Directory not found: {directory_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {directory_path}"
            )
            
        # List PDF files in directory
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        if not pdf_files:
            logging.error(f"No PDF files found in directory: {directory_path}")
            raise HTTPException(
                status_code=400,
                detail="No PDF files found in the specified directory"
            )
            
        logging.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process PDFs
        vector_db = process_and_create_vector_db(directory_path)
        if not vector_db:
            raise HTTPException(
                status_code=500, 
                detail="Failed to create vector database."
            )

        return {
            "message": "Vector database created successfully", 
            "status": "success",
            "files_processed": len(pdf_files)
        }

    except Exception as e:
        logging.error(f"Error processing PDFs: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing PDFs: {str(e)}"
        )


# @app.post("/rag-workflow")
# def run_rag_workflow(request: PromptRequest):
#     """
#     Run the RAG workflow for the given query prompt.

#     Args:
#         request: PromptRequest containing the query prompt.

#     Returns:
#         dict: Response containing the question and RAG-generated answer.
#     """
#     try:
#         # Connect to the existing OpenSearch vector database
#         vector_db = connect_to_existing_vector_db()
#         logging.info(f"Connected to vector database: {vector_db}")

#         # Initialize the language model
#         llm = ChatOllama(model=MODEL_NAME)

#         # Create the retriever using the vector database and language model
#         retriever = create_retriever(vector_db, llm)

#         # Create the RAG chain
#         chain = create_chain(retriever, llm)

#         # Use the RAG chain to generate an answer
#         res = chain.invoke(input=request.prompt)

#         # Return the response
#         return {"question": request.prompt, "response": res}

#     except Exception as e:
#         logging.error(f"Error running RAG workflow: {e}")
#         raise HTTPException(status_code=500, detail=f"Error running RAG workflow: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)



@app.get("/")
def read_root():
    return {"Hello, World!"}