<<<<<<< HEAD
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Define Pydantic models for request validation
class PromptRequest(BaseModel):
    prompt: str

# Create FastAPI app instance
app = FastAPI()

# OLLAMA_URL =  "http://ollama:11434"   //for docker compose
OLLAMA_URL =  "http://localhost:11434"  #//for local
=======
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import os
import sys
from pathlib import Path

from opensearch import OpensearchEngine

# Pydantic models for the requests
class PromptRequest(BaseModel):
    prompt: str

app = FastAPI(
    title="FastAPI with Ollama and OpenSearch",
    description="API for text generation and PDF search",
    version="1.0.0"
)

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
>>>>>>> origin/main
LLM_SERVER_URL = f"{OLLAMA_URL}/api/generate"
MODEL_NAME = "llama3.2:latest"
system_prompt = "You are a fast food ordering drive thru AI assistant."

<<<<<<< HEAD
@app.post("/generate-answer")
def generate_answer(request: PromptRequest):
    """
    Generate an answer based on the user's prompt.

=======
# Initialize OpenSearch
search_engine = OpensearchEngine(
    opensearch_url=OPENSEARCH_URL,
    index_name="pdf-index"
)




@app.post("/generate-answer")
def generate_answer(request: PromptRequest):  # Use the Pydantic model here
    """
    Generate an answer based on the user's prompt.
    
>>>>>>> origin/main
    Args:
        request: PromptRequest containing the user's prompt
    """
    try:
        response = requests.post(
            LLM_SERVER_URL,
            json={
                "model": MODEL_NAME,
                "system": system_prompt,
<<<<<<< HEAD
                "prompt": request.prompt,
=======
                "prompt": request.prompt,  # Access the prompt from the request model
>>>>>>> origin/main
                "stream": False,
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "frequency_penalty": 0,
                "presence_penalty": 0
            },
            timeout=30
        )
        response.raise_for_status()
        return {"answer": response.json()}

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to LLM server. Make sure it's running and accessible."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with LLM server: {str(e)}"
        )
<<<<<<< HEAD
    
@app.get('/')
def home():
    return {"hello" : "World"}
=======

# OpenSearch endpoints
@app.post("/index-pdf")
async def index_pdf(pdf_path: str):
    """Index a PDF file into OpenSearch."""
    try:
        search_engine.index_pdf(pdf_path)
        return {"message": "PDF indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str, top_k: int = 5):
    """Search for similar content in indexed PDFs."""
    try:
        results = search_engine.search_similar(query, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello, World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi.app:app", host="0.0.0.0", port=8000, reload=True)
>>>>>>> origin/main
