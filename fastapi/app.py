import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Define Pydantic models for request validation
class PromptRequest(BaseModel):
    prompt: str

# Create FastAPI app instance
app = FastAPI()

OLLAMA_URL =  "http://ollama:11434"   #//for docker compose
# OLLAMA_URL =  "http://localhost:11434"  #//for local
LLM_SERVER_URL = f"{OLLAMA_URL}/api/generate"
MODEL_NAME = "llama3.2:latest"
system_prompt = "Your name is Fasta. You are a fast food ordering drive thru AI assistant."

@app.post("/generate-answer")
def generate_answer(request: PromptRequest):
    """
    Generate an answer based on the user's prompt.

    Args:
        request: PromptRequest containing the user's prompt
    """
    try:
        response = requests.post(
            LLM_SERVER_URL,
            json={
                "model": MODEL_NAME,
                "system": system_prompt,
                "prompt": request.prompt,
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
    
@app.get('/')
def home():
    return {"hello" : "World"}
