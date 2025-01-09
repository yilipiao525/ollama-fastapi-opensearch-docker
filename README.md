# ollama-fastapi-opensearch-docker

run locally with:
   change fastapi/app.py : OLLAMA_URL =  "http://localhost:11434"  #//for local
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload


run in docker compose:
    change fastapi/app.py : OLLAMA_URL =  "http://ollama:11434"   #//for docker compose
    docker compose up --build

