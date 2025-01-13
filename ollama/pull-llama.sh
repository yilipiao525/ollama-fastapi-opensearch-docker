#!/bin/bash
set -e

# Start Ollama server
ollama serve &
SERVER_PID=$!

# Wait for Ollama server to be ready
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/version | grep -q "version"; do
    echo "Waiting for Ollama server..."
    sleep 2
done
echo "Ollama server is up!"

# Pull models
echo "Pulling models..."
ollama pull gemma2:2b
ollama pull nomic-embed-text

# Verify models are available
echo "Verifying models..."
until curl -sf http://localhost:11434/api/tags | grep -q "gemma2:2b"; do
    echo "Waiting for models to be ready..."
    sleep 5
done

echo "Models are ready!"
wait $SERVER_PID