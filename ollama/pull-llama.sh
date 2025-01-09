./bin/ollama serve &

pip=$!

sleep 5

echo "Pulling llama3.2"
ollama pull llama3.2

wait $pip