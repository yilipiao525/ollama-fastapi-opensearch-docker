
./bin/ollama serve &

pid=$!

sleep 5


echo "Pulling llama3 model"
# ollama pull llama3.2
# ollama pull qwen:0.5b
ollama run gemma2:2b
# qwen0.5B
# Gemma2

wait $pid
