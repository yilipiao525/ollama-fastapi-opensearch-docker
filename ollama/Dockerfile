# FROM ollama/ollama

# COPY ./pull-llama.sh /pull-llama.sh


FROM ollama/ollama

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY ./pull-llama.sh /pull-llama.sh
RUN chmod +x /pull-llama.sh