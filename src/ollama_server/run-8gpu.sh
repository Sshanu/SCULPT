#!/bin/bash

curl -fsSL https://ollama.com/install.sh | sudo sh

model=$1

case "$model" in
    "llama3.1")
        model_name="llama3.1"
        ;;
    "phi3")
        model_name="phi3:3.8b-mini-4k-instruct-fp16"
        ;;
    "mistralv2")
        model_name="mistral:7b-instruct-v0.2-fp16"
        ;;
    "phi3.5")
        model_name="phi3.5"
        ;;
    "gemma2")
        model_name="gemma2"
        ;;
    "mistralv3")
        model_name="mistral"
        ;;
    "llama3")
        model_name="llama3:8b-instruct-fp16"
        ;;
    "qwen2.5")
        model_name="qwen2.5"
        ;;
    "llama2-13b")
        model_name="llama2:13b-chat-fp16"
        ;;
    *)
        echo "Unknown model: $model"
        exit 1
        ;;
esac

echo "Model name is set to: $model_name"

(export CUDA_VISIBLE_DEVICES=0 && OLLAMA_HOST=127.0.0.1:114$20 ollama serve) & (export CUDA_VISIBLE_DEVICES=1 && OLLAMA_HOST=127.0.0.1:114$21 ollama serve) & (export CUDA_VISIBLE_DEVICES=2 && OLLAMA_HOST=127.0.0.1:114$22 ollama serve) & (export CUDA_VISIBLE_DEVICES=3 && OLLAMA_HOST=127.0.0.1:114$23 ollama serve) & (export CUDA_VISIBLE_DEVICES=4 && OLLAMA_HOST=127.0.0.1:114$24 ollama serve) & (export CUDA_VISIBLE_DEVICES=5 && OLLAMA_HOST=127.0.0.1:114$25 ollama serve) & (export CUDA_VISIBLE_DEVICES=6 && OLLAMA_HOST=127.0.0.1:114$26 ollama serve) & (export CUDA_VISIBLE_DEVICES=7 && OLLAMA_HOST=127.0.0.1:114$27 ollama serve) & (sleep 1m && ollama pull $model_name) & (sleep 2m && bash keep-alive.sh $model_name $2)