#!/bin/bash

# Define the model name for the curl request
model_name=="llama3.1"

# Set a single INSTANCE_ID for all GPUs
INSTANCE_ID=1  # Use the same INSTANCE_ID across all GPUs

# Function to start the service on a specific GPU_ID
start_service() {
    GPU_ID=$1

    # Set environment variables for the instance
    export OLLAMA_ORIGINS=*
    export OLLAMA_HOST=127.0.0.1:114${INSTANCE_ID}${GPU_ID}
    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    # Start the service for this GPU
    echo "Starting service on GPU ${GPU_ID} with Instance ID ${INSTANCE_ID} (Port 114${INSTANCE_ID}${GPU_ID})"
    nohup /usr/local/bin/ollama serve >/dev/null 2>&1 &
}

# Function to check if the service is running on a specific GPU_ID
check_service() {
    GPU_ID=$1

    # The URL for the instance
    url="http://localhost:114${INSTANCE_ID}${GPU_ID}/api/chat"

    # Make a curl request to check if the service is running
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST ${url} -d '{"model":"'"$model_name"'","system":"","prompt":"","template":""}')
    
    if [ "$response" -eq 200 ]; then
        echo "Service running on GPU ${GPU_ID} with Instance ID ${INSTANCE_ID} (Port 114${INSTANCE_ID}${GPU_ID})"
    else
        echo "Service NOT running on GPU ${GPU_ID} with Instance ID ${INSTANCE_ID} (Port 114${INSTANCE_ID}${GPU_ID})"
        # Restart the service if it's not running
        start_service $GPU_ID
        ollama pull $model_name
    fi
}

# Define GPU IDs
GPU_IDS=(0 1 2 3 4 5 6 7)  # GPU IDs are assumed to be 0-7

# Infinite loop to keep checking the services every 10 seconds
while true; do
    # Loop over all GPUs
    for GPU_ID in "${GPU_IDS[@]}"; do
        # Check if the service is running on the current GPU
        check_service $GPU_ID
    done

    # Wait for 10 seconds before checking again
    echo "Waiting for 10 seconds before checking services again..."
    sleep 10
done
