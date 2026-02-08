#!/bin/bash

# Loop wrapper for the detection script
while true; do
    echo "Starting detection cycle..."
    python -m detection.main \
        --model detection/best_openvino_model/best.xml \
        --base-api-url "$API_URL" \
        --api-token "$API_TOKEN" \
        --device "CPU"
    
    # Sleep to prevent tight loop if script fails immediately or to rate limit
    sleep 60
done
