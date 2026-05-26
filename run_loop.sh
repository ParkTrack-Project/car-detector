#!/bin/bash

# Loop wrapper for the detection script
while true; do
    mkdir -p "${CAR_DETECTOR_IMAGES_DIRECTORY_PATH:=/var/parktrack/car-detector/images}"

    echo "Starting detection cycle..."
    python -m detection.main \
        --model detection/best_openvino_model/best.xml \
        --base-api-url "$API_URL" \
        --api-token "$API_TOKEN" \
        --out_img "$CAR_DETECTOR_IMAGES_DIRECTORY_PATH" \
        --device "CPU"
    
    # Sleep to prevent tight loop if script fails immediately or to rate limit
    sleep ${CAR_DETECTOR_LOOP_DELAY_SECONDS:-60}
done
