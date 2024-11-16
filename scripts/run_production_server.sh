#!/bin/bash
set -x

# Default values for the parameters
GRPC_HOST=""
GPU=0
BATCH_SIZE=8
EPOCHS=50

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --grpc-host) GRPC_HOST="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if grpc-host is provided
if [ -z "$GRPC_HOST" ]; then
    echo "Usage: $0 --grpc-host <GRPC_HOST> [--gpu <GPU>] [--batch-size <BATCH_SIZE>] [--epochs <EPOCHS>]"
    exit 1
fi

export PROD=1

# Run the Python script with the specified parameters
/data/dl-env/bin/python dl-processing-pipeline/training/train_server.py /data/imagenet -a alexnet \
    --gpu "$GPU" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --grpc-host "$GRPC_HOST"

echo "Production servers are running."
set +x
