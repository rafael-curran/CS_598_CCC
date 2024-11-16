#!/bin/bash
set -x

# Default values for the parameters
OFFLOADING=0
COMPRESSION=0
BATCH_SIZE=8

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --offloading) OFFLOADING="$2"; shift ;;
        --compression) COMPRESSION="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export PROD=1

# Run the Python script with the specified parameters
python /data/dl-env/bin/python dl-processing-pipeline/training/storage_server.py --offloading "$OFFLOADING" --compression "$COMPRESSION" --batch_size "$BATCH_SIZE"
echo "Production servers are running."

set +x
