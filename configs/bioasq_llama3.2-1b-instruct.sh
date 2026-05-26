#!/bin/bash

# BioASQ Augmentation with Llama3.2-1B-Instruct
echo "Starting BioASQ augmentation with llama3.2-1b-instruct..."

python3 src/augment-bioasq.py \
    --model_name llama3.2-1b-instruct \
    --data_path data/trainining14b.json \
    --sample 10

echo "BioASQ augmentation completed!"