#!/bin/bash

# BioASQ Augmentation with Qwen2.5-1.5B-Instruct
echo "Starting BioASQ augmentation with qwen2.5-1.5b-instruct..."

python3 src/augment-bioasq.py \
    --model_name qwen2.5-1.5b-instruct \
    --data_path data/trainining14b.json \
    --sample 10

echo "BioASQ augmentation completed!"