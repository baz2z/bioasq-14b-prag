#!/bin/bash

# BioASQ Augmentation with Qwen2.5-7B-Instruct using golden 13B1 dataset
echo "Starting BioASQ augmentation with qwen2.5-7b-instruct..."

python3 src/augment_bioasq.py \
    --model_name qwen2.5-7b-instruct \
    --data_path data/13B1_golden_clean.json \
    --sample -1

echo "BioASQ augmentation completed!"
