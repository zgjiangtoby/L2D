#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python L2D.py \
  --train_path "your/path/to/train.csv" \
  --test_path "your/path/to/test.csv" \
  --output_dir "your/path/to/results/" \
  --method L2D \
  --task_name cr \
  --retriever_model "your/path/to/retirever" \
  --label_model "your/path/to/SLM" \
  --llm_model "your/path/to/LLM" \
  --seed 521 \
  --alpha 0.5 \
  --gpu 1 \
  --fine_tune      



