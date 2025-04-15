#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python L2D.py \
  --train_path "./all_datasets/cr_data/train.csv" \
  --test_path "./all_datasets/cr_data/test.csv" \
  --output_dir "./results/" \
  --method L2D \
  --task_name cr \
  --retriever_model "../LLMs/gte-multilingual-base" \
  --label_model "../LLMs/xlm-roberta-base" \
  --llm_model "../LLMs/Qwen2.5-7B-Instruct" \
  --seed 521 \
  --alpha 0.5 \
  --gpu 1 \
  --fine_tune      



