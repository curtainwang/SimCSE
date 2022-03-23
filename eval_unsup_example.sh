#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=$1
model_name=$2
model_prefix=$3

python evaluation.py \
    --model_name_or_path result/my-unsup-simcse-$model_prefix \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test

python evaluation.py \
    --model_name_or_path result/my-unsup-simcse-$model_prefix \
    --pooler cls \
    --task_set sts \
    --mode test
