#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=0
model_name=cls_condenser_bert
model_prefix=cls_condenser_bert_111

python3 train.py \
    --model_name_or_path $model_name \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-$model_prefix \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls_before_pooler \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16


python3 evaluation.py \
    --model_name_or_path result/my-unsup-simcse-$model_prefix \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test

python3 evaluation.py \
    --model_name_or_path result/my-unsup-simcse-$model_prefix \
    --pooler cls \
    --task_set sts \
    --mode test
