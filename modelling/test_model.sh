#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python3 train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --use_mixed \
    --do_eval \
    --use_mixed \
    --input_dir ./exp_1/1 \
    --generate \
    --output_dir "results_exp_use_mixed"


