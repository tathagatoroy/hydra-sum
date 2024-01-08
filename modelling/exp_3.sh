#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

source /home2/tathagato/miniconda3/bin/activate habitat

python3 -m memory_profiler train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_0.5(2)" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 0.5 \
    --use_mixed \
    --overlap_supervision \
    --experiment_name "overlap supervision divergence loss 0.5" 
    