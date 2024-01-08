#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
source /home2/tathagato/miniconda3/bin/activate habitat

echo "epoch 1 starting"
echo "head 0 starting"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_head 0 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_1/head_0" 

echo "head 1 starting"



python3   train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_head 1 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_1/head_1" 

echo "epoch 1 done, epoch 2 starting"

echo "head 0 starting"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_head 0 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_2/head_0" 

echo "head 1 starting"



python3   train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_head 1 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_2/head_1" 



echo "all done"