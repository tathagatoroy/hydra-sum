#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

source /home2/tathagato/miniconda3/bin/activate habitat

# python3 -m memory_profiler train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --do_train \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --per_gpu_train_batch_size 4 \
#     --gradient_accumulation_steps=1 \
#     --num_train_epochs 8 \
#     --learning_rate 1e-5 \
#     --output_dir "/scratch/tathagato/overlap_supervision_div_loss_10" \
#     --overwrite_output_dir \
#     --num_decoder_layers_shared 8 \
#     --divergence_loss "cosine" \
#     --divergence_weight 10 \
#     --use_mixed \
#     --overlap_supervision \
#     --use_distance_loss_post_lm_layer \
#     --use_two_head_distance_loss \
#     --current_epoch 0 \
#     --experiment_name "overlap supervision divergence loss 10" 

#gpu 2

# 17 : [10, yes, post, two_head]

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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_post_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 10 post lm two head"

# 18 : [10, yes, post, one_head]
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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_post_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 10 post lm one head"

# 19 : [10, yes, pre, two_head]
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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_10_pre_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_pre_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 10 pre lm two head"

# 20 : [10, yes, pre, one_head]

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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_10_pre_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_pre_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 10 pre lm one head"

# 21 : [10, no, post, two_head]
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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_10_post_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --use_distance_loss_post_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 10 post lm two head"

# 22 : [11, no, post, one_head]

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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_10_post_lm_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --use_distance_loss_post_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 10 post lm one head"

# 23 : [10, no, pre, two_head]
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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --use_distance_loss_pre_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 10 pre lm two head"

# 24 : [10, no, pre, one_head]

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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_10_pre_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 10 \
    --use_mixed \
    --use_distance_loss_pre_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 10 pre lm one head"



    
