#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
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
#     --output_dir "/scratch/tathagato/overlap_supervision_div_loss_1" \
#     --overwrite_output_dir \
#     --num_decoder_layers_shared 8 \
#     --divergence_loss "cosine" \
#     --divergence_weight 1 \
#     --use_mixed \
#     --overlap_supervision \
#     --use_distance_loss_post_lm_layer \
#     --use_two_head_distance_loss \
#     --current_epoch 0 \
#     --experiment_name "overlap supervision divergence loss 1" 
    
# 9 : [1, yes, post, two_head]

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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_1_post_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_post_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 1 post lm two head"

# 10 : [1, yes, post, one_head]

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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_1_post_lm_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_post_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 1 post lm one head"

# 11 : [1, yes, pre, two_head]
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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_pre_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 1 pre lm two head"

# 12 : [1, yes, pre, one_head]

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
    --output_dir "/scratch/tathagato/overlap_supervision_div_loss_1_pre_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --overlap_supervision \
    --use_distance_loss_pre_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "overlap supervision divergence loss 1 pre lm one head"

# 13 : [1, no, post, two_head]
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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_1_post_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --use_distance_loss_post_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 1 post lm two head"

# 14 : [1, no, post, one_head]

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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_1_post_lm_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --use_distance_loss_post_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 1 post lm one head"

# 15 : [1, no, pre, two_head]
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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_1_pre_lm_loss_two_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --use_distance_loss_pre_lm_layer \
    --use_two_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 1 pre lm two head"

# 16 : [1, no, pre, one_head]

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
    --output_dir "/scratch/tathagato/no_overlap_supervision_div_loss_1_pre_loss_one_head" \
    --overwrite_output_dir \
    --num_decoder_layers_shared 8 \
    --divergence_loss "cosine" \
    --divergence_weight 1 \
    --use_mixed \
    --use_distance_loss_pre_lm_layer \
    --use_one_head_distance_loss \
    --current_epoch 0 \
    --experiment_name "no overlap supervision divergence loss 1 pre lm one head"

#--------------------------------------------------------------------------------
