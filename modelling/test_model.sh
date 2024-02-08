#!/bin/bash
export CUDA_VISIBLE_DEVICES=3



source /home2/tathagato/miniconda3/bin/activate habitat
# echo "epoch 1 starting"

# python3 train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --do_train \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --per_gpu_train_batch_size 4 \
#     --gradient_accumulation_steps=1 \
#     --num_train_epochs 1 \
#     --learning_rate 1e-5 \
#     --output_dir "/scratch/tathagato/test0" \
#     --overwrite_output_dir \
#     --num_decoder_layers_shared 8 \
#     --divergence_loss "cosine" \
#     --divergence_weight 1 \
#     --use_mixed \
#     --overlap_supervision \
#     --experiment_name "overlap supervision divergence loss 1" \
#     --use_one_head_distance_loss \
#     --use_distance_loss_pre_lm_layer \
#     --train_data_size 4

# echo "epoch 2 starting"

# python3 train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --do_train \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --per_gpu_train_batch_size 4 \
#     --gradient_accumulation_steps=1 \
#     --num_train_epochs 1 \
#     --learning_rate 1e-5 \
#     --output_dir "/scratch/tathagato/test1" \
#     --overwrite_output_dir \
#     --num_decoder_layers_shared 8 \
#     --divergence_loss "cosine" \
#     --divergence_weight 1 \
#     --use_mixed \
#     --overlap_supervision \
#     --experiment_name "overlap supervision divergence loss 1" \
#     --use_one_head_distance_loss \
#     --use_distance_loss_post_lm_layer \
#     --train_data_size 4


# echo "epoch 3 starting"

# python3  train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_head 0 \
#     --input_dir "/scratch/tathagato/test0/1" \
#     --generate \
#     --generation_output_directory "./outputs/test0_epoch_1/head_0" \
#     --eval_data_size 4 \
#     --output_dir "./outputs/test0_epoch_1/head_0"

# echo "epoch 4 starting"


# python3  train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --gate_probability 0.3 \
#     --use_mixed \
#     --input_dir "/scratch/tathagato/test0/1" \
#     --generate \
#     --generation_output_directory "./outputs/test0_epoch_1/gate_prob_0.3" \
#     --eval_data_size 4 \
#     --output_dir "./outputs/test0_epoch_1/head_0"

# echo "epoch 5 starting"

# python3  train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_head 0 \
#     --input_dir "/scratch/tathagato/test1/1" \
#     --generate \
#     --generation_output_directory "outputs/test0_epoch_1/head_0" \
#     --eval_data_size 4 \
#     --output_dir "./outputs/test0_epoch_1/head_0"

# echo "epoch 6 starting"


# python3  train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_mixed \
#     --gate_probability 0.3 \
#     --input_dir "/scratch/tathagato/test1/1" \
#     --generate \
#     --generation_output_directory "outputs/test1_epoch_1/gate_prob_0.3" \
#     --eval_data_size 4 \
#     --output_dir "./outputs/test0_epoch_1/head_0"




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
#     --num_train_epochs 3 \
#     --learning_rate 1e-5 \
#     --output_dir "/scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head" \
#     --overwrite_output_dir \
#     --num_decoder_layers_shared 8 \
#     --divergence_loss "cosine" \
#     --divergence_weight 10 \
#     --use_mixed \
#     --overlap_supervision \
#     --use_distance_loss_post_lm_layer \
#     --use_two_head_distance_loss \
#     --current_epoch 0 \
#     --train_data_size 4 \
#     --experiment_name "overlap supervision divergence loss 10 post lm two head"


python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/1 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/head_0

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 1 --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/1 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/head_1

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/prob_0.25 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/prob_0.25 --gate_probability 0.25

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/prob_0.5 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/prob_0.5 --gate_probability 0.5

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/prob_0.75 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_1/prob_0.75 --gate_probability 0.75

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/head_0

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 1 --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/head_1

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/2  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/prob_0.25 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/prob_0.25 --gate_probability 0.25

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/2  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/prob_0.5 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/prob_0.5 --gate_probability 0.5

python3 train_seq2seq.py --test_data_size 16 --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed --input_dir /scratch/tathagato/overlap_supervision_div_loss_10_post_lm_loss_two_head/2  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/prob_0.75 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_10_post_lm_loss_two_head_epoch_2/prob_0.75 --gate_probability 0.75