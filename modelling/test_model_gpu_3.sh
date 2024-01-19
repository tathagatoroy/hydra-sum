#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
source /home2/tathagato/miniconda3/bin/activate habitat

# echo "epoch 1 starting"
# echo "head 0 starting"
# python3  train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_head 0 \
#     --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/1 \
#     --generate \
#     --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_1/head_0" 

# echo "head 1 starting"



# python3   train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_head 1 \
#     --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/1 \
#     --generate \
#     --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_1/head_1" 

# echo "epoch 1 done, epoch 2 starting"

# echo "head 0 starting"
# python3  train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_head 0 \
#     --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/2 \
#     --generate \
#     --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_2/head_0" 

# echo "head 1 starting"



# python3   train_seq2seq.py \
#     --model_type "bart_mult_heads_2" \
#     --model_name_or_path facebook/bart-large \
#     --train_data_file ./../data/cnn/lexical/train.tsv \
#     --eval_data_file ./../data/cnn/lexical/dev.tsv \
#     --test_data_file ./../data/cnn/lexical/test.tsv \
#     --per_gpu_eval_batch_size 4 \
#     --do_eval \
#     --use_head 1 \
#     --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.1/2 \
#     --generate \
#     --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.1_epoch_2/head_1" 



# echo "all done"

# 

echo "beg 1"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 0 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_0_pre_loss_one_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_loss_one_head_epoch_2/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_loss_one_head_epoch_2/head_0
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 2"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 1 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_0_pre_loss_one_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_loss_one_head_epoch_2/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_loss_one_head_epoch_2/head_1
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 3"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 0 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_1_post_lm_loss_two_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_1_post_lm_loss_two_head_epoch_2/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_1_post_lm_loss_two_head_epoch_2/head_0
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 4"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 1 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_1_post_lm_loss_two_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_1_post_lm_loss_two_head_epoch_2/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_1_post_lm_loss_two_head_epoch_2/head_1
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 5"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 0 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_0_pre_lm_loss_two_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_lm_loss_two_head_epoch_2/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_lm_loss_two_head_epoch_2/head_0
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 6"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 1 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_0_pre_lm_loss_two_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_lm_loss_two_head_epoch_2/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_0_pre_lm_loss_two_head_epoch_2/head_1
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 7"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_loss_one_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_1_pre_loss_one_head_epoch_2/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_1_pre_loss_one_head_epoch_2/head_0
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 8"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 1 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_loss_one_head/2 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_1_pre_loss_one_head_epoch_2/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/overlap_supervision_div_loss_1_pre_loss_one_head_epoch_2/head_1
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 9"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 0 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_10_post_lm_loss_one_head/1 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_post_lm_loss_one_head_epoch_1/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_post_lm_loss_one_head_epoch_1/head_0
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 10"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 1 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_10_post_lm_loss_one_head/1 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_post_lm_loss_one_head_epoch_1/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_post_lm_loss_one_head_epoch_1/head_1
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 11"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 0 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head/1 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head_epoch_1/head_0 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head_epoch_1/head_0
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi

echo "beg 12"
python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 8 --do_test --use_head 1 --input_dir /scratch/tathagato/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head/1 --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head_epoch_1/head_1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/tests/no_overlap_supervision_div_loss_10_pre_lm_loss_two_head_epoch_1/head_1
if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure: $?"
fi