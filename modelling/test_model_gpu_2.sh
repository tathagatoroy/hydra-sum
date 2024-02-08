#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
source /home2/tathagato/miniconda3/bin/activate habitat



python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed  --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_1/ --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_1/ --gate_probability 0.5


if [[ $? = 0 ]]; then
	echo "success"
else
	echo "failure"
fi

python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed  --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_1/ --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_1/ --gate_probability 0.75
 

if [[ $? = 0 ]]; then
	echo "success"
else
	echo "failure"
fi

python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed  --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_1/ --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_1/ --gate_probability 0.25
 

if [[ $? = 0 ]]; then
	echo "success"
else
	echo "failure"
fi


python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5/3  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_3/ --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_3/ 

if [[ $? = 0 ]]; then
	echo "success"
else
	echo "failure"
fi

python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 1 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5/3  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_3/ --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_3/ 

if [[ $? = 0 ]]; then
	echo "success"
else
	echo "failure"
fi

python3 train_seq2seq.py --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --test_data_file ./../data/cnn/lexical/test.tsv --per_gpu_eval_batch_size 4 --do_test --use_mixed  --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5/3  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_3/ --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/new_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e5_epoch_3/ --gate_probability 0.5


if [[ $? = 0 ]]; then
	echo "success"
else
	echo "failure"
fi