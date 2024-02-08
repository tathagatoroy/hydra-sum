export CUDA_VISIBLE_DEVICES=0 
source /home2/tathagato/miniconda3/bin/activate habitat

#beam search 1

python3 train_seq2seq_alternate.py  --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --train_data_size 10000 --test_data_file ./../data/cnn/lexical/train.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_cnn/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_1/ --num_beams 1 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_1/  


if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure"
fi

#beam search 2

python3 train_seq2seq_alternate.py  --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --train_data_size 10000 --test_data_file ./../data/cnn/lexical/train.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_cnn/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_2/ --num_beams 2 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_2/  


if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure"
fi

#beam search 3

python3 train_seq2seq_alternate.py  --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --train_data_size 10000 --test_data_file ./../data/cnn/lexical/train.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_cnn/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_3/ --num_beams 3 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_3/  


if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure"
fi

#beam search 4

python3 train_seq2seq_alternate.py  --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --train_data_size 10000 --test_data_file ./../data/cnn/lexical/train.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_cnn/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_4/ --num_beams 4 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_4/  


if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure"
fi


#beam search 5

python3 train_seq2seq_alternate.py  --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --train_data_size 10000 --test_data_file ./../data/cnn/lexical/train.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_cnn/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_5/ --num_beams 5 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_5/  


if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure"
fi

#beam search 6

python3 train_seq2seq_alternate.py  --model_type 'bart_mult_heads_2' --model_name_or_path 'facebook/bart-large' --train_data_size 10000 --test_data_file ./../data/cnn/lexical/train.tsv --per_gpu_eval_batch_size 4 --do_test --use_head 0 --input_dir /scratch/tathagato/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_cnn/1  --generate --output_dir /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_6/ --num_beams 6 --generation_output_directory /home2/tathagato/summarization/hydra-sum/modelling/again_tests/overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_6/  


if [[ $? = 0 ]]; then
    echo "success"
else
    echo "failure"
fi

#------gpu 0 done