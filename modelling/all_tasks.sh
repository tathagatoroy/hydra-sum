
echo "1"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.25 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0_epoch_1/prob_0.25/" 

echo "2"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.50 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0_epoch_1/prob_0.50/"

echo "3"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.75 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0_epoch_1/prob_0.75/"


echo "4"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.25 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0_epoch_2/prob_0.25/" 

echo "5"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.50 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0_epoch_2/prob_0.50/"

echo "6"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.75 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0_epoch_2/prob_0.75/"



echo "1"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.25 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.5/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.5_epoch_1/prob_0.25/" 

echo "2"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.50 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.5/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.5_epoch_1/prob_0.50/"

echo "3"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.75 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.5/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.5_epoch_1/prob_0.75/"


echo "4"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.25 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.5/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.5_epoch_2/prob_0.25/" 

echo "5"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.50 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.5/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.5_epoch_2/prob_0.50/"

echo "6"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.75 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_0.5/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_0.5_epoch_2/prob_0.75/"

echo "1"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.25 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_1/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_1_epoch_1/prob_0.25/" 

echo "2"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.50 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_1/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_1_epoch_1/prob_0.50/"

echo "3"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.75 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_1/1 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_1_epoch_1/prob_0.75/"


echo "4"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.25 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_1/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_1_epoch_2/prob_0.25/" 

echo "5"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.50 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_1/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_1_epoch_2/prob_0.50/"

echo "6"
python3  train_seq2seq.py \
    --model_type "bart_mult_heads_2" \
    --model_name_or_path facebook/bart-large \
    --train_data_file ./../data/cnn/lexical/train.tsv \
    --eval_data_file ./../data/cnn/lexical/dev.tsv \
    --test_data_file ./../data/cnn/lexical/test.tsv \
    --per_gpu_eval_batch_size 4 \
    --do_eval \
    --use_mixed  \
    --gate_probability 0.75 \
    --input_dir /scratch/tathagato/overlap_supervision_div_loss_1/2 \
    --generate \
    --output_dir "/home2/tathagato/summarization/hydra-sum/modelling/outputs/overlap_supervision_div_loss_1_epoch_2/prob_0.75/"