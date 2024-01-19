# divergence weight 0, 1, 10
# overlap supervision yes or no
# distance loss : pre - post
# distance loss : one head - two head # 
#number of experiments 3 * 2 * 2 * 2 = 24

#gpu 0

# 1 : [0, yes, post, two_head]
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
#     --output_dir "/scratch/tathagato/overlap_supervision_div_loss_0_post_lm_loss_two_head" \
#     --overwrite_output_dir \
#     --num_decoder_layers_shared 8 \
#     --divergence_loss "cosine" \
#     --divergence_weight 0 \
#     --use_mixed \
#     --overlap_supervision \
#     --use_distance_loss_post_lm_layer \
#     --use_two_head_distance_loss \
#     --current_epoch 0 \
#     --train_data_size 4 \
#     --experiment_name "overlap supervision divergence loss 0 post lm two head"


