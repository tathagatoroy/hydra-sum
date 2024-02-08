"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
code from the original hydrasum paper by goyal et al (SalesForce AI)
comments added by BARD/ChatGPT and Tathagato
modification by Tathagato

"""

import argparse
import json
import logging
import os
import random
from typing import Dict
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import train_seq2seq_utils
import single_head_utils
import multi_head_utils_3
import multi_head_utils
from torch import nn
import wandb
import time
import gc
from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    BartConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusConfig,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.utils import logging as transformers_logging
os.environ["WANDB_SILENT"] = "true"
#currently using giving issues with training 
#setting wandb to run in offline mode
os.environ["WANDB_MODE"] = "offline"



transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)
MODEL_CLASSES = {"bart": (BartConfig,
                          single_head_utils.ConditionalGenerationCustomBart,
                          BartTokenizer),
                 "bart_subpop": (BartConfig,
                                 single_head_utils.ConditionalGenerationCustomBart,
                                 BartTokenizer),
                 "bart_mult_heads_3": (BartConfig,
                                       multi_head_utils_3.ConditionalGenerationCustomBartMultHeads,
                                       BartTokenizer),
                 "bart_mult_heads_2": (BartConfig,
                                       multi_head_utils.ConditionalGenerationCustomBartMultHeads,
                                       BartTokenizer)
                 }


def set_seed(args):
    """
    Sets the random seed for various libraries (random, np, torch) for reproducible results.

    Args:
        args (argparse.Namespace): Training arguments containing the seed value.
    """

    random.seed(args.seed)  # set random seed for standard library
    np.random.seed(args.seed)  # set random seed for numpy
    torch.manual_seed(args.seed)  # set random seed for torch

    if args.n_gpu > 0:  # set random seed for all GPUs
        torch.cuda.manual_seed_all(args.seed)


def compute_accuracy_score(y_true, y_pred, ignore_ids=-1):
    """
    Calculates the accuracy score for a set of predictions and gold labels, ignoring specific labels.

    Args:
        y_true (list): List of ground truth labels.
        y_pred (list): List of predicted labels.
        ignore_ids (int, optional): Label ID to be ignored during accuracy calculation. Defaults to -1.

    Returns:
        float: The accuracy score.
    """

    correct, total = 0, 0  # initialize counters

    for pred, gold in zip(y_pred, y_true):  # iterate over predictions and gold labels
        if gold == ignore_ids:  # ignore labels with specific ID
            continue

        pred_label = np.argmax(pred)  # get the predicted label with highest probability

        if pred_label == gold:  # compare predicted and gold labels
            correct += 1  # increment correct predictions counter

        total += 1  # update total number of valid labels

    return correct / total  # calculate and return accuracy score


def save_checkpoints(args, output_dir, model, tokenizer, suffix=None):
    """
    Saves the model and tokenizer checkpoints to a specified directory.

    Args:
        args (argparse.Namespace): Training arguments.
        output_dir (str): Path to the directory where the checkpoints will be saved.
        model (nn.Module): Trained model object.
        tokenizer (PreTrainedTokenizer): Tokenizer used for pre-processing text data.
        suffix (str, optional): Suffix to add to the output directory name. Defaults to None.
    """
    print("calling save checkpoints")
    if suffix is not None:
        output_dir = os.path.join(output_dir, f'ckp_{suffix}')  # update output directory with suffix

    if not os.path.exists(output_dir):  # create output directory if it does not exist
        os.makedirs(output_dir)

    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # handle distributed/parallel training

    model_to_save.save_pretrained(output_dir)  # save model
    tokenizer.save_pretrained(output_dir)  # save tokenizer

    torch.save(args, os.path.join(output_dir, "training_args.bin"))  # save training arguments
    print("Saving model checkpoint to %s", output_dir)

    logger.info("Saving model checkpoint to %s", output_dir)  # log information about saved checkpoints


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    """
    Evaluates the model on the given dataset and saves the results.

    Args:
        args (argparse.Namespace): Training arguments.
        eval_dataset (Dataset): Dataset for evaluation.
        model (PreTrainedModel): Trained model object.
        tokenizer (PreTrainedTokenizer): Tokenizer used for pre-processing text data.
        prefix (str, optional): Prefix to add to the output directory and file names. Defaults to "".

    Returns:
        Dict: Dictionary containing evaluation results (loss, ppl)
    """

    eval_output_dir = args.generation_output_directory

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Set the batch size for evaluation
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Create a sequential sampler for evaluation
    # this is the default config of the sampler
    # looks a a pytorch dataset not huggingface
    eval_sampler = SequentialSampler(eval_dataset)

    # Build the data loader for evaluation
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Create a softmax function for calculating probabilities
    softmax_function = nn.Softmax(dim=-1)

    # Set the model to evaluation mode
    model.eval()

    # Initialize evaluation variables
    eval_loss = 0.0
    eval_steps = 0
    preds = None
    decoder_ids_all = None

    # Open a file to write posteriors if requested
    # I think this computes some probabilities of tokens
    if args.dump_posteriors:
        f_out = open(os.path.join(eval_output_dir, 'prob_out%s.txt' % prefix), 'w')
        
    with torch.no_grad():
        # Loop through the data loader
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move the batch to the specified device
            batch = tuple(t.to(args.device) for t in batch)

            # Unpack the batch
            input_ids, attention, decoder_ids, decoder_attention = batch[0], batch[1], batch[2], batch[3]

            # Handle sentence gate if available
            # what is sentence gate ? 
            try:
                sent_gate = batch[4]
            except:
                sent_gate = None
            try :
                overlap = batch[5]
            except:
                overlap = None
            try :
                overlap_bin = batch[6]
            except:
                overlap_bin = None

            #set overlap and overlap_bin to None if not using them
            overlap_bin = None
            overlap = None


            # Prepare the model inputs
            #using overlap abstraction as gate probability
            # inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
            #           'decoder_attention_mask': decoder_attention, 'generate': False,
            #           'use_gate_supervision': args.use_gate_supervision, 'gate': None, 'sent_gate': sent_gate,
            #           'use_sentence_gate_supervision': args.use_sentence_gate_supervision, 'overlap': overlap, 'overlap_bin': overlap_bin,'use_overlap_supervision': args.overlap_supervision}
            inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                      'decoder_attention_mask': decoder_attention, 'generate': False,
                      'use_gate_supervision': args.use_gate_supervision, 'gate': None, 'sent_gate': sent_gate,
                      'use_sentence_gate_supervision': args.use_sentence_gate_supervision, 'overlap': overlap, 'overlap_bin': overlap_bin,
                      "divergence_loss": None, "divergence_weight": None , 'use_overlap_supervision': False,
                      "use_mixed" : args.use_mixed, "use_head" : args.use_head, "gate_prob" : args.gate_probability, "use_one_head_distance_loss" : False,
                      "use_distance_loss_pre_lm_layer" : False , "use_distance_loss_post_lm_layer" : False, 
                      "use_last_layer_gating" : False, "use_feature_level_gating" : False, 
                      "use_two_head_distance_loss" : False}

            # Handle model-specific inputs
            if args.model_type in ['bart_mult_heads_2', 'bart_mult_heads_3']:
                inputs['use_mixed'] = args.use_mixed
                if not args.use_mixed and args.use_head is None:
                    print('Either set --use_mixed or set --use_head for chosen model_tpye')
                    exit()
                inputs['use_head'] = args.use_head

            # Perform the forward pass
            # gate supervision have different output format than other
            if args.use_gate_supervision:
                outputs, _, gate_prob = model(**inputs)

                # Accumulate gate predictions and labels
                # first if is for initilization (first iteration)
                if preds is None:
                    preds = gate_prob.detach().cpu().numpy()
                    out_label_ids_sent = gate.detach().cpu().numpy()
                    decoder_ids_all = decoder_ids.detach().cpu().numpy()
                else:
                    preds = np.append(preds, gate_prob.detach().cpu().numpy(), axis=0)
                    out_label_ids_sent = np.append(out_label_ids_sent, gate.detach().cpu().numpy(), axis=0)
                    decoder_ids_all = np.append(decoder_ids_all, decoder_ids.detach().cpu().numpy(), axis=0)

            else:
                outputs = model(**inputs)

            # Calculate the loss
            #sentence loss ?
            tmp_eval_loss_sentence = outputs.loss

            eval_loss += tmp_eval_loss_sentence.detach().cpu()
            eval_steps += 1

            if args.dump_posteriors:
                logits = outputs[1]
                softmax_scores = softmax_function(logits)

                decode_ids_shifted_left = train_seq2seq_utils.shift_tokens_left(decoder_ids, tokenizer.pad_token_id)
                decoder_softmax_scores = torch.gather(softmax_scores, dim=2,
                                                      index=decode_ids_shifted_left.unsqueeze(2)).detach().cpu().numpy()
                decoder_softmax_scores = decoder_softmax_scores.squeeze(2)

                for j in range(decoder_softmax_scores.shape[0]):
                    uncleaned_tokens = tokenizer.convert_ids_to_tokens(decode_ids_shifted_left[j],
                                                                       skip_special_tokens=False)
                    input = tokenizer.decode(input_ids[j]).replace('<pad>', '')
                    output = tokenizer.decode(decoder_ids[j]).replace('<pad>', '')
                    f_out.write(input + '\n')
                    f_out.write(output + '\n')

                    for k in range(len(uncleaned_tokens)):
                        f_out.write(uncleaned_tokens[k] + '\t' + str(decoder_softmax_scores[j][k]) + '\n')

                        if uncleaned_tokens[k] == '</s>':
                            break

                    f_out.write('\n\n')

            #if eval_steps > 200:
            #    break

    if args.use_gate_supervision:
        preds = preds.reshape(-1, 2)
        out_label_ids_sent = out_label_ids_sent.reshape(-1)
        acc = compute_accuracy_score(y_true=out_label_ids_sent, y_pred=preds)
        print(acc)

    eval_loss = eval_loss / eval_steps
    ppl = math.exp(eval_loss)

    if args.generate:
        print("starting generation")
        f_out = open(os.path.join(eval_output_dir, '%s_outfinal.txt' % prefix), 'w')
        print(eval_output_dir)

        with torch.no_grad():
            model.eval()
            batch_num = 0
            for batch in tqdm(eval_dataloader):

                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_attention_mask, decoder_ids = batch[0], batch[1], batch[2]

                num_return_sequences = 1
                do_sample = True
                if num_return_sequences > 1:
                    do_sample = False  # else the huggingface code returns same sequences
                input_args = {'input_ids': input_ids,
                              'attention_mask': input_attention_mask,
                              'num_beams': 6, 'length_penalty': 2, 'no_repeat_ngram_size': 3, 'max_length': 100,
                              'min_length': 12, 'top_k': 30, 'top_p': 0.5, 'do_sample': do_sample,
                              'decoder_start_token_id': tokenizer.bos_token_id,
                              'num_return_sequences': num_return_sequences}

                
                # if 'xsum' in args.input_dir:
                #     input_args = {'input_ids': input_ids,
                #                   'attention_mask': input_attention_mask,
                #                   'num_beams': 6, 'length_penalty': 1, 'no_repeat_ngram_size': 3, 'max_length': 60,
                #                   'min_length': 12, 'top_k': 30, 'top_p': 0.5, 'do_sample': do_sample,
                #                   'decoder_start_token_id': tokenizer.bos_token_id,
                #                   'num_return_sequences': num_return_sequences}

                if args.model_type in ['bart_mult_heads_2', 'bart_mult_heads_3']:
                    input_args['use_mixed'] = args.use_mixed
                    input_args['use_head'] = args.use_head
                    input_args['gate_prob'] = args.gate_probability
                    if args.use_mixed:
                        input_args['use_cache'] = False  # will NOT work if this is True for use_mixed
                    if args.use_mixed and args.use_head is not None:
                        print('Set ONLY ONE of --use_mixed or --use_head for chosen model_type during generate')
                    if not args.use_mixed and args.use_head is None:
                        print('Either set --use_mixed or set --use_head for chosen model_type')
                        exit()

                gen_ids = model.generate(**input_args)
                
                for j in range(len(input_ids)): 
                    input = tokenizer.decode(input_ids[j], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    gold = tokenizer.decode(decoder_ids[j], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

                    f_out.write(input.strip() + '\n')
                    f_out.write(gold.strip() + '\n')
                    #print("predicted summaries")
                    for k in range(num_return_sequences):
                        gen = tokenizer.decode(gen_ids[num_return_sequences * j + k],
                                               skip_special_tokens=True, clean_up_tokenization_spaces=False)

                        f_out.write(gen.strip() + '\n')
                        #print(gen)
                    f_out.write('\n')

                batch_num += 1
                #if not args.do_eval and batch_num >= 40:
                #    break

            f_out.close()

    result = {'loss': eval_loss, 'ppl': ppl}
    print(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('\n')
    return result


def evaluate_and_save_model(args, eval_dataset, model, tokenizer, global_step, max_ppl):
    #global step is the number of epochs
    # dynamically alter the 
    #result = evaluate(args, eval_dataset, model, tokenizer, global_step)
    save_checkpoints(args, os.path.join(args.output_dir, global_step), model, tokenizer)

    #if result['ppl'] < max_ppl:
    #    max_ppl = result['ppl']
        # save_checkpoints(args, os.path.join(args.output_dir, 'model-best'), model, tokenizer)

    #return max_ppl

def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """ Train the model """
    print("starting training function")
    #reducing the size of the dataset for debugging
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    print(args.train_batch_size)
    print("size of the train dataset is : " + str(len(train_dataset)))

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    print("freezing layers in main")
    # freeze layers
    model.train()
    model.freeze_weights()

    # Prepare optimizer and schedule (linear warmup and decay)

    no_decay = ["bias", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]



    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    print("num gpus : " + str(args.n_gpu))
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epoch_divergence_loss = 0.0
    epoch_masked_lm_loss = 0.0
    tr_loss = 0.0
    tr_gate_loss = 0.0
    logging_loss = 0.0
    logging_loss_gate = 0.0
    max_ppl = 30.0  # set to arbitrarily large value

    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproducibility

    torch.cuda.empty_cache()
    # alpha = 0.1

    #num_epochs = 1
    print("start iterating")
    for _ in train_iterator:
        epoch_start_time = time.time()
        print("start iterating for epoch : " + str(args.current_epoch))
        args.current_epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(args.device) for t in batch)
            # input_ids, attention, decoder_ids, decoder_attention = batch[0], batch[1], batch[2], batch[3]
            # gate = batch[4]
            # try:
            #     sent_gate = batch[5]
            # except:
            #     sent_gate = None

            # inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
            #           'decoder_attention_mask': decoder_attention, 'generate': False,
            #           'use_mixed': args.use_mixed, 'gate': gate, 'sent_gate': sent_gate,
            #           'use_sentence_gate_supervision': args.use_sentence_gate_supervision}

            # Unpack the batch
            input_ids, attention, decoder_ids, decoder_attention = batch[0], batch[1], batch[2], batch[3]

            # Handle sentence gate if available
            # what is sentence gate ? 
            try:
                sent_gate = batch[4]
            except:
                sent_gate = None
            try :
                overlap = batch[5]
            except:
                overlap = None
            try :
                overlap_bin = batch[6]
            except:
                overlap_bin = None


            # Prepare the model inputs
            # inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
            #           'decoder_attention_mask': decoder_attention, 'generate': False,
            #           'use_gate_supervision': args.use_gate_supervision, 'gate': None, 'sent_gate': sent_gate,
            #           'use_sentence_gate_supervision': args.use_sentence_gate_supervision, 'overlap': overlap, 'overlap_bin': overlap_bin, 'use_overlap_supervision': args.overlap_supervision}


            #if args.use_mixed and args.divergence_loss is not None:
            inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                    'decoder_attention_mask': decoder_attention, 'generate': False,
                    'use_gate_supervision': args.use_gate_supervision, 'gate': None, 'sent_gate': sent_gate,
                    'use_sentence_gate_supervision': args.use_sentence_gate_supervision, 'overlap': overlap, 'overlap_bin': overlap_bin,
                    "divergence_loss": args.divergence_loss, "divergence_weight": args.divergence_weight, 'use_overlap_supervision': args.overlap_supervision,
                    "use_mixed" : True, "use_head" : False, "gate_prob" : args.gate_probability, "use_one_head_distance_loss" : args.use_one_head_distance_loss,
                    "use_distance_loss_pre_lm_layer" : args.use_distance_loss_pre_lm_layer, "use_distance_loss_post_lm_layer" : args.use_distance_loss_post_lm_layer, 
                    "use_last_layer_gating" : args.use_last_layer_gating, "use_feature_level_gating" : args.use_feature_level_gating, 
                    "use_two_head_distance_loss" : args.use_two_head_distance_loss}
                #print("divergence loss : " + str(divergence_loss))
                #print("masked lm loss : " + str(masked_lm_loss))
                #print("total loss : " + str(outputs.loss.item()))

                #log the 3 losses
            if args.use_mixed and args.divergence_loss is not None:
                    outputs, divergence_loss , masked_lm_loss = model(**inputs)
                    #wandb.log({"divergence_loss": divergence_loss, "masked_lm_loss": masked_lm_loss, "total_loss": outputs.loss.item()})
                    #print("divergence loss : " + str(divergence_loss))
                    #print("masked lm loss : " + str(masked_lm_loss))
                    if step % 1000 == 0:
                        print("divergence loss : " + str(divergence_loss))
                        print("masked lm loss : " + str(masked_lm_loss))
                        print("total loss : " + str(outputs.loss.item()))

 
                

            elif args.use_gate_supervision:
                inputs['use_gate_supervision'] = True
                outputs, gate_loss, _ = model(**inputs)
            else:
                outputs = model(**inputs)

            loss = outputs.loss
            
            if args.use_mixed and args.divergence_loss is not None:
                epoch_divergence_loss += divergence_loss
                epoch_masked_lm_loss += masked_lm_loss

            #print(loss)

            tr_loss += loss.detach().item()

            if args.use_gate_supervision:
                tr_gate_loss += gate_loss.detach().item()
                loss += gate_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    """
                    print(getattr(model.model.decoder.layers, "3").fc1.weight)
                    print(getattr(model.model.decoder1.layers, "3").fc1.weight)

                    print(getattr(model.model.decoder.layers, "9").fc1.weight)
                    print(getattr(model.model.decoder1.layers, "9").fc1.weight)
                    """

                    logs = {}
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    loss_scalar_sent = (tr_loss - logging_loss) / args.save_steps
                    logs["loss_sent"] = loss_scalar_sent
                    logging_loss = tr_loss

                    loss_scalar_gate = (tr_gate_loss - logging_loss_gate) / args.save_steps
                    logs['gate_loss'] = loss_scalar_gate
                    logging_loss_gate = tr_gate_loss

                    # alpha = max([alpha - 0.02, 0])
                    print(logs)

                    # Evaluation  # should be str(global_step)
                    # evaluate_and_save_model(args, eval_dataset, model, tokenizer, str(num_epochs), max_ppl)
            gc.collect()

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        #epoch done
        epoch_end_time = time.time()
        print("epoch time : " + str(epoch_end_time - epoch_start_time))
        print("epoch done")
        print("calling evaluate and save model")
        evaluate_and_save_model(args, eval_dataset, model, tokenizer, str(args.current_epoch), max_ppl)
        #num_epochs += 1

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type choose from bart or pegasus (others may be available)",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_data_file",
        default="./../data/cnn_dailymail_val.csv",
        type=str,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--train_data_file",
        default="./../data/cnn_dailymail_train.csv",
        type=str,
        help="The input training data file (a text file)."
    )

    parser.add_argument(
        "--test_data_file",
        default="./../data/cnn_dailymail_test.csv",
        type=str,
        help="The input test data file (a text file)."
    )
    parser.add_argument(
        "--input_dir_weak_model",
        default=None,
        type=str,
        help="The input folder for the weak model. Only required when using example re-weighting"
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_decoder_length",
        default=128,
        type=int,
        help="The maximum total decoder sequence length after tokenization.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size evaluation.", )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="No. steps before backward/update", )
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument("--max_steps", default=-1, type=int, help="If>0: no. train steps. Override num_train_epochs.", )
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="Warmup steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--cache_directory", type=str, default="./cache", help="cache directory")
    

    # custom flags
    parser.add_argument("--generate", action="store_true", help="Generate summaries for dev set", )
    parser.add_argument("--dump_posteriors", action="store_true", help="Dump posterior probs at intermediate steps", )
    parser.add_argument("--use_mixed", action="store_true", help="Have multiple heads")
    parser.add_argument("--num_heads", type=int, default=2, help="no. of heads to use")
    parser.add_argument("--use_gate_supervision", action="store_true", help="Use supervision for gating")
    parser.add_argument("--use_head", type=int, default=None, help="use with --generate, options: 0, 1")
    parser.add_argument("--num_decoder_layers_shared", type=int, default=8, help="shared layers w/ mult heads. (1-12)")
    parser.add_argument("--gate_probability", type=float, default=None, help="gate prob")
    parser.add_argument("--subpop", type=int, default=0, help="subpopulation to train on")
    parser.add_argument("--use_sentence_gate_supervision", action="store_true", help="use sentence gating")
    parser.add_argument("--train_data_size", type=int, default=-1, help="size of training data")
    parser.add_argument("--eval_data_size", type=int, default=-1, help="size of eval data")
    parser.add_argument("--test_data_size", type=int, default=-1, help="size of test data")
    parser.add_argument("--num_bins", type=int, default=100, help="number of bins for dividing the controllable scoring")
    parser.add_argument("--divergence_loss", type = str, default = None , help = "divergence loss to use") # should be either kl or cosine
    parser.add_argument("--divergence_weight", type = float, default = 0.2, help = "weight for divergence loss")
    parser.add_argument("--experiment_name", type = str, default = "default", help = "name of the experiment")
    parser.add_argument("--overlap_supervision", action="store_true", help="use overlap score as gate probability")
    parser.add_argument("--use_one_head_distance_loss", action="store_true", help="use distance loss for one head")
    parser.add_argument("--use_two_head_distance_loss", action="store_true", help="use distance loss for two head")
    parser.add_argument("--use_distance_loss_pre_lm_layer", action="store_true", help="use distance loss for pre lm layer")
    parser.add_argument("--use_distance_loss_post_lm_layer", action="store_true", help="use distance loss for post lm layer") # this is the current default behaviour , needs to be changed
    parser.add_argument("--generate_after_training", action="store_true", help="generate summaries after training")
    parser.add_argument("--use_last_layer_gating", action="store_true", help="use gating on lm layer")
    parser.add_argument("--use_feature_level_gating", action="store_true", help="use gating on feature level(After 8th layer of decoder)")
    parser.add_argument("--generation_output_directory", type = str, default = None, help = "directory to save generated summaries")
    parser.add_argument("--current_epoch", type = int, default = 0, help = "current epoch number") # this is used to for restarting from the right checkpoint 

    



    # divergence for now can be only kl or cosine
    # divergence weight is the amount of weight to give the loss 
    args = parser.parse_args()
    print(args.generation_output_directory)

    if args.generate is True and args.use_head is not None:
        if args.use_head == 0:
            args.output_dir = os.path.join(args.output_dir, "head_0")
            args.generation_output_directory = os.path.join(args.generation_output_directory, "head_0")
        elif args.use_head == 1:
            args.output_dir = os.path.join(args.output_dir, "head_1")
            args.generation_output_directory = os.path.join(args.generation_output_directory, "head_1")
    if args.generate is True and args.gate_probability is not None:
        args.output_dir = os.path.join(args.output_dir, "gate_prob_" + str(args.gate_probability))
        args.generation_output_directory = os.path.join(args.generation_output_directory, "gate_prob_" + str(args.gate_probability))

    print(args.generation_output_directory)
    print(args.output_dir)

    #some validation stuff
    if args.use_two_head_distance_loss is False and args.use_one_head_distance_loss is False:
        args.use_two_head_distance_loss = True
    if args.use_distance_loss_pre_lm_layer is False and args.use_distance_loss_post_lm_layer is False:
        args.use_distance_loss_post_lm_layer = True
    if args.use_last_layer_gating is False and args.use_feature_level_gating is False:
        args.use_last_layer_gating = True

    #intialize wandb
    #wandb.init(project="hydra-sum", name=args.experiment_name)
    #wandb.config.update(args)
    print(args)


    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.n_gpu = 1
    device = torch.device("cuda", args.gpu_device)
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'model.log')
    )

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print('Starting...')
    if args.input_dir is not None:
        print('loading model')
        print("from args.input_dir : " + str(args.input_dir))
        tokenizer = tokenizer_class.from_pretrained(args.input_dir)
        model = model_class.from_pretrained(args.input_dir)
        print("model and tokenizer loaded")

        if args.model_type in ['bart_mult_heads_2', 'pegasus_mult_heads', 'bart_mult_heads_3']:
            args_saved = torch.load(os.path.join(args.input_dir, 'training_args.bin'))
            model.model.num_decoder_layers_shared = args_saved.num_decoder_layers_shared  # fix in old models
            #print(args_saved.num_decoder_layers_shared)

    else:
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        config.num_bins = args.num_bins
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

        if args.model_type in ['bart_mult_heads_2', 'pegasus_mult_heads', 'bart_mult_heads_3']:
            print(f'Using mixture of experts w/ {args.num_decoder_layers_shared} shared layers in decoder..')
            config.num_decoder_layers_shared = args.num_decoder_layers_shared
            config.num_bins = args.num_bins
            args.use_mixed = True  # Set this as true for training, evaluation can be controlled
            model.initialize_correct_weights(config, num_decoder_layers_shared=args.num_decoder_layers_shared)

    print("sending the model to : {0}".format(args.device))
    #model.to(args.device)
    model.cuda()
    print("model sent to : {0}".format(args.device))

    logger.info("Training/evaluation parameters %s", args)
    #test_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, 'test')
    #train_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, 'train')
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if args.do_eval:
        eval_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, 'dev')
        print("start evaluation")
        evaluate(args, eval_dataset, model, tokenizer, 'dev')
        print("end evaluation")
    
    if args.do_test:
        
        test_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, 'test')
        print("size of the test_dataset : {0}".format(len(test_dataset)))
        print("start testing")
        evaluate(args, test_dataset, model, tokenizer, 'test')
        print("end testing")



        

    if args.do_train:
        train_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, 'train')
        print('Starting training..')
        train(args, train_dataset, eval_dataset, model, tokenizer)
        print('Done training..')
        #if args.generate_after_training:
        #    print("start evaluation")
        #    evaluate(args, eval_dataset, model, tokenizer, 'test')
        #evaluate(args, eval_dataset, model, tokenizer, 'dev')
    
    print("Owari da")


if __name__ == "__main__":
    main()
