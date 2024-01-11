"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import os, csv, torch, copy
import logging
from torch.utils.data import TensorDataset
from torch import nn
from nltk import sent_tokenize
import pandas as pd
import tqdm
from nltk import word_tokenize, ngrams


def get_overlap(inp, out, ngram=2):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)

logger = logging.getLogger(__name__)


def skld_loss(prob1, prob2):
    """
    Calculates the symmetric KL divergence loss between two normalized probability distributions.

    Args:
    prob1: torch.Tensor of shape (batch_size, vocab_size) with normalized probabilities.
    prob2: torch.Tensor of shape (batch_size, vocab_size) with normalized probabilities.

    Returns:
    torch.Tensor of shape (batch_size,) with the SKLD loss for each sample.
    We want to maximise this loss, so we will return the negative of this value.
    """

    kl_div_12 = torch.sum(prob1 * (torch.log(prob1 + 1e-12) - torch.log(prob2 + 1e-12)), dim=1)
    kl_div_21 = torch.sum(prob2 * (torch.log(prob2 + 1e-12) - torch.log(prob1 + 1e-12)), dim=1)

    return -torch.mean((kl_div_12 + kl_div_21) / 2.0)


def cosine_similarity_on_features(features1, features2):
    """
    Calculates the cosine distance between two output hidden states from the deocder of the model.
    
    Args:
        features1: torch.Tensor of shape (batch_size, max_decoder_length, hidden_size) with hidden states.
        features2: torch.Tensor of shape (batch_size, max_decoder_length, hidden_size)
    """
    # make it batch_size , max_decoder_length * hidden_size
    features1 = features1.view(features1.shape[0], -1)
    features2 = features2.view(features2.shape[0], -1)
    # normalize the features
    features1 = nn.functional.normalize(features1, dim=1)
    features2 = nn.functional.normalize(features2, dim=1)
    # calculate the cosine similarity
    cosine_similarity = torch.sum(features1 * features2, dim=1)
    # return the mean of the cosine similarity for each sample  
    cosine_similarity = torch.mean(cosine_similarity)
    return cosine_similarity


def cosine_similarity(prob1, prob2):
    """
    Calculates the cosine distance between two softmax-normalized probability distributions.

    Args:
    prob1: torch.Tensor of shape (batch_size, vocab_size) with normalized probabilities.
    prob2: torch.Tensor of shape (batch_size, vocab_size) with normalized probabilities.

    Returns:
    torch.Tensor of shape (batch_size,) with the cosine distance for each sample.
    """


    dot_product = torch.sum(prob1 * prob2, dim=1)
    norm_1 = torch.linalg.norm(prob1, dim=1)
    norm_2 = torch.linalg.norm(prob2, dim=1)
    cosine_similarity = (dot_product / (norm_1 * norm_2 + 1e-12))
    # return the mean of the cosine similarity for each sample    
    cosine_similarity = torch.mean(cosine_similarity)
    return cosine_similarity

def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines



# def get_examples(filename):
#     """ filename is csv file with headers article, summary, id, abs score , return a list of dictionaries """
#     csv_file = pd.read_csv(filename, sep = "\t")
#     examples = []
#     for i in tqdm.tqdm(range(len(csv_file))):
#         examples.append({'article': csv_file['article'][i],
#                          'summary': csv_file['highlights'][i],
#                          'id': i,
#                          'overlap': csv_file['overlap'][i]})
#     return examples

def get_examples(filename):
    return _read_tsv(os.path.join(filename))

class InputFeatures(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def convert_abs_score_to_int(abs_score, num_bins):
    """Converts the abstract score to an integer."""
    bin_size = 1.0 / num_bins
    for i in range(num_bins):
        if abs_score <= (i + 1) * bin_size:
            return i
    return num_bins - 1


def convert_examples_to_features(examples, tokenizer, max_length=512, max_decoder_length=128, num_bins=10):
    print("converting examples to features")
    features = []
    #get the pad id
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    #iterate over the examples in dataset and convert them to features
    #print the size of the examples and an example of the example
    print("size of the examples : {0}".format(len(examples)))
    # print("sample example : ")
    # print(examples[0])
    for (ex_index, example) in tqdm.tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        #get the input and output text
        input = example['article']
        output = example['summary']
        id = example['id']
        overlap = get_overlap(input, output)
        overlap_bin = convert_abs_score_to_int(overlap, num_bins)
        #abs_score = example['overlap']

        # convert abs to int for nn.embedding layer
        # bin it to 5 values between 0 and 1
        #abs_score = convert_abs_score_to_int(abs_score, 5)


        #if input or output is empty, skip the example
        if input == '' or output == '':
            continue

        #get the input ids
        input_ids = tokenizer.encode(input, add_prefix_space=True)

        #if input length is greater than max length, truncate the input
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length - 1]

        #pad the input
        padding_length_a = max_length - len(input_ids)
        # 1 for input ids 0 for pad ids
        input_attention_mask = [1] * len(input_ids) + ([0] * padding_length_a)
        input_ids = input_ids + ([pad_id] * padding_length_a)

        """
        decoder_ids = tokenizer.encode(output, add_prefix_space=True)

        if 'gate' in example.keys():
            decoder_ids_2 = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in
                             output.split(' ')]

            decoder_ids_2_flattened = [item for sublist in decoder_ids_2 for item in sublist]
            assert decoder_ids_2_flattened == decoder_ids[1:-1], 'mismatch in splitting w/ gating_supervision'

            gate_tokenlevel = [int(x) for x in example['gate'].split(' ')]
            gate_wplevel = []
            assert len(decoder_ids_2) == len(gate_tokenlevel), 'mismatch in splitting w/ gating_supervision'
            for idx in range(len(decoder_ids_2)):
                gate_wplevel += [gate_tokenlevel[idx]] * len(decoder_ids_2[idx])
            gate_wplevel += [-1, -1]  # makin length equal to decoder_ids
        else:
            gate_wplevel = [-1] * len(decoder_ids)

        assert len(gate_wplevel) == len(decoder_ids), 'mismatch in splitting w/ gating_supervision'"""
        
        if 'gate_sent' in example.keys():
            sent_gates = [float(g) for g in example['gate_sent'].split()] # previously int
            #summary sentences
            #sent_gates has something to do with output sentences
            output_sents = sent_tokenize(output)
            
            #this is only necessary for the specificity where specificity is defined at the sentence level and not for abstractive level
            #assert len(sent_gates) == len(output_sents), 'mismatch in splitting w/ gating_supervision'

            decoder_ids = []
            gate_sent = []
            for sent, g in zip(output_sents, sent_gates):
                decoder_ids_sent = tokenizer.encode(sent, add_prefix_space=True)
                # looks like gate_sent is a list of ids for each word in a sentence map to a gate value
                gate_sent += [g] * len(decoder_ids_sent)
                decoder_ids += decoder_ids_sent

        else:
            decoder_ids = tokenizer.encode(output, add_prefix_space=True)
            gate_sent = [0] * len(decoder_ids)

        if len(decoder_ids) > max_decoder_length:
            decoder_ids = decoder_ids[:max_decoder_length - 1]
            # gate_wplevel = gate_wplevel[:max_decoder_length - 1]
            gate_sent = gate_sent[:max_decoder_length - 1]

        padding_length_b = max_decoder_length - len(decoder_ids)
        decoder_attention_mask = [1] * len(decoder_ids) + ([0] * padding_length_b)
        decoder_ids = decoder_ids + ([pad_id] * padding_length_b)
        # gate_wplevel = gate_wplevel + ([-1] * padding_length_b)
        gate_sent = gate_sent + ([0] * padding_length_b)


        features.append(InputFeatures(input_ids=input_ids,
                                      attention=input_attention_mask,
                                      decoder_attention=decoder_attention_mask,
                                      decoder_ids=decoder_ids,
                                      id=id,
                                      overlap=overlap,
                                      overlap_bin=overlap_bin,
                                      #gate=gate_wplevel,
                                      sent_gate=gate_sent))

    print("size of the features : {0} ".format(len(features)))
    return features


def convert_examples_to_features_pegasus(examples, tokenizer, max_length=512, max_decoder_length=128):
    features = []
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        input = example['article']
        output = example['summary']
        try:
            id = example['id']
        except:
            id = ex_index

        if input == '' or output == '':
            continue

        input_ids = tokenizer.encode(input)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length - 1]

        padding_length_a = max_length - len(input_ids)
        input_attention_mask = [1] * len(input_ids) + ([0] * padding_length_a)
        input_ids = input_ids + ([pad_id] * padding_length_a)

        decoder_ids = tokenizer.encode(output)
        if len(decoder_ids) > max_decoder_length:
            decoder_ids = decoder_ids[:max_decoder_length - 1]

        padding_length_b = max_decoder_length - len(decoder_ids)
        decoder_attention_mask = [1] * len(decoder_ids) + ([0] * padding_length_b)
        decoder_ids = decoder_ids + ([pad_id] * padding_length_b)
        gate_wplevel = [-1] * max_decoder_length
        sent_gate = 0

        features.append(InputFeatures(input_ids=input_ids,
                                      attention=input_attention_mask,
                                      decoder_attention=decoder_attention_mask,
                                      decoder_ids=decoder_ids,
                                      id=id,
                                      gate=gate_wplevel,
                                      sent_gate=sent_gate))
    #print(len(features))
    return features

def load_and_cache_examples(args, tokenizer, split, num_bins=10):
    """
    Loads or creates features from a dataset file and saves them as a cached file.
    Converts features to Tensors and builds a TensorDataset.

    Args:
        args (argparse.Namespace): Training arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer used for pre-processing text data.
        split (str): Dataset split ("train", "dev", or "test").

    Returns:
        TensorDataset: Dataset containing tokenized and padded sequences.
    """

    # Determine data directory and file name based on the split
    # will be using csv files till I get the data hence use different format for train and eval
    
    if split == 'dev':
        data_dir = '/'.join(args.eval_data_file.split('/')[:-1])
        file_name = args.eval_data_file
    elif split == 'test':
        data_dir = '/'.join(args.test_data_file.split('/')[:-1])
        file_name = args.test_data_file
    else:
        data_dir = '/'.join(args.train_data_file.split('/')[:-1])
        file_name = args.train_data_file
    
    data_dir = args.cache_directory
    if split == "dev":
        file_name = args.eval_data_file
    elif split == "test":
        file_name = args.test_data_file
    elif split == "train":
        file_name = args.train_data_file
    print("split : {0} data dir : {1} file name : {2}".format(split, data_dir, file_name))


    # Get the model type prefix based on model type
    model_type = args.model_type
    if model_type == 'bart_subpop' and split == 'train':
        model_type_prefix = model_type
    else:
        model_type_prefix = model_type.split('_')[0]
    print("model prefix : {0}".format(model_type_prefix))

    # Define the cached features file path
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            split,
            model_type_prefix,
            str(args.max_seq_length)
        ),
    )

    # Load features from cache if available and not overwritten
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if split == "dev" and args.eval_data_size != -1:
            features = features[:args.eval_data_size]
        if split == "test" and args.test_data_size != -1:
            features = features[:args.test_data_size]
        if split == "train" and args.train_data_size != -1:
            features = features[:args.train_data_size]

    else:
        # Load examples from dataset file
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = get_examples(file_name)
        if split == "dev" and args.eval_data_size != -1:
            examples = examples[:args.eval_data_size]
        if split == "test" and args.test_data_size != -1:
            examples = examples[:args.test_data_size]
        if split == "train" and args.train_data_size != -1:
            examples = examples[:args.train_data_size]

        
        #set the size to 100 for debugging 
        #examples = examples[:100]
    

        # Subset training data for bart_subpop model
        if model_type == 'bart_subpop' and split == 'train':
            gate = args.subpop
            examples_new = []
            for ex in examples:
                if int(ex['gate_sent']) == gate:
                    examples_new.append(ex)

            examples = examples_new

        # Convert examples to features
        print("converting examples to features")
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            max_decoder_length=args.max_decoder_length,
            num_bins = args.num_bins
        )

        # Save features to cache
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert features to Tensors
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_attention_mask = torch.tensor([f.attention for f in features], dtype=torch.long)
    decoder_ids = torch.tensor([f.decoder_ids for f in features], dtype=torch.long)
    decoder_attention_mask = torch.tensor([f.decoder_attention for f in features], dtype=torch.long)
    #gate = torch.tensor([f.gate for f in features], dtype=torch.long)
    sent_gate = torch.tensor([f.sent_gate for f in features], dtype=torch.float) # FIX THIS
    overlap = torch.tensor([f.overlap for f in features], dtype=torch.float)
    overlap_bin = torch.tensor([f.overlap_bin for f in features], dtype=torch.long)

    # Build and return the TensorDataset
    dataset = TensorDataset(input_ids, input_attention_mask, decoder_ids, decoder_attention_mask, sent_gate, overlap, overlap_bin)
    print("dataset size : {0}".format(len(dataset)))
    #print("example of dataset : ")
    #print(dataset[0])
    #one example of the dataset

    return dataset



def fix_endtoken_weight(weights, decoder_attention):
    batch_size = weights.shape[0]
    num_decoder_length = torch.sum(decoder_attention, dim=1) - 2  # get the index of </s> token
    j = torch.arange(batch_size).long()
    weights[j, num_decoder_length] = 1
    return weights


def shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids one token to the left"""
    prev_output_tokens = input_ids.clone()
    prev_output_tokens[:, :-1] = input_ids[:, 1:]
    prev_output_tokens[:, -1] = pad_token_id
    return prev_output_tokens


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer
