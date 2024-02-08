import os
import sys
import time
import pandas as pd
import time
import sys
from nltk import sent_tokenize
import logging
#print only errors
import tqdm
from multiprocessing import Pool, freeze_support
import json
sys.path.append("./..")
from postprocessing.rouge import get_rouge_score
import time
import logging
import pickle


# Use logging instead of print
logger = logging.getLogger('postprocessing.rouge')
logger.setLevel(logging.ERROR)



beam_files = ["again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_1", \
              "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_2", \
              "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_3", \
              "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_4", \
              "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_5", \
              "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_6"]
           


mixtures = ["head_0", "head_1"]
def extract_source_reference_and_prediction(file):
    article = []
    reference = []
    prediction = []
    with open(file, "r") as f:
        lines = f.readlines()
        cur_index = 0
        for index , line in enumerate(lines):
            if index % 4 == 0:
                article.append(line)
            elif index % 4 == 1:
                reference.append(line)
            elif index % 4 == 2:
                prediction.append(line)
            elif index % 4 == 3:
                cur_index += 1
    return article, reference, prediction

def create_all_possible_combinations(res):
    # indexed by beam size, mixture type and article index in the dataset
    keys = []
    candidates = []
    references = []
    articles = []
    for mixture_type in tqdm.tqdm(res.keys()):
        if mixture_type in ["head_0", "head_1"]:
            for index in res[mixture_type].keys():
                article = res[mixture_type][index]["article"]
                reference = res[mixture_type][index]["reference"]
                for beam_size in res[mixture_type][index]["predictions"].keys():
                    key = (beam_size, mixture_type, index)
                    keys.append(key)
                    candidates.append(res[mixture_type][str(index)]["predictions"][str(beam_size)])
                    references.append(reference)
                    articles.append(article)
    print(len(keys))
    print(len(candidates))
    print(len(references))
    print(len(articles))

    return keys, candidates, references, articles

def create_all_beam_results_dict(beam_files, mixtures, output_json_dict_file = "train_data_for_beam_size_prediction.json"):
    results = {}
    for mixture in mixtures:
        results[mixture] = {}
        print("mixture : {0}".format(mixture))
        for index,beam_file in tqdm.tqdm(enumerate(beam_files)):
            print(index, beam_file)
            mixture_folder = os.path.join(beam_file, mixture)
            relevant_filepath = os.path.join(mixture_folder, "test_outfinal.txt")
            article, reference, prediction = extract_source_reference_and_prediction(relevant_filepath)

            if index == 0:
                for idx, line in enumerate(article):
                    results[mixture][str(idx)] = {"article": line, "reference": reference[idx], "predictions" : {}}
                    results[mixture][str(idx)]["predictions"][str(index + 1)] = prediction[idx]
            else:
                for idx, line in enumerate(article):
                    results[mixture][str(idx)]["predictions"][str(index + 1)] = prediction[idx]
                #sort the result[mixture] by index 
                results[mixture] = dict(sorted(results[mixture].items(), key=lambda x: int(x[0])))
    #save the json file
    with open(output_json_dict_file, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":


    beam_files = ["again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_1", \
                "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_2", \
                "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_3", \
                "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_4", \
                "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_5", \
                "again_tests//overlap_supervision_div_loss_1_pre_lm_loss_two_head_grad_acc_10_lr_1e4_epoch_1_train_cnn_num_beam_6"]
            


    mixtures = ["head_0", "head_1"]

    output_json_dict_file = "train_data_for_beam_size_prediction.json"
    #read the json file
    with open(output_json_dict_file, "r") as f:
        results = json.load(f)
    keys, candidates, references, articles = create_all_possible_combinations(results)


    #test 
    

    start_time = time.time()

    final_res = get_rouge_score( candidates, references, keys)
    time_taken = time.time() - start_time
    print("Time taken: ", time.time() - start_time)
    print(len(candidates))
    print("expected total time : {0}".format((time_taken * len(candidates) / 1000)))
    print(final_res)

    #save final res as pickle 
    with open("beam_rouge_result_unprocessed.pkl", "wb") as f:
        pickle.dump(final_res, f)
    # for key in final_res.keys():
    #     print(key, final_res[key])  
