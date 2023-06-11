import argparse
import os
import json
from tqdm import tqdm
import pickle
from IPython import embed
import csv
import random

train = "datasets/qrecc/new_train.json"
test = "datasets/qrecc/new_test.json"
train_rel = "datasets/qrecc/train_rel_1.json"
test_rel = "datasets/qrecc/test_rel_1.json"
train_rel_gold = "datasets/qrecc/train_rel_gold_1.trec"
test_rel_gold = "datasets/qrecc/test_rel_gold_1.trec"

def create_label_rel_turn(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['sample_id']
            id_list = sample_id.split('-')
            conv_id = id_list[0]
            turn_id = id_list[1]
            history_query = obj[i]["context_queries"]
            query = obj[i]["query"]
            rewrite = obj[i]["oracle_query"]
            last_response = obj[i]["last_response"]
            #pos_docs = obj[i]["pos_docs"]
            if len(obj[i]["pos_docs"]) > 0:
                pos_docs_id = obj[i]["pos_docs"]
            else:
                continue

            if int(turn_id) > 1: # if first turn
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "query_pair": "",
                        "last_response": last_response,
                        #"pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, int(turn_id) - 1):
                    query_pair = history_query[tid]
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "query_pair": query_pair,
                            "last_response": last_response,
                            #"pos_docs": pos_docs,
                            "pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

def convert_gold_to_trec(gold_file, trec_file):
    with open(gold_file, "r") as f, open(trec_file, "w") as g:
        data = f.readlines()
        num = 0
        for line in data:
            line = json.loads(line)
            qid = line["id"]
            #query = line["query"]
            if len(line["pos_docs_id"]) == 0:
                continue
            doc_id = line["pos_docs_id"][0]
            g.write("{} {} {} {}".format(qid,
                                        "Q0",
                                        doc_id,
                                        1,
                                        ))
            g.write('\n')
            num += 1
        print(num)

def create_PRJ(label_file, query_file, output):
    with open(label_file, "r") as f1, open(query_file, "r") as f2, open(output, "w") as g:
        obj_1 = f1.readlines()
        obj_2 = f2.readlines()
        one = 0
        zero = 0
        total_nums = len(obj_2)
        print(len(obj_1), len(obj_2))
        idx = 0
        #assert len(obj_1) == len(obj_2)
        for i in range(total_nums):
            obj = json.loads(obj_1[idx])
            obj_2[i] = json.loads(obj_2[i])
            sample_id = obj['id']
            query_id = obj_2[i]['sample_id']
            if sample_id != query_id:
                continue
            conv_id = obj['conv_id']
            turn_id = obj['turn_id']
            rel_label = obj['rel_label']
            cur_query = obj_2[i]['query']
            history_query = obj_2[i]["context_queries"]
            assert len(history_query) == len(rel_label)
            if len(history_query) > 0:
                for index in range(len(history_query)):
                    if rel_label[index] == 1:
                        one += 1
                    else:
                        zero += 1
                    g.write(
                        json.dumps({
                            "id": sample_id + '-' + str(index + 1),
                            "query": cur_query,
                            "rel_query": history_query[index],
                            "rel_label": rel_label[index],
                        }) + "\n")
            idx += 1
        print("one", one)
        print("zero", zero)



if __name__ == "__main__":
    create_PRJ("output/qrecc/dense_rel/train_rel_label_rawq_1.json", train, "datasets/qrecc/filter_train_q_1.json") # one 15294 zero 72048
    create_PRJ("output/qrecc/dense_rel/test_rel_label_rawq_1.json", test, "datasets/qrecc/filter_test_q_1.json") # one 3978 zero 20351
    create_label_rel_turn(test_rel, test_rel)
    convert_gold_to_trec(test_rel, test_rel_gold)
    create_label_rel_turn(train, train_rel)
    convert_gold_to_trec(train_rel, train_rel_gold)
