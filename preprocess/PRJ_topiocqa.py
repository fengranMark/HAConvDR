import argparse
import os
import json
from tqdm import tqdm
import pickle
from IPython import embed
import csv
import random


collection_tsv = "datasets/topiocqa/full_wiki_segments.tsv"
collection_json = "datasets/topiocqa/full_wiki_segments.jsonl"
train = "datasets/topiocqa/topiocqa_train.json"
dev = "datasets/topiocqa/topiocqa_dev.json"
train_gold = "datasets/topiocqa/ir_all_history_train.json"
dev_gold = "datasets/topiocqa/ir_all_history_dev.json"
train_new = "datasets/topiocqa/train_new.json"
dev_new = "datasets/topiocqa/dev_new.json"
train_trec_gold = "datasets/topiocqa/train_gold.trec"
dev_trec_gold = "datasets/topiocqa/dev_gold.trec"
train_rel = "datasets/topiocqa/train_rel_1.json"
dev_rel = "datasets/topiocqa/dev_rel.json"
train_rel_gold = "datasets/topiocqa/train_rel_gold_1.trec"
dev_rel_gold = "datasets/topiocqa/dev_rel_gold.trec"


id_col= 0
text_col= 1
title_col = 2

# .tsv -> .jsonl
def convert_collection(collection_tsv, collection_json):
    with open(collection_tsv, 'r') as input, open(collection_json, 'w') as output:
        reader = csv.reader(input, delimiter="\t") # passage_nums = 25700592
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                # ['id', 'text', 'title'] id from 1
                continue
            title = row[title_col]
            text = row[text_col]
            title = ' '.join(title.split(' [SEP] '))
            break
                #obj = {"contents": " ".join([title, text]), "id": f"doc{i}"} # doc10
                #output.write(json.dumps(obj, ensure_ascii=False) + '\n')

def load_collection(collection_file, title = False):
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
    ext = collection_file[collection_file.rfind(".") + 1:]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    print("begin load")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                pid = int(obj["id"][3:])
                #passage = obj["title"] + "[SEP]" + obj["text"]
                passage = obj["title"] + obj["text"]
                all_passages[pid] = passage
        else:
            first_line = True
            for line in tqdm(f):
                if first_line:
                    first_line = False
                    continue
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = int(line_arr[0])
                    if title == True:
                        passage = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
                    else:
                        passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages

# combine original data and gold ir data for training
def combine_data_train(inputs, inputs_gold, inputs_rewrite, output, collection):
    with open(inputs, "r") as f, open(inputs_gold, "r") as gf, open(inputs_rewrite, "r") as rw, open(output, "w") as g:
        obj = json.load(f)
        obj_g = json.load(gf)
        obj_rw = json.load(rw)
        assert len(obj) == len(obj_g)
        assert len(obj) == len(obj_rw)
        total_nums = len(obj)
        all_passages = load_collection(collection)
        print("loading collection finish!")
        history_rewrite = []
        for i in range(total_nums):
            query = obj[i]["Question"]
            rewrite = obj_rw[i]["question"]
            answer = obj[i]["Answer"]
            conv_id = obj_g[i]["conv_id"]
            turn_id = obj_g[i]["turn_id"]
            history_query = []
            if int(turn_id) == 1:
                history_rewrite = []
                last_response = ""
            elif int(turn_id) > 1 and i > 0:
                history_rewrite.append(obj_rw[i - 1]["question"])
                last_response = ' '.join(obj_g[i - 1]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i - 1]["positive_ctxs"][0]["text"]
            history_answer = []
            idx = 0
            for key in obj[i]["Context"]:
                if idx % 2 == 0:
                    history_query.append(key)
                else:
                    history_answer.append(key)
                idx += 1
            topic = obj[i]["Topic"]
            sub_topic = obj[i]["Topic_section"]
            rationale = obj[i]["Rationale"]
            #additional_answers = obj[i]["Additional_answers"] # only dev
            is_nq = obj[i]["is_nq"]
            pos_docs = []
            pos_docs_id = []
            pos_docs.append(' '.join(obj_g[i]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i]["positive_ctxs"][0]["text"])
            pos_docs_id.append(int(obj_g[i]["positive_ctxs"][0]["passage_id"]))

            # random negatives
            neg_nums = 1
            neg_docs = []
            neg_docs_id = random.sample(range(0, 25700592), neg_nums)
            pos_id = pos_docs_id[0]
            if (pos_id - 1) in neg_docs_id:
                replace = True
                pos = pos_id - 1
                while replace:
                    neg_new = random.randint(0, 25700592)
                    neg_docs_id.remove(pos)
                    neg_docs_id.append(neg_new)
                    if neg_new != pos:
                        replace = False

            for j in range(len(neg_docs_id)):
                idx = neg_docs_id[j] + 1
                neg_docs.append(all_passages[idx])
            #print(len(neg_docs))
            #print(len(neg_docs_id))
            assert len(neg_docs) == len(neg_docs_id)

            # BM25 hard_neg
            hard_neg_docs = []
            hard_neg_docs_id = []
            
            g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id),
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "is_nq": is_nq,
                        "query": query,
                        "rewrite": rewrite,
                        "answer": answer,
                        "history_query": history_query,
                        "history_rewrite": history_rewrite,
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                        "neg_docs": neg_docs,
                        "neg_docs_id": neg_docs_id,
                        "hard_neg_docs": hard_neg_docs,
                        "hard_neg_docs_id": hard_neg_docs_id,
                    }) + "\n")
        print(total_nums)

def combine_data_test(inputs, inputs_gold, inputs_rewrite, output):
    with open(inputs, "r") as f, open(inputs_gold, "r") as gf, open(inputs_rewrite, "r") as rw, open(output, "w") as g:
        obj = json.load(f)
        obj_g = json.load(gf)
        total_nums = len(obj)
        obj_rw = json.load(rw)
        assert len(obj) == len(obj_g)
        assert len(obj) == len(obj_rw)
        history_rewrite = []
        for i in range(total_nums):
            query = obj[i]["Question"]
            rewrite = obj_rw[i]["question"]
            answer = obj[i]["Answer"]
            conv_id = obj_g[i]["conv_id"]
            turn_id = obj_g[i]["turn_id"]
            history_query = []
            if int(turn_id) == 1:
                history_rewrite = []
                last_response = ""
            elif int(turn_id) > 1 and i > 0:
                history_rewrite.append(obj_rw[i - 1]["question"])
                last_response = ' '.join(obj_g[i - 1]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i - 1]["positive_ctxs"][0]["text"]

            history_answer = []
            idx = 0
            for key in obj[i]["Context"]:
                if idx % 2 == 0:
                    history_query.append(key)
                else:
                    history_answer.append(key)
                idx += 1
            topic = obj[i]["Topic"]
            sub_topic = obj[i]["Topic_section"]
            rationale = obj[i]["Rationale"]
            additional_answers = obj[i]["Additional_answers"] # only test
            is_nq = obj[i]["is_nq"]
            pos_docs = []
            pos_docs_id = []
            pos_docs.append(' '.join(obj_g[i]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i]["positive_ctxs"][0]["text"])
            pos_docs_id.append(int(obj_g[i]["positive_ctxs"][0]["passage_id"]))

            g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id),
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "is_nq": is_nq,
                        "query": query,
                        "rewrite": rewrite, 
                        "answer": answer,
                        "history_query": history_query,
                        "history_rewrite": history_rewrite,
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        #"rationale": rationale,
                        #"additional_answers": additional_answers, # "Topic", "Topic_section"
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
        print(total_nums)

def create_label_rel_turn(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['conv_id']
            turn_id = obj[i]['turn_id']
            history_query = obj[i]["history_query"]
            history_rewrite = obj[i]["history_rewrite"]
            history_answer = obj[i]["history_answer"]
            last_response = obj[i]["last_response"]
            topic = obj[i]["topic"]
            sub_topic = obj[i]["sub_topic"]
            query = obj[i]["query"]
            rewrite = obj[i]["rewrite"]
            answer = obj[i]["answer"]
            pos_docs = obj[i]["pos_docs"]
            pos_docs_id = obj[i]["pos_docs_id"]

            if int(turn_id) > 1: # if first turn
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "rewrite": rewrite,
                        "query_pair": "",
                        "rewrite_query_pair": "",
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, int(turn_id) - 1):
                    query_pair = history_query[tid]
                    rewrite_query_pair = history_rewrite[tid]
                    #turn_pair_id = str(turn_id) + '-' + str(tid + 1)
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "rewrite": rewrite,
                            "query_pair": query_pair,
                            "rewrite_query_pair": rewrite_query_pair,
                            "history_answer": history_answer,
                            "last_response": last_response,
                            "topic": topic,
                            "sub_topic": sub_topic,
                            "pos_docs": pos_docs,
                            "pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

def create_label_rel_token(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['conv_id']
            turn_id = obj[i]['turn_id']
            history_query = obj[i]["history_query"]
            history_answer = obj[i]["history_answer"]
            query = obj[i]["query"]
            answer = obj[i]["answer"]
            pos_docs_id = obj[i]["pos_docs_id"]

            token_set = []
            for key in history_query:
                sent = key.strip().split()
                token_set.extend(sent)

            if int(turn_id) > 1: 
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "query_pair": "",
                        #"history_answer": history_answer,
                        #"last_response": last_response,
                        #"pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, len(token_set)):
                    query_pair = token_set[tid]
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "query_pair": query_pair,
                            #"history_answer": history_answer,
                            #"last_response": last_response,
                            #"pos_docs": pos_docs,
                            "pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

def create_topic_rel_turn(inputs, output, mode):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        cur_conv_index = 0
        cur_turn_rel = []
        rel_label = {}
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['conv_id']
            turn_id = obj[i]['turn_id']
            history_query = obj[i]["history_query"]
            #history_rewrite = obj[i]["history_rewrite"]
            #history_answer = obj[i]["history_answer"]
            #last_response = obj[i]["last_response"]
            topic = obj[i]["topic"]
            sub_topic = obj[i]["sub_topic"]
            query = obj[i]["query"]
            #rewrite = obj[i]["rewrite"]
            #answer = obj[i]["answer"]
            #pos_docs = obj[i]["pos_docs"]
            #pos_docs_id = obj[i]["pos_docs_id"]
            if int(turn_id) == 1:
                cur_conv_index = i
                cur_turn_rel = []
                rel_label[str(conv_id) + '-' + str(turn_id)] = cur_turn_rel
            else:
                for ind in range(cur_conv_index, i):
                    history_topic = obj[ind][mode]
                    if mode == "topic":
                        if topic == history_topic:
                            cur_turn_rel.append(1)
                        else:
                            cur_turn_rel.append(0)
                    elif mode == "sub_topic":
                        if sub_topic == history_topic:
                            cur_turn_rel.append(1)
                        else:
                            cur_turn_rel.append(0)
                rel_label[str(conv_id) + '-' + str(turn_id)] = cur_turn_rel
                cur_turn_rel = []

        #print(rel_label)
        for key, value in rel_label.items():
            id_list = key.split('-')
            conv_id = id_list[0]
            turn_id = id_list[1]
            if int(turn_id) == 1:
                g.write(
                    json.dumps({
                        "id": str(key),
                        "conv_id": str(conv_id),
                        "turn_id": str(turn_id),
                        "rel_label": []
                    }) + "\n")
            else:
                g.write(
                    json.dumps({
                        "id": str(key),
                        "conv_id": str(conv_id),
                        "turn_id": str(turn_id),
                        "rel_label": value
                    }) + "\n")
                


def passage_length(collection_file):
    max_passage_length = 0
    max_256_length = 0
    p_length = {}

    with open(collection_tsv, 'r') as input:
        reader = csv.reader(input, delimiter="\t") # passage_nums = 25700592
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                continue
            title = row[title_col]
            text = row[text_col]
            title = ' '.join(title.split(' [SEP] '))
            passage = text.strip().split()
            passage_length = len(passage)
            if passage_length not in p_length:
                p_length[passage_length] = 1
            else:
                p_length[passage_length] += 1

            if passage_length > max_passage_length:
                max_passage_length = passage_length
                print(max_passage_length)
            if passage_length > 256:
                max_256_length += 1
                #print(text)
        print("max_256_length", max_256_length)
        print("max_passage_length", max_passage_length)
        p_length = sorted(p_length.items(), key = lambda x: x[1]) # 194
        print(p_length)
        #print("max_passage_topic_length", max_passage_topic_length)


def convert_gold_to_trec(gold_file, trec_file):
    with open(gold_file, "r") as f, open(trec_file, "w") as g:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            qid = line["id"]
            #query = line["query"]
            doc_id = line["pos_docs_id"][0]
            g.write("{} {} {} {}".format(qid,
                                        "Q0",
                                        doc_id,
                                        1,
                                        ))
            g.write('\n')
            
def create_PRJ(label_file, query_file, output):
    with open(label_file, "r") as f1, open(query_file, "r") as f2, open(output, "w") as g:
        obj_1 = f1.readlines()
        obj_2 = f2.readlines()
        one = 0
        zero = 0
        total_nums = len(obj_1)
        assert len(obj_1) == len(obj_2)
        for i in range(total_nums):
            obj_1[i] = json.loads(obj_1[i])
            obj_2[i] = json.loads(obj_2[i])
            sample_id = obj_1[i]['id']
            conv_id = obj_1[i]['conv_id']
            turn_id = obj_1[i]['turn_id']
            rel_label = obj_1[i]['rel_label']
            cur_query = obj_2[i]['query']
            history_query = obj_2[i]["history_query"]
            last_response = obj_2[i]["last_response"]
            assert len(history_query) == len(rel_label)
            if len(history_query) > 0:
                for idx in range(len(history_query)):
                    if rel_label[idx] == 1:
                        one += 1
                    else:
                        zero += 1
                    g.write(
                        json.dumps({
                            "id": sample_id + '-' + str(idx + 1),
                            "query": cur_query,
                            "rel_query": history_query[idx],
                            "rel_label": rel_label[idx],
                            #"last_response": last_response
                        }) + "\n")
        print("one", one)
        print("zero", zero)


if __name__ == "__main__":
    create_label_rel_turn(train_new, train_rel)
    create_label_rel_turn(dev_new, dev_rel)
    convert_gold_to_trec(train_rel, train_rel_gold)
    convert_gold_to_trec(dev_rel, dev_rel_gold)
    create_PRJ("output/topiocqa/dense_rel/train_rel_label.json", train_new, "filter/data/topiocqa_train_qp_1.json")
    create_PRJ("output/topiocqa/dense_rel/dev_rel_label.json", dev_new, "filter/data/topiocqa_dev_qp_1.json")
    

    
